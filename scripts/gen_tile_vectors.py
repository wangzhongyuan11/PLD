"""Bank loading and output checks against A's independent layer goldens."""
from pathlib import Path
import argparse
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import Model, conv2d, require_equal
from reference.tinyunet import apply_lut, rescale_lut
from reference.tiling import choose_tile, choose_fused_decoder_tile, tiles, input_tile, deep_input_tile


def emit_case(f, data, weights, bias, mult, expected, repeat=False, reset=False):
    _, ci, ih, iw = data.shape
    co, _, kernel, _ = weights.shape
    oh, ow = expected.shape[2:]
    groups = (ci+3)//4
    f.write(f"0 {ih:x} {iw:x} {ci:x} {co:x} {int(kernel==3):x}\n")
    for c in range(ci):
        for y in range(ih):
            for x in range(iw):
                addr = (c//4)*ih*iw+y*iw+x
                f.write(f"1 {c%4:x} {addr:x} {int(data[0,c,y,x])&255:x} 0 0\n")
    for o in range(co):
        for i in range(ci):
            for y in range(kernel):
                for x in range(kernel):
                    bank = (o%8)*4+i%4
                    addr = (o//8)*groups*kernel*kernel+(i//4)*kernel*kernel+y*kernel+x
                    f.write(f"2 {bank:x} {addr:x} {int(weights[o,i,y,x])&255:x} 0 0\n")
        f.write(f"3 {o:x} {int(bias[o])&0xffffffff:x} 0 0 0\n")
        f.write(f"4 {o:x} {int(mult[o]):x} 0 0 0\n")
    for run in range(2 if repeat else 1):
        f.write(f"5 {int(reset and run==0):x} 0 0 0 0\n")
        for o in range(co):
            for y in range(oh):
                for x in range(ow):
                    addr = (o//8)*oh*ow+y*ow+x
                    f.write(f"6 {o%8:x} {addr:x} {int(expected[0,o,y,x])&255:x} 0 0\n")
    return (2 if repeat else 1), expected.size*(2 if repeat else 1)


def expand_deep(patch, ih, iw):
    """Nearest-2 expansion of a compact deep patch to full-resolution coords.

    Independent oracle for mode 6: full-res patch index (i,j) reads compact
    sample ((i+1)//2,(j+1)//2).  This direct formula must match the engine's
    incremental address registers without sharing any implementation.
    """
    rows = (np.arange(ih)+1)//2
    cols = (np.arange(iw)+1)//2
    return patch[:,:,rows][:,:,:,cols]


def emit_fused_case(f, skip_raw, deep_raw, lut0, lut1, weights, bias, mult,
                    expected, repeat=False, reset=False):
    """Mode 6 engine case: banks hold POST-LUT bytes in the two-branch layout.

    Skip channels use the normal full-resolution plane; deep channels use a
    compact (ih/2+1)x(iw/2+1) patch starting at bank address
    ceil(split/4)*ih*iw.  Expected values come from the reference
    convolution on the nearest-expanded branches, never from RTL state.
    """
    _, split, ih, iw = skip_raw.shape
    _, deep_ci, dh, dw = deep_raw.shape
    co, ci, kernel, _ = weights.shape
    if kernel != 3 or deep_ci != ci-split or (dh, dw) != (ih//2+1, iw//2+1):
        raise ValueError("inconsistent fused case geometry")
    oh, ow = expected.shape[2:]
    groups = (ci+3)//4
    plane, deep_plane = ih*iw, dh*dw
    deep_base = ((split+3)//4)*plane
    skip_mapped = apply_lut(skip_raw, lut0)
    deep_mapped = apply_lut(deep_raw, lut1)
    f.write(f"8 {ih:x} {iw:x} {ci:x} {co:x} {split:x}\n")
    for c in range(split):
        for y in range(ih):
            for x in range(iw):
                f.write(f"1 {c%4:x} {(c//4)*plane+y*iw+x:x} {int(skip_mapped[0,c,y,x])&255:x} 0 0\n")
    for c in range(deep_ci):
        for y in range(dh):
            for x in range(dw):
                f.write(f"1 {c%4:x} {deep_base+(c//4)*deep_plane+y*dw+x:x} {int(deep_mapped[0,c,y,x])&255:x} 0 0\n")
    for o in range(co):
        for i in range(ci):
            for y in range(kernel):
                for x in range(kernel):
                    bank = (o%8)*4+i%4
                    addr = (o//8)*groups*kernel*kernel+(i//4)*kernel*kernel+y*kernel+x
                    f.write(f"2 {bank:x} {addr:x} {int(weights[o,i,y,x])&255:x} 0 0\n")
        f.write(f"3 {o:x} {int(bias[o])&0xffffffff:x} 0 0 0\n")
        f.write(f"4 {o:x} {int(mult[o]):x} 0 0 0\n")
    for run in range(2 if repeat else 1):
        f.write(f"5 {int(reset and run==0):x} 0 0 0 0\n")
        for o in range(co):
            for y in range(oh):
                for x in range(ow):
                    f.write(f"6 {o%8:x} {(o//8)*oh*ow+y*ow+x:x} {int(expected[0,o,y,x])&255:x} 0 0\n")
    return (2 if repeat else 1), expected.size*(2 if repeat else 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="all 15 complete layers, using capacity-limited halo tiles")
    parser.add_argument("--full-fused-only", action="store_true", help="all fused decoder tiles without repeating the 15 baseline layers")
    args = parser.parse_args()
    if args.full and args.full_fused_only:
        parser.error("choose --full or --full-fused-only")
    model = Model(Path("role_a_final/int8"))
    path = Path("out/vectors/tile.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = runs = checks = 0
    with path.open("w", encoding="ascii") as f:
        if args.full or args.full_fused_only:
            if args.full:
                for idx, (layer, params) in enumerate(zip(model.layers, model.parameters)):
                    data, expected = model.golden(idx,"input"), model.golden(idx,"output")
                    co, ci, kernel, _ = params[0].shape
                    h, w = expected.shape[2:]
                    th, tw = choose_tile(ci,co,h,w,kernel)
                    for tile in tiles(h,w,th,tw):
                        local = input_tile(data,tile,kernel,layer["padding"])
                        golden = expected[:,:,tile.y:tile.y+tile.height,tile.x:tile.x+tile.width]
                        n, size = emit_case(f,local,*params,golden)
                        runs += n; checks += size
                    print(f"FULL_TILE_VECTORS layer={idx} name={layer['directory']} tile={th}x{tw}", flush=True)
            # Fused decoder-entry layers must reproduce the same frozen
            # goldens from the two compact branch patches (mode 6).
            for idx in (8,10,12):
                layer = model.layers[idx]
                weights, bias, mult = model.parameters[idx]
                co, ci, k, _ = weights.shape
                skip_index = {8:5,10:3,12:1}[idx]
                skip_all = model.golden(skip_index,"output")
                deep_all = model.golden(idx-1,"output")
                expected = model.golden(idx,"output")
                h, w = expected.shape[2:]
                th, tw = choose_fused_decoder_tile(ci,skip_all.shape[1],co,h,w)
                lut0 = rescale_lut(model.layers[skip_index]["output_scale"],layer["input_scale"])
                lut1 = rescale_lut(model.layers[idx-1]["output_scale"],layer["input_scale"])
                for tile in tiles(h,w,th,tw):
                    skip_patch = input_tile(skip_all,tile,k,layer["padding"])
                    deep_patch = deep_input_tile(deep_all,tile,k)
                    golden = expected[:,:,tile.y:tile.y+tile.height,tile.x:tile.x+tile.width]
                    n, size = emit_fused_case(f,skip_patch,deep_patch,lut0,lut1,
                                              weights,bias,mult,golden,
                                              repeat=(tile.y==0 and tile.x==0))
                    runs += n; checks += size
                print(f"FULL_FUSED_TILE_VECTORS layer={idx} name={layer['directory']} tile={th}x{tw}", flush=True)
            f.write(f"7 {runs:x} 0 {checks:x} 0 0\n")
            scope="15+fused_decoder" if args.full else "fused_decoder_only"
            print(f"TILE_VECTORS full_layers={scope} successful_runs={runs} output_bytes={checks} path={path}")
            return
        for idx, (layer, cfg, params) in enumerate(zip(model.layers, model.configs, model.parameters)):
            folder = model.root/layer["directory"]
            data = np.fromfile(folder/"input.bin", dtype=np.int8).reshape(cfg["input_shape"])
            expected = np.fromfile(folder/"output_golden.bin", dtype=np.int8).reshape(cfg["output_shape"])
            require_equal(conv2d(data, *params, cfg["padding"]), expected, layer["directory"])
            pad = cfg["padding"]
            data = np.pad(data, ((0,0),(0,0),(pad,pad),(pad,pad)))
            n, size = emit_case(f, data, *params, expected, repeat=idx==0, reset=idx==0)
            runs += n; checks += size
            cases += 1
        rng = np.random.default_rng(20260926)
        # Odd channels/output lanes, rectangular halo, and both bank capacity limits.
        for ci, co, h, w, kernel in ((5,9,7,11,3), (1,1,1,1,1), (16,8,64,128,1)):
            data = rng.integers(-127,128,(1,ci,h+kernel-1,w+kernel-1),dtype=np.int8)
            weights = rng.integers(-9,10,(co,ci,kernel,kernel),dtype=np.int8)
            bias = rng.integers(-10000,10001,co,dtype=np.int32)
            mult = rng.integers(1000,300001,co,dtype=np.int32)
            expected = conv2d(data,weights,bias,mult,0)
            n, size = emit_case(f,data,weights,bias,mult,expected)
            runs += n; checks += size
            cases += 1
        # Fused decoder-input cases: x/y parity, row wrap, multiple deep
        # groups, output-channel restart, rectangular patches, repeated runs
        # and a mid-accumulation reset.  LUTs are arbitrary byte maps; the
        # oracle conv2d sees exactly the post-LUT expanded branches.
        for ci, split, ih, iw, co, repeat, reset in ((8,4,6,6,8,False,False),
                                                      (8,4,6,6,12,True,False),
                                                      (16,4,4,8,9,False,False),
                                                      (12,8,8,6,16,False,True)):
            skip_raw = rng.integers(-127,128,(1,split,ih,iw),dtype=np.int8)
            deep_raw = rng.integers(-127,128,(1,ci-split,ih//2+1,iw//2+1),dtype=np.int8)
            lut0 = rng.integers(-127,128,256,dtype=np.int8)
            lut1 = rng.integers(-127,128,256,dtype=np.int8)
            weights = rng.integers(-9,10,(co,ci,3,3),dtype=np.int8)
            bias = rng.integers(-10000,10001,co,dtype=np.int32)
            mult = rng.integers(1000,300001,co,dtype=np.int32)
            fused_input = np.concatenate((apply_lut(skip_raw,lut0),
                                          expand_deep(apply_lut(deep_raw,lut1),ih,iw)),axis=1)
            expected = conv2d(fused_input,weights,bias,mult,0)
            n, size = emit_fused_case(f,skip_raw,deep_raw,lut0,lut1,
                                      weights,bias,mult,expected,repeat=repeat,reset=reset)
            runs += n; checks += size
            cases += 1
        # Invalid configurations must complete with error, never wrap RAM addresses.
        rejected = 0
        for ih, iw, ci, co, k3 in ((0,10,3,8,1),(2,2,3,8,1),(10,10,0,8,1),
                                   (10,10,97,8,1),(10,10,3,65,1),
                                   (258,258,3,8,1),(64,129,16,8,0),
                                   (65,128,4,8,0)):
            f.write(f"0 {ih:x} {iw:x} {ci:x} {co:x} {k3:x}\n5 0 1 0 0 0\n")
            rejected += 1
        # Mode 6 rejections: odd patch size, split boundary/group errors,
        # kernel 1 and bank-capacity overflow.
        for ih, iw, ci, co, split, op in ((5,6,8,8,4,8),(6,7,8,8,4,8),
                                           (6,6,8,8,0,8),(6,6,8,8,8,8),
                                           (6,6,8,8,2,8),(6,6,9,8,4,8),
                                           (6,6,8,8,4,9),(258,258,8,8,4,8)):
            f.write(f"{op} {ih:x} {iw:x} {ci:x} {co:x} {split:x}\n5 0 1 0 0 0\n")
            rejected += 1
        f.write(f"7 {runs:x} {rejected:x} {checks:x} 0 0\n")
    print(f"TILE_VECTORS cases={cases} successful_runs={runs} rejected={rejected} path={path}")


if __name__ == "__main__":
    main()
