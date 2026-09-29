"""Scheduling oracle tests; these do not count as hardware inference evidence."""
from pathlib import Path
import sys
import unittest
import hashlib

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"board"))
from tensor_accel import shape
from pl_graph import PLGraph,tensor_tile_shape
from reference.tinyunet import Model,conv2d,apply_lut,maxpool2,upsample2
from reference.tiling import deep_input_tile,choose_fused_decoder_tile,Tile


class OracleTile:
    """Software stand-in ONLY for validating host tiling and skip lifetimes."""
    def __init__(self):
        self.abi_version=0x00010001
        self.events=[]
        self.loads=0
        self.runs=0
        self.modes=[]

    def configure(self,mode,h,w,ci,co=0,kernel=1,split=0,frame=None):
        self.config=shape(mode,h,w,ci,co,kernel,split)
        self.modes.append(mode)

    def load_parameters(self,*params): self.params=params; self.loads+=1
    def load_lut(self,first,second=None): self.luts=(first,second)
    def load_input(self,data,second=None): self.input=(data.copy(),None if second is None else second.copy())

    def run(self):
        self.runs+=1
        mode=self.config["mode"]
        first,second=self.input
        if mode in (0,5):
            if mode==5: first=apply_lut(first,self.luts[0])
            return conv2d(first,*self.params,0)
        if mode==6:
            # Independent math: deep branch reads compact sample
            # ((i+1)//2,(j+1)//2) at full-resolution patch index (i,j).
            ih,iw=first.shape[2],first.shape[3]
            rows=(np.arange(ih)+1)//2; cols=(np.arange(iw)+1)//2
            expanded=apply_lut(second,self.luts[1])[:,:,rows][:,:,:,cols]
            return conv2d(np.concatenate((apply_lut(first,self.luts[0]),expanded),axis=1),*self.params,0)
        if mode==1: return apply_lut(first,self.luts[0])
        if mode==2: return apply_lut(maxpool2(first),self.luts[0])
        if mode==3: return apply_lut(upsample2(first),self.luts[0])
        return np.concatenate((apply_lut(first,self.luts[0]),upsample2(apply_lut(second,self.luts[1]))),axis=1)


class Tests(unittest.TestCase):
    def test_final_fused_scheduler_matches_all_raw_layer_outputs(self):
        model=Model()
        device=OracleTile()
        output,report=PLGraph(model,device,fuse_maps=True).run(model.golden(0,"input"),verify=True)
        np.testing.assert_array_equal(output,model.golden(14,"output"))
        self.assertEqual(device.loads,15)
        self.assertEqual(device.runs,143)
        self.assertEqual(sum(r["input_mapping"]=="PL load LUT" for r in report["layers"]),8)
        device.abi_version=0x00010000
        with self.assertRaisesRegex(ValueError,"ABI 1.1"):
            PLGraph(model,device,fuse_maps=True)

    def test_final_fused_decoder_scheduler_matches_all_raw_layer_outputs(self):
        model=Model()
        device=OracleTile()
        device.abi_version=0x00010002
        output,report=PLGraph(model,device,fuse_decoder=True).run(model.golden(0,"input"),verify=True)
        np.testing.assert_array_equal(output,model.golden(14,"output"))
        self.assertEqual(device.loads,15)
        # Decoder entries 8/10/12 use 2+4+8 fused conv tiles; every
        # CONCAT_UP transition (56 tile runs) disappears from the schedule.
        self.assertEqual(device.modes.count(6),14)
        self.assertNotIn(4,device.modes)
        baseline=OracleTile()
        PLGraph(model,baseline).run(model.golden(0,"input"),verify=True)
        self.assertEqual(baseline.modes.count(4),56)
        self.assertEqual(device.runs,baseline.runs-70)
        self.assertEqual(sum(r["input_mapping"]=="PL fused decoder input" for r in report["layers"]),3)
        self.assertTrue(report["fused_decoder"])
        self.assertFalse(report["fused_maps"])
        device.abi_version=0x00010001
        with self.assertRaisesRegex(ValueError,"ABI 1.2"):
            PLGraph(model,device,fuse_decoder=True)

    def test_full_scheduler_with_independent_tile_oracle(self):
        model=Model(ROOT/"role_a_handoff/role_a/int8")
        device=OracleTile()
        image=model.golden(0,"input")
        metadata={"model_spec_sha256":model.spec_hash,"input_sha256":hashlib.sha256(image.tobytes()).hexdigest(),
                  "valid_rect":[0,32,256,192]}
        output,report=PLGraph(model,device).run(image,verify=True,input_metadata=metadata)
        np.testing.assert_array_equal(output,model.golden(14,"output"))
        self.assertEqual(len(report["layers"]),15)
        self.assertEqual(device.loads,15)
        self.assertGreater(device.runs,80)
        self.assertEqual(report["result"]["valid_rect"],[0,32,256,192])
        self.assertEqual(report["result"]["output_sha256"],hashlib.sha256(output.tobytes()).hexdigest())
        metadata["input_sha256"]="wrong"
        with self.assertRaisesRegex(ValueError,"metadata"):
            PLGraph(model,device).run(image,input_metadata=metadata)

    def test_shapes_and_capacity_rejection(self):
        self.assertEqual(shape(0,34,258,3,8,3)["output_bytes"],65536)
        self.assertEqual(shape(4,8,64,96,split=32)["input_bytes"],24576)
        self.assertEqual(shape(4,8,64,96,split=32)["output_bytes"],49152)
        # Real decoder-entry patch geometries and the fused byte contract.
        self.assertEqual(shape(6,18,66,96,32,3,32)["input_bytes"],59776)
        self.assertEqual(shape(6,18,66,96,32,3,32)["lut_bytes"],512)
        self.assertEqual(shape(6,18,258,24,8,3,8)["input_bytes"],57952)
        for ci,split,co,height,width,expected in ((96,32,32,64,64,(32,64)),
                                                   (48,16,16,128,128,(32,128)),
                                                   (24,8,8,256,256,(32,256))):
            self.assertEqual(choose_fused_decoder_tile(ci,split,co,height,width),expected)
            shape(6,expected[0]+2,expected[1]+2,ci,co,3,split)
        for args in ((0,258,258,3,8,3),(4,7,8,96,0,1,32),(2,3,8,8),
                     (3,129,1,1),(0,1,1,1,1,0),(0,1,1,97,1,1)):
            with self.assertRaises(ValueError): shape(*args)
        for args in ((6,19,66,96,32,3,32),(6,18,67,96,32,3,32),(6,18,66,96,32,1,32),
                     (6,18,66,96,32,3,33),(6,18,66,96,32,3,0),(6,18,66,96,32,3,96),
                     (6,18,66,98,32,3,32),(6,259,66,96,32,3,32)):
            with self.assertRaises(ValueError): shape(*args)
        # Group-aligned splits on both sides of the boundary stay valid.
        self.assertEqual(shape(6,18,66,96,32,3,36)["input_bytes"],63168)
        for mode,ci,h,w,split in ((1,8,256,256,0),(2,32,32,32,0),(4,96,64,64,32)):
            th,tw=tensor_tile_shape(mode,ci,h,w,split)
            ih,iw=(th*2,tw*2) if mode==2 else (th,tw)
            shape(mode,ih,iw,ci,split=split)
            self.assertEqual(h%th,0); self.assertEqual(w%tw,0)

    def test_deep_input_tile_gathers_exact_half_resolution_patch(self):
        rng=np.random.default_rng(20260929)
        source=rng.integers(-127,128,(1,16,128,128),dtype=np.int8)
        # Tile coordinates live in conv-output space (256x256 here); the
        # source rectangle is rows/cols [origin//2-1, origin//2+extent//2+1).
        tile=Tile(64,64,16,16)
        patch=deep_input_tile(source,tile,3)
        self.assertEqual(patch.shape,(1,16,10,10))
        np.testing.assert_array_equal(patch[0,:,1:9,1:9],source[0,:,32:40,32:40])
        # Full-image borders zero-pad outside the source, matching conv halo.
        for tile in (Tile(0,0,16,64),Tile(0,248,16,8),Tile(248,0,8,64),Tile(248,248,8,8)):
            patch=deep_input_tile(source,tile,3)
            ih,iw=tile.height+2,tile.width+2
            self.assertEqual(patch.shape,(1,16,ih//2+1,iw//2+1))
            dy0,dx0=tile.y//2-1,tile.x//2-1
            for dy in range(patch.shape[2]):
                for dx in range(patch.shape[3]):
                    sy,sx=dy0+dy,dx0+dx
                    want=source[0,:,sy,sx] if 0<=sy<128 and 0<=sx<128 else np.zeros(16,dtype=np.int8)
                    np.testing.assert_array_equal(patch[0,:,dy,dx],want)
        for tile,kernel in ((Tile(63,0,16,64),3),(Tile(0,63,16,64),3),(Tile(0,0,16,64),1)):
            with self.assertRaises(ValueError): deep_input_tile(source,tile,kernel)


if __name__=="__main__": unittest.main()
