"""Reproducible host benchmark plus RTL-calibrated cycle/dataflow analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import re
import time
import numpy as np
from .model import DEFAULT_MODEL_ROOT, load_model
from .plan import build_plan
from .pipeline import CamusPipeline


def engine_cycles(cfg):
    if cfg['mode'] == 0:
        h, w = cfg['h']-cfg['k']+1, cfg['w']-cfg['k']+1
        groups = (cfg['co']+7)//8
        return 4+h*w*(groups*((cfg['ci']+3)//4*cfg['k']**2+11)+cfg['co'])
    count = cfg['ci'] * cfg['h'] * cfg['w']
    if cfg['mode'] == 3: count *= 4
    return count+8


def calibrate_rtl(vector_file, log_file):
    registers, completed, current = {}, [], None
    for line in Path(vector_file).read_text().splitlines():
        op, *v = (int(x,16) for x in line.split())
        if op == 0 and v[3] == 0:
            address, value, strobe = v[:3]
            old = registers.get(address, 0)
            for byte in range(4):
                if strobe & (1 << byte): old = (old & ~(255 << (8*byte))) | (value & (255 << (8*byte)))
            registers[address] = old
            if address == 8 and value & 1:
                current = dict(mode=registers.get(0x18), h=registers.get(0x1c), w=registers.get(0x20),
                               ci=registers.get(0x24), co=registers.get(0x28), k=registers.get(0x2c), command=registers.get(0x14))
            if address == 8 and value & 2: current = None
        if op == 2 and v[0] == 0 and v[4] and current and current['command'] == 4:
            completed.append(dict(current))
    log = Path(log_file).read_text()
    if 'TB_PASS tb_axis_tensor_accel_camus' not in log:
        raise ValueError('RTL regression did not finish successfully')
    observed = [int(m[2]) for m in re.findall(r'CAMUS_RUN run=(\d+) bytes=(\d+) cycles=(\d+)', log)]
    if len(observed) != len(completed) or not observed: raise ValueError('RTL cycle sample count mismatch')
    rows = [dict(configuration=cfg, observed_cycles=actual, predicted_cycles=engine_cycles(cfg)) for cfg,actual in zip(completed,observed)]
    if any(row['observed_cycles'] != row['predicted_cycles'] for row in rows):
        raise ValueError('FSM cycle model does not match RTL samples')
    return dict(scope='exact RTL engine cycles on directed regression; full-frame totals are projections',
                matched_samples=len(rows), rows=rows)


def analysis(model, calibration=None):
    plan = build_plan(model)
    tensor_cycles = sum((4 if r['mode']==2 else 1)*r['output_bytes']+8*r['tile_count'] for r in plan['tensor_schedule'])
    clocks = []
    for mhz in (100,125):
        engine = (plan['projected_conv_compute_cycles']+tensor_cycles)/(mhz*1e6)
        serial = plan['total_packet_bytes']/(mhz*1e6)
        clocks.append(dict(clock_mhz=mhz, ideal_32_mac_seconds=plan['total_mac_count']/(32*mhz*1e6),
                           projected_engine_seconds=engine, minimum_byte_serial_seconds=serial,
                           projected_serial_pipeline_floor_seconds=engine+serial,
                           projected_throughput_ceiling_fps=1/(engine+serial)))
    return dict(scope='analytical projection, not measured board latency/FPS', plan=plan, clocks=clocks,
                rtl_cycle_calibration=calibration,
                decoder_entry_mac_fraction=sum(plan['layers'][i]['mac_count'] for i in (8,10,12))/plan['total_mac_count'],
                performance_conclusion='32 MAC/cycle cannot deliver CAMUS 4 FPS at 100/125 MHz; PS tuning alone cannot remove compute and byte-serial limits')


def main(argv=None):
    p=argparse.ArgumentParser(prog='python -m camus.performance')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL_ROOT)
    p.add_argument('--input', type=Path); p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--backend', choices=('reference','emulator','pl'), default='emulator')
    p.add_argument('--runtime', type=Path); p.add_argument('--output', type=Path, default=Path('out/camus/performance.json'))
    args=p.parse_args(argv)
    if not 1 <= args.repeat <= 1000: p.error('repeat must be 1..1000')
    if args.output.exists(): p.error('choose a new output report')
    model=load_model(args.model)
    from .pipeline import protect_output
    protect_output(args.output,model.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    vectors=Path('out/camus/vectors/accel.txt'); log=Path('out/camus/simulation.log')
    calibration=calibrate_rtl(vectors,log) if vectors.exists() and log.exists() else None
    result=dict(schema='camus-performance-v1', platform=platform.platform(), python=platform.python_version(),
                numpy=np.__version__, analysis=analysis(model,calibration), host_benchmark=None)
    if args.input:
        rows=[]; wall=time.perf_counter()
        with CamusPipeline(args.model,args.backend,args.runtime) as pipeline:
            for i in range(args.repeat):
                row=pipeline.run(args.input,args.output.parent/(args.output.stem+f'_frame_{i:04d}'),html=False)
                row.pop('report'); rows.append(row)
        result['host_benchmark']=dict(backend=args.backend, input=str(args.input), frames=args.repeat,
            scope='actual end-to-end artifact generation on this host; not board performance unless backend=pl',
            session_wall_seconds=time.perf_counter()-wall,
            median_seconds=float(np.median([r['end_to_end_seconds'] for r in rows])),
            p95_seconds=float(np.percentile([r['end_to_end_seconds'] for r in rows],95)), rows=rows)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(args.output),calibrated_samples=calibration['matched_samples'] if calibration else 0,
                          benchmark=result['host_benchmark'] and {k:result['host_benchmark'][k] for k in ('backend','frames','median_seconds','p95_seconds')})))


if __name__=='__main__': main()
