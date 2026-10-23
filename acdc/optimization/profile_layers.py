"""Diagnostic single-layer measurements from real prefix-produced DDR inputs.

Does not change the deployed driver or feed golden intermediate activations.
Every measured layer output is checked against an independent integer oracle.
Production frame timing is recorded by board_verify, separately from this tool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np

from acdc.contract import SHAPES,ContractError,atomic_json,file_sha
from acdc.model import load_bundle
from acdc.optimization.board_verify import load_cases
from acdc.optimization.device import SpatialDevice

TARGETS=('work_a','skip0','work_a','skip1','work_a','skip2','work_a','work_b',
         'work_a','work_b','work_a','work_b','work_a','work_b','head')


def run(args):
    if args.output.exists():raise ContractError('choose a new layer profile report')
    model=load_bundle(args.model,expected_sha=args.model_spec_sha)
    if file_sha(args.node_oracle)!=args.node_oracle_sha:raise ContractError('node oracle SHA mismatch')
    first,_=load_cases(model,args.cases)[0]
    oracle=np.load(args.node_oracle,allow_pickle=False)
    if str(oracle['model_spec_sha256'])!=model.model_sha256:raise ContractError('oracle/model mismatch')
    import hashlib
    if str(oracle['input_sha256'])!=hashlib.sha256(first.tobytes()).hexdigest():raise ContractError('oracle/input mismatch')
    device=None;rows=[]
    report=dict(schema='pld-acdc-layer-profile-v1',complete=False,model_spec_sha256=model.model_sha256,
        runtime_manifest_sha256=file_sha(args.runtime/'runtime_manifest.json'),tool_sha256=file_sha(Path(__file__)),
        node_oracle_sha256=args.node_oracle_sha,rows=rows,
        scope='isolated layer jobs including their DMA, implicit CAT and fused Pool; inputs produced by fresh PL prefix; no intermediate injection',
        timing_scope='PL cycles and PS START-to-checked-DONE; debug output copy/check reported separately; not frame FPS')
    try:
        device=SpatialDevice(args.runtime,model);original=device.program_words.copy()
        for index,layer in enumerate(device.program['layers']):
            device.program_words[:]=original
            produced,_=device.forward(first,debug_layer=index)
            if not np.array_equal(produced[layer['name']],oracle[layer['name']]):raise ContractError('prefix oracle mismatch')
            samples=[]
            start,count=layer['first_job'],layer['jobs']
            for repeat in range(args.repeat):
                device.epoch+=1
                device.program_words[:count]=original[start:start+count]
                device.program_words[:count,12]=np.arange(count,dtype=np.uint32)
                device.program_words[:count,15]=device.epoch
                device._sync('program',device.to_device)
                if device.core.read(0x10)&5:raise ContractError('not idle before isolated layer')
                device.core.write(0x14,device.program['addresses']['program'])
                device.core.write(0x18,count);device.core.write(0x1c,device.epoch)
                begin=time.perf_counter();device.core.write(0x0c,1);deadline=time.monotonic()+device.timeout
                try:
                    while True:
                        status=device.core.read(0x10)
                        if status&4:raise RuntimeError('isolated layer fault '+str(device.snapshot()))
                        if status&2:break
                        if time.monotonic()>=deadline:raise TimeoutError('isolated layer deadline '+str(device.snapshot()))
                        time.sleep(.0002)
                    elapsed=time.perf_counter()-begin;cycles=device.core.read(0x28)
                    if device.core.read(0x2c)!=((count<<16)|count):raise ContractError('isolated job count mismatch')
                    tick=time.perf_counter();_,channels,height,_=SHAPES[index]
                    actual=device._feature(TARGETS[index],channels,height)
                    if not np.array_equal(actual,oracle[layer['name']]):raise ContractError('isolated layer output mismatch')
                    if index in (1,3,5):
                        level=(index-1)//2;actual_pool=device._feature(f'pool{level}',channels,height//2)
                        if not np.array_equal(actual_pool,oracle[f'P{level}']):raise ContractError('fused pool output mismatch')
                    copied=time.perf_counter()-tick;device.core.write(0x0c,4)
                except BaseException:
                    device.recover();raise
                samples.append(dict(pl_cycles=cycles,pl_seconds=cycles/(device.clock_mhz*1e6),
                    start_to_done_seconds=elapsed,debug_copy_check_seconds=copied,oracle_equal=True))
            rows.append(dict(layer=layer['name'],jobs=count,repeat=args.repeat,samples=samples,
                median_pl_seconds=float(np.median([s['pl_seconds'] for s in samples])),
                median_start_to_done_seconds=float(np.median([s['start_to_done_seconds'] for s in samples]))))
            atomic_json(args.output,report);print(layer['name'],'isolated checked',flush=True)
        device.program_words[:]=original
        report.update(complete=True,layers=15,intermediate_golden_injected=False)
        return report
    finally:
        oracle.close()
        try:
            if device is not None:device.close()
        finally:atomic_json(args.output,report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','model','cases','node-oracle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model-spec-sha',required=True);p.add_argument('--node-oracle-sha',required=True)
    p.add_argument('--repeat',type=int,default=3);a=p.parse_args()
    if not 1<=a.repeat<=10:p.error('repeat must be 1..10')
    run(a)
