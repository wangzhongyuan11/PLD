"""Actual ACD8 board runner: owned heads, fresh wall time and bounded recovery.

CPU-generated input/head cases are verified before loading the overlay. Nothing
in this runner computes the CNN on the PS or fabricates a PL response.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import hashlib
import platform
import numpy as np

from acdc.contract import INPUT_SHAPE,HEAD_SHAPE,ContractError,file_sha,atomic_json,checked_int8,safe_member
from acdc.model import load_bundle
from acdc.board_verify import memory
from .device import SpatialDevice


def telemetry():
    values=dict(monotonic_seconds=time.monotonic(),**memory())
    for path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:values[path.parent.name+'_celsius']=int(path.read_text())/1000
        except (OSError,ValueError):pass
    frequency=Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
    if frequency.exists():
        try:values['cpu0_frequency_khz']=int(frequency.read_text())
        except (OSError,ValueError):pass
    return values


def load_cases(model,filename=None):
    if filename is None:
        return [(np.fromfile(safe_member(model.root,model.spec['golden']['input']['path']),np.int8).reshape(INPUT_SHAPE),
                 np.fromfile(safe_member(model.root,model.spec['golden']['head']['path']),np.int8).reshape(HEAD_SHAPE))]
    filename=Path(filename).resolve();report=json.loads(filename.read_text('utf-8'))
    if report.get('model_spec_sha256')!=model.model_sha256 or not report.get('cases'):
        raise ContractError('empty or wrong-model numerical cases')
    cases=[]
    for row in report['cases']:
        pair=[]
        for key,shape in [('input',INPUT_SHAPE),('head',HEAD_SHAPE)]:
            p=safe_member(filename.parent,row[key]['path'])
            if file_sha(p)!=row[key]['sha256'] or p.stat().st_size!=int(np.prod(shape)):
                raise ContractError('numerical case size/hash mismatch')
            pair.append(checked_int8(np.fromfile(p,np.int8).reshape(shape),shape))
        cases.append(tuple(pair))
    return cases


def recovery_checks(device,first,expected):
    result=[]
    for name,word,value in [('descriptor_tag',12,0x7fffffff),('invalid_ci',11,0)]:
        old=int(device.program_words[0,word]);device.program_words[0,word]=value
        caught=None
        try:
            try:device.forward(first)
            except (RuntimeError,TimeoutError) as exc:caught=str(exc)
            if caught is None:raise ContractError('invalid descriptor accepted: '+name)
        finally:device.program_words[0,word]=old
        if device.unsafe:raise RuntimeError('recovery left unsafe DMA')
        state=device.last_fault_snapshot
        head,_=device.forward(first)
        if not np.array_equal(head,expected):raise ContractError('fresh recovery head mismatch: '+name)
        result.append(dict(fault=name,error=caught,before_reset=state,fresh_head_equal=True))
    class CancelActive:
        def is_set(self):return device.core.read(0x28)>50000
    try:device.forward(first,cancel=CancelActive())
    except InterruptedError:
        state=device.last_fault_snapshot
        if not state['registers']['10']&1:raise ContractError('cancel did not catch active frame')
    else:raise ContractError('active cancel injection not observed')
    head,_=device.forward(first)
    if not np.array_equal(head,expected):raise ContractError('fresh head after cancel mismatch')
    result.append(dict(fault='active_frame_cancel',before_reset=state,fresh_head_equal=True))
    return result


def run(args):
    if args.output.exists():raise ContractError('choose a new board report')
    model=load_bundle(args.model,expected_sha=args.model_spec_sha)
    cases=load_cases(model,args.cases)
    report=dict(schema='pld-acdc-spatial-board-v1',model_spec_sha256=model.model_sha256,
        synthetic=model.synthetic,backend='pynq_pl',passed=False,frames=[],faults=[],memory_samples=[],
        unique_inputs=len({hashlib.sha256(x.tobytes()).hexdigest() for x,_ in cases}),warmup=args.warmup,
        runtime_manifest_sha256=file_sha(args.runtime/'runtime_manifest.json'),initial_memory=telemetry(),
        cases_manifest_sha256=file_sha(args.cases) if args.cases else None,
        node_oracle_sha256=args.node_oracle_sha if args.node_oracle else None,
        environment=dict(python=platform.python_version(),numpy=np.__version__,uname=list(platform.uname()),
                         wall_clock_note='board calendar clock is untrusted; elapsed time is monotonic'))
    device=None
    try:
        tick=time.perf_counter();device=SpatialDevice(args.runtime,model,timeout=args.timeout)
        report['initialization_seconds']=time.perf_counter()-tick
        report['hardware']=json.loads((args.runtime/'overlay/build_manifest.json').read_text('utf-8'))
        if args.node_oracle:
            if file_sha(args.node_oracle)!=args.node_oracle_sha:raise ContractError('node oracle SHA mismatch')
            expected_nodes=np.load(args.node_oracle,allow_pickle=False)
            if str(expected_nodes['model_spec_sha256'])!=model.model_sha256:
                raise ContractError('node oracle model mismatch')
            if str(expected_nodes['input_sha256'])!=hashlib.sha256(cases[0][0].tobytes()).hexdigest():
                raise ContractError('node oracle input mismatch')
            actual_nodes=device.capture_nodes(cases[0][0])
            for name,actual in actual_nodes.items():
                if not np.array_equal(actual,expected_nodes[name]):raise ContractError('physical board node mismatch: '+name)
            report['physical_nodes_compared']=list(actual_nodes)
            report['prefix_timings']=device.capture_timings
            report['prefix_timing_scope']='fresh cumulative prefix runs; differences are not isolated layer measurements'
            report['virtual_CAT_scope']='not materialized; checked through C09/C11/C13 and separate loader RTL'
            expected_nodes.close();del actual_nodes
        for i in range(args.warmup):
            x,expected=cases[i%len(cases)];head,_=device.forward(x)
            if not np.array_equal(head,expected):raise ContractError('warmup head mismatch')
        if args.recovery_checks:report['faults']=recovery_checks(device,*cases[0])
        started=time.perf_counter()
        for i in range(args.repeat):
            index=i%len(cases);x,expected=cases[index]
            head,timing=device.forward(x)
            if not np.array_equal(head,expected):
                different=np.argwhere(head!=expected);coord=tuple(different[0])
                raise ContractError(f'frame {i} mismatch at {coord}: {head[coord]} != {expected[coord]}')
            report['frames'].append(dict(index=i,case=index,head_sha256=hashlib.sha256(head.tobytes()).hexdigest(),
                                         mismatch_count=0,**timing))
            if (i+1)%20==0:
                report['memory_samples'].append(dict(frame=i+1,**telemetry()));atomic_json(args.output,report)
                print('fresh frames',i+1,flush=True)
        wall=time.perf_counter()-started
        times=[r['slice_seconds'] for r in report['frames']]
        p95=float(np.percentile(times,95));fps=len(times)/wall
        report.update(passed=True,fresh_wall_seconds=wall,sustained_fps=fps,
            p50_seconds=float(np.median(times)),p95_seconds=p95,p99_seconds=float(np.percentile(times,99)),
            max_seconds=max(times),sample_4fps_met=p95<=.25 and fps>=4,sample_8fps_met=p95<=.125 and fps>=8,
            G4_performance_met=args.repeat>=1000 and args.recovery_checks and p95<=.25 and fps>=4,
            G8_performance_met=args.repeat>=1000 and args.recovery_checks and p95<=.125 and fps>=8,
            G4_accepted=False,G8_accepted=False,acceptance_note='final acceptance additionally audits memory, full-volume results, original-system regression and delivery evidence',
            scope='warm PS-owned NCHW input through complete DMA/PL to independent NCHW head; wall includes comparisons/report gaps',
            final_memory=telemetry(),volume_quality_evaluated=False)
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        if device is not None:report['last_state']=device.snapshot()
        raise
    finally:
        try:
            if device is not None:device.close()
        finally:atomic_json(args.output,report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-spec-sha',required=True);p.add_argument('--cases',type=Path)
    p.add_argument('--node-oracle',type=Path);p.add_argument('--node-oracle-sha')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeat',type=int,default=5)
    p.add_argument('--warmup',type=int,default=0);p.add_argument('--timeout',type=float,default=5)
    p.add_argument('--recovery-checks',action='store_true');a=p.parse_args()
    if not 1<=a.repeat<=10000 or not 0<=a.warmup<=100:p.error('invalid repeat/warmup')
    print(json.dumps({k:v for k,v in run(a).items() if k not in ('frames','hardware')}))
