"""Explicit board numerical/stability runner. Never succeeds with zero cases."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

from .contract import atomic_json,ContractError,NODE_NAMES,INPUT_SHAPE,checked_int8,file_sha
from .model import load_bundle
from .backend import GraphBackend
from .verify import reference_case


def recovery_checks(backend,first,expected,rows,*,fault_type=None):
    """Inject three real protocol faults, then require a fresh full-network head.

    Uses the mature bounded recovery path. This function is never run by PC
    emulation; fault_type is injectable only for control-flow unit tests.
    """
    if fault_type is None:
        from tensor_accel import AcceleratorFault
        fault_type=AcceleratorFault
    io=backend.graph.device.io
    def after(name,state):
        begin=time.perf_counter();result=backend.graph.forward(first)
        if not np.array_equal(result['head'],expected):raise ContractError(name+' next complete head mismatch')
        rows.append(dict(fault=name,recovered=True,state=state,next_full_head_equal=True,
                         recovery_frame_seconds=time.perf_counter()-begin))
    io.configure(0,1,16,1,1,1)
    def short():
        io.tx[:8]=1;io._execute(2,io.tx,8)
    try:io._guard(short)
    except fault_type as exc:
        if not exc.recovered or exc.state['core']['error']!=2:raise
        after('short_input',exc.state)
    else:raise ContractError('short packet was accepted')
    io.configure(0,1,16,1,1,1);old_timeout=io.timeout;io.timeout=.05
    def missing():io._begin(2);io._wait(16,0)
    try:
        try:io._guard(missing)
        except fault_type as exc:
            if not exc.recovered or 'core completion timeout' not in str(exc):raise
            state=exc.state
        else:raise ContractError('missing packet was accepted')
    finally:io.timeout=old_timeout
    after('missing_input',state)
    layer=backend.model.layers[7]
    # Use the full 128KiB input bank capacity, giving PS a wider active-compute
    # observation window on the faster ACD4 core than a tiny 16x16 probe.
    data=np.random.default_rng(18).integers(-127,128,size=(1,128,32,32),dtype=np.int8)
    cfg=io.configure(0,32,32,128,56,3,relu=True)
    io.load_parameters(layer.weight[:56],layer.bias[:56],layer.multipliers[:56]);io.load_input(data)
    io.dma.submit(destination=io.rx,receive_nbytes=cfg['output_bytes']);io._begin(4)
    deadline=time.monotonic()+io.timeout
    while io.core.read(0x58)<64:
        if time.monotonic()>=deadline:raise TimeoutError('reset test compute never began')
    state=io.snapshot();state['compute_cycles']=int(io.core.read(0x58))
    if state['core']['status']&7!=1 or state['core']['sent']!=0:
        raise ContractError('reset did not catch active computation')
    io.recover()
    if io.core.read(0x0c)&0x7f or io.dma.active:raise ContractError('active state survived reset')
    after('reset_during_compute',state)


def memory():
    report={}
    for path,fields in [(Path('/proc/self/status'),('VmRSS:',)),(Path('/proc/meminfo'),('CmaFree:','CmaTotal:'))]:
        if path.exists():
            for line in path.read_text().splitlines():
                if any(line.startswith(field) for field in fields):report[line.split(':')[0]+'_bytes']=int(line.split()[1])*1024
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-spec-sha');p.add_argument('--allow-synthetic',action='store_true')
    p.add_argument('--repeat',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inputs',type=Path,help='optional directory of complete 196608-byte quantized Axial inputs')
    p.add_argument('--recovery-checks',action='store_true',help='inject short/missing packets and active-compute reset, requiring full-head recovery')
    args=p.parse_args()
    if not 1<=args.repeat<=10000:p.error('repeat must be 1..10000')
    if args.output.exists():p.error('choose a fresh report')
    model=load_bundle(args.model,expected_sha=args.model_spec_sha,allow_synthetic=args.allow_synthetic)
    if args.inputs:
        cases=sorted(args.inputs.glob('*.bin'))
        if not cases or args.repeat<len(cases):p.error('nonempty input directory and repeat>=case count required')
    else:cases=[None]
    folder=args.output.with_suffix('.oracles');folder.mkdir(parents=True,exist_ok=False)
    backend=None
    report=dict(synthetic=model.synthetic,model_sha256=model.model_sha256,backend='pl',passed=False,frames=[],cases=[],
                faults=[],fault_injection_executed=False,volume_quality_evaluated=False)
    started=time.perf_counter()
    try:
        # Compute each independent oracle once, outside measured inference. Persist
        # only input/head, not all node captures, so board RSS stays bounded.
        for index,path in enumerate(cases):
            if path is None:x,oracle=reference_case(model)
            else:
                x=checked_int8(np.fromfile(path,np.int8).reshape(INPUT_SHAPE),INPUT_SHAPE)
                x,oracle=reference_case(model,x)
            if backend is None:backend=GraphBackend(model,'pl',args.runtime)
            verification=backend.graph.forward(x,capture=True)
            for node in NODE_NAMES:
                if not np.array_equal(oracle['captures'][node],verification['captures'][node]):
                    raise ContractError('board node mismatch: '+node)
            input_path=folder/f'{index:04}.input.bin';head_path=folder/f'{index:04}.head.bin'
            x.tofile(input_path);oracle['head'].tofile(head_path)
            report['cases'].append(dict(index=index,input_sha256=file_sha(input_path),head_sha256=file_sha(head_path),
                                         nodes_compared=21,mismatch_count=0))
            del x,oracle,verification
        report['initialization_seconds']=backend.initialization_seconds
        report['hardware']=backend.hardware_info
        if args.recovery_checks:
            first=np.fromfile(folder/'0000.input.bin',np.int8).reshape(INPUT_SHAPE)
            expected=np.fromfile(folder/'0000.head.bin',np.int8).reshape(model.head_shape)
            recovery_checks(backend,first,expected,report['faults'])
            report['fault_injection_executed']=True
            del first,expected
        for i in range(args.repeat):
            index=i%len(cases)
            x=np.fromfile(folder/f'{index:04}.input.bin',np.int8).reshape(INPUT_SHAPE)
            begin=time.perf_counter()
            result=backend.graph.forward(x)
            seconds=time.perf_counter()-begin
            expected=np.fromfile(folder/f'{index:04}.head.bin',np.int8).reshape(model.head_shape)
            if not np.array_equal(expected,result['head']):raise ContractError(f'board head mismatch at frame {i}')
            events=list(backend.graph.device.io.events)
            report['frames'].append(dict(index=i,case_index=index,slice_seconds=seconds,memory=memory(),nodes=result['nodes'],
                event_count=len(events),pl_compute_cycles=sum(e.get('compute_cycles',0) for e in events if e.get('command')==4)))
            atomic_json(args.output,report)
        times=[r['slice_seconds'] for r in report['frames']]
        report.update(passed=True,wall_seconds=time.perf_counter()-started,
            slice_median_seconds=float(np.median(times)),slice_p95_seconds=float(np.percentile(times,95)),
            sustained_fps=len(times)/sum(times),target_4fps_sample_met=float(np.percentile(times,95))<=.25 and len(times)/sum(times)>=4,
            scope='warm complete PS-DMA-PL tensor-to-owned-head; excludes CPU oracle/disk reads/comparison',
            volume_quality_evaluated=False)
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        try:
            if backend is not None:backend.close()
        finally:atomic_json(args.output,report)


if __name__=='__main__':main()
