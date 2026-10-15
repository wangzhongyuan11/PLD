"""Real-board numerical, recovery and sustained performance gates. No CPU fallback."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from .accelerator import CamusTileAccelerator
from .graph import CamusGraph
from .model import load_model, DEFAULT_MODEL_ROOT, load_golden
from .pipeline import CamusPipeline
from .preprocess import preprocess
from .postprocess import apply_island5_hole1
from .result import native_labels_from_head
from .verify import _golden_rows


def write(path, data):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8');temporary.replace(path)


def check_equal(actual, expected, name):
    if actual.shape != expected.shape or not np.array_equal(actual,expected):
        where = np.argwhere(actual != expected)[0].tolist() if actual.shape == expected.shape else None
        raise RuntimeError(f'{name}: first mismatch {where}; shapes {actual.shape}/{expected.shape}')


def event_summary(events):
    return dict(commands=len(events), packet_bytes=sum(e['bytes'] for e in events),
                command_seconds=sum(e['seconds'] for e in events), dma_seconds=sum(e['dma_seconds'] for e in events),
                engine_cycles=sum(e['compute_cycles'] for e in events),
                core_cycles=sum(e['core_cycles'] for e in events))


def recovery_checks(device, model, graph, first):
    from tensor_accel import AcceleratorFault
    io=device.io; rows=[]
    def after(name, state):
        t=time.perf_counter();result=graph.forward(first)
        check_equal(result['head'],load_golden(model,model.layers[-1],'output_full_golden'),name+' next frame')
        rows.append(dict(fault=name,recovered=True,state=state,next_full_head_equal=True,
                         recovery_frame_seconds=time.perf_counter()-t))
    io.configure(0,1,16,1,1,1)
    def short():
        io.tx[:8]=1;io._execute(2,io.tx,8)
    try:io._guard(short)
    except AcceleratorFault as exc:
        if not exc.recovered or exc.state['core']['error'] != 2:raise
        after('short_input',exc.state)
    else:raise RuntimeError('short packet accepted')
    io.configure(0,1,16,1,1,1);old=io.timeout;io.timeout=.05
    def missing():io._begin(2);io._wait(16,0)
    try:
        try:io._guard(missing)
        except AcceleratorFault as exc:
            if not exc.recovered or 'core completion timeout' not in str(exc):raise
            state=exc.state
        else:raise RuntimeError('missing packet accepted')
    finally:io.timeout=old
    after('missing_input',state)
    # A real Ci=128 deep layer, partial Co=56, with RX armed during abort.
    layer=model.layers[7];data=load_golden(model,layer,'input_full')[:,:,:16,:16]
    cfg=io.configure(0,16,16,128,56,3,relu=True)
    io.load_parameters(layer.weight[:56],layer.bias[:56],layer.multipliers[:56]);io.load_input(data)
    io.dma.submit(destination=io.rx,receive_nbytes=cfg['output_bytes']);io._begin(4)
    deadline=time.monotonic()+io.timeout
    while io.core.read(0x58)<64:
        if time.monotonic()>=deadline:raise TimeoutError('reset test compute never began')
    state=io.snapshot();state['compute_cycles']=int(io.core.read(0x58))
    if state['core']['status']&7 != 1 or state['core']['sent'] != 0:
        raise RuntimeError('reset did not catch active computation')
    io.recover()
    if io.core.read(0x0c)&0x7f or io.dma.active:raise RuntimeError('active state survived reset')
    after('reset_during_compute',state)
    return rows


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='camus-board-validation-v1',scope='real PYNQ PL; all operators use CAM2; no CPU inference',
                complete=False,started_host_label=args.run_label,frames=[],faults=[])
    write(args.output/'summary.json',report)
    model=load_model(args.model);first=load_golden(model,model.layers[0],'input_full')
    start=time.perf_counter()
    try:
        with CamusPipeline(args.model,'pl',args.runtime) as pipeline:
            graph=pipeline.network;io=pipeline.device.io
            report['hardware']=dict(id=hex(io.core.read(0)),abi=hex(io.core.read(4)),clock_mhz=io.clock_mhz,
                                    initialization_seconds=pipeline.initialization_seconds)
            initial=graph.forward(first,capture=True)
            rows=_golden_rows(model,initial['captures']);report['golden_tensors']=rows
            if not all(r['equal'] for r in rows):raise RuntimeError('first-input full graph differs from golden')
            del initial
            report['faults']=recovery_checks(pipeline.device,model,graph,first)
            manifest=json.loads((args.dataset/'dataset.json').read_text()); cases=manifest['cases']
            if len(cases)!=200 or manifest['patients']!=50:raise ValueError('invalid dev dataset manifest')
            for row in cases:
                for key,hashkey in (('input','input_sha256'),('expected','expected_sha256')):
                    p=(args.dataset/row[key]).resolve()
                    if p.parent!=args.dataset.resolve() or hashlib.sha256(p.read_bytes()).hexdigest()!=row[hashkey]:
                        raise ValueError('dataset hash/path mismatch')
            # One complete artifact pass through the public pipeline before the sustained graph loop.
            artifact=pipeline.run(args.dataset/cases[0]['input'],args.output/'full_pipeline')
            with np.load(args.dataset/cases[0]['expected'],allow_pickle=False) as oracle:
                head=np.fromfile(args.output/'full_pipeline/head.bin',np.int8).reshape(model.head_shape)
                check_equal(head,oracle['logits_int8'],'full artifact head')
            report['full_pipeline']={k:v for k,v in artifact.items() if k!='report'}
            report['full_pipeline']['head_equal']=True
            write(args.output/'summary.json',report)
            wall=time.perf_counter(); dice=[]; poor=[]
            with (args.output/'frames.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                for index in range(args.frames):
                    case=cases[index%len(cases)];tick=time.perf_counter()
                    prep=preprocess(args.dataset/case['input']);pre_seconds=time.perf_counter()-tick
                    with np.load(args.dataset/case['expected'],allow_pickle=False) as oracle:
                        check_equal(prep.input_int8,oracle['input_int8'],case['stem']+' input')
                        t=time.perf_counter();r=graph.forward(prep.input_int8);network=time.perf_counter()-t
                        check_equal(r['head'],oracle['logits_int8'],case['stem']+' head')
                        t=time.perf_counter();labels=native_labels_from_head(r['head'],*prep.raw_gray.shape,model.head_scale)
                        processed=apply_island5_hole1(labels).labels;post=time.perf_counter()-t
                        check_equal(labels,oracle['labels'],case['stem']+' native')
                        check_equal(processed,oracle['processed'],case['stem']+' postprocess')
                        if index<len(cases):
                            scores=[]
                            for cls in (1,2,3):
                                pred=processed==cls;gt=oracle['gt_labels']==cls
                                denom=int(pred.sum()+gt.sum());scores.append(2*int(np.count_nonzero(pred & gt))/denom if denom else 1.)
                            dice.append(float(np.mean(scores)))
                            if case['quality'].lower()=='poor':poor.append(float(np.mean(scores)))
                    row=dict(frame=index+1,case=case['stem'],input_equal=True,head_equal=True,native_equal=True,processed_equal=True,
                             preprocess_seconds=pre_seconds,network_seconds=network,postprocess_seconds=post,
                             online_compute_seconds=pre_seconds+network+post,
                             rss_kib=int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0]),
                             **event_summary(r['events']))
                    report['frames'].append(row);log.write(json.dumps(row)+'\n')
                    if index%20==0 or index+1==args.frames:
                        write(args.output/'summary.json',report)
                        print(json.dumps(dict(progress=index+1,total=args.frames,network_seconds=network,rss_kib=row['rss_kib'])),flush=True)
            elapsed=time.perf_counter()-wall;values=np.array([r['network_seconds'] for r in report['frames']])
            online=np.array([r['online_compute_seconds'] for r in report['frames']])
            report['performance']=dict(frames=args.frames,unique_images=min(args.frames,len(cases)),sustained_wall_seconds=elapsed,
                sustained_validated_fps=args.frames/elapsed,network_median_seconds=float(np.median(values)),
                network_p95_seconds=float(np.percentile(values,95)),online_median_seconds=float(np.median(online)),
                online_p95_seconds=float(np.percentile(online,95)),rss_first_kib=report['frames'][0]['rss_kib'],
                rss_last_kib=report['frames'][-1]['rss_kib'],max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            report['metrics']=dict(dev_frames=len(dice),dice_mean=float(np.mean(dice)),poor_frames=len(poor),
                                   poor_dice_mean=float(np.mean(poor)) if poor else None)
            report['complete']=True
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        report['wall_seconds']=time.perf_counter()-start;write(args.output/'summary.json',report)
    print(json.dumps(dict(complete=True,performance=report['performance'],metrics=report['metrics'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL_ROOT);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=1000)
    p.add_argument('--run-label',default='host timestamp not supplied')
    a=p.parse_args()
    if not 1<=a.frames<=1000:p.error('frames must be 1..1000')
    run(a)


if __name__=='__main__':main()
