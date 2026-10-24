"""Explicit isolated board test; never edits web services or startup settings."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
from frame_driver import FrameSession


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--model',choices=('skin','camus'),required=True)
    parser.add_argument('--frames',type=int,default=1)
    parser.add_argument('--min-seconds',type=float,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.frames<1 or not math.isfinite(args.min_seconds) or args.min_seconds<0:raise ValueError('invalid measurement duration')
    root=args.package.resolve()
    package=json.loads((root/'package.json').read_text())
    for name,digest in package['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('package integrity failure: '+name)
    receipt=json.loads((root/'build_receipt.json').read_text())
    if not receipt['passed']:raise ValueError('build did not qualify')
    for name in ('skin8.bit','skin8.hwh'):
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=receipt['artifacts'][name]:raise ValueError('overlay pair mismatch')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'kind':'isolated_board_quantized_input_to_owned_head','model':args.model,
            'build_artifacts':receipt['artifacts'],'frames':0,'passed':False,'timings':[],
            'scope':'repeated frozen full-size input; not dataset accuracy or webpage latency'}
    def save():
        tmp=args.output.with_suffix(args.output.suffix+'.tmp')
        tmp.write_text(json.dumps(report,indent=2));tmp.replace(args.output)
    save()
    with open('/run/pld-unified-fpga.lock','a+b') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from pynq import Overlay, MMIO, allocate
        overlay=Overlay(str(root/'skin8.bit'))
        mmio=MMIO(0x43c00000,0x1000)
        session=FrameSession(root/'models'/args.model,mmio,allocate)
        try:
            manifest=session.manifest
            c,h,w=manifest['layers'][0]['input_shape']
            packed=np.frombuffer((root/'models'/args.model/'first_input_nchwc8.bin').read_bytes(),np.int8)
            x=packed.reshape(-1,h,w,8).transpose(0,3,1,2).reshape(-1,h,w)[:c].copy()
            co,oh,ow=manifest['layers'][-1]['output_shape']
            golden_raw=(root/'models'/args.model/'golden_head_nchwc8.bin').read_bytes()
            if hashlib.sha256(golden_raw).hexdigest()!=manifest['artifacts']['golden_head_nchwc8.bin']['sha256']:raise ValueError('golden mismatch')
            golden=np.frombuffer(golden_raw,np.int8).reshape(-1,oh,ow,8).transpose(0,3,1,2).reshape(-1,oh,ow)[:co]
            begin=time.perf_counter()
            while report['frames']<args.frames or time.perf_counter()-begin<args.min_seconds:
                head,timing=session.run(x,timeout=10)
                if not np.array_equal(head,golden):
                    where=np.argwhere(head!=golden)[0]
                    report['first_difference']={'coordinate':where.tolist(),'expected':int(golden[tuple(where)]),'actual':int(head[tuple(where)])}
                    raise AssertionError('hardware head differs from frozen model')
                report['frames']+=1;report['timings'].append(timing)
                if report['frames']%25==0:save()
            report['elapsed_seconds']=time.perf_counter()-begin
            report['sustained_fps']=report['frames']/report['elapsed_seconds']
            durations=[v['input_to_owned_head_seconds'] for v in report['timings']]
            report['median_seconds']=float(np.median(durations));report['p95_seconds']=float(np.percentile(durations,95))
            report['passed']=True;save()
        except BaseException as exc:
            report['error']=type(exc).__name__+': '+str(exc)
            report['hardware_activity_unresolved']=session.active;save()
            if session.active:
                # Keep DMA allocations alive for the supervisor. Do not resume
                # or kill this process until board reset has cleared bus work.
                report['recovery']='process stopped retaining DMA memory; board reset required before termination';save()
                os.kill(os.getpid(),signal.SIGSTOP)
                raise RuntimeError('unsafe resume after unresolved hardware activity') from exc
            raise
        finally:
            if not session.active:session.close()


if __name__=='__main__':main()
