"""Repeatable PL timing baseline; cProfile is a separate diagnostic frame."""
import argparse
from collections import defaultdict
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import DEFAULT_MODEL_ROOT,Model,require_equal,digest
from tensor_accel import TileAccelerator
from pl_graph import PLGraph


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bit',required=True)
    p.add_argument('--model',default=DEFAULT_MODEL_ROOT)
    p.add_argument('--repeat',type=int,default=5)
    p.add_argument('--report',required=True)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--native-copy',help='verified native copy library; default remains NumPy')
    p.add_argument('--tx-range-sync',action='store_true',help='XRT TX prefix cache synchronization')
    p.add_argument('--native-control',help='verified native command library')
    p.add_argument("--fuse-maps",action="store_true",help="ABI 1.1 PL input-LUT convolution")
    p.add_argument("--fuse-decoder",action="store_true",help="ABI 1.2 PL fused decoder-entry convolution")
    args=p.parse_args()
    if args.repeat<2: p.error('at least two measured frames required')
    model=Model(Path(args.model)); image=model.golden(0,'input'); golden=model.golden(14,'output')
    report={'passed':False,'warmup_frames':1,'frames':[],
            'scope':'one quantized input; PS scheduling + PL/DMA + stitching; correctness comparison excluded',
            'model_spec_sha256':model.spec_hash,
            'files':{str(f):digest(f) for f in [Path(args.bit),Path(args.bit).with_suffix('.hwh'),
                     Path(__file__),Path(__file__).with_name('pl_graph.py'),
                     Path(__file__).with_name('tensor_accel.py'),Path(__file__).with_name('simple_dma.py')]}}
    try:
        with TileAccelerator(args.bit,native_copy=args.native_copy,tx_range_sync=args.tx_range_sync,
                             native_control=args.native_control) as device:
            report['clock_mhz']=device.clock_mhz
            report['copy_backend']=device.copy_info
            report['tx_sync_backend']=device.tx_sync_info
            report['control_backend']=device.control_info
            graph=PLGraph(model,device,fuse_maps=args.fuse_maps,fuse_decoder=args.fuse_decoder)
            output,_=graph.run(image,frame=0)
            require_equal(output,golden,'warmup head')
            for frame in range(1,args.repeat+1):
                device.events=[]
                output,record=graph.run(image,frame=frame)
                require_equal(output,golden,f'frame {frame} head')
                groups=defaultdict(lambda:{'commands':0,'bytes':0,'seconds':0.,'dma_seconds':0.,'core_cycles':0,'compute_cycles':0})
                for event in device.events:
                    group=groups[f"command{event['command']}_mode{event['mode']}"]
                    group['commands']+=1
                    for key in ('bytes','seconds','dma_seconds','core_cycles','compute_cycles'): group[key]+=event[key]
                record['groups']=dict(groups)
                record['head_sha256']=hashlib.sha256(output.tobytes()).hexdigest()
                report['frames'].append(record)
                print(f"PROFILE_FRAME_PASS frame={frame} seconds={record['seconds']:.6f}",flush=True)
            seconds=[r['seconds'] for r in report['frames']]
            report['median_seconds']=float(np.median(seconds))
            report['p95_seconds']=float(np.percentile(seconds,95))
            if args.profile:
                profiler=cProfile.Profile()
                device.events=[]
                profiler.enable(); output,_=graph.run(image,frame=args.repeat+1); profiler.disable()
                require_equal(output,golden,'profiled head')
                profiler.dump_stats(args.report+'.prof')
                with open(args.report+'.profile.txt','w') as stream:
                    stats=pstats.Stats(profiler,stream=stream)
                    stats.strip_dirs().sort_stats('cumulative').print_stats(40)
                    stats.sort_stats('tottime').print_stats(25)
        report['passed']=True
    finally:
        Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(f"PROFILE_PASS median={report['median_seconds']:.6f} p95={report['p95_seconds']:.6f}")


if __name__=='__main__': main()
