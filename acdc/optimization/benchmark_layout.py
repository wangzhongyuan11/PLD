"""Compare checked PS boundary copies on the actual host; never an FPGA FPS test."""
from __future__ import annotations
import argparse
import platform
import time
from pathlib import Path
import numpy as np
from acdc.contract import atomic_json
from .layout import pack_feature
from .image import decode_head
from .native_layout import load_native_layout


def run(output):
    output=Path(output)
    if output.exists():raise ValueError('choose a fresh benchmark report')
    native=load_native_layout()
    rng=np.random.default_rng(20102026)
    source=rng.integers(-127,128,(1,3,256,256),dtype=np.int16).astype(np.int8)
    head=rng.integers(-127,128,(4,256,256),dtype=np.int16).astype(np.int8)
    guarded=np.full(524288+128,0xa5,np.uint8);packed=guarded[64:-64]
    native.pack(packed,source)
    assert packed.tobytes()==pack_feature(source[0]).tobytes()
    assert np.all(guarded[:64]==0xa5) and np.all(guarded[-64:]==0xa5)
    physical=pack_feature(head).view(np.uint8).ravel()
    result=native.head(physical)
    assert np.array_equal(result[0],head)
    physical[:]=0
    assert np.array_equal(result[0],head)
    for c in range(3):
        bad=source.copy();bad[0,c,-1,-1]=-128
        try:native.pack(packed,bad)
        except ValueError:pass
        else:raise AssertionError('accepted invalid input')
    for c in range(8):
        bad=np.zeros(524288,np.uint8);bad[-8+c]=128 if c<4 else 1
        try:native.head(bad)
        except ValueError:pass
        else:raise AssertionError('accepted invalid output')
    physical=pack_feature(head).view(np.uint8).ravel()
    def numpy_pack():packed[:]=np.frombuffer(pack_feature(source[0]).tobytes(),np.uint8)
    operations={'numpy_pack':numpy_pack,'native_pack':lambda:native.pack(packed,source),
                'numpy_head':lambda:decode_head(physical.tobytes()),
                'native_head':lambda:native.head(physical)}
    samples={name:[] for name in operations}
    for iteration in range(35):
        for name,fn in operations.items():
            start=time.perf_counter();fn();elapsed=time.perf_counter()-start
            if iteration>=5:samples[name].append(elapsed)
    report=dict(schema='pld-acdc-layout-benchmark-v1',machine=platform.machine(),
        numpy=np.__version__,python=platform.python_version(),native_build=native.info,
        exact_layout=True,invalid_input_head_padding_rejected=True,independent_output=True,
        guard_bytes_preserved=True,physical_fpga=False,
        scope='PS checked boundary copies in ordinary NumPy memory; excludes CMA cache synchronization and FPGA',
        samples_seconds=samples,median_seconds={k:float(np.median(v)) for k,v in samples.items()})
    atomic_json(output,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(run(args.output)['median_seconds'])
