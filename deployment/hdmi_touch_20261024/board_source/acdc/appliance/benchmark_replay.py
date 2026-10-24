"""Native ARM qualification using saved heads; explicitly not fresh PL timing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np

from acdc.contract import ContractError,atomic_json,file_sha
from .native import load_native
from .volume import load_volume,AxialInputs
from .reconstruct import Collector


def run(fixture,output):
    report=json.loads((fixture/'fixture.json').read_text('utf-8'))
    for rel,sha in report['files'].items():
        if file_sha(fixture/rel)!=sha:raise ContractError('fixture hash mismatch: '+rel)
    tick=time.perf_counter();kernel=load_native();initialization=time.perf_counter()-tick
    table=np.load(fixture/'exp.npy',allow_pickle=False)
    tick=time.perf_counter()
    volume=load_volume(fixture/'image.nii.gz',kernel,policy='acdc-direction-only-sform-v1',binding=report['binding'])
    preparation=time.perf_counter()-tick
    if hashlib.sha256(volume.working.tobytes()).hexdigest()!=report['working_sha256']:
        raise ContractError('ARM working float32 mismatch')
    inputs=AxialInputs(volume,kernel);collector=Collector(volume,kernel,table)
    nz=volume.working.shape[2]
    heads=np.memmap(fixture/'heads.bin',np.int8,mode='r',shape=(nz,1,4,256,256))
    input_time=0.;restoration=0.
    for z in range(nz):
        tick=time.perf_counter();x,_=inputs.get(z);input_time+=time.perf_counter()-tick
        if hashlib.sha256(x.tobytes()).hexdigest()!=report['input_sha256'][z]:
            raise ContractError('ARM input mismatch at '+str(z))
        tick=time.perf_counter();collector.add(heads[z],z,inputs.spec);restoration+=time.perf_counter()-tick
    tick=time.perf_counter();_,labels=collector.finish();restoration+=time.perf_counter()-tick
    if hashlib.sha256(labels.tobytes()).hexdigest()!=report['labels_sha256']:
        raise ContractError('ARM reconstructed label mismatch')
    result=dict(schema='pld-acdc-appliance-native-arm-replay-v2',machine=platform.machine(),slices=nz,
                model_sha256=report['model_sha256'],fixture_sha256=file_sha(fixture/'fixture.json'),
                native_build=kernel.info,working_float32_equal=True,integer_inputs_equal=True,native_labels_equal=True,
                native_initialization_seconds=initialization,prepare_seconds=preparation,input_seconds=input_time,
                restore_seconds=restoration,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                restore_stages=collector.timings,
                fresh_pl_inferences=0,scope='ARM native math on one phase with authenticated saved heads; not full job latency')
    atomic_json(output,result)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.fixture,args.output)
