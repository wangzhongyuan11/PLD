"""Extend the evolving-DDR oracle with a hardware-fetched execution table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gen_executor_vectors import generate as generate_executor

PROGRAM_ADDRESS = 0x10800000


def generate(output,p,divisor=8,model=None):
    metadata=generate_executor(output,p,divisor,model)
    jobs=(output/'jobs.txt').read_text('ascii').splitlines()[1:]
    data=b''.join(int(row,16).to_bytes(64,'little') for row in jobs)
    rows=(output/'source_memory.txt').read_text('ascii').splitlines()
    rows[0]=str(int(rows[0])+len(data)//8)
    rows += [f'{PROGRAM_ADDRESS+off:08x} {int.from_bytes(data[off:off+8],"little"):016x}'
             for off in range(0,len(data),8)]
    (output/'source_memory.txt').write_text('\n'.join(rows)+'\n',encoding='ascii')
    metadata.update(program_address=PROGRAM_ADDRESS,program_bytes=len(data),ps_submissions=1)
    (output/'fixture.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--p',type=int,choices=(4,6),default=4)
    parser.add_argument('--divisor',type=int,choices=(1,2,4,8),default=8)
    parser.add_argument('--model',type=Path);parser.add_argument('--model-spec-sha')
    args=parser.parse_args()
    model=None
    if args.model:
        from acdc.model import load_bundle
        model=load_bundle(args.model,expected_sha=args.model_spec_sha)
    print(generate(args.output,args.p,args.divisor,model))
