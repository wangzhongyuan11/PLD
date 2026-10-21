"""Current V2 numerical board entry. Full Web readiness is a later stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('numerics',help='real board ED/ES numerical path; excludes meshes and Web')
    for name in ('model','runtime','assets','ed','es','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model-sha',required=True);p.add_argument('--assets-sha',required=True)
    p.add_argument('--geometry-policy',choices=('strict','acdc-direction-only-sform-v1'),default='strict')
    p.add_argument('--serial',action='store_true')
    args=parser.parse_args()
    from acdc.model import load_bundle
    from acdc.backend import SpatialBackend
    from .assets import load
    from .native import load_native
    from .pipeline import NumericalPipeline
    model=load_bundle(args.model,expected_sha=args.model_sha)
    table=load(model,args.assets,args.assets_sha);kernel=load_native()
    backend=SpatialBackend(model,args.runtime)
    print(json.dumps(dict(status='numerical_backend_ready',initialization_seconds=backend.initialization_seconds,
                          model_sha256=model.model_sha256,runtime_sha256=backend.runtime_sha)),flush=True)
    def progress(row):
        if row.get('phase_completed',0)%10==0:print(json.dumps(row),flush=True)
    try:
        result=NumericalPipeline(backend,kernel,table,binding=model.implementation_binding,
            policy=args.geometry_policy).run(args.ed,args.es,args.output,overlap=not args.serial,progress=progress)
        print(json.dumps(dict(status=result['status'],numerical_seconds=result['numerical_seconds'])),flush=True)
    finally:backend.close()


if __name__=='__main__':main()
