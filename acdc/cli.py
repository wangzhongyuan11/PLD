from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser():
    p=argparse.ArgumentParser(prog='python -m acdc',description='Independent Axial-only ACDC system')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('plan',help='static 15-layer/tile/cycle budget; no model or PYNQ required')
    for name in ('run','web','worker'):
        q=sub.add_parser(name)
        q.add_argument('--backend',choices=('reference','reference_torch','emulator','pl','remote','demo'),required=True)
        q.add_argument('--model',type=Path);q.add_argument('--model-spec-sha');q.add_argument('--allow-synthetic',action='store_true')
        q.add_argument('--runtime',type=Path);q.add_argument('--board-url')
        if name=='run':
            q.add_argument('--ed',type=Path,required=True);q.add_argument('--es',type=Path,required=True)
            q.add_argument('--output',type=Path,required=True);q.add_argument('--save-heads',action='store_true')
            q.add_argument('--resume-from',type=Path,help='reuse only byte/geometry/model/backend-identical completed checkpoints')
        if name=='web':q.add_argument('--output',type=Path,default=Path('out/acdc/jobs'))
        if name in ('web','worker'):q.add_argument('--port',type=int,default=8092 if name=='web' else 8093)
        q.add_argument('--geometry-policy',choices=('strict','acdc-direction-only-sform-v1'),default='strict')
        q.add_argument('--recognized-acdc',action='store_true',help='explicit operator attestation of recognized ACDC source')
    q=sub.add_parser('phantom');q.add_argument('output',type=Path)
    q=sub.add_parser('synthetic-model');q.add_argument('output',type=Path)
    q=sub.add_parser('import-release',help='import the authenticated Axial model and required research evidence')
    q.add_argument('source',type=Path);q.add_argument('output',type=Path)
    q.add_argument('--model-spec-sha',required=True)
    q=sub.add_parser('verify-model');q.add_argument('model',type=Path);q.add_argument('--model-spec-sha');q.add_argument('--allow-synthetic',action='store_true')
    q.add_argument('--full',action='store_true',help='execute full 256x256 integer reference/emulator and compare all 21 nodes')
    q=sub.add_parser('package');q.add_argument('output',type=Path)
    q=sub.add_parser('evaluate');q.add_argument('--cases',type=Path,required=True)
    q.add_argument('--model-spec-sha',required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('runtime');q.add_argument('--output',type=Path,required=True);q.add_argument('--build',type=Path,required=True)
    q.add_argument('--model',type=Path,required=True);q.add_argument('--model-spec-sha');q.add_argument('--allow-synthetic',action='store_true')
    return p


def backend(args):
    from .backend import DemoBackend,GraphBackend,RemoteBackend
    from .model import load_bundle
    if args.backend=='demo':return DemoBackend()
    if args.model is None:raise ValueError('--model is required except for explicit analytic demo')
    model=load_bundle(args.model,expected_sha=args.model_spec_sha,allow_synthetic=args.allow_synthetic)
    if args.backend=='reference_torch':
        from .backend import TorchReferenceBackend
        return TorchReferenceBackend(model)
    if args.backend=='pl' and args.runtime:
        cert=json.loads((args.runtime/'overlay/build_manifest.json').read_text('utf-8'))
        if cert.get('hardware_id')==0x41434438:
            from .backend import SpatialBackend
            return SpatialBackend(model,args.runtime)
    if args.backend=='remote':
        if not args.board_url:raise ValueError('--board-url required')
        return RemoteBackend(args.board_url,model_sha=model.model_sha256,head_scale=model.head_scale,
                             synthetic=model.synthetic,binding=model.implementation_binding)
    return GraphBackend(model,args.backend,args.runtime)


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='plan':
        from .planning import workload
        print(json.dumps(workload(),indent=2));return
    if args.command=='phantom':
        from .volume import make_phantom
        print(make_phantom(args.output));return
    if args.command=='synthetic-model':
        from .model import create_synthetic_bundle
        print(create_synthetic_bundle(args.output));return
    if args.command=='import-release':
        from .release import import_release
        model=import_release(args.source,args.output,expected_sha=args.model_spec_sha)
        print(json.dumps(dict(model_sha256=model.model_sha256,head_scale=model.head_scale,
                              members=len(model.spec['release_members']),qualification=model.spec['qualification'])));return
    if args.command=='verify-model':
        from .model import load_bundle
        model=load_bundle(args.model,expected_sha=args.model_spec_sha,allow_synthetic=args.allow_synthetic)
        report=dict(model_sha256=model.model_sha256,synthetic=model.synthetic,head_scale=model.head_scale,
                    scope='bundle contract and qualification records')
        if args.full:
            from .verify import verify_model
            report['full_numerics']=verify_model(model)
        print(json.dumps(report));return
    if args.command=='package':
        from .package import source_package
        print(source_package(args.output));return
    if args.command=='evaluate':
        from .evaluate import evaluate
        from .contract import atomic_json
        if args.output.exists():raise ValueError('choose a new evaluation report')
        atomic_json(args.output,evaluate(args.cases,args.model_spec_sha));print(args.output);return
    if args.command=='runtime':
        from .model import load_bundle
        from .package import runtime_package
        m=load_bundle(args.model,expected_sha=args.model_spec_sha,allow_synthetic=args.allow_synthetic)
        print(runtime_package(args.output,args.build,m));return
    runner=backend(args)
    if args.command=='web':
        from .web import serve
        serve(runner,args.output,port=args.port,policy=args.geometry_policy,recognized_acdc=args.recognized_acdc)
    elif args.command=='worker':
        from .worker import make_worker
        server=make_worker(runner,args.port)
        from .lifecycle import graceful_signals
        graceful_signals(server)
        print(f'ACDC worker loopback port {server.server_port}; backend={runner.kind}; synthetic={runner.synthetic}',flush=True)
        try:server.serve_forever()
        finally:server.server_close();runner.close()
    else:
        from .pipeline import Pipeline
        try:
            result=Pipeline(runner,policy=args.geometry_policy,recognized_acdc=args.recognized_acdc).run(
                args.ed,args.es,args.output,save_heads=args.save_heads,resume_from=args.resume_from)
            print(json.dumps(result))
        finally:runner.close()
