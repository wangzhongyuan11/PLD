"""Independent board engine/Web and numerical validation entry points."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for command in ('engine','web'):
        q=commands.add_parser(command,help='production standalone '+command)
        q.add_argument('--config',type=Path,required=True);q.add_argument('--app-sha',required=True)
    p=commands.add_parser('numerics',help='real board ED/ES numerical path; excludes meshes and Web')
    for name in ('model','runtime','assets','ed','es','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model-sha',required=True);p.add_argument('--assets-sha',required=True)
    p.add_argument('--geometry-policy',choices=('strict','acdc-direction-only-sform-v1'),default='strict')
    p.add_argument('--serial',action='store_true')
    args=parser.parse_args()
    if args.command in ('engine','web'):
        production(args);return
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


def production(args):
    from display_web.latency import Stages
    timing=Stages('acdc_'+args.command+'_startup')
    from .identity import verify
    root=args.config.resolve().parent
    config,identity=verify(root,args.app_sha,check=False)
    timing.mark('application_load')
    data=Path(config['data_root']).resolve();data.mkdir(parents=True,exist_ok=True)
    from .native import load_native
    kernel=load_native()
    timing.mark('native_image')
    if args.command=='web':
        from .engine import Client
        from .web import WebApplication,serve
        app=WebApplication(Client(config['socket']),data,kernel)
        serve(app,config.get('host','0.0.0.0'),config.get('port',8092));return
    from acdc.model import load_bundle
    from acdc.backend import SpatialBackend
    from .assets import load
    from .pipeline import NumericalPipeline
    from .full_pipeline import FullPipeline
    from .surface import SurfaceGenerator
    from .engine import JobManager,serve
    import nibabel
    timing.mark('engine_imports')
    began=time.monotonic()
    model=load_bundle(root/config['model'],expected_sha=config['model_sha256'],verify=False)
    timing.mark('model_load')
    table=load(model,root/config['assets'],config['assets_sha256'],verify=False)
    timing.mark('assets_load')
    meshes=SurfaceGenerator(root/config['mesh_binary'],config['mesh_binary_sha256'],verify=False)
    timing.mark('mesh_generator')
    from .startup_cache import install_plan_cache
    install_plan_cache(root)
    resident_handoff=os.environ.get('PLD_RESIDENT_MODELS')=='1'
    if resident_handoff:
        from .resident_lease import install
        install()
    from display_web.overlay_metadata import enable_metadata_cache
    enable_metadata_cache(verify=False)
    display_agent=None
    if os.environ.get('PLD_WARM_HDMI')=='1':
        from display_web.warm_hdmi import WarmHDMI
        display_agent=WarmHDMI(data/'warm_hdmi.log')
    def make_backend():
        trace=Stages('acdc_backend')
        if os.environ.get('PLD_PROFILE_STARTUP')=='1':
            import cProfile,pstats
            profiler=cProfile.Profile();profiler.enable()
            try:return SpatialBackend(model,root,verify=False)
            finally:
                profiler.disable();pstats.Stats(profiler).sort_stats('cumulative').print_stats(18)
                trace.mark('initialize')
        try:return SpatialBackend(model,root,verify=False)
        finally:trace.mark('initialize')
    backend=make_backend()
    timing.mark('backend')
    manager=None
    try:
        warm=None
        if display_agent:display_agent.prepare()
        numerical=NumericalPipeline(backend,kernel,table,binding=model.implementation_binding,
                                    policy=config.get('geometry_policy','strict'),verify=False)
        manager=JobManager(FullPipeline(numerical,meshes,application_sha=args.app_sha),data,initialization=dict(
            ready_seconds=time.monotonic()-began,pl_selftest=warm,application_sha256=args.app_sha,
            timing=timing.snapshot()),backend_factory=make_backend,display_agent=display_agent,
            backend_check=None,resident_handoff=resident_handoff)
        print(json.dumps(dict(status='ready',initialization=manager.initialization)),flush=True)
        def notify_ready():
            if os.environ.get('NOTIFY_SOCKET'):
                import socket
                address=os.environ['NOTIFY_SOCKET']
                if address.startswith('@'):address='\0'+address[1:]
                with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as channel:
                    channel.connect(address);channel.sendall(b'READY=1\nSTATUS=ACDC model and mesh ready')
        serve(manager,config['socket'],on_ready=notify_ready)
    finally:
        if manager is None:
            backend.close()
            if display_agent:display_agent.close()
        elif manager.thread is None or not manager.thread.is_alive():manager.pipeline.backend.close()


if __name__=='__main__':main()
