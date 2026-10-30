"""Original presentation with Skin8 inference and guarded hardware lifecycle."""
import argparse,hashlib,importlib,json,os,signal,sys,threading,time
from pathlib import Path
BOOT_STARTED=time.monotonic()
import numpy as np
from web_lifecycle import OWNER
HERE=Path(__file__).resolve().parent
CONFIG=dict(original_release='/home/xilinx/jupyter_notebooks/pld/releases/pld_dual_20261017',workspace='/home/xilinx/jupyter_notebooks/PLD_Dev',data='/home/xilinx/skin8_web_data')
BASE=Path(CONFIG['original_release'])
CORE_ROOT=Path(CONFIG['workspace'])
DATA=Path(CONFIG['data'])
sys.path.insert(0,str(CORE_ROOT))

def startup_stage(name):
    print('PLD_STARTUP',json.dumps(dict(stage=name,monotonic=time.monotonic(),seconds=time.monotonic()-BOOT_STARTED)),flush=True)

startup_stage('numpy_import')

def prepare_core_software():
    # Import classes only: no device selection, Overlay, MMIO or CMA allocation.
    # Do this before the HDMI child starts competing for startup CPU time.
    from pynq import Overlay,MMIO,allocate
    from frame_driver import FrameSession
    startup_stage('pure_core_imports')

def runtime_identity(kind):
    if kind=='skin':return 'skin8-processing-only'
    return 'skin8-processing-only:'+kind

class Core:
    def __init__(self,kind,prepared=None):
        from latency import Stages
        trace=Stages('skin8_'+kind+'_core')
        OWNER.acquire()
        import asyncio
        try:asyncio.get_event_loop()
        except RuntimeError:asyncio.set_event_loop(asyncio.new_event_loop())
        from types import SimpleNamespace
        from pynq import Overlay,MMIO,allocate
        from overlay_metadata import enable_metadata_cache
        enable_metadata_cache(verify=False)
        from frame_driver import FrameSession
        from pynq.pl_server.global_state import load_global_state, global_state_file_exists
        trace.mark('pynq_frame_driver_imports')
        root=BASE/'skin/role_a_final/int8' if kind=='skin' else BASE/'camus/model/int8_export'
        raw=(root/'model_spec.json').read_bytes()
        manifest=json.loads((HERE/'models'/kind/'manifest.json').read_text())
        spec=json.loads(raw)
        self.model=SimpleNamespace(layers=spec['layers'],spec_sha256=manifest['spec_sha256'])
        trace.mark('model_load')
        bit=HERE/'skin8.bit'
        self.bit_sha256=hashlib.sha256(bit.read_bytes()).hexdigest()
        # Overlay(download=False) updates process-local device metadata; its
        # is_loaded() is insufficient to identify a download by another process.
        loaded=global_state_file_exists() and load_global_state().bitfile_name==str(bit)
        self.overlay=Overlay(str(bit),download=False)
        trace.mark('metadata_bind')
        if not loaded:self.overlay.download()
        trace.mark('program_overlay')
        self.session=FrameSession(HERE/'models'/kind,MMIO(0x43c00000,0x1000),allocate,prepared=prepared,verify=False)
        trace.mark('frame_session')
        self.startup_timing=trace.snapshot()
    def run(self,q):
        head,t=self.session.run(q,timeout=10)
        return dict(head=head,host_timing=dict(seconds=t['input_to_owned_head_seconds'],**t))
    def close(self):
        if self.session.closed:return
        if not self.overlay.is_loaded():
            # The old register address may now target unrelated hardware or an
            # unresponsive AXI slave. Do not read MMIO or release DMA memory.
            raise RuntimeError('FPGA configuration changed; retain buffers and inspect hardware before recovery')
        self.session.close()


class SkinWorker:
    def __init__(self,runtime,*args,**kwargs):
        self.core=None;self.runtime=runtime;self.failure=None;self.prepared=None
    def check_manifest(self):
        assert (HERE/'skin8.bit').is_file()
    def run(self,argv,log_path):
        from persistent_worker import PersistentWorkerError
        if self.failure:raise PersistentWorkerError(self.failure)
        try:return self._run(argv,log_path)
        except Exception as exc:
            self.failure=str(exc)
            raise PersistentWorkerError(self.failure) from exc
    def _run(self,argv,log_path):
        from reference.preprocess import decode_rgb,letterbox_int8
        started=time.perf_counter();reused=self.core is not None
        artifacts_reused=self.prepared is not None
        if not self.core:
            self.core=Core('skin',prepared=self.prepared)
            self.prepared=self.core.session.prepared
        startup=time.perf_counter()-started
        if '--warmup-only' not in argv:
            source=Path(argv[0]);out=Path(argv[argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=False)
            t=time.perf_counter()
            with decode_rgb(source) as image:q,valid,g=letterbox_int8(image,self.core.model.layers[0]['input_scale'])
            prep=time.perf_counter()-t
            result=self.core.run(q[0]);head=result['head'][None];inference=result['host_timing']['seconds']
            rect=[g['left'],g['top'],g['resized_width'],g['resized_height']]
            spec=self.core.model.spec_sha256
            metadata=dict(g,model_spec_sha256=spec,valid_rect=rect,valid_pixels=rect[2]*rect[3])
            report=dict(status='success',processing_only=True,shape=[1,1,256,256],dtype='int8',layout='NCHW',comparison='>=',model_name='soup_v5_a050',model_spec_sha256=spec,model_manifest_sha256='e47eb18b94369451b36790971f4e0db62b2d128f424937e00f2768e866a30936',threshold_q=-1,threshold_probability=.45,head_output_scale=self.core.model.layers[-1]['output_scale'],frame_id=1,valid_rect=rect,input_metadata=metadata,online_b_seconds=prep+inference,inference_seconds=inference,preprocess_seconds=prep,abi_version='SKIN8-v0',execution_backend='skin8_125',execution_phases=result['host_timing'],source_name=source.name)
            report['bit_sha256']=self.core.bit_sha256
            (out/'frame_000001.bin').write_bytes(head.tobytes());(out/'frame_000001.json').write_text(json.dumps(report,indent=2))
            Path(log_path).write_text('SKIN8_SKIN_INFERENCE_COMPLETED_NOT_REFERENCE_CHECKED\n'+json.dumps(result['host_timing'],indent=2))
        return dict(model_reused=reused,hardware_reused=reused,artifacts_reused=artifacts_reused,
                    core_startup_stages={} if reused else self.core.startup_timing['stages_seconds'],
                    startup_seconds=startup,request_seconds=time.perf_counter()-started,worker_pid=os.getpid(),alive_after_reply=True)
    def release_hardware(self,*args):
        # Keep CPU parameter bytes across HDMI, but release every hardware buffer.
        if self.core:self.core.close();self.core=None
    def close(self):
        self.release_hardware()
        self.prepared=None


def serve_skin(display_agent=None):
    sys.path.insert(0,str(BASE/'web'))
    import server as old
    startup_stage('skin_web_import')
    prepare_core_software()
    from hdmi_policy import guard_skin
    guard_skin(old)
    old.PersistentBWorker=SkinWorker
    token=Path(json.loads((BASE/'hub/config.json').read_text())['token_file']).read_text().strip()
    app=old.Console(DATA/'skin',BASE/'skin',token=token,persistent_b_worker=True,display_agent=display_agent,processing_only=True)
    startup_stage('skin_console_init')
    app.runtime_identity=runtime_identity('skin')
    warmup=app._warmup
    def warmup_then_prepare_display():
        warmup()
        if app.warmup_state=='ready' and display_agent:display_agent.prepare()
    app._warmup=warmup_then_prepare_display
    server=old.make_server(app,'127.0.0.1',8766)
    from web_lifecycle import install
    install(old,app,'skin')
    def stop(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    app.begin_warmup()
    startup_stage('skin_http_ready')
    try:server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        while app.active:time.sleep(.2)
        app.stop_hdmi()
        app.close()


def serve_camus(display_agent=None):
    sys.path.insert(0,str(BASE/'hub'))
    sys.path.insert(1,str(BASE/'hub'))
    from camus import web
    from camus.pipeline import CamusPipeline
    startup_stage('camus_web_import')
    class Pipeline(CamusPipeline):
        def __init__(self,model_root,backend='pl',runtime=None,timeout=3.):
            if backend!='pl':super().__init__(model_root,backend,runtime,timeout);self.core=None;return
            t=time.perf_counter();super().__init__(model_root,'reference',verify=False)
            from camus.native import load_native
            # Use only the existing PS image operations; no CAM2 MMIO binding.
            self.native=load_native()
            self.core=Core('camus');self.backend='pl'
            pipeline=self
            class Network:
                def forward(self,x):
                    r=pipeline.core.run(x[0]);return dict(head=r['head'][None],events=r.get('events',[]),conv_records=r.get('records',[]))
            self.network=Network();self.initialization_seconds=time.perf_counter()-t
        def restore_hardware(self):
            if self.core is not None:raise RuntimeError('CAMUS hardware already active')
            self.core=Core('camus')
        def close(self):
            if self.core:self.core.close();self.core=None
    web.CamusPipeline=Pipeline
    from camus_input_adapter import install
    install(web)
    from camus_quad import install as quad
    quad(web)
    from hdmi_policy import guard_camus
    guard_camus(web)
    original=web.Handler.do_GET
    def get(handler):
        if web.urlsplit(handler.path).path=='/api/status':
            hdmi=handler.server.hdmi_status()
            core=handler.server.pipeline.core
            with handler.server.guard:active=any(j.get('state') in ('queued','running') for j in handler.server.jobs.values())
            return handler.json(dict(backend=handler.server.pipeline.backend,spec_sha256=web.MODEL_SPEC_SHA256,ready=not handler.server.fault and not getattr(handler.server,'suspended',False),hdmi=hdmi,hdmi_available=not handler.server.fault,active=active,fault=handler.server.fault,model='CAMUS hidden_histogram INT8',lifecycle='pause-resume-v1',suspended=getattr(handler.server,'suspended',False),hardware_released=OWNER.handle is None,runtime_identity=handler.server.runtime_identity,owner_pid=os.getpid(),hardware_ready=bool(OWNER.handle is not None and core is not None and not core.session.closed)))
        return original(handler)
    web.Handler.do_GET=get
    server_init=web.CamusServer.__init__
    def init(server,*args,**kwargs):
        kwargs['display_agent']=display_agent
        server_init(server,*args,**kwargs)
        server.runtime_identity=runtime_identity('camus')
        from web_lifecycle import install
        install(web,server,'camus',Pipeline)
    web.CamusServer.__init__=init
    try:
        if display_agent is None and os.environ.get('PLD_WARM_HDMI')=='1':
            from warm_hdmi import WarmHDMI
            display_agent=WarmHDMI(DATA/'camus/warm_hdmi.log');display_agent.prepare()
        web.main(['--backend','pl','--runtime',str(BASE/'camus'),'--model',str(BASE/'camus/model/int8_export'),'--host','127.0.0.1','--port','8090','--data',str(DATA/'camus')])
    finally:
        if display_agent:display_agent.close()


def serve_hub():
    sys.path.insert(0,str(BASE/'hub'))
    import hub
    hub.UNITS={'skin':'pld-original-rtl-skin.service','camus':'pld-original-rtl-camus.service'}
    sys.argv=[sys.argv[0],'--config',str(HERE/'hub/config.json')]
    hub.main()

if __name__=='__main__':
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('skin','camus','hub'));a=p.parse_args()
    DATA.mkdir(parents=True,exist_ok=True)
    if a.kind=='hub':serve_hub()
    else:
        # Parent lease covers inference AND original HDMI subprocess lifetime.
        OWNER.acquire()
        display_agent=None
        try:
            if a.kind=='skin' and os.environ.get('PLD_WARM_HDMI')=='1':
                sys.path.insert(0,str(BASE/('web' if a.kind=='skin' else 'hub')))
                from warm_hdmi import WarmHDMI
                display_agent=WarmHDMI(DATA/a.kind/'warm_hdmi.log')
            (serve_skin if a.kind=='skin' else serve_camus)(display_agent)
        finally:
            if display_agent:display_agent.close()
            OWNER.release()
