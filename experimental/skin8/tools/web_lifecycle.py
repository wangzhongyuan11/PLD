"""Application pause/resume around the unchanged Skin8 frame driver."""
import json
import os


class Ownership:
    def __init__(self):self.handle=None

    def acquire(self):
        if self.handle is not None:return
        import fcntl
        handle=open('/run/pld-unified-fpga.lock','a+b')
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:handle.close();raise
        self.handle=handle

    def release(self):
        if self.handle is not None:self.handle.close();self.handle=None


OWNER=Ownership()


def install(web,app,kind,pipeline_factory=None):
    app.suspended=False
    original=web.Handler.do_POST
    original_get=web.Handler.do_GET
    def get(handler):
        if handler.path.split('?',1)[0]!='/api/ownership':return original_get(handler)
        if handler.client_address[0]!='127.0.0.1':
            return handler.send(403,dict(error='board-local ownership')) if kind=='skin' else handler.json(dict(error='board-local ownership'),403)
        value=dict(lifecycle='pause-resume-v1',suspended=app.suspended,hardware_released=OWNER.handle is None,
                   hdmi='off' if app.suspended else 'active',active=not app.suspended,fault=app.fault,owner_pid=os.getpid())
        return handler.send(200,value) if kind=='skin' else handler.json(value)
    web.Handler.do_GET=get
    if kind=='skin':
        status=app.status
        def snapshot():
            result=status()
            result.update(lifecycle='pause-resume-v1',suspended=app.suspended,
                          hardware_released=OWNER.handle is None,runtime_identity=getattr(app,'runtime_identity',None),owner_pid=os.getpid())
            core=getattr(app.b_worker,'core',None)
            result['hardware_ready']=bool(OWNER.handle is not None and core is not None and not core.session.closed and not app.suspended)
            if app.suspended:result['inference_available']=False
            return result
        app.status=snapshot

    def post(handler):
        path=handler.path.split('?',1)[0]
        def reply(code,value):
            return handler.send(code,value) if kind=='skin' else handler.json(value,code)
        if path!='/api/lifecycle':
            if app.suspended and path in ('/api/jobs','/api/hdmi'):
                return reply(409,dict(error='模型已暂停，请先通过统一入口选择该模型'))
            return original(handler)
        if handler.client_address[0]!='127.0.0.1':return reply(403,dict(error='board-local lifecycle'))
        size=int(handler.headers.get('Content-Length','0'))
        if not 0<size<=1024:return reply(400,dict(error='invalid lifecycle request'))
        body=json.loads(handler.rfile.read(size));action=body.get('action')
        if action not in ('pause','resume'):return reply(400,dict(error='invalid lifecycle action'))
        acquired=False
        try:
            if kind=='skin':
                acquired=app.lock.acquire(blocking=False)
                if not acquired or app.active or app.warmup_state=='running':return reply(409,dict(error='模型处理中'))
            else:
                acquired=app.busy.acquire(blocking=False)
                if not acquired:return reply(409,dict(error='模型处理中'))
            if action=='pause' and not app.suspended:
                app.stop_hdmi()
                if kind=='skin':
                    if app.b_worker:app.b_worker.release_hardware(app.data/'pause_release.log')
                else:
                    app.pipeline.close();app.released=True
                OWNER.release();app.suspended=True
            elif action=='resume' and app.suspended:
                OWNER.acquire()
                if kind=='skin':
                    app.warmup_state='running';app._warmup()
                    if app.warmup_state!='ready':raise RuntimeError('皮肤镜恢复预热失败')
                else:
                    if hasattr(app.pipeline,'restore_hardware'):app.pipeline.restore_hardware()
                    else:app.pipeline=pipeline_factory(app.pipeline.model.root,'pl',app.runtime)
                    app.released=False
                app.suspended=False
            return reply(200,dict(suspended=app.suspended,hardware_released=OWNER.handle is None))
        except Exception as exc:
            app.fault=str(exc) if kind=='skin' else True
            return reply(409,dict(error=str(exc)))
        finally:
            if acquired:
                if kind=='skin':app.lock.release()
                else:app.busy.release()
    web.Handler.do_POST=post
