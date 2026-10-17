"""One shared FPGA lease for either existing, otherwise unmodified backend."""
import argparse
import fcntl
import os
from pathlib import Path
import runpy
import signal
import sys
import threading

def install_skin_shutdown(worker_module):
    """EOF closes the idle worker through its finally block, without SIGKILL."""
    original=worker_module.PersistentBWorker._stop
    def stop(self,reason):
        if reason!='closed_by_service':return original(self,reason)
        self.close_reason=reason
        process=self.process
        if process is not None:
            if process.poll() is None:
                process.stdin.close()
                process.wait(timeout=60)
            if process.returncode!=0:raise RuntimeError('旧推理进程未正常释放，禁止切换模型')
            process.stdout.close()
            if self.reader is not None and self.reader is not threading.current_thread():self.reader.join(timeout=5)
        self._shutdown_launcher()
    worker_module.PersistentBWorker._stop=stop

def main():
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['skin','camus'])
    p.add_argument('source',type=Path)
    p.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    # Parent keeps the lease while its model/HDMI subprocesses run. systemd
    # verifies the whole cgroup has exited before another backend can start.
    with open('/run/pld-unified-fpga.lock','a+b') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=str(a.source.resolve());sys.path.insert(0,source);os.chdir(source)
        argv=a.args[1:] if a.args[:1]==['--'] else a.args
        if a.kind=='skin':
            signal.signal(signal.SIGINT,signal.default_int_handler)
            import persistent_worker
            install_skin_shutdown(persistent_worker)
            sys.argv=[source+'/server.py']+argv
            runpy.run_path(sys.argv[0],run_name='__main__')
        else:
            from camus import web
            from camus_input_adapter import install
            install(web)
            from camus_quad import install as install_quad
            install_quad(web)
            original=web.Handler.do_GET
            def get(handler):
                if web.urlsplit(handler.path).path=='/api/status':
                    hdmi=handler.server.hdmi_status()
                    with handler.server.guard:
                        active=any(j.get('state') in ('queued','running') for j in handler.server.jobs.values())
                        errors=[dict(id=j['id'],error=j.get('error'),error_kind=j.get('error_kind','runtime'),close_error=j.get('close_error')) for j in handler.server.jobs.values() if j.get('state')=='error']
                    return handler.json(dict(backend=handler.server.pipeline.backend,
                        spec_sha256=web.MODEL_SPEC_SHA256,ready=not handler.server.fault,
                        hdmi=hdmi,active=active,fault=handler.server.fault,last_error=errors[-1] if errors else None,model='CAMUS hidden_histogram INT8'))
                return original(handler)
            web.Handler.do_GET=get
            web.main(argv)

if __name__=='__main__':main()
