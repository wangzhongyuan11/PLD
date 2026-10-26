"""A display child parks after closing video hardware; no fork of XRT state."""
import argparse
import contextlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import uuid


class SharedWorker:
    """One checked connection to the board-local resident display service."""
    def __init__(self, path):
        self.socket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.socket.connect(path)
        self.stdin=self.socket.makefile('w',buffering=1)
        self.stdout=self.socket.makefile('r')
        self.finished=threading.Event();self.returncode=None;self.pid=None

    def poll(self):return self.returncode if self.finished.is_set() else None

    def eof(self):
        if self.returncode is None:self.returncode=1
        self.stdin.close();self.stdout.close();self.socket.close();self.finished.set()

    def wait(self,timeout=None):
        if not self.finished.wait(timeout):raise subprocess.TimeoutExpired('shared HDMI close',timeout)
        return self.returncode


class DisplaySession:
    def __init__(self, agent, ident):
        self.agent = agent; self.ident = ident
        self.finished = threading.Event(); self.returncode = None

    def poll(self):
        if self.returncode is None and self.agent.process.poll() is not None:
            self.returncode = self.agent.process.returncode or 1
            self.finished.set()
        return self.returncode

    def terminate(self): self.agent.send(dict(action='stop', id=self.ident))

    def update(self, view):self.agent.send(dict(action='update',id=self.ident,view=view))

    def wait(self, timeout=None):
        if not self.finished.wait(timeout):
            if self.poll() is None: raise subprocess.TimeoutExpired('resident HDMI close', timeout)
        return self.returncode


class WarmHDMI:
    def __init__(self, log, *, shared_socket=None):
        self.log = Path(log); self.process = None; self.session = None
        self.ready = threading.Event(); self.lock = threading.Lock()
        self.error = ''
        self.parked = threading.Event(); self.park_id = None; self.park_error = ''
        self.shared_socket=os.environ.get('PLD_SHARED_HDMI_SOCKET','') if shared_socket is None else shared_socket

    def prepare(self):
        with self.lock:
            if self.process is not None: return
            if self.shared_socket:
                self.process=SharedWorker(self.shared_socket)
                threading.Thread(target=self.read,daemon=True,name='shared-hdmi-responses').start()
                return
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open('a') as output:
                self.process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=output, text=True, bufsize=1,
                    env=dict(os.environ,PYTHONPATH=os.pathsep.join(p for p in sys.path if p)))
            threading.Thread(target=self.read, daemon=True, name='hdmi-responses').start()

    def read(self):
        for line in self.process.stdout:
            try: value = json.loads(line)
            except ValueError: continue
            if isinstance(self.process,SharedWorker):
                if value.get('state')=='ready':self.process.pid=value.get('worker_pid')
                if value.get('state')=='closed':self.process.returncode=int(value.get('returncode',1))
            if value.get('id') == self.park_id and value.get('state') in ('parked','error'):
                self.park_error = value.get('error','');self.parked.set()
            if value.get('state') == 'ready': self.ready.set()
            if value.get('state') == 'error':
                self.error = value.get('error', 'HDMI failed'); self.ready.set()
            session = self.session
            if session and value.get('id') == session.ident and value.get('state') in ('idle', 'error'):
                session.returncode = 1 if value['state'] == 'error' else 0
                session.finished.set()
        if isinstance(self.process,SharedWorker):self.process.eof()
        self.ready.set()
        if self.session and self.session.returncode is None:
            self.session.returncode = self.process.wait() or 1; self.session.finished.set()

    def send(self, body):
        with self.lock:
            if not self.process or self.process.poll() is not None:
                raise RuntimeError('resident HDMI process unavailable; release unconfirmed')
            self.process.stdin.write(json.dumps(body)+'\n'); self.process.stdin.flush()

    def start(self, command, log, *, pythonpath=None):
        self.prepare()
        if not self.ready.wait(90): raise RuntimeError('resident HDMI initialization timed out')
        if self.error: raise RuntimeError(self.error)
        if self.session and self.session.poll() is None: raise RuntimeError('HDMI already active')
        self.session = DisplaySession(self, uuid.uuid4().hex)
        self.send(dict(action='show', id=self.session.ident, command=[str(x) for x in command], log=str(log),
                       pythonpath=list(sys.path) if pythonpath is None else pythonpath))
        return self.session

    def close(self):
        if self.process is None: return
        if self.process.poll() is not None:
            if self.process.returncode: raise RuntimeError('resident HDMI process failed')
            return
        self.send(dict(action='quit'))
        self.process.wait(timeout=90)
        if self.process.returncode: raise RuntimeError('resident HDMI did not close normally')

    def park(self):
        if self.process is None:return
        if self.session and self.session.poll() is None:raise RuntimeError('close HDMI before parking metadata')
        if not self.ready.wait(90):raise RuntimeError('HDMI software initialization timed out')
        if self.error:raise RuntimeError(self.error)
        if not self.shared_socket:return
        self.park_id=uuid.uuid4().hex;self.park_error='';self.parked.clear()
        self.send(dict(action='park',id=self.park_id))
        if not self.parked.wait(30):raise RuntimeError('HDMI metadata park timed out')
        if self.park_error:raise RuntimeError(self.park_error)


def broker(path):
    """Serialize unchanged renderers in one SDK process across all models."""
    path=Path(path)
    agent=WarmHDMI(path.parent/'worker.log',shared_socket='')
    guard=threading.Lock();owner=[None]
    class Handler(socketserver.StreamRequestHandler):
        def setup(self):
            super().setup();self.output_lock=threading.Lock();self.session=None;self.failed=False;self.watcher=None

        def reply(self,**value):
            with self.output_lock:
                try:self.wfile.write((json.dumps(value)+'\n').encode());self.wfile.flush()
                except OSError:pass  # A disconnected owner is closed below.

        def release(self):
            if self.session and self.session.poll() is None:
                self.session.terminate();self.session.wait(90)
            if self.watcher:
                self.watcher.join(10)
                if self.watcher.is_alive():raise RuntimeError('HDMI cleanup acknowledgement did not complete')
            if self.session and self.session.returncode:self.failed=True
            with guard:
                if owner[0] is self:owner[0]=None

        def monitor(self,session,ident):
            code=session.wait()
            with guard:
                if owner[0] is self and self.session is session:owner[0]=None
            if code:self.failed=True
            self.reply(state='error' if code else 'idle',id=ident,
                       **({'error':'shared HDMI renderer failed; inspect worker log'} if code else {}))

        def handle(self):
            ident=None
            try:
                agent.prepare()
                if not agent.ready.wait(90):raise RuntimeError('shared HDMI imports timed out')
                if agent.error:raise RuntimeError(agent.error)
                self.reply(state='ready',worker_pid=agent.process.pid)
                for raw in self.rfile:
                    body=json.loads(raw);ident=body.get('id');action=body.get('action')
                    if action=='show':
                        with guard:
                            if owner[0] is not None:raise RuntimeError('another HDMI session is still active')
                            self.session=agent.start(body['command'],body['log'],pythonpath=body.get('pythonpath'));owner[0]=self
                        self.watcher=threading.Thread(target=self.monitor,args=(self.session,ident),daemon=True)
                        self.watcher.start()
                    elif action in ('stop','update'):
                        if self.session and ident==getattr(self,'client_id',None):
                            if action=='stop':self.session.terminate()
                            else:self.session.update(body['view'])
                    elif action=='park':
                        if self.session and self.session.poll() is None:raise RuntimeError('close HDMI before parking')
                        self.reply(state='parked',id=ident)
                    elif action=='quit':
                        self.release();self.reply(state='closed',returncode=int(self.failed));return
                    else:raise RuntimeError('unknown shared HDMI command')
                    if action=='show':self.client_id=ident
            except Exception as exc:
                self.failed=True;self.reply(state='error',id=ident,error=str(exc))
            finally:
                try:self.release()
                except Exception as exc:self.reply(state='error',id=ident,error=str(exc))

    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads=True
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as probe:
            try:probe.connect(str(path))
            except ConnectionRefusedError:path.unlink()
            else:raise RuntimeError('shared HDMI broker already running')
    server=Server(str(path),Handler);path.chmod(0o600)
    def shutdown(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
    if os.environ.get('NOTIFY_SOCKET'):
        address=os.environ['NOTIFY_SOCKET']
        with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as notify:
            notify.connect('\0'+address[1:] if address.startswith('@') else address)
            notify.sendall(b'READY=1')
    try:server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        try:agent.close()
        finally:path.unlink(missing_ok=True)


def worker():
    output = sys.stdout
    def respond(**value):
        output.write(json.dumps(value)+'\n'); output.flush()
    commands = queue.Queue(); stop = threading.Event(); quitting = threading.Event()
    active = [None]
    def reader():
        for line in sys.stdin:
            try: body = json.loads(line)
            except ValueError: continue
            if body.get('action') == 'quit': quitting.set(); stop.set(); commands.put(None)
            elif body.get('action') == 'stop':
                if body.get('id') == active[0]: stop.set()
            elif body.get('action') == 'update':
                if body.get('id') == active[0]:stop.display_update=body
            elif body.get('action') == 'show':
                # Publish the ID before the main thread starts initialization,
                # so even a stop immediately after show is retained.
                if active[0] is not None:
                    respond(state='error', id=body.get('id'), error='HDMI already active')
                else:
                    active[0] = body['id']; stop.display_update=None; stop.clear(); commands.put(body)
        quitting.set(); stop.set(); commands.put(None)
    threading.Thread(target=reader, daemon=True).start()
    def shutdown(*_): quitting.set(); stop.set(); commands.put(None)
    signal.signal(signal.SIGTERM, shutdown); signal.signal(signal.SIGINT, shutdown)
    with contextlib.redirect_stdout(sys.stderr):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        # Prepare pure software metadata. This does not download a bitstream,
        # instantiate video drivers, read MMIO or allocate a video frame.
        import pynq
        import pynq.lib.video
        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from overlay_metadata import enable_metadata_cache
        enable_metadata_cache()
        if quitting.is_set():return
        try:video_module=importlib.import_module('display_web.video_overlay')
        except ModuleNotFoundError:video_module=importlib.import_module('video_overlay')
        # All model renderers share the same prepared software description in
        # this child, regardless of package/standalone import conventions.
        sys.modules['video_overlay']=video_module
        sys.modules['display_web.video_overlay']=video_module
        # Keep imports warm, but allocate the large video metadata graph only
        # when a display is requested. No competing graph build at model startup.
    respond(state='ready')
    modules = {}; failed = False
    while not quitting.is_set():
        body = commands.get()
        if body is None: break
        ident = body['id']
        try:
            if quitting.is_set(): break
            command = body['command']
            with Path(body['log']).open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                if 'pythonpath' in body:sys.path[:]=body['pythonpath']
                if '-m' in command:
                    index = command.index('-m'); key = command[index+1]; argv = command[index+2:]
                    if key not in modules: modules[key] = importlib.import_module(key)
                else:
                    index = next(i for i in range(1,len(command)) if not command[i].startswith('-'))
                    script = Path(command[index]).resolve(); key = str(script); argv = command[index+1:]
                    if key not in modules:
                        sys.path.insert(0, str(script.parent))
                        spec = importlib.util.spec_from_file_location('pld_hdmi_'+str(len(modules)), script)
                        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); modules[key] = module
                modules[key].main(argv=argv, stop=stop)
            active[0] = None
            respond(state='idle', id=ident)
        except Exception as exc:
            failed = True
            active[0] = None
            respond(state='error', id=ident, error=type(exc).__name__+': '+str(exc))
        finally:
            if active[0] == ident: active[0] = None
    return 1 if failed else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--broker',type=Path); args = parser.parse_args()
    if args.broker:broker(args.broker)
    if args.worker:
        try:
            code=worker()
            # main() has returned through video/touch cleanup, and each session
            # has acknowledged idle/error. Avoid expensive interpreter teardown
            # of the software-only PYNQ metadata graph after hardware release.
            sys.stdout.flush();sys.stderr.flush()
            os._exit(code or 0)
        except Exception as exc:
            print(json.dumps(dict(state='error', error=str(exc))), flush=True)
            raise
