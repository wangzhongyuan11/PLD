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
import subprocess
import sys
import threading
import time
import uuid


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
    def __init__(self, log):
        self.log = Path(log); self.process = None; self.session = None
        self.ready = threading.Event(); self.lock = threading.Lock()
        self.error = ''

    def prepare(self):
        with self.lock:
            if self.process is not None: return
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
            if value.get('state') == 'ready': self.ready.set()
            if value.get('state') == 'error':
                self.error = value.get('error', 'HDMI failed'); self.ready.set()
            session = self.session
            if session and value.get('id') == session.ident and value.get('state') in ('idle', 'error'):
                session.returncode = 1 if value['state'] == 'error' else 0
                session.finished.set()
        self.ready.set()
        if self.session and self.session.returncode is None:
            self.session.returncode = self.process.wait() or 1; self.session.finished.set()

    def send(self, body):
        with self.lock:
            if not self.process or self.process.poll() is not None:
                raise RuntimeError('resident HDMI process unavailable; release unconfirmed')
            self.process.stdin.write(json.dumps(body)+'\n'); self.process.stdin.flush()

    def start(self, command, log):
        self.prepare()
        if not self.ready.wait(90): raise RuntimeError('resident HDMI initialization timed out')
        if self.error: raise RuntimeError(self.error)
        if self.session and self.session.poll() is None: raise RuntimeError('HDMI already active')
        self.session = DisplaySession(self, uuid.uuid4().hex)
        self.send(dict(action='show', id=self.session.ident, command=[str(x) for x in command], log=str(log)))
        return self.session

    def close(self):
        if self.process is None: return
        if self.process.poll() is not None:
            if self.process.returncode: raise RuntimeError('resident HDMI process failed')
            return
        self.send(dict(action='quit'))
        self.process.wait(timeout=90)
        if self.process.returncode: raise RuntimeError('resident HDMI did not close normally')


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
        video_module.prepare_video_overlay()
    respond(state='ready')
    modules = {}; failed = False
    while not quitting.is_set():
        body = commands.get()
        if body is None: break
        command = body['command']; ident = body['id']
        try:
            if quitting.is_set(): break
            with Path(body['log']).open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
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
            if not quitting.is_set():
                try:video_module.prepare_video_overlay()
                except Exception as exc:print('HDMI_SOFTWARE_PREPARE_ERROR',str(exc),file=sys.stderr,flush=True)
        except Exception as exc:
            failed = True
            active[0] = None
            respond(state='error', id=ident, error=type(exc).__name__+': '+str(exc))
        finally:
            if active[0] == ident: active[0] = None
    return 1 if failed else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', action='store_true'); args = parser.parse_args()
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
