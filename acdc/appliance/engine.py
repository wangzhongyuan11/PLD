"""One persistent PL owner; Unix-socket control carries no slice tensors."""
from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import re
import secrets
import socket
import socketserver
import stat
import threading
import time

from acdc.contract import ContractError,canonical

JOB=re.compile(r'[0-9a-f]{32}\Z')


def checked_job(job):
    if not isinstance(job,str) or not JOB.fullmatch(job):raise ContractError('invalid job ID')
    return job


class JobManager:
    def __init__(self,pipeline,root,*,initialization=None):
        self.pipeline=pipeline;self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        (self.root/'inputs').mkdir(exist_ok=True);(self.root/'jobs').mkdir(exist_ok=True)
        self.lock=threading.Lock();self.active=None;self.states=OrderedDict();self.thread=None
        self.initialization=initialization or {}

    def _expire(self):
        if self.active and self.active['stage']=='receiving' and time.monotonic()>self.active['expires']:
            job=self.active['job'];self.states[job]=dict(run_id=job,status='upload_expired');self.active=None

    def health(self):
        with self.lock:
            self._expire();active=self.active['job'] if self.active else None
        backend=self.pipeline.backend
        device=getattr(backend,'device',None)
        fault=bool(getattr(device,'unsafe',False) or getattr(device,'closed',False))
        return dict(status='fault' if fault else 'busy' if active else 'ready',ready=not fault,active_job=active,backend=backend.kind,
            synthetic=backend.synthetic,model_sha256=backend.model_sha,runtime_sha256=backend.runtime_sha,
            initialization=self.initialization,generator='pynq_ps',inference='pynq_pl',renderer='browser_webgl')

    def reserve(self,job):
        checked_job(job)
        if not self.health()['ready']:raise ContractError('PL engine is not ready')
        with self.lock:
            self._expire()
            if self.active:raise ContractError('another task is active')
            if (self.root/'jobs'/job).exists() or (self.root/'inputs'/job).exists():
                raise ContractError('job ID already exists')
            token=secrets.token_hex(24)
            self.active=dict(job=job,token=token,stage='receiving',expires=time.monotonic()+90,cancel=threading.Event())
            self.states[job]=dict(run_id=job,status='receiving')
            while len(self.states)>64:self.states.popitem(last=False)
            return dict(run_id=job,reservation=token)

    def release(self,job,token):
        checked_job(job)
        with self.lock:
            if (self.active and self.active['job']==job and self.active['stage']=='receiving'
                    and secrets.compare_digest(self.active['token'],str(token))):
                self.active=None;self.states[job]=dict(run_id=job,status='upload_failed')
        return dict(released=True)

    def start(self,job,token,suffixes,upload_seconds):
        checked_job(job)
        if (not isinstance(suffixes,list) or len(suffixes)!=2 or any(s not in ('.nii','.nii.gz') for s in suffixes)
                or not isinstance(upload_seconds,(int,float)) or not 0<=upload_seconds<=90):
            raise ContractError('invalid upload metadata')
        files=[self.root/'inputs'/job/(phase+suffix) for phase,suffix in zip(('ED','ES'),suffixes)]
        if not all(p.is_file() and not p.is_symlink() for p in files):raise ContractError('missing uploaded pair')
        with self.lock:
            self._expire()
            if (not self.active or self.active['job']!=job or self.active['stage']!='receiving'
                    or not secrets.compare_digest(self.active['token'],str(token))):
                raise ContractError('upload reservation expired or mismatched')
            self.active['stage']='computing';cancel=self.active['cancel']
            self.states[job]=dict(run_id=job,status='queued')
        def progress(state):
            with self.lock:self.states[job]=dict(state)
        def run():
            try:self.pipeline.run(*files,self.root/'jobs'/job,cancel=cancel,progress=progress,upload_seconds=upload_seconds)
            except Exception as exc:
                with self.lock:
                    if self.states[job].get('status') not in ('failed','cancelled'):
                        self.states[job]=dict(run_id=job,status='failed',error=f'{type(exc).__name__}: {exc}')
            finally:
                with self.lock:self.active=None
        self.thread=threading.Thread(target=run,name='acdc-v2-job',daemon=False);self.thread.start()
        return dict(run_id=job,status='queued')

    def status(self,job):
        checked_job(job)
        with self.lock:
            self._expire()
            if job in self.states:return dict(self.states[job])
        path=self.root/'jobs'/job/'status.json'
        state=json.loads(path.read_text('utf-8'))
        if state['status'] not in ('assets_ready','failed','cancelled'):
            return dict(run_id=job,status='interrupted',error='engine restarted before completion')
        return state

    def cancel(self,job):
        checked_job(job)
        with self.lock:
            if not self.active or self.active['job']!=job:raise ContractError('task is not active')
            self.active['cancel'].set()
        return dict(cancelling=True)

    def dispatch(self,message):
        op=message.get('op');job=message.get('job')
        if op=='health':return self.health()
        if op=='reserve':return self.reserve(job)
        if op=='release':return self.release(job,message.get('reservation'))
        if op=='start':return self.start(job,message.get('reservation'),message.get('suffixes'),message.get('upload_seconds'))
        if op=='status':return self.status(job)
        if op=='cancel':return self.cancel(job)
        raise ContractError('unknown engine operation')

    def close(self):
        with self.lock:
            if self.active:self.active['cancel'].set()
        if self.thread:
            self.thread.join(timeout=80)
            if self.thread.is_alive():raise RuntimeError('job not stopped; retain hardware ownership')


class Client:
    def __init__(self,path):self.path=str(path)
    def request(self,**message):
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
            connection.settimeout(5);connection.connect(self.path);connection.sendall(canonical(message)+b'\n')
            with connection.makefile('rb') as stream:raw=stream.readline(65537)
        if len(raw)>65536 or not raw.endswith(b'\n'):raise ContractError('invalid engine response')
        result=json.loads(raw)
        if not result['ok']:raise ContractError(result['error'])
        return result['result']


def serve(manager,path,*,on_ready=None):
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not stat.S_ISSOCK(path.lstat().st_mode):raise ContractError('engine path is not a socket')
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as probe:
            probe.settimeout(.2)
            try:probe.connect(str(path))
            except (ConnectionRefusedError,FileNotFoundError):path.unlink()
            else:raise ContractError('another engine is listening')
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(5)
            try:
                raw=self.rfile.readline(65537)
                if len(raw)>65536 or not raw.endswith(b'\n'):raise ContractError('invalid engine request')
                result=dict(ok=True,result=manager.dispatch(json.loads(raw)))
            except Exception as exc:result=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
            self.wfile.write(canonical(result)+b'\n')
    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads=True
    server=Server(str(path),Handler);os.chmod(path,0o660)
    if os.geteuid()==0:
        import grp
        os.chown(path,-1,grp.getgrnam('xilinx').gr_gid)
    from acdc.lifecycle import graceful_signals
    graceful_signals(server)
    if on_ready:on_ready()
    try:server.serve_forever()
    finally:server.server_close();manager.close();path.unlink(missing_ok=True)
