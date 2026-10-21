"""Local asynchronous volume UI. Exactly one backend owner; bounded raw pair uploads."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit,parse_qs
import uuid

from .contract import ContractError,canonical
from .pipeline import Pipeline

MAX_UPLOAD=128*1024*1024


class JobManager:
    def __init__(self,root,pipeline):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.pipeline=pipeline;self.lock=threading.Lock();self.active=None;self.jobs={}
        self.thread=None
        self.presentation_lock=threading.Lock()

    def create(self,ed,es,job_id,upload_seconds=0.):
        if self.pipeline is None:raise ContractError('saved-results viewer does not execute inference')
        with self.lock:
            if self.active is not None:raise ContractError('another volume job is active')
            cancel=threading.Event()
            self.jobs[job_id]=dict(status='queued',run_id=job_id,synthetic=self.pipeline.backend.synthetic)
            self.active=(job_id,cancel)
        def progress(state):
            with self.lock:self.jobs[job_id]=state
        def run():
            try:self.pipeline.run(ed,es,self.root/job_id,cancel=cancel,progress=progress,upload_seconds=upload_seconds)
            except Exception as exc:
                with self.lock:
                    if self.jobs[job_id].get('status') not in ('failed','cancelled'):
                        self.jobs[job_id].update(status='failed',error=f'{type(exc).__name__}: {exc}')
            finally:
                with self.lock:self.active=None
        self.thread=threading.Thread(target=run,name='acdc-job',daemon=False);self.thread.start()

    def status(self,job_id):
        with self.lock:
            if job_id in self.jobs:return dict(self.jobs[job_id])
        path=self.root/job_id/'status.json'
        if not path.is_file():raise FileNotFoundError(job_id)
        return json.loads(path.read_text('utf-8'))

    def cancel(self,job_id):
        with self.lock:
            if self.active and self.active[0]==job_id:self.active[1].set();return
        raise ContractError('job is not active')

    def presentation(self,job_id,phase):
        from .surface import prepare_presentation
        with self.presentation_lock:
            key,report=prepare_presentation(self.root/job_id,self.root/'_presentations'/job_id,phase)
        return dict(report,base_url=f'/presentations/{job_id}/{key}/{phase}/')


def handler(manager):
    static=Path(__file__).with_name('web_assets')
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def setup(self):
            super().setup();self.connection.settimeout(30)
        def log_message(self,*args):pass

        def send(self,code,body,ctype='application/json'):
            if ctype=='application/json':body=canonical(body)
            self.send_response(code);self.send_header('Content-Type',ctype)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)

        def error(self,exc):
            self.close_connection=True
            self.send(400 if isinstance(exc,(ValueError,ContractError)) else 404 if isinstance(exc,FileNotFoundError) else 500,
                      dict(error=f'{type(exc).__name__}: {exc}'))

        def asset(self,path):
            if not path.is_file():raise FileNotFoundError(path.name)
            mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8',
                  '.json':'application/json','.zip':'application/zip'}.get(path.suffix,'application/octet-stream')
            self.send_response(200);self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(path.stat().st_size));self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(256*1024),b''):self.wfile.write(chunk)

        def do_GET(self):
            try:
                path=urlsplit(self.path).path
                if path=='/api/capabilities':
                    backend=manager.pipeline.backend if manager.pipeline is not None else None
                    self.send(200,dict(profile='axial_only_v1',backend=backend.kind if backend else 'viewer',
                        synthetic=backend.synthetic if backend else False,model_sha256=backend.model_sha if backend else None,
                        head_scale=backend.head_scale if backend else None,max_upload_bytes=MAX_UPLOAD,
                        read_only=backend is None));return
                if path in ('/','/app.js','/surface_view.js','/style.css'):
                    self.asset(static/('index.html' if path=='/' else path[1:]));return
                match=re.fullmatch(r'/api/jobs/([0-9a-f]{32})',path)
                if match:self.send(200,manager.status(match[1]));return
                match=re.fullmatch(r'/presentations/([0-9a-f]{32})/([0-9a-f]{64})/(ED|ES)/([^/]+)',path)
                if match:
                    job,key,phase,name=match.groups()
                    if name not in {n+s for n in ('RV','MYO','LV') for s in ('.vertices.f32','.normals.f32','.vtp','.ply')}:
                        raise FileNotFoundError('unknown presentation asset')
                    self.asset(manager.root/'_presentations'/job/key/phase/name);return
                match=re.fullmatch(r'/results/([0-9a-f]{32})/(.+)',path)
                if match:
                    root=manager.root/match[1];candidate=(root/match[2]).resolve()
                    if not candidate.is_relative_to(root.resolve()) or candidate.is_symlink():raise ContractError('unsafe result path')
                    # Never expose uploaded input files; viewer data is deliberate result output.
                    self.asset(candidate);return
                raise FileNotFoundError('unknown endpoint')
            except (BrokenPipeError,ConnectionResetError):return
            except Exception as exc:self.error(exc)

        def do_POST(self):
            upload_begin=time.perf_counter()
            try:
                origin=self.headers.get('Origin')
                if origin is not None and urlsplit(origin).netloc!=self.headers.get('Host'):
                    raise ContractError('cross-origin requests are not accepted')
                path=urlsplit(self.path).path
                match=re.fullmatch(r'/api/jobs/([0-9a-f]{32})/presentation/(ED|ES)',path)
                if match:self.send(200,manager.presentation(*match.groups()));return
                match=re.fullmatch(r'/api/jobs/([0-9a-f]{32})/cancel',path)
                if match:
                    manager.cancel(match[1]);self.send(200,dict(status='cancelling'));return
                if path!='/api/jobs':raise FileNotFoundError('unknown endpoint')
                if manager.pipeline is None:raise ContractError('saved-results viewer does not execute inference')
                if self.headers.get('Transfer-Encoding'):raise ContractError('chunked upload not supported')
                total=int(self.headers.get('Content-Length','0'));ed_bytes=int(self.headers.get('X-ED-Bytes','0'))
                if not 0<ed_bytes<total<=MAX_UPLOAD:raise ContractError('invalid ED/ES upload lengths')
                query=parse_qs(urlsplit(self.path).query)
                suffixes=[]
                for key in ('ed','es'):
                    name=query.get(key,[''])[0].lower()
                    if name.endswith('.nii.gz'):suffixes.append('.nii.gz')
                    elif name.endswith('.nii'):suffixes.append('.nii')
                    else:raise ContractError('upload two explicit .nii/.nii.gz files')
                with manager.lock:
                    if manager.active is not None:raise ContractError('another volume job is active')
                job_id=uuid.uuid4().hex
                folder=manager.root/'inputs'/job_id;folder.mkdir(parents=True,exist_ok=False)
                files=[]
                self.connection.settimeout(60)
                for phase,length,suffix in zip(('ED','ES'),(ed_bytes,total-ed_bytes),suffixes):
                    target=folder/(phase+suffix);remaining=length
                    with target.open('wb') as stream:
                        while remaining:
                            chunk=self.rfile.read(min(remaining,256*1024))
                            if not chunk:raise ContractError('truncated upload')
                            stream.write(chunk);remaining-=len(chunk)
                    files.append(target)
                manager.create(*files,job_id,upload_seconds=time.perf_counter()-upload_begin)
                self.send(202,dict(run_id=job_id,status='queued'))
            except (BrokenPipeError,ConnectionResetError):return
            except Exception as exc:self.error(exc)
    return Handler


def serve(backend,output,*,port=8092,policy='strict',recognized_acdc=False):
    manager=JobManager(output,Pipeline(backend,policy=policy,recognized_acdc=recognized_acdc) if backend is not None else None)
    server=ThreadingHTTPServer(('127.0.0.1',port),handler(manager))
    from .lifecycle import graceful_signals
    graceful_signals(server)
    print(f'ACDC Axial UI: http://127.0.0.1:{server.server_port} ; backend={backend.kind if backend else "saved-results"}',flush=True)
    try:server.serve_forever()
    finally:
        with manager.lock:
            if manager.active:manager.active[1].set()
        server.server_close()
        if manager.thread is not None:
            manager.thread.join(90)
            if manager.thread.is_alive():raise RuntimeError('job did not stop: backend buffers deliberately remain owned')
        if backend is not None:backend.close()
