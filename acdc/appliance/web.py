"""LAN browser service. All image processing stays on the board."""
from __future__ import annotations

from collections import OrderedDict
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
from urllib.parse import urlsplit,parse_qs
import uuid

from acdc.contract import ContractError,canonical,atomic_json
from .engine import checked_job
from .mpr import MPRVolume
from .storage import ResultUsage

MAX_UPLOAD=32*1024*1024


class WebApplication:
    def __init__(self,client,root,kernel,access_code):
        if not isinstance(access_code,str) or len(access_code)<12:raise ContractError('access code too short')
        self.client=client;self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.kernel=kernel;self.access_code=access_code;self.sessions=OrderedDict();self.lock=threading.Lock()
        self.mpr_lock=threading.Lock();self.volumes=OrderedDict();self.images=OrderedDict()
        try:active=self.client.request(op='health').get('active_job')
        except (OSError,TimeoutError):active=None
        self.storage=ResultUsage(self.root,active_job=active)

    def login(self,code):
        if not isinstance(code,str) or not secrets.compare_digest(code,self.access_code):
            raise PermissionError('incorrect access code')
        token=secrets.token_urlsafe(32)
        with self.lock:
            self.sessions[token]=time.monotonic()+8*3600
            while len(self.sessions)>32:self.sessions.popitem(last=False)
        return token

    def authorized(self,cookie):
        try:
            parsed=SimpleCookie(cookie or '');token=parsed['acdc_v2'].value
        except (KeyError,ValueError):return False
        with self.lock:
            expires=self.sessions.get(token)
            return expires is not None and expires>time.monotonic()

    def health(self):
        try:return self.client.request(op='health')
        except (OSError,TimeoutError):return dict(status='initializing',ready=False,active_job=None)

    def result(self,job):
        checked_job(job);root=self.root/'jobs'/job
        state=json.loads((root/'status.json').read_text('utf-8'))
        if state['status']!='assets_ready':raise ContractError('complete results are not ready')
        return root,json.loads((root/'manifest.json').read_text('utf-8'))

    def _volume(self,root,job,phase):
        """Caller holds mpr_lock; both preload and rendering share the same LRU."""
        identity=(job,phase)
        if identity not in self.volumes:self.volumes[identity]=MPRVolume(root/'viewer'/phase,self.kernel)
        self.volumes.move_to_end(identity)
        while len(self.volumes)>2:self.volumes.popitem(last=False)
        return self.volumes[identity]

    def preload_mpr(self,root,job,phase):
        # The browser always reads both volume manifests before enabling the
        # viewer. Charge mmap/validation to that measured result-loading stage,
        # so the first cursor movement need not initialize the medical volume.
        with self.mpr_lock:self._volume(root,job,phase)

    def mpr(self,job,phase,plane,values):
        values=dict(values)
        root,_=self.result(job)
        if phase not in ('ED','ES'):raise ContractError('invalid phase')
        allowed={'position','shape','level','window','alpha','visible','outline'}
        if set(values)-allowed:raise ContractError('unknown MPR parameter')
        key=canonical([job,phase,plane,values])
        with self.mpr_lock:
            if key in self.images:
                self.images.move_to_end(key);return self.images[key]
            position=values.pop('position',None)
            result=self._volume(root,job,phase).render(plane,position,**values)
            self.images[key]=result
            while len(self.images)>12:self.images.popitem(last=False)
            return result


def handler(app):
    static=Path(__file__).with_name('web_assets')
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        disable_nagle_algorithm=True
        def setup(self):super().setup();self.connection.settimeout(60)
        def log_message(self,*args):pass

        def send(self,code,body,ctype='application/json',headers=None):
            if ctype=='application/json':body=canonical(body)
            self.send_response(code);self.send_header('Content-Type',ctype)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            for key,value in (headers or {}).items():self.send_header(key,value)
            self.end_headers();self.wfile.write(body)

        def failure(self,exc):
            self.close_connection=True
            code=401 if isinstance(exc,PermissionError) else 404 if isinstance(exc,FileNotFoundError) else 400 if isinstance(exc,ValueError) else 503 if isinstance(exc,OSError) else 500
            self.send(code,dict(error=f'{type(exc).__name__}: {exc}'))

        def auth(self):
            if not app.authorized(self.headers.get('Cookie')):raise PermissionError('login required')

        def body(self,maximum=16384):
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=maximum:raise ContractError('invalid request length')
            raw=self.rfile.read(n)
            if len(raw)!=n:raise ContractError('truncated request')
            data=json.loads(raw)
            if not isinstance(data,dict):raise ContractError('JSON object required')
            return data

        def file(self,path):
            if not path.is_file():raise FileNotFoundError(path.name)
            mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8',
                  '.css':'text/css; charset=utf-8','.json':'application/json','.png':'image/png',
                  '.zip':'application/zip'}.get(path.suffix,'application/octet-stream')
            self.send_response(200);self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(path.stat().st_size));self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(128*1024),b''):self.wfile.write(chunk)

        def do_GET(self):
            try:
                path=urlsplit(self.path).path
                if path in ('/','/app.js','/style.css','/surface_view.js'):
                    self.file(static/('index.html' if path=='/' else path[1:]));return
                self.auth()
                if path=='/api/v2/health':self.send(200,dict(app.health(),max_upload_bytes=MAX_UPLOAD));return
                match=re.fullmatch(r'/api/v2/jobs/([0-9a-f]{32})',path)
                if match:self.send(200,app.client.request(op='status',job=match[1]));return
                match=re.fullmatch(r'/api/v2/jobs/([0-9a-f]{32})/assets/(.+)',path)
                if match:
                    root,manifest=app.result(match[1]);name=match[2]
                    if name not in manifest['files'] and name not in ('manifest.json','result.zip','completion.json'):
                        raise FileNotFoundError('asset is not public')
                    target=(root/name).resolve()
                    if not target.is_relative_to(root.resolve()):raise ContractError('invalid asset path')
                    volume=re.fullmatch(r'viewer/(ED|ES)/volume.json',name)
                    if volume:app.preload_mpr(root,match[1],volume[1])
                    self.file(target);return
                raise FileNotFoundError('unknown endpoint')
            except (BrokenPipeError,ConnectionResetError):return
            except Exception as exc:self.failure(exc)

        def do_POST(self):
            began=time.perf_counter();reservation=None;job=None;submitted=False
            try:
                origin=self.headers.get('Origin')
                if origin is not None and urlsplit(origin).netloc!=self.headers.get('Host'):
                    raise PermissionError('cross-origin request refused')
                path=urlsplit(self.path).path
                if path=='/api/v2/session':
                    token=app.login(self.body().get('code'))
                    self.send(200,dict(authenticated=True),headers={'Set-Cookie':f'acdc_v2={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=28800'})
                    return
                self.auth()
                if path=='/api/v2/jobs':
                    length=int(self.headers.get('Content-Length','0'));ed_bytes=int(self.headers.get('X-ED-Bytes','0'))
                    if not 0<ed_bytes<length<=MAX_UPLOAD:raise ContractError('invalid ED/ES upload length')
                    suffixes=[]
                    query=parse_qs(urlsplit(self.path).query)
                    for phase in ('ed','es'):
                        name=query.get(phase,[''])[0].lower()
                        if name.endswith('.nii.gz'):suffixes.append('.nii.gz')
                        elif name.endswith('.nii'):suffixes.append('.nii')
                        else:raise ContractError('upload explicit ED and ES NIfTI files')
                    if shutil.disk_usage(app.root).free<MAX_UPLOAD*2+256*1024*1024:
                        raise ContractError('insufficient board storage')
                    job=uuid.uuid4().hex
                    reservation=app.client.request(op='reserve',job=job)['reservation']
                    app.storage.admit(job,length)
                    folder=app.root/'inputs'/job;folder.mkdir(parents=True,exist_ok=False)
                    for phase,count,suffix in zip(('ED','ES'),(ed_bytes,length-ed_bytes),suffixes):
                        with (folder/(phase+suffix)).open('wb') as stream:
                            while count:
                                chunk=self.rfile.read(min(count,128*1024))
                                if not chunk:raise ContractError('interrupted upload')
                                stream.write(chunk);count-=len(chunk)
                    result=app.client.request(op='start',job=job,reservation=reservation,suffixes=suffixes,
                                              upload_seconds=time.perf_counter()-began)
                    submitted=True;self.send(202,result);return
                match=re.fullmatch(r'/api/v2/jobs/([0-9a-f]{32})/cancel',path)
                if match:
                    self.body();self.send(200,app.client.request(op='cancel',job=match[1]));return
                match=re.fullmatch(r'/api/v2/jobs/([0-9a-f]{32})/mpr/(ED|ES)/(axial|coronal|sagittal)',path)
                if match:
                    png,meta=app.mpr(*match.groups(),self.body())
                    self.send(200,png,'image/png',{'X-MPR-Metadata':canonical(meta).decode()});return
                match=re.fullmatch(r'/api/v2/jobs/([0-9a-f]{32})/client-timing',path)
                if match:
                    root,_=app.result(match[1]);data=self.body()
                    duration=data.get('interactive_seconds');client=data.get('client_id')
                    if not isinstance(duration,(int,float)) or not math.isfinite(duration) or not 0<duration<=3600:
                        raise ContractError('invalid client duration')
                    if not isinstance(client,str) or not re.fullmatch(r'[0-9a-f]{32}',client):raise ContractError('invalid client ID')
                    atomic_json(root/'client_timings'/(client+'.json'),dict(data,scope='browser reported upload-to-interactive duration'))
                    app.storage.refresh(match[1])
                    self.send(200,dict(recorded=True));return
                raise FileNotFoundError('unknown endpoint')
            except (BrokenPipeError,ConnectionResetError):return
            except Exception as exc:self.failure(exc)
            finally:
                if reservation and not submitted:
                    try:app.client.request(op='release',job=job,reservation=reservation)
                    except OSError:pass
    return Handler


def serve(app,host='0.0.0.0',port=8092):
    class Server(ThreadingHTTPServer):
        slots=threading.BoundedSemaphore(16)
        def process_request(self,request,address):
            if not self.slots.acquire(blocking=False):self.shutdown_request(request);return
            try:super().process_request(request,address)
            except BaseException:self.slots.release();raise
        def process_request_thread(self,request,address):
            try:super().process_request_thread(request,address)
            finally:self.slots.release()
    server=Server((host,port),handler(app))
    from acdc.lifecycle import graceful_signals
    graceful_signals(server)
    print(f'ACDC V2 LAN Web: http://{host}:{server.server_port}',flush=True)
    try:server.serve_forever()
    finally:server.server_close()
