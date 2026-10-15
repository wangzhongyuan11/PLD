"""Independent CAMUS web app; shared pipeline, one job, explicit backend."""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

from .model import DEFAULT_MODEL_ROOT, MODEL_SPEC_SHA256
from .pipeline import CamusPipeline, protect_output

MAX_BODY = 32 * 1024 * 1024
ARTIFACTS = {'overlay.png','segmentation.png','label.png','label_raw.png','labels_native.npz',
             'head.bin','head.json','report.json','result.zip','preview.html'}


def decode_upload(payload):
    if not isinstance(payload, dict): raise ValueError('请求必须是 JSON 对象')
    files = payload.get('files')
    if not isinstance(files, list) or not 1 <= len(files) <= 4:
        raise ValueError('上传 1～4 个文件，MHD 请同时选择对应 RAW')
    result = {}
    for item in files:
        if not isinstance(item, dict): raise ValueError('文件记录无效')
        name = item.get('name', '')
        if not isinstance(name, str) or not name or len(name) > 128 or name in ('.','..') or any(c in name for c in '/\\:\x00') or not name.isprintable():
            raise ValueError('文件名无效')
        if name in result: raise ValueError('文件名重复')
        data = base64.b64decode(item['data_base64'], validate=True)
        if not data: raise ValueError('文件为空')
        result[name] = data
    selected = payload.get('input_name')
    if selected not in result: raise ValueError('缺少主输入文件')
    if Path(selected).suffix.lower() not in ('.png','.jpg','.jpeg','.npy','.npz','.mhd','.nii','.gz'):
        raise ValueError('请选择灰度图、数组或 MHD/NIfTI')
    return result, selected


class CamusServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, pipeline, data, runtime=None):
        self.pipeline, self.data = pipeline, Path(data).resolve()
        protect_output(self.data/'_probe_not_created', pipeline.model.root)
        self.data.mkdir(parents=True, exist_ok=True)
        self.jobs, self.guard, self.busy = {}, threading.Lock(), threading.BoundedSemaphore(1)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='camus-inference')
        self.fault = False
        self.runtime,self.released,self.hdmi = runtime,False,None
        self.hdmi_folder,self.hdmi_started = None,0.
        super().__init__(address, Handler)

    def hdmi_status(self):
        process=self.hdmi
        if process is None:return 'off'
        code=process.poll()
        if code is not None:
            if code != 0:self.fault=True
            return 'error' if code else 'off'
        if (self.hdmi_folder/'hdmi.ready.json').exists():return 'showing'
        if time.monotonic()-self.hdmi_started>45:self.fault=True;return 'error'
        return 'starting'

    def stop_hdmi(self):
        if self.hdmi is not None:
            if self.hdmi.poll() is None:
                self.hdmi.terminate()
                try:self.hdmi.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    self.fault=True
                    raise RuntimeError('HDMI did not confirm hardware release; inspect board before restarting')
            if self.hdmi.returncode != 0:
                self.fault=True;raise RuntimeError('HDMI child failed; hardware release requires inspection')
            self.hdmi=None

    def start_hdmi(self, identifier, view):
        if self.pipeline.backend!='pl':raise ValueError('HDMI requires the real board PL backend')
        if view not in ('overlay','segmentation','label','label_raw'):raise ValueError('invalid CAMUS HDMI view')
        with self.guard:state=self.jobs.get(identifier,{}).get('state')
        if state!='done':raise ValueError('completed CAMUS job required')
        self.stop_hdmi()
        self.pipeline.close();self.released=True
        self.hdmi_folder=self.data/identifier/'output'
        (self.hdmi_folder/'hdmi.ready.json').unlink(missing_ok=True)
        with (self.hdmi_folder/'hdmi.log').open('w',encoding='utf-8') as log:
            self.hdmi=subprocess.Popen([sys.executable,'-u','-m','camus.hdmi','--result',str(self.hdmi_folder),'--view',view],stdout=log,stderr=log)
        self.hdmi_started=time.monotonic()

    def execute(self, job, selected, spacing):
        job['state'] = 'running'
        finished = {}
        try:
            self.stop_hdmi()
            if self.released:
                self.pipeline=CamusPipeline(self.pipeline.model.root,'pl',self.runtime)
                self.released=False
            result = self.pipeline.run(self.data/job['id']/'input'/selected,
                                       self.data/job['id']/'output', spacing)
            result.pop('output_dir'); result.pop('report')
            finished.update(state='done', result=result)
        except Exception as exc:
            finished.update(state='error', error=str(exc))
            if self.pipeline.backend == 'pl':
                self.fault = True
                try: self.pipeline.close()
                except Exception as close_error: finished['close_error'] = str(close_error)
        finally:
            with self.guard:
                job['server_seconds'] = time.perf_counter()-job.pop('_started')
                job.update(finished)
            self.busy.release()

    def server_close(self):
        super().server_close()
        self.executor.shutdown(wait=True)
        self.stop_hdmi()
        self.pipeline.close()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup(); self.connection.settimeout(20)

    def send_bytes(self, data, status=200, kind='application/json; charset=utf-8'):
        self.send_response(status)
        self.send_header('Content-Type', kind); self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers(); self.wfile.write(data)

    def json(self, payload, status=200):
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode('utf-8'), status)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/':
            return self.send_bytes(Path(__file__).with_name('web.html').read_bytes(), kind='text/html; charset=utf-8')
        if path == '/api/status':
            hdmi=self.server.hdmi_status()
            return self.json(dict(backend=self.server.pipeline.backend, spec_sha256=MODEL_SPEC_SHA256,
                                  ready=not self.server.fault, hdmi=hdmi,
                                  model='CAMUS hidden_histogram INT8'))
        match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})', path)
        if match:
            with self.server.guard: job = dict(self.server.jobs.get(match[1], {}))
            job.pop('_started', None)
            return self.json(job, 200 if job else 404)
        match = re.fullmatch(r'/files/([0-9a-f]{32})/([a-z_.]+)', path)
        if match and match[2] in ARTIFACTS:
            with self.server.guard: state = self.server.jobs.get(match[1], {}).get('state')
            file = self.server.data/match[1]/'output'/match[2]
            if state == 'done' and file.is_file() and not file.is_symlink():
                kind = 'image/png' if file.suffix == '.png' else 'application/json' if file.suffix == '.json' else 'application/zip' if file.suffix == '.zip' else 'application/octet-stream'
                return self.send_bytes(file.read_bytes(), kind=kind)
        self.json(dict(error='not found'), 404)

    def do_POST(self):
        origin=self.headers.get('Origin')
        if origin and origin!='http://'+self.headers.get('Host',''):
            return self.json(dict(error='cross-origin board control is not allowed'),403)
        if urlsplit(self.path).path == '/api/hdmi':
            if self.server.fault:return self.json(dict(error='hardware session faulted; inspect board'),503)
            if not self.server.busy.acquire(blocking=False):return self.json(dict(error='inference is busy'),409)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=2048:raise ValueError('invalid control request size')
                request=json.loads(self.rfile.read(length))
                if not isinstance(request,dict):raise ValueError('control request must be a JSON object')
                if request.get('action')=='stop':self.server.stop_hdmi()
                elif request.get('action')=='show':self.server.start_hdmi(request.get('id'),request.get('view','overlay'))
                else:raise ValueError('unknown HDMI action')
                return self.json(dict(hdmi=self.server.hdmi_status()),202)
            except (ValueError,TypeError) as exc:return self.json(dict(error=str(exc)),400)
            except Exception as exc:
                self.server.fault=True;return self.json(dict(error=str(exc)),503)
            finally:self.server.busy.release()
        if urlsplit(self.path).path != '/api/jobs': return self.json(dict(error='not found'), 404)
        started = time.perf_counter()
        if self.server.fault: return self.json(dict(error='PL session faulted; restart this CAMUS app after checking hardware'), 503)
        if not self.server.busy.acquire(blocking=False): return self.json(dict(error='已有任务正在执行'), 409)
        submitted = False
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY: raise ValueError('上传体积超过 32 MiB 或为空')
            raw = self.rfile.read(length)
            if len(raw) != length: raise ValueError('上传未完成')
            payload = json.loads(raw)
            files, selected = decode_upload(payload)
            identifier = uuid.uuid4().hex
            folder = self.server.data/identifier/'input'; folder.mkdir(parents=True)
            for name, content in files.items(): (folder/name).write_bytes(content)
            job = dict(id=identifier, state='queued', backend=self.server.pipeline.backend, _started=started)
            with self.server.guard: self.server.jobs[identifier] = job
            self.server.executor.submit(self.server.execute, job, selected, payload.get('spacing'))
            submitted = True
            self.json(dict(id=identifier, state='queued'), 202)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.json(dict(error=str(exc)), 400)
        finally:
            if not submitted: self.server.busy.release()


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m camus.web')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8090)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL_ROOT)
    p.add_argument('--backend', choices=('reference','emulator','pl'), default='reference')
    p.add_argument('--runtime', type=Path); p.add_argument('--data', type=Path, default=Path('out/camus/web'))
    args = p.parse_args(argv)
    if args.host not in ('127.0.0.1', 'localhost', '::1'):
        p.error('this independent local app binds loopback only; remote access can use an SSH tunnel')
    pipeline = CamusPipeline(args.model, args.backend, args.runtime)
    try:
        with CamusServer((args.host,args.port), pipeline, args.data, args.runtime) as server:
            def stop_server(*_):
                threading.Thread(target=server.shutdown,daemon=True).start()
            signal.signal(signal.SIGTERM,stop_server);signal.signal(signal.SIGINT,stop_server)
            print(f'CAMUS_WEB http://{args.host}:{server.server_port} backend={args.backend}', flush=True)
            server.serve_forever()
    finally: pipeline.close()


if __name__ == '__main__': main()
