"""Independent CAMUS web app; shared pipeline, one job, explicit backend."""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
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
    def __init__(self, address, pipeline, data):
        self.pipeline, self.data = pipeline, Path(data).resolve()
        protect_output(self.data/'_probe_not_created', pipeline.model.root)
        self.data.mkdir(parents=True, exist_ok=True)
        self.jobs, self.guard, self.busy = {}, threading.Lock(), threading.BoundedSemaphore(1)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='camus-inference')
        self.fault = False
        super().__init__(address, Handler)

    def execute(self, job, selected, spacing):
        job['state'] = 'running'
        finished = {}
        try:
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
            return self.json(dict(backend=self.server.pipeline.backend, spec_sha256=MODEL_SPEC_SHA256,
                                  ready=not self.server.fault, model='CAMUS hidden_histogram INT8'))
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
        with CamusServer((args.host,args.port), pipeline, args.data) as server:
            print(f'CAMUS_WEB http://{args.host}:{server.server_port} backend={args.backend}', flush=True)
            server.serve_forever()
    finally: pipeline.close()


if __name__ == '__main__': main()
