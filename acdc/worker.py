"""Single-owner binary inference worker, normally reached over a board SSH tunnel."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from collections import OrderedDict
import hashlib
import threading
import numpy as np

from .contract import INPUT_SHAPE,INPUT_BYTES,HEAD_BYTES,canonical,checked_int8,ContractError
from .protocol import encode,decode,validate_request


def make_worker(backend,port=8093):
    lock=threading.Lock()
    cache=OrderedDict()
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def setup(self):
            super().setup();self.connection.settimeout(30)
        def log_message(self,*args):pass
        def reply(self,status,data,ctype='application/octet-stream'):
            self.send_response(status);self.send_header('Content-Type',ctype)
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path!='/health':self.reply(404,b'unknown');return
            self.reply(200,canonical(dict(backend=backend.kind,synthetic=backend.synthetic,model_sha256=backend.model_sha,
                                          head_scale=backend.head_scale,busy=lock.locked(),runtime_sha256=getattr(backend,'runtime_sha',None),
                                          hardware=getattr(backend,'hardware_info',None))),'application/json')
        def do_POST(self):
            if self.path!='/infer':self.reply(404,b'unknown');return
            if not lock.acquire(blocking=False):self.close_connection=True;self.reply(409,b'worker busy');return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not INPUT_BYTES<n<=INPUT_BYTES+65540 or self.headers.get('Transfer-Encoding'):
                    raise ContractError('invalid request length')
                self.connection.settimeout(30)
                data=self.rfile.read(n)
                meta,raw=decode(data,INPUT_BYTES);validate_request(meta)
                key=hashlib.sha256(data).hexdigest()
                if key in cache:
                    cached_head,cached_meta=cache[key];cache.move_to_end(key)
                    self.reply(200,encode(dict(cached_meta,cached_response=True,nodes=[],events=[],slice_seconds=0.),cached_head))
                    return
                x=checked_int8(np.frombuffer(raw,np.int8).reshape(INPUT_SHAPE))
                head,result=backend.infer(x,meta)
                # Per-command events may exceed the 64KiB metadata cap. Keep node
                # timings + totals on the wire; detailed events remain in board verification.
                events=result.pop('events',[])
                result['event_summary']=dict(commands=len(events),compute_cycles=sum(e.get('compute_cycles',0) for e in events),
                    payload_bytes=sum(e.get('bytes',0) for e in events))
                cache[key]=(head.tobytes(),result)
                if len(cache)>8:cache.popitem(last=False)
                self.reply(200,encode(result,head.tobytes()))
            except Exception as exc:
                self.close_connection=True
                self.reply(400,canonical(dict(error=f'{type(exc).__name__}: {exc}')),'application/json')
            finally:lock.release()
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)
