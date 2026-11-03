"""Two independently launchable localhost APIs and pages, no board changes."""
import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import urlparse
from .skin_api import SkinAPI
from .camus_api import CamusAPI
from .images import decode_image,MAX_FILE_BYTES
from .service import png_data
from .contracts import InputError
from .models import PROJECT

STATIC = PROJECT/'host/interactive'


def sample(model):
    if model=='skin':
        exclusions = json.loads((PROJECT/'runs/role_a_full2017/overlap_exclusions.json').read_text())['external_validation']
        path = next(p for p in sorted((PROJECT/'data/raw/isic2017_validation/images').glob('*.jpg')) if p.stem not in exclusions)
    else:
        path = PROJECT/'datasets/camus_public/official/CAMUS_public/database_nifti/patient0200/patient0200_2CH_ED.nii.gz'
    image = decode_image(model,base64.b64encode(path.read_bytes()).decode(),path.name)
    return {'image':png_data(image),'name':path.name,'width':image.shape[1],'height':image.shape[0],
            'source':'development_demo','reference_available':False}


def make_server(model,port=0,controller=None):
    if model not in ('skin','camus'):
        raise InputError('请选择 skin 或 camus')
    controller = controller or (SkinAPI() if model=='skin' else CamusAPI())
    gate = threading.Lock()
    prefix = '/api/'+model

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*args):
            # Do not log uploads, credentials or patient filenames.
            pass

        def send(self,status,payload,content_type='application/json; charset=utf-8'):
            if isinstance(payload,(dict,list)):
                payload = json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8')
            if isinstance(payload,str): payload = payload.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(payload)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer')
            self.end_headers()
            try: self.wfile.write(payload)
            except (BrokenPipeError,ConnectionResetError): pass

        def allowed(self):
            port = self.server.server_address[1]
            hosts = {f'127.0.0.1:{port}',f'localhost:{port}'}
            if self.headers.get('Host') not in hosts:
                self.send(403,{'error':'请通过本机地址访问此服务'})
                return False
            origin = self.headers.get('Origin')
            if origin and origin not in {'http://'+h for h in hosts}:
                self.send(403,{'error':'拒绝跨站请求'})
                return False
            return True

        def do_GET(self):
            if not self.allowed(): return
            path = urlparse(self.path).path
            if path in ('/',f'/{model}',f'/{model}/'):
                html = (STATIC/'index.html').read_text(encoding='utf-8').replace('__MODEL__',model)
                return self.send(200,html,'text/html; charset=utf-8')
            files = {'/app.js':'application/javascript; charset=utf-8','/style.css':'text/css; charset=utf-8'}
            if path in files:
                return self.send(200,(STATIC/path[1:]).read_bytes(),files[path])
            if path==prefix+'/capabilities':
                return self.send(200,controller.capabilities())
            if path==prefix+'/sample':
                try: return self.send(200,sample(model))
                except Exception: return self.send(503,{'error':'本机开发样例不可用，请自行选择图像'})
            if path=='/health':
                return self.send(200,{'ok':True,'model':model,'software_only':True})
            return self.send(404,{'error':'该独立服务没有此接口'})

        def do_POST(self):
            if not self.allowed(): return
            path = urlparse(self.path).path
            if path not in (prefix+'/image',prefix+'/predict'):
                return self.send(404,{'error':'该独立服务没有此接口'})
            if self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
                return self.send(415,{'error':'请使用 application/json'})
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0<length<=MAX_FILE_BYTES*4//3+8192:
                    return self.send(413,{'error':'请求为空或图像超过16 MB'})
                body = json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(body,dict) or set(body)-{'image','filename','parameters'}:
                    raise InputError('请求只接受 image、filename、parameters 字段')
                image = decode_image(model,body.get('image'),body.get('filename','image.png'))
                if path.endswith('/image'):
                    return self.send(200,{'image':png_data(image),'width':image.shape[1],'height':image.shape[0]})
                if not gate.acquire(blocking=False):
                    return self.send(409,{'error':'正在处理前一帧，请稍后重新运行'})
                try:
                    result = controller.predict(image,body.get('parameters',{}))
                finally:
                    gate.release()
                return self.send(200,result)
            except (ValueError,UnicodeError) as error:
                return self.send(400,{'error':str(error)})
            except Exception:
                return self.send(500,{'error':'推理失败，请检查冻结模型与本地运行环境；未返回旧结果'})

    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    parser = argparse.ArgumentParser(description='皮肤/CAMUS 独立交互分割接口')
    parser.add_argument('--model',choices=('skin','camus'),required=True)
    parser.add_argument('--port',type=int)
    args = parser.parse_args()
    port = args.port if args.port is not None else (8766 if args.model=='skin' else 8767)
    with make_server(args.model,port) as server:
        print(json.dumps({'ready':True,'model':args.model,'pid':__import__('os').getpid(),
            'url':f'http://127.0.0.1:{server.server_address[1]}/{args.model}/'},ensure_ascii=False),flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass


if __name__=='__main__': main()
