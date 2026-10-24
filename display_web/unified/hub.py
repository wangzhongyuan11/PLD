"""Authenticated common UI and serialized systemd model switching. Stdlib only."""
import argparse
import http.client
from http.cookies import SimpleCookie, CookieError
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import signal
import subprocess
import threading
import time
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parent
UNITS={'skin':'pld-web.service','camus':'pld-camus.service'}
PORTS={'skin':8766,'camus':8090}

def memory():
    values={}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,value=line.split(':',1)
            if key in ('MemAvailable','CmaFree','CmaTotal'):values[key]=value.strip()
    except OSError:pass
    return values

class Board:
    def __init__(self,token):self.token=token
    def request(self,model,method,path,body=None,content_type='application/json'):
        port=PORTS[model];conn=http.client.HTTPConnection('127.0.0.1',port,timeout=45)
        headers={'Content-Type':content_type}
        if model=='skin':headers['Authorization']='Bearer '+self.token
        try:
            conn.request(method,path,body,headers)
            r=conn.getresponse();return r.status,r.getheader('Content-Type','application/octet-stream'),r.read()
        finally:conn.close()
    def status(self,model):
        code,_,raw=self.request(model,'GET','/api/status')
        if code!=200:raise RuntimeError('模型状态接口返回 '+str(code))
        return json.loads(raw)
    def props(self,model):
        r=subprocess.run(['systemctl','show',UNITS[model],'-p','ActiveState','-p','SubState','-p','Result','-p','ControlGroup','-p','MainPID'],check=True,capture_output=True,text=True,timeout=10)
        return dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
    def stop(self,model):
        before=self.props(model);group=before.get('ControlGroup','')
        if group and (not group.startswith('/') or '..' in group.split('/')):raise RuntimeError('异常 cgroup 路径')
        r=subprocess.run(['systemctl','stop',UNITS[model]],capture_output=True,text=True,timeout=260)
        after=self.props(model)
        if r.returncode or after.get('ActiveState')!='inactive' or after.get('Result') not in ('success',None) or after.get('MainPID')!='0':
            raise RuntimeError(model+' 未正常停止，禁止加载另一个模型；请检查服务日志')
        if group:
            # cgroup v2 and systemd v1. No manual kill or forced reset fallback.
            for path in [Path('/sys/fs/cgroup'+group)/'cgroup.procs',Path('/sys/fs/cgroup/systemd'+group)/'tasks']:
                if path.exists() and path.read_text().strip():raise RuntimeError(model+' 仍有子进程，禁止切换')
    def start(self,model):
        subprocess.run(['systemctl','start',UNITS[model]],check=True,capture_output=True,text=True,timeout=100)

class Controller:
    def __init__(self,board,state_dir):
        self.board=board;self.state_dir=Path(state_dir);self.state_dir.mkdir(exist_ok=True)
        self.lock=threading.RLock();self.model=None;self.target=None;self.phase='idle';self.error=None
        self.generation=0;self.thread=None;self.stopping=False
        self.metrics={};self.last_backend={}
    def snapshot(self):
        with self.lock:return dict(model=self.model,target=self.target,phase=self.phase,error=self.error,
            generation=self.generation,interactive_busy=getattr(self,'interactive_busy',False),memory=memory(),switch=self.metrics,backend=self.last_backend)
    def switch(self,target):
        if target not in UNITS:raise ValueError('未知模型')
        with self.lock:
            if self.stopping:raise RuntimeError('入口正在停止')
            if getattr(self,'interactive_busy',False):raise RuntimeError('交互任务处理中，请完成后切换模型')
            if self.phase in ('waiting','stopping','loading'):raise RuntimeError('已有模型切换进行中')
            if self.phase=='error':raise RuntimeError('硬件状态待检查；请查看日志并重新启动统一服务')
            if target==self.model and self.phase=='ready':return self.snapshot()
            if hasattr(self,'interactive'):self.interactive.clear()
            self.target=target;self.phase='waiting';self.error=None;self.generation+=1
            self.thread=threading.Thread(target=self._switch,args=(target,),daemon=True);self.thread.start()
            return self.snapshot()
    def wait_idle(self,model):
        deadline=time.monotonic()+300
        while True:
            props=self.board.props(model)
            if props['ActiveState']=='inactive':return
            if props['ActiveState']!='active':raise RuntimeError(model+' 服务状态异常，请检查后重试')
            state=self.board.status(model)
            with self.lock:self.last_backend=state
            if 'active' not in state:raise RuntimeError('后台缺少任务状态，不能安全切换')
            if not state['active'] and state.get('warmup')!='running':
                if state.get('fault') or state.get('hdmi')=='error':
                    detail=state.get('last_error') or state.get('fault') or state.get('hdmi')
                    raise RuntimeError(model+' 服务报告故障，禁止切换：'+str(detail))
                if state.get('hdmi') in ('showing','starting'):
                    code,_,raw=self.board.request(model,'POST','/api/hdmi',b'{"action":"stop"}')
                    if code not in (200,202) or json.loads(raw).get('hdmi')!='off':raise RuntimeError('HDMI 正常释放未确认')
                return
            if time.monotonic()>deadline:raise RuntimeError('等待当前任务结束超时，保留当前服务')
            time.sleep(.5)
    def _switch(self,target):
        started=time.monotonic();before=memory();stages={}
        try:
            for model in UNITS:self.wait_idle(model)
            stages['wait_idle']=time.monotonic()-started
            stop_started=time.monotonic()
            with self.lock:self.phase='stopping'
            for model in UNITS:self.board.stop(model)
            stages['release']=time.monotonic()-stop_started
            released=memory()
            load_started=time.monotonic()
            with self.lock:self.model=None;self.phase='loading'
            self.board.start(target)
            deadline=time.monotonic()+180
            while True:
                try:state=self.board.status(target)
                except (OSError,http.client.HTTPException):state={}
                if state.get('fault') or state.get('runtime_error') or state.get('hdmi')=='error':raise RuntimeError('目标模型初始化报告故障')
                ready=(state.get('mode')=='board' and state.get('inference_available') and state.get('warmup')=='ready') if target=='skin' else state.get('ready') and state.get('backend')=='pl'
                if ready:break
                if self.board.props(target)['ActiveState'] not in ('active','activating'):raise RuntimeError('目标服务启动失败')
                if time.monotonic()>deadline:raise RuntimeError('目标模型初始化超时')
                time.sleep(.2)
            stages['initialize']=time.monotonic()-load_started
            with self.lock:
                self.model=target;self.phase='ready';self.last_backend=state
                self.metrics=dict(seconds=time.monotonic()-started,stages_seconds=stages,before=before,after_release=released,after_ready=memory())
                temp=self.state_dir/'last_model.tmp';temp.write_text(target);temp.replace(self.state_dir/'last_model')
        except Exception as exc:
            with self.lock:self.phase='error';self.error=str(exc)
            print('SWITCH_ERROR',str(exc),flush=True)
    def proxy(self,model,generation,method,path,body,kind):
        with self.lock:
            if self.phase!='ready' or self.model!=model or generation!=self.generation:raise RuntimeError('模型已切换或正在初始化，请等待新页面')
            if method=='POST' and getattr(self,'interactive_busy',False):raise RuntimeError('交互任务处理中，请稍候')
            # Serialize submit/control vs switch. Switching cannot overtake an
            # upload accepted by the backend but not yet returned to the hub.
            other='camus' if model=='skin' else 'skin'
            if self.board.props(other)['ActiveState']!='inactive':raise RuntimeError('另一个模型服务意外启动，请检查硬件占用')
            return self.board.request(model,method,path,body,kind)
    def close(self):
        with self.lock:self.stopping=True
        if hasattr(self,'interactive') and self.interactive.thread:self.interactive.thread.join()
        if self.thread:self.thread.join()
        for model in UNITS:
            self.wait_idle(model);self.board.stop(model)

def rewrite(data,kind,prefix):
    if not any(t in kind for t in ('text/html','javascript')):return data
    text=data.decode('utf-8')
    text=text.replace('href="/"','href="'+prefix+'"')
    return re.sub(r'''(["'])/(api/|files/|app\.js|style\.css|favicon\.svg)''',lambda m:m[1]+prefix+m[2],text).encode('utf-8')

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def setup(self):super().setup();self.connection.settimeout(45)
    def send(self,code,body,kind='application/json; charset=utf-8',cookie=None):
        if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
        self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('X-Frame-Options','SAMEORIGIN')
        if cookie:self.send_header('Set-Cookie',cookie)
        self.end_headers();self.wfile.write(body)
    def authed(self):
        try:
            c=SimpleCookie();c.load(self.headers.get('Cookie',''));value=c['pld_hub'].value
            return secrets.compare_digest(value,self.server.session)
        except (KeyError,ValueError,CookieError):return False
    def handle_request(self):
        path=urlsplit(self.path).path;post=self.command=='POST';body=None
        if post:
            if self.headers.get('Origin')!='http://'+self.headers.get('Host',''):return self.send(403,{'error':'不允许跨站控制'})
            n=int(self.headers.get('Content-Length','0'));limit=4096 if path.startswith('/hub/') else 32*1024*1024
            if not 0<n<=limit:return self.send(413,{'error':'请求大小无效'})
            body=self.rfile.read(n)
            if len(body)!=n:return self.send(400,{'error':'上传未完成'})
        if path=='/' and not post:return self.send(200,(ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
        if path=='/hub/login' and post:
            value=json.loads(body).get('token','')
            if not isinstance(value,str) or not secrets.compare_digest(value,self.server.token):return self.send(401,{'error':'访问码不正确'})
            return self.send(200,{'ok':True},cookie='pld_hub='+self.server.session+'; HttpOnly; SameSite=Strict; Path=/')
        if not self.authed():return self.send(401,{'error':'请使用原网页访问码登录'})
        if path.startswith('/interactive'):
            from interactive_bridge import handle
            return handle(self,path,post,body)
        if path=='/hub/status' and not post:return self.send(200,self.server.controller.snapshot())
        if path=='/hub/switch' and post:return self.send(202,self.server.controller.switch(json.loads(body).get('model')))
        m=re.fullmatch(r'/(skin|camus)/g/(\d+)/(.*)',path)
        if not m:return self.send(404,{'error':'页面不存在'})
        model,epoch,rest=m.groups();prefix='/'+model+'/g/'+epoch+'/'
        code,kind,raw=self.server.controller.proxy(model,int(epoch),self.command,'/'+rest,body,self.headers.get('Content-Type','application/json'))
        self.send(code,rewrite(raw,kind,prefix),kind)
    def do_GET(self):
        try:self.handle_request()
        except (ValueError,TypeError,AttributeError):self.send(400,{'error':'请求格式无效'})
        except (RuntimeError,OSError,subprocess.SubprocessError,http.client.HTTPException) as exc:self.send(409,{'error':str(exc)})
    do_POST=do_GET

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');a=p.parse_args()
    config=json.loads(a.config.read_text());token=Path(config['token_file']).read_text().strip()
    if len(token)<24:raise ValueError('现有访问码无效，拒绝启动')
    controller=Controller(Board(token),ROOT/'state')
    server=ThreadingHTTPServer(('0.0.0.0',8765),Handler)
    server.controller=controller;server.token=token;server.session=secrets.token_urlsafe(32)
    from interactive_bridge import Manager
    server.interactive=Manager(controller,ROOT);controller.interactive=server.interactive
    def stop(*args):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        saved=ROOT/'state/last_model';model=saved.read_text().strip() if saved.exists() else 'skin'
        controller.switch(model if model in UNITS else 'skin');server.serve_forever()
    finally:server.server_close();controller.close()

if __name__=='__main__':main()
