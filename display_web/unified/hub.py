"""Authenticated common UI and serialized systemd model switching. Stdlib only."""
import argparse
import http.client
from http.cookies import SimpleCookie, CookieError
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
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
        r=subprocess.run(['systemctl','show',UNITS[model],'-p','ActiveState','-p','SubState','-p','Result','-p','ControlGroup','-p','MainPID','-p','Conflicts'],check=True,capture_output=True,text=True,timeout=10)
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
        if self.props(model)['ActiveState']=='active':
            state=self.status(model)
            if state.get('lifecycle')=='pause-resume-v1' and state.get('suspended'):
                code,_,body=self.request(model,'POST','/api/lifecycle',b'{"action":"resume"}')
                if code!=200 or json.loads(body).get('suspended'):raise RuntimeError('目标模型恢复失败')
                return
        subprocess.run(['systemctl','start',UNITS[model]],check=True,capture_output=True,text=True,timeout=100)

    def park(self,model):
        props=self.props(model)
        if props['ActiveState']=='inactive':return
        if set(props.get('Conflicts','').split()).intersection(UNITS.values()):return self.stop(model)
        state=self.status(model)
        if state.get('lifecycle')!='pause-resume-v1':return self.stop(model)
        code,_,body=self.request(model,'POST','/api/lifecycle',b'{"action":"pause"}')
        value=json.loads(body)
        if code!=200 or not value.get('suspended') or not value.get('hardware_released'):
            raise RuntimeError('旧模型硬件释放未确认')

    def check_external_owner(self):
        result=subprocess.run(['systemctl','show','pld-acdc-v2-engine.service','-p','ActiveState','-p','MainPID','-p','Result'],
                              check=True,capture_output=True,text=True,timeout=5)
        props=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
        if props.get('ActiveState')=='inactive' and props.get('MainPID')=='0' and props.get('Result')=='success':return
        if props.get('ActiveState')!='active':raise RuntimeError('ACDC 服务尚未正常释放')
        connection=http.client.HTTPConnection('127.0.0.1',8092,timeout=5)
        try:
            connection.request('GET','/api/v2/health');response=connection.getresponse()
            if response.status!=200:raise RuntimeError('ACDC 释放状态不可用')
            state=json.loads(response.read())
        finally:connection.close()
        if not (str(state.get('owner_pid'))==props.get('MainPID') and state.get('suspended')
                and state.get('hardware_released') and state.get('hdmi')=='off' and not state.get('active_job')):
            raise RuntimeError('ACDC 未确认暂停并释放硬件')

class Controller:
    def __init__(self,board,state_dir):
        self.board=board;self.state_dir=Path(state_dir);self.state_dir.mkdir(exist_ok=True)
        self.lock=threading.RLock();self.model=None;self.target=None;self.phase='idle';self.error=None
        self.generation=0;self.thread=None;self.stopping=False
        self.metrics={};self.last_backend={}
        self.parked=False
        self.proxy_busy=False
        self.proxy_idle=threading.Event();self.proxy_idle.set()
    def snapshot(self):
        with self.lock:return dict(model=self.model,target=self.target,phase=self.phase,error=self.error,
            generation=self.generation,interactive_busy=getattr(self,'interactive_busy',False),evaluation_busy=getattr(self,'evaluation_busy',False),proxy_busy=self.proxy_busy,memory=memory(),switch=self.metrics,backend=self.last_backend,
            lifecycle='resident-hub-v1',owner_pid=os.getpid(),hardware_released=self.parked)
    def lifecycle(self,action,target=None):
        if action=='resume':return self.switch(target,resume=True)
        if action!='pause':raise ValueError('未知生命周期操作')
        with self.lock:
            if self.parked:return self.snapshot()
            if self.stopping or self.proxy_busy or self.phase!='ready' or getattr(self,'interactive_busy',False) or getattr(self,'evaluation_busy',False):
                raise RuntimeError('当前任务或切换未完成')
            self.phase='pausing';self.generation+=1
        try:
            if hasattr(self,'interactive'):self.interactive.clear()
            for model in UNITS:self.wait_idle(model)
            for model in UNITS:self.board.park(model)
            with self.lock:self.parked=True;self.phase='paused';self.last_backend={}
        except Exception as exc:
            with self.lock:self.phase='error';self.error=str(exc)
            raise
        return self.snapshot()

    def switch(self,target,*,resume=False):
        if target not in UNITS:raise ValueError('未知模型')
        with self.lock:
            if self.stopping:raise RuntimeError('入口正在停止')
            if self.proxy_busy:raise RuntimeError('操作正在提交，请稍候再切换')
            if getattr(self,'evaluation_busy',False):raise RuntimeError('评测进行中，请等待恢复模型')
            if self.parked and not resume:raise RuntimeError('入口已暂停，请通过统一入口选择模型')
            if getattr(self,'interactive_busy',False):raise RuntimeError('交互任务处理中，请完成后切换模型')
            if self.phase in ('waiting','stopping','loading','pausing'):raise RuntimeError('已有模型切换进行中')
            if self.phase=='error':raise RuntimeError('硬件状态待检查；请查看日志并重新启动统一服务')
            if target==self.model and self.phase=='ready':return self.snapshot()
            if hasattr(self,'interactive'):self.interactive.clear()
            self.target=target;self.phase='waiting';self.error=None;self.generation+=1;self.parked=False
            self.thread=threading.Thread(target=self._switch,args=(target,),daemon=True);self.thread.start()
            return self.snapshot()
    def wait_idle(self,model):
        deadline=time.monotonic()+300
        while True:
            props=self.board.props(model)
            if props['ActiveState']=='inactive':return
            if props['ActiveState']!='active':raise RuntimeError(model+' 服务状态异常，请检查后重试')
            try:state=self.board.status(model)
            except (OSError,http.client.HTTPException):
                if time.monotonic()>deadline:raise RuntimeError(model+' 软件启动状态不可用')
                time.sleep(.2);continue
            with self.lock:self.last_backend=state
            if 'active' not in state:raise RuntimeError('后台缺少任务状态，不能安全切换')
            if not state['active'] and state.get('warmup')!='running':
                if state.get('fault') or state.get('hdmi')=='error':
                    detail=state.get('last_error') or state.get('fault') or state.get('hdmi')
                    raise RuntimeError(model+' 服务报告故障，禁止切换：'+str(detail))
                if state.get('hdmi') in ('showing','starting'):
                    code,_,raw=self.board.request(model,'POST','/api/hdmi',b'{"action":"stop","_automatic":true}')
                    if code not in (200,202) or json.loads(raw).get('hdmi')!='off':raise RuntimeError('HDMI 正常释放未确认')
                return
            if time.monotonic()>deadline:raise RuntimeError('等待当前任务结束超时，保留当前服务')
            time.sleep(.5)
    def _switch(self,target):
        started=time.monotonic();before=memory();stages={}
        try:
            if hasattr(self.board,'check_external_owner'):self.board.check_external_owner()
            for model in UNITS:self.wait_idle(model)
            stages['wait_idle']=time.monotonic()-started
            stop_started=time.monotonic()
            with self.lock:self.phase='stopping'
            state=self.board.status(target) if self.board.props(target)['ActiveState']=='active' else {}
            adopted=self.model is None and state.get('hardware_ready',False)
            for model in UNITS:
                if adopted and model==target:continue
                if hasattr(self.board,'park'):self.board.park(model)
                else:self.board.stop(model)
            stages['release']=time.monotonic()-stop_started
            released=memory()
            load_started=time.monotonic()
            with self.lock:self.model=None;self.phase='loading'
            if not adopted:self.board.start(target)
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
                self.metrics=dict(seconds=time.monotonic()-started,stages_seconds=stages,before=before,after_release=released,after_ready=memory(),adopted_preloaded_target=bool(adopted))
                temp=self.state_dir/'last_model.tmp';temp.write_text(target);temp.replace(self.state_dir/'last_model')
        except Exception as exc:
            with self.lock:self.phase='error';self.error=str(exc)
            print('SWITCH_ERROR',str(exc),flush=True)
    def proxy(self,model,generation,method,path,body,kind):
        mutation=method=='POST'
        with self.lock:
            if self.stopping:raise RuntimeError('入口正在停止')
            if getattr(self,'evaluation_busy',False):raise RuntimeError('评测独占硬件，等待恢复模型')
            if self.phase!='ready' or self.model!=model or generation!=self.generation:raise RuntimeError('模型已切换或正在初始化，请等待新页面')
            if mutation and (self.proxy_busy or getattr(self,'interactive_busy',False)):raise RuntimeError('任务或操作正在提交，请稍候')
            if mutation:
                # Reserve admission while network I/O runs without the state
                # lock. Switch/evaluation/ROI cannot overtake an unacked POST.
                self.proxy_busy=True;self.proxy_idle.clear()
        try:
            other='camus' if model=='skin' else 'skin'
            if self.board.props(other)['ActiveState']!='inactive':
                state=self.board.status(other)
                if not (state.get('suspended') and state.get('hardware_released') and state.get('hdmi')=='off'):
                    raise RuntimeError('另一个模型未释放硬件，请检查占用')
            with self.lock:
                if self.phase!='ready' or self.model!=model or generation!=self.generation:
                    raise RuntimeError('模型已切换，请等待新页面')
            return self.board.request(model,method,path,body,kind)
        finally:
            if mutation:
                with self.lock:self.proxy_busy=False;self.proxy_idle.set()
    def close(self):
        with self.lock:self.stopping=True
        if not self.proxy_idle.wait(60):raise RuntimeError('代理操作未结束，保留当前服务与硬件现场')
        if getattr(self,'evaluation',None) and self.evaluation.thread:
            self.evaluation.cancelled.set();self.evaluation.thread.join()
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
        if path=='/hub/ownership' and not post:
            if self.client_address[0]!='127.0.0.1':return self.send(403,{'error':'board-local ownership'})
            value=self.server.controller.snapshot()
            return self.send(200,{key:value.get(key) for key in ('phase','owner_pid','hardware_released','interactive_busy','evaluation_busy')})
        if not self.authed():return self.send(401,{'error':'请使用原网页访问码登录'})
        if path=='/hub/lifecycle' and post:
            if self.client_address[0]!='127.0.0.1':return self.send(403,{'error':'board-local lifecycle'})
            value=json.loads(body)
            return self.send(200,self.server.controller.lifecycle(value.get('action'),value.get('model')))
        if path.startswith('/evaluation'):
            from evaluation_bridge import handle
            return handle(self,path,post,body)
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
    from evaluation_bridge import Manager as EvaluationManager
    server.evaluation=EvaluationManager(controller,ROOT) if (ROOT/'evaluation_config.json').is_file() else None
    controller.evaluation=server.evaluation
    def stop(*args):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        saved=ROOT/'state/last_model';model=saved.read_text().strip() if saved.exists() else 'skin'
        controller.switch(model if model in UNITS else 'skin');server.serve_forever()
    finally:server.server_close();controller.close()

if __name__=='__main__':main()
