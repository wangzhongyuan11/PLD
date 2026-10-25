"""Hub-owned sequential CPU/PL evaluation; hardware is used only by a locked child."""
import base64
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

class Manager:
    def __init__(self, controller, root):
        self.controller=controller;self.root=Path(root);self.thread=None;self.job=None
        self.cancelled=threading.Event();self.guard=threading.RLock()
        self.config=json.loads((self.root/'evaluation_config.json').read_text())
        self.store=Path(self.config['data']);self.store.mkdir(parents=True,exist_ok=True)
        self.process=None;self.began=None
    def save(self, **values):
        with self.guard:
            self.job.update(values)
            p=self.store/self.job['id']/'status.json';tmp=p.with_suffix('.tmp')
            tmp.write_text(json.dumps(self.job,ensure_ascii=False),encoding='utf-8');tmp.replace(p)
    def snapshot(self):
        with self.guard:
            if not self.job:return None
            result=json.loads(json.dumps(self.job))
            if self.began and result['state']=='running':result['elapsed_seconds']=round(time.monotonic()-self.began,1)
            log=self.store/result['id']/'execution.log'
            if log.is_file():
                lines=log.read_text(errors='replace').splitlines()
                result['progress_line']=next((line for line in reversed(lines) if line.startswith(('INITIALIZED','FIRST_FORWARD_DONE','FRAME_DONE'))),'')
            return result
    def submit(self, body):
        from interactive_bridge import decode_image
        from PIL import Image
        c=self.controller
        with c.lock:
            if c.stopping or c.phase!='ready' or getattr(c,'interactive_busy',False) or getattr(c,'evaluation_busy',False):raise RuntimeError('当前模型或任务未就绪')
            if body.get('model')!=c.model or body.get('generation')!=c.generation:raise ValueError('模型已改变，请刷新')
            repeat=body.get('repeat',1);warmup=body.get('warmup',0)
            if type(repeat)!=int or not 1<=repeat<=10 or type(warmup)!=int or not 0<=warmup<=2:raise ValueError('重复次数1–10，预热0–2')
            image=decode_image(body.get('image'))
            ident=uuid.uuid4().hex;folder=self.store/ident;folder.mkdir()
            frame=Image.fromarray(image)
            if c.model=='camus':frame=frame.convert('L')
            frame.save(folder/('input.jpg' if c.model=='skin' else 'input.png'),**({'quality':100,'subsampling':0} if c.model=='skin' else {}))
            self.job=dict(id=ident,model=c.model,state='running',stage='等待已有任务结束',repeat=repeat,warmup=warmup,results={},error=None,restored=False,
                scope='256×256网络网格；无普通网页小岛/孔洞后处理；同一准备输入；NumPy INT8参考；无GT精度',
                input_encoding='JPEG quality100 4:4:4' if c.model=='skin' else '8-bit grayscale PNG')
            self.began=time.monotonic();self.cancelled.clear();c.evaluation_busy=True
            self.save();self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
            return self.snapshot()
    def command(self,args,folder):
        env=dict(os.environ,XILINX_XRT='/usr',PLD_EVAL_HUB_PARENT=str(os.getpid()),PYTHONUNBUFFERED='1')
        with (folder/'execution.log').open('ab') as log:
            p=subprocess.Popen([sys.executable,str(Path(self.config['tools'])/'run_eval.py')]+list(map(str,args)),stdout=log,stderr=log,env=env)
            self.process=p
            try:p.wait(timeout=1800)
            except subprocess.TimeoutExpired:raise RuntimeError('测试超过30分钟，进程未强杀；禁止恢复硬件，请检查日志')
            self.process=None
            if p.returncode:raise RuntimeError('评测失败：'+(folder/'execution.log').read_text(errors='replace')[-1800:])
    def preview(self,folder,backend):
        import numpy as np
        from PIL import Image
        labels=np.load(folder/backend/'case_0000_labels.npy',allow_pickle=False)
        a=np.load(folder/'inputs/case_0000_fp32.npy',allow_pickle=False)[0]
        rgb=np.moveaxis(a,0,-1)
        if rgb.shape[-1]==1:rgb=np.repeat(rgb,3,axis=-1)
        rgb=np.uint8(np.clip(rgb*255,0,255));mask=labels>0
        colors=np.array([[0,0,0],[255,185,0],[40,210,240],[230,70,170]],np.uint8)
        overlay=rgb.copy();overlay[mask]=(.55*rgb[mask]+.45*colors[labels[mask]]).astype(np.uint8)
        for name,img in [('overlay',overlay),('cut',rgb*mask[:,:,None]),('mask',colors[labels]),('original',rgb)]:Image.fromarray(img).save(folder/(backend+'_'+name+'.png'))
    def run(self):
        c=self.controller;job=self.snapshot();model=job['model'];folder=self.store/job['id'];cfg=self.config[model]
        released=False;failed=None
        try:
            for m in ('skin','camus'):c.wait_idle(m)
            self.save(stage='正常释放模型与HDMI')
            for m in ('skin','camus'):c.board.stop(m)
            released=True
            if hasattr(c,'interactive'):c.interactive.clear()
            common=['--source',self.config['source'],'--model',model,'--model-root',cfg['model']]
            self.save(stage='准备CPU与FPGA共用输入')
            self.command(['prepare']+common+['--input',folder/('input.jpg' if model=='skin' else 'input.png'),'--output',folder/'inputs'],folder)
            for backend in ('pl','cpu-int8'):
                if self.cancelled.is_set():break
                label='FPGA' if backend=='pl' else 'CPU（整数参考较慢，请等待）'
                self.save(stage=label+'：初始化、首次推理及正式测量')
                args=['bench']+common+['--dataset',folder/'inputs/dataset.json','--backend',backend,'--warmup',job['warmup'],'--repeat',job['repeat'],'--output',folder/backend]
                if backend=='pl':
                    if model=='skin':args+=['--bit',cfg['bit'],'--native-copy',cfg['native_copy'],'--native-control',cfg['native_control'],'--fuse']
                    else:args+=['--runtime',cfg['runtime']]
                self.command(args,folder);self.preview(folder,backend)
                report=json.loads((folder/backend/'benchmark.json').read_text())
                if not report.get('complete'):raise RuntimeError('不完整的测试结果')
                with self.guard:self.job['results'][backend]=report
                self.save()
            if not self.cancelled.is_set():
                self.save(stage='比较输出并生成报告')
                self.command(['compare',folder/'cpu-int8/benchmark.json',folder/'pl/benchmark.json','--output',folder/'comparison.json'],folder)
                self.save(comparison=json.loads((folder/'comparison.json').read_text()))
                import numpy as np
                from PIL import Image
                a=np.load(folder/'cpu-int8/case_0000_labels.npy',allow_pickle=False);b=np.load(folder/'pl/case_0000_labels.npy',allow_pickle=False)
                diff=np.zeros((256,256,3),np.uint8);diff[a!=b]=[255,70,70];Image.fromarray(diff).save(folder/'difference.png')
                self.command(['report','--run',folder/'cpu-int8/benchmark.json','--run',folder/'pl/benchmark.json','--evidence',folder/'comparison.json','--output',folder/'report'],folder)
        except Exception as exc:failed=str(exc);self.save(error=failed)
        finally:
            unsafe=self.process is not None
            for p in folder.glob('*/benchmark.json'):
                try:unsafe=unsafe or bool(json.loads(p.read_text()).get('close_error'))
                except Exception:unsafe=True
            if released and not unsafe:
                self.save(stage='恢复原模型，请稍候')
                c._switch(model)
                self.save(restored=c.phase=='ready')
                if c.phase!='ready':self.save(error=(failed or '')+' 模型恢复失败：'+str(c.error))
            elif unsafe or failed:
                with c.lock:c.phase='error';c.error='评测进程或硬件释放状态待检查，未自动恢复'
            with c.lock:
                self.save(elapsed_seconds=round(time.monotonic()-self.began,1) if self.began else None,state='error' if failed or not self.job['restored'] else ('cancelled' if self.cancelled.is_set() else 'done'),stage='已结束')
                c.evaluation_busy=False
    def history(self):
        rows=[]
        for p in sorted(self.store.glob('*/status.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:30]:
            d=json.loads(p.read_text());rows.append({k:d.get(k) for k in ('id','model','state','stage')})
        return rows

def handle(h,path,post,body):
    m=h.server.evaluation
    if m is None:return h.send(503,{'error':'评测工具尚未安装，请配置evaluation_config.json'})
    if path=='/evaluation' and not post:return h.send(200,(m.root/'evaluation.html').read_bytes(),'text/html; charset=utf-8')
    if path=='/evaluation/status' and not post:return h.send(200,{'job':m.snapshot(),'controller':m.controller.snapshot(),'history':m.history()})
    if path=='/evaluation/start' and post:return h.send(202,m.submit(json.loads(body)))
    if path=='/evaluation/cancel' and post:m.cancelled.set();return h.send(200,{'message':'将在当前后端完成后取消；不会强杀硬件进程'})
    match=re.fullmatch(r'/evaluation/files/([0-9a-f]{32})/(status.json|comparison.json|difference.png|(?:pl|cpu-int8)_(?:overlay|cut|mask|original).png|report/(?:report.html|report.json|report.md|comparison.csv))',path)
    if match and not post:
        p=m.store/match[1]/match[2]
        if not p.is_file():return h.send(404,{'error':'结果尚未生成'})
        kind='image/png' if p.suffix=='.png' else ('text/html; charset=utf-8' if p.suffix=='.html' else 'application/json' if p.suffix=='.json' else 'text/plain; charset=utf-8')
        return h.send(200,p.read_bytes(),kind)
    return h.send(404,{'error':'未知评测接口'})
