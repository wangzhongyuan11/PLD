"""Authenticated hub add-on: reuse sole existing PL owner, refine on CPU."""
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import threading
import time
import uuid
import numpy as np
from PIL import Image, ImageOps
from prompt_seg.contracts import Request
from prompt_seg.service import SegmentationService

MAX_PIXELS = 1024*1024
SPECS = {'skin':'fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef',
         'camus':'68b4f425cc3612ad72d5e83955c2018981edee912fbc0937e9ac8c1718e7b5ae'}


def decode_image(value):
    if not isinstance(value,str) or len(value)>12*1024*1024: raise ValueError('图片编码无效或过大')
    raw=base64.b64decode(value.split(',')[-1],validate=True)
    with Image.open(io.BytesIO(raw)) as im:
        if im.format not in ('JPEG','PNG') or im.width*im.height>MAX_PIXELS:
            raise ValueError('交互模式支持 JPG/PNG，最多1048576像素；请先缩小图片')
        return np.array(ImageOps.exif_transpose(im).convert('RGB'))


class Predictor:
    def __init__(self, manager):
        self.manager=manager;self.cache_key=None;self.cached=None;self.identity=None

    def clear(self):
        self.cache_key=None;self.cached=None;self.identity=None

    def predict(self,image,model,variant,backend):
        began=time.monotonic()
        m=self.manager;c=m.controller
        key=(c.generation,model,image.shape,hashlib.sha256(image.tobytes()).hexdigest())
        if key==self.cache_key:
            probs=self.cached.copy()
            return probs,dict(self.identity,prediction_cache_hit=True,
                prediction_timings_ms={'cache_lookup_copy':(time.monotonic()-began)*1000})
        self.clear()
        encoded=io.BytesIO()
        # The frozen skin runtime accepts JPEG only. Document the encoding.
        frame=Image.fromarray(image)
        # The frozen CAMUS reader accepts scalar gray images only. Browser
        # canvas/decode_image always produces RGB, including gray uploads.
        if model=='camus':frame=frame.convert('L')
        frame.save(encoded,format='JPEG' if model=='skin' else 'PNG',**({'quality':100,'subsampling':0} if model=='skin' else {'compress_level':1}))
        b64=base64.b64encode(encoded.getvalue()).decode()
        body={'mode':'infer','image':b64,'name':'interactive ROI','thickness':2} if model=='skin' else {'input_name':'roi.png','files':[{'name':'roi.png','data_base64':b64}]}
        submitted=time.monotonic()
        row=m.request(model,'POST','/api/jobs',body)
        ident=row['id']
        if not re.fullmatch('[0-9a-f]{32}',ident): raise RuntimeError('后端任务标识无效')
        deadline=time.monotonic()+300
        while row['state'] not in ('done','error'):
            if time.monotonic()>deadline: raise RuntimeError('等待后端超时；原推理未被强杀，请检查服务状态')
            time.sleep(.25);row=m.request(model,'GET','/api/jobs/'+ident)
        if row['state']!='done': raise RuntimeError('FPGA任务失败：'+str(row.get('error')))
        completed=time.monotonic()
        folder=Path(m.backends[model]['data'])/ident
        h,w=image.shape[:2]
        if model=='skin':
            from pld_result import load_result
            result=load_result(folder/'inference/frame_000001.json')
            if result.metadata['model_spec_sha256']!=SPECS[model]:raise RuntimeError('皮肤模型不匹配')
            left,top,nw,nh=result.metadata['valid_rect']
            p=1/(1+np.exp(-result.head[0,0,top:top+nh,left:left+nw].astype(np.float32)*result.metadata['head_output_scale']))
            p=np.asarray(Image.fromarray(p).resize((w,h),getattr(Image, 'Resampling', Image).NEAREST))
            probs=np.stack([1-p,p])
        else:
            from camus.result import letterbox_meta_for_native, _resize_output_channel
            out=folder/'output';meta=json.loads((out/'head.json').read_text())
            if meta['spec_sha256']!=SPECS[model] or meta['shape']!=[1,4,256,256] or meta['dtype']!='int8':raise RuntimeError('CAMUS模型或输出不匹配')
            scale=float(meta['scale'])
            if not np.isfinite(scale) or scale<=0:raise RuntimeError('CAMUS输出尺度无效')
            head=np.frombuffer((out/'head.bin').read_bytes(),np.int8).reshape(4,256,256)
            geom=letterbox_meta_for_native(h,w);t,l=geom['top'],geom['left'];nh,nw=geom['new_height'],geom['new_width']
            logits=np.stack([_resize_output_channel(a[t:t+nh,l:l+nw].astype(np.float32)*np.float32(scale),h,w) for a in head])
            logits-=logits.max(axis=0);probs=np.exp(logits);probs/=probs.sum(axis=0)
        identity=dict(model=model,variant='delivery' if model=='skin' else 'candidate',backend='pl_int8',spec_sha256=SPECS[model],
                      prediction_cache_hit=False,software_only=False,refinement_backend='PS/CPU scipy maximum_flow',
                      board_verified=False,execution='existing PL backend completed; interactive integration needs board acceptance',
                      source_job=ident,input_encoding='JPEG quality100 4:4:4' if model=='skin' else '8-bit grayscale PNG (RGB converted to L)',
                      prediction_timings_ms=dict(input_encoding=(submitted-began)*1000,
                          backend_job_wait=(completed-submitted)*1000,
                          probability_restore=(time.monotonic()-completed)*1000),
                      backend_timing=row.get('result',{}).get('stages_seconds',
                          row.get('statistics',{}).get('final_stage_seconds',{})))
        self.cache_key=key;self.cached=probs.copy();self.identity=identity
        return probs,identity


class Manager:
    def __init__(self,controller,root):
        self.controller=controller;self.root=Path(root);self.guard=threading.Lock();self.job=None;self.thread=None
        self.display_cache_key=None;self.display_cache=None
        self.backends=json.loads((self.root/'installation.json').read_text())['backends']
        for kind in ('skin','camus'):sys.path.insert(0,self.backends[kind]['source'])
        self.predictor=Predictor(self);self.service=SegmentationService(self.predictor)

    def request(self,model,method,path,body=None):
        code,_,raw=self.controller.board.request(model,method,path,None if body is None else json.dumps(body).encode())
        data=json.loads(raw)
        if code not in (200,202):raise RuntimeError(data.get('error','后端请求失败'))
        return data

    def submit(self,body):
        began=time.monotonic()
        c=self.controller
        with c.lock:
            if c.phase!='ready' or c.stopping or getattr(c,'interactive_busy',False):raise RuntimeError('请等待模型或当前任务完成')
            if body.get('model')!=c.model or body.get('generation')!=c.generation:raise ValueError('模型已改变，请刷新交互页面')
            # Reject concurrently running normal-page jobs before allocating arrays.
            other='camus' if c.model=='skin' else 'skin'
            if c.board.props(other)['ActiveState']!='inactive':raise RuntimeError('另一个模型未释放，禁止交互推理')
            status=c.board.status(c.model)
            if status.get('active') or status.get('fault') or status.get('warmup')=='running':raise RuntimeError('后端忙或异常，请先完成普通分割')
            image=decode_image(body.get('image'));params=body.get('parameters',{})
            req=Request.parse(c.model,params,image.shape[1],image.shape[0])
            if req.backend!='int8' or req.variant!=('delivery' if c.model=='skin' else 'candidate'):raise ValueError('交互模式固定使用当前板端模型，不能切换软件权重')
            identifier=uuid.uuid4().hex
            with self.guard:self.job=dict(id=identifier,state='running')
            c.interactive_busy=True
            self.thread=threading.Thread(target=self.run,args=(identifier,c.model,image,params,began,time.monotonic()-began),daemon=False);self.thread.start()
            return dict(id=identifier,state='running')

    def run(self,identifier,model,image,params,began=None,submit_seconds=0):
        if began is None:began=time.monotonic()
        try:
            result=self.service.process(model,image,params)
            result['parameters']['backend']='pl_int8';result['evidence_scope']='FPGA概率 + PS交互图割；本补丁尚待板端验收'
            from prompt_seg.service import png_data
            from quad_display import enhance,thumbnail
            started=time.monotonic()
            display_key=(self.controller.generation,result['image']['sha256'])
            hit=display_key==self.display_cache_key
            if not hit:
                self.display_cache={'original_png':png_data(image),
                    'enhanced_png':png_data(np.asarray(enhance(thumbnail(image))))}
                self.display_cache_key=display_key
            result['outputs'].update(self.display_cache)
            result['identity']['display_cache_hit']=hit
            result['timings_ms']['original_enhanced_png']=(time.monotonic()-started)*1000
            result['timings_ms']['request_validation_decode']=submit_seconds*1000
            result['elapsed_ms']=round((time.monotonic()-began)*1000,2)
            result['timing_scope']='server submit through all five PNG/base64 outputs; excludes HTTP JSON transfer and browser polling/render'
            result['warnings'].append('交互输出单独下载；原任务历史、ZIP及HDMI仍是交互前结果。皮肤ROI以JPEG质量100编码。')
            if model=='camus':result['warnings'].append('CAMUS 输入转换为8位灰度；彩色图不代表该模型的有效医学输入。')
            with self.guard:self.job=dict(id=identifier,state='done',result=result)
        except Exception as exc:
            self.predictor.clear()
            with self.guard:self.job=dict(id=identifier,state='error',error=str(exc))
        finally:
            with self.controller.lock:self.controller.interactive_busy=False

    def snapshot(self,identifier):
        with self.guard:
            if not self.job or self.job['id']!=identifier:raise ValueError('结果已被新任务替换，请重新处理')
            return self.job.copy()

    def clear(self):
        with self.guard:self.job=None
        self.predictor.clear()
        self.display_cache_key=None;self.display_cache=None


def handle(handler,path,post,body):
    root=Path(__file__).resolve().parent
    if path in ('/interactive','/interactive.js') and not post:
        file='interactive.html' if path=='/interactive' else 'interactive.js'
        return handler.send(200,(root/file).read_bytes(),'text/html; charset=utf-8' if file.endswith('html') else 'application/javascript')
    manager=handler.server.interactive
    if path=='/interactive/jobs' and post:return handler.send(202,manager.submit(json.loads(body)))
    match=re.fullmatch(r'/interactive/jobs/([0-9a-f]{32})',path)
    if match and not post:return handler.send(200,manager.snapshot(match[1]))
    return handler.send(404,{'error':'交互接口不存在'})
