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
from prompt_seg.service import SegmentationService, png_sha256

MAX_PIXELS = 1024*1024
MAX_UPLOAD_BYTES = 16*1024*1024
REPLAY_SCHEMA = 'pld-interactive-replay-v1'
SPECS = {'skin':'fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef',
         'camus':'68b4f425cc3612ad72d5e83955c2018981edee912fbc0937e9ac8c1718e7b5ae'}


def decode_image(value):
    if not isinstance(value,str) or len(value)>12*1024*1024: raise ValueError('图片编码无效或过大')
    raw=base64.b64decode(value.split(',')[-1],validate=True)
    if len(raw)>MAX_UPLOAD_BYTES: raise ValueError('图片压缩后超过16MB')
    with Image.open(io.BytesIO(raw)) as im:
        if im.format not in ('JPEG','PNG') or im.width*im.height>MAX_PIXELS:
            raise ValueError('交互模式支持 JPG/PNG，最多1048576像素；请先缩小图片')
        return np.array(ImageOps.exif_transpose(im).convert('RGB'))


def decode_png_rgb(value):
    """Decode an exported processed-image PNG and return RGB pixels (no alpha)."""
    if not isinstance(value,str) or not value.startswith('data:image/png;base64,'):
        raise ValueError('回放记录缺少处理图 PNG')
    if len(value)>12*1024*1024: raise ValueError('回放处理图过大')
    raw=base64.b64decode(value.split(',',1)[1],validate=True)
    if len(raw)>MAX_UPLOAD_BYTES: raise ValueError('回放处理图压缩后超过16MB')
    with Image.open(io.BytesIO(raw)) as im:
        if im.format!='PNG' or im.width*im.height>MAX_PIXELS:
            raise ValueError('回放处理图必须是 PNG 且不超过1048576像素')
        return np.array(im.convert('RGB'))


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
                pl_executed_this_request=False,
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
                      prediction_cache_hit=False,pl_executed_this_request=True,software_only=False,
                      refinement_backend='PS/CPU scipy maximum_flow',
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
        self.image_slot=None
        self.backends=json.loads((self.root/'installation.json').read_text())['backends']
        for kind in ('skin','camus'):sys.path.insert(0,self.backends[kind]['source'])
        self.predictor=Predictor(self);self.service=SegmentationService(self.predictor)

    # -- shared helpers -----------------------------------------------------
    def _check_context(self,body):
        c=self.controller
        if body.get('model')!=c.model or body.get('generation')!=c.generation:
            raise ValueError('模型已改变，请刷新交互页面')
        return c

    def _busy(self):
        c=self.controller
        return bool(c.phase!='ready' or c.stopping or getattr(c,'interactive_busy',False)
                    or getattr(c,'evaluation_busy',False))

    def upload(self,body):
        """Single authenticated image slot; validated, decoded and replaced under c.lock.

        ``expected_sha256`` lets a caller pin the exact pixels it believes it is
        uploading (for example an imported replay). Everything is verified before
        the slot is replaced, so a failed upload leaves the previous token usable.
        """
        c=self.controller
        requested=body.get('model')
        generation=body.get('generation')
        expected=body.get('expected_sha256')
        with c.lock:
            if requested!=c.model or generation!=c.generation:
                raise ValueError('模型已改变，请刷新交互页面')
            if self._busy():
                raise RuntimeError('请等待当前任务完成后再上传图片')
            image=decode_image(body.get('image'))
            digest=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            if expected is not None:
                if not isinstance(expected,str) or len(expected)!=64:
                    raise ValueError('expected_sha256 格式无效')
                if digest!=expected:
                    raise ValueError('上传图像像素哈希与预期不一致，已保留原有图像槽')
            slot=dict(token=uuid.uuid4().hex,image=image,model=c.model,generation=c.generation,
                width=int(image.shape[1]),height=int(image.shape[0]),sha256=digest)
            with self.guard:self.image_slot=slot
        return {'image_token':slot['token'],'sha256':slot['sha256'],
                'width':slot['width'],'height':slot['height'],
                'model':slot['model'],'generation':slot['generation']}

    def _image_for(self,body):
        c=self.controller
        token=body.get('image_token')
        if token is None:
            return decode_image(body.get('image')),None
        if not isinstance(token,str) or not re.fullmatch('[0-9a-f]{32}',token):
            raise ValueError('图像 token 无效')
        with c.lock:
            with self.guard:slot=self.image_slot
            if slot is None or slot['token']!=token:
                raise ValueError('图像 token 已失效，请重新上传图片')
            if slot['model']!=c.model or slot['generation']!=c.generation:
                raise ValueError('图像 token 与当前模型不匹配，请重新上传')
            return slot['image'].copy(),token

    def request(self,model,method,path,body=None):
        code,_,raw=self.controller.board.request(model,method,path,None if body is None else json.dumps(body).encode())
        data=json.loads(raw)
        if code not in (200,202):raise RuntimeError(data.get('error','后端请求失败'))
        return data

    def submit(self,body):
        began=time.monotonic()
        c=self.controller
        with c.lock:
            if self._busy():raise RuntimeError('请等待模型或当前任务完成')
            self._check_context(body)
            # Reject concurrently running normal-page jobs before allocating arrays.
            other='camus' if c.model=='skin' else 'skin'
            if c.board.props(other)['ActiveState']!='inactive':raise RuntimeError('另一个模型未释放，禁止交互推理')
            status=c.board.status(c.model)
            if status.get('active') or status.get('fault') or status.get('warmup')=='running':raise RuntimeError('后端忙或异常，请先完成普通分割')
            image,image_token=self._image_for(body)
            params=body.get('parameters',{})
            req=Request.parse(c.model,params,image.shape[1],image.shape[0])
            if req.backend!='int8' or req.variant!=('delivery' if c.model=='skin' else 'candidate'):raise ValueError('交互模式固定使用当前板端模型，不能切换软件权重')
            identifier=uuid.uuid4().hex
            with self.guard:self.job=dict(id=identifier,state='running')
            c.interactive_busy=True
            self.thread=threading.Thread(target=self.run,args=(identifier,c.model,image,params,began,
                time.monotonic()-began,image_token),daemon=False);self.thread.start()
            return dict(id=identifier,state='running')

    def run(self,identifier,model,image,params,began=None,submit_seconds=0,image_token=None):
        if began is None:began=time.monotonic()
        try:
            c=self.controller
            scope=(c.generation,SPECS[model])
            result=self.service.process(model,image,params,scope=scope)
            # Canonical, resubmittable parameters: the request parser only accepts
            # int8/fp32, so pl_int8 must never leak into the replay record.
            canonical=dict(result['parameters'])
            canonical['backend']='int8'
            canonical['variant']='delivery' if model=='skin' else 'candidate'
            result['parameters']['backend']='pl_int8'
            result['evidence_scope']='FPGA概率 + PS交互图割；本补丁尚待板端验收'
            from prompt_seg.service import png_data
            from quad_display import enhance,thumbnail
            started=time.monotonic()
            display_key=(c.generation,result['image']['sha256'])
            hit=display_key==self.display_cache_key
            if not hit:
                self.display_cache={'original_png':png_data(image),
                    'enhanced_png':png_data(np.asarray(enhance(thumbnail(image))))}
                self.display_cache_key=display_key
            result['outputs']=dict(result['outputs'])
            result['outputs'].update(self.display_cache)
            identity=result['identity']
            identity['display_cache_hit']=hit
            identity['probability_cache_hit']=bool(identity.get('prediction_cache_hit'))
            identity['pl_executed_this_request']=bool(identity.get('pl_executed_this_request',
                not identity['probability_cache_hit'] and not identity.get('software_only',False)))
            identity['baseline_cache_hit']=bool(result['baseline']['cache_hit'])
            result['execution']={'pl_executed_this_request':identity['pl_executed_this_request'],
                'probability_cache_hit':identity['probability_cache_hit'],
                'baseline_cache_hit':identity['baseline_cache_hit'],
                'refinement_backend':'PS/CPU scipy maximum_flow',
                'network_prompt_conditioned':False,
                'board_verified':False,
                'note':'本次请求未运行FPGA时不得把上次预测耗时计入本次PL时间'}
            result['timings_ms']['original_enhanced_png']=(time.monotonic()-started)*1000
            result['timings_ms']['request_validation_decode']=submit_seconds*1000
            result['timing_scope']='server submit through baseline/final PNG outputs and replay metadata; excludes HTTP JSON transfer and browser polling/render'
            result['image_token']=image_token
            result['replay']={'schema':REPLAY_SCHEMA,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'model':model,'generation':c.generation,'spec_sha256':SPECS[model],
                'coordinate_space':'processed_image_pixels',
                'image':dict(result['image']),
                'parameters':canonical,
                'results':{'baseline_labels_sha256':result['label_hashes']['baseline'],
                    'final_labels_sha256':result['label_hashes']['final'],
                    'baseline_target_mask_png_sha256':png_sha256(result['baseline']['outputs']['target_mask_png']),
                    'final_target_mask_png_sha256':png_sha256(result['outputs']['target_mask_png'])},
                'execution':dict(result['execution']),'timings_ms':dict(result['timings_ms']),
                'note':'导入回放需要同一 model/spec；处理图 PNG 随导出 JSON 一起保存'}
            result['warnings'].append('交互输出单独下载；原任务历史、ZIP及HDMI仍是交互前结果。皮肤ROI以JPEG质量100编码。')
            if model=='camus':result['warnings'].append('CAMUS 输入转换为8位灰度；彩色图不代表该模型的有效医学输入。')
            result['elapsed_ms']=round((time.monotonic()-began)*1000,2)
            with self.guard:self.job=dict(id=identifier,state='done',result=result)
        except Exception as exc:
            self.predictor.clear()
            # A failed request must not leave a baseline/context keyed to
            # probabilities whose version is no longer trustworthy.
            self.service.clear()
            with self.guard:self.job=dict(id=identifier,state='error',error=str(exc))
        finally:
            with self.controller.lock:self.controller.interactive_busy=False

    def snapshot(self,identifier):
        with self.guard:
            if not self.job or self.job['id']!=identifier:raise ValueError('结果已被新任务替换，请重新处理')
            return self.job.copy()

    def replay_check(self,body):
        """Validate an imported replay record, including its processed PNG pixels.

        The record must carry the full canonical parameter set, the live model's
        fixed variant, int8 backend and 64-hex label hashes; anything else is
        refused before the image slot is touched.
        """
        record=body.get('replay')
        if not isinstance(record,dict):raise ValueError('回放记录无效')
        if record.get('schema')!=REPLAY_SCHEMA:raise ValueError('回放记录版本不支持')
        c=self.controller
        model=record.get('model')
        if model!=c.model:raise ValueError('回放记录的模型与当前模型不同，拒绝重建')
        if record.get('spec_sha256')!=SPECS.get(model):
            raise ValueError('回放记录的模型规格不匹配')
        if record.get('coordinate_space')!='processed_image_pixels':
            raise ValueError('回放记录坐标空间不支持')
        image=record.get('image') or {}
        if not isinstance(image.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',image['sha256']):
            raise ValueError('回放记录缺少图像像素哈希')
        width,height=image.get('width'),image.get('height')
        if type(width) is not int or type(height) is not int or min(width,height)<2 or width*height>MAX_PIXELS:
            raise ValueError('回放记录处理图尺寸无效')
        parameters=record.get('parameters')
        allowed={'variant','backend','target_class','threshold','roi','box','points'}
        if not isinstance(parameters,dict) or set(parameters)!=allowed:
            raise ValueError('回放参数必须完整包含 variant/backend/target_class/threshold/roi/box/points')
        if parameters.get('backend')!='int8':
            raise ValueError('回放参数 backend 必须为 int8')
        expected_variant='delivery' if model=='skin' else 'candidate'
        if parameters.get('variant')!=expected_variant:
            raise ValueError(f'回放参数 variant 必须为当前模型的 {expected_variant}')
        req=Request.parse(model,parameters,width,height)
        canonical={'variant':req.variant,'backend':req.backend,'target_class':req.target_class,
            'threshold':req.threshold,'roi':None if req.roi is None else req.roi.as_dict(),
            'box':None if req.box is None else req.box.as_dict(),
            'points':[{'x':p.x,'y':p.y,'label':'foreground' if p.foreground else 'background'} for p in req.points]}
        results=record.get('results')
        if not isinstance(results,dict):raise ValueError('回放记录缺少结果哈希')
        for key in ('baseline_labels_sha256','final_labels_sha256'):
            value=results.get(key)
            if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
                raise ValueError(f'回放记录 {key} 必须是64位小写十六进制')
        original=record.get('original')
        if not isinstance(original,dict):
            raise ValueError('回放记录缺少原图尺寸信息')
        ow,oh=original.get('width'),original.get('height')
        if type(ow) is not int or type(oh) is not int or ow<=0 or oh<=0:
            raise ValueError('回放记录原图尺寸无效')
        transform=original.get('transform')
        if not isinstance(transform,dict) or not isinstance(transform.get('mode'),str) or not transform['mode']:
            raise ValueError('回放记录缺少合法变换说明')
        pw,ph=transform.get('processed_width'),transform.get('processed_height')
        if type(pw) is not int or type(ph) is not int or (pw,ph)!=(width,height):
            raise ValueError('回放记录变换尺寸与处理图不一致')
        pixels=decode_png_rgb(record.get('image_png'))
        if pixels.shape[:2]!=(height,width):
            raise ValueError(f'回放处理图尺寸 {pixels.shape[1]}x{pixels.shape[0]} 与记录 {width}x{height} 不一致')
        actual=hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()
        if actual!=image['sha256']:
            raise ValueError('回放处理图像素哈希不一致，已拒绝导入')
        return {'model':model,'spec_sha256':record['spec_sha256'],
                'generation_expected':record.get('generation'),
                'image':{'sha256':image['sha256'],'width':width,'height':height,
                         'png_sha256':png_sha256(record['image_png'])},
                'original':{'width':ow,'height':oh,'transform':dict(transform)},
                'parameters':canonical,
                'expected':{'baseline_labels_sha256':results['baseline_labels_sha256'],
                            'final_labels_sha256':results['final_labels_sha256']},
                'replay_ready':True}

    def clear(self):
        with self.guard:self.job=None;self.image_slot=None
        self.predictor.clear();self.service.clear()
        self.display_cache_key=None;self.display_cache=None


def handle(handler,path,post,body):
    root=Path(__file__).resolve().parent
    if path in ('/interactive','/interactive.js') and not post:
        file='interactive.html' if path=='/interactive' else 'interactive.js'
        return handler.send(200,(root/file).read_bytes(),'text/html; charset=utf-8' if file.endswith('html') else 'application/javascript')
    manager=handler.server.interactive
    if path=='/interactive/images' and post:return handler.send(200,manager.upload(json.loads(body)))
    if path=='/interactive/replay' and post:return handler.send(200,manager.replay_check(json.loads(body)))
    if path=='/interactive/jobs' and post:return handler.send(202,manager.submit(json.loads(body)))
    match=re.fullmatch(r'/interactive/jobs/([0-9a-f]{32})',path)
    if match and not post:return handler.send(200,manager.snapshot(match[1]))
    return handler.send(404,{'error':'交互接口不存在'})
