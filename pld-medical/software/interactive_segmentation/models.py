"""Read-only adapters for the accepted frozen segmentation models."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import threading
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .contracts import InputError

PROJECT = Path(__file__).resolve().parents[2]
torch.set_num_threads(2)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class ModelInfo:
    name: str
    weights: str
    spec: str
    weights_sha256: str
    spec_sha256: str


REGISTRY = {
    ('skin','delivery'): ModelInfo('皮肤病灶原交付',
        'deliverables/role_a_final/best.pt','deliverables/role_a_final/int8/model_spec.json',
        'f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77',
        'fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef'),
    ('camus','delivery'): ModelInfo('CAMUS 原交付',
        'deliverables/role_a_camus_opt_20261013/weights/model_state_fp32.pt',
        'deliverables/role_a_camus_opt_20261013/int8_export/model_spec.json',
        '74d0da6f13b48528b221e6c295d90de73b6e16f1320a860bc086a05e3c2b774f',
        '26e200660f91d228afe7127e046f4d69d52000e4a240c9fc8f7068baf074cd1a'),
    ('camus','candidate'): ModelInfo('CAMUS 隐藏层直方图候选',
        'runs/role_a_camus/20261013_auto03/selection/model_state_fp32.pt',
        'runs/role_a_camus/20261014_auto10/exports/hidden_histogram/model_spec.json',
        '3d8f268910783ea524513b3506f4425aec991a54dcb23bf25cff5598fb942378',
        '68b4f425cc3612ad72d5e83955c2018981edee912fbc0937e9ac8c1718e7b5ae'),
}


def capabilities(model):
    if model not in ('skin','camus'):
        raise InputError('未知模型')
    classes = {'1':'皮肤病灶'} if model=='skin' else {'1':'左室腔','2':'心肌','3':'左房'}
    return {'model':model,'variants':{v:info.name for (m,v),info in REGISTRY.items() if m==model},
        'classes':classes,'backends':['int8','fp32'],
        'default_threshold':.45 if model=='skin' else None,
        'threshold_semantics':'前景概率阈值' if model=='skin' else 'null保留多类别argmax；数值为所选结构的概率阈值',
        'coordinate_space':'original_image_pixels','rect_convention':'x0,y0 inclusive; x1,y1 exclusive',
        'roi':True,'point_prompts':True,'box_prompt':True,'threshold_adjustment':True,
        'prompt_algorithm':'seeded_min_cut','network_prompt_conditioned':False,
        'software_only':True,'board_verified':False}


class Predictor:
    def __init__(self):
        self._models = {}
        self._lock = threading.RLock()
        self._cache_key = None
        self._cache = None

    def _load(self, model, variant, backend):
        key = model,variant,backend
        if key in self._models:
            return self._models[key]
        info = REGISTRY[model,variant]
        weights,spec_path = PROJECT/info.weights,PROJECT/info.spec
        if sha256(weights)!=info.weights_sha256 or sha256(spec_path)!=info.spec_sha256:
            raise RuntimeError('冻结模型文件校验失败，请恢复已验收交付文件')
        if model == 'skin':
            from software.role_a import quant
            loaded = quant._load_fp32(weights)[0] if backend=='fp32' else quant._load_quantized(spec_path)[0]
            result = ('torch',loaded)
        elif backend == 'fp32':
            from software.role_a_camus.model import CamusUNet
            checkpoint = torch.load(weights,map_location='cpu',weights_only=True)
            config = checkpoint['config']
            loaded = CamusUNet(base_channels=config['base_channels'],num_pools=config['num_pools'])
            loaded.load_state_dict(checkpoint['model_state'],strict=True)
            result = ('torch',loaded.eval())
        else:
            from software.role_a_camus_int8 import pipeline,graph
            spec = json.loads(spec_path.read_text(encoding='utf-8'))
            if spec['graph']['layer_names'] != list(graph.LAYER_NAMES):
                raise RuntimeError('CAMUS 拓扑不匹配')
            for name in ('weights_int8','bias_int32'):
                info = spec['packed'][name]
                member = pipeline._safe_member_path(spec_path.parent,info['file'])
                if member.stat().st_size!=int(info['bytes']) or sha256(member)!=info['sha256']:
                    raise RuntimeError('CAMUS packed 参数校验失败')
            layers = pipeline._read_layers(spec_path.parent,spec)
            result = ('camus_int8',layers)
        self._models[key] = result
        return result

    def predict(self, image, model, variant='delivery', backend='int8'):
        """Native H×W class probabilities; cache excludes prompts/thresholds."""
        image = np.asarray(image,dtype=np.uint8)
        key = hashlib.sha256(image.tobytes()).hexdigest(),image.shape,model,variant,backend
        with self._lock:
            if key==self._cache_key:
                return self._cache.copy(), self.identity(model,variant,backend,True)
            kind,loaded = self._load(model,variant,backend)
            h,w = image.shape[:2]
            if model=='skin':
                rgb = Image.fromarray(image).convert('RGB')
                scale = min(256/w,256/h)
                nw,nh = max(1,round(w*scale)),max(1,round(h*scale))
                left,top = (256-nw)//2,(256-nh)//2
                canvas = Image.new('RGB',(256,256))
                canvas.paste(rgb.resize((nw,nh),Image.Resampling.BILINEAR),(left,top))
                tensor = torch.from_numpy(np.asarray(canvas,dtype=np.float32).transpose(2,0,1).copy()/255.)[None]
                with torch.inference_mode():
                    probability = torch.sigmoid(loaded(tensor).float())
                    probability = F.interpolate(probability[...,top:top+nh,left:left+nw],size=(h,w),mode='nearest')[0,0].numpy()
                probabilities = np.stack((1-probability,probability))
            else:
                from software.role_a_camus import geometry
                gray = np.asarray(Image.fromarray(image).convert('L'),dtype=np.float32)
                canvas,valid,meta = geometry.letterbox(geometry.normalize_gray(gray))
                with torch.inference_mode():
                    if kind=='torch':
                        logits = loaded(torch.from_numpy(canvas)[None,None]).float()
                    else:
                        from software.role_a_camus_int8.graph import forward_graph
                        result = forward_graph(None,loaded,canvas,valid=valid)
                        logits = torch.from_numpy(np.asarray(result['logits'],dtype=np.float32))[None]
                    cropped = logits[...,meta['top']:meta['top']+meta['new_height'],meta['left']:meta['left']+meta['new_width']]
                    native = F.interpolate(cropped,size=(h,w),mode='bilinear',align_corners=False)
                    probabilities = torch.softmax(native,dim=1)[0].numpy()
            if not np.isfinite(probabilities).all():
                raise RuntimeError('模型输出包含无效数值')
            self._cache_key,self._cache = key,probabilities.copy()
            return probabilities,self.identity(model,variant,backend,False)

    @staticmethod
    def identity(model,variant,backend,cached=False):
        info = REGISTRY[model,variant]
        return {'model':model,'variant':variant,'name':info.name,'backend':backend,
            'weights_sha256':info.weights_sha256,'spec_sha256':info.spec_sha256,
            'prediction_cache_hit':cached,'software_only':True,
            'integer_simulation':backend=='int8','board_verified':False}
