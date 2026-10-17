"""Native-coordinate ROI, thresholding and model-assisted prompt segmentation."""
import base64
import hashlib
import struct
import time
import uuid
import zlib
import numpy as np
from scipy import ndimage
from .contracts import Request,Rect,InputError
from .refinement import seeded_cut



def png_data(array):
    # Same lossless Sub + RLE strategy already used by camus.png. Avoid
    # Pillow's adaptive filter search on the Cortex-A9; pixels are unchanged.
    a=np.asarray(array,dtype=np.uint8)
    if not (a.ndim==2 or a.ndim==3 and a.shape[2]==3):
        raise ValueError('PNG requires gray or RGB pixels')
    h,w=a.shape[:2];channels=1 if a.ndim==2 else 3
    raw=np.ascontiguousarray(a).reshape(h,w*channels)
    rows=np.empty((h,w*channels+1),np.uint8);rows[:,0]=1
    rows[:,1:1+channels]=raw[:,:channels]
    np.subtract(raw[:,channels:],raw[:,:-channels],out=rows[:,1+channels:])
    def chunk(kind,data):
        return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(data,zlib.crc32(kind))&0xffffffff)
    compressor=zlib.compressobj(level=1,strategy=zlib.Z_RLE)
    compressed=compressor.compress(rows.tobytes())+compressor.flush()
    header=struct.pack('!IIBBBBB',w,h,8,0 if channels==1 else 2,0,0,0)
    out=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',header)+chunk(b'IDAT',compressed)+chunk(b'IEND',b'')
    return 'data:image/png;base64,'+base64.b64encode(out).decode('ascii')


class SegmentationService:
    def __init__(self,predictor=None):
        if predictor is None: raise ValueError("PL predictor required")
        self.predictor = predictor

    def process(self,model,image,parameters):
        began = time.monotonic()
        image = np.asarray(image)
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise InputError('输入图像必须为 uint8 RGB')
        h,w = image.shape[:2]
        req = Request.parse(model,parameters,w,h)
        roi = req.roi or Rect(0,0,w,h)
        crop = image[roi.y0:roi.y1,roi.x0:roi.x1]
        prepared = time.monotonic()
        probabilities,identity = self.predictor.predict(crop,model,req.variant,req.backend)
        predicted = time.monotonic()
        nclasses = 2 if model=='skin' else 4
        if probabilities.shape!=(nclasses,*crop.shape[:2]):
            raise RuntimeError('模型输出尺寸或类别数不匹配')
        target = probabilities[req.target_class]
        other = probabilities.copy(); other[req.target_class] = -1
        alternatives = other.argmax(axis=0).astype(np.uint8)
        if req.threshold is None:
            labels = probabilities.argmax(axis=0).astype(np.uint8)
            # Pairwise target-vs-strongest-other odds match multiclass argmax.
            prior = target/np.maximum(target+other.max(axis=0),1e-8)
        else:
            labels = np.where(target>=req.threshold,req.target_class,alternatives).astype(np.uint8)
            epsilon = 1e-6
            p = np.clip(target,epsilon,1-epsilon)
            odds = np.log(p/(1-p))-np.log(req.threshold/(1-req.threshold))
            prior = 1/(1+np.exp(-np.clip(odds,-30,30)))
        if model=='camus':
            from camus.postprocess import apply_island5_hole1
            labels = apply_island5_hole1(labels).labels.astype(np.uint8)
        baseline = labels.copy()
        decided = time.monotonic()
        refinement = None
        if req.interactive:
            points = [(p.x-roi.x0,p.y-roi.y0,p.foreground) for p in req.points]
            box = None if req.box is None else (req.box.x0-roi.x0,req.box.y0-roi.y0,req.box.x1-roi.x0,req.box.y1-roi.y0)
            mask,refinement = seeded_cut(crop,prior,points,box)
            # Other classes retain their original decision unless the chosen
            # target is added/removed; removed pixels select a non-target class.
            labels[(labels==req.target_class)&~mask] = alternatives[(labels==req.target_class)&~mask]
            labels[mask] = req.target_class
        refined = time.monotonic()
        native = np.zeros((h,w),np.uint8)
        native[roi.y0:roi.y1,roi.x0:roi.x1] = labels
        selected = native==req.target_class
        boundary = selected & ~ndimage.binary_erosion(selected)
        overlay = image.copy().astype(float)
        colors = ((0,0,0),(25,174,114),(232,150,43),(98,107,213))
        # Display only the selected structure; labels.png still contains all
        # classes. The drawn rectangle itself is never part of this overlay.
        color = np.asarray(colors[req.target_class],float)
        overlay[selected] = .6*overlay[selected]+.4*color
        outline = ndimage.binary_dilation(boundary,iterations=1)
        overlay[outline] = color
        warnings = []
        if not selected.any():
            warnings.append('当前条件下未提取到目标，可调整提示或阈值')
        if req.roi:
            warnings.append('仅处理 ROI；区域外未推理并输出背景，面积占比以 ROI 为分母')
        if req.interactive:
            warnings.append('提示由模型辅助图割处理；原网络未经过提示式训练，交互结果的精度需另行验证')
        pixels = int(selected.sum()); domain_pixels = crop.shape[0]*crop.shape[1]
        statistics = {'target_pixels':pixels,'processed_pixels':domain_pixels,'area_ratio':pixels/domain_pixels,
            'per_class_pixels':{str(i):int((labels==i).sum()) for i in range(nclasses)},
            'changed_pixels_by_refinement':int((labels!=baseline).sum())}
        rendered = time.monotonic()
        outputs = {'labels_png':png_data(native),'target_mask_png':png_data(selected.astype(np.uint8)*255),
            'overlay_png':png_data(np.rint(overlay).clip(0,255).astype(np.uint8))}
        encoded = time.monotonic()
        return {'request_id':uuid.uuid4().hex,'model':model,'identity':identity,
            'image':{'width':w,'height':h,'sha256':hashlib.sha256(image.tobytes()).hexdigest()},
            'parameters':{'variant':req.variant,'backend':req.backend,'target_class':req.target_class,
                'threshold':req.threshold,'roi':None if req.roi is None else req.roi.as_dict(),
                'box':None if req.box is None else req.box.as_dict(),
                'points':[{'x':p.x,'y':p.y,'label':'foreground' if p.foreground else 'background'} for p in req.points]},
            'operation':'prompt_refinement' if req.interactive else ('roi_model_inference' if req.roi else 'model_inference'),
            'refinement':refinement,'network_prompt_conditioned':False,
            'postprocess':'island5_hole1_before_refinement' if model=='camus' else 'none',
            'outputs':outputs,'statistics':statistics,
            'timings_ms':dict(input_validation=(prepared-began)*1000,prediction=(predicted-prepared)*1000,
                decision_postprocess=(decided-predicted)*1000,prompt_refinement=(refined-decided)*1000,
                render_statistics=(rendered-refined)*1000,result_png=(encoded-rendered)*1000),
            'reference_available':False,'accuracy_metrics':None,'warnings':warnings,
            'elapsed_ms':round((time.monotonic()-began)*1000,2)}
