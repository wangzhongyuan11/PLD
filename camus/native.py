"""Source-bound CAM2 control, checked copies and exact float32 reconstruction."""
from __future__ import annotations
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import numpy as np
from .native_control import NativeControl


def load_native():
    root=Path(__file__).resolve().parent
    sources=[root/'native/control.c',root/'native/ops.c',root/'native.py',root/'native_control.py']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    flags=['-O3','-std=c11','-shared','-fPIC','-fno-fast-math','-ffp-contract=off']
    link_flags=['-lm']
    if os.name=='nt':link_flags+=['-Wl,--export-all-symbols']
    if platform.machine().startswith('arm'):flags+=['-mcpu=cortex-a9','-mfpu=neon','-mfloat-abi=hard']
    key=hashlib.sha256(json.dumps([hashes,flags,link_flags,platform.machine()],sort_keys=True).encode()).hexdigest()
    folder=Path(tempfile.gettempdir())/'pld-camus-native'/key
    folder.mkdir(parents=True,exist_ok=True)
    library=folder/'camus.so';manifest=folder/'build.json'
    if not library.exists() or not manifest.exists():
        pending=folder/'camus.pending.so'
        command=['gcc',*flags,*(str(p) for p in sources[:2]),'-o',str(pending),*link_flags]
        subprocess.run(command,check=True,capture_output=True,text=True,timeout=90)
        pending.replace(library)
        manifest.write_text(json.dumps(dict(source_hashes=hashes,flags=flags,link_flags=link_flags,
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest()),indent=2)+'\n')
    record=json.loads(manifest.read_text())
    if (record['source_hashes']!=hashes or record['flags']!=flags or
            record.get('link_flags')!=link_flags or record['library_sha256']!=hashlib.sha256(library.read_bytes()).hexdigest()):
        raise ValueError('CAMUS native build drift')
    return NativeOps(library,record)


class NativeOps(NativeControl):
    def __init__(self,path,record):
        super().__init__(path)
        self.info.update(kind='camus_native',build=record)
        p=ct.c_void_p;i=ct.c_int
        self.copy_fn=self.library.camus_copy
        self.copy_fn.argtypes=[p,ct.c_size_t,p,p,p];self.copy_fn.restype=i
        self.gather_fn=self.library.camus_gather
        self.gather_fn.argtypes=[p,p,*([i]*7)];self.gather_fn.restype=i
        self.scatter_fn=self.library.camus_scatter
        self.scatter_fn.argtypes=[p,p,*([i]*7)];self.scatter_fn.restype=None
        self.scatter_checked_fn=self.library.camus_scatter_checked
        self.scatter_checked_fn.argtypes=[p,p,ct.c_size_t,*([i]*7)];self.scatter_checked_fn.restype=i
        self.restore_fn=self.library.camus_restore
        self.restore_fn.argtypes=[p,p,*([i]*5),ct.c_float,*([p]*6)];self.restore_fn.restype=i
        self.post_fn=self.library.camus_postprocess
        self.post_fn.argtypes=[p,p,i,i,p];self.post_fn.restype=i
        self.render_fn=self.library.camus_render_views
        self.render_fn.argtypes=[*([p]*8),ct.c_size_t];self.render_fn.restype=i
        self.preprocess_fn=self.library.camus_preprocess
        self.preprocess_fn.argtypes=[*([p]*4),*([i]*6),*([p]*7)];self.preprocess_fn.restype=i

    def copy(self,destination,source):
        if (source.dtype!=np.int8 or source.ndim!=4 or destination.dtype not in (np.uint8,np.int8)
                or not destination.flags.c_contiguous or not destination.flags.writeable
                or np.may_share_memory(destination,source) or source.nbytes>destination.nbytes):
            raise ValueError('invalid native copy arrays')
        shape=np.asarray(source.shape,np.int64);strides=np.asarray(source.strides,np.int64)
        if self.copy_fn(destination.ctypes.data,destination.nbytes,source.ctypes.data,shape.ctypes.data,strides.ctypes.data):
            raise ValueError('native copy rejected dimensions or -128 input')

    def gather(self,destination,source,tile,kernel,padding):
        _,c,h,w=source.shape;th=tile.height+kernel-1;tw=tile.width+kernel-1
        if source.dtype!=np.int8 or not source.flags.c_contiguous or c*th*tw>destination.nbytes:
            raise ValueError('invalid native gather arrays')
        if self.gather_fn(destination.ctypes.data,source.ctypes.data,c,h,w,tile.y-padding,tile.x-padding,th,tw):
            raise ValueError('input contains -128')
        return c*th*tw

    def scatter(self,destination,source,tile):
        _,c,h,w=destination.shape
        if not destination.flags.c_contiguous or source.nbytes<c*tile.height*tile.width:
            raise ValueError('invalid native scatter arrays')
        self.scatter_fn(destination.ctypes.data,source.ctypes.data,c,h,w,tile.y,tile.x,tile.height,tile.width)

    def scatter_checked(self,destination,source,tile,length):
        _,c,h,w=destination.shape
        if (destination.dtype!=np.int8 or not destination.flags.c_contiguous or
                source.dtype!=np.uint8 or not source.flags.c_contiguous or
                source.nbytes<length or length!=c*tile.height*tile.width):
            raise ValueError('invalid native checked scatter arrays')
        return self.scatter_checked_fn(destination.ctypes.data,source.ctypes.data,source.nbytes,
                                       c,h,w,tile.y,tile.x,tile.height,tile.width)

    def restore(self,head,height,width,scale):
        from .result import letterbox_meta_for_native, output_axis_plan
        meta=letterbox_meta_for_native(height,width)
        axes=(*output_axis_plan(meta['new_width'],width),*output_axis_plan(meta['new_height'],height))
        # C ABI uses int32 indices independently of the host's pointer width.
        axes=tuple(np.ascontiguousarray(a,dtype=np.float32 if n%3==2 else np.int32) for n,a in enumerate(axes))
        if head.dtype!=np.int8 or head.shape!=(1,4,256,256) or not head.flags.c_contiguous or not np.isfinite(scale) or scale<=0:
            raise ValueError('invalid native restore head/scale')
        labels=np.empty((height,width),np.uint8)
        if self.restore_fn(head.ctypes.data,labels.ctypes.data,height,width,meta['top'],meta['left'],meta['new_height'],scale,*(a.ctypes.data for a in axes)):
            raise MemoryError('native reconstruction allocation failed')
        return labels

    def postprocess(self,labels):
        from .postprocess import PostprocessOutcome
        if labels.dtype!=np.uint8 or labels.ndim!=2 or not labels.flags.c_contiguous or not 0<labels.size<=8_000_000:
            raise ValueError('native postprocess requires contiguous uint8 labels')
        output=np.empty_like(labels);stats=np.zeros(10,np.int32)
        if self.post_fn(labels.ctypes.data,output.ctypes.data,*labels.shape,stats.ctypes.data):
            raise ValueError('native postprocessing rejected labels or allocation')
        before,after,components,largest,secondary,island,filled,hole,changed,closed=(int(v) for v in stats)
        retained=before-(secondary if island else 0)
        return PostprocessOutcome('island5_hole1',output,before,after,components,largest,secondary,
            secondary/before if before else 0.,bool(island),filled,closed/retained if retained else 0.,bool(hole),changed)

    def render_views(self,gray,labels,processed):
        from .result import _overlay_lut
        if (labels.dtype!=np.uint8 or labels.ndim!=2 or not labels.flags.c_contiguous or
                processed.dtype!=np.uint8 or processed.shape!=labels.shape or
                not processed.flags.c_contiguous or gray.shape!=labels.shape):
            raise ValueError('invalid CAMUS view arrays')
        gray8=np.ascontiguousarray(np.clip(np.asarray(gray,dtype=np.float32),0.,255.).astype(np.uint8))
        h,w=labels.shape
        raw=np.empty((h,w),np.uint8);label=np.empty_like(raw)
        segmentation=np.empty((h,w,3),np.uint8);overlay=np.empty_like(segmentation)
        lut=_overlay_lut(.45)
        args=(labels,processed,gray8,lut,raw,label,segmentation,overlay)
        if self.render_fn(*(a.ctypes.data for a in args),labels.size):
            raise ValueError('CAMUS render labels outside 0..3')
        return {'label_raw.png':raw,'label.png':label,
                'segmentation.png':segmentation,'overlay.png':overlay}

    def preprocess(self,path):
        from .preprocess import LoadedGray, PreprocessResult, load_gray_frame, input_axis_plan
        loaded=load_gray_frame(path)
        raw=np.ascontiguousarray(np.asarray(loaded.data,dtype=np.float32))
        h,w=raw.shape
        scale=min(256/h,256/w)
        nh=max(1,min(256,int(round(h*scale))))
        nw=max(1,min(256,int(round(w*scale))))
        top,left=(256-nh)//2,(256-nw)//2
        meta=dict(original_height=h,original_width=w,scale=scale,new_height=nh,
                  new_width=nw,top=top,left=left,size=256)
        axes=(*input_axis_plan(h,nh),*input_axis_plan(w,nw))
        axes=tuple(np.ascontiguousarray(a,dtype=np.float32 if j%3==2 else np.int32)
                   for j,a in enumerate(axes))
        normalized=np.empty_like(raw)
        canvas=np.empty((256,256),np.float32)
        quantized=np.empty((256,256),np.int8)
        counts=np.empty(2,np.int32)
        args=(raw,normalized,canvas,quantized)
        code=self.preprocess_fn(*(a.ctypes.data for a in args),h,w,nh,nw,top,left,
                                *(a.ctypes.data for a in axes),counts.ctypes.data)
        if code:raise MemoryError(f'native CAMUS preprocessing failed code={code}')
        return PreprocessResult(loaded.data.copy(),normalized,canvas,quantized[None,None],
                                meta,int(counts[0]),int(counts[1]),loaded)


class TXPrefix:
    """Synchronize only valid TX cache lines; retain the owning root buffer."""
    def __init__(self,buffer,length,direction):
        if buffer.offset!=0 or buffer.coherent or buffer.physical_address%64 or not 0<length<=buffer.nbytes or buffer.nbytes%64:
            raise ValueError('invalid TX prefix allocation')
        self.buffer=buffer;self.physical_address=buffer.physical_address;self.nbytes=length
        self.direction=direction
    def flush(self):self.buffer.bo.sync(self.direction,(self.nbytes+63)//64*64,0)
    def invalidate(self):raise RuntimeError('TX prefix cannot receive')
