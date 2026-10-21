"""Source-bound native kernels. Importing this module needs no Torch, SciPy or PYNQ."""
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

from acdc.contract import ContractError, file_sha


def load_native():
    source=Path(__file__).with_name('native')/'image.c'
    compiler=os.environ.get('ACDC_IMAGE_CC',os.environ.get('CC','gcc'))
    version=subprocess.run([compiler,'--version'],capture_output=True,text=True,check=True,timeout=10).stdout
    flags=['-O3','-std=c11','-shared','-fno-fast-math','-ffp-contract=off','-fno-strict-aliasing']
    if os.name!='nt':flags+=['-fPIC']
    if platform.machine().startswith('arm'):
        flags+=['-mcpu=cortex-a9','-mfpu=neon','-mfloat-abi=hard']
    identity=dict(source_sha256=file_sha(source),wrapper_sha256=file_sha(Path(__file__)),
                  compiler=compiler,compiler_version=version,flags=flags,machine=platform.machine())
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    folder=Path(tempfile.gettempdir())/'pld-acdc-image'/key
    folder.mkdir(parents=True,exist_ok=True)
    library=folder/('image.dll' if os.name=='nt' else 'image.so')
    manifest=folder/'build.json'
    if not library.exists() or not manifest.exists():
        pending=folder/(str(os.getpid())+library.suffix)
        subprocess.run([compiler,*flags,str(source),'-o',str(pending),'-lm'],
                       capture_output=True,text=True,check=True,timeout=120)
        pending.replace(library)
        report=dict(identity=identity,library_sha256=file_sha(library))
        temporary=folder/(str(os.getpid())+'.json')
        temporary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        temporary.replace(manifest)
    report=json.loads(manifest.read_text('utf-8'))
    if report['identity']!=identity or report['library_sha256']!=file_sha(library):
        raise ContractError('native image library drift')
    return NativeImage(library,report)


class NativeImage:
    def __init__(self,path,report):
        self.library=ct.CDLL(str(path));self.info=report
        if self.library.acdc_image_abi()!=1:raise ContractError('native image ABI mismatch')
        pointer=ct.c_void_p
        signatures=dict(acdc_normalize=[pointer,pointer,ct.c_size_t,pointer],
                        acdc_resize=[pointer,pointer,*([ct.c_int]*5)],
                        acdc_quantize=[pointer,pointer,ct.c_size_t],
                        acdc_softmax=[pointer,pointer,ct.c_size_t,pointer],
                        acdc_mpr=[pointer,pointer,pointer,pointer,pointer,pointer,pointer,ct.c_int,ct.c_int,
                                  ct.c_double,ct.c_double,ct.c_double,ct.c_int,ct.c_int],
                        acdc_affine=[pointer,pointer,pointer,pointer,pointer,pointer,ct.c_float])
        for name,types in signatures.items():
            fn=getattr(self.library,name);fn.argtypes=types;fn.restype=ct.c_int

    @staticmethod
    def check(code):
        if code:raise ContractError(f'native image kernel rejected input: {code}')

    @staticmethod
    def floats(value,ndim=None):
        a=np.asarray(value,dtype=np.float32,order='C')
        if a.size==0 or (ndim is not None and a.ndim!=ndim) or not np.all(np.isfinite(a)):
            raise ContractError('nonempty finite float32 image required')
        return a

    def normalize(self,value):
        a=self.floats(value,3);b=np.empty_like(a);limits=np.empty(2,np.float64)
        self.check(self.library.acdc_normalize(a.ctypes.data,b.ctypes.data,a.size,limits.ctypes.data))
        return b,dict(low=float(limits[0]),high=float(limits[1]),method='nonzero-p1-p99-linear-native-phase')

    def resize(self,value,shape):
        a=self.floats(value)
        if a.ndim not in (2,3):raise ContractError('resize requires HW or CHW')
        h,w=map(int,shape)
        if min(h,w)<1 or max(h,w)>16384:raise ContractError('invalid resize shape')
        c=1 if a.ndim==2 else a.shape[0]
        if c not in (1,4):raise ContractError('qualified resize supports one gray or four probability channels')
        b=np.empty((c,h,w),np.float32)
        self.check(self.library.acdc_resize(a.ctypes.data,b.ctypes.data,c,*a.shape[-2:],h,w))
        return b[0] if a.ndim==2 else b

    def quantize(self,value):
        a=self.floats(value);b=np.empty(a.shape,np.int8)
        self.check(self.library.acdc_quantize(a.ctypes.data,b.ctypes.data,a.size))
        return b

    def softmax(self,head,table):
        from acdc.contract import checked_int8,HEAD_SHAPE
        a=checked_int8(head,HEAD_SHAPE)
        t=self.floats(table,2)
        if t.shape!=(255,255) or np.any(t<0) or not np.all(np.diag(t)==1):
            raise ContractError('invalid exp lookup table')
        b=np.empty((4,256,256),np.float32)
        self.check(self.library.acdc_softmax(a.ctypes.data,b.ctypes.data,256*256,t.ctypes.data))
        return b

    def affine(self,value,shape,matrix,offset,fill=0.):
        return self.prepare_affine(value).apply(shape,matrix,offset,fill)

    def prepare_affine(self,value):
        """Validate a finished source once, then reuse it across bounded blocks.

        The owner must finish writing the source before making this sampler and
        must not mutate it until all blocks have been consumed.
        """
        return AffineSource(self,self.floats(value,3))

    def mpr(self,gray,labels,origin,dx,dy,shape,*,level,window,alpha=.5,visible=14,outline=False):
        a=self.floats(gray,3)
        labels=np.ascontiguousarray(labels)
        if labels.dtype!=np.uint8 or labels.shape!=a.shape or np.any(labels>3):
            raise ContractError('invalid MPR label volume')
        h,w=map(int,shape)
        if min(h,w)<1 or max(h,w)>768:raise ContractError('invalid MPR raster size')
        vectors=[np.ascontiguousarray(v,np.float64) for v in (origin,dx,dy)]
        if any(v.shape!=(3,) or not np.all(np.isfinite(v)) for v in vectors):
            raise ContractError('invalid MPR coordinate mapping')
        if not all(np.isfinite(v) for v in (level,window,alpha)) or window<=0 or not 0<=alpha<=1 or not 0<=visible<=14:
            raise ContractError('invalid MPR display parameters')
        result=np.empty((h,w,4),np.uint8);size=np.array(a.shape,np.int64)
        self.check(self.library.acdc_mpr(a.ctypes.data,labels.ctypes.data,result.ctypes.data,size.ctypes.data,
            *(v.ctypes.data for v in vectors),h,w,float(level),float(window),float(alpha),int(visible),int(outline)))
        return result


class AffineSource:
    def __init__(self,kernel,array):
        self.kernel=kernel;self.array=array.view();self.array.flags.writeable=False
        self.ins=np.array(array.shape,np.int64)

    def apply(self,shape,matrix,offset,fill=0.):
        a=self.array
        shape=tuple(int(n) for n in shape)
        if len(shape)!=3 or min(shape)<1 or np.prod(shape,dtype=np.int64)>4_000_000:
            raise ContractError('native affine output exceeds working limit')
        m=np.ascontiguousarray(matrix,dtype=np.float64);o=np.ascontiguousarray(offset,dtype=np.float64)
        if m.shape!=(3,3) or o.shape!=(3,) or not np.all(np.isfinite(m)) or not np.all(np.isfinite(o)):
            raise ContractError('invalid affine mapping')
        if not np.isfinite(fill):raise ContractError('nonfinite fill')
        b=np.empty(shape,np.float32);outs=np.array(shape,np.int64)
        self.kernel.check(self.kernel.library.acdc_affine(a.ctypes.data,b.ctypes.data,self.ins.ctypes.data,outs.ctypes.data,
                                                         m.ctypes.data,o.ctypes.data,float(fill)))
        return b
