"""Source-bound, checked NCHW/C8 boundary copies, with ARM NEON acceleration."""
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


def load_native_layout():
    source=Path(__file__).parent/'native/layout.c'
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source,Path(__file__))}
    flags=['-O3','-std=c11','-shared','-fno-strict-aliasing']
    if os.name!='nt':flags+=['-fPIC']
    if platform.machine().startswith('arm'):
        flags+=['-mcpu=cortex-a9','-mfpu=neon','-mfloat-abi=hard']
    compiler=os.environ.get('CC','gcc')
    key=hashlib.sha256(json.dumps([hashes,flags,platform.machine()],sort_keys=True).encode()).hexdigest()
    folder=Path(tempfile.gettempdir())/'pld-acdc-layout'/key
    folder.mkdir(parents=True,exist_ok=True)
    library=folder/('layout.dll' if os.name=='nt' else 'layout.so')
    manifest=folder/'build.json'
    if not library.exists() or not manifest.exists():
        pending=library.with_name('pending-'+str(os.getpid())+library.suffix)
        subprocess.run([compiler,*flags,str(source),'-o',str(pending)],
                       check=True,capture_output=True,text=True,timeout=90)
        pending.replace(library)
        record=dict(source_hashes=hashes,flags=flags,machine=platform.machine(),
                    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
        manifest.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    record=json.loads(manifest.read_text('utf-8'))
    if (record['source_hashes']!=hashes or record['flags']!=flags or
            record['library_sha256']!=hashlib.sha256(library.read_bytes()).hexdigest()):
        raise ValueError('ACDC native layout build drift')
    return NativeLayout(library,record)


class NativeLayout:
    def __init__(self,path,record):
        self.library=ct.CDLL(str(path));self.info=record
        self.library.acdc_layout_abi.restype=ct.c_int
        if self.library.acdc_layout_abi()!=1:raise ValueError('native layout ABI mismatch')
        for name in ('acdc_pack_input','acdc_unpack_head'):
            fn=getattr(self.library,name)
            fn.argtypes=[ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t]
            fn.restype=ct.c_int

    @staticmethod
    def _array(a,dtype,shape,writable=False):
        if (not isinstance(a,np.ndarray) or a.dtype!=dtype or a.shape!=shape or
                not a.flags.c_contiguous or (writable and not a.flags.writeable) or
                a.ctypes.data%2):
            raise ValueError('invalid contiguous native boundary tensor')

    def pack(self,destination,source):
        self._array(destination,np.uint8,(524288,),True)
        self._array(source,np.int8,(1,3,256,256))
        if np.may_share_memory(destination,source):raise ValueError('overlapping layout arrays')
        if self.library.acdc_pack_input(destination.ctypes.data,destination.nbytes,
                                       source.ctypes.data,source.nbytes):
            raise ValueError('invalid native input, including forbidden -128')

    def head(self,source):
        self._array(source,np.uint8,(524288,))
        # Always return an independent allocation, never a CMA/staging view.
        result=np.empty((1,4,256,256),dtype=np.int8)
        if self.library.acdc_unpack_head(result.ctypes.data,result.nbytes,
                                        source.ctypes.data,source.nbytes):
            raise ValueError('invalid native head or nonzero padding lanes')
        return result
