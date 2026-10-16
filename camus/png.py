"""Lossless four-view output with bounded, joined PNG writers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
import time
import zlib
import numpy as np
from .result import label_display_png, segmentation_rgb, overlay_rgb


def save_png(image, path):
    # Same Sub + RLE encoder as the existing display_web implementation.
    a=np.asarray(image)
    if a.dtype!=np.uint8 or not (a.ndim==2 or a.ndim==3 and a.shape[2]==3):
        raise ValueError('PNG requires uint8 L or RGB')
    h,w=a.shape[:2];channels=1 if a.ndim==2 else 3
    raw=np.ascontiguousarray(a).reshape(h,w*channels)
    rows=np.empty((h,w*channels+1),np.uint8);rows[:,0]=1
    rows[:,1:1+channels]=raw[:,:channels]
    np.subtract(raw[:,channels:],raw[:,:-channels],out=rows[:,1+channels:])
    def chunk(kind,data):
        crc=zlib.crc32(data,zlib.crc32(kind)) & 0xffffffff
        return struct.pack('!I',len(data))+kind+data+struct.pack('!I',crc)
    compressor=zlib.compressobj(level=1,strategy=zlib.Z_RLE)
    compressed=compressor.compress(rows.tobytes())+compressor.flush()
    header=struct.pack('!IIBBBBB',w,h,8,0 if channels==1 else 2,0,0,0)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',header)+chunk(b'IDAT',compressed)+chunk(b'IEND',b''))


def save_views(output,gray,labels,processed,native=None):
    if native is None:
        jobs=[('label_raw.png',label_display_png,(labels,)),('label.png',label_display_png,(processed,)),
              ('segmentation.png',segmentation_rgb,(processed,)),('overlay.png',overlay_rgb,(gray,processed))]
    else:
        started=time.perf_counter()
        rendered=native.render_views(gray,labels,processed)
        elapsed=time.perf_counter()-started
        jobs=[(name,lambda image=image:image,()) for name,image in rendered.items()]
    def group(items):
        result={}
        for name,render,args in items:
            t=time.perf_counter();image=render(*args);render_time=time.perf_counter()-t
            t=time.perf_counter();save_png(image,output/name)
            result[name]=dict(render=render_time,encode_write=time.perf_counter()-t)
        return result
    with ThreadPoolExecutor(max_workers=2,thread_name_prefix='camus-png') as pool:
        futures=[pool.submit(group,jobs[index::2]) for index in range(2)]
        details={name:row for future in futures for name,row in future.result().items()}
        if native is not None:details['overlay.png']['render']+=elapsed
        return details
