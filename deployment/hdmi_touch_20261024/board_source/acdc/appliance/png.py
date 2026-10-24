"""Lossless RGBA PNG using the existing PLD fixed-Sub/RLE encoder strategy.

Adapted from display_web/server.py:save_lossless_png for an in-memory RGBA
response. Avoid importing that application or changing its deployed source.
"""
import struct
import zlib

import numpy as np


def rgba_png(pixels):
    a=np.asarray(pixels)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=4 or min(a.shape[:2])<1:
        raise ValueError('PNG requires nonempty uint8 RGBA')
    h,w=a.shape[:2];raw=np.ascontiguousarray(a).reshape(h,w*4)
    rows=np.empty((h,w*4+1),np.uint8);rows[:,0]=1
    rows[:,1:5]=raw[:,:4]
    np.subtract(raw[:,4:],raw[:,:-4],out=rows[:,5:])
    def chunk(kind,data):
        crc=zlib.crc32(data,zlib.crc32(kind))&0xffffffff
        return struct.pack('!I',len(data))+kind+data+struct.pack('!I',crc)
    compressor=zlib.compressobj(level=1,strategy=zlib.Z_RLE)
    compressed=compressor.compress(rows.tobytes())+compressor.flush()
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!IIBBBBB',w,h,8,6,0,0,0))
            +chunk(b'IDAT',compressed)+chunk(b'IEND',b''))
