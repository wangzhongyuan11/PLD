"""Bounded two-core DEFLATE ZIP writer, independently readable by zipfile.

The appliance exports well below ZIP32 limits. Each worker owns one immutable
file; the main thread writes members in the requested order. No shared ZipFile
mutation or extra compressed temporary files are involved.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
import zlib


def write_zip(root, names, destination, check_cancel):
    root=Path(root);names=iter(names);central=[]
    def compress(name):
        raw_name=name.encode('utf-8')
        if len(raw_name)>65535:raise ValueError('ZIP member name too long')
        path=root/name
        if path.stat().st_size>32*1024*1024:raise ValueError('ZIP member exceeds bounded compression memory')
        # Geometry floats/indices contain many short byte runs. RLE avoids the
        # expensive LZ match search on A9 while retaining standard ZIP DEFLATE.
        compressor=zlib.compressobj(1,zlib.DEFLATED,-15,8,zlib.Z_RLE);parts=[];crc=size=0
        with path.open('rb') as source:
            for block in iter(lambda:source.read(256*1024),b''):
                check_cancel();crc=zlib.crc32(block,crc);size+=len(block);parts.append(compressor.compress(block))
        parts.append(compressor.flush())
        return raw_name,crc&0xffffffff,size,b''.join(parts)
    with Path(destination).open('xb') as stream, ThreadPoolExecutor(max_workers=2,thread_name_prefix='v2-zip') as pool:
        pending=deque()
        for _ in range(2):
            name=next(names,None)
            if name is not None:pending.append(pool.submit(compress,name))
        while pending:
            check_cancel();name,crc,size,data=pending.popleft().result();offset=stream.tell()
            if offset+len(data)>=0xffffffff or len(central)>=65535:raise ValueError('ZIP32 capacity exceeded')
            # UTF-8 names, raw DEFLATE, fixed valid 1980-01-01 DOS timestamp.
            stream.write(struct.pack('<I5H3I2H',0x04034b50,20,0x800,8,0,33,crc,len(data),size,len(name),0))
            stream.write(name);stream.write(data)
            central.append(struct.pack('<I6H3I5H2I',0x02014b50,20,20,0x800,8,0,33,
                crc,len(data),size,len(name),0,0,0,0,0,offset)+name)
            following=next(names,None)
            if following is not None:pending.append(pool.submit(compress,following))
        start=stream.tell()
        for record in central:stream.write(record)
        length=stream.tell()-start
        stream.write(struct.pack('<I4H2IH',0x06054b50,0,0,len(central),len(central),length,start,0))
