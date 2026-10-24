"""Streaming ZIP32 exports with bounded memory and an inline transport digest.

Measured Cortex-A9 DEFLATE cost dominates this already binary result set.
ZIP_STORED retains every file and its CRC while trading a larger download for
shorter result readiness. It needs one 256 KiB copy buffer, not whole files.
"""
from pathlib import Path
import hashlib
import zipfile


class _DigestWriter:
    def __init__(self,stream):
        self.stream=stream;self.digest=hashlib.sha256()
    def write(self,data):
        written=self.stream.write(data)
        if written!=len(data):raise OSError('short ZIP write')
        self.digest.update(data)
        return written
    def tell(self):return self.stream.tell()
    def flush(self):return self.stream.flush()
    # No seek: ZipFile emits standard data descriptors, so headers are never
    # rewritten behind the streaming digest.


def write_zip(root,names,destination,check_cancel):
    root=Path(root)
    with Path(destination).open('xb') as stream:
        sink=_DigestWriter(stream)
        with zipfile.ZipFile(sink,'w',compression=zipfile.ZIP_STORED,allowZip64=False) as archive:
            for name in names:
                check_cancel();path=root/name
                size=path.stat().st_size
                if size>32*1024*1024:raise ValueError('ZIP member exceeds export limit')
                info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.file_size=size
                with path.open('rb') as source,archive.open(info,'w') as target:
                    for block in iter(lambda:source.read(256*1024),b''):
                        check_cancel();target.write(block)
        return sink.digest.hexdigest()
