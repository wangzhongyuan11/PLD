"""Capture the board's actual libc/C++ headers and link libraries for cross builds."""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path);args=parser.parse_args()
    if args.output.exists():raise ValueError('sysroot snapshot must be a new file')
    paths={Path('/usr/include'),Path('/lib/ld-linux-armhf.so.3')}
    patterns=('libc.so*','libc_nonshared.a','libpthread*','libdl*','librt*','libm.so*','libm-*.so',
              'libresolv*','libgcc_s*','libstdc++*','libatomic*','crt*.o','Scrt1.o','ld-linux*')
    for base in ('/usr/lib/arm-linux-gnueabihf','/lib/arm-linux-gnueabihf'):
        for pattern in patterns:paths.update(Path(p) for p in glob.glob(base+'/'+pattern) if Path(p).is_file())
    for pattern in ('crt*.o','libgcc*.a','libstdc++.so'):
        paths.update(Path(p) for p in glob.glob('/usr/lib/gcc/arm-linux-gnueabihf/11/'+pattern))
    with tarfile.open(args.output,'w:gz',compresslevel=1,dereference=True) as archive:
        for path in sorted(paths):archive.add(path,arcname=path.as_posix().lstrip('/'))
    sha=hashlib.sha256()
    with args.output.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block)
    print(json.dumps(dict(path=str(args.output),bytes=args.output.stat().st_size,sha256=sha.hexdigest())),flush=True)


if __name__=='__main__':main()
