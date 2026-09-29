"""Build the pinned ARM/Pillow row resizer and run pixel/bounds tests."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'board'))
from native_resize import SOURCES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('out/native/rgb_resize.so'))
    args=parser.parse_args()
    if platform.system()!='Linux' or platform.machine() not in ('armv7l','armv8l') or Image.__version__!='9.0.1':
        parser.error('requires the verified ARM PYNQ / Pillow 9.0.1 runtime')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    flags=['-O3','-std=c11','-Wall','-Wextra','-Werror','-shared','-fPIC',
           '-mcpu=cortex-a9','-mfpu=neon','-mfloat-abi=hard']
    manifest={'source_hashes':{name:digest(ROOT/name) for name in SOURCES},'flags':flags,
              'pillow':Image.__version__,'pillow_core_sha256':digest(Path(Image.core.__file__)),
              'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
              'machine':platform.machine()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['gcc',*flags,str(ROOT/'board/native/rgb_resize.c'),'-lm','-o',str(args.output)],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/check_native_resize.py'),str(args.output),'--pillow-rows'],check=True)
    if any(digest(ROOT/name)!=value for name,value in manifest['source_hashes'].items()):
        raise RuntimeError('native resize source changed during build')
    manifest['library_sha256']=digest(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('NATIVE_RESIZE_BUILD_PASS '+str(args.output),flush=True)


if __name__=='__main__': main()
