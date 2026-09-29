"""Compile the bounded command candidate and run its independent register model."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('out/native/tensor_control.so'))
    args=parser.parse_args()
    if platform.system()!='Linux' or platform.machine() not in ('armv7l','armv8l'):
        parser.error('build on the 32-bit ARM PYNQ target')
    root=Path(__file__).resolve().parents[1]
    names=('board/native/tensor_control.c','board/native_control.py','scripts/verify_native_control.py')
    source=root/names[0]
    hashes={name:digest(root/name) for name in names}
    flags=['-O3','-std=c11','-Wall','-Wextra','-Werror','-shared','-fPIC','-mcpu=cortex-a9',
           '-mfpu=neon','-mfloat-abi=hard']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    test_library=args.output.with_name(args.output.stem+'_test.so')
    subprocess.run(['gcc',*flags,'-DPLD_CONTROL_TEST',str(source),'-o',str(test_library)],check=True)
    subprocess.run(['gcc',*flags,str(source),'-o',str(args.output)],check=True)
    subprocess.run([sys.executable,str(root/names[2]),str(test_library),
                    '--production-library',str(args.output)],check=True)
    library=ctypes.CDLL(str(args.output.resolve()))
    library.tensor_control_abi.restype=ctypes.c_uint32
    if library.tensor_control_abi()!=1:
        raise RuntimeError('production control ABI mismatch')
    if any(digest(root/name)!=h for name,h in hashes.items()):
        raise RuntimeError('control source changed during build')
    manifest={'source_hashes':hashes,'flags':flags,'machine':platform.machine(),
              'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
              'library_sha256':digest(args.output),'test_library_sha256':digest(test_library)}
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('NATIVE_CONTROL_BUILD_PASS '+str(args.output),flush=True)


if __name__=='__main__': main()
