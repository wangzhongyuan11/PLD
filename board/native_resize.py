"""Manifest-checked, version-pinned Pillow row resize for the PYNQ runtime."""
import hashlib
import json
from pathlib import Path

from PIL import Image
from reference.pillow_resize import NativePillowBilinear

SOURCES=("board/native/rgb_resize.c","board/native_resize.py","reference/native_resize.py",
         "reference/pillow_resize.py","scripts/check_native_resize.py","scripts/build_native_resize.py")


class NativeResize(NativePillowBilinear):
    def __init__(self, library):
        path=Path(library).resolve()
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        build=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        root=Path(__file__).resolve().parents[1]
        if (set(build['source_hashes'])!=set(SOURCES) or
                any(digest(root/name)!=value for name,value in build['source_hashes'].items()) or
                digest(path)!=build['library_sha256'] or
                Image.__version__!=build['pillow'] or
                digest(Path(Image.core.__file__))!=build['pillow_core_sha256']):
            raise ValueError('native resize source/library/Pillow build manifest mismatch')
        super().__init__(path)
        self.info['build']=build
