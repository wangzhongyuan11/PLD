"""Checked invocation of the board-native, headless mesh executable."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from acdc.contract import ContractError,LABELS,COLORS,file_sha


class SurfaceGenerator:
    def __init__(self,binary,expected_sha,*,timeout=60,parameters=(0,40,.02)):
        self.binary=Path(binary).resolve();self.timeout=timeout
        if file_sha(self.binary)!=expected_sha:raise ContractError('native mesh binary drift')
        result=subprocess.run([str(self.binary),'--version'],capture_output=True,text=True,check=True,timeout=10)
        self.info=json.loads(result.stdout)
        if self.info.get('abi')!=1 or self.info.get('vtk')!='9.6.0':raise ContractError('native mesh ABI/VTK mismatch')
        self.info['binary_sha256']=expected_sha
        self.parameters=tuple(parameters);self.info['parameters']=list(parameters)

    def generate(self,labels,affine,directory,*,cancel):
        labels=np.ascontiguousarray(labels)
        if labels.dtype!=np.uint8 or labels.ndim!=3 or labels.size>2_000_000 or np.any(labels>3):
            raise ContractError('invalid mesh label volume')
        affine=np.asarray(affine,np.float64)
        if affine.shape!=(4,4) or not np.all(np.isfinite(affine)):raise ContractError('invalid mesh affine')
        directory=Path(directory).resolve();directory.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='mesh-',dir=directory.parent) as scratch:
            scratch=Path(scratch);labels.tofile(scratch/'labels.u8')
            np.savetxt(scratch/'affine.txt',affine,fmt='%.17g')
            with (scratch/'error.log').open('wb') as errors:
                process=subprocess.Popen([str(self.binary),str(scratch/'labels.u8'),*map(str,labels.shape),
                    str(scratch/'affine.txt'),str(directory),*map(str,self.parameters)],stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,stderr=errors)
                deadline=time.monotonic()+self.timeout
                try:
                    while process.poll() is None:
                        if cancel.is_set():raise InterruptedError('mesh generation cancelled')
                        if time.monotonic()>deadline:raise TimeoutError('native mesh deadline exceeded')
                        time.sleep(.02)
                    if process.returncode:
                        raise RuntimeError('native mesh failed: '+(scratch/'error.log').read_text(errors='replace')[:2000])
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        try:process.wait(timeout=3)
                        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
                    raise
        meshes=[]
        for c in (1,2,3):
            name=LABELS[c];report=json.loads((directory/(name+'.json')).read_text('utf-8'))
            row=dict(label=c,name=name,color=COLORS[c],fov_truncated=report['fov_truncated'],
                     generator='pynq_ps',coordinate_system='RAS',quantitative=False,raw=None,smooth=None)
            for kind in ('raw','smooth'):
                prefix=name+'.'+kind
                meta=json.loads((directory/(prefix+'.json')).read_text('utf-8'))
                count,triangles=meta['points'],meta['triangles']
                files=dict(points=prefix+'.points.f32',normals=prefix+'.normals.f32',indices=prefix+'.indices.u32')
                for key,size in (('points',count*12),('normals',count*12),('indices',triangles*12)):
                    if (directory/files[key]).stat().st_size!=size:raise ContractError('native mesh asset length mismatch')
                row[kind]=dict(meta,vertex_count=count,triangle_count=triangles,**files,
                               ply=prefix+'.ply',vtp=prefix+'.vtp',smoothed=kind=='smooth')
            meshes.append(row)
        return meshes
