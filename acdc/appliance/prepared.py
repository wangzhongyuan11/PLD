"""Portable PC-prepared 2.5D tensors; reconstruction stays in native geometry."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from .volume import Volume,AxialInputs,load_volume,MAX_NATIVE_VOXELS,MAX_WORKING_VOXELS


def prepare(source,destination,kernel,*,binding,policy='acdc-direction-only-sform-v1'):
    volume=load_volume(source,kernel,policy=policy,binding=binding)
    axial=AxialInputs(volume,kernel)
    planes=np.stack([axial.get(z)[0][0,1] for z in range(volume.working_shape[2])])
    metadata=dict(schema='pld-acdc-prepared-v1',source_name=Path(source).name,
        geometry=volume.geometry,letterbox=axial.spec,input_shape=[len(planes),1,3,256,256],
        dtype='int8',layout='slices,NCHW')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        archive.writestr('metadata.json',json.dumps(metadata))
        # Neighboring 2.5D inputs share two planes. Ship each quantized plane
        # once; the board only gathers already-prepared bytes into NCHW.
        for name,array in [('planes',planes),('native_gray',volume.native)]:
            with archive.open(name+'.npy','w') as stream:np.save(stream,array,allow_pickle=False)
    return metadata


class PreparedInputs:
    def __init__(self,array,spec):self.array=array;self.spec=spec;self.resized_planes=0
    def get(self,z):
        nz=len(self.array)
        neighbors=[max(z-1,0),z,min(z+1,nz-1)]
        return self.array[neighbors][None],neighbors


def load(path):
    with zipfile.ZipFile(path) as archive:
        if sum(row.file_size for row in archive.infolist())>80*1024*1024:raise ValueError('prepared ACDC exceeds memory limit')
        meta=json.loads(archive.read('metadata.json'))
        inputs=np.load(io.BytesIO(archive.read('planes.npy')),allow_pickle=False)
        native=np.load(io.BytesIO(archive.read('native_gray.npy')),allow_pickle=False)
    geometry=meta['geometry'];nz=geometry['working_shape'][2]
    if (meta['schema']!='pld-acdc-prepared-v1' or inputs.dtype!=np.int8 or inputs.shape!=(nz,256,256)
            or native.dtype!=np.float32 or list(native.shape)!=geometry['native_shape']
            or native.size>MAX_NATIVE_VOXELS or int(np.prod(geometry['working_shape']))>MAX_WORKING_VOXELS):
        raise ValueError('invalid prepared ACDC representation')
    volume=Volume(native,np.asarray(geometry['native_affine'],np.float64),None,geometry)
    return volume,PreparedInputs(inputs,meta['letterbox'])
