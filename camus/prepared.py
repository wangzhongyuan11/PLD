"""PC-prepared CAMUS INT8 input, with native pixels for board result rendering."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from .preprocess import LoadedGray,PreprocessResult


def prepare(source,destination):
    from .native import load_native
    prep=load_native().preprocess(source)
    metadata=dict(schema='pld-camus-prepared-v1',source_name=Path(source).name,
        letterbox=prep.letterbox_meta,spacing_yx_mm=prep.source.spacing_yx_mm,
        clipped_low=prep.clipped_low,clipped_high=prep.clipped_high,
        input_shape=list(prep.input_int8.shape),dtype='int8',layout='NCHW')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        archive.writestr('metadata.json',json.dumps(metadata))
        for name,array in [('input',prep.input_int8),('native_gray',prep.raw_gray)]:
            with archive.open(name+'.npy','w') as stream:np.save(stream,array,allow_pickle=False)
    return metadata


def load(path):
    with zipfile.ZipFile(path) as archive:
        if sum(row.file_size for row in archive.infolist())>64*1024*1024:raise ValueError('prepared CAMUS exceeds memory limit')
        meta=json.loads(archive.read('metadata.json'))
        q=np.load(io.BytesIO(archive.read('input.npy')),allow_pickle=False)
        gray=np.load(io.BytesIO(archive.read('native_gray.npy')),allow_pickle=False)
    if (meta['schema']!='pld-camus-prepared-v1' or q.dtype!=np.int8 or q.shape!=(1,1,256,256)
            or gray.ndim!=2 or not 0<gray.size<=8_000_000):
        raise ValueError('invalid prepared CAMUS representation')
    source=LoadedGray(gray,None if meta['spacing_yx_mm'] is None else tuple(meta['spacing_yx_mm']),
                      'pc_prepared_int8',str(path))
    return PreprocessResult(gray,None,None,q,meta['letterbox'],meta['clipped_low'],meta['clipped_high'],source)
