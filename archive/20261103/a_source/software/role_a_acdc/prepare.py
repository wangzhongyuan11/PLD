"""Patient-stratified split and physical-grid cache; no testing directory access."""
from __future__ import annotations
import argparse,json,random,collections,shutil,time
from pathlib import Path
import numpy as np
import nibabel as nib
from .common import DATA,RUN,save_json,sha256,utc_now
from .geometry import prepare_volume,SPACING,physical_affine,ACDC_GEOMETRY_POLICY

def cfg(path):
    return dict(line.split(':',1) for line in path.read_text().splitlines() if ':' in line)

def make_split():
    verified=json.loads((DATA/'DOWNLOAD_VERIFIED.json').read_text(encoding='utf-8'))
    assert verified['patients']==100 and len(verified['files'])==500 and not verified['test_access']
    groups=collections.defaultdict(list);info={}
    for folder in sorted((DATA/'training').iterdir()):
        c={k.strip():v.strip() for k,v in cfg(folder/'Info.cfg').items()}
        group=c['Group'];groups[group].append(folder.name);info[folder.name]=c
    if len(groups)!=5 or any(len(x)!=20 for x in groups.values()):raise ValueError('expected five disease groups with 20 patients each')
    rng=random.Random(20261017);train=[];dev=[]
    for group,patients in sorted(groups.items()):
        pts=list(patients);rng.shuffle(pts);train.extend(pts[:16]);dev.extend(pts[16:])
    result={'schema':'acdc-patient-split-v1','seed':20261017,'created':utc_now(),
            'train':sorted(train),'dev':sorted(dev),'groups':dict(groups),'patient_info':info,
            'official_test':'NOT_DOWNLOADED_NOT_ACCESSED','source_sha256':sha256(DATA/'DOWNLOAD_VERIFIED.json')}
    save_json(DATA/'SPLIT.json',result)
    headers=[]
    for pt in result['train']:
        for phase in ('ED','ES'):
            path=DATA/'training'/pt/f'{pt}_frame{int(info[pt][phase]):02}.nii.gz';image=nib.load(path)
            headers.append({'patient':pt,'phase':phase,'shape':list(image.shape),
                            'spacing':nib.affines.voxel_sizes(physical_affine(image,ACDC_GEOMETRY_POLICY)[0]).tolist(),
                            'source_affine_spacing':nib.affines.voxel_sizes(image.affine).tolist(),
                            'source_pixdim':list(map(float,image.header.get_zooms()[:3])),
                            'axes':list(nib.aff2axcodes(image.affine))})
    spacing=np.array([h['spacing'] for h in headers])
    save_json(DATA/'TRAIN_FINGERPRINT.json',{'training_volumes':160,'headers':headers,
        'median_native_spacing_mm':np.median(spacing,axis=0).tolist(),'working_spacing_mm':list(SPACING),
        'decision':'2 mm RAS grid for three orthogonal directions with bounded PC memory; measured source thickness retained in metadata; interpolation adds no acquired detail',
        'dev_or_test_used_to_choose_grid':False})
    return result

def build_cache():
    split=make_split() if not (DATA/'SPLIT.json').exists() else json.loads((DATA/'SPLIT.json').read_text(encoding='utf-8'))
    cache=DATA/'cache_ras2mm_physical_v2';cache.mkdir(parents=True,exist_ok=True);records=[]
    start=time.time()
    for part in ('train','dev'):
        for patient in split[part]:
            for phase in ('ED','ES'):
                n=int(split['patient_info'][patient][phase]);key=f'{patient}_{phase}'
                source=DATA/'training'/patient/f'{patient}_frame{n:02}.nii.gz'
                label=source.with_name(source.name.replace('.nii.gz','_gt.nii.gz'))
                d=cache/key;d.mkdir(exist_ok=True)
                if not (d/'READY.json').exists():
                    x,y,m=prepare_volume(source,label,geometry_policy=ACDC_GEOMETRY_POLICY)
                    if x.nbytes*2>500_000_000:raise ValueError('unusually large volume; review geometry')
                    np.save(d/'image.npy',x.astype(np.float16),allow_pickle=False)
                    np.save(d/'label.npy',y,allow_pickle=False)
                    m.update(patient=patient,phase=phase,split=part,source=str(source.relative_to(DATA)),
                             source_sha256=sha256(source),label_sha256=sha256(label),split_sha256=sha256(DATA/'SPLIT.json'))
                    save_json(d/'geometry.json',m)
                    save_json(d/'READY.json',{'image_sha256':sha256(d/'image.npy'),'label_sha256':sha256(d/'label.npy')})
                m=json.loads((d/'geometry.json').read_text(encoding='utf-8'))
                if m['split_sha256']!=sha256(DATA/'SPLIT.json'):raise ValueError('stale cache split')
                records.append({'key':key,'patient':patient,'phase':phase,'split':part,'shape':m['working_shape'],
                                'relative':d.relative_to(DATA).as_posix()})
            if len(records)%20==0:
                save_json(RUN/'PREPARE_PROGRESS.json',{'volumes':len(records),'total':200,'seconds':time.time()-start})
                print(f'prepared {len(records)}/200',flush=True)
    save_json(DATA/'CACHE_MANIFEST.json',{'created':utc_now(),'split_sha256':sha256(DATA/'SPLIT.json'),
                 'working_spacing_mm':list(SPACING),'volumes':records,'test_access':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['split','cache']);a=p.parse_args()
    make_split() if a.action=='split' else build_cache()
