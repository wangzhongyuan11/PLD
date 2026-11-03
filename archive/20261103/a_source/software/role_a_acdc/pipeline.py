"""PC native-volume inference, probability fusion, NIfTI, physical PLY and viewer."""
from __future__ import annotations
import argparse,json,time,collections
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from .common import DATA,RUN,LABELS,ARCHITECTURE,save_json,sha256,utc_now
from .geometry import prepare_volume,probabilities_to_native,ACDC_GEOMETRY_POLICY
from .inference import predict_working
from .quantize import load_fp32,load_integer

COLORS={1:(255,0,0),2:(0,255,0),3:(0,0,255)}

def segmentation_metrics(pred,gt,affine):
    result={};voxel_volume=abs(float(np.linalg.det(affine[:3,:3])))
    def surface(mask):
        edge=mask & ~binary_erosion(mask,border_value=0)
        return nib.affines.apply_affine(affine,np.argwhere(edge))
    for c in (1,2,3):
        p=pred==c;g=gt==c;den=int(p.sum()+g.sum())
        row={'dice':float(2*np.count_nonzero(p&g)/den) if den else 1.,
             'pred_volume_ml':float(p.sum()*voxel_volume/1000),'reference_volume_ml':float(g.sum()*voxel_volume/1000),
             'pred_empty':not bool(p.any()),'reference_empty':not bool(g.any())}
        if not p.any() or not g.any():
            row.update(hd95_mm=0. if not den else None,hd_mm=0. if not den else None,assd_mm=0. if not den else None,
                       surface_status='both_empty' if not den else 'one_empty_undefined_distance')
        else:
            ps=surface(p);gs=surface(g);a=cKDTree(ps).query(gs,workers=1)[0];b=cKDTree(gs).query(ps,workers=1)[0]
            distances=np.concatenate((a,b))
            row.update(hd95_mm=float(np.percentile(distances,95)),hd_mm=float(distances.max()),assd_mm=float(distances.mean()),
                       surface_status='physical affine; pooled directed voxel-surface center distances')
        result[str(c)]=row
    return result

def export_surface(labels,affine,path,class_id):
    mask=labels==class_id
    if not mask.any():return {'class':class_id,'empty':True,'file':None}
    padded=np.pad(mask.astype(np.uint8),1)
    vertices,faces,_,_=marching_cubes(padded,.5,allow_degenerate=False)
    vertices=nib.affines.apply_affine(affine,vertices-1)
    if np.linalg.det(affine[:3,:3])<0:faces=faces[:,::-1]
    color=COLORS[class_id]
    with Path(path).open('w',encoding='ascii',newline='\n') as f:
        f.write(f'ply\nformat ascii 1.0\ncomment physical RAS millimetres; no smoothing\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        for v in vertices:f.write(f'{v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {color[0]} {color[1]} {color[2]}\n')
        for face in faces:f.write(f'3 {face[0]} {face[1]} {face[2]}\n')
    return {'class':class_id,'label':LABELS[str(class_id)],'empty':False,'file':Path(path).name,'vertices':len(vertices),'faces':len(faces),'coordinate_system':'RAS mm'}

def infer_native(image,models,*,baseline=False,geometry_policy='strict',return_probabilities=False):
    x,_,metadata=prepare_volume(image,geometry_policy=geometry_policy);native_sum=None;rows=[]
    device=next(iter(models.values()))[1]
    for direction,(model,dev) in models.items():
        t=time.perf_counter()
        p=predict_working(model,x,direction,torch.device(dev),baseline=baseline)
        native=probabilities_to_native(p,metadata)
        native_sum=native if native_sum is None else native_sum+native
        rows.append({'direction':direction,'seconds':time.perf_counter()-t,'backend':'integer_software_reference' if hasattr(model,'quantized') else 'pytorch_fp32'})
    fused=native_sum/len(models)
    result=(fused.argmax(0).astype(np.uint8),metadata,rows)
    return (*result,fused) if return_probabilities else result

def run_case(image,output,backend='fp32',directions=('axial','coronal','sagittal'),geometry_policy='strict',model_root=None):
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    models={};identity={}
    for direction in directions:
        if backend=='int8':
            d=Path(model_root)/direction/'int8' if model_root else RUN/'exports'/direction
            models[direction]=(load_integer(d,device)[0],device);identity[direction]=sha256(d/'model_spec.json')
        else:
            path=Path(model_root)/direction/'model_state_fp32.pt' if model_root else RUN/'training'/direction/'best.pt'
            models[direction]=(load_fp32(path,device)[0],device);identity[direction]=sha256(path)
    labels,metadata,timing=infer_native(image,models,geometry_policy=geometry_policy)
    return save_prediction(image,output,labels,metadata,timing,identity,backend)

def save_prediction(image,output,labels,metadata,timing,identity,backend,board_inference=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=nib.load(str(image));affine=np.array(metadata['native_affine'])
    prediction=nib.Nifti1Image(labels,affine,source.header.copy());prediction.set_data_dtype(np.uint8)
    prediction.set_qform(affine,1);prediction.set_sform(affine,2);prediction.header.set_xyzt_units('mm')
    nib.save(prediction,output/'prediction_native.nii.gz')
    surfaces=[export_surface(labels,affine,output/f'{LABELS[str(c)]}.ply',c) for c in (1,2,3)]
    summary={'created':utc_now(),'architecture':ARCHITECTURE,'source_sha256':sha256(image),'backend':backend,
          'model_identity':identity,'geometry':metadata,'directions':timing,'fusion':'equal arithmetic mean of per-model softmax probabilities in native physical grid',
          'native_volumes_ml':{LABELS[str(c)]:float(np.count_nonzero(labels==c)*metadata['native_voxel_volume_mm3']/1000) for c in (1,2,3)},
          'surfaces':surfaces,'board_inference':board_inference,'independent_board_acceptance':False,
          'limits':'research/demo; ventricles and myocardium only; interpolation does not create acquired anatomical detail'}
    save_json(output/'result.json',summary)
    from .viewer import create_viewer
    create_viewer(image,output/'prediction_native.nii.gz',metadata,output/'viewer.html')
    return summary

def cardiac_function(ed_volumes,es_volumes):
    result={}
    for name in ('RV_cavity','LV_cavity'):
        ed=float(ed_volumes[name]);es=float(es_volumes[name]);valid=ed>0 and 0<=es<=ed
        result[name]={'edv_ml':ed,'esv_ml':es,'raw_ef_percent':100*(ed-es)/ed if ed>0 else None,
                      'ef_valid_range':valid,'ef_percent':100*(ed-es)/ed if valid else None,
                      'status':'derived from native-grid segmentation' if valid else 'invalid segmentation/phase order: review ED/ES and masks'}
    return result

def run_pair(ed_image,es_image,output,backend='fp32',geometry_policy='strict',model_root=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ed=run_case(ed_image,output/'ED',backend,geometry_policy=geometry_policy,model_root=model_root)
    es=run_case(es_image,output/'ES',backend,geometry_policy=geometry_policy,model_root=model_root)
    result={'created':utc_now(),'backend':backend,'source_ed_sha256':sha256(ed_image),'source_es_sha256':sha256(es_image),
            'cardiac_function':cardiac_function(ed['native_volumes_ml'],es['native_volumes_ml']),
            'volume_source':'unsmoothed native label voxels and physical affine determinant, not mesh volume',
            'research_only':True,'phase_verification':'ED and ES supplied explicitly by caller; no inferred phase detection'}
    save_json(output/'cardiac_function.json',result)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>ACDC ED/ES</title><h1>ACDC ED / ES</h1><p><a href="ED/viewer.html">舒张末 ED 三视图与三维</a></p><p><a href="ES/viewer.html">收缩末 ES 三视图与三维</a></p><p><a href="cardiac_function.json">原生网格容积与 EF 记录</a></p><p>科研演示；EF由两期分割推导，不构成临床诊断。</p>',encoding='utf-8')
    return result

def validate_native(name,backend='fp32'):
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    names=('axial','coronal','sagittal') if name=='fusion' else (name,)
    models={};identities={}
    for n in names:
        direction='axial' if n=='baseline2d' else n
        model=load_integer(RUN/'exports'/n,device)[0] if backend=='int8' else load_fp32(RUN/'training'/n/'best.pt',device)[0]
        models[direction]=(model,device)
        identities[direction]=sha256(RUN/'exports'/n/'model_spec.json') if backend=='int8' else sha256(RUN/'training'/n/'best.pt')
    split=json.loads((DATA/'SPLIT.json').read_text(encoding='utf-8'));rows=[];start=time.time()
    for patient in split['dev']:
        for phase in ('ED','ES'):
            frame=int(split['patient_info'][patient][phase]);image=DATA/'training'/patient/f'{patient}_frame{frame:02}.nii.gz'
            label=image.with_name(image.name.replace('.nii.gz','_gt.nii.gz'))
            if name=='fusion':
                probabilities=[];timing=[]
                for direction in names:
                    d=RUN/'validation/probabilities'/f'{direction}_{backend}';record=json.loads((d/f'{patient}_{phase}.json').read_text(encoding='utf-8'))
                    file=d/f'{patient}_{phase}.npz'
                    if record['model_identity']!=identities[direction] or record['split_sha256']!=sha256(DATA/'SPLIT.json') or record['source_sha256']!=sha256(image) or sha256(file)!=record['probability_sha256']:raise ValueError('stale/corrupt probability cache')
                    with np.load(file,allow_pickle=False) as cache:probabilities.append(cache['probabilities'])
                    metadata=record['geometry'];timing.append({'direction':direction,'backend':backend,'reused_verified_native_probabilities':True})
                if any(p.dtype!=np.float32 or p.shape!=(4,*metadata['native_shape']) for p in probabilities):raise ValueError('invalid native probability cache')
                pred=((probabilities[0]+probabilities[1]+probabilities[2])/3).argmax(0).astype(np.uint8)
            else:
                pred,metadata,timing,prob=infer_native(image,models,baseline=name=='baseline2d',geometry_policy=ACDC_GEOMETRY_POLICY,return_probabilities=True)
                if name!='baseline2d':
                    d=RUN/'validation/probabilities'/f'{name}_{backend}';d.mkdir(parents=True,exist_ok=True);file=d/f'{patient}_{phase}.npz'
                    np.savez_compressed(file,probabilities=prob.astype(np.float32))
                    save_json(d/f'{patient}_{phase}.json',{'model_identity':identities[name],'source_sha256':sha256(image),
                          'split_sha256':sha256(DATA/'SPLIT.json'),'probability_sha256':sha256(file),'geometry':metadata,
                          'labels_or_ground_truth_stored':False})
            gt=np.asarray(nib.load(label).dataobj,dtype=np.uint8);metrics=segmentation_metrics(pred,gt,np.array(metadata['native_affine']))
            rows.append({'patient':patient,'phase':phase,'metrics':metrics,'direction_times':timing})
        save_json(RUN/'validation'/f'{name}_{backend}_progress.json',{'volumes':len(rows),'total':40,'seconds':time.time()-start})
    per_patient=collections.defaultdict(list)
    for row in rows:per_patient[row['patient']].append([row['metrics'][str(c)]['dice'] for c in (1,2,3)])
    score=np.mean([np.mean(v,axis=0) for v in per_patient.values()],axis=0)
    functions=[]
    for patient in split['dev']:
        pts={r['phase']:r for r in rows if r['patient']==patient}
        pred={phase:{LABELS[str(c)]:pts[phase]['metrics'][str(c)]['pred_volume_ml'] for c in (1,2,3)} for phase in ('ED','ES')}
        ref={phase:{LABELS[str(c)]:pts[phase]['metrics'][str(c)]['reference_volume_ml'] for c in (1,2,3)} for phase in ('ED','ES')}
        functions.append({'patient':patient,'predicted':cardiac_function(pred['ED'],pred['ES']),'reference':cardiac_function(ref['ED'],ref['ES'])})
    report={'name':name,'backend':backend,'model_identity':identities,'patients':20,'volumes':40,'per_class_dice':score.tolist(),
         'patient_macro_dice':float(score.mean()),'rows':rows,'split_sha256':sha256(DATA/'SPLIT.json'),
         'cardiac_function':functions,
         'space':'native physical grid','official_test_access':False,'seconds':time.time()-start,'board_measured':False,
         'precision':'FP32 no autocast/TF32' if backend=='fp32' else 'exact Q24 INT8 feature graph; validated CUDA int8/int32 accumulator or exact FP64 CPU reference'}
    save_json(RUN/'validation'/f'{name}_{backend}.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--image');p.add_argument('--ed');p.add_argument('--es');p.add_argument('--output');p.add_argument('--models');p.add_argument('--backend',choices=['fp32','int8'],default='fp32')
    p.add_argument('--geometry-policy',choices=['strict',ACDC_GEOMETRY_POLICY],default='strict')
    p.add_argument('--validate',choices=['baseline2d','axial','coronal','sagittal','fusion']);a=p.parse_args()
    if a.validate:print(json.dumps({k:v for k,v in validate_native(a.validate,a.backend).items() if k!='rows'}))
    elif a.image and a.output:print(json.dumps(run_case(a.image,a.output,a.backend,geometry_policy=a.geometry_policy,model_root=a.models)))
    elif a.ed and a.es and a.output:print(json.dumps(run_pair(a.ed,a.es,a.output,a.backend,a.geometry_policy,a.models)))
    else:p.error('provide --validate, --image/--output, or --ed/--es/--output')
