"""Complete Axial ED/ES job: bounded prefetch, actual execution, restoration and export."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import threading
import time
import zipfile
import numpy as np

from .contract import ARCHITECTURE,PROFILE,PREPROCESSING,atomic_json,file_sha,digest
from .volume import load_volume,AxialInputs
from .reconstruct import Collector
from .statistics import volumes,paired
from .surface import export_surfaces


def _memory():
    # Linux VmRSS is actual RSS; Windows ru_maxrss is unavailable, never fabricate it.
    p=Path('/proc/self/status')
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    return None


class Pipeline:
    def __init__(self,backend,*,policy='strict',recognized_acdc=False,binding=None):
        self.backend=backend;self.policy=policy;self.recognized=recognized_acdc
        self.binding=binding if binding is not None else getattr(backend,'binding',None)

    def run(self,ed,es,output,*,cancel=None,progress=None,save_heads=False,save_probabilities=False,resume_from=None,upload_seconds=0.):
        import nibabel as nib
        output=Path(output)
        output.mkdir(parents=True,exist_ok=False)
        cancel=cancel or threading.Event()
        progress=progress or (lambda _:None)
        run_id=output.name;started=time.perf_counter();records=[];phase_stats={};phase_reports={}
        status=dict(run_id=run_id,status='preparing',execution_profile=PROFILE,synthetic=self.backend.synthetic,
                    backend=self.backend.kind,completed_slices=0,total_slices=None)

        def update(**values):
            status.update(values);atomic_json(output/'status.json',status);progress(dict(status))

        def check_cancel():
            if cancel.is_set():raise InterruptedError('job cancelled')

        try:
            # Read/prepare one phase at a time: bounded working probability memory.
            for phase,path in [('ED',ed),('ES',es)]:
                check_cancel();update(status='preparing',phase=phase)
                st=time.perf_counter()
                volume=load_volume(path,policy=self.policy,recognized_acdc=self.recognized,binding=self.binding)
                inputs=AxialInputs(volume);nz=volume.working.shape[2]
                phase_report=dict(native_shape=list(volume.native.shape),working_shape=list(volume.working.shape),
                                  expected_slices=nz,prepare_seconds=time.perf_counter()-st,geometry_sha256=volume.geometry_sha)
                geom=dict(volume.geometry,letterbox=inputs.spec)
                atomic_json(output/'geometry'/f'{phase}.json',geom)
                collector=Collector(volume,self.backend.model_sha,self.backend.head_scale,phase)
                update(status='running',phase=phase,phase_completed=0,phase_total=nz)
                # One prepared request ahead; only the main thread owns the hardware backend.
                with ThreadPoolExecutor(max_workers=1,thread_name_prefix='acdc-input') as prefetch, \
                     ThreadPoolExecutor(max_workers=1,thread_name_prefix='acdc-restore') as restore_pool:
                    pending=prefetch.submit(inputs.get,0)
                    restoring=None
                    def restore_plane(head,meta):
                        tick=time.perf_counter();collector.add(head,meta,inputs.spec)
                        return time.perf_counter()-tick
                    for z in range(nz):
                        check_cancel()
                        x,neighbors=pending.result()
                        if z+1<nz:pending=prefetch.submit(inputs.get,z+1)
                        request=dict(architecture_id=ARCHITECTURE,execution_profile=PROFILE,preprocessing_id=PREPROCESSING,
                            run_id=run_id,phase=phase,orientation='axial',center_index=z,neighbor_indices=neighbors,
                            total_slices=nz,geometry_sha256=volume.geometry_sha,model_sha256=self.backend.model_sha,
                            shape=[1,3,256,256],dtype='int8',layout='NCHW',input_scale=1/127)
                        request['runtime_sha256']=getattr(self.backend,'runtime_sha',None)
                        input_sha=hashlib.sha256(x.tobytes()).hexdigest()
                        saved=Path(resume_from)/'checkpoint'/phase/f'{z:04}.json' if resume_from else None
                        reused=False
                        if saved is not None and saved.is_file():
                            old=json.loads(saved.read_text('utf-8'))
                            expected_backend='pl' if self.backend.kind=='remote' else self.backend.kind
                            for field in ('phase','center_index','neighbor_indices','geometry_sha256','model_sha256','runtime_sha256'):
                                if old.get(field)!=request[field]:raise ValueError('resume identity mismatch: '+field)
                            raw=saved.with_suffix('.bin').read_bytes()
                            if (old.get('input_sha256')!=input_sha or old.get('head_sha256')!=hashlib.sha256(raw).hexdigest()
                                    or old.get('backend')!=expected_backend or old.get('synthetic') is not self.backend.synthetic):
                                raise ValueError('resume bytes/backend identity mismatch')
                            head=np.frombuffer(raw,np.int8).reshape((1,4,256,256)).copy()
                            meta=dict(old,run_id=run_id,nodes=[],events=[],source_run_id=old['run_id'])
                            seconds=0.;reused=True
                        else:
                            tick=time.perf_counter()
                            head,meta=self.backend.infer(x,request,cancel)
                            seconds=time.perf_counter()-tick
                            reused=bool(meta.get('cached_response',False))
                        if restoring is not None:records[-1]['restore_plane_seconds']=restoring.result()
                        restoring=restore_pool.submit(restore_plane,head,meta)
                        head_sha=hashlib.sha256(head.tobytes()).hexdigest()
                        checkpoint=output/'checkpoint'/phase;checkpoint.mkdir(parents=True,exist_ok=True)
                        pending_head=checkpoint/f'{z:04}.bin.pending';pending_head.write_bytes(head.tobytes())
                        pending_head.replace(checkpoint/f'{z:04}.bin')
                        atomic_json(checkpoint/f'{z:04}.json',dict(request,backend=meta['backend'],synthetic=meta['synthetic'],
                            head_scale=meta['head_scale'],head_sha256=head_sha,input_sha256=input_sha,
                            source_run_id=meta.get('source_run_id')))
                        row=dict(phase=phase,center_index=z,seconds=seconds,input_sha256=input_sha,reused=reused,
                                 head_sha256=hashlib.sha256(head.tobytes()).hexdigest(),backend=meta['backend'],
                                 synthetic=meta['synthetic'],nodes=meta.get('nodes',[]),events=meta.get('events',[]),
                                 event_summary=meta.get('event_summary'),ps_slice_seconds=meta.get('slice_seconds'),rss_bytes=_memory())
                        records.append(row)
                        if save_heads:
                            folder=output/'heads'/phase;folder.mkdir(parents=True,exist_ok=True)
                            head.tofile(folder/f'{z:04}.bin');atomic_json(folder/f'{z:04}.json',meta)
                        update(phase_completed=z+1,completed_slices=len(records))
                    if restoring is not None:records[-1]['restore_plane_seconds']=restoring.result()
                phase_report['resized_planes']=inputs.resized_planes
                update(status='restoring',phase=phase)
                st=time.perf_counter();prob,labels=collector.finish()
                phase_report['restore_seconds']=time.perf_counter()-st
                folder=output/'labels';folder.mkdir(exist_ok=True)
                image=nib.Nifti1Image(labels,volume.affine);image.header.set_xyzt_units('mm')
                nib.save(image,str(folder/f'{phase}.nii.gz'))
                phase_stats[phase]=volumes(labels,volume.affine)
                check_cancel();update(status='meshing',phase=phase)
                st=time.perf_counter();meshes=export_surfaces(labels,volume.affine,output/'meshes'/phase)
                phase_report['meshing_seconds']=time.perf_counter()-st
                if save_probabilities:np.save(output/f'{phase}_native_probabilities.npy',prob,allow_pickle=False)
                # Static binary assets for local browser sampling. X is fastest in the browser.
                viewer=output/'viewer'/phase;viewer.mkdir(parents=True)
                volume.native.astype('<f4').ravel(order='F').tofile(viewer/'gray.f32')
                labels.ravel(order='F').tofile(viewer/'labels.u8')
                corners=np.array(list(product(*[(0,n-1) for n in labels.shape])))
                world=nib.affines.apply_affine(volume.affine,corners)
                atomic_json(viewer/'volume.json',dict(shape=list(labels.shape),affine=volume.affine.tolist(),
                    inverse_affine=np.linalg.inv(volume.affine).tolist(),bounds=[world.min(0).tolist(),world.max(0).tolist()],
                    window=[float(np.percentile(volume.native,1)),float(np.percentile(volume.native,99))],
                    meshes=meshes,coordinate_system='RAS',phase=phase))
                phase_reports[phase]=phase_report
                del collector,prob,labels,inputs,volume
            check_cancel();update(status='exporting',total_slices=len(records))
            stats=paired(phase_stats['ED'],phase_stats['ES']);atomic_json(output/'statistics.json',stats)
            with (output/'statistics.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['phase','structure','voxels','volume_mL'])
                for phase,report in phase_stats.items():
                    for name,row in report['classes'].items():writer.writerow([phase,name,row['voxels'],row['mL']])
            times=[r['seconds'] for r in records if not r['reused']]
            median=float(np.median(times)) if times else None
            p95=float(np.percentile(times,95)) if times else None
            fps=len(times)/sum(times) if times else None
            atomic_json(output/'performance.json',dict(schema='pld-acdc-performance-v1',backend=self.backend.kind,
                synthetic=self.backend.synthetic,measured_on_this_host=True,board_validation=self.backend.kind=='pl',
                slice_count=len(records),fresh_inference_count=len(times),resumed_count=len(records)-len(times),
                slice_median_seconds=median,slice_p95_seconds=p95,measured_slice_throughput_fps=fps,
                target_4fps_passed=(self.backend.kind=='pl' and len(times)>=1000 and p95<=.25 and fps>=4),
                slice_sample_meets_250ms=(self.backend.kind=='pl' and bool(times) and p95<=.25),
                backend_initialization_seconds=getattr(self.backend,'initialization_seconds',None),upload_seconds=upload_seconds,
                hardware=getattr(self.backend,'hardware_info',None),runtime_sha256=getattr(self.backend,'runtime_sha',None),
                phases=phase_reports,records=records,pre_export_seconds=time.perf_counter()-started))
            atomic_json(output/'quality.json',dict(ground_truth_evaluation=False,notice='No accuracy claim without independent GT evaluation',
                        completeness=True,required_orientations=['axial'],fusion_mode='none',postprocessing='none'))
            (output/'README.txt').write_text('Axial-only ED/ES segmentation. Labels: 0 background,1 RV,2 MYO,3 LV.\n'
                'Coordinates RAS, millimeters; adopted physical affine. No smoothing or morphology.\n'
                f'Backend={self.backend.kind}; synthetic={self.backend.synthetic}.\n'
                '工程教学与算法演示，不作为临床诊断依据。\n',encoding='utf-8')
            assets={f.relative_to(output).as_posix():file_sha(f) for f in output.rglob('*')
                    if f.is_file() and 'checkpoint' not in f.relative_to(output).parts
                    and f.name not in ('status.json','manifest.json','result.zip')}
            atomic_json(output/'manifest.json',dict(schema='pld-acdc-result-v1',execution_profile=PROFILE,fusion_mode='none',
                model_sha256=self.backend.model_sha,head_scale=self.backend.head_scale,backend=self.backend.kind,
                synthetic=self.backend.synthetic,files=assets))
            # ZIP cannot contain its own final duration. external completion.json reports finalization wall time.
            with zipfile.ZipFile(output/'result.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
                for rel in sorted([*assets,'manifest.json']):archive.write(output/rel,rel)
            job_seconds=time.perf_counter()-started
            completion=dict(end_to_end_seconds=upload_seconds+job_seconds,job_to_zip_seconds=job_seconds,
                            upload_seconds=upload_seconds,zip_sha256=file_sha(output/'result.zip'),
                            timing_scope='server upload body read start to ZIP ready; browser rendering/download reported separately')
            atomic_json(output/'completion.json',completion)
            update(status='completed',completion=completion)
            return status
        except InterruptedError:
            update(status='cancelled');raise
        except Exception as exc:
            update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
