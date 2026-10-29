"""V2 two-phase native numerical path with bounded CPU/PL overlap.

This stage produces numerical results only. The final appliance must also run
meshing, MPR, export and browser readiness before publishing a completed job.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import threading
import time

import numpy as np

from acdc.contract import ARCHITECTURE,PROFILE,PREPROCESSING,ContractError,atomic_json,checked_int8,HEAD_SHAPE
from acdc.statistics import volumes,paired
from .volume import load_volume,AxialInputs
from .reconstruct import Collector


class NumericalPipeline:
    def __init__(self,backend,kernel,exp_table,*,binding,policy='strict',verify=True):
        self.backend=backend;self.kernel=kernel;self.table=exp_table
        self.binding=binding;self.policy=policy
        self.verify=verify

    def run(self,ed,es,output,*,cancel=None,progress=None,overlap=True,save_heads=False,phase_ready=None):
        import nibabel as nib
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
        cancel=cancel or threading.Event();progress=progress or (lambda _:None)
        stats={};reports={};records=[];began=time.perf_counter();run_id=output.name

        def update(status,**fields):
            record=dict(run_id=run_id,status=status,**fields)
            atomic_json(output/'status.json',record);progress(record)

        def check_cancel():
            if cancel.is_set():raise InterruptedError('V2 job cancelled')

        phase_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='v2-phase-input')
        next_volume=None
        def prepare_volume(path):
            tick=time.perf_counter()
            value=load_volume(path,self.kernel,policy=self.policy,binding=self.binding)
            return value,tick,time.perf_counter()-tick

        try:
            for phase,path in (('ED',ed),('ES',es)):
                check_cancel();update('preparing',phase=phase)
                volume,prepare_begin,prepare=next_volume.result() if next_volume is not None else prepare_volume(path)
                next_volume=None
                inputs=AxialInputs(volume,self.kernel);nz=volume.working.shape[2]
                collector=Collector(volume,self.kernel,self.table)
                # One additional gray/working volume only; never preallocate a
                # second four-channel collector while ED restoration is alive.
                if overlap and phase=='ED':next_volume=phase_pool.submit(prepare_volume,es)
                inference_started=time.perf_counter()
                atomic_json(output/'geometry'/f'{phase}.json',dict(volume.geometry,letterbox=inputs.spec))

                def prepare_input(z):
                    prepared=time.perf_counter()
                    x,neighbors=inputs.get(z)
                    return x,neighbors,hashlib.sha256(x.tobytes()).hexdigest() if self.verify else None,time.perf_counter()-prepared

                with ThreadPoolExecutor(max_workers=1,thread_name_prefix='v2-input') as prefetch, \
                     ThreadPoolExecutor(max_workers=1,thread_name_prefix='v2-restore') as restoring:
                    pending=prefetch.submit(prepare_input,0) if overlap else None
                    previous=None
                    for z in range(nz):
                        check_cancel()
                        x,neighbors,input_sha,input_seconds=pending.result() if overlap else prepare_input(z)
                        if overlap and z+1<nz:pending=prefetch.submit(prepare_input,z+1)
                        request=dict(architecture_id=ARCHITECTURE,execution_profile=PROFILE,preprocessing_id=PREPROCESSING,
                            run_id=run_id,phase=phase,orientation='axial',center_index=z,neighbor_indices=neighbors,
                            total_slices=nz,geometry_sha256=volume.geometry_sha,model_sha256=self.backend.model_sha,
                            runtime_sha256=self.backend.runtime_sha,shape=[1,3,256,256],dtype='int8',layout='NCHW',input_scale=1/127)
                        tick=time.perf_counter();head,meta=self.backend.infer(x,request,cancel)
                        seconds=time.perf_counter()-tick
                        if self.verify:
                            for key in ('architecture_id','execution_profile','preprocessing_id','run_id','phase','orientation',
                                        'center_index','neighbor_indices','total_slices','geometry_sha256','model_sha256','runtime_sha256'):
                                if meta.get(key)!=request[key]:raise ContractError('V2 inference response mismatch: '+key)
                            if (meta.get('cached_response') or meta.get('head_scale')!=self.backend.head_scale
                                    or meta.get('backend')!=self.backend.kind or meta.get('synthetic') is not self.backend.synthetic
                                    or meta.get('status')!='completed' or meta.get('shape')!=list(HEAD_SHAPE)
                                    or meta.get('dtype')!='int8' or meta.get('layout')!='NCHW'):
                                raise ContractError('unexpected inference source/scale or reused response')
                            head=checked_int8(head,HEAD_SHAPE)
                        if previous is not None:previous.result()
                        if overlap:previous=restoring.submit(collector.add,head,z,inputs.spec)
                        else:collector.add(head,z,inputs.spec)
                        records.append(dict(phase=phase,center_index=z,input_sha256=input_sha,
                            head_sha256=hashlib.sha256(head.tobytes()).hexdigest() if self.verify else None,seconds=seconds,
                            input_prepare_seconds=input_seconds,inference_begin_seconds=tick-began,
                            inference_end_seconds=tick+seconds-began,
                            device_timing={k:meta[k] for k in ('input_validation_seconds','input_layout_seconds',
                                'flush_seconds','submit_wait_seconds','head_seconds','pl_cycles','jobs','epoch') if k in meta},
                            pl_seconds=meta.get('pl_seconds'),backend=meta['backend'],synthetic=meta['synthetic']))
                        if save_heads:
                            folder=output/'heads'/phase;folder.mkdir(parents=True,exist_ok=True)
                            head.tofile(folder/f'{z:04}.bin')
                        update('infer_restore',phase=phase,phase_completed=z+1,phase_total=nz,
                               completed_slices=len(records))
                    if previous is not None:previous.result()
                inference_window=time.perf_counter()-inference_started
                check_cancel();update('restoring',phase=phase)
                tick=time.perf_counter();_,labels=collector.finish();finish=time.perf_counter()-tick
                label_dir=output/'labels';label_dir.mkdir(exist_ok=True)
                image=nib.Nifti1Image(labels,volume.affine);image.header.set_xyzt_units('mm')
                nib.save(image,str(label_dir/f'{phase}.nii.gz'))
                # Compact uncompressed files feed future board MPR without another decode.
                viewer=output/'viewer'/phase;viewer.mkdir(parents=True)
                np.save(viewer/'gray.npy',volume.native,allow_pickle=False)
                np.save(viewer/'labels.npy',labels,allow_pickle=False)
                atomic_json(viewer/'geometry.json',dict(shape=list(labels.shape),affine=volume.affine.tolist()))
                stats[phase]=volumes(labels,volume.affine)
                reports[phase]=dict(slices=nz,prepare_seconds=prepare,inference_restore_window_seconds=inference_window,
                    prepare_begin_monotonic=prepare_begin,inference_begin_monotonic=inference_started,
                    native_restore_begin_monotonic=tick,
                    final_restore_seconds=finish,restore_stages=collector.timings,resized_planes=inputs.resized_planes,
                    native_shape=list(volume.native.shape),working_shape=list(volume.working.shape))
                del collector,volume,inputs,labels
                if phase_ready is not None:phase_ready(phase)
            atomic_json(output/'statistics.json',paired(stats['ED'],stats['ES']))
            report=dict(schema='pld-acdc-appliance-numerics-v2',status='numerics_completed',
                job_id=run_id,overlap=overlap,model_sha256=self.backend.model_sha,runtime_sha256=self.backend.runtime_sha,
                backend=self.backend.kind,synthetic=self.backend.synthetic,phases=reports,records=records,
                numerical_seconds=time.perf_counter()-began,native_build=self.kernel.info,
                numerical_begin_monotonic=began,
                scope='ED/ES numerical path only; no mesh, upload, ZIP or browser readiness')
            atomic_json(output/'numerics.json',report)
            update('numerics_completed',completed_slices=len(records),total_slices=len(records))
            return report
        except InterruptedError:
            update('cancelled');raise
        except Exception as exc:
            update('failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            phase_pool.shutdown(wait=True,cancel_futures=True)
