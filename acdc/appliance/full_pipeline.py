"""Numerics, two-phase surfaces/MPR, statistics and exports on one board."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

import numpy as np

from acdc.contract import atomic_json,file_sha,is_sha,ContractError
from .mpr import MPRVolume
from .archive import write_zip


class FullPipeline:
    def __init__(self,numerics,surfaces,*,application_sha=None):
        self.numerics=numerics;self.surfaces=surfaces;self.backend=numerics.backend
        if application_sha is not None and not is_sha(application_sha):raise ContractError('invalid application SHA')
        self.application_sha=application_sha

    def run(self,ed,es,output,*,cancel=None,progress=None,upload_seconds=0.):
        output=Path(output);cancel=cancel or threading.Event();progress=progress or (lambda _:None)
        began=time.perf_counter();phase_reports={}

        def update(status,**values):
            state=dict(run_id=output.name,status=status,**values)
            atomic_json(output/'status.json',state);progress(state)

        def check_cancel():
            if cancel.is_set():raise InterruptedError('full job cancelled')

        def prepare_display(phase):
            # Only the main numerical thread publishes job status. ED
            # display generation may overlap ES inference, with one display
            # future and no access to the FPGA owner or probability arrays.
            check_cancel()
            viewer=output/'viewer'/phase
            labels=np.load(viewer/'labels.npy',allow_pickle=False,mmap_mode='r')
            geometry=json.loads((viewer/'geometry.json').read_text('utf-8'))
            tick=time.perf_counter();volume=MPRVolume(viewer,self.numerics.kernel)
            low,high=np.percentile(volume.gray,[1,99]);window=max(1.,float(high-low));level=float((low+high)/2)
            occupied=self.numerics.kernel.foreground_bounds(labels)
            if occupied is not None:
                center=occupied.mean(axis=0)
                position=(volume.affine[:3,:3]@center+volume.affine[:3,3]).tolist()
            else:position=volume.bounds.mean(axis=0).tolist()
            initial={}
            for plane in ('axial','coronal','sagittal'):
                check_cancel()
                png,meta=volume.render(plane,position,level=level,window=window)
                destination=output/'mpr'/phase;destination.mkdir(parents=True,exist_ok=True)
                (destination/(plane+'.png')).write_bytes(png);initial[plane]=meta
            mpr_seconds=time.perf_counter()-tick
            preview=dict(phase=phase,bounds=volume.bounds.tolist(),position=position,window=[level,window],initial_mpr=initial)
            if not (output/'preview.json').exists():
                atomic_json(output/'preview.json',preview)
                progress(dict(run_id=output.name,status='preview_ready',phase=phase,preview_ready=True))
            mesh_begin=time.perf_counter()
            meshes=self.surfaces.generate(labels,geometry['affine'],output/'meshes'/phase,cancel=cancel)
            mesh_seconds=time.perf_counter()-mesh_begin
            atomic_json(viewer/'volume.json',dict(phase=phase,shape=geometry['shape'],affine=geometry['affine'],
                bounds=volume.bounds.tolist(),position=position,window=[level,window],meshes=meshes,
                initial_mpr=initial,coordinate_system='RAS',generator='pynq_ps'))
            result=dict(mesh_seconds=mesh_seconds,mpr_seconds=mpr_seconds,
                mesh_begin_monotonic=mesh_begin,mpr_begin_monotonic=tick)
            del volume,labels
            return result

        try:
            with ThreadPoolExecutor(max_workers=1,thread_name_prefix='v2-display') as displays:
                futures={}
                def phase_ready(phase):
                    futures[phase]=displays.submit(prepare_display,phase)
                report=self.numerics.run(ed,es,output,cancel=cancel,progress=progress,phase_ready=phase_ready)
                for phase in ('ED','ES'):
                    update('meshing',phase=phase)
                    phase_reports[phase]=futures[phase].result()
            check_cancel();update('exporting')
            statistics=json.loads((output/'statistics.json').read_text('utf-8'))
            with (output/'statistics.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['phase','structure','voxels','volume_mL'])
                for phase,values in statistics['phases'].items():
                    for name,row in values['classes'].items():writer.writerow([phase,name,row['voxels'],row['mL']])
            with (output/'cardiac_function.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['structure','EDV_mL','ESV_mL','SV_mL','EF_percent','valid','reason'])
                for name,row in statistics['cavities'].items():
                    writer.writerow([name,*(row[k] for k in ('EDV_mL','ESV_mL','SV_mL','EF_percent','valid','reason'))])
            (output/'README.txt').write_text(
                'ACDC Axial B16: 0 background, 1 RV, 2 MYO, 3 LV. RAS millimeters.\n'
                'Preprocessing, reconstruction, MPR, statistics and meshes: PYNQ ARM. Network: PYNQ PL.\n'
                'Smoothed geometry is a display derivative. Volumes/EF use native labels.\n'
                'Internal gray/label NPY files are not included in the public ZIP.\n'
                '工程教学与算法演示，不作为临床诊断依据。\n',encoding='utf-8')
            atomic_json(output/'performance.json',dict(schema='pld-acdc-appliance-performance-v2',
                application_sha256=self.application_sha,
                numerical_seconds=report['numerical_seconds'],phases=phase_reports,
                processing_begin_monotonic=began,numerical_begin_monotonic=report['numerical_begin_monotonic'],
                before_export_seconds=time.perf_counter()-began,upload_seconds=upload_seconds,
                model_sha256=self.backend.model_sha,runtime_sha256=self.backend.runtime_sha,
                backend=self.backend.kind,synthetic=self.backend.synthetic,
                generator='pynq_ps',renderer='browser_webgl',mesh_binary=self.surfaces.info))
            # Public assets are explicit. Native MPR NPY volumes remain server-side.
            assets={p.relative_to(output).as_posix():(file_sha(p) if getattr(self.numerics,'verify',True) else None) for p in output.rglob('*')
                    if p.is_file() and p.suffix!='.npy' and p.name not in ('status.json','result.zip','manifest.json')}
            atomic_json(output/'manifest.json',dict(schema='pld-acdc-appliance-result-v2',
                application_sha256=self.application_sha,
                model_sha256=self.backend.model_sha,runtime_sha256=self.backend.runtime_sha,
                backend=self.backend.kind,synthetic=self.backend.synthetic,files=assets,
                generator='pynq_ps',inference='pynq_pl' if self.backend.kind=='pl' else self.backend.kind))
            tick=time.perf_counter()
            zip_sha=write_zip(output,sorted([*assets,'manifest.json']),output/'result.zip',check_cancel,verify=getattr(self.numerics,'verify',True))
            server_seconds=upload_seconds+time.perf_counter()-began
            completion=dict(application_sha256=self.application_sha,server_upload_to_assets_seconds=server_seconds,zip_seconds=time.perf_counter()-tick,zip_begin_monotonic=tick,
                zip_sha256=zip_sha,browser_interactive_seconds=None,
                scope='server upload read to both-phase MPR/meshes/ZIP; client display and transfer measured separately')
            atomic_json(output/'completion.json',completion)
            update('assets_ready',completion=completion,completed_slices=len(report['records']),total_slices=len(report['records']))
            return completion
        except InterruptedError:
            update('cancelled');raise
        except Exception as exc:
            update('failed',error=f'{type(exc).__name__}: {exc}');raise
