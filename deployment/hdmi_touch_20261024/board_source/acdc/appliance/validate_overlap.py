"""Paired fresh P6 numerical runs with per-slice overlap off/on.

Run only with the managed Web/engine stopped and the FPGA lease free. Both
variants retain dual-core final native restoration; the changed variable is
input preparation / PL / plane restoration overlap, not model arithmetic.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from acdc.backend import SpatialBackend
from acdc.contract import atomic_json,file_sha
from acdc.model import load_bundle
from acdc.appliance.assets import load,selftest
from acdc.appliance.identity import verify
from acdc.appliance.native import load_native
from acdc.appliance.pipeline import NumericalPipeline
from acdc.appliance.volume import load_volume


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--app-sha',required=True)
    parser.add_argument('--pair-root',type=Path,required=True);parser.add_argument('--oracle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.release.resolve();config,_=verify(root,args.app_sha)
    args.output.mkdir(parents=True,exist_ok=False)
    model=load_bundle(root/config['model'],expected_sha=config['model_sha256'])
    table=load(model,root/config['assets'],config['assets_sha256']);kernel=load_native()
    backend=SpatialBackend(model,root);reports={}
    try:
        selftest(backend,root/config['assets'])
        pipeline=NumericalPipeline(backend,kernel,table,binding=model.implementation_binding,policy=config['geometry_policy'])
        images=[args.pair_root/(phase+'.nii.gz') for phase in ('ED','ES')]
        # Explicitly warm file/import paths equally before either timed run.
        for image in images:
            volume=load_volume(image,kernel,policy=config['geometry_policy'],binding=model.implementation_binding)
            del volume
        expected=json.loads(args.oracle.read_text())
        for name,overlap in (('serial',False),('overlap',True)):
            report=pipeline.run(*images,args.output/name,overlap=overlap)
            values=[(r['phase'],r['center_index'],r['input_sha256'],r['head_sha256']) for r in report['records']]
            if values!=[tuple(row) for row in expected['records']]:raise AssertionError('independent oracle differs')
            reports[name]=dict(numerical_seconds=report['numerical_seconds'],phases=report['phases'],
                fresh_physical_pl_heads=len(values),input_and_head_hashes_equal_independent_oracle=True)
        for phase in ('ED','ES'):
            a=nib.load(args.output/'serial/labels'/(phase+'.nii.gz'))
            b=nib.load(args.output/'overlap/labels'/(phase+'.nii.gz'))
            np.testing.assert_array_equal(a.dataobj,b.dataobj);np.testing.assert_array_equal(a.affine,b.affine)
        report=dict(schema='pld-acdc-appliance-overlap-board-v2',complete=True,
            application_sha256=args.app_sha,model_sha256=backend.model_sha,runtime_sha256=backend.runtime_sha,
            oracle_sha256=file_sha(args.oracle),runs=reports,native_labels_affines_equal=True,
            numerical_speedup=reports['serial']['numerical_seconds']/reports['overlap']['numerical_seconds'],
            scope='paired real P6 numerics, warm file/import caches, same dual-core final restoration; no meshes/upload/browser')
        atomic_json(args.output/'audit.json',report);print(json.dumps(report))
    finally:backend.close()


if __name__=='__main__':main()
