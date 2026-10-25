"""Independent offline audit of downloaded native labels and board stage times."""
import argparse
import json
from pathlib import Path
import zipfile

import nibabel as nib
import numpy as np
from acdc.contract import file_sha,atomic_json


def run(results,download,output):
    cases=json.loads((results/'cases.json').read_text())['cases']
    reference={(r['patient'],r['phase']):Path(r['result_dir']) for r in cases}
    source=json.loads((download/'audit.json').read_text());rows=[]
    for row in source['rows']:
        root=download/row['job_id'];labels={}
        with zipfile.ZipFile(root/'result.zip') as archive:
            performance=json.loads(archive.read('performance.json'))
            numerical=json.loads(archive.read('numerics.json'))
        for phase in ('ED','ES'):
            baseline=reference[(row['patient'],phase)]
            name='labels/'+phase+'.nii.gz'
            manifest=json.loads((baseline/'manifest.json').read_text())
            if file_sha(baseline/name)!=manifest['files'][name]:raise ValueError('reference labels drift')
            expected=nib.load(baseline/name);actual=nib.load(root/(phase+'.nii.gz'))
            np.testing.assert_array_equal(actual.dataobj,expected.dataobj)
            np.testing.assert_array_equal(actual.affine,expected.affine)
            if actual.header.get_xyzt_units()[0]!='mm':raise ValueError('native units lost')
            labels[phase]=dict(native_label_mismatches=0,affine_equal=True,shape=list(actual.shape))
        times=np.array([r['seconds'] for r in numerical['records']])
        rows.append(dict(patient=row['patient'],repeat=row['repeat'],job_id=row['job_id'],labels=labels,
            stages=performance,server=row['server'],initial_assets_downloaded_seconds=row['initial_assets_downloaded_seconds'],
            slice_seconds=dict(p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95)),maximum=float(times.max()))))
    values=np.array([r['initial_assets_downloaded_seconds'] for r in rows])
    report=dict(schema='pld-acdc-appliance-offline-native-audit-v2',complete=source['complete'],jobs=len(rows),
        fresh_physical_pl_heads=sum(r['physical_pl_heads'] for r in source['rows']),rows=rows,
        downloaded_assets_seconds=dict(p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),maximum=float(values.max())),
        over_40=[r['job_id'] for r in rows if r['initial_assets_downloaded_seconds']>=40],
        browser_gpu_readiness_verified=False,scope='independent native labels/affines and stage audit; browser evidence separate')
    atomic_json(output,report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('results','download','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();report=run(args.results,args.download,args.output)
    print(json.dumps({k:report[k] for k in ('complete','jobs','fresh_physical_pl_heads','downloaded_assets_seconds','over_40')}))
