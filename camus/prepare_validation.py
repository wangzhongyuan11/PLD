"""Freeze only the 200 exported development endpoints and their raw NIfTI inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
import nibabel as nib
from .preprocess import preprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(dataset, candidate, output):
    dataset, candidate, output = map(Path, (dataset, candidate, output))
    output.mkdir(parents=True, exist_ok=False)
    allowed = set((dataset/'database_split/subgroup_validation.txt').read_text().split())
    cases = sorted((candidate/'evaluations/int8/cases').glob('*_int8.npz'))
    if len(cases) != 200 or len({p.name.split('_')[0] for p in cases}) != 50:
        raise ValueError('requires the frozen 50-patient/200-frame dev export')
    if any(p.name.split('_')[0] not in allowed for p in cases):
        raise ValueError('export crosses the official development split')
    rows = []
    for index, case in enumerate(cases):
        stem = case.name.removesuffix('_int8.npz'); patient, view, phase = stem.split('_')
        source = dataset/'database_nifti'/patient/(stem+'.nii.gz')
        ground = source.with_name(stem+'_gt.nii.gz')
        prep = preprocess(source)
        with np.load(case, allow_pickle=False) as payload:
            expected = {k: payload[k] for k in ('logits_int8','labels','gt_labels')}
        with np.load(case.with_name(stem+'_processed.npz'), allow_pickle=False) as payload:
            expected['processed'] = payload['labels']
        gt = np.squeeze(np.asarray(nib.load(str(ground)).dataobj)).T
        if not np.array_equal(gt, expected['gt_labels']):
            raise ValueError('original GT differs from dev export: '+stem)
        expected['input_int8'] = prep.input_int8
        image = output/(stem+'.nii.gz'); shutil.copyfile(source, image)
        archive = output/(stem+'.npz'); np.savez_compressed(archive, **expected)
        cfg = (source.parent/('Info_'+view+'.cfg')).read_text()
        quality = next(line.split(':',1)[1].strip() for line in cfg.splitlines() if line.strip().startswith('ImageQuality:'))
        rows.append(dict(stem=stem, input=image.name, expected=archive.name,
                         input_sha256=sha(image), expected_sha256=sha(archive), original_gt_sha256=sha(ground),
                         native_hw=list(prep.raw_gray.shape), spacing_yx_mm=prep.source.spacing_yx_mm, quality=quality))
        if index % 25 == 0: print('PREPARED',index+1,flush=True)
    report = dict(schema='camus-board-dataset-v1', scope='official dev only; no sealed test access',
                  patients=50, frames=200, split_sha256=sha(dataset/'database_split/subgroup_validation.txt'), cases=rows)
    (output/'dataset.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(output.iterdir()): tar.add(p, arcname=p.name, recursive=False)
    print(json.dumps(dict(dataset=str(archive),sha256=sha(archive),frames=200)),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.dataset,a.candidate,a.output)


if __name__=='__main__':main()
