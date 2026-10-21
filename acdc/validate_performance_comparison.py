"""Audit a historical matched-input CPU-reference / remote-PL comparison.

This does not rerun inference or claim a controlled hardware speedup experiment.
The CPU implementation is the FP64 exact-integer witness, not an optimized
INT8/FP32 inference library or the PYNQ ARM baseline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np

from .contract import ContractError,atomic_json,file_sha


def paired_records(cpu,pl):
    def index(records):
        result={}
        for row in records:
            key=(row['phase'],row['center_index'])
            if key in result or row['reused'] or row['synthetic']:
                raise ContractError('duplicate, reused or synthetic comparison sample')
            if not math.isfinite(row['seconds']) or row['seconds']<=0:
                raise ContractError('invalid sample duration')
            result[key]=row
        return result
    first,second=index(cpu),index(pl)
    if not first or first.keys()!=second.keys():raise ContractError('different/incomplete sample sets')
    for key,a in first.items():
        b=second[key]
        if a['backend']!='reference_torch' or b['backend']!='pl':
            raise ContractError('wrong comparison backend')
        if any(a[name]!=b[name] for name in ('input_sha256','head_sha256')):
            raise ContractError('input/head identity mismatch at '+str(key))
    return len(first)


def run(cpu_root,pl_root,output,model_sha,runtime_sha):
    cpu_root,pl_root=Path(cpu_root),Path(pl_root);output=Path(output)
    if output.exists():raise ContractError('choose a new comparison output')
    first={p.parent.name:p for p in cpu_root.glob('patient*/performance.json')}
    second={p.parent.name:p for p in pl_root.glob('patient*/performance.json')}
    if not first or first.keys()!=second.keys():raise ContractError('different/incomplete patient sets')
    rows=[];sources=[];all_times={'cpu':[],'pl':[]};windows={'cpu':0.,'pl':0.}
    for patient in sorted(first):
        pair=[];item={'patient':patient}
        for kind,path in (('cpu',first[patient]),('pl',second[patient])):
            manifest_path=path.parent/'manifest.json';manifest=json.loads(manifest_path.read_text('utf-8'))
            if manifest['model_sha256']!=model_sha or file_sha(path)!=manifest['files']['performance.json']:
                raise ContractError('performance/model manifest mismatch')
            report=json.loads(path.read_text('utf-8'))
            expected='reference_torch' if kind=='cpu' else 'remote'
            if report['backend']!=expected or report['synthetic'] or report['resumed_count']:
                raise ContractError('comparison needs fresh real CPU-reference/remote runs')
            if report['runtime_sha256']!=(None if kind=='cpu' else runtime_sha):
                raise ContractError('runtime mismatch')
            if report['fresh_inference_count']!=len(report['records']):raise ContractError('fresh sample count mismatch')
            duration=report['fresh_inference_wall_seconds']
            if not math.isfinite(duration) or duration<=0:raise ContractError('invalid fresh wall window')
            windows[kind]+=duration;all_times[kind].extend(r['seconds'] for r in report['records'])
            item[kind+'_fresh_window_seconds']=duration
            item[kind+'_pre_export_seconds']=report['pre_export_seconds']
            pair.append(report['records'])
            sources.append(dict(path=path.as_posix(),sha256=file_sha(path),manifest_sha256=file_sha(manifest_path)))
        item['paired_slices']=paired_records(*pair);rows.append(item)
    count=sum(r['paired_slices'] for r in rows)
    metrics={kind:dict(fresh_slices=count,sum_fresh_window_seconds=windows[kind],
        fresh_window_fps=count/windows[kind],slice_p50_seconds=float(np.percentile(all_times[kind],50)),
        slice_p95_seconds=float(np.percentile(all_times[kind],95)),slice_max_seconds=max(all_times[kind]))
        for kind in ('cpu','pl')}
    report=dict(schema='pld-acdc-matched-reference-performance-v1',complete=True,
        model_spec_sha256=model_sha,runtime_manifest_sha256=runtime_sha,patients=len(rows),paired_slices=count,
        input_and_head_hash_mismatches=0,metrics=metrics,rows=rows,sources=sources,
        observed_window_throughput_ratio=metrics['pl']['fresh_window_fps']/metrics['cpu']['fresh_window_fps'],
        fresh_inference_count_this_audit=0,
        scope='historical pipeline windows on identical recorded INT8 inputs/heads; CPU exact-integer FP64 witness vs remote physical PL; includes scheduling/checkpoints, not full process wall',
        limits=['CPU is a PC reference implementation, not an optimized INT8/FP32 library or PYNQ ARM baseline',
                'Runs occurred at different times; host load and cold/warm conditions were not controlled',
                'Ratio describes these recorded pipeline windows, not isolated FPGA/CPU hardware speedup',
                'PS-DMA-PL 1000-frame 8-FPS acceptance is a separate measurement'])
    atomic_json(output,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','pl','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model-spec-sha',required=True);p.add_argument('--runtime-sha',required=True)
    a=p.parse_args();report=run(a.cpu,a.pl,a.output,a.model_spec_sha,a.runtime_sha)
    print(json.dumps({k:report[k] for k in ('patients','paired_slices','metrics','observed_window_throughput_ratio','complete')}))
