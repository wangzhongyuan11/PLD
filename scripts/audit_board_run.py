"""Independently audit a completed dataset run against its exact deployment.

Checks every output/metric, report statistics and packaged source hashes. It
reports performance failures without calling a numerically correct run failed;
memory trends remain evidence for review, not an automatic no-leak verdict.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.datasets import validate_manifest
from reference.evaluation import segmentation_metrics


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit(report_path, dataset_path, deployment_path):
    raw=report_path.read_bytes()
    report=json.loads(raw)
    manifest_bytes=(dataset_path/'manifest.json').read_bytes()
    manifest=json.loads(manifest_bytes)
    if (report.get('passed') is not True or not report.get('full_test_set') or
            any(k in report for k in ('error','close_error','resizer_close_error')) or
            report['dataset_manifest_sha256']!=sha(manifest_bytes) or
            report['model_spec_sha256']!=manifest['model_spec_sha256']):
        raise ValueError('incomplete report or model/dataset mismatch')
    with tarfile.open(deployment_path) as archive:
        deployment=json.load(archive.extractfile('deployment_manifest.json'))
        files=deployment['files']
        for name,expected in files.items():
            if sha(archive.extractfile(name).read())!=expected:
                raise ValueError('corrupt deployment file: '+name)
    if (files['overlay/dma_top.bit']!=report['bit_sha256'] or
            files['overlay/dma_top.hwh']!=report['hwh_sha256'] or
            deployment['model_spec_sha256']!=report['model_spec_sha256']):
        raise ValueError('report does not identify this hardware/model deployment')
    for path,expected in report['source_hashes'].items():
        matches=[name for name in files if path.endswith('/'+name)]
        if len(matches)!=1 or files[matches[0]]!=expected:
            raise ValueError('runtime source mismatch: '+path)
    copy=report.get('copy_backend',{})
    if copy.get('kind')=='native':
        build=copy['build']
        if (build['source_sha256']!=files['board/native/tensor_copy.c'] or
                build['wrapper_sha256']!=files['board/native_copy.py'] or
                build['library_sha256']!=copy['library_sha256']):
            raise ValueError('native gather build mismatch')
    control=report.get('control_backend',{})
    if control.get('kind')=='native_packet':
        if (any(files[name]!=value for name,value in control['build']['source_hashes'].items()) or
                control['library_sha256']!=control['build']['library_sha256']):
            raise ValueError('native control build mismatch')
    tx=report.get('tx_sync_backend',{})
    resize=report.get('resize_backend',{})
    if report.get('preprocessing_backend')=='pillow_rows_neon':
        build=resize['build']
        if (resize['kind']!='pillow_rows_neon' or
                build['library_sha256']!=resize['library_sha256'] or
                build['pillow']!=report['runtime']['pillow'] or
                any(files[name]!=value for name,value in build['source_hashes'].items())):
            raise ValueError('native resize build mismatch')
    if tx.get('kind')=='xrt_tx_prefix_64' and tx['source_sha256']!=files['board/cache_sync.py']:
        raise ValueError('TX cache synchronization source mismatch')
    if (report.get('preprocessing_backend')=='parallel_bilinear_2' and
            report['resizer_sha256']!=files['reference/parallel_resize.py']):
        raise ValueError('parallel resize source mismatch')
    records=manifest['records']
    if not validate_manifest(manifest):
        raise ValueError('reference set is incomplete')
    count=len(records)
    if manifest.get('schema_version')==2:
        if (report.get('threshold')!=manifest['threshold'] or
                deployment.get('threshold')!=manifest['threshold'] or
                report.get('collection')!=manifest['collection']):
            raise ValueError('deployment/report threshold or dataset identity differs')
    threshold=manifest.get('threshold',{}).get('threshold_q',0)
    for record in records:
        path=dataset_path/record['file']
        if path.resolve().parent!=dataset_path.resolve() or sha(path.read_bytes())!=record['sha256']:
            raise ValueError('dataset artifact hash mismatch')
        with np.load(path,allow_pickle=False) as data:
            if (sha(data['head'].tobytes())!=record['head_sha256'] or
                    segmentation_metrics(data['head'][0,0]>=threshold,data['target'],data['valid'])!=record['int8']):
                raise ValueError('dataset head or threshold metric differs')
    by_id={r['id']:r for r in records}
    frames=report['frames']; seen=set()
    if len(by_id)!=count or len(frames)<count:
        raise ValueError('missing test-set coverage')
    for index,frame in enumerate(frames):
        expected=by_id[frame['id']]
        if (frame['frame']!=index+1 or frame['id']!=records[index%count]['id'] or
                frame['output_sha256']!=expected['head_sha256'] or frame['metrics']!=expected['int8']):
            raise ValueError(f'frame/reference mismatch at {index+1}')
        seen.add(frame['id'])
        for key in ('seconds','online_b_seconds'):
            if not math.isfinite(frame[key]) or frame[key]<=0:
                raise ValueError('invalid measured duration')
    if seen!=set(by_id):
        raise ValueError('test IDs not fully covered')
    jsonl=report_path.with_suffix('.jsonl')
    if jsonl.exists() and [json.loads(line) for line in jsonl.read_text().splitlines()]!=frames:
        raise ValueError('JSONL and final report differ')
    times=np.array([f['seconds'] for f in frames])
    online=np.array([f['online_b_seconds'] for f in frames])
    recomputed={'median_seconds':float(np.median(times)),'p95_seconds':float(np.percentile(times,95)),
                'inference_fps':len(times)/float(np.sum(times)),
                'dice_mean':float(np.mean([f['metrics']['dice'] for f in frames[:count]])),
                'iou_mean':float(np.mean([f['metrics']['iou'] for f in frames[:count]]))}
    if report['original_jpeg_inputs']:
        recomputed.update(online_b_median_seconds=float(np.median(online)),
                          online_b_p95_seconds=float(np.percentile(online,95)))
    for key,value in recomputed.items():
        if not math.isclose(report[key],value,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('reported statistic differs: '+key)
    stable_count=len(frames)>=1000 and report['wall_seconds']>=600
    if stable_count!=report['stability_duration_and_frames']:
        raise ValueError('stability frame/time flag differs')
    faults=report.get('faults',[])
    recovery=(len(faults)==3 and {f['fault'] for f in faults}==
              {'short_input','missing_input','reset_during_compute'} and
              all(f['recovered'] for f in faults) and report.get('post_recovery_frame_passed') is True)
    rss=np.array([f['memory']['VmRSS_KiB'] for f in frames])
    paired=rss[count:]-rss[:-count]
    memory={'rss_min_KiB':int(rss.min()),'rss_max_KiB':int(rss.max()),
            'first_100_median_KiB':float(np.median(rss[:100])),
            'last_100_median_KiB':float(np.median(rss[-100:])),
            'same_image_next_cycle_delta_median_KiB':float(np.median(paired)) if len(paired) else None,
            'hundred_frame_medians_KiB':[float(np.median(rss[i:i+100])) for i in range(0,len(rss),100)],
            'cma_free_min_KiB':min(f['memory']['CmaFree_KiB'] for f in frames),
            'cma_free_max_KiB':max(f['memory']['CmaFree_KiB'] for f in frames),
            'manual_trend_review_required':True}
    return {'numeric_provenance_verified':True,'report_sha256':sha(raw),
            'deployment_sha256':sha(deployment_path.read_bytes()),'frames':len(frames),'unique_images':len(seen),
            'collection':manifest.get('collection'),
            'primary_evaluation_complete':manifest.get('collection',{}).get('primary',False),
            'bit_sha256':report['bit_sha256'],'wall_seconds':report['wall_seconds'],**recomputed,
            'online_b_fps':len(online)/float(np.sum(online)) if report['original_jpeg_inputs'] else None,
            'stability_duration_and_frames':stable_count,'fault_recovery_verified':recovery,'memory':memory,
            'performance_met':bool(report['original_jpeg_inputs'] and len(online)/sum(online)>=1 and np.percentile(online,95)<=1),
            'scope':'B only; decode/file IO/comparison/C display excluded; performance and memory acceptance remain separate'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.report,args.dataset,args.deployment)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
