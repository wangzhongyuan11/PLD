"""Artifact comparison, label accuracy and portable reports."""
import csv
import hashlib
import html
import json
from pathlib import Path
from .core import digest


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _artifact(root,name,expected):
    path=(root/name).resolve()
    if not path.is_relative_to(root) or digest(path)!=expected:raise ValueError('artifact path/hash mismatch')
    return path


def compare(first,second):
    import numpy as np
    from .metrics import array_difference
    a,b=read(first),read(second)
    for row in (a,b):
        if row.get('schema')!='pld-benchmark-v1' or not row.get('complete'):raise ValueError('complete benchmark reports required')
    for key in ('model','dataset_sha256','label_scope'):
        if a[key]!=b[key]:raise ValueError('comparison mismatch: '+key)
    if a['provenance']['spec_sha256']!=b['provenance']['spec_sha256']:raise ValueError('model spec mismatch')
    if set(a['artifacts'])!=set(b['artifacts']):raise ValueError('case set differs')
    rows=[];all_exact=True
    for key in a['artifacts']:
        x,y=a['artifacts'][key],b['artifacts'][key]
        if x['valid_sha256']!=y['valid_sha256']:raise ValueError('valid region mismatch')
        roots=[Path(first).resolve().parent,Path(second).resolve().parent]
        heads=[];labels=[]
        for root,item in zip(roots,(x,y)):
            heads.append(np.load(_artifact(root,item['head'],item['head_file_sha256']),allow_pickle=False))
            labels.append(np.load(_artifact(root,item['labels'],item['labels_file_sha256']),allow_pickle=False))
        head_diff=array_difference(*heads) if heads[0].dtype==heads[1].dtype and heads[0].dtype.kind in 'iu' else None
        label_diff=array_difference(*labels);all_exact=all_exact and head_diff is not None and head_diff['exact']
        rows.append(dict(case_id=key,head_difference=head_diff,label_difference=label_diff))
    problems=[]
    if {a['backend'],b['backend']}!={'cpu-int8','pl'}:problems.append('automatic acceleration ratio limited to CPU-INT8 versus PL same integer task')
    if a['environment']['machine']!=b['environment']['machine'] or a['environment']['os']!=b['environment']['os']:problems.append('different host architectures/OS; report cross-platform times separately')
    if not a['environment'].get('host_fingerprint') or a['environment'].get('host_fingerprint')!=b['environment'].get('host_fingerprint'):problems.append('reports not identified as the same host; use cross-platform comparison without an automatic FPGA speedup claim')
    if not all_exact:problems.append('heads not bit-exact')
    if any(not r.get('repeated_output_consistent',False) for r in a['frames']+b['frames']):problems.append('a repeated output changed')
    signatures=lambda d:[(r['case_id'],r['repetition'],r['input_sha256']) for r in d['frames']]
    if signatures(a)!=signatures(b):problems.append('input order/hash/repetition differs')
    ratio=None
    if not problems:
        cpu=a if a['backend']=='cpu-int8' else b;pl=b if a['backend']=='cpu-int8' else a
        ratio=cpu['summary']['inference']['mean_seconds']/pl['summary']['inference']['mean_seconds']
    return dict(kind='benchmark_comparison',model=a['model'],case_count=len(rows),cases=rows,
        source_report_sha256=[digest(first),digest(second)],integer_head_all_exact=all_exact,
        inference_acceleration_cpu_int8_over_pl=ratio,acceleration_blocked_reasons=problems,
        accuracy_warning='label agreement is not ground-truth accuracy; FP32/INT8 Dice changes require labeled data')


def score_pairs(manifest):
    import numpy as np
    from .metrics import confusion,scores
    from PIL import Image
    path=Path(manifest).resolve();doc=read(path);classes=doc['classes'];rows=[];pooled=None
    if not doc.get('pairs'):raise ValueError('no labeled pairs')
    seen=set();reference_members=[]
    def load(name):
        member=(path.parent/name).resolve()
        if member.suffix=='.npy':return np.load(member,allow_pickle=False)
        with Image.open(member) as image:return np.array(image)
    for pair in doc['pairs']:
        identifier=pair['id']
        if identifier in seen:raise ValueError('duplicate pair ID')
        seen.add(identifier)
        prediction,truth=load(pair['prediction']),load(pair['truth'])
        valid=load(pair['valid']) if pair.get('valid') else None
        reference_members.append(dict(id=identifier,truth_sha256=digest((path.parent/pair['truth']).resolve()),valid_sha256=digest((path.parent/pair['valid']).resolve()) if pair.get('valid') else None))
        mat=confusion(prediction,truth,classes,valid)
        pooled=mat.copy() if pooled is None else pooled+mat
        rows.append(dict(id=identifier,metrics=scores(mat)))
    macro={}
    for metric in ('foreground_macro_dice','foreground_macro_iou'):
        values=[row['metrics'][metric] for row in rows if row['metrics'][metric] is not None]
        macro[metric]=sum(values)/len(values) if values else None
        macro[metric+'_image_count']=len(values)
    return dict(kind='labeled_accuracy',evidence='computed_on_user_labels',source_manifest_sha256=digest(path),
        reference_set_sha256=hashlib.sha256(json.dumps(sorted(reference_members,key=lambda x:x['id']),sort_keys=True).encode()).hexdigest(),
        classes=classes,dataset_split=doc.get('dataset_split','unspecified; do not assume independent test set'),
        label_scope=doc.get('label_scope','unspecified'),pair_count=len(rows),per_image=rows,
        mean_per_image=macro,pooled_pixel_metrics=scores(pooled),
        warning='macro per-image and pooled-pixel scores have different weighting; no resizing or display-label remapping performed')


def bundle(runs,evidence,output):
    root=Path(output).resolve()
    if root.exists():raise ValueError('report output must be a new directory')
    data=[read(p) for p in runs];extra=[read(p) for p in evidence]
    for run in data:
        if run.get('schema')!='pld-benchmark-v1' or not run.get('complete'):raise ValueError('complete benchmark runs required')
    root.mkdir(parents=True)
    headers=['model','backend','frames','inference_mean_ms','inference_p95_ms','inference_active_fps','processing_mean_ms','loop_throughput_fps','initialization_s','cold_forward_s','synthetic']
    table=[]
    for run in data:
        a,b=run['summary']['inference'],run['summary']['processing']
        table.append([run['model'],run['backend'],a['frames'],a['mean_seconds']*1000,a['p95_seconds']*1000,a['active_fps'],b['mean_seconds']*1000,b['wall_throughput_fps'],run['model_initialization_seconds'],run['first_forward_seconds'],run['synthetic']])
    with (root/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(headers);writer.writerows(table)
    note='本报告区分实测、历史导入、工具估算。CPU NumPy参考不是充分优化的ARM基线；合成输入不用于医学精度结论。FPGA/整板功耗与DDR总带宽缺失时保留待测，不能填零。'
    md=['# PLD 性能与证据汇总','',note,'','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    md+=['| '+' | '.join(str(round(x,6)) if isinstance(x,float) else str(x) for x in row)+' |' for row in table]
    md+=['','推理FPS按实际处理帧数/推理累计时间计算；loop吞吐率按完整测量循环墙钟计算。P95为本次样本分位数。不同平台与不同计时范围不自动生成加速比。',
         '', '## 附加证据', '', '```json',json.dumps(extra,ensure_ascii=False,indent=2), '```']
    (root/'report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    display_headers=['模型','后端','测量帧数','推理均值 ms','推理 P95 ms','推理 FPS','处理均值 ms','循环吞吐 FPS','初始化 s','首次前向 s','合成输入']
    head=''.join('<th>'+html.escape(x)+'</th>' for x in display_headers)
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(round(x,6)) if isinstance(x,float) else str(x))+'</td>' for x in row)+'</tr>' for row in table)
    page='<!doctype html><meta charset="utf-8"><title>PLD 性能评测</title><style>body{font:16px system-ui;margin:32px;color:#173646}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #bbcbd1;padding:9px}th{background:#e8f4f4}pre{white-space:pre-wrap}</style><h1>PLD 性能评测</h1><p>'+html.escape(note)+'</p><table><tr>'+head+'</tr>'+body+'</table><h2>附加证据与待测项</h2><pre>'+html.escape(json.dumps(extra,ensure_ascii=False,indent=2))+'</pre>'
    (root/'report.html').write_text(page,encoding='utf-8')
    (root/'report.json').write_text(json.dumps(dict(schema='pld-evaluation-bundle-v1',runs=data,evidence=extra),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return root/'report.html'


def precision_change(baseline,candidate):
    a,b=read(baseline),read(candidate)
    for row in (a,b):
        if row.get('kind')!='labeled_accuracy':raise ValueError('labeled accuracy reports required')
    for key in ('reference_set_sha256','classes','label_scope','dataset_split'):
        if not a.get(key) or a[key]!=b[key]:raise ValueError('accuracy comparison mismatch: '+key)
    change={}
    for key in ('foreground_macro_dice','foreground_macro_iou'):
        x,y=a['mean_per_image'][key],b['mean_per_image'][key]
        change[key]=dict(baseline=x,candidate=y,delta_percentage_points=None if x is None or y is None else 100*(y-x))
    classes=[]
    for x,y in zip(a['pooled_pixel_metrics']['per_class'],b['pooled_pixel_metrics']['per_class']):
        classes.append(dict(class_id=x['class_id'],dice_delta_percentage_points=None if x['dice'] is None or y['dice'] is None else 100*(y['dice']-x['dice']),
                            iou_delta_percentage_points=None if x['iou'] is None or y['iou'] is None else 100*(y['iou']-x['iou'])))
    return dict(kind='algorithm_precision_change',scope='same labeled set and label coordinate scope; candidate minus baseline',
        reference_set_sha256=a['reference_set_sha256'],mean_per_image=change,pooled_class_changes=classes,
        source_reports_sha256=[digest(baseline),digest(candidate)],
        warning='development split is not independent validation; label agreement without GT is not accuracy')
