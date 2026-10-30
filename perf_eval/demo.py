"""Compile verified report files into portable display-only comparison records."""
import json
import math
from pathlib import Path
from .core import digest,summarize,positive
from .report import compare,precision_change,read
from .evidence import utilization,power_report,integrate_power


def prepare_record(config,root):
    from display_web.demo_metrics import FIELDS,record_digest
    root=Path(root)
    def path(value):return (root/value).resolve()
    cpu_path,pl_path=path(config['cpu']),path(config['fpga'])
    cpu,pl=read(cpu_path),read(pl_path)
    if cpu.get('backend') not in ('cpu-int8','cpu-fp32') or pl.get('backend')!='pl' or pl.get('hardware_execution') is not True:
        raise ValueError('CPU/software and PL reports required in that order')
    for report in (cpu,pl):
        if report.get('complete') is not True or type(report.get('synthetic')) is not bool:
            raise ValueError('complete benchmark and explicit input scope required')
        measured=summarize([f['inference_seconds'] for f in report['frames']])
        for key in ('mean_seconds','median_seconds','p95_seconds','active_fps'):
            summary_value=positive(report['summary']['inference'][key],'inference summary')
            if not math.isclose(summary_value,measured[key],rel_tol=1e-6,abs_tol=1e-9):
                raise ValueError('benchmark summary differs from measured frame times')
    comparison=compare(cpu_path,pl_path)  # Checks artifact hashes, model, inputs and valid regions.
    runtime=pl['provenance'].get('runtime_manifest_sha256') or pl['provenance'].get('bit_sha256')
    model_sha=pl['provenance']['spec_sha256']
    sources=[]
    def source(role,p):
        sources.append(dict(role=role,sha256=digest(p)))
    source('cpu_benchmark',cpu_path);source('fpga_benchmark',pl_path)
    if cpu.get('synthetic') is not pl.get('synthetic'):
        raise ValueError('synthetic and real benchmark reports cannot be paired')
    result=dict(model=pl['model'],model_sha256=model_sha,runtime_sha256=runtime,
                dataset_sha256=pl['dataset_sha256'],label_scope=pl['label_scope'],
                synthetic=bool(cpu.get('synthetic') or pl.get('synthetic')),
                sources=sources,checks=comparison,accuracy_scope='unmeasured; no paired GT reports')
    for key,run in (('cpu',cpu),('fpga',pl)):
        frames=run['frames']
        if not frames:raise ValueError('measured frames required')
        wall=positive(run['summary']['processing']['wall_seconds'],'loop wall time')
        timings=summarize([f['inference_seconds'] for f in frames])
        processing=summarize([f['processing_seconds'] for f in frames],wall)
        metrics=dict.fromkeys(FIELDS)
        metrics.update(processing_fps=processing['active_fps'],median_ms=1000*timings['median_seconds'],
                       p95_ms=1000*timings['p95_seconds'],throughput_fps=processing['wall_throughput_fps'])
        environment=run.get('environment',{})
        result[key]=dict(metrics=metrics,frames=len(frames),baseline=run.get('baseline',run['backend']),
            platform=str(environment.get('machine','unknown'))+' / '+str(environment.get('os','unknown')),
            power_method='unmeasured',timing_scope=run.get('timing_scope','inference / processing / wall loop'))
        if key=='fpga' and all(isinstance(f.get('dma'),dict) for f in frames):
            byte_count=sum(positive(f['dma']['total_packet_bytes'],'DMA bytes',zero=True) for f in frames)
            metrics['ddr_MB_s']=byte_count/timings['active_seconds']/1e6
    def bound(item):
        if item.get('model_sha256')!=model_sha or item.get('runtime_sha256')!=runtime:
            raise ValueError('build evidence does not match model/runtime')
        p=path(item['path'])
        if digest(p)!=item.get('sha256'):raise ValueError('build report digest mismatch')
        return p
    if config.get('utilization'):
        p=bound(config['utilization']);report=utilization(p);source('routed_utilization',p)
        if not report.get('design_state') or 'rout' not in report['design_state'].lower():
            raise ValueError('routed utilization report required')
        mapping={'LUT':'lut','FF':'ff','BRAM36_equivalent':'bram','DSP':'dsp'}
        result['fpga']['resource_percent']={}
        for name,field in mapping.items():
            value=report['resources'].get(name)
            if value:
                if value['used']>value['available']:raise ValueError('routed resource usage exceeds device')
                result['fpga']['metrics'][field]=value['used']
                result['fpga']['resource_percent'][field]=100*value['used']/value['available']
        result['resource_scope']='routed build report; supplied exact model/runtime binding'
    for backend,item in config.get('power',{}).items():
        if backend not in ('cpu','fpga'):raise ValueError('unknown power backend')
        if item.get('kind')=='chip_estimate':
            if backend!='fpga':raise ValueError('Vivado chip estimate applies to FPGA only')
            p=bound(item);report=power_report(p)
            value=report['values'].get('total_on_chip_w')
            method='Vivado chip estimate (not board W)'
        elif item.get('kind')=='board_samples':
            p=path(item['path'])
            expected=digest(cpu_path if backend=='cpu' else pl_path)
            if item.get('benchmark_sha256')!=expected or item.get('sha256')!=digest(p):
                raise ValueError('power sample/benchmark binding mismatch')
            report=integrate_power(read(p))
            if report['frames']!=result[backend]['frames']:raise ValueError('power interval completion count differs')
            value=report['mean_watts'];method='measured board input / '+report['instrument']
        else:raise ValueError('power must identify chip estimate or measured board samples')
        result[backend]['metrics']['power_w']=value;result[backend]['power_method']=method
        source(backend+'_power',p)
    if config.get('workload'):
        item=config['workload'];p=path(item['path']);work=read(p)
        if digest(p)!=item.get('sha256') or work.get('model_sha256')!=model_sha or not work.get('method'):
            raise ValueError('operation count provenance missing')
        macs=positive(work['macs_per_frame'],'MACs per input')
        for key,run in (('cpu',cpu),('fpga',pl)):
            result[key]['metrics']['gmac_s']=macs*len(run['frames'])/sum(f['inference_seconds'] for f in run['frames'])/1e9
        source('model_operation_count',p)
    if config.get('accuracy'):
        a=config['accuracy']
        if a.get('cpu_benchmark_sha256')!=digest(cpu_path) or a.get('fpga_benchmark_sha256')!=digest(pl_path):
            raise ValueError('accuracy/model association must be declared for both benchmark reports')
        first,second=path(a['cpu']),path(a['fpga'])
        change=precision_change(first,second)
        score_a,score_b=read(first),read(second)
        if score_a['label_scope']!=pl['label_scope']:raise ValueError('accuracy coordinate scope differs from benchmark')
        for field,metric in (('dice','foreground_macro_dice'),('iou','foreground_macro_iou')):
            row=change['mean_per_image'][metric]
            result['cpu']['metrics'][field]=row['baseline'];result['fpga']['metrics'][field]=row['candidate']
            result['fpga']['metrics'][field+'_delta_pp']=row['delta_percentage_points']
        result['accuracy_scope']=str(score_a['dataset_split'])+'; mean per image; N='+str(score_a['pair_count'])
        source('cpu_labeled_accuracy',first);source('fpga_labeled_accuracy',second)
    result['record_sha256']=record_digest(result)
    return result


def prepare_catalog(config_path,output):
    from display_web.demo_metrics import SCHEMA,validate_catalog
    p=Path(config_path).resolve();config=read(p)
    if config.get('schema')!='pld-demo-metrics-input-v1' or not isinstance(config.get('records'),list):
        raise ValueError('invalid comparison import configuration')
    doc=dict(schema=SCHEMA,records=[prepare_record(row,p.parent) for row in config['records']])
    validate_catalog(doc)
    # An existing evidence snapshot is never overwritten.
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    with Path(output).open('x',encoding='utf-8') as stream:
        json.dump(doc,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return doc
