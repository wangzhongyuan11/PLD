"""Portable recorded CPU/FPGA comparison; no inference or hardware imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import re

SCHEMA = 'pld-demo-metrics-v1'
FIELDS = ('processing_fps','median_ms','p95_ms','throughput_fps','gmac_s','power_w',
          'lut','ff','bram','dsp','ddr_MB_s','dice','iou','dice_delta_pp','iou_delta_pp')


def record_digest(record):
    data = {k:v for k,v in record.items() if k != 'record_sha256'}
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def validate_catalog(doc):
    if not isinstance(doc,dict) or doc.get('schema') != SCHEMA or not isinstance(doc.get('records'),list):
        raise ValueError('invalid comparison catalog')
    identities = set()
    for r in doc['records']:
        if not isinstance(r,dict) or r.get('model') not in ('skin','camus','acdc'):
            raise ValueError('invalid comparison model')
        identity = tuple(r.get(k) for k in ('model','model_sha256','runtime_sha256'))
        if not all(_sha(x) for x in identity[1:]) or identity in identities:
            raise ValueError('ambiguous comparison identity')
        identities.add(identity)
        if r.get('record_sha256') != record_digest(r):
            raise ValueError('comparison content changed')
        if type(r.get('synthetic')) is not bool or not isinstance(r.get('sources'),list) or len(r['sources']) < 2:
            raise ValueError('comparison provenance missing')
        if not _sha(r.get('dataset_sha256')) or not isinstance(r.get('label_scope'),str) or not isinstance(r.get('accuracy_scope'),str):
            raise ValueError('comparison scope missing')
        checks=r.get('checks')
        if not isinstance(checks,dict) or type(checks.get('integer_head_all_exact')) is not bool:
            raise ValueError('comparison checks missing')
        ratio=checks.get('inference_acceleration_cpu_int8_over_pl')
        if ratio is not None and (type(ratio) not in (int,float) or not math.isfinite(ratio) or ratio<=0):
            raise ValueError('invalid acceleration ratio')
        if ratio is not None and (not checks['integer_head_all_exact'] or checks.get('acceleration_blocked_reasons')):
            raise ValueError('acceleration ratio violates comparison checks')
        for source in r['sources']:
            if not isinstance(source,dict) or not _sha(source.get('sha256')):
                raise ValueError('invalid evidence digest')
        for backend in ('cpu','fpga'):
            row = r.get(backend)
            if not isinstance(row,dict) or type(row.get('frames')) is not int or row['frames'] < 1:
                raise ValueError('comparison sample count missing')
            if any(not isinstance(row.get(key),str) for key in ('baseline','platform','power_method','timing_scope')):
                raise ValueError('baseline and measurement scope missing')
            metrics = row.get('metrics')
            if not isinstance(metrics,dict) or set(metrics) != set(FIELDS):
                raise ValueError('comparison metric fields missing')
            for name,value in metrics.items():
                if value is None:continue
                if type(value) not in (int,float) or not math.isfinite(value):
                    raise ValueError('comparison values must be finite')
                if name.endswith('_delta_pp'):
                    if not -100 <= value <= 100:raise ValueError('invalid accuracy change')
                elif value < 0 or (name in ('dice','iou') and value > 1):
                    raise ValueError('invalid metric range')
            percentages=row.get('resource_percent',{})
            if not isinstance(percentages,dict):raise ValueError('invalid resource percentages')
            for value in percentages.values():
                if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=100:
                    raise ValueError('invalid resource percentage')
    return doc


def catalog_status(path=None):
    source = path or os.environ.get('PLD_DEMO_METRICS')
    if not source:return dict(status='no_evidence',records=[])
    try:
        p = Path(source)
        with p.open('rb') as stream:
            data = stream.read(2*1024*1024+1)
        if len(data) > 2*1024*1024:raise ValueError('catalog too large')
        doc = validate_catalog(json.loads(data))
        return dict(status='ready' if doc['records'] else 'no_evidence',records=doc['records'])
    except (OSError,ValueError,TypeError,KeyError,RecursionError):
        # The viewer must not expose private report paths or break image display.
        return dict(status='invalid_evidence',records=[])


def for_task(info,path=None):
    status = catalog_status(path)
    matches = [r for r in status['records'] if r['model'] == info['model'].lower()
               and r['model_sha256'] == info.get('model_sha256')
               and r['runtime_sha256'] == info.get('runtime_sha256')]
    return dict(status='matched' if len(matches)==1 else status['status'] if status['status']!='ready' else 'identity_mismatch',
                record=matches[0] if len(matches)==1 else None)


def comparison_rows(record):
    """The HDMI and web viewers use the same numeric fields and units."""
    def fmt(backend,name,digits=2):
        value = record[backend]['metrics'].get(name) if record else None
        if value is not None and name in ('lut','ff','bram','dsp'):
            return f'{value:g}'
        return '--' if value is None else f'{value:,.{digits}f}'
    unit = 'slice/s' if record and record['model']=='acdc' else 'frame/s'
    rows=[]
    definitions=[('Active processing ('+unit+')',('processing_fps',)),
                 ('Network median / P95 (ms)',('median_ms','p95_ms')),
                 ('Wall loop throughput ('+unit+')',('throughput_fps',)),
                 ('Effective throughput (GMAC/s)',('gmac_s',)),
                 ('Power (W; see method below)',('power_w',)),
                 ('Build LUT / FF',('lut','ff')),('Build BRAM36 / DSP',('bram','dsp')),
                 ('DMA packet estimate (MB/s)',('ddr_MB_s',)),
                 ('GT Dice / IoU',('dice','iou')),('GT change Dice / IoU (pp)',('dice_delta_pp','iou_delta_pp'))]
    for label,fields in definitions:
        values=[' / '.join(fmt(b,k,4 if k in ('dice','iou') else 2) for k in fields) for b in ('cpu','fpga')]
        if fields[0] in ('lut','bram'):values[0]='N/A (software)'
        rows.append((label,*values))
    return rows


def draw_comparison(screen,selection,y=0):
    """Readable 640x480 overlay, leaving ACDC's top controls intact at y=48."""
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    image=Image.fromarray(screen).convert('RGB');draw=ImageDraw.Draw(image)
    def font(size):
        for name in ('DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf'):
            try:return ImageFont.truetype(name,size)
            except OSError:pass
        return ImageFont.load_default()
    small,regular,title=font(11),font(12),font(15)
    def text(x,top,value,style=small,color=(227,238,244),width=None):
        value=str(value).encode('ascii','replace').decode()
        width=width or image.width-x-8
        while value and draw.textbbox((0,0),value,font=style)[2]>width:value=value[:-1]
        draw.text((x,top),value,font=style,fill=color)
    record=(selection or {}).get('record')
    draw.rectangle((0,y,image.width,image.height),fill=(8,22,34))
    text(10,y+6,'CPU / SOFTWARE vs FPGA',title,(125,232,206))
    if record:
        label='SYNTHETIC FIXTURE - NOT BENCHMARK RESULTS' if record['synthetic'] else 'RECORDED BENCHMARK - NOT A LIVE COUNTER'
        text(10,y+28,label,color=(242,203,137))
        text(10,y+44,f"{record['model'].upper()} | N CPU={record['cpu']['frames']} FPGA={record['fpga']['frames']} | model {record['model_sha256'][:10]} / runtime {record['runtime_sha256'][:10]}")
    else:
        text(10,y+28,'No matching verified report. All missing values remain unmeasured.',color=(242,203,137))
        text(10,y+44,'Import paired CPU/FPGA reports; model and runtime identities must match.')
    text(10,y+64,'Metric / scope',regular);text(315,y+64,'CPU / software',regular);text(475,y+64,'FPGA',regular)
    for i,(label,cpu,pl) in enumerate(comparison_rows(record)):
        at=y+86+i*21
        draw.line((8,at-2,image.width-8,at-2),fill=(35,57,72))
        text(10,at,label,width=300);text(315,at,cpu,width=150);text(475,at,pl,width=155)
    foot=y+302
    if record:
        text(10,foot,record['cpu'].get('baseline','Software baseline'),color=(165,194,208))
        text(10,foot+16,'Power: CPU '+record['cpu'].get('power_method','unmeasured')+'; FPGA '+record['fpga'].get('power_method','unmeasured'))
        text(10,foot+32,'DMA packets exclude CPU copies/cache/HDMI; not total DDR bandwidth.')
        text(10,foot+48,'GT: '+record.get('accuracy_scope','unmeasured; output agreement is not accuracy'))
        percentages=record['fpga'].get('resource_percent',{})
        usage=' | '.join(name.upper()+(' --' if percentages.get(name) is None else f' {percentages[name]:.1f}%') for name in ('lut','ff','bram','dsp'))
        text(10,foot+64,'Build utilization: '+usage)
        text(10,foot+80,'Network latency includes PL/DMA; excludes init, decode, labels and exports.')
    else:
        text(10,foot,'-- = unmeasured | LUT/FF/BRAM/DSP are static build utilization.')
        text(10,foot+16,'A single-sample benchmark is not sustained performance evidence.')
    text(10,image.height-31,'Tap below the top controls to return to images.',color=(125,232,206))
    text(10,image.height-16,'Teaching and algorithm demo only - not for clinical diagnosis',color=(242,203,137))
    return np.asarray(image).copy()
