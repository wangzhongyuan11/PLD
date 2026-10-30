"""Display annotations from saved job evidence; never touches PL, DMA or exports."""
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import statistics

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _read(path):
    try:
        # A malformed or missing optional annotation must not break result display.
        if path.stat().st_size > 4 * 1024 * 1024:
            return {}
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, RecursionError):
        return {}


def _number(value, minimum=0):
    if type(value) in (int, float) and math.isfinite(value) and value > minimum:
        return float(value)
    return None


def _text(value, limit=48):
    return ''.join(c for c in str(value) if 32 <= ord(c) < 127)[:limit]


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _load_task_telemetry(folder, model, build_info=None):
    """Known schemas only. Single-job latency is never inverted into live FPS.

    Build usage is static, and only displayed for an exact model/runtime pair.
    PLD_HUD_BUILD_INFO is an optional deployment-owned JSON catalog, not a model
    manifest or a reason to bypass any existing runtime identity check.
    """
    folder = Path(folder)
    info = dict(model=model.upper(), mode='saved result', precision='unreported',
                core=None, network_seconds=None, network_scope='Network + DMA',
                task_seconds=None, threshold='unreported', area_percent=None,
                model_sha256=None, runtime_sha256=None, resources=None)
    if model == 'skin':
        job = _read(folder/'job.json').get('job', {})
        if not isinstance(job, dict) or job.get('state') != 'done':
            return info
        stats = job.get('statistics', {})
        if not isinstance(stats, dict):
            stats = {}
        mode = job.get('mode')
        info['mode'] = {'infer': 'Full image', 'demo': 'Replay / no new PL',
                        'mask': 'Supplied mask / no PL'}.get(mode, 'saved result')
        info['task_seconds'] = _number(stats.get('request_to_job_complete_seconds'))
        area = stats.get('area_percent')
        if type(area) in (int, float) and math.isfinite(area) and 0 <= area <= 100:
            info['area_percent'] = float(area)
        frame = _read(folder/'inference/frame_000001.json')
        if mode == 'infer' and frame.get('status') == 'success' and frame.get('dtype') == 'int8':
            info.update(precision='INT8', core=_text(frame.get('model_name', 'SKIN')),
                        model_sha256=frame.get('model_spec_sha256'), runtime_sha256=frame.get('bit_sha256'),
                        network_seconds=_number(frame.get('inference_seconds')))
            q = frame.get('threshold_q')
            if type(q) is int and -128 <= q <= 127:
                info['threshold'] = 'q >= ' + str(q)
    elif model == 'camus':
        report = _read(folder/'report.json')
        if report.get('schema') != 'pld-camus-result-v1':
            return info
        source = report.get('model', {})
        timings = report.get('timings_seconds', {})
        if not isinstance(source, dict) or not isinstance(timings, dict):
            return info
        info.update(mode='Full image / '+_text(report.get('backend', 'unknown'), 12),
                    threshold='Argmax / 4 classes', model_sha256=source.get('spec_sha256'),
                    runtime_sha256=report.get('runtime_sha256'),
                    task_seconds=_number(report.get('seconds_to_metadata_finalize')))
        # This task scope excludes the final metadata append (see pipeline.py).
        info['task_scope'] = 'To metadata'
        if isinstance(source.get('head'), dict) and source['head'].get('dtype') == 'int8':
            info['precision'] = 'INT8'
        if report.get('backend') == 'pl':
            info['network_seconds'] = _number(timings.get('network'))
    elif model == 'acdc':
        report = _read(folder/'numerics.json')
        if report.get('schema') != 'pld-acdc-appliance-numerics-v2' or report.get('status') != 'numerics_completed':
            return info
        info.update(mode='ED/ES volume / '+_text(report.get('backend', 'unknown'), 12),
                    model_sha256=report.get('model_sha256'), runtime_sha256=report.get('runtime_sha256'),
                    threshold='Argmax / 4 classes', network_scope='Median slice + DMA')
        completion = _read(folder/'completion.json')
        info['task_seconds'] = _number(completion.get('server_upload_to_assets_seconds'))
        info['task_scope'] = 'Upload to assets'
        records = report.get('records')
        if report.get('backend') == 'pl' and report.get('synthetic') is False and isinstance(records, list) and records:
            values = [_number(r.get('seconds')) for r in records if isinstance(r, dict)]
            if len(values) == len(records) and all(v is not None for v in values):
                info.update(precision='INT8', network_seconds=statistics.median(values))
                windows = [(r.get('inference_begin_seconds'), r.get('inference_end_seconds')) for r in records]
                if (len(windows) > 1 and all(type(a) in (int,float) and type(b) in (int,float)
                        and math.isfinite(a) and math.isfinite(b) and 0 <= a < b for a,b in windows)
                        and all(windows[i-1][1] <= windows[i][0] for i in range(1,len(windows)))):
                    info['slice_rate'] = len(windows)/(windows[-1][1]-windows[0][0])
    else:
        raise ValueError('unknown HUD model')
    catalog_path = build_info or os.environ.get('PLD_HUD_BUILD_INFO')
    if catalog_path and _sha(info['model_sha256']) and _sha(info['runtime_sha256']):
        catalog = _read(Path(catalog_path))
        if catalog.get('schema') == 'pld-hud-build-info-v1' and isinstance(catalog.get('builds'), list):
            matches = [r for r in catalog['builds'] if isinstance(r, dict)
                       and r.get('model_sha256') == info['model_sha256']
                       and r.get('runtime_sha256') == info['runtime_sha256']]
            if len(matches) == 1:
                row = matches[0]
                # Report provenance is required even though these are annotations,
                # never accelerator validation or measured live utilization.
                if _sha(row.get('utilization_report_sha256')):
                    info['resources'] = row
    return info


def load_telemetry(folder,model,build_info=None,*,metrics_path=None):
    info=_load_task_telemetry(folder,model,build_info)
    try:from .demo_metrics import for_task
    except ImportError:from demo_metrics import for_task
    info['comparison']=for_task(info,metrics_path)
    return info


def telemetry_lines(info):
    def millis(value):
        number = _number(value)
        return 'unmeasured' if number is None else f'{1000*number:.1f} ms'
    resource = info.get('resources') or {}
    def usage(name):
        used, total = resource.get(name+'_used'), resource.get(name+'_total')
        if (type(used) in (int, float) and math.isfinite(used) and used >= 0
                and _number(total) is not None and used <= total):
            return f'{used:g}/{total:g} ({100*used/total:.1f}%)'
        return 'unreported'
    clock = _number(resource.get('clock_mhz'))
    core = _text(resource.get('core') or info.get('core') or info['model'], 30)
    if clock is not None:
        core += f' @ {clock:g} MHz'
    area = info.get('area_percent')
    result = '' if area is None else f' | Area {area:.2f}%'
    rate = _number(info.get('slice_rate'))
    speed = 'FPS: unmeasured (single result)' if rate is None else f'Slice/s (job window): {rate:.2f}'
    return [f"{info['model']} | {info['precision']} | {info['mode']} | saved task",
            f"{info['network_scope']}: {millis(info.get('network_seconds'))} | {info.get('task_scope', 'Upload to ZIP')}: {millis(info.get('task_seconds'))}",
            f"{core} | {info['threshold']}{result}",
            f"{speed} | Build DSP {usage('dsp')} / BRAM {usage('bram')}"]


@lru_cache(maxsize=4)
def _font(size):
    for name in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def annotate(screen, info, y=0, mode=None):
    """Return independent RGB pixels; keep source panels/exports immutable.

    Call after touch enlargement so both four-panel and enlarged views retain
    the task annotation. ACDC passes y=48 to preserve its top touch controls.
    """
    rgb = np.asarray(screen)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 160 or rgb.shape[1] < 320:
        raise ValueError('HUD requires a uint8 RGB screen of at least 320 x 160')
    if type(y) is not int or not 0 <= y <= rgb.shape[0]-88:
        raise ValueError('HUD origin outside screen')
    local = dict(info)
    if mode is not None:
        local['mode'] = _text(mode)
    lines = telemetry_lines(local)
    base = Image.fromarray(rgb).convert('RGBA')
    layer = Image.new('RGBA', base.size)
    draw = ImageDraw.Draw(layer)
    draw.rectangle((0, y, base.width, y+67), fill=(8, 22, 34, 255))
    for i, line in enumerate(lines):
        size = 11
        font = _font(size)
        while size > 9 and draw.textbbox((0, 0), line, font=font)[2] > base.width-16:
            size -= 1
            font = _font(size)
        # Fit within the screen even with a long model name or tiny fallback font.
        while line and draw.textbbox((0, 0), line, font=font)[2] > base.width-16:
            line = line[:-1]
        draw.text((8, y+2+i*16), line, font=font,
                  fill=(125, 232, 206, 255) if i == 0 else (236, 243, 248, 255))
    footer = 'Teaching demo / no clinical diagnosis | Tap HUD: CPU-FPGA metrics'
    font = _font(11)
    draw.rectangle((0, base.height-16, base.width, base.height), fill=(8, 22, 34, 255))
    draw.text((8, base.height-15), footer, font=font, fill=(242, 203, 137, 255))
    return np.asarray(Image.alpha_composite(base, layer).convert('RGB')).copy()
