"""Import report evidence without confusing estimates with measurements."""
import json
import math
from pathlib import Path
import re
from .core import digest,positive


def utilization(path):
    text=Path(path).read_text(encoding='utf-8-sig',errors='replace');out={}
    aliases={'Slice LUTs':'LUT','CLB LUTs':'LUT','Register as Flip Flop':'FF','Block RAM Tile':'BRAM36_equivalent','DSPs':'DSP','DSP48E1 only':'DSP','DSP48E2 only':'DSP'}
    for line in text.splitlines():
        cells=[c.strip() for c in line.split('|')[1:-1]]
        if len(cells)<5:continue
        name=cells[0].rstrip('*').strip();key=aliases.get(name)
        if key and key not in out:
            try:
                used=float(cells[1].replace(',',''));available=float(cells[-2].replace(',',''))
                percent=float(cells[-1])
            except ValueError:continue
            if not all(math.isfinite(v) and v>=0 for v in (used,available,percent)) or available==0:raise ValueError('invalid resource report')
            out[key]=dict(used=used,available=available,percent=percent)
    # Some reports list Slice Registers as a total with no separate FF row.
    # Do not silently call latches flip-flops.
    if not out:raise ValueError('no supported utilization table found')
    state=re.search(r'\|\s*Design State\s*:\s*(.*?)\s*\|',text)
    return dict(kind='vivado_utilization',evidence='tool_report',source_sha256=digest(path),
        design_state=state[1] if state else None,resources=out,
        missing=[k for k in ('LUT','FF','BRAM36_equivalent','DSP') if k not in out],
        bitstream_association_verified=False,
        warning='must bind this report to the deployed build manifest; presence of report alone does not verify current bitstream')


def power_report(path):
    text=Path(path).read_text(encoding='utf-8-sig',errors='replace');fields={}
    labels={'Total On-Chip Power (W)':'total_on_chip_w','Dynamic (W)':'dynamic_w','Device Static (W)':'device_static_w'}
    for line in text.splitlines():
        cells=[x.strip() for x in line.split('|')[1:-1]]
        if len(cells)>=2 and cells[0] in labels:
            try:fields[labels[cells[0]]]=positive(float(cells[1]),cells[0],zero=True)
            except ValueError:raise ValueError('invalid Vivado power value')
    if not fields:raise ValueError('no recognized Vivado power entries')
    return dict(kind='vivado_power_estimate',evidence='tool_estimate',scope='chip estimate; not measured board input power',
        source_sha256=digest(path),values=fields,bitstream_association_verified=False)


def integrate_power(payload):
    if payload.get('scope')!='board_input':raise ValueError('power samples require scope=board_input')
    rows=payload.get('samples',[])
    if len(rows)<2:raise ValueError('at least two power measurements required')
    energy=0.;first=None;prev=None
    for row in rows:
        t=positive(row['time_s'],'sample time',zero=True);p=positive(row['power_w'],'power',zero=True)
        if first is None:first=t
        if prev is not None:
            if t<=prev[0]:raise ValueError('sample times must strictly increase')
            energy+=(t-prev[0])*(p+prev[1])/2
        prev=(t,p)
    duration=prev[0]-first;frames=payload.get('frames')
    if type(frames) is not int or frames<=0:raise ValueError('frames must count completions in the exact sampled interval')
    idle=payload.get('idle_power_w')
    if idle is not None:idle=positive(idle,'idle_power_w',zero=True)
    return dict(kind='board_power_samples',evidence='measured_user_samples',scope='board_input',
        instrument=str(payload.get('instrument','unspecified')),sample_count=len(rows),duration_seconds=duration,
        frames=frames,energy_joules=energy,mean_watts=energy/duration,joules_per_frame=energy/frames,
        incremental_joules_per_frame=None if idle is None else (energy-idle*duration)/frames,
        warning='interval must contain exactly the stated completed frames; idle subtraction is an estimate and may be negative')


def imported_timing(path):
    raw=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if isinstance(raw.get('server_stages'),dict):
        server=raw['server_stages'];browser=raw.get('client_seconds') or {};scope='skin web timing; historical single request'
    elif isinstance(raw.get('timings_seconds'),dict):
        server=raw['timings_seconds'];browser={};scope='CAMUS result timing; historical single request'
    else:raise ValueError('unsupported timing JSON; expected server_stages or timings_seconds')
    def durations(d):
        return {k:float(v) for k,v in d.items() if not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and v>=0 and not k.endswith(('_pid','_pixels','_bytes'))}
    # Keep known timing fields; backend flags/PIDs and patient paths never enter exported summary.
    excluded={'model_reused','hardware_reused','persistent_worker_alive_after_reply'}
    allowed={'request_receive_parse','upload_decode_validate','request_to_job_start','stop_hdmi','input_write','runtime_verify','persistent_worker_startup','persistent_worker_request','persistent_worker_total','result_load','mask_remap','contour_process','preview_build','request_to_preview_ready','comparison_build','png_parallel_wait','zip_images','statistics_write','zip_report','zip_total','job_total','request_to_zip_done','inference_subprocess','input_read_preprocess','network','native_reconstruction_postprocess','head_labels_write','png_encoding','zip_payload','quad_display'}
    server={k:v for k,v in server.items() if k in allowed and k not in excluded}
    return dict(kind='imported_timing',evidence='historical_import_not_new_measurement',source_sha256=digest(path),
        scope=scope,browser_seconds=durations(browser),server_seconds=durations(server),
        ranked_top_level_seconds=sorted(durations(server).items(),key=lambda x:x[1],reverse=True),
        warning='totals and nested stages overlap; never add ranked values; single-request reciprocal is not measured throughput')


def imported_dma(path):
    from .core import dma_summary
    raw=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    events=raw.get('command_events')
    seconds=(raw.get('timings_seconds') or {}).get('network')
    if not isinstance(events,list) or not events or seconds is None:
        raise ValueError('DMA import requires command_events and timings_seconds.network')
    return dict(kind='historical_dma_packet_estimate',evidence='historical_import_not_new_measurement',
        source_sha256=digest(path),summary=dma_summary(events,seconds),
        warning='DMA packet accounting excludes PS copies/cache writeback/HDMI and is not a DDR controller measurement')
