"""Reproducible model-only capacity projection; explicitly not board evidence."""
from types import SimpleNamespace as S
import math

from .contract import SHAPES


def workload():
    from camus.plan import build_plan
    from camus.device import output_channel_chunks
    model=S(layers=[S(index=i,name=f'C{i+1:02}',full_input_shape=(1,ci,h,h),full_output_shape=(1,co,h,h),
                     kernel=k,padding=k//2,relu=i<14) for i,(ci,co,h,k) in enumerate(SHAPES)])
    plan=build_plan(model)
    plan.update(schema='pld-acdc-capacity-plan-v1',device='ACD3/ACD4 ABI 0x00030000; static capacity schedule',
                planner_donor='read-only camus.plan; same bank capacities, ACDC base16 layer shapes')
    tensor=sum((4 if r['mode']==2 else 1)*r['output_bytes']+8*r['tile_count'] for r in plan['tensor_schedule'])
    issue=sum(h*h*math.ceil(ci/8)*sum(math.ceil(c/16) for c in output_channel_chunks(ci,co,k))*k*k for ci,co,h,k in SHAPES)
    fast_conv=sum(4*block['tile_count']+row['output_hw'][0]*row['output_hw'][1]*(
        math.ceil(block['output_channels']/16)*(math.ceil(row['ci']/8)*row['kernel']**2+13)+block['output_channels'])
        for row in plan['layers'] for block in row['chunk_schedule'])
    fast_cycles=fast_conv+tensor
    return dict(scope='static projection, not measured board performance',execution_profile='axial_only_v1',
                compatibility=plan,engine_cycles=plan['projected_conv_compute_cycles']+tensor,
                compatibility_serial_floor_seconds={str(mhz):(plan['projected_conv_compute_cycles']+tensor+plan['total_packet_bytes'])/(mhz*1e6) for mhz in (100,125)},
                candidate_128_issue_cycles=issue,implemented_128_engine_cycles=fast_cycles,
                implemented_128_serial_floor_seconds={str(mhz):(fast_cycles+plan['total_packet_bytes'])/(mhz*1e6) for mhz in (100,125)},
                remaining_bottlenecks=['byte-serial load/drain','serialized requantization','command gaps'],target_seconds=.250,
                slices_formula='Nz_ED+Nz_ES from RAS 2mm center-corner bounding box')


def calibrate(vector_file,log_file,fast=False):
    import re
    from pathlib import Path
    regs={};configs=[];current=None
    for line in Path(vector_file).read_text().splitlines():
        op,*v=[int(s,16) for s in line.split()]
        if op==0 and v[3]==0:
            address,value,strobe=v[:3];old=regs.get(address,0)
            for byte in range(4):
                if strobe&(1<<byte):old=(old&~(255<<(8*byte)))|(value&(255<<(8*byte)))
            regs[address]=old
            if address==8 and value&1:current=dict(regs)
            if address==8 and value&2:current=None
        if op==2 and v[0]==0 and v[4] and current and current.get(0x14)==4:configs.append(current)
    log=Path(log_file).read_text()
    if 'TB_PASS tb_axis_tensor_accel_acdc' not in log:raise ValueError('RTL regression did not pass')
    observed=[int(v) for v in re.findall(r'ACDC_RUN run=\d+ bytes=\d+ cycles=(\d+)',log)]
    if len(observed)!=len(configs) or not observed:raise ValueError('missing RTL cycle samples')
    rows=[]
    for c,actual in zip(configs,observed):
        mode,h,w,ci,co,k=[c.get(key,0) for key in (0x18,0x1c,0x20,0x24,0x28,0x2c)]
        if mode==0:
            pi,po,setup,overhead=(8,16,4,13) if fast else (4,8,4,11)
            predicted=setup+(h-k+1)*(w-k+1)*(math.ceil(co/po)*(math.ceil(ci/pi)*k*k+overhead)+co)
        else:predicted=ci*h*w*(4 if mode==3 else 1)+8
        if predicted!=actual:raise ValueError(f'cycle model mismatch: mode={mode} {predicted} != {actual}')
        rows.append(dict(mode=mode,observed=actual,predicted=predicted))
    return dict(fast=fast,samples=len(rows),matched=True,rows=rows)
