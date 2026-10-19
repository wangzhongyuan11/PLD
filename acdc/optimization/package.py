"""Certification of a complete routed spatial SoC, separate from OOC evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as et

# Vivado's process environment carries vendor DLL paths. Keep the build-time
# snapshot/certification entrypoint stdlib-only; importing NumPy here can load
# incompatible vendor runtime DLLs before the actual build even starts.
ROOT=Path(__file__).resolve().parents[2]
ContractError=ValueError


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    pending.replace(path)


def source_snapshot(p):
    if p not in (4,6):raise ContractError('unknown spatial profile')
    packed=ROOT/f'ip_repo/acdc_spatial_p{p}/spatial_accel'
    packed_sv={path.name:file_sha(path) for path in packed.rglob('*.sv')}
    if not {'spatial_accel.sv','tile_conv.sv','requant_q24.sv','frame_engine.sv'}<=packed_sv.keys():
        raise ContractError('missing packaged spatial RTL')
    sources={}
    for name,sha in packed_sv.items():
        path=ROOT/('rtl' if name=='requant_q24.sv' else 'acdc/optimization/rtl')/name
        if not path.is_file() or file_sha(path)!=sha:raise ContractError('packaged spatial RTL drift: '+name)
        sources[path.relative_to(ROOT).as_posix()]=sha
    for name in ('acdc/optimization/vivado/package_spatial.tcl','acdc/optimization/vivado/build_spatial.tcl','vivado/ps7_config.tcl'):
        sources[name]=file_sha(ROOT/name)
    sources['acdc/optimization/package.py']=file_sha(Path(__file__).resolve())
    return sources


def begin(build,p):
    build=Path(build).resolve();build.mkdir(parents=True,exist_ok=True)
    atomic_json(build/'source_start.json',source_snapshot(p))


def certify(build,p):
    from scripts.check_build import check
    build=Path(build).resolve();check(build)
    sources=source_snapshot(p)
    if sources!=json.loads((build/'source_start.json').read_text('utf-8')):
        raise ContractError('spatial build source changed since launch')
    resources={}
    text=(build/'utilization.rpt').read_text('utf-8')
    for key,label in [('LUT','Slice LUTs'),('FF','Slice Registers'),('BRAM36_equivalent','Block RAM Tile'),('DSP','DSPs')]:
        match=re.search(r'\|\s*'+re.escape(label)+r'\*?\s*\|\s*([\d.]+)',text)
        if not match:raise ContractError('missing resource '+key)
        resources[key]=float(match[1])
    if resources['DSP']>220 or resources['BRAM36_equivalent']>140 or resources['LUT']>53200 or resources['FF']>106400:
        raise ContractError('spatial design exceeds Z2 resources')
    lines=(build/'timing_summary.rpt').read_text('utf-8').splitlines()
    i=next(i for i,l in enumerate(lines) if 'WNS(ns)' in l and 'WPWS(ns)' in l)
    timing=list(map(float,lines[i+2].split()))
    hwh=et.parse(build/'dma_top.hwh')
    clk=hwh.findall('.//PARAMETER[@NAME="PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ"]')
    if len(clk)!=1 or abs(float(clk[0].attrib['VALUE'])-100)>.1:raise ContractError('unexpected spatial FCLK')
    artifacts={name:file_sha(build/name) for name in ('dma_top.bit','dma_top.hwh','dma_top.xsa','timing_summary.rpt','utilization.rpt','drc.rpt','source_start.json')}
    report=dict(schema='pld-acdc-build-v1',hardware_id=0x41434438,abi=0x00020000,spatial_p=p,
        compute_lanes=p*64,clock_mhz=100,resources=resources,sources=sources,artifacts=artifacts,
        routed_timing=dict(WNS_ns=timing[0],WHS_ns=timing[4],WPWS_ns=timing[8]),board_validation=False)
    atomic_json(build/'build_manifest.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('build',type=Path);p.add_argument('--p',type=int,choices=(4,6),required=True)
    p.add_argument('--begin',action='store_true')
    a=p.parse_args()
    if a.begin:begin(a.build,a.p)
    else:print(json.dumps(certify(a.build,a.p)))
