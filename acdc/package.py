"""Reproducible source package and hash-bound ACD3 runtime verification."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import zipfile

from .contract import file_sha,atomic_json,ContractError,safe_member

ROOT=Path(__file__).resolve().parents[1]
SHARED_FILES=('camus/contract.py','camus/device.py','camus/integer.py','camus/accelerator.py','camus/model.py','camus/plan.py','camus/gen_vectors.py',
              'camus/native.py','camus/native_control.py','camus/native/control.c','camus/native/ops.c',
              'camus/__init__.py','reference/__init__.py','reference/tiling.py',
              'board/tensor_accel.py','board/simple_dma.py','rtl/conv_mac_array.sv','rtl/requant_q24.sv',
              'rtl/axi_lite_endpoint.sv','rtl/axis_byte_pack.sv','rtl/axis_byte_unpack.sv',
              'reference/tinyunet.py','vivado/ps7_config.tcl','scripts/extract_xsa.py','scripts/check_build.py')


def source_files():
    own=[p for p in (ROOT/'acdc').rglob('*') if p.is_file() and not any(
        part in ('.venv','__pycache__','.pytest_cache') for part in p.parts) and p.suffix not in ('.pyc',)]
    return {p.relative_to(ROOT).as_posix():p for p in [*own,*(ROOT/r for r in SHARED_FILES)]}


def source_package(output):
    output=Path(output).resolve()
    if output.exists():raise ContractError('package output already exists')
    if output.is_relative_to(ROOT/'acdc'):raise ContractError('source archive must be outside acdc source tree')
    files=source_files()
    if output in [p.resolve() for p in files.values()]:raise ContractError('package must be outside source tree')
    manifest=dict(schema='pld-acdc-source-v1',files={r:file_sha(p) for r,p in files.items()},
                  production_model_included=False,board_validated=False,profile='axial_only_v1')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel,path in sorted(files.items()):z.write(path,rel)
        z.writestr('source_manifest.json',json.dumps(manifest,indent=2))
    return output


def build_sources(fast=False):
    modules=['conv_tile_engine_acdc','tensor_tile_engine_acdc','axis_tensor_accel_acdc']
    paths=[ROOT/'acdc/rtl'/(n+'.sv') for n in modules]
    paths += [ROOT/r for r in SHARED_FILES if r.startswith('rtl/')]
    if fast:
        paths += [ROOT/'acdc/performance_ip'/(n+'.sv') for n in (
            'conv_mac_128','conv_tile_engine_acdc_128','axis_tensor_accel_acdc_fast')]
    paths += [ROOT/'acdc/vivado'/n for n in (
        ('package_fast.tcl','build_fast.tcl') if fast else ('package.tcl','build.tcl'))]
    paths += [ROOT/n for n in ('vivado/ps7_config.tcl','scripts/extract_xsa.py','scripts/check_build.py')]
    return paths


def certify_build(build=None,fast=False):
    from scripts.check_build import check
    build=Path(build or ROOT/'out/acdc/vivado').resolve();check(build)
    artifacts={n:file_sha(build/n) for n in ('dma_top.bit','dma_top.hwh','dma_top.xsa','timing_summary.rpt','utilization.rpt','drc.rpt')}
    sources={p.relative_to(ROOT).as_posix():file_sha(p) for p in build_sources(fast)}
    packed=ROOT/('ip_repo/acdc_fast/axis_tensor_accel_acdc' if fast else 'ip_repo/acdc/axis_tensor_accel_acdc')
    expected={Path(r).name:sha for r,sha in sources.items() if r.endswith('.sv')}
    actual={p.name:file_sha(p) for p in packed.rglob('*.sv')}
    if any(actual.get(n)!=sha for n,sha in expected.items()):raise ContractError('packaged RTL source drift')
    import re
    import xml.etree.ElementTree as ET
    text=(build/'utilization.rpt').read_text('utf-8')
    resources={}
    for key,label in [('LUT','Slice LUTs'),('FF','Slice Registers'),('BRAM36_equivalent','Block RAM Tile'),('DSP','DSPs')]:
        match=re.search(r'\|\s*'+re.escape(label)+r'\*?\s*\|\s*([\d.]+)',text)
        if match:resources[key]=float(match[1])
    hwh=ET.parse(build/'dma_top.hwh')
    clocks=hwh.findall('.//PARAMETER[@NAME="PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ"]')
    if len(clocks)!=1:raise ContractError('unidentified HWH clock')
    lines=(build/'timing_summary.rpt').read_text('utf-8').splitlines()
    header=next(i for i,l in enumerate(lines) if 'WNS(ns)' in l and 'WPWS(ns)' in l)
    timing=[float(x) for x in lines[header+2].split()]
    record=dict(schema='pld-acdc-build-v1',sources=sources,artifacts=artifacts,board_validation=False,
                clock_mhz=float(clocks[0].attrib['VALUE']),resources=resources,
                routed_timing=dict(WNS_ns=timing[0],WHS_ns=timing[4],WPWS_ns=timing[8]),
                hardware_id=0x41434434 if fast else 0x41434433,abi=0x00030000,compute_lanes=128 if fast else 32)
    atomic_json(build/'build_manifest.json',record)
    return record


def runtime_package(output,build,model):
    import shutil
    build=Path(build).resolve();output=Path(output).resolve()
    if output.is_relative_to(model.root.resolve()) or output.is_relative_to(ROOT/'acdc'):
        raise ContractError('runtime must be outside model and ACDC source trees')
    certificate=json.loads((build/'build_manifest.json').read_text('utf-8'))
    if certificate.get('schema')!='pld-acdc-build-v1':raise ContractError('uncertified build')
    for rel,sha in certificate['sources'].items():
        if file_sha(ROOT/rel)!=sha:raise ContractError('build source drift: '+rel)
    for name,sha in certificate['artifacts'].items():
        if file_sha(build/name)!=sha:raise ContractError('build artifact drift: '+name)
    if output.exists():raise ContractError('runtime output already exists')
    output.mkdir(parents=True)
    executed=source_files()
    for rel,src in executed.items():
        dst=output/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in model.root.rglob('*'):
        if src.is_symlink():raise ContractError('symlink in model package')
        if src.is_file():
            dst=output/'model'/src.relative_to(model.root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for name in (*certificate['artifacts'],'build_manifest.json'):
        dst=output/'overlay'/('evidence' if name.endswith('.xsa') else '')/name
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(build/name,dst)
    atomic_json(output/'runtime_manifest.json',dict(schema='pld-acdc-runtime-v1',
        model_spec_sha256=model.model_sha256,synthetic=model.synthetic,hardware_id=certificate['hardware_id'],abi=certificate['abi'],
        compute_lanes=certificate['compute_lanes'],board_validation=False,
        executed_sources={r:file_sha(p) for r,p in executed.items()},
        files={p.relative_to(output).as_posix():file_sha(p) for p in output.rglob('*') if p.is_file()}))
    return output


def verify_runtime(runtime,model):
    root=Path(runtime).resolve()
    record=json.loads((root/'runtime_manifest.json').read_text('utf-8'))
    if record.get('schema')!='pld-acdc-runtime-v1' or record.get('model_spec_sha256')!=model.model_sha256:
        raise ContractError('runtime/model identity mismatch')
    for rel,sha in record['files'].items():
        if file_sha(safe_member(root,rel))!=sha:raise ContractError('runtime member drift: '+rel)
    for rel,sha in record['executed_sources'].items():
        if file_sha(ROOT/rel)!=sha:raise ContractError('executed source drift: '+rel)
    if 'overlay/dma_top.bit' not in record['files'] or 'overlay/dma_top.hwh' not in record['files']:
        raise ContractError('runtime lacks paired overlay')
    certificate=json.loads(safe_member(root,'overlay/build_manifest.json').read_text('utf-8'))
    if (certificate.get('schema')!='pld-acdc-build-v1' or certificate.get('hardware_id')!=record.get('hardware_id')
            or certificate.get('abi')!=record.get('abi')):
        raise ContractError('runtime/build capability mismatch')
    for name,sha in certificate['artifacts'].items():
        relative='overlay/'+('evidence/' if name.endswith('.xsa') else '')+name
        if record['files'].get(relative)!=sha:raise ContractError('runtime/build artifact mismatch')
    for name,sha in certificate['sources'].items():
        if record['executed_sources'].get(name)!=sha:raise ContractError('runtime/build source mismatch')
    return root


if __name__=='__main__':
    # Pass the path as argv so MSYS converts /d/... for Windows Python.
    import argparse
    parser=argparse.ArgumentParser(description='Certify an existing ACDC routed build')
    parser.add_argument('build',type=Path);parser.add_argument('--fast',action='store_true')
    args=parser.parse_args()
    print(json.dumps(certify_build(args.build,fast=args.fast)))
