"""Source-bound local CAMUS build certificate and independent runtime package."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import subprocess
import xml.etree.ElementTree as ET

from .model import MODEL_SPEC_SHA256, DEFAULT_MODEL_ROOT, load_model

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'out/camus/vivado'
SHARED = ('conv_mac_array', 'requant_q24', 'axi_lite_endpoint', 'axis_byte_pack', 'axis_byte_unpack')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8', newline='\n')


def build_sources():
    return [*(ROOT / 'rtl' / (name+'.sv') for name in SHARED),
            *sorted((ROOT/'camus/rtl').glob('*.sv')),
            *sorted((ROOT/'camus/vivado').glob('*.tcl')),
            ROOT/'camus/build.sh', ROOT/'camus/package.py', ROOT/'vivado/ps7_config.tcl',
            ROOT/'scripts/extract_xsa.py', ROOT/'scripts/check_build.py']


def certify_build():
    from scripts.check_build import check
    check(BUILD)
    files = [BUILD/name for name in ('dma_top.bit', 'dma_top.hwh', 'dma_top.xsa', 'timing_summary.rpt', 'utilization.rpt', 'drc.rpt')]
    packed = ROOT/'ip_repo/camus/axis_tensor_accel_camus'
    packed_sources = list(packed.rglob('*.sv'))
    source_map = {p.name: digest(p) for p in build_sources() if p.suffix == '.sv'}
    found = {p.name: digest(p) for p in packed_sources}
    if any(found.get(k) != v for k, v in source_map.items()):
        raise ValueError('packaged CAM2 RTL differs from the compiled source')
    tree = ET.parse(BUILD/'dma_top.hwh')
    clocks = tree.findall('.//PARAMETER[@NAME="PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ"]')
    if len(clocks) != 1 or abs(float(clocks[0].attrib['VALUE']) - 100.) > .01:
        raise ValueError('CAMUS candidate is not the 100 MHz design')
    certificate = dict(schema='camus-build-v1', clock_mhz=100,
                       git_head_at_certification=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                       provenance='source SHA256 and packed RTL bind this build; Git HEAD alone does not identify uncommitted candidate source',
                       sources={p.relative_to(ROOT).as_posix(): digest(p) for p in build_sources()},
                       artifacts={p.name: digest(p) for p in files},
                       packed_rtl={p.relative_to(ROOT).as_posix(): digest(p) for p in packed_sources})
    write_json(BUILD/'build_manifest.json', certificate)
    print('CAMUS_BUILD_CERTIFIED')
    return certificate


def package_runtime(output=None, model_root=DEFAULT_MODEL_ROOT):
    model = load_model(model_root)
    manifest = json.loads((BUILD/'build_manifest.json').read_text())
    for rel, expected in manifest['sources'].items():
        if digest(ROOT/rel) != expected:
            raise ValueError('build source drift: '+rel)
    for name, expected in manifest['artifacts'].items():
        if digest(BUILD/name) != expected:
            raise ValueError('build artifact drift: '+name)
    for rel, expected in manifest['packed_rtl'].items():
        if digest(ROOT/rel) != expected:
            raise ValueError('packaged IP drift: '+rel)
    paths = {}
    for p in (ROOT/'camus').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
            paths[p.relative_to(ROOT).as_posix()] = p
    for folder, names in (('board', ('tensor_accel.py', 'simple_dma.py')),
                          ('reference', ('__init__.py', 'tiling.py', 'tinyunet.py'))):
        for name in names:
            if (ROOT/folder/name).is_file(): paths[folder+'/'+name] = ROOT/folder/name
    for p in model.root.rglob('*'):
        if p.is_file(): paths['model/int8_export/'+p.relative_to(model.root).as_posix()] = p
    for name in (*manifest['artifacts'], 'build_manifest.json'):
        paths['overlay/'+name] = BUILD/name
    for rel in manifest['sources']:
        paths[rel] = ROOT/rel
    runtime = dict(schema='camus-runtime-v1', model_spec_sha256=MODEL_SPEC_SHA256,
                   git_head_at_packaging=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                   files={rel:digest(p) for rel,p in paths.items()}, model_directory='model/int8_export',
                   clock_mhz=100, board_validation=False)
    output = Path(output or ROOT/'out/camus/runtime.tar.gz')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT/'out/camus/runtime_manifest.json'; write_json(manifest_path, runtime)
    with tarfile.open(output, 'w:gz') as archive:
        for rel,p in sorted(paths.items()): archive.add(p, arcname=rel, recursive=False)
        archive.add(manifest_path, arcname='runtime_manifest.json', recursive=False)
    print(json.dumps(dict(package=str(output), members=len(paths)+1, sha256=digest(output))))
    return output


def verify_runtime(runtime, model_root):
    runtime = Path(runtime).resolve()
    manifest = json.loads((runtime/'runtime_manifest.json').read_text())
    if manifest.get('schema') != 'camus-runtime-v1' or manifest.get('model_spec_sha256') != MODEL_SPEC_SHA256:
        raise ValueError('not a frozen CAMUS runtime')
    if manifest.get('clock_mhz') != 100:
        raise ValueError('unsupported CAMUS runtime clock')
    for rel, expected in manifest['files'].items():
        p = runtime/rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or not p.resolve().is_relative_to(runtime):
            raise ValueError('unsafe runtime member')
        if p.is_symlink() or digest(p) != expected:
            raise ValueError('CAMUS runtime drift: '+rel)
        if p.suffix == '.py' and rel.split('/')[0] in ('camus','board','reference'):
            if not (ROOT/rel).is_file() or digest(ROOT/rel) != expected:
                raise ValueError('executing source differs from frozen runtime: '+rel)
    required = ('overlay/dma_top.bit', 'overlay/dma_top.hwh', 'overlay/build_manifest.json', 'camus/accelerator.py')
    if any(rel not in manifest['files'] for rel in required):
        raise ValueError('incomplete CAMUS runtime')
    if Path(model_root).resolve() != (runtime/manifest['model_directory']).resolve():
        raise ValueError('PL backend must use the model inside this verified runtime')
    return runtime


def main():
    p = argparse.ArgumentParser(); p.add_argument('--certify-build', action='store_true')
    p.add_argument('--output', type=Path); p.add_argument('--model', type=Path, default=DEFAULT_MODEL_ROOT)
    args = p.parse_args()
    if args.certify_build: certify_build()
    else: package_runtime(args.output, args.model)


if __name__ == '__main__': main()
