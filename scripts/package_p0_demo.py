"""Reproducible allowlisted source candidate; never installs or publishes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE = 'cb710efeb2debea9a5c17f938b227eb9beb1b3e0'
FILES = (
    'README.md', 'acdc/appliance/hdmi.py', 'camus/hdmi.py',
    'display_web/entry/README.md', 'display_web/entry/entry.py',
    'display_web/entry/index.html', 'display_web/entry/recovery_demo.py',
    'display_web/entry/test_recovery_demo.py', 'display_web/entry/test_ui.cjs',
    'display_web/hdmi_worker.py', 'display_web/touch_display.py',
    'display_web/telemetry_hud.py', 'display_web/hud_build_info.example.json',
    'display_web/tests/test_telemetry_hud.py', 'display_web/unified/install_quad.py',
    'display_web/unified/tests/test_quad_install.py',
    'display_web/unified/camus_quad.py',
    'display_web/unified/quad_camus_hdmi.py', 'scripts/package_dual_model_web.py',
    'scripts/package_quad_view.py', 'scripts/package_p0_demo.py',
    'docs/P0_DEMO_HUD_RECOVERY.md', 'docs/validation/P0_DEMO_20261030.md',
    'display_web/demo_metrics.py','display_web/demo_metrics_input.example.json',
    'display_web/entry/metrics.html','display_web/entry/test_metrics.cjs','display_web/entry/test_metrics_gateway.py',
    'perf_eval/demo.py','perf_eval/tests/test_demo.py','scripts/prepare_demo_metrics.py',
    'camus/pipeline.py','display_web/camus_display_identity.py',
    'display_web/unified/tests/test_camus_display_identity.py','acdc/tests/test_hdmi_touch.py',
    'docs/DEMO_METRICS.md',
)


def package(output):
    output = Path(output)
    members = {name:(ROOT/name).read_bytes() for name in FILES}
    manifest = dict(schema='pld-p0-source-candidate-v1', status='candidate',
        base_commit=BASE,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        physical_board_verified=False, physical_hdmi_verified=False,
        files={name:hashlib.sha256(data).hexdigest() for name,data in members.items()})
    members['MANIFEST.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    try:
        with zipfile.ZipFile(temporary,'w') as archive:
            for name,data in sorted(members.items()):
                info=zipfile.ZipInfo('PLD_P0_Candidate/'+name,(1980,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
                archive.writestr(info,data)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('candidate ZIP CRC failed')
            for name,digest in manifest['files'].items():
                if hashlib.sha256(archive.read('PLD_P0_Candidate/'+name)).hexdigest()!=digest:
                    raise ValueError('candidate SHA256 mismatch: '+name)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'out/PLD_P0_Demo_Candidate_20261030.zip')
    args=parser.parse_args()
    print('SOURCE_CANDIDATE_VERIFIED',package(args.output),str(args.output))
