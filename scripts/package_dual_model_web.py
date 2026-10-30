"""Package the optional dual-model gateway without local state or model assets."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'display_web/unified'
SHARED = ('display_completion.py', 'display_lease.py', 'warm_hdmi.py', 'video_overlay.py', 'overlay_metadata.py', 'latency.py', 'telemetry_hud.py')
FILES = (
    'interactive_bridge.py', 'interactive.html', 'interactive.js',
    'interactive_workspace.py', 'interactive_hdmi.py',
    'evaluation_bridge.py', 'evaluation.html', 'evaluation_config.example.json',
    'prompt_seg/__init__.py', 'prompt_seg/contracts.py', 'prompt_seg/refinement.py', 'prompt_seg/service.py',
    'hub.py', 'backend_launcher.py', 'camus_input_adapter.py', 'install.py', 'diagnose.py', 'index.html',
    'accepted_backends.json', 'README.md', 'TEST_REPORT.md', 'QUAD_VIEW.md',
    'camus_quad.py', 'quad_display.py', 'quad_camus.html', 'quad_camus_hdmi.py',
    'tests/test_hub.py', 'tests/test_ui.cjs',
) + SHARED


def member_path(source, name):
    source=Path(source)
    return source/name if (source/name).is_file() or name not in SHARED else source.parent/name


def package(output, source=SOURCE):
    source, output = Path(source), Path(output)
    # Fixed allowlist: never collect credentials, state, models or arbitrary files.
    members = {name: member_path(source,name).read_bytes() for name in FILES}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
    members['SHA256.json'] = (json.dumps(hashes, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w') as archive:
            for name, data in sorted(members.items()):
                info = zipfile.ZipInfo('PLD_DualModel/' + name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP CRC verification failed')
            for name, digest in hashes.items():
                if hashlib.sha256(archive.read('PLD_DualModel/' + name)).hexdigest() != digest:
                    raise ValueError('ZIP member hash mismatch: ' + name)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'out/PLD_DualModel.zip')
    args = parser.parse_args()
    hashes = package(args.output)
    print(f'PACKAGE_PASS {args.output} ({len(hashes)} source files + SHA256.json)')
    print('SHA256 ' + hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
