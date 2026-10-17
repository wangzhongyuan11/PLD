"""Install/rollback a verified release; preserve private data and bounded DMA shutdown."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
import zipfile

UNITS = ('pld-hub.service', 'pld-web.service', 'pld-camus.service')
SYSTEM = Path('/etc/systemd/system')
BASE = Path('/home/xilinx/jupyter_notebooks/pld')
RELEASE = 'pld_dual_20261017'


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=300).stdout.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stop():
    # The gateway first drains/ends its current backend; never SIGKILL DMA owners.
    run('systemctl', 'stop', UNITS[0])
    for unit in UNITS:
        values = run('systemctl', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID', '-p', 'Result')
        if 'ActiveState=inactive' not in values or 'MainPID=0' not in values or 'Result=success' not in values:
            raise RuntimeError('Shutdown not confirmed: ' + values)
        group = run('systemctl', 'show', unit, '-p', 'ControlGroup', '--value')
        procs = Path('/sys/fs/cgroup' + group) / 'cgroup.procs' if group else None
        if procs and procs.exists() and procs.read_text().strip():
            raise RuntimeError('Unit cgroup still owns processes: ' + unit)
    import fcntl
    with open('/run/pld-unified-fpga.lock', 'a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)


def unit_files():
    return [p for u in UNITS for p in [SYSTEM/u, *sorted((SYSTEM/(u+'.d')).glob('*.conf'))] if p.is_file()]


def apply_units(root, receipt):
    # Reject edited/missing templates before touching the current service files.
    manifest = json.loads((root/'RELEASE_MANIFEST.json').read_text())
    for name in UNITS:
        if sha(root/'units'/name) != manifest['files']['units/'+name]:
            raise ValueError('Release unit template drift: ' + name)
    for p in unit_files():
        p.unlink()
    for name in UNITS:
        text = (root/'units'/name).read_text()
        (SYSTEM/name).write_text(text)
    run('systemctl', 'daemon-reload')
    run('systemd-analyze', 'verify', *[str(SYSTEM/u) for u in UNITS])
    run('systemctl', 'disable', *UNITS[1:])
    run('systemctl', 'enable', UNITS[0])
    receipt['installed_units'] = {str(SYSTEM/u): sha(SYSTEM/u) for u in UNITS}


def install(a):
    manifest = json.loads(a.manifest.read_text())
    if manifest['release'] != RELEASE or sha(a.archive) != manifest['archive_sha256']:
        raise ValueError('Release identity mismatch')
    root = BASE/'releases'/RELEASE
    if root.exists():
        raise ValueError('Release already exists; use reapply with its rollback record')
    # Verify every member before extracting; models and raw datasets stay on host.
    with zipfile.ZipFile(a.archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest['files']):
            raise ValueError('Release member mismatch')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe release path')
            if z.getinfo(name).external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError('Release symlink forbidden')
            if hashlib.sha256(z.read(name)).hexdigest() != manifest['files'][name]:
                raise ValueError('Release member drift: ' + name)
        for name in names:
            if name.split('/')[0] in ('skin', 'camus', 'hub', 'web', 'units', 'config'):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(z.read(name))
    (root/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    old_hub = Path('/home/xilinx/jupyter_notebooks/PLD_DualModel')
    config = json.loads((old_hub/'config.json').read_text())
    if not Path(config['token_file']).is_file():
        raise ValueError('Existing access token missing')
    # Configuration carries paths only, never copy the token into the release.
    (root/'hub/config.json').write_text(json.dumps(config, indent=2))
    installation = json.loads((root/'config/installation.json').read_text())
    (root/'hub/installation.json').write_text(json.dumps(installation, indent=2))
    stop()
    backup = BASE/'rollback'/('pre_'+RELEASE)
    backup.mkdir(parents=True, exist_ok=False)
    receipt = {'release_root': str(root), 'original_files': {},
               'enabled': {u: subprocess.run(['systemctl', 'is-enabled', u],
                   capture_output=True, text=True).stdout.strip() for u in UNITS}}
    if any(v not in ('enabled', 'disabled') for v in receipt['enabled'].values()):
        raise ValueError('Unsupported unit enablement state')
    for p in unit_files():
        rel = p.relative_to(SYSTEM)
        dest = backup/'units'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        receipt['original_files'][str(p)] = sha(dest)
    # Keep one complete private fallback, rather than relying on old source trees.
    old = json.loads((old_hub/'installation.json').read_text())
    mapping = {str(old_hub): str(backup/'hub')}
    sources = {'hub': old_hub, 'web': Path(old['backends']['skin']['source']),
               'skin': Path(old['backends']['skin']['runtime']),
               'camus': Path(old['backends']['camus']['runtime'])}
    for name, source in sources.items():
        shutil.copytree(source, backup/name, ignore=shutil.ignore_patterns('data', '__pycache__', 'backup_*'))
        mapping[str(source)] = str(backup/name)
    # Relocate fallback source/config references; shared histories remain in place.
    receipt['fallback_mapping'] = mapping
    (backup/'receipt.json').write_text(json.dumps(receipt, indent=2))
    apply_units(root, receipt)
    (backup/'receipt.json').write_text(json.dumps(receipt, indent=2))
    (root/'hub/state').mkdir(exist_ok=True)
    (root/'hub/state/last_model').write_text('skin')
    run('systemctl', 'start', UNITS[0])
    print('DEPLOY_STARTED', root, 'ROLLBACK', backup)


def rollback(a):
    backup = a.backup.resolve()
    if not backup.is_relative_to(BASE/'rollback'):
        raise ValueError('Rollback path outside managed root')
    receipt = json.loads((backup/'receipt.json').read_text())
    root = Path(receipt['release_root'])
    expected_units = receipt.get('rollback_units') if a.action == 'reapply' else receipt['installed_units']
    if not expected_units or {str(p) for p in unit_files()} != set(expected_units):
        raise ValueError('Unexpected unit/drop-in changes; preserve them for manual review')
    for path, expected in expected_units.items():
        if sha(path) != expected:
            raise ValueError('Current unit was edited; retained: ' + path)
    if a.action == 'rollback':
        for target, expected in receipt['original_files'].items():
            original = backup/'units'/Path(target).relative_to(SYSTEM)
            if sha(original) != expected:
                raise ValueError('Rollback file hash drift')
    stop()
    if a.action == 'reapply':
        apply_units(root, receipt)
    else:
        for p, expected in receipt['installed_units'].items():
            if sha(p) != expected:
                raise ValueError('Installed unit was edited; preserve it for manual review: ' + p)
        for p in unit_files():
            p.unlink()
        for target, expected in receipt['original_files'].items():
            original = backup/'units'/Path(target).relative_to(SYSTEM)
            if sha(original) != expected:
                raise ValueError('Rollback file hash drift')
            text = original.read_text()
            for old, new in sorted(receipt['fallback_mapping'].items(), key=lambda x: -len(x[0])):
                text = text.replace(old, new)
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        for name in ('installation.json',):
            text = (backup/'hub'/name).read_text()
            for old, new in sorted(receipt['fallback_mapping'].items(), key=lambda x: -len(x[0])):
                text = text.replace(old, new)
            (backup/'hub'/name).write_text(text)
        run('systemctl', 'daemon-reload')
        for unit, enabled in receipt['enabled'].items():
            run('systemctl', 'enable' if enabled == 'enabled' else 'disable', unit)
        receipt['rollback_units'] = {str(p): sha(p) for p in unit_files()}
    (backup/'receipt.json').write_text(json.dumps(receipt, indent=2))
    run('systemctl', 'start', UNITS[0])
    print(a.action.upper()+'_STARTED', root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('install', 'rollback', 'reapply'))
    p.add_argument('--archive', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--backup', type=Path, default=BASE/'rollback'/('pre_'+RELEASE))
    a = p.parse_args()
    if os.name != 'posix' or os.geteuid() != 0:
        p.error('Run on the board with sudo PYNQ Python')
    os.umask(0o077)
    if a.action == 'install':
        if not a.archive or not a.manifest:
            p.error('install needs --archive and --manifest')
        install(a)
    else:
        rollback(a)


if __name__ == '__main__':
    main()
