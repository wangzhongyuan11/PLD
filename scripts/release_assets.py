"""Verify/prepare the single frozen dual-model asset archive (no network needed)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'delivery/pld_dual_20261017'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe asset member: ' + name)
    return path


def verify(archive):
    manifest = json.loads((RELEASE / 'MANIFEST.json').read_text(encoding='utf-8'))
    if digest(archive) != manifest['archive_sha256']:
        raise ValueError('Archive SHA256 mismatch')
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest['files']):
            raise ValueError('Asset member list mismatch')
        for name in names:
            member(name)
            info = source.getinfo(name)
            if info.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError('Asset symlink forbidden')
            if hashlib.sha256(source.read(name)).hexdigest() != manifest['files'][name]:
                raise ValueError('Asset hash mismatch: ' + name)
    return manifest


def prepare(archive, root):
    manifest = verify(archive)
    root = Path(root).resolve()
    mappings = {'models/role_a_final/': 'role_a_final/',
                'models/role_a_handoff/': 'role_a_handoff/',
                'models/camus_candidate_hidden_histogram_int8_20261015/':
                    'camus_candidate_hidden_histogram_int8_20261015/',
                'hardware/skin/': 'out/vivado_cnn/',
                'hardware/camus/': 'out/camus/vivado/'}
    # Preflight ALL destinations before writing; preserve conflicting user files.
    pending = []
    for name, expected in manifest['files'].items():
        for prefix, target in mappings.items():
            if name.startswith(prefix):
                path = root / target / str(member(name[len(prefix):]))
                ancestors = [path, *[p for p in path.parents if p != root and p.is_relative_to(root)]]
                if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in ancestors):
                    raise ValueError('Unsafe destination: ' + str(path))
                if path.exists() and digest(path) != expected:
                    raise ValueError('Existing file differs; retained: ' + str(path))
                pending.append((name, path))
    with zipfile.ZipFile(archive) as source:
        for name, path in pending:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.open(name) as src, path.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
    print('PREPARE_PASS', len(pending), 'model/build files', root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--root', type=Path, default=ROOT)
    a = p.parse_args()
    if a.prepare:
        prepare(a.archive, a.root)
    else:
        m = verify(a.archive)
        print('RELEASE_ASSETS_PASS', m['release'], len(m['files']))


if __name__ == '__main__':
    main()
