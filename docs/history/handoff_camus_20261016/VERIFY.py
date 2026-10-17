"""Offline integrity check for the complete CAMUS handoff directory."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    errors = []
    for relative, expected in manifest['files'].items():
        path = root / relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or path.is_symlink() or not path.resolve().is_relative_to(root)
                or not path.is_file()):
            errors.append(relative + ': missing or unsafe')
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            errors.append(relative + ': SHA256 differs')
    if errors:
        print('\n'.join(errors))
        return 1
    print('HANDOFF_ALL_PASS files=' + str(len(manifest['files'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
