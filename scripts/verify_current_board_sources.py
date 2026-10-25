"""Compare the published source map with this checkout or the existing board files.

This reads files only. It does not load an overlay, start a service or run inference.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify(manifest, root=ROOT, board=False):
    record = json.loads(Path(manifest).read_text(encoding='utf-8'))
    failures = []
    for row in record['files']:
        path = Path(row['board_path']) if board else Path(root)/row['repo_path']
        try:
            data = path.read_bytes()
        except OSError as error:
            failures.append(dict(path=str(path), error=str(error)))
            continue
        expected = row['board_sha256'] if board else row['lf_sha256']
        actual = hashlib.sha256(data if board else data.replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected:
            failures.append(dict(path=str(path), expected=expected, actual=actual))
    return dict(scope='board exact bytes' if board else 'repository source with LF normalization',
                checked=len(record['files']), passed=not failures, failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT/'deployment/current_board_20261025/SOURCE_MAP.json')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--board', action='store_true', help='read the absolute deployed paths on PYNQ')
    args = parser.parse_args()
    result = verify(args.manifest, args.root, args.board)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
