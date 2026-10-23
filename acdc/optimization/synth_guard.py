"""Bind experimental OOC synthesis to immutable sources; not a board certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    begin = sub.add_parser('begin')
    begin.add_argument('output', type=Path)
    begin.add_argument('sources', nargs='+', type=Path)
    finish = sub.add_parser('finish')
    finish.add_argument('output', type=Path)
    finish.add_argument('--dsp', type=int, required=True)
    args = parser.parse_args()
    start = args.output/'source_start.json'
    if args.mode == 'begin':
        args.output.mkdir(parents=True, exist_ok=True)
        files = {p.resolve().relative_to(ROOT).as_posix(): sha(p.resolve())
                 for p in args.sources}
        start.write_text(json.dumps(files, indent=2)+'\n', encoding='utf-8')
    else:
        files = json.loads(start.read_text(encoding='utf-8'))
        for name, expected in files.items():
            if sha(ROOT/name) != expected:
                raise ValueError('synthesis source changed during build: '+name)
        artifacts = {name: sha(args.output/name) for name in
                     ('utilization.rpt', 'timing.rpt', 'core.dcp')}
        result = dict(schema='pld-acdc-ooc-synthesis-v1', board_validated=False,
                      full_system_timing=False, expected_dsp_checked_by_tcl=args.dsp,
                      sources=files, artifacts=artifacts)
        (args.output/'synth_manifest.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
