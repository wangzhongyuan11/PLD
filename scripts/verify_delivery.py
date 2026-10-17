"""Compatibility entry for the current single release; see release_assets.py."""
import argparse
from pathlib import Path
from release_assets import ROOT, verify, prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=ROOT/'out/pld_dual_20261017_assets.zip')
    p.add_argument('--prepare-models', action='store_true')
    a = p.parse_args()
    if a.prepare_models:
        prepare(a.archive, ROOT)
    else:
        m = verify(a.archive)
        print('DELIVERY_PASS', m['release'], len(m['files']))


if __name__ == '__main__':
    main()
