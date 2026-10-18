"""Render a reviewable unit only; does not SSH/install/stop/restart a service."""
from pathlib import Path
import argparse
import re


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--spec-sha',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not re.fullmatch(r'/home/xilinx/jupyter_notebooks/pld/releases/acdc_[a-zA-Z0-9_]+',a.release):
        p.error('release must be a distinct versioned ACDC directory')
    if not re.fullmatch('[0-9a-f]{64}',a.spec_sha):p.error('invalid SHA')
    if a.output.exists():p.error('output already exists')
    text=Path(__file__).with_name('acdc-worker.service.in').read_text('utf-8')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(text.replace('@RELEASE@',a.release).replace('@SPEC_SHA@',a.spec_sha),encoding='utf-8')


if __name__=='__main__':main()
