from __future__ import annotations
from pathlib import Path
import hashlib, json, os, tempfile, time

PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / 'runs/acdc_extension/20261017'
DATA = PROJECT / 'datasets/acdc'
ARCHITECTURE = 'ACDC25D-B16-P3-v1'
LABELS = {'0':'background','1':'RV_cavity','2':'LV_myocardium','3':'LV_cavity'}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20), b''): h.update(b)
    return h.hexdigest()

def save_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def code_identity():
    return {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}

def utc_now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
