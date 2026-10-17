"""Explicit minimal update package; never copy training data or credentials."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'display_web/unified'))
from install_interactive import HUB
VERSION='PLD_Interactive_20261016_v4'
def package():
    members={};manifest={}
    # Reviewed v1-v3 manifest hashes, including the actual deployed v3 payload.
    # The supplied source snapshot has no Git objects for its original history.
    accepted=json.loads((ROOT/'display_web/unified/interactive_accepted_versions.json').read_text())
    if set(accepted)!={'hub/'+name for name in HUB}:raise ValueError('Baseline member mismatch')
    for name in HUB:
        rel='display_web/unified/'+name;new=(ROOT/rel).read_bytes()
        key='hub/'+name;manifest[key]=dict(old=accepted[key],new=hashlib.sha256(new).hexdigest());members['payload/'+key]=new
    members['patch_manifest.json']=json.dumps(manifest,indent=2).encode()
    for name in ('install_interactive.py','INTERACTIVE.md'):members[name]=(ROOT/'display_web/unified'/name).read_bytes()
    dest=ROOT/('out/'+VERSION+'.zip');dest.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(members.items()):
            info=zipfile.ZipInfo(VERSION+'/'+name,date_time=(2026,10,16,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,raw)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for key,value in manifest.items():assert hashlib.sha256(z.read(VERSION+'/payload/'+key)).hexdigest()==value['new']
    print('PACKAGE_PASS',dest,hashlib.sha256(dest.read_bytes()).hexdigest())
if __name__=='__main__':package()
