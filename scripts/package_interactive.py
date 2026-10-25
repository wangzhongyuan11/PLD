"""Explicit minimal update package; never copy training data or credentials."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'display_web/unified'))
from install_interactive import HUB,SKIN,SHARED
VERSION='PLD_Skin_ROI_Touch_20261025_v8'
def package():
    members={};manifest={}
    # Reviewed v1-v4 manifest hashes, including the frozen v4 release payload.
    # The supplied source snapshot has no Git objects for its original history.
    accepted=json.loads((ROOT/'display_web/unified/interactive_accepted_versions.json').read_text())
    names=[('hub',name,'display_web/' if name in SHARED else 'display_web/unified/') for name in HUB]+[('skin',name,'display_web/') for name in SKIN]
    if set(accepted)!={kind+'/'+name for kind,name,_ in names}:raise ValueError('Baseline member mismatch')
    for kind,name,base in names:
        new=(ROOT/(base+name)).read_bytes()
        key=kind+'/'+name;manifest[key]=dict(old=accepted[key],new=hashlib.sha256(new).hexdigest());members['payload/'+key]=new
    members['patch_manifest.json']=json.dumps(manifest,indent=2).encode()
    for name in ('install_interactive.py','INTERACTIVE.md'):members[name]=(ROOT/'display_web/unified'/name).read_bytes()
    dest=ROOT/('out/'+VERSION+'.zip');dest.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(members.items()):
            info=zipfile.ZipInfo(VERSION+'/'+name,date_time=(2026,10,17,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,raw)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for key,value in manifest.items():assert hashlib.sha256(z.read(VERSION+'/payload/'+key)).hexdigest()==value['new']
    print('PACKAGE_PASS',dest,hashlib.sha256(dest.read_bytes()).hexdigest())
if __name__=='__main__':package()
