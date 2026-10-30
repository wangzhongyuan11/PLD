"""Reviewable, hash-gated patch for the deployed 2026-10-16 unified system."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKIN = ('static/index.html','static/app.js','static/style.css','hdmi_worker.py','quad_display.py')
HUB = ('backend_launcher.py','accepted_backends.json','camus_quad.py','quad_display.py',
       'quad_camus.html','quad_camus_hdmi.py')
ADDITIONS = tuple(kind+'/'+name for kind in ('skin','hub') for name in ('telemetry_hud.py','demo_metrics.py','camus_display_identity.py'))


def package(output=None):
    output = Path(output or ROOT/'out/PLD_QuadView_20261016_v2.zip')
    members = {}
    manifest = {}
    # Preserve exact hashes from the supplied, verified v2 update package.
    # Its original Git history is absent from the imported source snapshot.
    accepted = json.loads((ROOT/'display_web/unified/quad_accepted_versions.json').read_text())
    if set(accepted) != {'skin/'+n for n in SKIN}|{'hub/'+n for n in HUB}:
        raise ValueError('Baseline member mismatch')
    for kind,folder,names in [('skin','display_web',SKIN),('hub','display_web/unified',HUB)]:
        for name in names:
            key = kind+'/'+name
            data = (ROOT/folder/name).read_bytes()
            expected = accepted[key]
            manifest[key] = dict(new=hashlib.sha256(data).hexdigest(),old=expected)
            members['payload/'+key] = data
    # New display-only dependency. Only absent or exact candidate bytes may be
    # installed; unrelated existing files retain the original refusal behavior.
    for key in ADDITIONS:
        data = (ROOT/'display_web'/key.split('/')[-1]).read_bytes()
        manifest[key] = dict(new=hashlib.sha256(data).hexdigest(),old=[None])
        members['payload/'+key] = data
    members['patch_manifest.json'] = (json.dumps(manifest,indent=2)+'\n').encode()
    members['install_quad.py'] = (ROOT/'display_web/unified/install_quad.py').read_bytes()
    members['README.md'] = (ROOT/'display_web/unified/QUAD_VIEW.md').read_bytes()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(members.items()):archive.writestr('PLD_QuadView_20261016_v2/'+name,data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for key,record in manifest.items():
            assert hashlib.sha256(archive.read('PLD_QuadView_20261016_v2/payload/'+key)).hexdigest() == record['new']
    print('QUAD_PACKAGE_PASS',output)
    print('SHA256',hashlib.sha256(output.read_bytes()).hexdigest())
    return output


if __name__ == '__main__':package()
