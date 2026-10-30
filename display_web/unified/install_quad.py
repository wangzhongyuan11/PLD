"""Apply the reviewed four-panel patch to an existing unified installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SKIN = ('static/index.html','static/app.js','static/style.css','hdmi_worker.py','quad_display.py','telemetry_hud.py')
HUB = ('backend_launcher.py','accepted_backends.json','camus_quad.py','quad_display.py',
       'quad_camus.html','quad_camus_hdmi.py','telemetry_hud.py')
UNITS = ('pld-hub.service','pld-web.service','pld-camus.service')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def destinations(root):
    receipt = json.loads((root/'installation.json').read_text())
    skin = Path(receipt['backends']['skin']['source']).resolve()
    if not skin.is_absolute() or not (skin/'server.py').is_file():
        raise ValueError('Cannot identify the installed skin backend')
    result = {}
    for kind,base,names in [('skin',skin,SKIN),('hub',root,HUB)]:
        for name in names:
            target = base/name
            if target.is_symlink() or not target.resolve().is_relative_to(base):
                raise ValueError('Unsafe target: '+name)
            result[kind+'/'+name] = target
    return result


def call(*args):
    return subprocess.run(args,check=True,capture_output=True,text=True,timeout=360).stdout


def stop_safely():
    # The existing hub waits for tasks, releases HDMI and stops both models.
    call('systemctl','stop','pld-hub.service')
    for unit in UNITS:
        text = call('systemctl','show',unit,'-p','ActiveState','-p','MainPID','-p','Result')
        props = dict(line.split('=',1) for line in text.splitlines() if '=' in line)
        if props != dict(ActiveState='inactive', MainPID='0', Result='success'):
            raise RuntimeError('Normal shutdown not confirmed: '+unit+' '+str(props))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hub',type=Path,default=Path('/home/xilinx/jupyter_notebooks/PLD_DualModel'))
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--rollback',type=Path)
    args = parser.parse_args()
    if sys.platform != 'linux' or os.geteuid() != 0:
        parser.error('Run on the board using sudo and the PYNQ Python')
    if bool(args.apply) == bool(args.rollback):
        parser.error('Choose --apply or --rollback BACKUP_DIRECTORY')
    root = args.hub.resolve()
    targets = destinations(root)
    if args.rollback:
        backup = args.rollback.resolve()
        if backup.parent != root or not backup.name.startswith('backup_quad_'):
            raise ValueError('Backup must be backup_quad_* inside the hub directory')
        record = json.loads((backup/'record.json').read_text())
        # Old backups predate this optional module. Restore precisely their
        # original member set; never remove a file they did not back up.
        if set(record) == set(targets)-{'skin/telemetry_hud.py','hub/telemetry_hud.py'}:
            targets = {key:value for key,value in targets.items() if key in record}
        if set(record) != set(targets):raise ValueError('Backup member mismatch')
        for key,target in targets.items():
            item = record[key]
            actual = digest(target) if target.exists() else None
            if actual not in (item['new'], item['old']):
                raise ValueError('Installed file changed; rollback stopped: '+key)
            if item['old'] is not None and digest(backup/key) != item['old']:
                raise ValueError('Backup hash mismatch: '+key)
        stop_safely()
        for key,target in targets.items():
            if record[key]['old'] is None:target.unlink(missing_ok=True)
            else:shutil.copy2(backup/key,target)
        call('systemctl','start','pld-hub.service')
        print('ROLLBACK_APPLIED; wait for model initialization, then refresh the browser')
        return
    package = Path(__file__).resolve().parent
    manifest = json.loads((package/'patch_manifest.json').read_text())
    if set(manifest) != set(targets):raise ValueError('Patch member mismatch')
    for key,target in targets.items():
        entry = manifest[key]
        if digest(package/'payload'/key) != entry['new']:
            raise ValueError('Package hash mismatch: '+key)
        actual = digest(target) if target.exists() else None
        if actual not in entry['old']+[entry['new']]:
            raise ValueError('Existing version differs; no files changed: '+key)
    print('Preflight passed. Waiting for current jobs and normal hardware release.',flush=True)
    stop_safely()
    backup = root/('backup_quad_'+str(time.time_ns()))
    backup.mkdir()
    record = {}
    for key,target in targets.items():
        old = digest(target) if target.exists() else None
        record[key] = {'old':old,'new':manifest[key]['new']}
        if old is not None:
            (backup/key).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target,backup/key)
    (backup/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Backup: '+str(backup),flush=True)
    # Backups and record are complete before the first replacement.
    for key,target in targets.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary = target.with_name(target.name+'.quad-new')
        shutil.copy2(package/'payload'/key,temporary)
        temporary.replace(target)
    call('systemctl','start','pld-hub.service')
    print('QUAD_PATCH_APPLIED')
    print('Backup: '+str(backup))
    print('Wait for model initialization, then Ctrl+F5: http://192.168.2.99:8765/')


if __name__ == '__main__':
    main()
