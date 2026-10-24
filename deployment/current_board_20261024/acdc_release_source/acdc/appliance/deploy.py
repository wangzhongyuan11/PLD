"""Install only the independent V2 units; retain V1 and original deployments."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from acdc.contract import ContractError,atomic_json,file_sha
from .identity import verify

UNITS=('pld-acdc-v2-engine.service','pld-acdc-v2-web.service')


def render(root,app_sha):
    root=Path(root).resolve()
    def command(role):
        shell='source /etc/profile; exec env '+shlex.quote('PYTHONPATH='+str(root/'v2_deps'))
        shell+=' /usr/local/share/pynq-venv/bin/python3 -m acdc.appliance '+role
        shell+=' --config '+shlex.quote(str(root/'v2_config.json'))+' --app-sha '+app_sha
        return '/bin/bash -lc '+json.dumps(shell)
    common=f'WorkingDirectory={root}\nGroup=xilinx\nUMask=0007\nEnvironment=PYTHONUNBUFFERED=1\nEnvironment=OMP_NUM_THREADS=1\nEnvironment=OPENBLAS_NUM_THREADS=1\nEnvironment=VTK_SMP_MAX_THREADS=1\n'
    engine='''[Unit]
Description=ACDC V2 standalone board engine
After=network.target pld-acdc.service pld-hub.service pld-web.service pld-camus.service
Conflicts=pld-acdc.service pld-hub.service pld-web.service pld-camus.service

[Service]
Type=notify
NotifyAccess=main
User=root
RuntimeDirectory=acdc-board-v2
RuntimeDirectoryMode=0770
'''+common+'ExecStart='+command('engine')+'''
TimeoutStartSec=180
TimeoutStopSec=90
KillMode=control-group
SendSIGKILL=no
Restart=no

[Install]
WantedBy=multi-user.target
'''
    web='''[Unit]
Description=ACDC V2 browser upload and board MPR service
After=network.target
Wants=pld-acdc-v2-engine.service

[Service]
Type=simple
User=xilinx
'''+common+'ExecStart='+command('web')+'''
TimeoutStopSec=30
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
'''
    return dict(zip(UNITS,(engine,web)))


def install(root,app_sha):
    import grp
    import pwd
    if os.geteuid()!=0:raise ContractError('install requires board root privileges')
    root=Path(root).resolve();config,_=verify(root,app_sha)
    base=Path('/home/xilinx/acdc_board_v2/releases').resolve()
    if not root.is_relative_to(base) or root==base:raise ContractError('V2 release must be inside its dedicated releases directory')
    owner=pwd.getpwnam('xilinx');group=grp.getgrnam('xilinx')
    data=Path(config['data_root']).resolve()
    if not data.is_relative_to(base.parent) or data.is_relative_to(base):raise ContractError('invalid V2 data directory')
    for directory in (data,data/'jobs',data/'inputs',data/'startup'):
        directory.mkdir(parents=True,exist_ok=True);os.chown(directory,owner.pw_uid,group.gr_gid);directory.chmod(0o770)
    # The new copy only: make it readable to the unprivileged V2 Web process.
    for path in (root,*root.rglob('*')):
        if path.is_symlink():raise ContractError('symlink in V2 release')
        os.chown(path,0,group.gr_gid)
        path.chmod(0o750 if path.is_dir() or path==root/config['mesh_binary'] else 0o640)
    state_root=Path('/var/lib/acdc-board-v2');state_root.mkdir(exist_ok=True)
    state_file=state_root/'deployment.json'
    previous=json.loads(state_file.read_text()) if state_file.exists() else None
    directory=Path('/etc/systemd/system');units=render(root,app_sha)
    old={}
    for name in UNITS:
        path=directory/name
        if path.exists():
            if previous is None or file_sha(path)!=previous['unit_sha256'].get(name):
                raise ContractError('existing V2 unit has unrecognized edits: '+name)
            old[name]=path.read_text()
    # Verify both before mutating either unit.
    for name,text in units.items():(directory/name).write_text(text)
    state=dict(release=str(root),application_sha256=app_sha,previous=previous,
               previous_units=old,unit_sha256={n:file_sha(directory/n) for n in UNITS})
    atomic_json(state_file,state)
    subprocess.run(['systemctl','daemon-reload'],check=True,timeout=30)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--app-sha',required=True)
    args=parser.parse_args();state=install(args.release,args.app_sha)
    print(json.dumps({k:state[k] for k in ('release','application_sha256')}))
