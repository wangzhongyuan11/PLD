"""Install/switch the independent board service without editing the ISIC unit."""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


UNIT='pld-camus.service'


def command(*args):
    return subprocess.run(args,check=True,capture_output=True,text=True,timeout=100).stdout


def unit_text(runtime,dependencies,data,port):
    # Already-resolved Linux paths are supplied by install; keep generation
    # platform-independent so the service contract can be checked on Windows.
    paths=[str(p) for p in (runtime,dependencies,data)]
    if any(re.fullmatch(r'/[A-Za-z0-9_./-]+',p) is None for p in paths):
        raise ValueError('board service paths must not contain whitespace or systemd metacharacters')
    runtime,dependencies,data=paths
    # Login shell initializes the installed PYNQ/XRT environment on this board.
    invocation=shlex.join(['/usr/local/share/pynq-venv/bin/python3','-u','-m','camus.web',
        '--backend','pl','--runtime',runtime,'--host','127.0.0.1','--port',str(port),'--data',data])
    return f'''[Unit]
Description=Independent CAMUS CAM2 PL segmentation web
Conflicts=pld-web.service
After=pld-web.service network.target

[Service]
Type=simple
User=root
WorkingDirectory={runtime}
Environment=PYTHONPATH={dependencies}
ExecStart=/bin/bash -lc 'exec {invocation}'
TimeoutStartSec=90
TimeoutStopSec=90
Restart=no
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
'''


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('action',choices=('install','camus','isic','stop','status'))
    p.add_argument('--runtime',type=Path);p.add_argument('--dependencies',type=Path)
    p.add_argument('--data',type=Path);p.add_argument('--port',type=int,default=8090)
    args=p.parse_args(argv)
    if sys.platform!='linux' or os.geteuid()!=0:p.error('run through sudo on the PYNQ board')
    if args.action=='install':
        if None in (args.runtime,args.dependencies,args.data):p.error('install requires runtime/dependencies/data')
        if not 1024<=args.port<=65535:p.error('port must be 1024..65535')
        from .model import load_model
        from .package import verify_runtime
        root=args.runtime.resolve();model=load_model(root/'model/int8_export');verify_runtime(root,model.root)
        target=Path('/etc/systemd/system')/UNIT
        if target.exists():p.error('unit already exists; inspect the installed deployment before replacing it')
        text=unit_text(root,args.dependencies,args.data,args.port)
        with tempfile.TemporaryDirectory() as tmp:
            probe=Path(tmp)/UNIT;probe.write_text(text,encoding='utf-8')
            command('systemd-analyze','verify',str(probe))
        target.write_text(text,encoding='utf-8')
        command('systemctl','daemon-reload')
        print(json.dumps(dict(installed=str(target),boot_enabled=False,isic_unit_modified=False)))
    elif args.action in ('camus','isic'):
        active=command('systemctl','list-units','--type=service','--state=running','--plain','--no-legend','pld-camus-validation*.service')
        if active.strip():raise RuntimeError('board validation owns PL; wait for its completion before switching services')
        # systemd Conflicts + ordering stops the previous owner before startup.
        print(command('systemctl','start',UNIT if args.action=='camus' else 'pld-web.service'))
    elif args.action=='stop':print(command('systemctl','stop',UNIT))
    for name in (UNIT,'pld-web.service'):
        print(command('systemctl','show',name,'-p','ActiveState','-p','SubState','-p','MainPID','-p','WorkingDirectory'))


if __name__=='__main__':main()
