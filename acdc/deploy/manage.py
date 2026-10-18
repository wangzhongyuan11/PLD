"""Board-local install/activate/stop/rollback. No SSH or changes to legacy unit files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from acdc.contract import ContractError,atomic_json,file_sha

OLD_UNITS=('pld-hub.service','pld-web.service','pld-camus.service')
UNIT='pld-acdc.service'


def command(argv):
    return subprocess.run(argv,capture_output=True,text=True,timeout=120)


class Manager:
    def __init__(self,state_dir=Path('/var/lib/pld-acdc-manager'),unit_path=Path('/etc/systemd/system')/UNIT,run=command):
        self.state_dir=Path(state_dir);self.unit_path=Path(unit_path);self.run=run

    def invoke(self,*args):
        r=self.run(['systemctl',*args])
        if r.returncode:raise RuntimeError('systemctl '+args[0]+' failed: '+r.stderr.strip())
        return r.stdout

    def props(self,unit):
        text=self.invoke('show',unit,'-p','ActiveState','-p','MainPID','-p','ControlGroup','-p','Result','-p','ExecMainStatus')
        return dict(line.split('=',1) for line in text.splitlines() if '=' in line)

    def state(self):
        p=self.state_dir/'state.json'
        if not p.is_file():raise ContractError('ACDC unit has not been installed by this manager')
        s=json.loads(p.read_text('utf-8'))
        if not self.unit_path.is_file() or file_sha(self.unit_path)!=s['managed_unit_sha256']:
            raise ContractError('unit changed outside ACDC manager; refusing to overwrite or switch')
        return s

    def install(self,release,spec_sha):
        if not re.fullmatch(r'/home/xilinx/jupyter_notebooks/pld/releases/acdc_[A-Za-z0-9_]+',release):
            raise ContractError('versioned ACDC release path required')
        if not re.fullmatch('[0-9a-f]{64}',spec_sha):raise ContractError('invalid model SHA')
        previous=None
        prior_state=None
        if self.unit_path.exists():
            prior_state=self.state()
            if self.props(UNIT).get('ActiveState') not in ('inactive','failed'):
                raise ContractError('stop the current ACDC worker before installing another version')
            previous=self.unit_path.read_text('utf-8')
        text=Path(__file__).with_name('acdc-worker.service.in').read_text('utf-8')
        text=text.replace('@RELEASE@',release).replace('@SPEC_SHA@',spec_sha)
        self.state_dir.mkdir(parents=True,exist_ok=True)
        review=self.state_dir/f'acdc_candidate_{time.time_ns()}.service';review.write_text(text,encoding='utf-8')
        checked=self.run(['systemd-analyze','verify',str(review)])
        if checked.returncode:raise ContractError('systemd unit validation failed: '+checked.stderr)
        state=dict(release=release,spec_sha=spec_sha,previous_unit=previous,previous_state=prior_state,previous_active=[],
                   managed_unit_sha256=hashlib.sha256(text.encode()).hexdigest())
        # Save rollback evidence before the only system configuration write.
        atomic_json(self.state_dir/'state.pending.json',state)
        self.unit_path.write_text(text,encoding='utf-8',newline='\n')
        try:self.invoke('daemon-reload')
        except BaseException:
            if previous is None:self.unit_path.unlink()
            else:self.unit_path.write_text(previous,encoding='utf-8',newline='\n')
            self.run(['systemctl','daemon-reload'])
            raise
        atomic_json(self.state_dir/'state.json',state)
        return state

    def stopped(self,unit):
        s=self.props(unit)
        if s.get('MainPID')!='0' or s.get('ActiveState') not in ('inactive','failed'):
            raise ContractError(unit+' has not stopped; DMA ownership is uncertain')
        group=s.get('ControlGroup','')
        if group:
            base=Path('/sys/fs/cgroup').resolve();group_path=(base/group.lstrip('/')).resolve()
            if not group_path.is_relative_to(base):raise ContractError('invalid cgroup path')
            p=group_path/'cgroup.procs'
            if p.is_file() and p.read_text().strip():raise ContractError(unit+' still has live child processes')
        return s

    def activate(self):
        state=self.state()
        if self.props(UNIT).get('ActiveState')=='active':raise ContractError('ACDC is already active')
        previous=[u for u in OLD_UNITS if self.props(u).get('ActiveState')=='active']
        state['previous_active']=previous;atomic_json(self.state_dir/'state.json',state)
        for u in OLD_UNITS:
            self.invoke('stop',u);self.stopped(u)
        self.invoke('start',UNIT)
        if self.props(UNIT).get('ActiveState')!='active':raise ContractError('ACDC did not become active')
        # Worker/overlay initialization can take time: actual /health must be checked by the operator/PC handshake.
        return dict(status='started',health='http://127.0.0.1:8093/health',previous_active=previous)

    def stop(self,restore=True):
        state=self.state();self.invoke('stop',UNIT);p=self.stopped(UNIT)
        if p.get('Result')!='success' or p.get('ExecMainStatus')!='0':
            raise ContractError('ACDC did not exit cleanly; refuse to hand DMA to another model')
        if restore:
            previous=state.get('previous_active',[])
            # Hub owns backend selection. Do not separately start both backends.
            units=['pld-hub.service'] if 'pld-hub.service' in previous else [u for u in previous if u in OLD_UNITS]
            if len(units)>1:raise ContractError('ambiguous legacy hardware owners in saved state')
            for u in units:self.invoke('start',u)
        return dict(status='stopped',restored=restore)

    def rollback(self):
        state=self.state();self.stop(restore=False)
        old=state.get('previous_unit')
        if old is None:self.unit_path.unlink() # exact owned file, hash checked by state(), never a recursive delete
        else:self.unit_path.write_text(old,encoding='utf-8',newline='\n')
        self.invoke('daemon-reload')
        previous=state.get('previous_active',[])
        if 'pld-hub.service' in previous:self.invoke('start','pld-hub.service')
        elif len(previous)==1:self.invoke('start',previous[0])
        atomic_json(self.state_dir/'last_rollback.json',dict(restored_unit=old is not None,release=state['release']))
        if state.get('previous_state') is not None:atomic_json(self.state_dir/'state.json',state['previous_state'])
        return dict(status='rolled_back',release_files_preserved=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    install=sub.add_parser('install');install.add_argument('--release',required=True);install.add_argument('--spec-sha',required=True)
    for action in ('activate','stop','rollback'):sub.add_parser(action)
    args=p.parse_args()
    if sys.platform!='linux' or os.geteuid()!=0:p.error('run locally on the PYNQ board as root')
    manager=Manager()
    if args.action=='install':
        from acdc.model import load_bundle
        from acdc.package import verify_runtime
        release=Path(args.release).resolve()
        model=load_bundle(release/'model',expected_sha=args.spec_sha)
        verify_runtime(release,model)
        result=manager.install(str(release),args.spec_sha)
    else:result=getattr(manager,args.action)()
    print(json.dumps(result))


if __name__=='__main__':main()
