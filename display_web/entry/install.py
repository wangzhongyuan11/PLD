"""Install/rollback only the new entry unit and its boot selection. Board root required."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
UNIT = 'pld-entry.service'
MANAGED = ('pld-acdc-v2-engine.service', 'pld-acdc-v2-web.service', 'pld-hub.service')
ENGINE_DROPIN = Path('/etc/systemd/system/pld-acdc-v2-engine.service.d/96-entry-hdmi-shutdown.conf')
ENGINE_POLICY = '[Service]\n# The engine owns the ordered HDMI shutdown.\nKillMode=mixed\n'
RECORD = ROOT / 'installation.json'
UNIT_PATH = Path('/etc/systemd/system') / UNIT


def systemctl(*args):
    return subprocess.run(['systemctl', *args], capture_output=True, text=True, timeout=1200)


def checked(*args):
    result = systemctl(*args)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def install():
    if RECORD.exists() or UNIT_PATH.exists():
        raise RuntimeError('入口已安装；更新前先回退，保留安装记录')
    hub = checked('show', 'pld-hub.service', '-p', 'WorkingDirectory', '--value')
    if not (Path(hub) / 'hub.py').is_file():
        raise RuntimeError('未找到原双模型入口')
    for unit in MANAGED:
        if checked('show', unit, '-p', 'LoadState', '--value') != 'loaded':
            raise RuntimeError('缺少服务 ' + unit)
    text = f'''[Unit]
Description=PLD three-model entry (default cardiac 2.5D)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={ROOT}
ExecStart=/usr/local/share/pynq-venv/bin/python3 -u {shlex.quote(str(ROOT / 'entry.py'))} --hub-root {shlex.quote(hub)}
Restart=no
SendSIGKILL=no
TimeoutStopSec=1500
UMask=0077

[Install]
WantedBy=multi-user.target
'''
    state = dict(enabled={unit: systemctl('is-enabled', unit).stdout.strip() for unit in MANAGED}, unit=text,
                 engine_dropin=ENGINE_DROPIN.read_text() if ENGINE_DROPIN.exists() else None)
    RECORD.write_text(json.dumps(state, indent=2) + '\n')
    UNIT_PATH.write_text(text)
    ENGINE_DROPIN.parent.mkdir(parents=True, exist_ok=True)
    ENGINE_DROPIN.write_text(ENGINE_POLICY)
    checked('daemon-reload')
    for unit in MANAGED:
        checked('disable', unit)
    checked('enable', UNIT)
    checked('start', UNIT)
    print('ENTRY_INSTALLED http://192.168.2.99:8088/')


def rollback():
    state = json.loads(RECORD.read_text())
    if UNIT_PATH.exists() and UNIT_PATH.read_text() != state['unit']:
        raise RuntimeError('入口 unit 已被修改，保留现场')
    if 'engine_dropin' in state and (not ENGINE_DROPIN.exists() or ENGINE_DROPIN.read_text() != ENGINE_POLICY):
        raise RuntimeError('HDMI 停止策略已被修改，保留现场')
    checked('stop', UNIT)
    # Leave original services and data intact; return hardware to the previous default.
    checked('stop', 'pld-hub.service')
    checked('disable', UNIT)
    UNIT_PATH.unlink(missing_ok=True)
    if 'engine_dropin' in state:
        if state['engine_dropin'] is None:
            ENGINE_DROPIN.unlink()
        else:
            ENGINE_DROPIN.write_text(state['engine_dropin'])
    for unit, enabled in state['enabled'].items():
        checked('enable' if enabled == 'enabled' else 'disable', unit)
    checked('daemon-reload')
    checked('start', 'pld-acdc-v2-web.service')
    RECORD.rename(ROOT / 'installation.rolled-back.json')
    print('ENTRY_ROLLBACK_PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'rollback'), nargs='?', default='install')
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise RuntimeError('需要板端 root 权限')
    (install if args.action == 'install' else rollback)()
