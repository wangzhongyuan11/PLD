"""Order PLD after PYNQ boot setup without restarting services or changing boot.py."""
import argparse
import os
from pathlib import Path
import subprocess


POLICY = Path(__file__).resolve().parents[1] / 'deployment/pynq_boot_order/99-pynq-boot-order.conf'
UNITS = ('pld-entry.service', 'pld-acdc-v2-engine.service', 'pld-web.service',
         'pld-camus.service', 'pld-hdmi.service')
SYSTEMD = Path('/etc/systemd/system')


def checked(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def apply(action):
    policy = POLICY.read_text(encoding='utf-8').encode('utf-8')
    paths = [SYSTEMD / (unit + '.d') / POLICY.name for unit in UNITS]
    # Check every destination before touching any installed file. These are
    # dedicated new drop-ins; rollback must not remove somebody else's edits.
    for unit, path in zip(UNITS, paths):
        if checked('systemctl', 'show', unit, '-p', 'LoadState', '--value') != 'loaded':
            raise RuntimeError('missing service: ' + unit)
        if path.exists() and path.read_bytes() != policy:
            raise RuntimeError('drop-in differs; preserve existing file: ' + str(path))
    if action == 'install':
        for unit in ('bootpy.service', 'clear_pl_statefile.service'):
            if checked('systemctl', 'is-enabled', unit) != 'enabled':
                raise RuntimeError('expected enabled PYNQ boot service: ' + unit)
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(policy)
            path.chmod(0o644)
    else:
        for path in paths:
            path.unlink(missing_ok=True)
    checked('systemctl', 'daemon-reload')
    for unit in UNITS:
        after = checked('systemctl', 'show', unit, '-p', 'After', '--value').split()
        if action == 'install' and not {'bootpy.service', 'clear_pl_statefile.service'} <= set(after):
            raise RuntimeError('boot order not applied: ' + unit)
    print('PYNQ_BOOT_ORDER_' + action.upper() + '_PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'rollback'))
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise RuntimeError('run as root on the PYNQ board')
    apply(args.action)
