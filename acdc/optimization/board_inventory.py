"""Read-only board inventory; never downloads an overlay or changes a service."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


def read(path):
    try:
        return Path(path).read_text().strip().strip('\x00')
    except OSError:
        return None


def inventory():
    packages = {}
    for name in ('pynq', 'numpy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    services = {}
    for unit in ('pld-hub.service','pld-web.service','pld-camus.service'):
        result = subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True,timeout=5)
        services[unit] = result.stdout.strip()
    mem = {}
    for line in (read('/proc/meminfo') or '').splitlines():
        key, value = line.split(':',1)
        if key in ('MemTotal','MemAvailable','CmaTotal','CmaFree','SwapTotal','SwapFree'):
            mem[key] = value.strip()
    result = dict(schema='pld-acdc-board-inventory-v1',read_only=True,
                  machine=platform.machine(),kernel=platform.release(),
                  model=read('/sys/firmware/devicetree/base/model'),
                  python=platform.python_version(),packages=packages,services=services,
                  memory=mem,cpus=os.cpu_count(),
                  cpu_khz=read('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'),
                  fpga_state=read('/sys/class/fpga_manager/fpga0/state'))
    try:
        from pynq import Clocks
        result['fclk_mhz']=[float(getattr(Clocks,f'fclk{i}_mhz')) for i in range(4)]
    except Exception as exc:
        result['fclk_read_error']=type(exc).__name__+': '+str(exc)
    return result


if __name__ == '__main__':
    print(json.dumps(inventory(),indent=2))
