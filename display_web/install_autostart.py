"""Install a board-local systemd service, preserving application and access code.

Run with the PYNQ Python as root after stopping standalone hardware programs.
No IP changes, reboots, model changes, or automatic hardware-fault restarts.
"""
import argparse
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.request

APP = Path('/home/xilinx/jupyter_notebooks/PLD_Web')
RUNTIME = Path('/home/xilinx/jupyter_notebooks/pld/final_mode6_fastq_20260929_v3')
PYTHON = Path('/usr/local/share/pynq-venv/bin/python3')
UNIT_NAME = 'pld-web.service'


def unit_text():
    return '''[Unit]
Description=PLD image segmentation web console
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/xilinx/jupyter_notebooks/PLD_Web
Environment=PYTHONUNBUFFERED=1
ExecStart=/bin/bash -lc 'exec /usr/local/share/pynq-venv/bin/python3 /home/xilinx/jupyter_notebooks/PLD_Web/server.py --host 0.0.0.0 --port 8765 --data /home/xilinx/jupyter_notebooks/PLD_Web/data --runtime /home/xilinx/jupyter_notebooks/pld/final_mode6_fastq_20260929_v3'
KillSignal=SIGINT
KillMode=mixed
TimeoutStopSec=240
Restart=no
UMask=0077

[Install]
WantedBy=multi-user.target
'''


def hardware_processes(proc=Path('/proc')):
    names = {'server.py', 'hdmi_worker.py', 'show_result.py', 'infer_then_display.py',
             'infer_images.py', 'hdmi_buttons.py', 'verify_cnn.py', 'profile_cnn.py', 'evaluate_dataset.py'}
    found = []
    for directory in proc.iterdir():
        if not directory.name.isdigit(): continue
        try:
            args = (directory / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError, ProcessLookupError): continue
        if any(Path(arg).name in names for arg in args):
            found.append((directory.name, ' '.join(args)))
    return found


def check_status(url, token):
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer '+token})
    with urllib.request.urlopen(req, timeout=2) as response:
        result = json.load(response)
    if result.get('mode') != 'board' or not result.get('inference_available'):
        raise RuntimeError('网页已启动，但板端推理环境未就绪：'+str(result.get('runtime_error') or result.get('fault') or result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Only check environment; do not install')
    args = parser.parse_args()
    if platform.system() != 'Linux' or not platform.machine().lower().startswith('arm'):
        raise RuntimeError('请在 PYNQ 开发板运行此程序，而不是 Windows 电脑')
    if os.geteuid() != 0: raise RuntimeError('请通过 sudo -E 运行')
    if not Path('/run/systemd/system').is_dir() or not shutil.which('systemctl'):
        raise RuntimeError('未发现正在运行的 systemd，未修改系统')
    for f in (APP/'server.py', APP/'static/app.js', RUNTIME/'deployment_manifest.json', PYTHON):
        if not f.is_file(): raise RuntimeError('缺少文件：'+str(f))
    if 'def persistent_token(' not in (APP/'server.py').read_text(encoding='utf-8'):
        raise RuntimeError('请先安装已提供的固定网址更新包')
    (APP/'data').mkdir(exist_ok=True)
    # Import the installed application: keeps the exact same token and runtime checks.
    import sys
    sys.path.insert(0, str(APP))
    from server import Console
    app = Console(APP/'data', RUNTIME)
    if not app.runtime_ok: raise RuntimeError('计算包未就绪：'+str(app.runtime_error))
    if args.check:
        print('PREFLIGHT_PASS：固定网址与计算包检查通过；尚未安装自启')
        return
    active = subprocess.run(['systemctl', 'is-active', '--quiet', UNIT_NAME]).returncode == 0
    if active:
        print('服务已经运行，不覆盖或中断它。请使用 systemctl status pld-web.service 查看状态。')
        return
    owners = hardware_processes()
    if owners:
        raise RuntimeError('请先正常停止以下旧程序，再运行安装命令：\n'+ '\n'.join(pid+' '+cmd for pid,cmd in owners))
    unit_path = Path('/etc/systemd/system') / UNIT_NAME
    content = unit_text()
    with tempfile.TemporaryDirectory(prefix='pld-web-unit-') as tmp:
        candidate = Path(tmp) / UNIT_NAME
        candidate.write_text(content, encoding='utf-8')
        subprocess.run(['systemd-analyze', 'verify', str(candidate)], check=True)
        if unit_path.exists():
            backup = unit_path.with_name(UNIT_NAME+'.backup-'+str(time.time_ns()))
            shutil.copy2(unit_path, backup)
        shutil.copyfile(candidate, unit_path)
        unit_path.chmod(0o644)
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', 'enable', '--now', UNIT_NAME], check=True)
    error = None
    for _ in range(30):
        try:
            check_status('http://127.0.0.1:8765/api/status', app.token)
            if subprocess.run(['systemctl','is-active','--quiet',UNIT_NAME]).returncode:
                raise RuntimeError('网页可访问，但系统服务未保持运行')
            print('AUTOSTART_PASS：已启用开机自启，网页与板端推理环境检查通过。')
            print('固定网址：http://192.168.2.99:8765/ （前提是板子 IP 保持此地址）')
            print('未执行推理或 HDMI 测试，也未重启板子。请完成一次重启验收。')
            return
        except Exception as exc:
            error = exc
            time.sleep(1)
    raise RuntimeError('服务已配置，但启动验证失败：'+str(error)+'\n请执行：journalctl -u pld-web.service -n 60 --no-pager')


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print('安装未完成：'+str(exc))
        raise SystemExit(1)
