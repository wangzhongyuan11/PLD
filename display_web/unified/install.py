"""Install reversible drop-ins around the already-installed board services."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
SYSTEM=Path('/etc/systemd/system')
PYTHON='/usr/local/share/pynq-venv/bin/python3'
UNITS={'skin':'pld-web.service','camus':'pld-camus.service'}
RECEIPT=ROOT/'installation.json'

def run(*args):
    return subprocess.run(args,check=True,capture_output=True,text=True,timeout=300).stdout.strip()

def safe_path(value):
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',str(value)) or '..' in str(value).split('/'):
        raise ValueError('路径包含不支持的字符：'+str(value))
    return str(value)

def command_from_unit(text):
    """Parse supported ExecStart without executing any shell from the unit."""
    section='';commands=[]
    for line in text.replace('\\\n',' ').splitlines():
        line=line.strip()
        if line.startswith('['):section=line
        if section=='[Service]' and line.startswith('ExecStart='):
            value=line.split('=',1)[1]
            if not value:commands=[]
            else:commands.append(value)
    if len(commands)!=1:raise ValueError('无法唯一确定原服务 ExecStart')
    args=shlex.split(commands[0])
    if args[:2]==['/bin/bash','-lc'] and len(args)==3:
        args=shlex.split(args[2])
        if args[:1]==['exec']:args=args[1:]
    if not args or args[0]!=PYTHON or any(x in args for x in [';','&&','||','|','&']):
        raise ValueError('原服务启动命令不是已支持的 PYNQ Python 方式')
    return args[1:]

def option(args,name,default=None):
    if args.count(name)>1:raise ValueError('重复参数：'+name)
    if name not in args:return default
    i=args.index(name)
    if i+1>=len(args):raise ValueError('缺少参数值：'+name)
    return args[i+1]

def inspect_backend(kind,accepted):
    unit=UNITS[kind]
    state=run('systemctl','show',unit,'--property=ActiveState','--value')
    if state!='inactive':raise ValueError(unit+' 必须先正常停止；当前为 '+state)
    load=run('systemctl','show',unit,'--property=LoadState','--value')
    if load!='loaded':raise ValueError(unit+' 尚未安装，请先完成原模型的独立部署')
    source=Path(safe_path(run('systemctl','show',unit,'--property=WorkingDirectory','--value')))
    args=command_from_unit(run('systemctl','cat',unit))
    if kind=='skin':
        if not any(Path(a).name=='server.py' for a in args):raise ValueError('皮肤镜启动入口不匹配')
        if '--portable' in args:raise ValueError('皮肤镜服务不是板端运行方式')
        runtime=option(args,'--runtime')
        if not runtime:raise ValueError('原皮肤镜服务缺少 --runtime')
        data=option(args,'--data',str(source/'data'))
    else:
        if '-m' not in args or option(args,'-m')!='camus.web' or option(args,'--backend')!='pl':
            raise ValueError('CAMUS 原服务必须使用 -m camus.web --backend pl')
        runtime=option(args,'--runtime')
        if not runtime:raise ValueError('CAMUS 缺少 --runtime')
        data=option(args,'--data',str(source/'out/camus/web'))
    runtime=Path(safe_path(runtime));data=Path(safe_path(data))
    if not runtime.is_dir() or not data.is_dir():raise ValueError('运行包或结果目录不存在')
    for name,hashes in accepted[kind].items():
        path=source/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() not in hashes:
            raise ValueError('代码版本与已审查版本不同：'+str(path)+'；请提供该文件后适配，不要跳过检查')
    if kind=='skin':
        worker=runtime/'board/infer_worker.py'
        if not worker.is_file() or 'if request is None:' not in worker.read_text():raise ValueError('推理进程不支持正常 EOF 退出')
        extra=['--runtime',str(runtime),'--data',str(data),'--persistent-b-worker']
    else:
        model=Path(safe_path(option(args,'--model',str(source/'model/int8_export'))))
        if not model.is_dir():raise ValueError('CAMUS 模型目录不存在')
        extra=['--backend','pl','--runtime',str(runtime),'--model',str(model),'--data',str(data)]
    return dict(source=str(source),runtime=str(runtime),data=str(data),extra=extra)

def dropin(kind,backend,root):
    other=UNITS['camus' if kind=='skin' else 'skin']
    port='8766' if kind=='skin' else '8090'
    args=[PYTHON,'-u',safe_path(root/'backend_launcher.py'),kind,backend['source'],'--',
          '--host','127.0.0.1','--port',port]+backend['extra']
    command=shlex.join(args)
    # Every supplied path is restricted above; no quote is present in command.
    return f'''[Unit]
Conflicts={other}

[Service]
Type=simple
ExecStart=
ExecStart=/bin/bash -lc 'exec {command}'
Restart=no
KillMode=mixed
KillSignal={'SIGINT' if kind=='skin' else 'SIGTERM'}
SendSIGKILL=no
TimeoutStopSec=240
'''

def hub_unit(root):
    root=safe_path(root)
    return f'''[Unit]
Description=PLD single-owner dual-model workspace
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={root}
ExecStart={PYTHON} -u {root}/hub.py --config {root}/config.json
Restart=no
KillSignal=SIGTERM
SendSIGKILL=no
TimeoutStopSec=1200
UMask=0077

[Install]
WantedBy=multi-user.target
'''

def enabled(unit):
    result=subprocess.run(['systemctl','is-enabled',unit],capture_output=True,text=True)
    value=result.stdout.strip()
    if value not in ('enabled','disabled'):raise ValueError(unit+' 的启用状态不受支持：'+value)
    return value

def verify_no_rogue():
    # Standard units must be stopped, so any matching live owner is unexpected.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().decode(errors='replace').split('\x00')
        except (OSError,ProcessLookupError):continue
        names={Path(a).name for a in args if a}
        if names.intersection({'server.py','infer_worker.py','infer_images.py','hdmi_worker.py','backend_launcher.py'}) or any(a in ('camus.web','camus.hdmi') for a in args):
            raise ValueError('仍存在模型/显示进程 PID '+entry.name+'，请正常停止后再安装')

def install():
    safe_path(ROOT)
    if RECEIPT.exists():raise ValueError('已存在安装记录；不要覆盖解压正在使用的目录，需要更新时先回退')
    if not Path(PYTHON).is_file():raise ValueError('PYNQ Python 不存在')
    accepted=json.loads((ROOT/'accepted_backends.json').read_text())
    backends={kind:inspect_backend(kind,accepted) for kind in UNITS}
    verify_no_rogue()
    token=Path(backends['skin']['data'])/'.access_token'
    if not token.is_file() or len(token.read_text().strip())<24:raise ValueError('原皮肤镜访问码文件不存在或损坏')
    generated={SYSTEM/(unit+'.d')/'zz-pld-unified.conf':dropin(kind,backends[kind],ROOT) for kind,unit in UNITS.items()}
    generated[SYSTEM/'pld-hub.service']=hub_unit(ROOT)
    for p in generated:
        if p.exists():raise ValueError('目标配置已存在，未覆盖：'+str(p))
    states={unit:enabled(unit) for unit in UNITS.values()}
    receipt={'enabled':states,'files':{str(p):hashlib.sha256(s.encode()).hexdigest() for p,s in generated.items()},'backends':backends}
    (ROOT/'config.json').write_text(json.dumps({'token_file':str(token)},indent=2))
    # Save ownership before mutation, so a failed installation is recoverable.
    RECEIPT.write_text(json.dumps(receipt,indent=2))
    try:
        for p,s in generated.items():
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
        run('systemctl','daemon-reload')
        run('systemd-analyze','verify',str(SYSTEM/'pld-hub.service'),*[str(SYSTEM/u) for u in UNITS.values()])
        run('systemctl','disable',*UNITS.values())
        run('systemctl','enable','pld-hub.service')
    except Exception:
        print('安装中断，已保留 installation.json；运行 python3 install.py rollback 恢复。')
        raise
    print('INSTALL_PASS：原模型源码/权重/bitstream 未覆盖；统一入口已设开机启动。')
    print('现在执行：sudo systemctl start pld-hub.service')
    print('原网址 http://192.168.2.99:8765/ 保持不变（前提是板子 IP 未变）。')

def rollback():
    receipt=json.loads(RECEIPT.read_text())
    allowed={str(SYSTEM/(u+'.d')/'zz-pld-unified.conf') for u in UNITS.values()}|{str(SYSTEM/'pld-hub.service')}
    if set(receipt['files'])!=allowed or set(receipt['enabled'])!=set(UNITS.values()):raise ValueError('安装记录无效')
    for p,h in receipt['files'].items():
        path=Path(p)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise ValueError('安装后的配置被修改，停止自动回退：'+p)
    for unit in ['pld-hub.service',*UNITS.values()]:
        state=run('systemctl','show',unit,'--property=ActiveState','--value')
        if state!='inactive':raise ValueError('先正常停止 '+unit+'，确认 inactive 后再回退')
    verify_no_rogue()
    run('systemctl','disable','pld-hub.service')
    for p in receipt['files']:Path(p).unlink(missing_ok=True)
    run('systemctl','daemon-reload')
    for unit,state in receipt['enabled'].items():
        if state not in ('enabled','disabled'):raise ValueError('启用状态无效')
        run('systemctl','enable' if state=='enabled' else 'disable',unit)
    RECEIPT.rename(ROOT/'installation.rolled_back.json')
    print('ROLLBACK_PASS：已恢复原服务配置和开机启用状态；可执行 sudo systemctl start pld-web.service')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['install','rollback'],nargs='?',default='install');args=parser.parse_args()
    if sys.platform!='linux' or os.geteuid()!=0:raise SystemExit('请在板子上用 sudo 执行，不是在 Windows 上执行')
    try:(install if args.action=='install' else rollback)()
    except Exception as exc:raise SystemExit('未完成：'+str(exc))

if __name__=='__main__':main()
