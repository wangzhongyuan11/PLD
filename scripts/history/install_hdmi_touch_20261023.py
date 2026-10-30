"""Board installer: isolate presentation changes, retaining current runtime copies.

Run as root with --source pointing to the uploaded checkout subset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--name',default='hdmi_touch_20261024');args=p.parse_args()
    source=args.source.resolve()
    def call(*cmd):return subprocess.check_output(cmd,text=True).strip()
    old=Path(call('systemctl','show','pld-acdc-v2-engine.service','-p','WorkingDirectory','--value'))
    new=old.parent/args.name
    ui=Path('/home/xilinx')/('pld_'+args.name)
    if new.exists() or ui.exists():raise RuntimeError('candidate directory already exists')
    shutil.copytree(old,new)
    for name in ('__main__.py','engine.py','web.py','hdmi.py','identity.py','web_assets/index.html','web_assets/app.js'):
        shutil.copy2(source/'acdc/appliance'/name,new/'acdc/appliance'/name)
    (new/'display_web').mkdir(exist_ok=True)
    for name in ('pld_hdmi.py','touch_display.py'):
        shutil.copy2(source/'display_web'/name,new/'display_web'/name)
    base=Path('/home/xilinx/jupyter_notebooks/pld/releases/pld_dual_20261017')
    core=Path('/home/xilinx/jupyter_notebooks/PLD_Dev')
    ui.mkdir()
    for name in ('web','hub'):shutil.copytree(base/name,ui/name)
    for name in ('camus','board','reference'):shutil.copytree(core/name,ui/name)
    (ui/'display_web').mkdir()
    for name in ('pld_hdmi.py','touch_display.py'):
        shutil.copy2(source/'display_web'/name,ui/'display_web'/name)
    shutil.copy2(source/'display_web/hdmi_worker.py',ui/'web/hdmi_worker.py')
    shutil.copy2(source/'display_web/touch_display.py',ui/'web/touch_display.py')
    shutil.copy2(source/'display_web/unified/quad_camus_hdmi.py',ui/'hub/quad_camus_hdmi.py')
    shutil.copy2(source/'camus/hdmi.py',ui/'camus/hdmi.py')
    original=Path('/home/xilinx/skin8_candidate_20261023/web_entry.py')
    text=original.read_text()
    replacements={
        'HERE=Path(__file__).resolve().parent':f'HERE=Path({str(original.parent)!r})\nUI=Path({str(ui)!r})',
        'sys.path.insert(0,str(CORE_ROOT))':'sys.path.insert(0,str(CORE_ROOT))\nsys.path.insert(0,str(UI))',
        "str(BASE/'web')":"str(UI/'web')",
        "str(BASE/'hub')":"str(UI/'hub')",
        '    guard_skin(old)':'    # HDMI uses the existing release_hardware lifecycle.',
        '    guard_camus(web)':'    # HDMI uses the existing pipeline.close lifecycle.',
        'hdmi_available=False':'hdmi_available=True',
    }
    for before,after in replacements.items():
        if before not in text:raise RuntimeError('unexpected Skin8 adapter: '+before)
        text=text.replace(before,after)
    # Keep the original policy import available for its status string only.
    text=text.replace('sys.path.insert(0,str(UI))',f'sys.path.insert(0,str(UI))\nsys.path.append({str(original.parent)!r})')
    (ui/'web_entry.py').write_text(text)
    # Child python -m camus.hdmi must resolve the same presentation copy.
    units={}
    for model in ('skin','camus'):
        unit='pld-web.service' if model=='skin' else 'pld-camus.service'
        directory=Path('/etc/systemd/system')/(unit+'.d');directory.mkdir(exist_ok=True)
        path=directory/'90-hdmi-touch.conf'
        if path.exists():raise RuntimeError('HDMI drop-in already exists')
        units[unit]=str(path)
        path.write_text('[Service]\nEnvironment="PYTHONPATH='+str(ui)+':'+str(core)+'"\nExecStart=\nExecStart=/bin/bash -lc "source /etc/profile; exec /usr/local/share/pynq-venv/bin/python3 '+str(ui/'web_entry.py')+' '+model+'"\n')
    import sys
    sys.path.insert(0,str(new))
    from acdc.appliance.identity import create
    from acdc.appliance.deploy import install
    sha=create(new);state=install(new,sha)
    record=dict(acdc_release=str(new),previous_acdc=str(old),application_sha256=sha,display_root=str(ui),dropins=units,
                skin8_adapter_sha256=hashlib.sha256(original.read_bytes()).hexdigest())
    (ui/'installation.json').write_text(json.dumps(record,indent=2)+'\n')
    subprocess.run(['systemctl','daemon-reload'],check=True)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
