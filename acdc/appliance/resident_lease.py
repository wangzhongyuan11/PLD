"""Application lease for acknowledged, paused three-model services."""
import http.client
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from acdc.hardware import Lease as OriginalLease


def properties(unit):
    response=subprocess.run(['systemctl','show',unit,'-p','ActiveState','-p','MainPID','-p','Result'],
                            check=True,capture_output=True,text=True,timeout=5)
    return dict(line.split('=',1) for line in response.stdout.splitlines() if '=' in line)


def status(port,path):
    connection=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
    try:
        connection.request('GET',path)
        response=connection.getresponse()
        if response.status!=200:raise RuntimeError('ownership status unavailable')
        return json.loads(response.read())
    finally:connection.close()


def released(unit,port,path,hub=False):
    props=properties(unit)
    if props.get('ActiveState')=='inactive' and props.get('MainPID')=='0':
        if props.get('Result') not in ('success',None,''):raise RuntimeError(unit+' failed; release requires inspection')
        return
    if props.get('ActiveState')!='active':raise RuntimeError(unit+' is transitioning or faulted')
    value=status(port,path)
    if str(value.get('owner_pid'))!=props.get('MainPID'):raise RuntimeError(unit+' status PID mismatch')
    if hub:
        idle=(value.get('phase')=='paused' and value.get('hardware_released')
              and not value.get('interactive_busy') and not value.get('evaluation_busy'))
    else:
        idle=(value.get('lifecycle')=='pause-resume-v1' and value.get('suspended')
              and value.get('hardware_released') and value.get('hdmi')=='off'
              and not value.get('active') and not value.get('fault'))
    if not idle:raise RuntimeError(unit+' has not acknowledged hardware release')


class ResidentLease(OriginalLease):
    def __init__(self):
        if sys.platform!='linux':raise RuntimeError('PL requires the Linux board')
        import fcntl
        self.handles=[]
        try:
            released('pld-hub.service',8765,'/hub/ownership',hub=True)
            released('pld-web.service',8766,'/api/ownership')
            released('pld-camus.service',8090,'/api/status')
            for path in [Path('/run/pld-unified-fpga.lock'),
                         *(Path(tempfile.gettempdir())/name for name in ('pld-fpga.lock','pld-camus-pl.lock'))]:
                handle=path.open('a+b');self.handles.append(handle)
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.close();raise


def install():
    # Only the application ownership boundary changes. The frozen accelerator,
    # DMA, allocator, model and numerical source files remain byte-identical.
    from acdc.optimization import device
    device.Lease=ResidentLease
