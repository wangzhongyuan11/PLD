"""CAMUS display ownership compatible with a paused skin application."""
import http.client
import json
from pathlib import Path
import subprocess
import tempfile


def skin_released():
    result=subprocess.run(['systemctl','show','pld-web.service','-p','ActiveState','-p','MainPID','-p','Result'],
                          check=True,capture_output=True,text=True,timeout=5)
    props=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    if props.get('ActiveState')=='inactive' and props.get('MainPID')=='0' and props.get('Result')=='success':return
    if props.get('ActiveState')!='active':raise RuntimeError('skin service has not released hardware')
    connection=http.client.HTTPConnection('127.0.0.1',8766,timeout=5)
    try:
        connection.request('GET','/api/ownership');response=connection.getresponse()
        if response.status!=200:raise RuntimeError('skin ownership acknowledgement unavailable')
        state=json.loads(response.read())
    finally:connection.close()
    if not (str(state.get('owner_pid'))==props.get('MainPID') and state.get('lifecycle')=='pause-resume-v1'
            and state.get('suspended') and state.get('hardware_released') and state.get('hdmi')=='off'
            and not state.get('active') and not state.get('fault')):
        raise RuntimeError('skin service has not acknowledged hardware release')


class CamusDisplayLease:
    def __init__(self):
        import fcntl
        skin_released()
        self.handle=(Path(tempfile.gettempdir())/'pld-camus-pl.lock').open('a+b')
        try:fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.close();raise

    def close(self):
        if self.handle is not None:self.handle.close();self.handle=None
