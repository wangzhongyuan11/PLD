"""Exercise the real child protocol with software-only PYNQ/render fixtures."""
import importlib.util
from pathlib import Path
import sys
import time

import pytest

spec=importlib.util.spec_from_file_location('warm_hdmi',Path(__file__).parents[1]/'warm_hdmi.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def agent_fixture(tmp_path,monkeypatch):
    sdk=tmp_path/'pynq'
    files={
        '__init__.py':"__version__='3.1.1'\nclass Overlay:\n def __init__(self,*args,download=True,**kw):\n  assert download is False,'software preparation downloaded a bitstream'\n",
        'lib/__init__.py':'','lib/video.py':'','pl_server/__init__.py':'',
        'pl_server/embedded_device.py':'',
        'pl_server/device.py':"from types import SimpleNamespace\nclass Device:\n active_device=SimpleNamespace(_pld_metadata_cache=True,get_bitfile_metadata=lambda path:{})\n",
    }
    for name,text in files.items():
        p=sdk/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path))
    script=tmp_path/'renderer.py'
    script.write_text("from pathlib import Path\ndef main(argv=None,stop=None):\n p=Path(argv[0])\n try:\n  if p.name=='bad':raise ValueError('render failure')\n  p.with_suffix('.opened').write_text('open')\n  while not stop.wait(.01):\n   update=getattr(stop,'display_update',None)\n   if update:\n    stop.display_update=None\n    p.with_suffix('.updated').write_text(update['view'])\n finally:p.with_suffix('.closed').write_text('closed')\n")
    return module.WarmHDMI(tmp_path/'worker.log'),script


def test_stop_immediately_after_show_acknowledges_cleanup_before_exit(tmp_path,monkeypatch):
    agent,script=agent_fixture(tmp_path,monkeypatch)
    session=agent.start([sys.executable,'-u',str(script),str(tmp_path/'ok')],tmp_path/'frame.log')
    session.terminate()
    assert session.wait(5)==0
    assert (tmp_path/'ok.closed').exists()
    agent.close()
    assert agent.process.returncode==0


def test_render_failure_remains_failure_on_worker_shutdown(tmp_path,monkeypatch):
    agent,script=agent_fixture(tmp_path,monkeypatch)
    session=agent.start([sys.executable,str(script),str(tmp_path/'bad')],tmp_path/'frame.log')
    assert session.wait(5)==1
    assert (tmp_path/'bad.closed').exists()
    with pytest.raises(RuntimeError,match='close normally'):agent.close()


def test_view_update_keeps_the_existing_session_and_child(tmp_path,monkeypatch):
    agent,script=agent_fixture(tmp_path,monkeypatch)
    session=agent.start([sys.executable,str(script),str(tmp_path/'ok')],tmp_path/'frame.log')
    pid=agent.process.pid
    session.update('label')
    deadline=time.monotonic()+3
    while not (tmp_path/'ok.updated').exists() and time.monotonic()<deadline:time.sleep(.01)
    assert (tmp_path/'ok.updated').read_text()=='label'
    assert session.poll() is None and agent.process.pid==pid
    session.terminate();assert session.wait(5)==0
    agent.close()
