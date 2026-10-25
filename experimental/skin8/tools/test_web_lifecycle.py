import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import Mock

spec=importlib.util.spec_from_file_location('web_lifecycle',Path(__file__).with_name('web_lifecycle.py'))
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)


def fixture(monkeypatch):
    owner=SimpleNamespace(handle=object())
    owner.release=lambda:setattr(owner,'handle',None)
    owner.acquire=lambda:setattr(owner,'handle',object())
    monkeypatch.setattr(life,'OWNER',owner)
    app=SimpleNamespace(lock=threading.RLock(),active=None,warmup_state='ready',fault=None,
        stop_hdmi=Mock(),b_worker=SimpleNamespace(release_hardware=Mock()),data=Path('.'),
        status=lambda:dict(inference_available=True,hdmi='off'))
    app._warmup=lambda:setattr(app,'warmup_state','ready')
    class Handler:
        do_POST=Mock()
        do_GET=Mock()
        def __init__(self,action,path='/api/lifecycle'):
            self.path=path;self.client_address=('127.0.0.1',1)
            raw=json.dumps(dict(action=action)).encode();self.rfile=io.BytesIO(raw)
            self.headers={'Content-Length':str(len(raw))};self.reply=None
        def send(self,code,body):self.reply=(code,body)
    life.install(SimpleNamespace(Handler=Handler),app,'skin')
    return app,owner,Handler


def test_pause_releases_before_ack_and_resume_restores(monkeypatch):
    app,owner,handler=fixture(monkeypatch)
    request=handler('pause');request.do_POST()
    assert request.reply==(200,dict(suspended=True,hardware_released=True))
    app.b_worker.release_hardware.assert_called_once()
    assert not app.status()['inference_available'] and owner.handle is None
    request=handler('ignored','/api/jobs');request.do_POST();assert request.reply[0]==409
    request=handler('resume');request.do_POST()
    assert request.reply==(200,dict(suspended=False,hardware_released=False))
    assert app.status()['inference_available']


def test_failed_release_keeps_lease_and_does_not_claim_pause(monkeypatch):
    app,owner,handler=fixture(monkeypatch)
    app.b_worker.release_hardware.side_effect=RuntimeError('DMA not released')
    request=handler('pause');request.do_POST()
    assert request.reply[0]==409 and owner.handle is not None and not app.suspended
