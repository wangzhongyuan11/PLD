from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from perf_eval import engines


def test_hub_child_still_requires_stopped_backends_and_the_original_file_lock(monkeypatch):
    monkeypatch.setattr(engines.sys,'platform','linux')
    monkeypatch.setattr(engines.os,'getppid',lambda:101)
    monkeypatch.setenv('PLD_EVAL_HUB_PARENT','101')
    states={'pld-hub.service':('active',101),'pld-web.service':('inactive',0),'pld-camus.service':('inactive',0)}
    def props(args,**kwargs):
        state,pid=states[args[2]]
        return SimpleNamespace(stdout=f'ActiveState={state}\nMainPID={pid}\n')
    monkeypatch.setattr(engines.subprocess,'run',props)
    handle=Mock();opened=Mock(return_value=handle);lock=Mock()
    monkeypatch.setattr(engines,'open',opened,raising=False)
    monkeypatch.setitem(engines.sys.modules,'fcntl',SimpleNamespace(flock=lock,LOCK_EX=1,LOCK_NB=2))
    lease=engines.Lease()
    opened.assert_called_once_with('/run/pld-unified-fpga.lock','a+b')
    lock.assert_called_once_with(handle,3)
    lease.close();handle.close.assert_called_once()
    opened.reset_mock();states['pld-web.service']=('active',202)
    with pytest.raises(RuntimeError,match='normally stopped'):engines.Lease()
    opened.assert_not_called()
    states['pld-web.service']=('inactive',0);states['pld-hub.service']=('active',999)
    with pytest.raises(RuntimeError,match='normally stopped'):engines.Lease()
    opened.assert_not_called()
