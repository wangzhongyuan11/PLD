import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import evaluation_bridge
import hub


def test_unconfigured_evaluation_returns_503_without_starting_any_backend():
    handler=SimpleNamespace(server=SimpleNamespace(evaluation=None),send=Mock())
    evaluation_bridge.handle(handler,'/evaluation',False,None)
    assert handler.send.call_args.args[0]==503


def test_evaluation_blocks_normal_requests_and_shutdown_joins_before_stopping_backends():
    events=[]
    controller=hub.Controller.__new__(hub.Controller)
    controller.lock=threading.RLock();controller.evaluation_busy=True
    controller.stopping=False;controller.proxy_busy=False
    controller.proxy_idle=threading.Event();controller.proxy_idle.set()
    with pytest.raises(RuntimeError,match='评测'):
        controller.proxy('skin',1,'POST','/api/jobs',b'{}','application/json')
    controller.evaluation=SimpleNamespace(cancelled=threading.Event(),
        thread=SimpleNamespace(join=lambda:events.append('evaluation joined')))
    controller.thread=None
    controller.board=SimpleNamespace(stop=lambda model:events.append(model),
        props=lambda model:dict(ActiveState='inactive'))
    controller.close()
    assert controller.evaluation.cancelled.is_set()
    assert events==['evaluation joined','skin','camus']
