"""The inner hub must not reintroduce the entry's removed head-of-line wait."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import hub
from interactive_bridge import Manager as Interactive
from evaluation_bridge import Manager as Evaluation


@pytest.mark.parametrize('fail',[False,True])
def test_reads_remain_live_but_mutations_and_switches_cannot_overtake_post(tmp_path,fail):
    entered=threading.Event(); release=threading.Event(); errors=[]
    def request(model,method,*args):
        if method=='POST':
            entered.set();assert release.wait(3)
            if fail:raise TimeoutError('lost backend reply')
        return 200,'application/json',b'{}'
    board=SimpleNamespace(props=lambda _:dict(ActiveState='inactive'),request=request)
    c=hub.Controller(board,tmp_path);c.model='skin';c.phase='ready';c.generation=1
    def post():
        try:c.proxy('skin',1,'POST','/api/hdmi',b'{}','application/json')
        except Exception as exc:errors.append(exc)
    t=threading.Thread(target=post);t.start()
    try:
        assert entered.wait(1)
        assert c.snapshot()['proxy_busy']
        assert c.proxy('skin',1,'GET','/api/status',None,'application/json')[0]==200
        with pytest.raises(RuntimeError):c.switch('camus')
        with pytest.raises(RuntimeError):c.lifecycle('pause')
        with pytest.raises(RuntimeError):c.proxy('skin',1,'POST','/api/jobs',b'{}','application/json')
        roi=Interactive.__new__(Interactive);roi.controller=c
        assert roi._busy()
        evaluation=Evaluation.__new__(Evaluation);evaluation.controller=c
        with pytest.raises(RuntimeError):evaluation.submit(dict(model='skin',generation=1))
    finally:release.set();t.join(2)
    assert not t.is_alive() and not c.proxy_busy and c.proxy_idle.is_set()
    assert len(errors)==int(fail)


def test_shutdown_waits_for_unacknowledged_post(tmp_path):
    entered=threading.Event();release=threading.Event()
    def request(*args):entered.set();assert release.wait(3);return 202,'application/json',b'{}'
    board=SimpleNamespace(props=lambda _:dict(ActiveState='inactive'),request=request,stop=Mock())
    c=hub.Controller(board,tmp_path);c.model='skin';c.phase='ready';c.generation=1
    post=threading.Thread(target=lambda:c.proxy('skin',1,'POST','/api/jobs',b'{}','application/json'))
    closing=threading.Thread(target=c.close)
    post.start();assert entered.wait(1);closing.start()
    try:
        assert not c.proxy_idle.wait(.1);board.stop.assert_not_called()
    finally:release.set();post.join(2);closing.join(2)
    assert not closing.is_alive() and board.stop.call_count==2
