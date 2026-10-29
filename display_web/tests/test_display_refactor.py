"""Notification, preview and first-frame error boundaries; no real hardware."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import display_completion as completion
import quad_display
import server
import warm_hdmi
import pytest


def test_notification_retries_same_identity_and_closes_every_connection():
    first=Mock(); first.getresponse.side_effect=TimeoutError('lost response')
    second=Mock(); second.getresponse.return_value.status=200
    second.getresponse.return_value.read.return_value=b'{"token":"accepted"}'
    with patch.object(completion.sys,'platform','linux'), patch.object(completion.http.client,'HTTPConnection',side_effect=[first,second]):
        c=completion.Completion('skin','job')
    assert c.token=='accepted'
    assert first.request.call_args.args[2]==second.request.call_args.args[2]
    assert len(json.loads(first.request.call_args.args[2])['request_id'])==32
    first.close.assert_called_once(); second.close.assert_called_once()


def test_begin_failure_is_visible_and_finish_does_not_start_late_task():
    conn=Mock(); conn.getresponse.side_effect=TimeoutError('entry unavailable')
    with patch.object(completion.sys,'platform','linux'),patch.object(completion.http.client,'HTTPConnection',return_value=conn) as factory:
        c=completion.Completion('skin','job'); result=c.finish(True)
    assert factory.call_count==2
    assert not result['accepted'] and 'unavailable' in result['error']


def test_preview_path_avoids_full_png_decode_and_preserves_exports(tmp_path):
    for name,color in [('original',(200,10,20)),('mask',(255,255,255)),('outlined',(10,200,20))]:
        Image.new('RGB',(100,60),color).save(tmp_path/('preview_'+name+'.png'))
        (tmp_path/(name+'.png')).write_bytes(b'full exports must not be opened')
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    screen=quad_display.skin_frame(tmp_path)
    assert screen.shape==(480,640,3) and screen.dtype==np.uint8
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()}==before


def test_old_job_without_previews_still_renders(tmp_path):
    for name in ('original','mask','outlined'):Image.new('RGB',(40,30),'white').save(tmp_path/(name+'.png'))
    assert quad_display.skin_frame(tmp_path).shape==(480,640,3)


def test_async_warmup_ack_and_early_cancel_never_send_show(tmp_path):
    agent=warm_hdmi.WarmHDMI(tmp_path/'worker.log')
    agent.process=Mock(); agent.process.poll.return_value=None
    with patch.object(agent,'prepare'),patch.object(agent,'send') as send:
        session=agent.start(['python','renderer.py'],tmp_path/'frame.log')
        assert not agent.ready.is_set() and session.poll() is None
        session.terminate(); assert session.wait(1)==0
        agent.ready.set()
        send.assert_not_called()


def test_first_frame_timeout_does_not_claim_hardware_is_released(tmp_path):
    agent=warm_hdmi.WarmHDMI(tmp_path/'worker.log')
    agent.process=Mock(); agent.process.poll.return_value=None; agent.ready.set()
    with patch.object(agent,'prepare'),patch.object(agent,'send') as send:
        session=agent.start(['python','renderer.py'],tmp_path/'frame.log',ready_path=tmp_path/'missing',first_frame_timeout=0)
        # Launcher may send stop, but only a child idle acknowledgement may
        # set finished. A blocked native driver is not a successful cleanup.
        for _ in range(50):
            if session.failure:break
            threading.Event().wait(.01)
        assert 'timed out' in session.failure
        assert not session.finished.is_set() and session.poll() is None
        assert [call.args[0]['action'] for call in send.call_args_list]==['show','stop']


def test_cold_first_frame_timeout_blocks_future_hardware(tmp_path):
    app=server.Console(tmp_path,token='test',boot_id='boot'); process=Mock()
    process.poll.return_value=None; app.hdmi=process; app.hdmi_job='job'
    app._watch_hdmi(process,tmp_path,timeout=0)
    assert app.fault and app.hdmi is process
    process.terminate.assert_called_once()
    assert not app.status()['inference_available']
    app.hdmi=None


def test_timing_uses_first_frame_submission_as_separate_endpoint(tmp_path):
    app=server.Console(tmp_path,token='test',boot_id='boot')
    folder=tmp_path/'job';folder.mkdir();(folder/'hdmi.ready').write_text('ready')
    (folder/'hdmi.timing.json').write_text(json.dumps(dict(first_frame_monotonic=15,seconds=3)))
    app.jobs['job']=dict(zip_complete_monotonic=10,statistics=dict(request_to_job_complete_seconds=2))
    app.hdmi=Mock();app.hdmi.poll.return_value=None;app.hdmi_job='job'
    state=app.status()
    assert state['hdmi_timing']['request_to_first_frame_submit_seconds']==7
    assert app.jobs['job']['statistics']['request_to_job_complete_seconds']==2
    app.hdmi=None


def test_async_display_fault_blocks_submit_even_without_status_poll(tmp_path):
    app=server.Console(tmp_path,token='test',boot_id='boot')
    app.hdmi=Mock(failure='first frame timed out');app.hdmi.poll.return_value=None;app.hdmi_job='job'
    with pytest.raises(RuntimeError,match='timed out'):app.submit(dict(mode='demo'))
    assert app.active is None
    app.hdmi=None
