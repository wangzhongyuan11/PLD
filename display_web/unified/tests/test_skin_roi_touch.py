import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'display_web'))
sys.path.insert(0,str(ROOT/'display_web/unified'))
from interactive_bridge import Manager
from interactive_hdmi import Surface
from interactive_workspace import parameters
from touch_drag import DragInput


def manager():
    m=Manager.__new__(Manager)
    m.controller=SimpleNamespace(model='skin',generation=4,lock=threading.RLock(),
        phase='ready',stopping=False,interactive_busy=False)
    m.guard=threading.Lock();m.job=None
    m.image_slot=dict(token='a'*32,width=100,height=50,png='image')
    m.reset_workspace()
    return m


def test_skin_only_shared_revision_and_stale_display():
    m=manager();context=dict(model='skin',generation=4,image_token='a'*32,revision=0)
    m.control(dict(context,parameters=dict(parameters(),points=[dict(x=20,y=20,label='foreground')])))
    state=m.state('a'*32)
    assert state['parameters']['points'][0]['x']==20 and state['revision']==1
    assert 'image_png' not in state and not state['result_ready']
    with pytest.raises(ValueError,match='同步'):m.control(dict(context,tool='roi'))
    m.controller.model='camus'
    with pytest.raises(ValueError,match='仅支持皮肤镜'):m.state()


def test_hdmi_current_result_and_cache_only_update_do_not_restart_display():
    m=manager();calls=[]
    def request(model,method,path,body=None):
        calls.append((method,path,body))
        return dict(hdmi='showing',hdmi_view='interactive')
    m.request=request
    context=dict(model='skin',generation=4)
    with pytest.raises(ValueError,match='先运行'):m.set_hdmi(dict(context,action='show'))
    result=dict(identity=dict(source_job='b'*32))
    m.job=dict(id='c'*32,state='done',result=result);m.result_revision=0
    m.set_hdmi(dict(context,action='show'))
    assert m.hdmi_requested and [c[0] for c in calls]==['GET']
    m.set_hdmi(dict(context,action='leave'))
    assert not m.hdmi_requested and len(calls)==1
    m.set_hdmi(dict(context,action='stop'))
    assert calls[-1][2]==dict(action='stop')


def test_native_touch_tools_point_and_roi_map_to_processed_image():
    m=manager();surface=Surface();surface.image=Image.fromarray(np.zeros((50,100,3),np.uint8))
    surface.update(m.state('a'*32))
    ox,oy,rw,rh=surface.geometry()
    surface.gesture('down',ox+rw//2,oy+rh//2)
    path,body=surface.gesture('up',ox+rw//2,oy+rh//2)
    assert path=='/interactive/control'
    assert body['parameters']['points']==[dict(x=50,y=25,label='foreground')]
    surface.update(m.control(body))
    surface.gesture('down',250,20)
    path,body=surface.gesture('up',250,20)
    assert body['tool']=='roi'
    surface.update(m.control(body))
    surface.gesture('down',ox+30,oy+30)
    surface.gesture('move',ox+220,oy+120)
    assert surface.drag
    path,body=surface.gesture('up',ox+220,oy+120)
    assert body['parameters']['roi'] and body['parameters']['points']==[]
    assert body['parameters']['box'] is None
    surface.update(m.control(body))
    assert surface.frame().shape==(480,640,3)
    surface.gesture('down',450,20)
    assert surface.gesture('up',450,20)[0]=='/interactive/jobs'


def test_hdmi_unchanged_state_and_updated_image_do_not_show_stale_thumbnails():
    m=manager();surface=Surface();surface.image=Image.new('RGB',(100,50),'red')
    current=m.state('a'*32);assert surface.update(current)
    assert not surface.update(dict(current))
    frame=surface.frame();ox,oy,rw,rh=surface.geometry();point=(oy+rh//2,ox+rw//2)
    assert tuple(frame[point])==(255,0,0)
    assert np.array_equal(frame,surface.frame())
    surface.image=Image.new('RGB',(100,50),'green')
    assert tuple(surface.frame()[point])==(0,128,0)
    surface.overlay=Image.new('RGB',(100,50),'blue')
    assert tuple(surface.frame()[point[0],point[1]+320])==(0,0,255)
    surface.overlay=None
    assert tuple(surface.frame()[point[0],point[1]+320])==(15,28,40)


@pytest.mark.parametrize('tool',('foreground','background','roi','box'))
def test_hdmi_mark_accumulates_same_mode_points_and_replaces_rectangles(tool):
    m=manager();m.tool=tool
    m.parameters.update(roi=dict(x0=2,y0=2,x1=90,y1=45),
        box=dict(x0=5,y0=5,x1=70,y1=35),
        points=[dict(x=20,y=20,label='foreground'),dict(x=30,y=20,label='background')])
    surface=Surface();surface.image=Image.new('RGB',(100,50),'red');surface.update(m.state('a'*32))
    ox,oy,rw,rh=surface.geometry()
    surface.gesture('down',ox+30,oy+30)
    path,body=surface.gesture('up',ox+220,oy+120)
    assert path=='/interactive/control'
    state=m.control(body);p=state['parameters']
    assert (p['roi'] is None)==(tool!='roi')
    assert (p['box'] is None)==(tool!='box')
    assert len(p['points'])==(2 if tool in ('foreground','background') else 0)
    if p['points']:assert p['points'][0]['label']==tool
    assert p['threshold']==.45 and m.image_slot['token']=='a'*32


def test_cancelled_or_empty_hdmi_rectangle_preserves_previous_marks():
    m=manager();m.tool='box';m.parameters['points']=[dict(x=20,y=20,label='foreground')]
    surface=Surface();surface.image=Image.new('RGB',(100,50),'red');surface.update(m.state('a'*32))
    ox,oy,rw,rh=surface.geometry()
    surface.gesture('down',ox+30,oy+30)
    assert surface.gesture('up',ox+30,oy+30) is None
    surface.gesture('down',ox+30,oy+30)
    assert surface.gesture('cancel',ox+50,oy+50) is None
    assert m.parameters['points']==[dict(x=20,y=20,label='foreground')]


def test_drag_input_keeps_motion_release_and_partial_packets(monkeypatch):
    reader=DragInput();reader.fd=42;reader.axes=(0,1);reader.ranges=[(0,4096),(0,4096)]
    events=[(3,0,1024),(3,1,1024),(1,330,1),(0,0,0),
        (3,0,3072),(3,1,3072),(0,0,0),(1,330,0),(0,0,0)]
    data=b''.join(reader.event.pack(0,0,*event) for event in events)
    chunks=iter([data[:11],data[11:]])
    def read(*args):
        try:return next(chunks)
        except StopIteration:raise BlockingIOError()
    monkeypatch.setattr('os.read',read)
    result=reader.poll_events()
    assert [r[0] for r in result]==['down','move','up']
    assert result[0][1:] == (159,119) and result[-1][1:] == (479,359)


def test_clear_all_shared_marks_preserves_image_and_requires_fresh_result():
    m=manager();m.parameters.update(roi=dict(x0=10,y0=10,x1=90,y1=45),
        box=dict(x0=20,y0=15,x1=60,y1=40),points=[dict(x=30,y=20,label='foreground'),dict(x=40,y=30,label='background')])
    m.job=dict(id='b'*32,state='done');m.result_revision=0
    surface=Surface();surface.image=Image.new('RGB',(100,50),'red')
    surface.update(m.state('a'*32))
    surface.gesture('down',570,460)
    path,body=surface.gesture('up',570,460)
    assert path=='/interactive/control'
    previous_image=m.image_slot
    state=m.control(body)
    assert m.image_slot is previous_image and state['revision']==1
    assert state['parameters']==parameters() and not state['result_ready']
    with pytest.raises(ValueError,match='同步'):m.control(body)
    surface.update(state);surface.state['busy']=True
    surface.gesture('down',570,460)
    assert surface.gesture('up',570,460) is None
