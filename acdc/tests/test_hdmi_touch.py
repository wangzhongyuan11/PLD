import json
from types import SimpleNamespace

import numpy as np
import pytest

from acdc.appliance.hdmi import HeartView
from acdc.appliance.engine import JobManager
from acdc.contract import ContractError
from display_web.touch_display import QuadTouch, TouchInput


def test_real_evdev_packet_layout_and_release(monkeypatch):
    touch=TouchInput();touch.fd=42;touch.axes=(0,1);touch.ranges=[(100,1100),(200,2200)]
    data=b''.join(touch.event.pack(0,0,*event) for event in
                  [(3,0,600),(3,1,1200),(1,330,1),(0,0,0),(1,330,0),(0,0,0)])
    chunks=iter([data])
    def read(*_):
        try:return next(chunks)
        except StopIteration:raise BlockingIOError()
    monkeypatch.setattr('os.read',read)
    assert touch.poll()==[(320,240)]


def test_quad_tap_enlarges_and_returns():
    pixels=np.zeros((480,640,3),np.uint8);pixels[30:236,6:317]=[255,0,0]
    view=QuadTouch(pixels);original=view.frame()
    enlarged=view.tap(100,100)
    assert enlarged[200,320].tolist()==[255,0,0]
    assert np.array_equal(view.tap(100,100),original)


def test_heart_controls_phase_plane_slice_and_cursor(tmp_path,monkeypatch):
    monkeypatch.setattr('acdc.appliance.hdmi.load_mesh',lambda *_:None)
    monkeypatch.setattr('acdc.appliance.hdmi.render',lambda *a,size=(640,432),**kw:np.zeros((size[1],size[0],3),np.uint8))
    class Kernel:
        def prepare_mpr(self,*_):return self
        def render(self,origin,dx,dy,shape,**kw):return np.full((*shape,4),100,np.uint8)
    for phase in ('ED','ES'):
        folder=tmp_path/'viewer'/phase;folder.mkdir(parents=True)
        np.save(folder/'gray.npy',np.ones((10,12,14),np.float32))
        np.save(folder/'labels.npy',np.ones((10,12,14),np.uint8))
        (folder/'geometry.json').write_text(json.dumps(dict(shape=[10,12,14],affine=np.eye(4).tolist())))
        (folder/'volume.json').write_text(json.dumps(dict(position=[4,5,6],window=[100,200])))
    (tmp_path/'statistics.json').write_text(json.dumps({'phases':{p:{'classes':{c:{'mL':2.5} for c in ('LV','RV','MYO')}} for p in ('ED','ES')}}))
    view=HeartView(tmp_path,Kernel());assert view.frame().shape==(480,640,3)
    view.tap(150,20);assert view.phase=='ES'
    view.tap(30,80);assert view.metrics_open
    view.tap(30,150);assert not view.metrics_open and view.quad
    view.tap(30,150);assert not view.quad and view.plane==0
    view.tap(580,20);assert view.positions['ES'][2]==8
    view.tap(320,250);assert view.positions['ES'][0]!=4
    view.tap(250,20);assert view.quad


def test_hdmi_release_restores_both_pipeline_references(tmp_path):
    old=SimpleNamespace(kind='pl',synthetic=False,model_sha='x',runtime_sha='y',device=SimpleNamespace(closed=True))
    new=SimpleNamespace(device=SimpleNamespace(closed=False))
    pipeline=SimpleNamespace(backend=old,numerics=SimpleNamespace(backend=old))
    manager=JobManager(pipeline,tmp_path,backend_factory=lambda:new)
    manager.released=True;manager.hdmi_state='showing'
    assert manager.health()['ready']
    manager.hdmi_state='starting'
    with pytest.raises(ContractError):manager.reserve('a'*32)
    manager.restore_inference()
    assert pipeline.backend is pipeline.numerics.backend is new
    assert manager.hdmi_state=='off' and not manager.released


def test_pause_releases_hardware_and_resume_rebinds_pipeline(tmp_path):
    closed=[]
    old=SimpleNamespace(kind='pl',synthetic=False,model_sha='x',runtime_sha='y',
        device=SimpleNamespace(closed=False),close=lambda:closed.append('old'))
    new=SimpleNamespace(device=SimpleNamespace(closed=False))
    pipeline=SimpleNamespace(backend=old,numerics=SimpleNamespace(backend=old))
    manager=JobManager(pipeline,tmp_path,backend_factory=lambda:new)
    manager.lifecycle('pause');manager.hdmi_thread.join(2)
    assert closed==['old'] and manager.suspended and manager.released
    assert not manager.health()['ready']
    with pytest.raises(ContractError):manager.reserve('a'*32)
    with pytest.raises(ContractError):manager.show_hdmi(None,'stop')
    manager.lifecycle('resume');manager.hdmi_thread.join(2)
    assert not manager.suspended and pipeline.backend is pipeline.numerics.backend is new
