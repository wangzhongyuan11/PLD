"""Exercise browser HTTP upload/result/MPR against a synthetic numerical owner."""
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import nibabel as nib
import numpy as np
import pytest

from acdc.backend import DemoBackend
from acdc.contract import ContractError
from acdc.appliance.engine import JobManager
from acdc.appliance.full_pipeline import FullPipeline
from acdc.appliance.native import load_native
from acdc.appliance.pipeline import NumericalPipeline
from acdc.appliance.reconstruct import make_exp_table
from acdc.appliance.web import WebApplication,handler


class MeshFixture:
    info={'test_double':True}
    def generate(self,labels,affine,directory,*,cancel):
        directory.mkdir(parents=True);(directory/'fixture.json').write_text('{"test_double":true}')
        return []


class DirectClient:
    def __init__(self,manager):self.manager=manager
    def request(self,**message):return self.manager.dispatch(message)


def setup(tmp_path):
    backend=DemoBackend();backend.runtime_sha=None;kernel=load_native()
    pipeline=FullPipeline(NumericalPipeline(backend,kernel,make_exp_table(backend.head_scale),binding={}),MeshFixture())
    manager=JobManager(pipeline,tmp_path)
    return manager,WebApplication(DirectClient(manager),tmp_path,kernel,'test-only-access-code')


def test_reservation_is_exclusive_and_releases_on_abort(tmp_path):
    manager,_=setup(tmp_path);first='a'*32;second='b'*32
    r=manager.reserve(first)
    with pytest.raises(ContractError):manager.reserve(second)
    manager.release(first,'wrong-token')
    with pytest.raises(ContractError):manager.reserve(second)
    manager.release(first,r['reservation'])
    assert manager.reserve(second)['run_id']==second
    with pytest.raises(ContractError):manager.reserve('../escape')
    manager.close()


def test_http_upload_mpr_export_and_client_timing(tmp_path):
    manager,app=setup(tmp_path)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(app))
    thread=threading.Thread(target=server.serve_forever);thread.start()
    cookie=None
    def call(method,path,body=None,headers=None):
        connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=15)
        head=dict(headers or {})
        if cookie:head['Cookie']=cookie
        connection.request(method,path,body=body,headers=head)
        response=connection.getresponse();raw=response.read();status=response.status;result_headers=dict(response.getheaders())
        connection.close();return status,raw,result_headers
    try:
        assert call('GET','/api/v2/health')[0]==401
        status,_,headers=call('POST','/api/v2/session',json.dumps({'code':'test-only-access-code'}))
        assert status==200 and 'HttpOnly' in headers['Set-Cookie']
        cookie=headers['Set-Cookie'].split(';')[0]
        assert call('GET','/api/v2/health')[0]==200
        rng=np.random.default_rng(500)
        image=nib.Nifti1Image(rng.integers(1,250,(9,11,5)).astype(np.float32),np.diag([2.,2.,2.,1.]))
        image.header.set_xyzt_units('mm');raw=image.to_bytes()
        status,reply,_=call('POST','/api/v2/jobs?ed=ed.nii&es=es.nii',raw+raw,{'X-ED-Bytes':str(len(raw))})
        assert status==202;job=json.loads(reply)['run_id']
        for _ in range(100):
            state=json.loads(call('GET','/api/v2/jobs/'+job)[1])
            if state['status'] in ('assets_ready','failed'):break
            time.sleep(.03)
        assert state['status']=='assets_ready',state
        base='/api/v2/jobs/'+job
        assert call('GET',base+'/assets/result.zip')[0]==200
        assert call('GET',base+'/assets/viewer/ED/gray.npy')[0]==404
        volume=json.loads(call('GET',base+'/assets/viewer/ED/volume.json')[1])
        assert (job,'ED') in app.volumes
        prepared=app.volumes[(job,'ED')]
        status,png,headers=call('POST',base+'/mpr/ED/axial',json.dumps({'position':volume['position'],'shape':[80,100]}))
        assert status==200 and png.startswith(b'\x89PNG')
        assert app.volumes[(job,'ED')] is prepared
        assert json.loads(headers['X-MPR-Metadata'])['generator']=='pynq_ps'
        assert call('POST',base+'/client-timing',json.dumps({'client_id':'f'*32,'interactive_seconds':1.25}))[0]==200
        assert (tmp_path/'jobs'/job/'client_timings'/('f'*32+'.json')).is_file()
    finally:server.shutdown();server.server_close();thread.join(timeout=5);manager.close()
