"""Board-local real HTTP lifecycle checks; no replacement CPU inference."""
import argparse
import http.client
import json
from pathlib import Path
import socket
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    def request(method,path,data=None,headers=None):
        connection=http.client.HTTPConnection('127.0.0.1',8092,timeout=15)
        head=dict(headers or {})
        try:
            connection.request(method,path,data,head);response=connection.getresponse()
            return response.status,response.read(),dict(response.getheaders())
        finally:connection.close()
    def post(path,data):return request('POST',path,json.dumps(data).encode(),{'Content-Type':'application/json'})
    def health():
        status,data,_=request('GET','/api/v2/health');assert status==200;return json.loads(data)
    assert health()['ready'] and not health()['active_job']
    # A truncated TCP body must release its exclusive receiving reservation.
    connection=http.client.HTTPConnection('127.0.0.1',8092,timeout=10)
    connection.putrequest('POST','/api/v2/jobs?ed=ED.nii.gz&es=ES.nii.gz')
    for key,value in {'Content-Length':'1024','X-ED-Bytes':'512','Cookie':cookie}.items():connection.putheader(key,value)
    connection.endheaders();connection.send(b'truncated');connection.sock.shutdown(socket.SHUT_WR)
    response=connection.getresponse();assert response.status!=202;response.read();connection.close()
    deadline=time.monotonic()+10
    while health()['active_job']:
        if time.monotonic()>deadline:raise TimeoutError('upload reservation was not released')
        time.sleep(.05)
    def upload(images):
        status,data,_=request('POST','/api/v2/jobs?ed=ED.nii.gz&es=ES.nii.gz',b''.join(images),
            {'Content-Type':'application/octet-stream','X-ED-Bytes':str(len(images[0]))})
        return status,json.loads(data)
    def wait(job,targets,seconds=15):
        deadline=time.monotonic()+seconds
        while True:
            status,raw,_=request('GET','/api/v2/jobs/'+job);assert status==200;state=json.loads(raw)
            if state['status'] in targets:return state
            if state['status'] in ('failed','cancelled','assets_ready'):raise AssertionError(state)
            if time.monotonic()>deadline:raise TimeoutError(str(state))
            time.sleep(.02)
    status,result=upload([b'not a gzip nifti',b'not a gzip nifti']);assert status==202
    invalid=result['run_id'];wait(invalid,{'failed'})
    assert request('GET','/api/v2/jobs/'+invalid+'/assets/result.zip')[0]!=200
    deadline=time.monotonic()+10
    while health()['active_job']:
        if time.monotonic()>deadline:raise TimeoutError('failed input retained admission')
        time.sleep(.05)
    images=[(args.pair_root/(phase+'.nii.gz')).read_bytes() for phase in ('ED','ES')]
    status,result=upload(images);assert status==202;cancelled=result['run_id']
    during=wait(cancelled,{'infer_restore'},60)
    assert during.get('phase_completed',0)>=1
    competing_status,competing=upload([b'x',b'y'])
    assert competing_status!=202 and 'another task is active' in competing.get('error','')
    assert post('/api/v2/jobs/'+cancelled+'/cancel',{})[0]==200
    wait(cancelled,{'cancelled'},15)
    assert request('GET','/api/v2/jobs/'+cancelled+'/assets/result.zip')[0]!=200
    deadline=time.monotonic()+10
    while health()['active_job']:
        if time.monotonic()>deadline:raise TimeoutError('cancelled job retained admission')
        time.sleep(.05)
    final=health();assert final['ready'] and final['backend']=='pl' and not final['synthetic']
    report=dict(schema='pld-acdc-appliance-lifecycle-board-v2',complete=True,
        transport='real board-local HTTP/TCP; not an Ethernet unplug test',interrupted_upload_released=True,
        invalid_nifti_failed_without_result=True,concurrent_upload_rejected=True,cancelled_during_inference=True,
        cancelled_job_without_complete_result=True,admission_ready_after_cancel=True,
        invalid_job=invalid,cancelled_job=cancelled,phase_at_cancel=during,health=final,
        next_full_fresh_job_must_be_verified_separately=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
