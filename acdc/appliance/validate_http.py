"""Fresh board HTTP jobs + asset transfer/oracle audit. Does not claim GPU readiness.

The SSH credential is used only to retrieve the board's access code, never
printed or written to a report. Medical preprocessing/inference remains remote.
"""
from __future__ import annotations
import argparse
import getpass
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--host',default='192.168.2.99');parser.add_argument('--port',type=int,default=8092)
    parser.add_argument('--repeat',type=int,default=1);parser.add_argument('--limit',type=int)
    parser.add_argument('--patient',action='append',help='select named development patients for a bounded candidate test')
    parser.add_argument('--access-file',default='/home/xilinx/acdc_board_v2/data/access_code.txt')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import paramiko
    password=os.environ.get('PLD_BOARD_PASS') or getpass.getpass('Board password: ')
    with paramiko.SSHClient() as client:
        client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(args.host,username='xilinx',password=password,timeout=10,allow_agent=False,look_for_keys=False)
        with client.open_sftp() as sftp:
            with sftp.open(args.access_file) as stream:code=stream.read().decode().strip()
    del password
    # This is a direct LAN benchmark. Windows/system proxy settings can route
    # 192.168.x.x through a local proxy and inject unrelated 502/connect faults.
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    base=f'http://{args.host}:{args.port}'
    def request(path,body=None,headers=None):
        r=urllib.request.Request(base+path,data=body,headers=headers or {})
        with opener.open(r,timeout=120) as response:return response.read()
    def post(path,body):return json.loads(request(path,json.dumps(body).encode(),{'Content-Type':'application/json'}))
    post('/api/v2/session',{'code':code});del code
    health=json.loads(request('/api/v2/health'))
    if not health.get('ready') or health.get('backend')!='pl' or health.get('synthetic'):
        raise RuntimeError('board engine is not ready for real PL jobs: '+str(health))
    cases=json.loads((args.results/'cases.json').read_text())['cases'];patients={}
    for case in cases:patients.setdefault(case['patient'],{})[case['phase']]=case
    selected=[(name,pair) for name,pair in patients.items() if not args.patient or name in args.patient][:args.limit];rows=[]
    if not selected:raise ValueError('no matching development patients')
    for repeat in range(args.repeat):
        for patient,pair in selected:
            images=[]
            for phase in ('ED','ES'):
                case=pair[phase]
                path=Path(case['input']) if 'input' in case else Path(case['gt']).with_name(Path(case['gt']).name.replace('_gt.nii','.nii'))
                images.append(path.read_bytes())
            began=time.perf_counter()
            created=json.loads(request('/api/v2/jobs?ed=ED.nii.gz&es=ES.nii.gz',b''.join(images),
                {'Content-Type':'application/octet-stream','X-ED-Bytes':str(len(images[0]))}))
            job=created['run_id'];folder=args.output/job;folder.mkdir()
            while True:
                state=json.loads(request('/api/v2/jobs/'+job))
                if state['status']=='assets_ready':break
                if state['status'] in ('failed','cancelled','interrupted'):raise RuntimeError(str(state))
                if time.perf_counter()-began>180:raise TimeoutError('fresh HTTP job deadline')
                time.sleep(.25)
            server_ready=time.perf_counter()-began
            assetbase='/api/v2/jobs/'+job+'/assets/'
            def asset(name):
                data=request(assetbase+name);target=folder/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                return data
            for phase in ('ED','ES'):
                volume=json.loads(asset('viewer/'+phase+'/volume.json'))
                for plane in ('axial','coronal','sagittal'):asset('mpr/'+phase+'/'+plane+'.png')
                for structure in volume['meshes']:
                    for key in ('points','normals','indices'):asset('meshes/'+phase+'/'+structure['smooth'][key])
            statistics=json.loads(asset('statistics.json'))
            transfer_seconds=time.perf_counter()-began
            numerical=json.loads(asset('numerics.json'));completion=json.loads(asset('completion.json'))
            app_sha=health['initialization']['application_sha256']
            if completion.get('application_sha256')!=app_sha:raise AssertionError('application identity changed or missing')
            expected_stats=json.loads((Path(pair['ED']['result_dir'])/'statistics.json').read_text())
            if statistics!=expected_stats:raise AssertionError('native statistics differ: '+patient)
            for row in numerical['records']:
                old=json.loads((Path(pair[row['phase']]['result_dir'])/'checkpoint'/row['phase']/f"{row['center_index']:04}.json").read_text())
                if row['synthetic'] or row['backend']!='pl':raise AssertionError('not a fresh physical PL record')
                for key in ('input_sha256','head_sha256'):
                    if row[key]!=old[key]:raise AssertionError(patient+'/'+row['phase']+'/'+key)
            raw=asset('result.zip')
            if hashlib.sha256(raw).hexdigest()!=completion['zip_sha256']:raise AssertionError('ZIP transport SHA mismatch')
            with zipfile.ZipFile(folder/'result.zip') as archive:
                if archive.testzip() is not None:raise AssertionError('ZIP CRC mismatch')
                manifest=json.loads(archive.read('manifest.json'))
                for name,sha in manifest['files'].items():
                    if hashlib.sha256(archive.read(name)).hexdigest()!=sha:raise AssertionError('result member drift: '+name)
                for phase in ('ED','ES'):(folder/(phase+'.nii.gz')).write_bytes(archive.read('labels/'+phase+'.nii.gz'))
            summary=dict(patient=patient,repeat=repeat,job_id=job,server_ready_observed_seconds=server_ready,
                initial_assets_downloaded_seconds=transfer_seconds,server=completion,physical_pl_heads=len(numerical['records']),
                all_input_and_head_hashes_equal=True,statistics_equal=True,zip_sha_crc_members_equal=True,
                browser_gpu_readiness_verified=False)
            rows.append(summary)
            report=dict(schema='pld-acdc-appliance-http-validation-v2',complete=False,rows=rows,transport='direct LAN HTTP; system proxies disabled in this client only',
                        scope='real LAN upload, fresh board computation and initial asset download; no browser GPU claim')
            (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
            print(patient,repeat,job,'server',round(server_ready,3),'assets',round(transfer_seconds,3),flush=True)
    report['complete']=True;(args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
