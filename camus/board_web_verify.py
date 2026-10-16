"""Host HTTP gate against a real PL service, including ZIP and HDMI handoff."""
from __future__ import annotations
import argparse
import base64
import io
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import zipfile
import numpy as np
from PIL import Image
from .model import MODEL_SPEC_SHA256
from .preprocess import preprocess
from .result import CLASS_COLORS


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8091')
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=20);p.add_argument('--hdmi',action='store_true')
    a=p.parse_args(argv)
    if not 1<=a.frames<=200 or a.output.exists():p.error('frames must be 1..200 and output must be new')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def fetch(path,payload=None):
        request=Request(a.url.rstrip('/')+path,data=None if payload is None else json.dumps(payload).encode(),
                        headers={} if payload is None else {'Content-Type':'application/json'})
        with urlopen(request,timeout=100) as response:return response.read()
    def api(path,payload=None):return json.loads(fetch(path,payload))
    status=api('/api/status')
    if status['backend']!='pl' or not status['ready'] or status['spec_sha256']!=MODEL_SPEC_SHA256:
        raise RuntimeError('HTTP target is not a ready real CAMUS PL service')
    report=dict(schema='camus-board-http-validation-v1',url=a.url,scope='host HTTP through SSH loopback tunnel to real PL',
                complete=False,status=status,frames=[],hdmi=[],busy_409=False)
    cases=json.loads((a.dataset/'dataset.json').read_text())['cases'];indices=np.linspace(0,len(cases)-1,a.frames,dtype=int)
    def save():a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def infer(case,check_busy=False,reinitialized=False):
        content=(a.dataset/case['input']).read_bytes()
        payload=dict(input_name=case['input'],files=[dict(name=case['input'],data_base64=base64.b64encode(content).decode())])
        start=time.perf_counter();job=api('/api/jobs',payload)
        if check_busy:
            try:api('/api/jobs',payload)
            except HTTPError as exc:
                if exc.code!=409:raise
                report['busy_409']=True
            else:raise RuntimeError('concurrent request was not rejected')
        deadline=time.monotonic()+100
        while True:
            job=api('/api/jobs/'+job['id'])
            if job['state']=='error':raise RuntimeError(str(job))
            if job['state']=='done':break
            if time.monotonic()>deadline:raise TimeoutError('HTTP job timeout')
            time.sleep(.1)
        ready=time.perf_counter()-start;root='/files/'+job['id']+'/'
        packed=fetch(root+'result.zip');download=time.perf_counter()-start
        with zipfile.ZipFile(io.BytesIO(packed)) as archive,np.load(a.dataset/case['expected'],allow_pickle=False) as expected:
            if len(set(archive.namelist()))!=len(archive.namelist()):raise RuntimeError('ZIP duplicate entries')
            if archive.testzip() is not None:raise RuntimeError('ZIP CRC failure')
            if archive.read('head.bin')!=expected['logits_int8'].tobytes():raise RuntimeError('HTTP PL head differs from A')
            if archive.read('report.json')!=fetch(root+'report.json'):raise RuntimeError('ZIP metadata differs from standalone report')
            result=json.loads(archive.read('report.json'))
            if result['backend']!='pl':raise RuntimeError('software fallback reported')
            with np.load(io.BytesIO(archive.read('labels_native.npz')),allow_pickle=False) as labels:
                if not np.array_equal(labels['labels_raw'],expected['labels']) or not np.array_equal(labels['labels_processed'],expected['processed']):
                    raise RuntimeError('native labels differ from A')
            for name in ('overlay.png','segmentation.png','label.png','label_raw.png'):
                with Image.open(io.BytesIO(archive.read(name))) as image:
                    if image.size!=tuple(reversed(case['native_hw'])):raise RuntimeError('native PNG size changed')
                    actual=np.array(image)
                processed=expected['processed'];raw=expected['labels']
                if name=='label_raw.png':pixels=np.asarray([0,85,170,255],np.uint8)[raw]
                elif name=='label.png':pixels=np.asarray([0,85,170,255],np.uint8)[processed]
                elif name=='segmentation.png':pixels=CLASS_COLORS[processed]
                else:
                    gray=np.clip(np.asarray(preprocess(a.dataset/case['input']).raw_gray,dtype=np.float32),0,255).astype(np.uint8)
                    rgb=np.repeat(gray[:,:,None],3,axis=2)
                    pixels=np.rint(.55*rgb.astype(np.float32)+.45*CLASS_COLORS[processed].astype(np.float32)).astype(np.uint8)
                if not np.array_equal(actual,pixels):raise RuntimeError('PNG decoded pixels changed: '+name)
            if sum(r['pixels'] for r in result['output']['areas'].values())!=int(np.prod(case['native_hw'])):
                raise RuntimeError('area total mismatch')
        row=dict(case=case['stem'],job_id=job['id'],head_equal=True,native_equal=True,postprocess_equal=True,
                 pngs=4,native_hw=case['native_hw'],zip_bytes=len(packed),post_to_zip_ready_seconds=ready,
                 post_to_zip_downloaded_seconds=download,server_seconds=job['server_seconds'],
                 stages_seconds=job['result']['stages_seconds'],after_hdmi_reinitialization=reinitialized)
        row.update(png_pixels_equal=True,timing_details=result.get('timing_details'))
        report['frames'].append(row);save();print(json.dumps(row),flush=True)
        return job['id']
    try:
        last=None
        for index in indices:last=infer(cases[int(index)],check_busy=last is None)
        if a.hdmi:
            for view in ('overlay','segmentation','label'):
                t=time.perf_counter();state=api('/api/hdmi',dict(action='show',id=last,view=view));deadline=time.monotonic()+50
                while state['hdmi']=='starting':
                    if time.monotonic()>deadline:raise TimeoutError('HDMI initialization timeout')
                    time.sleep(.3);state=api('/api/status')
                if state['hdmi']!='showing':raise RuntimeError('HDMI frame was not submitted: '+str(state))
                report['hdmi'].append(dict(view=view,frame_submitted=True,start_seconds=time.perf_counter()-t,
                                           physical_screen_observed=False));save()
            state=api('/api/hdmi',dict(action='stop'))
            if state['hdmi']!='off':raise RuntimeError('HDMI did not stop')
            infer(cases[0],reinitialized=True)
        values=[r['post_to_zip_ready_seconds'] for r in report['frames'] if not r['after_hdmi_reinitialization']]
        report['performance']=dict(frames=len(values),median_seconds=float(np.median(values)),p95_seconds=float(np.percentile(values,95)),
                                   max_seconds=max(values),within_5_seconds=sum(v<=5 for v in values),
                                   includes='POST upload through ready-to-download ZIP; excludes initial service start and explicit HDMI reinitialization')
        report['complete']=True
    except BaseException as exc:report['error']=repr(exc);raise
    finally:save()
    print(json.dumps(dict(complete=True,performance=report['performance'],hdmi=report['hdmi'])),flush=True)


if __name__=='__main__':main()
