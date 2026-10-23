"""Measure uncached board MPR HTTP responses, with explicit idle/busy buckets.

This client performs no medical processing and requires no login.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='192.168.2.99');p.add_argument('--port',type=int,default=8092)
    p.add_argument('--job',required=True);p.add_argument('--samples',type=int,default=30)
    p.add_argument('--wait-busy',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not re.fullmatch('[0-9a-f]{32}',a.job) or not 1<=a.samples<=120:p.error('bounded samples and valid job ID required')
    if a.output.exists():raise FileExistsError(a.output)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base=f'http://{a.host}:{a.port}'
    def request(path,value=None):
        body=None if value is None else json.dumps(value).encode()
        req=urllib.request.Request(base+path,body,headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=15) as r:return r.read(),r.headers
    def health():return json.loads(request('/api/v2/health')[0])
    volumes={phase:json.loads(request(f'/api/v2/jobs/{a.job}/assets/viewer/{phase}/volume.json')[0])
             for phase in ('ED','ES')}
    if a.wait_busy:
        deadline=time.monotonic()+60
        while not health().get('active_job'):
            if time.monotonic()>deadline:raise TimeoutError('no active real job')
            time.sleep(.1)
    rows=[]
    for i in range(a.samples):
        phase=('ED','ES')[i%2];plane=('axial','coronal','sagittal')[(i//2)%3]
        axis={'axial':2,'coronal':1,'sagittal':0}[plane]
        v=volumes[phase];bounds=np.asarray(v['bounds']);position=list(v['position'])
        position[axis]=float(bounds[0,axis]+(.1+.8*(i+.5)/a.samples)*(bounds[1,axis]-bounds[0,axis]))
        before=health();began=time.perf_counter()
        png,headers=request(f'/api/v2/jobs/{a.job}/mpr/{phase}/{plane}',dict(position=position,
            shape=[400,400],level=v['window'][0],window=v['window'][1],alpha=.5,visible=14,outline=False))
        seconds=time.perf_counter()-began;after=health()
        meta=json.loads(headers['X-MPR-Metadata'])
        if not png.startswith(b'\x89PNG\r\n\x1a\n') or meta['generator']!='pynq_ps':raise AssertionError('invalid board MPR response')
        if meta['plane']!=plane or not np.allclose(meta['position'],position,rtol=0,atol=1e-10):raise AssertionError('MPR cursor response differs')
        busy=bool(before.get('active_job'));stable=before.get('active_job')==after.get('active_job')
        rows.append(dict(phase=phase,plane=plane,position=position,seconds=seconds,bytes=len(png),
            png_sha256=hashlib.sha256(png).hexdigest(),load='busy' if busy and stable else 'idle' if stable else 'transition'))
    groups={}
    for load in ('idle','busy','transition'):
        values=[r['seconds'] for r in rows if r['load']==load]
        if values:groups[load]=dict(samples=len(values),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),maximum=max(values))
    report=dict(schema='pld-acdc-appliance-mpr-http-v2',application_sha256=before['initialization']['application_sha256'],
        job_id=a.job,complete=True,groups_seconds=groups,rows=rows,transport='direct LAN HTTP, proxies disabled',
        scope='uncached 400x400 MPR HTTP responses after loading both volume manifests; volume initialization is charged to result asset loading, no browser/GPU timing claim')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(groups))


if __name__=='__main__':main()
