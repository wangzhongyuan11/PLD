"""One bounded HTTP HDMI lifecycle per model, using existing board test inputs."""
import base64,hashlib,json,time,urllib.request
from pathlib import Path
OUT=Path('/home/xilinx/hdmi_lifecycle_20261024.json');rows=[]
def save():OUT.write_text(json.dumps(rows,indent=2))
def request(port,path,body=None,raw=None,headers=None):
 h={'Origin':f'http://127.0.0.1:{port}'};h.update(headers or {})
 if body is not None:raw=json.dumps(body).encode();h['Content-Type']='application/json'
 with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=raw,headers=h),timeout=35) as r:return json.load(r)
def wait(fn,limit=180):
 end=time.monotonic()+limit
 while time.monotonic()<end:
  s=fn()
  if s:return s
  time.sleep(.4)
 raise TimeoutError('bounded wait expired')
def switch(model):
 request(8088,'/entry/switch',{'model':model})
 def ready():
  s=request(8088,'/entry/status')
  if s['phase']=='error':raise RuntimeError(s)
  return s if s['phase']=='ready' and s['model']==model else None
 return wait(ready)['url']
def job(port,path,body):
 s=request(port,path+'api/jobs',body);ident=s['id'];t=time.monotonic()
 def done():
  r=request(port,path+'api/jobs/'+ident)
  if r.get('state')=='error':raise RuntimeError(r)
  return r if r.get('state')=='done' else None
 r=wait(done);return ident,time.monotonic()-t
try:
 h=request(8092,'/api/v2/health');assert h['ready'] and h['hdmi']=='off',h
 rows.append({'model':'acdc','hdmi_showing_observed':True,'restore_health':h});save()
 folder=Path('/home/xilinx/acdc_board_v2/final_data_20261022/inputs/04d9a8445ca245138cc3c1e1698ddda4')
 ed=(folder/'ED.nii.gz').read_bytes();es=(folder/'ES.nii.gz').read_bytes()
 t=time.monotonic();r=request(8092,'/api/v2/jobs?ed=ED.nii.gz&es=ES.nii.gz',raw=ed+es,headers={'X-ED-Bytes':str(len(ed)),'Content-Type':'application/octet-stream'})
 ident=r['run_id']
 def acdc_done():
  s=request(8092,'/api/v2/jobs/'+ident)
  if s['status'] in ('failed','cancelled','interrupted'):raise RuntimeError(s)
  return s if s['status']=='assets_ready' else None
 wait(acdc_done,240);rows[-1].update(resumed_job=ident,resumed_seconds=time.monotonic()-t,passed=True);save();print('ACDC_PASS',flush=True)
 for model in ('skin','camus'):
  path=switch(model);row={'model':model,'path':path};rows.append(row);save()
  if model=='skin':
   folder=Path('/home/xilinx/skin8_web_data/skin/37dce47fc49d4d58bca8f5ddaae36073')
   body={'mode':'infer','image':base64.b64encode((folder/'input.jpg').read_bytes()).decode(),'thickness':2}
  else:
   folder=Path('/home/xilinx/skin8_web_data/camus/815b04489e3f465db2543d68ca2db20f/input')
   files=sorted(p for p in folder.iterdir() if p.is_file());selected=next(p.name for p in files if p.suffix.lower()!='.raw')
   body={'files':[{'name':p.name,'data_base64':base64.b64encode(p.read_bytes()).decode()} for p in files],'input_name':selected}
  ident,seconds=job(8088,path,body);row.update(before_job=ident,before_seconds=seconds);save()
  state=request(8088,path+'api/status');assert state.get('hdmi_available',True),state
  request(8088,path+'api/hdmi',{'action':'show','id':ident,'view':'quad' if model=='camus' else 'outlined'})
  def showing():
   s=request(8088,path+'api/status');v=s.get('hdmi')
   if s.get('fault'):raise RuntimeError(s)
   if isinstance(v,dict):v=v.get('state')
   return s if v=='showing' else None
  row['show_status']=wait(showing,70);save()
  row['stop_status']=request(8088,path+'api/hdmi',{'action':'stop'});save()
  after,seconds=job(8088,path,body);row.update(after_job=after,after_seconds=seconds)
  root=Path('/home/xilinx/skin8_web_data')/model
  relative='inference/frame_000001.bin' if model=='skin' else 'output/head.bin'
  hashes=[hashlib.sha256((root/j/relative).read_bytes()).hexdigest() for j in (ident,after)]
  row.update(head_sha256=hashes,head_equal=hashes[0]==hashes[1]);assert row['head_equal'];row['passed']=True;save();print(model.upper()+'_PASS',flush=True)
 switch('acdc');rows.append({'restored_entry':request(8088,'/entry/status')});save()
except Exception as exc:
 rows.append({'error':str(exc)});save();raise
