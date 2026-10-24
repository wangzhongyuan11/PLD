import json,time
from pathlib import Path
from acdc.appliance.hdmi import HeartView
from acdc.appliance.native import load_native
root=Path('/home/xilinx/acdc_board_v2/final_data_20261022/jobs/04d9a8445ca245138cc3c1e1698ddda4')
v=HeartView(root,load_native());v.frame()
r={}
for phase in ('ED','ES'):
 v.phase=phase
 r[phase]={'full_triangles':len(v.meshes[phase][0][1]),'preview_triangles':len(v.meshes[phase][1][1])}
 for preview in (False,True):
  samples=[]
  for i in range(4):
   v.yaw+=.06;t=time.monotonic();v.frame(preview=preview);samples.append(time.monotonic()-t)
  r[phase]['preview' if preview else 'full']=samples
for x,y in ((100,150),(450,150),(100,360),(450,360)):
 v.tap(x,y);assert not v.quad;v.frame();v.tap(30,20);assert v.quad and not v.surface
r['back_all_four']='PASS'
print(json.dumps(r),flush=True)
