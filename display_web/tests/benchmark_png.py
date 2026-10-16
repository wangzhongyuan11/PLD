"""Local/board lossless PNG benchmark; deterministic synthetic data, no FPGA."""
from pathlib import Path
import time,statistics,json,tempfile,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import server
from contour import process

def benchmark(destination):
 rng=np.random.default_rng(51);h,w=768,1024
 y,x=np.mgrid[:h,:w]
 base=np.stack([(x/w)*160+30,(y/h)*170+20,((x+y)/(w+h))*100+50],axis=2)
 rgb=np.clip(base+rng.normal(0,5,(h,w,3)),0,255).astype(np.uint8)
 mask=((x-w/2)**2/(w*.25)**2+(y-h/2)**2/(h*.3)**2)<1
 views=process(mask,rgb);arrays={'original':rgb,**{k:views[k] for k in ('mask','contour','outlined','cutout')}}
 rows=[]
 for kind in ['pillow','fast_sub']:
  server.PNG_BACKEND=kind;elapsed=[]
  for trial in range(5):
   start=time.perf_counter()
   for key,a in arrays.items():server.save_lossless_png(a,destination/(key+'.png'))
   elapsed.append(time.perf_counter()-start)
  size=sum((destination/(k+'.png')).stat().st_size for k in arrays)
  for k,a in arrays.items():
   with Image.open(destination/(k+'.png')) as im:np.testing.assert_array_equal(np.array(im),a.astype(np.uint8)*255 if a.dtype==bool else a)
  rows.append({'encoder':kind,'median_seconds':statistics.median(elapsed),'total_bytes':size})
 return rows
if __name__=='__main__':
 with tempfile.TemporaryDirectory() as d:print(json.dumps(benchmark(Path(d)),indent=2))
