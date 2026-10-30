"""Skin ROI HDMI surface; controls use the same hub workspace as the browser."""
import base64
import http.client
import io
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from interactive_workspace import TOOLS


class Client:
    def __init__(self, hub):
        config=json.loads((hub/'config.json').read_text())
        token=Path(config['token_file']).read_text().strip()
        self.cookie=''
        self.request('/hub/login',dict(token=token))

    def request(self, path, body=None):
        conn=http.client.HTTPConnection('127.0.0.1',8765,timeout=3)
        try:
            conn.request('GET' if body is None else 'POST',path,
                None if body is None else json.dumps(body).encode(),
                {'Content-Type':'application/json','Origin':'http://127.0.0.1:8765','Cookie':self.cookie})
            response=conn.getresponse();raw=response.read()
            cookie=response.getheader('Set-Cookie')
            if cookie:self.cookie=cookie.split(';',1)[0]
            result=json.loads(raw)
            if response.status not in (200,202):raise RuntimeError(result.get('error','ROI request failed'))
            return result
        finally:conn.close()


def decode(value):
    with Image.open(io.BytesIO(base64.b64decode(value.split(',')[-1]))) as image:
        return image.convert('RGB')


class Surface:
    def __init__(self):
        self.state=None;self.image=None;self.overlay=None;self.result_id=None
        self.anchor=None;self.drag=None;self.message=''

    def update(self,state,result=None):
        changed=self.state!=state or result is not None
        self.state=state
        if state.get('image_png'):
            self.image=decode(state['image_png'])
        if not state['result_ready']:
            self.overlay=None;self.result_id=None
        elif result is not None:
            self.overlay=decode(result['outputs']['overlay_png'])
            self.result_id=state['job_id']
        if state['busy']:
            self.anchor=None;self.drag=None
        return changed

    def geometry(self):
        if self.image is None:return None
        w,h=self.image.size;scale=min(304/w,354/h)
        rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
        return 8+(304-rw)//2,78+(354-rh)//2,rw,rh

    def point(self,x,y,clip=False):
        geometry=self.geometry()
        if geometry is None:return None
        ox,oy,rw,rh=geometry;w,h=self.image.size
        if not clip and not (ox<=x<ox+rw and oy<=y<oy+rh):return None
        return dict(x=max(0,min(w-1,math.floor((x-ox)*w/rw))),
                    y=max(0,min(h-1,math.floor((y-oy)*h/rh))))

    def gesture(self,action,x,y):
        """Return one shared-workspace command on release; motion only previews."""
        if not self.state or self.state['busy']:return None
        if action=='cancel':self.anchor=None;self.drag=None;return None
        if action=='down':
            self.anchor=(x,y);self.drag=None
            return None
        if self.anchor is None:return None
        ax,ay=self.anchor
        start=self.point(ax,ay)
        if action=='move':
            if start and self.state['tool'] in ('roi','box'):
                self.drag=(start,self.point(x,y,True))
            return None
        if action!='up':return None
        self.anchor=None;self.drag=None
        context={k:self.state[k] for k in ('model','generation','image_token','revision')}
        if 520<=ax<640 and 438<=ay<480 and 520<=x<640 and 438<=y<480:
            return '/interactive/control',dict(context,parameters=dict(self.state['parameters'],roi=None,box=None,points=[]))
        if ay<48 and y<48:
            a=next((i for i,(lo,hi) in enumerate(((0,100),(100,200),(200,300),(300,400),(400,520),(520,640))) if lo<=ax<hi),None)
            if a is None:return None
            if a<4:return '/interactive/control',dict(context,tool=TOOLS[a],**({} if TOOLS[a]==self.state['tool'] else dict(parameters=dict(self.state['parameters'],roi=None,box=None,points=[]))))
            if a==4:return '/interactive/jobs',context
            return '/interactive/hdmi',dict(context,action='leave')
        if start is None:return None
        end=self.point(x,y,True);params=dict(self.state['parameters'],roi=None,box=None,points=[dict(p) for p in self.state['parameters']['points'] if p['label']==self.state['tool']] if self.state['tool'] in ('foreground','background') else [])
        tool=self.state['tool']
        if tool in ('foreground','background'):
            if self.point(x,y) is None:return None
            params['points'].append(dict(end,label=tool))
        else:
            rect=dict(x0=min(start['x'],end['x']),y0=min(start['y'],end['y']),
                x1=max(start['x'],end['x'])+1,y1=max(start['y'],end['y'])+1)
            if rect['x1']-rect['x0']<2 or rect['y1']-rect['y0']<2:return None
            params[tool]=rect
        return '/interactive/control',dict(context,parameters=params)

    def frame(self):
        image=Image.new('RGB',(640,480),(15,28,40));draw=ImageDraw.Draw(image)
        tool=self.state['tool'] if self.state else ''
        for i,(label,x,w) in enumerate((('FG point',0,100),('BG point',100,100),('ROI',200,100),
                ('Box',300,100),('Run',400,120),('Web / Stop',520,120))):
            fill=(0,115,135) if i<4 and TOOLS[i]==tool else (35,65,80)
            draw.rectangle((x+2,2,x+w-3,46),fill=fill)
            draw.text((x+9,18),label,fill='white')
        draw.text((8,57),'SKIN ROI / draw on original',fill='white')
        draw.text((328,57),'Corrected segmentation',fill='white')
        geometry=self.geometry()
        if geometry:
            ox,oy,rw,rh=geometry;w,h=self.image.size
            key=(id(self.image),id(self.overlay),rw,rh)
            if getattr(self,'thumbnail_key',None)!=key:
                self.thumbnails=(self.image.resize((rw,rh)),self.overlay.resize((rw,rh)) if self.overlay is not None else None)
                self.thumbnail_key=key
            image.paste(self.thumbnails[0],(ox,oy))
            if self.thumbnails[1] is not None:image.paste(self.thumbnails[1],(ox+320,oy))
            def rectangle(rect,color):
                draw.rectangle((ox+rect['x0']*rw/w,oy+rect['y0']*rh/h,
                    ox+rect['x1']*rw/w,oy+rect['y1']*rh/h),outline=color,width=2)
            params=self.state['parameters']
            for key,color in (('roi','#00d6ff'),('box','#ffd43b')):
                if params[key]:rectangle(params[key],color)
            for point in params['points']:
                x=ox+point['x']*rw/w;y=oy+point['y']*rh/h
                draw.ellipse((x-3,y-3,x+3,y+3),fill='#00ff70' if point['label']=='foreground' else '#ff435c',outline='white')
            if self.drag:
                a,b=self.drag;rectangle(dict(x0=min(a['x'],b['x']),y0=min(a['y'],b['y']),
                    x1=max(a['x'],b['x'])+1,y1=max(a['y'],b['y'])+1),'white')
        status='Working: FPGA inference may pause HDMI' if self.state and self.state['busy'] else 'Select tool, touch / drag, then Run'
        draw.text((8,446),status,fill=(150,225,215))
        # Native default fonts need ASCII; the web page carries full error text.
        if self.message:draw.text((8,463),self.message.encode('ascii','replace').decode()[:70],fill=(255,185,100))
        draw.rectangle((522,440,637,477),fill=(35,65,80))
        draw.text((536,453),'Clear all',fill='white')
        return np.asarray(image).copy()


def run(folder,hub,stop):
    from pld_hdmi import PynqHdmiDisplay
    from touch_drag import DragInput
    from video_overlay import video_overlay
    client=Client(hub);surface=Surface()
    state=client.request('/interactive/state');surface.update(state)
    if state['model']!='skin' or state['phase']!='ready':return
    if state['result_ready']:
        surface.update(state,client.request('/interactive/jobs/'+state['job_id'])['result'])
    base=video_overlay();display=PynqHdmiDisplay(base.video.hdmi_out);touch=DragInput()
    ready=folder/'hdmi.ready'
    next_sync=0
    try:
        display.start();display.show(surface.frame());ready.write_text('interactive')
        while not stop.wait(.02):
            changed=False
            try:
                if time.monotonic()>=next_sync:
                    token=surface.state.get('image_token') or ''
                    state=client.request('/interactive/state?image_token='+token)
                    if state['phase']!='ready':break
                    result=None
                    if state['result_ready'] and surface.result_id!=state['job_id']:
                        result=client.request('/interactive/jobs/'+state['job_id'])['result']
                    changed=surface.update(state,result) or changed;next_sync=time.monotonic()+.5
                for action,x,y in touch.poll_events():
                    command=surface.gesture(action,x,y);changed=True
                    if command:
                        path,body=command;reply=client.request(path,body)
                        print('ROI_TOUCH',action,x,y,path,flush=True)
                        surface.message=''
                        if path=='/interactive/hdmi':stop.set();break
                        if path=='/interactive/control':surface.update(reply)
                        else:surface.state['busy']=True
                        next_sync=0
            except (OSError,RuntimeError,ValueError) as exc:
                if stop.is_set():break
                surface.message=str(exc);changed=True;next_sync=time.monotonic()+1
            if changed and not stop.is_set():display.show(surface.frame())
    finally:
        touch.close();display.close();ready.unlink(missing_ok=True)
