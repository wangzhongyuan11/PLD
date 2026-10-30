"""Completed ACDC MPR results on the existing PYNQ HDMI transport."""
from display_web.latency import Stages
startup = Stages("acdc_hdmi")

import argparse
import io
import json
from pathlib import Path
import signal
import threading

import numpy as np
from PIL import Image, ImageDraw

from .mpr import MPRVolume
from .native import load_native
from .surface_hdmi import load_mesh, render
from display_web.telemetry_hud import load_telemetry, annotate
from display_web.demo_metrics import draw_comparison


class HeartView:
    def __init__(self, folder, kernel):
        self.folder = Path(folder)
        self.telemetry = load_telemetry(self.folder, 'acdc')
        self.metrics_open = False
        self.volumes = {p: MPRVolume(self.folder/'viewer'/p, kernel) for p in ('ED', 'ES')}
        self.meta = {p: json.loads((self.folder/'viewer'/p/'volume.json').read_text()) for p in self.volumes}
        self.phase = 'ED'
        self.plane = 0
        self.positions = {p: list(self.meta[p]['position']) for p in self.volumes}
        self.planes = ('axial', 'coronal', 'sagittal')
        self.quad = True
        self.surface = False
        self.meshes = {p: load_mesh(self.folder, p) for p in ('ED', 'ES')}
        self.yaw, self.pitch = .55, -1.05
        self.slice_cache = {}

    def drag(self, dx, dy):
        self.yaw += dx * .008
        self.pitch += dy * .008
        return None

    def tap(self, x, y):
        if 48 <= y < 116 or (self.metrics_open and y >= 48):
            self.metrics_open = not self.metrics_open
            return None
        if y < 48:
            button = min(5, x//107)
            if button == 0 and not self.quad:
                self.quad = True; self.surface = False
            elif button < 2: self.phase = ('ED', 'ES')[button]
            elif button == 2: self.quad = True; self.surface = False
            elif button == 3: self.plane = (self.plane+1)%3;self.quad = False;self.surface = False
            else:
                axis = (2, 1, 0)[self.plane]
                bounds = self.volumes[self.phase].bounds
                pos = self.positions[self.phase]
                pos[axis] = float(np.clip(pos[axis]+(-2 if button==4 else 2), bounds[0,axis], bounds[1,axis]))
        elif self.quad:
            index = (int(y >= 264)*2 + int(x >= 320))
            if index < 3: self.plane = index;self.quad = False;self.surface = False
            else: self.surface = True;self.quad = False
        elif self.surface:
            pass
        else:
            # Tap the enlarged slice to move the RAS cursor in that plane.
            m = self.last_meta[self.plane]
            pos = self.positions[self.phase];b = self.volumes[self.phase].bounds
            pos[m['u']] = float(np.clip(b[0,m['u']]+(x-m['ox'])/m['scale'], b[0,m['u']], b[1,m['u']]))
            pos[m['v']] = float(np.clip(b[1,m['v']]-(y-72-m['oy'])/m['scale'], b[0,m['v']], b[1,m['v']]))
        return None

    def frame(self, preview=False):
        phase=self.phase
        key=(phase,self.quad,self.surface,self.plane,tuple(self.positions[phase]))
        if getattr(self,'base_key',None)!=key:
            image = Image.new('RGB', (640,480), (15,28,40));draw = ImageDraw.Draw(image)
            for i, label in enumerate(('ED' if self.quad else '< BACK','ES','4 views','Plane','Slice -','Slice +')):
                x=i*107;draw.rectangle((x+2,2,min(x+104,639),46),fill=(160,75,20) if i==0 and not self.quad else (35,85,105))
                draw.text((x+12,19),label,fill='white')
            self.last_meta = {}
            volume=self.volumes[phase];meta=self.meta[phase]
            indices=() if self.surface else (range(3) if self.quad else (self.plane,))
            for i in indices:
                x=(i%2)*320 if self.quad else 0;y=48+(i//2)*216 if self.quad else 48
                h,w=(192,320) if self.quad else (408,640)
                slice_key=(phase,i,h,w,tuple(self.positions[phase]))
                if slice_key not in self.slice_cache:
                    raw,m=volume.render(self.planes[i],self.positions[phase],shape=(h,w),level=meta['window'][0],window=meta['window'][1])
                    with Image.open(io.BytesIO(raw)) as im: tile=im.convert('RGB')
                    if len(self.slice_cache)>=12:self.slice_cache.clear()
                    self.slice_cache[slice_key]=(tile,m)
                tile,m=self.slice_cache[slice_key];self.last_meta[i]=m
                draw.text((x+8,y+6),phase+' / '+self.planes[i]+' / %.1f mm'%self.positions[phase][(2,1,0)[i]],fill='white')
                image.paste(tile,(x,y+24))
            if self.quad:draw.text((330,270),'Surface / drag to rotate',fill='white')
            self.base_frame=np.asarray(image).copy();self.base_key=key
        screen=self.base_frame.copy()
        if self.surface:
            screen[48:480,:,:]=render(self.meshes[phase],self.yaw,self.pitch,phase,preview=preview,size=(640,432))
        elif self.quad:
            screen[288:480,320:640,:]=render(self.meshes[phase],self.yaw,self.pitch,phase,preview=preview,size=(320,192))
        mode = self.phase + (' / four views' if self.quad else ' / surface' if self.surface else ' / '+self.planes[self.plane])
        if self.metrics_open:return draw_comparison(screen,self.telemetry.get('comparison'),y=48)
        return annotate(screen, self.telemetry, y=48, mode=mode)


startup.mark("module_imports")

def main(argv=None, stop=None):
    global startup
    if stop is not None:startup=Stages('acdc_hdmi')
    parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args(argv)
    from display_web.pld_hdmi import PynqHdmiDisplay
    from .touch_drag import DragInput
    from display_web.video_overlay import video_overlay
    startup.mark('pynq_import')
    if stop is None:
        stop=threading.Event()
        signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    view=HeartView(args.result,load_native());screen=view.frame()
    startup.mark("prepare_view")
    base=video_overlay();startup.mark('base_overlay');display=PynqHdmiDisplay(base.video.hdmi_out);touch=DragInput()
    ready=args.result/'hdmi.ready.json'
    try:
        display.start();startup.mark("video_start");display.show(screen);startup.mark("first_frame")
        ready.write_text(json.dumps(dict(view='acdc_mpr',resolution=[640,480],scope='frame submitted',timing=startup.snapshot())))
        anchor=None;rotate=False;moved=False
        while not stop.wait(.02):
            changed=False
            for action,x,y in touch.poll_events():
                if action=='down':
                    anchor=(x,y);moved=False
                    rotate=not view.metrics_open and ((view.surface and y>=116) or (view.quad and x>=320 and y>=264))
                elif action=='move' and anchor is not None:
                    dx,dy=x-anchor[0],y-anchor[1]
                    if abs(dx)+abs(dy)>2:moved=True
                    if rotate:
                        view.yaw+=dx*.008;view.pitch+=dy*.008;changed=True
                    anchor=(x,y)
                elif action=='up':
                    if rotate and moved:changed=True
                    if not moved:view.tap(x,y);changed=True
                    anchor=None;rotate=False
                elif action=='cancel':anchor=None;rotate=False
            if changed:
                display.show(view.frame(preview=rotate and moved));print('TOUCH_UPDATE',view.phase,view.yaw,view.pitch,flush=True)
    finally:
        touch.close();display.close();ready.unlink(missing_ok=True)


if __name__=='__main__':main()
