"""Exclusive CAMUS result display. Inference DMA must be closed before entry."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import signal
import threading
import time
import numpy as np
from PIL import Image


def main(argv=None, stop=None):
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True)
    p.add_argument('--view',choices=('overlay','segmentation','label','label_raw'),default='overlay')
    p.add_argument('--duration',type=float,default=3600.)
    args=p.parse_args(argv)
    if not 0<args.duration<=86400:p.error('duration must be in (0,86400] seconds')
    folder=args.result.resolve();source=folder/(args.view+'.png')
    if not source.is_file():p.error('completed CAMUS result PNG required')
    if stop is None:
        stop=threading.Event()
        signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    # Reuse C's display transport without changing its binary-mask/UI contract.
    from display_web.pld_hdmi import PynqHdmiDisplay,fit_rgb
    from display_web.touch_display import TouchInput
    from display_web.telemetry_hud import load_telemetry,annotate
    from display_web.demo_metrics import draw_comparison
    try:from display_web.video_overlay import video_overlay
    except ModuleNotFoundError:from video_overlay import video_overlay
    try:from display_web.display_lease import CamusDisplayLease
    except ModuleNotFoundError:from display_lease import CamusDisplayLease
    ready=folder/'hdmi.ready.json';display=None;lease=CamusDisplayLease()
    try:
        rgb=np.asarray(Image.open(source).convert('RGB'))
        telemetry=load_telemetry(folder,'camus')
        screen=annotate(fit_rgb(rgb,640,480,nearest=args.view in ('label','label_raw','segmentation')),telemetry)
        base=video_overlay();display=PynqHdmiDisplay(base.video.hdmi_out)
        display.start();display.show(screen)
        evidence=dict(scope='HDMI overlay/configure/frame submission; physical screen appearance requires observation',
                      view=args.view,resolution=[640,480],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      frame_sha256=hashlib.sha256(screen.tobytes()).hexdigest())
        ready.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(evidence),flush=True)
        touch=TouchInput();views=['overlay','segmentation','label','label_raw'];selected=views.index(args.view);metrics_open=False
        deadline=time.monotonic()+args.duration
        try:
            while not stop.wait(.02) and time.monotonic()<deadline:
                for x,y in touch.poll():
                    if y<68 or metrics_open:
                        metrics_open=not metrics_open
                        with Image.open(folder/(views[selected]+'.png')) as im:
                            base_frame=fit_rgb(np.asarray(im.convert('RGB')),640,480,nearest=selected!=0)
                        display.show(draw_comparison(base_frame,telemetry.get('comparison')) if metrics_open else annotate(base_frame,telemetry))
                        continue
                    selected=(selected+1)%len(views)
                    with Image.open(folder/(views[selected]+'.png')) as im:
                        display.show(annotate(fit_rgb(np.asarray(im.convert('RGB')),640,480,nearest=selected!=0),telemetry))
                    print('TOUCH_TAP',x,y,views[selected],flush=True)
        finally:touch.close()
    finally:
        try:
            if display is not None:display.close()
            ready.unlink(missing_ok=True)
        finally:lease.close()


if __name__=='__main__':main()
