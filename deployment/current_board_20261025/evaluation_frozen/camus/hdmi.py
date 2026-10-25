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
from .accelerator import HardwareLease


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True)
    p.add_argument('--view',choices=('overlay','segmentation','label','label_raw'),default='overlay')
    p.add_argument('--duration',type=float,default=3600.)
    args=p.parse_args(argv)
    if not 0<args.duration<=86400:p.error('duration must be in (0,86400] seconds')
    folder=args.result.resolve();source=folder/(args.view+'.png')
    if not source.is_file():p.error('completed CAMUS result PNG required')
    stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    # Reuse C's display transport without changing its binary-mask/UI contract.
    from display_web.pld_hdmi import PynqHdmiDisplay,fit_rgb
    from pynq.overlays.base import BaseOverlay
    ready=folder/'hdmi.ready.json';display=None;lease=HardwareLease()
    try:
        rgb=np.asarray(Image.open(source).convert('RGB'))
        screen=fit_rgb(rgb,640,480,nearest=args.view in ('label','label_raw','segmentation'))
        base=BaseOverlay('base.bit');display=PynqHdmiDisplay(base.video.hdmi_out)
        display.start();display.show(screen)
        evidence=dict(scope='HDMI overlay/configure/frame submission; physical screen appearance requires observation',
                      view=args.view,resolution=[640,480],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      frame_sha256=hashlib.sha256(screen.tobytes()).hexdigest())
        ready.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(evidence),flush=True);stop.wait(args.duration)
    finally:
        try:
            if display is not None:display.close()
            ready.unlink(missing_ok=True)
        finally:lease.close()


if __name__=='__main__':main()
