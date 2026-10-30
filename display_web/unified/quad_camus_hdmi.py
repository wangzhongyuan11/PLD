"""Submit completed four-panel CAMUS frame after inference releases PL."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import numpy as np
from PIL import Image


def main(argv=None, stop=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--view',choices=('quad','overlay','segmentation','label','label_raw'),default='quad')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.source.resolve()))
    if stop is None:
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:from display_web.display_lease import CamusDisplayLease
    except ModuleNotFoundError:from display_lease import CamusDisplayLease
    from display_web.pld_hdmi import PynqHdmiDisplay,fit_rgb
    from display_web.touch_display import TouchInput,QuadTouch
    try:from display_web.telemetry_hud import load_telemetry,annotate
    except ModuleNotFoundError:from telemetry_hud import load_telemetry,annotate
    try:from display_web.video_overlay import video_overlay
    except ModuleNotFoundError:from video_overlay import video_overlay
    folder = args.result.resolve()
    telemetry = load_telemetry(folder, 'camus')
    view=args.view
    def frame(selected):
        with Image.open(folder/(selected+'.png')) as im:rgb=np.asarray(im.convert('RGB')).copy()
        return fit_rgb(rgb,640,480,nearest=selected in ('label','label_raw','segmentation'))
    screen=frame(view)
    lease = CamusDisplayLease()
    display = None;touch=None
    ready = folder/'hdmi.ready.json'
    try:
        base = video_overlay()
        display = PynqHdmiDisplay(base.video.hdmi_out)
        display.start()
        quad=QuadTouch(screen,telemetry) if view=='quad' else None
        shown=quad.frame() if quad else annotate(screen,telemetry)
        display.show(shown)
        ready.write_text(json.dumps(dict(view=view, resolution=[640,480],
            frame_sha256=hashlib.sha256(shown.tobytes()).hexdigest(),
            scope='frame submission only; observe physical screen separately'))+'\n')
        touch=TouchInput()
        while not stop.wait(.02):
            update=getattr(stop,'display_update',None)
            if update is not None:
                stop.display_update=None
                selected=update.get('view')
                if selected not in ('quad','overlay','segmentation','label','label_raw'):raise ValueError('invalid display update')
                view=selected;screen=frame(view);quad=QuadTouch(screen,telemetry) if view=='quad' else None
                display.show(quad.frame() if quad else annotate(screen,telemetry))
                ready.write_text(json.dumps(dict(view=view,resolution=[640,480],scope='frame update submitted')))
            for x,y in touch.poll():
                if quad:display.show(quad.tap(x,y))
                else:
                    views=('overlay','segmentation','label','label_raw')
                    view=views[(views.index(view)+1)%len(views)];display.show(annotate(frame(view),telemetry))
                print('CAMUS_TOUCH',x,y,view,flush=True)
    finally:
        try:
            if display is not None:
                display.close()
            if touch is not None:touch.close()
        finally:
            ready.unlink(missing_ok=True)
            lease.close()


if __name__ == '__main__':
    main()
