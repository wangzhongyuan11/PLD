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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    from camus.accelerator import HardwareLease
    from display_web.pld_hdmi import PynqHdmiDisplay
    from pynq.overlays.base import BaseOverlay
    folder = args.result.resolve()
    with Image.open(folder/'quad.png') as im:
        screen = np.asarray(im.convert('RGB')).copy()
    if screen.shape != (480, 640, 3):
        raise ValueError('640x480 four-panel frame required')
    lease = HardwareLease()
    display = None
    ready = folder/'hdmi.ready.json'
    try:
        base = BaseOverlay('base.bit')
        display = PynqHdmiDisplay(base.video.hdmi_out)
        display.start()
        display.show(screen)
        ready.write_text(json.dumps(dict(view='quad', resolution=[640,480],
            frame_sha256=hashlib.sha256(screen.tobytes()).hexdigest(),
            scope='frame submission only; observe physical screen separately'))+'\n')
        stop.wait(3600)
    finally:
        try:
            if display is not None:
                display.close()
        finally:
            ready.unlink(missing_ok=True)
            lease.close()


if __name__ == '__main__':
    main()
