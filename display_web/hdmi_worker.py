"""Exclusive HDMI child. Started only after the inference subprocess exits."""
import signal
import sys
import time
import threading
from pathlib import Path
from PIL import Image
import numpy as np
from pld_hdmi import PynqHdmiDisplay, fit_rgb
from quad_display import skin_frame

stop_requested = threading.Event()


def main():
    folder, view = Path(sys.argv[1]), sys.argv[2]
    if view == 'interactive':
        hub = Path(__file__).resolve().parent / 'unified'
        if not hub.is_dir(): hub = Path(__file__).resolve().parent.parent / 'hub'
        sys.path.insert(0, str(hub))
        from interactive_hdmi import run
        return run(folder, hub, stop_requested)
    screen = skin_frame(folder)
    from pynq.overlays.base import BaseOverlay
    from hdmi_buttons import DebouncedButtons, handle_press
    screens = [screen, screen, screen]
    base = BaseOverlay("base.bit")
    display = PynqHdmiDisplay(base.video.hdmi_out)
    try:
        display.start(); display.show(screen)
        (folder / "hdmi.ready").write_text("ready")
        debounce = DebouncedButtons(base.buttons.read(), time.monotonic())
        while not stop_requested.is_set():
            pressed = debounce.update(base.buttons.read(), time.monotonic())
            if pressed: handle_press(pressed, display, screens)
            time.sleep(.01)
    finally:
        display.close()
        (folder / "hdmi.ready").unlink(missing_ok=True)


if __name__ == "__main__":
    # Do not interrupt overlay initialization halfway through and claim cleanup.
    # The parent refuses further hardware work if initialization cannot return.
    def stop(signum, frame): stop_requested.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try: main()
    except KeyboardInterrupt: pass
