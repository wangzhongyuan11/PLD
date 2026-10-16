"""Display B's .bin + .json result on a PC or PYNQ-Z2 HDMI."""
import argparse
import math
import platform
import signal
import time
from pathlib import Path

from pld_hdmi import PynqHdmiDisplay
from pld_result import load_result, save_result_images

RESULT_BUTTON_VIEWS = ("original", "cutout", "dashboard")

def show_on_board(result, view="cutout", buttons=False, hold=0):
    if platform.system() != "Linux" or not platform.machine().lower().startswith("arm"):
        raise RuntimeError("HDMI mode must run on PYNQ-Z2, not this PC")
    # Prepare/validate everything before replacing the computation overlay.
    screen = result.screen(view)
    screens = None
    if buttons:
        screens = [result.screen(v) for v in (RESULT_BUTTON_VIEWS
                   if result.rgb is not None else ("mask", "mask", "mask"))]
    from pynq.overlays.base import BaseOverlay
    from hdmi_buttons import DebouncedButtons, handle_press
    print("Loading base.bit. The B inference process must already have exited and released DMA.", flush=True)
    base = BaseOverlay("base.bit")
    display = PynqHdmiDisplay(base.video.hdmi_out)
    try:
        display.start()
        display.show(screen)
        debounce = DebouncedButtons(base.buttons.read(), time.monotonic()) if buttons else None
        print("Showing {}. Ctrl+C closes HDMI.".format(view), flush=True)
        if buttons:
            print("BTN0 original | BTN1 cutout | BTN2 comparison | BTN3 black screen" if result.rgb is not None
                  else "Mask-only: BTN0/1/2 mask | BTN3 black screen", flush=True)
        started = time.monotonic()
        while hold == 0 or time.monotonic() - started < hold:
            if debounce is not None:
                pressed = debounce.update(base.buttons.read(), time.monotonic())
                if pressed:
                    handle_press(pressed, display, screens)
            time.sleep(0.01 if buttons else 0.1)
    finally:
        display.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="frame .json or .bin; both must exist")
    parser.add_argument("--image", help="Exact original JPEG used for inference")
    parser.add_argument("--view", choices=("cutout", "crop", "mask", "original", "dashboard"), default=None)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview", action="store_true", help="Write PNGs only (default)")
    modes.add_argument("--hdmi", action="store_true", help="Load base.bit AFTER inference has stopped")
    parser.add_argument("--buttons", action="store_true", help="Enable board buttons with --hdmi")
    parser.add_argument("--hold", type=float, default=0, help="Display seconds; 0 waits for Ctrl+C")
    parser.add_argument("--out", help="New preview directory (must not exist)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.hold) or args.hold < 0:
        parser.error("--hold must be finite and >= 0")
    if args.buttons and not args.hdmi:
        parser.error("--buttons requires --hdmi")
    view = args.view or ("cutout" if args.image else "mask")
    result = load_result(args.result, args.image)
    destination = args.out or str(Path("preview") / ("result_" + time.strftime("%Y%m%d_%H%M%S")))
    save_result_images(result, destination, view)
    print("Saved:", Path(destination).resolve(), flush=True)
    print("Area: {:.2f}% of VALID region".format(result.stats["area_percent"]), flush=True)
    if args.hdmi:
        show_on_board(result, view, args.buttons, args.hold)


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped; HDMI released.")
