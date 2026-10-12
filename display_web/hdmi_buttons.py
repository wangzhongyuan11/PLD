"""Run on PYNQ-Z2: python3 hdmi_buttons.py. Ctrl+C exits."""
import platform
import signal
import time

from pld_hdmi import (PynqHdmiDisplay, compose_dashboard,
                      synthetic_sample, test_pattern)


class DebouncedButtons:
    def __init__(self, initial, now, delay=0.04):
        self.stable = self.candidate = initial
        self.changed_at = now
        self.delay = delay

    def update(self, value, now):
        if value != self.candidate:
            self.candidate, self.changed_at = value, now
        if value != self.stable and now - self.changed_at >= self.delay:
            pressed = value & ~self.stable
            self.stable = value
            return pressed
        return 0


def handle_press(pressed, display, screens):
    # Blank wins when several buttons are pressed together. Do not tear down
    # VDMA on BTN3: subsequent frames must use the same running pipeline.
    if pressed & 8:
        display.blank()
        print("Screen blanked; HDMI remains active. Press BTN0/1/2 to restore.", flush=True)
        return
    for index in range(3):
        if pressed & (1 << index):
            display.start()
            display.show(screens[index])
            print("Showing BTN{} screen".format(index), flush=True)
            return


def main():
    import sys
    if len(sys.argv) > 1:
        # Existing no-argument white box/bars/demo mode is preserved.
        # Result mode immediately displays the cutout and enables the buttons.
        from show_result import main as result_main
        result_main(["--hdmi", "--buttons"] + sys.argv[1:])
        return
    if platform.system() != "Linux" or not platform.machine().lower().startswith("arm"):
        raise RuntimeError("Run this program on the PYNQ-Z2, not on Windows.")
    from pynq.overlays.base import BaseOverlay
    print("Loading base.bit; stop other HDMI/DMA programs first.", flush=True)
    base = BaseOverlay("base.bit")
    display = PynqHdmiDisplay(base.video.hdmi_out, 640, 480)
    try:
        rgb, mask = synthetic_sample()
        dashboard, _ = compose_dashboard(rgb, mask, demo=True)
        screens = [test_pattern(pattern="rectangle"), test_pattern(), dashboard]
        buttons = DebouncedButtons(base.buttons.read(), time.monotonic())
        print("Ready: BTN0 rectangle | BTN1 bars | BTN2 demo | BTN3 black screen", flush=True)
        print("Use the four user buttons, not RESET. Ctrl+C exits.", flush=True)
        while True:
            pressed = buttons.update(base.buttons.read(), time.monotonic())
            if pressed:
                handle_press(pressed, display, screens)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Exiting.", flush=True)
    finally:
        display.close()


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    main()
