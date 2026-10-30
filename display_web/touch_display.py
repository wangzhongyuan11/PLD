"""Linux USB touch input and tap-to-enlarge for the existing HDMI frames."""
import glob
import os
import struct
import time

import numpy as np
from PIL import Image, ImageDraw


class TouchInput:
    def __init__(self):
        self.fd = None
        self.path = None
        self.retry = 0
        self.x = self.y = 0
        self.down = False
        self.pending = False
        self.slot = 0
        self.event = struct.Struct('@llHHi')

    def connect(self):
        import fcntl
        self.retry = time.monotonic() + 2
        for path in sorted(glob.glob('/dev/input/event*')):
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                for axes in ((0, 1), (53, 54)):
                    try:
                        ranges = []
                        for axis in axes:
                            info = bytearray(24)
                            fcntl.ioctl(fd, 0x80184540 + axis, info)
                            _, low, high, *_ = struct.unpack('6i', info)
                            if high <= low:
                                raise ValueError('no absolute range')
                            ranges.append((low, high))
                        self.axes, self.ranges = axes, ranges
                        self.fd, self.path = fd, path
                        print('TOUCH_CONNECTED', path, axes, ranges, flush=True)
                        return
                    except (OSError, ValueError):
                        continue
            finally:
                if self.fd != fd:
                    os.close(fd)

    def poll(self):
        if self.fd is None:
            if time.monotonic() >= self.retry:
                self.connect()
            return []
        taps = []
        try:
            while True:
                data = os.read(self.fd, self.event.size * 64)
                if not data:
                    raise OSError('touch disconnected')
                for _, _, kind, code, value in self.event.iter_unpack(data):
                    if kind == 3 and code == 47:
                        self.slot = value
                    if kind == 3 and (self.axes == (0, 1) or self.slot == 0):
                        if code == self.axes[0]: self.x = value
                        if code == self.axes[1]: self.y = value
                        if code == 57 and self.axes == (53, 54):
                            self.pending |= value < 0 and self.down
                            self.down = value >= 0
                    if kind == 1 and code == 330:
                        self.pending |= not value and self.down
                        self.down = bool(value)
                    if kind == 0 and code == 0 and self.pending:
                        self.pending = False
                        xy = [int(round((v-lo) * (size-1) / (hi-lo)))
                              for v, (lo, hi), size in zip((self.x, self.y), self.ranges, (640, 480))]
                        taps.append((max(0, min(639, xy[0])), max(0, min(479, xy[1]))))
        except BlockingIOError:
            pass
        except OSError:
            self.close()
        return taps

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
        self.fd = None


class QuadTouch:
    def __init__(self, screen, telemetry=None):
        self.original = Image.fromarray(screen)
        self.selected = None
        self.telemetry = telemetry
        self.metrics_open = False

    def tap(self, x, y):
        if self.telemetry is not None and (y < 68 or self.metrics_open):
            self.metrics_open = not self.metrics_open
            return self.frame()
        self.selected = None if self.selected is not None else (int(y >= 240)*2 + int(x >= 320))
        return self.frame()

    def frame(self):
        if self.metrics_open:
            try:from .demo_metrics import draw_comparison
            except ImportError:from demo_metrics import draw_comparison
            return draw_comparison(np.asarray(self.original),self.telemetry.get('comparison'))
        if self.selected is None:
            image = self.original.copy()
            label = 'Touch a panel to enlarge'
        else:
            x = 6 + (self.selected % 2)*317
            y = 30 + (self.selected // 2)*212
            image = self.original.crop((x, y, x+311, y+206)).resize((640, 460))
            canvas = Image.new('RGB', (640, 480));canvas.paste(image);image = canvas
            label = 'Touch to return to four views'
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 460, 640, 480), fill=(15, 28, 40))
        draw.text((10, 464), label, fill='white')
        if self.telemetry is not None:
            try:from .telemetry_hud import annotate
            except ImportError:from telemetry_hud import annotate
            # Annotate after enlargement; never zoom the metadata into a panel.
            draw.text((10, 447), label, fill='white')
            return annotate(np.asarray(image), self.telemetry)
        return np.asarray(image).copy()


def interact(display, screen, stop, duration=3600):
    touch = TouchInput();view = QuadTouch(screen)
    display.show(view.frame())
    deadline = time.monotonic()+duration
    try:
        while not stop.wait(.02) and time.monotonic() < deadline:
            for x, y in touch.poll():
                display.show(view.tap(x, y))
                print('TOUCH_TAP', x, y, 'panel', view.selected, flush=True)
    finally:
        touch.close()
