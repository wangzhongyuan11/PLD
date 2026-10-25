"""Drag events adapted from the deployed ACDC touch reader."""
import os
import time
from touch_display import TouchInput


class DragInput(TouchInput):
    def __init__(self):
        super().__init__()
        self.previous = None
        self.buffer = b''

    def poll_events(self):
        if self.fd is None:
            if time.monotonic() >= self.retry:
                self.connect()
            return []
        events = []
        try:
            for _ in range(16):
                data = os.read(self.fd, self.event.size*64)
                if not data:
                    raise OSError('disconnected')
                self.buffer += data
                while len(self.buffer) >= self.event.size:
                    _, _, kind, code, value = self.event.unpack(self.buffer[:self.event.size])
                    self.buffer = self.buffer[self.event.size:]
                    if kind==3 and code==47:
                        self.slot = value
                    if kind==3 and (self.axes==(0,1) or self.slot==0):
                        if code==self.axes[0]: self.x=value
                        if code==self.axes[1]: self.y=value
                        if code==57 and self.axes==(53,54): self.down=value>=0
                    if kind==1 and code==330: self.down=bool(value)
                    if kind==0 and code==3:
                        self.down=False; self.previous=None; events.append(('cancel',0,0))
                    if kind==0 and code==0:
                        xy=tuple(max(0,min(size-1,int((v-lo)*(size-1)/(hi-lo))))
                            for v,(lo,hi),size in zip((self.x,self.y),self.ranges,(640,480)))
                        if self.down:
                            events.append(('down' if self.previous is None else 'move',*xy))
                            self.previous=xy
                        elif self.previous is not None:
                            events.append(('up',*xy)); self.previous=None
        except BlockingIOError:
            pass
        except OSError:
            self.close(); self.down=False; self.previous=None; self.buffer=b''
            events.append(('cancel',0,0))
        return events
