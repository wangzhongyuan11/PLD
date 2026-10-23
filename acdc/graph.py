"""21-node graph with a full oracle and the read-only mature tiled device."""
from __future__ import annotations

import time
import numpy as np

from . import integer
from .contract import ContractError, NODE_NAMES, checked_int8


class FullDevice:
    def conv(self, x, w, b, m, *, padding, relu):
        return integer.conv_full(x, w, b, m, padding, relu)

    max_pool = staticmethod(integer.pool)
    concat_upsample_rescale = staticmethod(integer.concat)

    def begin_frame(self):
        pass

    def close(self):
        pass


class Graph:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or FullDevice()

    def forward(self, first_input, *, capture=False, cancel=None):
        x = checked_int8(first_input, self.model.input_shape)
        self.device.begin_frame()
        skips, captures, timings = {}, {}, []
        cursor = 0

        def execute(name, fn):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("job cancelled at a network node boundary")
            started = time.perf_counter()
            start_event=len(getattr(self.device,'events',[]))
            result = fn()
            events=getattr(self.device,'events',[])[start_event:]
            timings.append(dict(node=name, seconds=time.perf_counter()-started,
                                commands=len(events),compute_cycles=sum(e.get('compute_cycles',0) for e in events)))
            if capture:
                captures[name] = result.copy()
            return result

        def conv(value):
            nonlocal cursor
            l = self.model.layers[cursor]
            cursor += 1
            return execute(l.name, lambda: self.device.conv(
                value, l.weight, l.bias, l.multipliers, padding=l.padding, relu=l.relu))

        for level in range(4):
            x = conv(conv(x))
            if level < 3:
                skips[level] = x
                x = execute(f"P{level}", lambda: self.device.max_pool(x))
        for i, cat in enumerate(self.model.concats):
            skip = skips.pop(cat.level)
            x = execute(f"CAT{i+1}", lambda: self.device.concat_upsample_rescale(
                skip, x, cat.multiplier_q24))
            del skip
            x = conv(conv(x))
        x = conv(x)
        if cursor != 15 or x.shape != self.model.head_shape or tuple(t['node'] for t in timings) != NODE_NAMES:
            raise ContractError("graph execution incomplete")
        return dict(head=x.copy(), head_scale=self.model.head_scale, captures=captures,
                    nodes=timings, events=list(getattr(self.device, 'events', [])))

    def close(self):
        self.device.close()


def emulator():
    # Exact tiled algorithm, kept distinct from the independent oracle above.
    from camus.device import EmulatedTileAccelerator
    return EmulatedTileAccelerator()
