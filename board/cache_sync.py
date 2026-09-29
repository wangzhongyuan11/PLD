"""TX-only XRT prefix synchronization. RX retains full-buffer guard checking."""
import operator


class XrtTXPrefix:
    def __init__(self, buffer, length, direction):
        length = operator.index(length)
        if buffer.offset != 0 or buffer.coherent or buffer.physical_address % 64:
            raise ValueError('prefix sync requires a cacheable aligned root allocation')
        if not 0 < length <= buffer.nbytes or buffer.nbytes % 64:
            raise ValueError('invalid prefix or allocation size')
        self.buffer = buffer  # Keep the allocation alive through DMA wait/reset.
        self.physical_address = buffer.physical_address
        self.nbytes = length
        self.sync_bytes = (length+63)//64*64
        self.direction = direction

    def flush(self):
        # The PYNQ 3.1.1 device wrapper discards size/offset. Call the supported
        # XRT overload directly: sync(direction, size, offset), rounded outward.
        self.buffer.bo.sync(self.direction, self.sync_bytes, 0)

    def invalidate(self):
        raise RuntimeError('TX prefix must never be used as a receive buffer')
