"""Bounded AXI DMA v7.1 direct-register control for simple-mode overlays.

PG021 register map; deliberately does not instantiate PYNQ's DMA driver,
whose constructor/start/stop/wait contain unbounded loops in PYNQ 3.1.1.
Single owner, one packet at a time. Caller owns allocation and PL reset.
"""
import math
import operator
import time


class DMAFault(RuntimeError):
    def __init__(self, reason, registers):
        self.reason = reason
        self.registers = registers
        super().__init__(f"{reason}: {registers}")


class SimpleDMA:
    def __init__(self, mmio, parameters, timeout=2.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.mmio, self.timeout = mmio, timeout
        self.active = False
        self._pending = None
        p = {key.lower(): int(value, 0) if isinstance(value, str) else int(value)
             for key, value in parameters.items() if key.lower() in {
                 "c_include_sg", "c_micro_dma", "c_addr_width", "c_sg_length_width",
                 "c_m_axi_mm2s_data_width", "c_m_axi_s2mm_data_width"}}
        if p.get("c_include_sg", 1) or p.get("c_micro_dma", 0):
            raise ValueError("requires simple, non-micro DMA")
        self.max_size = (1 << p["c_sg_length_width"]) - 1
        self.address_limit = 1 << p["c_addr_width"]
        self.align = tuple(p[f"c_m_axi_{name}_data_width"] // 8 for name in ("mm2s", "s2mm"))

    def snapshot(self):
        return {name: f"0x{self.mmio.read(offset):08x}" for name, offset in (
            ("MM2S_CR", 0), ("MM2S_SR", 4), ("MM2S_ADDR", 0x18), ("MM2S_LEN", 0x28),
            ("S2MM_CR", 0x30), ("S2MM_SR", 0x34), ("S2MM_ADDR", 0x48), ("S2MM_LEN", 0x58))}

    def _poll(self, predicate, reason):
        deadline = time.monotonic() + self.timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise DMAFault(reason, self.snapshot())

    def reset(self):
        # Reset either channel resets the entire DMA, not the external PL core.
        self.mmio.write(0, 4)
        self._poll(lambda: not (self.mmio.read(0) & 4) and
                   (self.mmio.read(4) & 1) and (self.mmio.read(0x34) & 1),
                   "reset timeout")
        self.active = False
        self._pending = None

    def start(self):
        for base in (0, 0x30):
            self.mmio.write(base + 4, 0x7000)  # W1C IRQ status
            self.mmio.write(base, 1)
        self._poll(lambda: not ((self.mmio.read(4) | self.mmio.read(0x34)) & 1),
                   "start timeout")

    def _validate(self, buffer, nbytes, alignment):
        try:
            nbytes = operator.index(nbytes)
        except TypeError as exc:
            raise ValueError("DMA byte count must be an integer") from exc
        addr = int(buffer.physical_address)
        if not 0 < nbytes <= min(buffer.nbytes, self.max_size):
            raise ValueError("invalid DMA byte count")
        if addr < 0 or addr % alignment or addr + nbytes > self.address_limit:
            raise ValueError("invalid DMA address/alignment")

    def submit(self, source=None, destination=None, *, send_nbytes=None, receive_nbytes=None):
        """Arm one or both channels; wait() completes the same transaction.

        RX is armed before TX when both are present. Buffers are retained until
        completion or successful reset. The caller must not free/reuse them.
        """
        self._submit(source, destination, send_nbytes, receive_nbytes)

    def _submit(self, source, destination, send_nbytes, receive_nbytes, skip_send=False):
        if self.active:
            raise RuntimeError("previous transfer still active; reset required")
        if source is None and destination is None:
            raise ValueError("at least one DMA channel is required")
        for buf, length, alignment in ((source, send_nbytes, self.align[0]),
                                       (destination, receive_nbytes, self.align[1])):
            if buf is None:
                if length is not None:
                    raise ValueError("byte count requires a buffer")
            else:
                self._validate(buf, length, alignment)
        if source is not None:
            source.flush()
        # Flush dirty destination cache lines before device writes, invalidate
        # only after completion; otherwise later writeback can corrupt results.
        if destination is not None:
            destination.flush()
        channels = tuple(base for base, buf in ((0, source), (0x30, destination)) if buf is not None)
        self._pending = (source, destination, receive_nbytes, channels)
        self.active = True
        for base in channels:
            self.mmio.write(base + 4, 0x7000)
        for base, buf, length in ((0x30, destination, receive_nbytes), (0, source, send_nbytes)):
            if buf is None or (base == 0 and skip_send):
                continue
            addr = int(buf.physical_address)
            self.mmio.write(base + 0x18, addr & 0xFFFFFFFF)
            self.mmio.write(base + 0x1C, addr >> 32)
            self.mmio.write(base + 0x28, int(length))  # length starts channel; RX first

    def wait(self):
        if not self.active or self._pending is None:
            raise RuntimeError("no pending DMA transaction")
        _, destination, receive_nbytes, channels = self._pending

        def complete():
            status = tuple(self.mmio.read(base+4) for base in channels)
            if any(s & 0x70 for s in status):
                raise DMAFault("DMA error", self.snapshot())
            if any(s & 1 for s in status):
                raise DMAFault("unexpected halt", self.snapshot())
            # Require fresh IOC and idle on every participating channel.
            return all(s & 0x1002 == 0x1002 for s in status)

        self._poll(complete, "transfer timeout")
        if destination is not None:
            received = self.mmio.read(0x58)
            if received != receive_nbytes:
                raise DMAFault(f"short packet: received={received}, expected={receive_nbytes}", self.snapshot())
            destination.invalidate()
        self.active = False
        self._pending = None

    def send(self, source, nbytes):
        self.submit(source=source, send_nbytes=nbytes)
        self.wait()

    def receive(self, destination, nbytes):
        self.submit(destination=destination, receive_nbytes=nbytes)
        self.wait()

    def transfer(self, source, destination, nbytes=None, *, send_nbytes=None,
                 receive_nbytes=None, receive_only=False):
        """Duplex transfer, with equal legacy or independently specified lengths.

        receive_only remains the stage-2 timeout injection: it deliberately
        skips TX but waits for BOTH channels. Use receive() for real RX-only.
        """
        if nbytes is not None:
            if send_nbytes is not None or receive_nbytes is not None:
                raise ValueError("choose equal or independent lengths, not both")
            send_nbytes = receive_nbytes = nbytes
        self._submit(source, destination, send_nbytes, receive_nbytes, skip_send=receive_only)
        self.wait()
