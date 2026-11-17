import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Event, FallingEdge, ReadOnly, RisingEdge, Timer

TCLK_NS = 25                           # clk 40 MHz
TSCLK_NS = 50                          # SCLK_IN 20 MHz
GAP = 5                                   # SCLK_IN periods with CS high per sample


def s8(v):
    """Signed value of an 8-bit field."""
    v = int(v) & 0xFF
    return v - 256 if v & 0x80 else v


def bit(sig):
    """'0' or '1', or 'x' for anything unresolved (gate level)."""
    v = str(sig.value).lower()
    return v if v in ("0", "1") else "x"


class DutTB:
    def __init__(self, dut, tclk_ns=TCLK_NS, tsclk_ns=TSCLK_NS, gap=GAP):
        self.dut = dut
        self.tclk_ns, self.tsclk_ns, self.gap = tclk_ns, tsclk_ns, gap
        self.frames = []                  # (y, e, w), signed; None for a lane with X
        self.bad_frames = 0               # frames cut short by a new FS_OUT
        self.sent = 0
        self._sclk_rise = Event()
        self._sclk_fall = Event()

    async def start(self):
        """Clocks running, every input idle, chip in reset."""
        d = self.dut
        d.ena.value = 1
        d.rst_n.value = 0
        d.ui_in.value = 1 << 1  # SCLK_IN=0, CS_IN=1, MOSI lanes=0
        d.uio_in.value = 0
        Clock(d.clk, self.tclk_ns, "ns").start(start_high=False)
        cocotb.start_soon(self._sclk_in())
        cocotb.start_soon(self._receive())

    async def reset(self):
        await self.start()
        await ClockCycles(self.dut.clk, 10)
        self.dut.rst_n.value = 1
        await ClockCycles(self.dut.clk, 10)  # reset synchronizer

    async def _sclk_in(self):
        # SCLK_IN is not aligned with clk; an odd period gets the extra ns low
        await Timer(3300, "ns")
        half = self.tsclk_ns // 2
        while True:
            self._set_input(sclk=1)
            await Timer(1, "ps")
            self._sclk_rise.set()
            await Timer(half, "ns")
            self._set_input(sclk=0)
            await Timer(1, "ps")
            self._sclk_fall.set()
            await Timer(half, "ns")

    async def _receive(self):
        d = self.dut
        n, ys, es, ws = -1, "", "", ""
        sclk_prev = 0
        fs_prev = 1
        while True:
            await RisingEdge(d.clk)
            await ReadOnly()
            out = int(d.uo_out.value)
            sclk = out & 1
            fs = (out >> 1) & 1
            if not sclk_prev and sclk:
                # This serializer uses FS_OUT low while transmitting and
                # high while idle; its data lanes are LSB-first.
                if fs == 0:
                    if fs_prev == 1 and n > 0:
                        self.bad_frames += 1
                    if n < 0:
                        n, ys, es, ws = 0, "", "", ""
                if n >= 0:
                    ys += str((out >> 2) & 1)
                    es += str((out >> 3) & 1)
                    ws += str((out >> 4) & 1)
                    n += 1
                    if n == 8:
                        self.frames.append(tuple(s8(int(b[::-1], 2))
                                                 for b in (ys, es, ws)))
                        n = -1
            sclk_prev = sclk
            fs_prev = fs

    def set_ctl(self, adapt_en, mu_sel):
        """Control pins; they reach the core through 2-flop synchronizers."""
        self.dut.uio_in.value = ((mu_sel & 0x7) << 1) | int(bool(adapt_en))

    def ovf(self):
        return (int(self.dut.uo_out.value) >> 5) & 1

    def sat(self):
        return (int(self.dut.uo_out.value) >> 6) & 1

    def _set_input(self, *, sclk=None, cs_n=None, mosi_u=None, mosi_r=None):
        value = int(self.dut.ui_in.value)
        for bit_num, setting in ((0, sclk), (1, cs_n), (2, mosi_u), (3, mosi_r)):
            if setting is not None:
                value = (value & ~(1 << bit_num)) | (int(bool(setting)) << bit_num)
        self.dut.ui_in.value = value

    async def _wait_fall(self):
        # Discard an edge that occurred while the host was between samples.
        self._sclk_fall.clear()
        await self._sclk_fall.wait()
        self._sclk_fall.clear()

    async def send(self, u, r):
        """One sample: CS low, 8 bits MSB first (data changes on falling
        SCLK_IN, the chip samples on rising), then CS high for `gap` periods."""
        d = self.dut
        await self._wait_fall()
        self._set_input(cs_n=0)
        await Timer(1, "ps")
        for i in range(8):
            self._set_input(mosi_u=(u >> i) & 1, mosi_r=(r >> i) & 1)
            await self._wait_fall()
        self._set_input(cs_n=1)
        for _ in range(self.gap - 1):
            await self._wait_fall()
        self.sent += 1

    async def send_partial(self, u, r, bits):
        """An aborted frame: CS low for only `bits` (1 .. 7) bits, then high."""
        d = self.dut
        await self._wait_fall()
        self._set_input(cs_n=0)
        await Timer(1, "ps")
        for i in range(bits):
            self._set_input(mosi_u=(u >> i) & 1, mosi_r=(r >> i) & 1)
            await self._wait_fall()
        self._set_input(cs_n=1)
        for _ in range(self.gap - 1):
            await self._wait_fall()

    async def drain(self, expect, limit=2000):
        """Wait until `expect` frames have arrived (or `limit` clk cycles)."""
        for _ in range(limit):
            if len(self.frames) >= expect:
                break
            await RisingEdge(self.dut.clk)
        await ClockCycles(self.dut.clk, 40)   # let any frame in flight finish
