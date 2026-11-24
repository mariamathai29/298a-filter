"""
study_top.py - cocotb driver for filter_study.py, TOP-LEVEL / SPI-ONLY variant.

Drives the whole project (tt_um_filter_top through tb.v: SPI receiver, filter
core, SPI transmitter) using ONLY the chip pins, through DutTB (dut_tb.py):

  in : ui_in (SCLK_IN, CS_IN, MOSI_u, MOSI_r), uio_in[3:0] (adapt_en, mu_sel)
  out: uo_out (SCLK_OUT, FS_OUT, y/e/w serial lanes, ovf_flag, sat_flag)

No internal signal (dut.w, dut.x, ...) is read, so the same module also runs on
the gate-level netlist.  Per sample the chip returns y, e and ONE weight (tap
n % 4, top 8 bits, Q2.6, pre-update) - full-resolution weights are not visible.

Launched by `filter_study.py --backend top|netlist`; by hand, from test/:

    STUDY_RUNS=results/cache/_runlist.json \
        make SUITE=top COCOTB_TEST_MODULES=study_top                # RTL top level
    ... make GATES=yes SUITE=top COCOTB_TEST_MODULES=study_top      # netlist

(any non-empty SUITE other than "core" stops the Makefile from also running the
core suite first).  The run-list JSON format is the one documented in study_sim.py.
"""

import json
import os
import time

import cocotb
import numpy as np
from cocotb.triggers import ClockCycles

from dut_tb import DutTB

SEQ_GAP = 4            # SCLK periods with CS high per sample: 8 + 4 = 12 SCLK = 24 clk,
                       # the max-rate setting of test_max_rate (no dropped samples)
CLK_PER_SCLK = 2       # dut_tb: TSCLK_NS (50) / TCLK_NS (25)
FRAME_SCLK = 8         # SCLK periods CS is low per sample
MIN_GAP = 2            # smallest gap exercised by test.py (test_overrun)
RESET_CLK = 10         # clocks in reset / after reset (DutTB.reset)
CTL_SYNC_CLK = 5       # control synchronizer settle time (test.py)
PROGRESS_EVERY = 500   # steps between progress lines


# ---- small helpers (same as study_sim.py; kept local so importing this module
# ---- can never pull a second cocotb test into the run)
def load_runlist():
    text = os.environ["STUDY_RUNS"]
    if text.lstrip().startswith("{"):
        return json.loads(text)
    with open(text) as f:
        return json.load(f)


class Progress:
    # prints steps/second and an ETA for the whole run list
    def __init__(self, total):
        self.total, self.done, self.t0 = total, 0, time.time()

    def tick(self, label):
        self.done += 1
        if self.done % PROGRESS_EVERY == 0:
            rate = self.done / max(time.time() - self.t0, 1e-9)
            eta = (self.total - self.done) / max(rate, 1e-9)
            print(f"[study_top] {self.done:,}/{self.total:,} steps  "
                  f"{rate:,.0f} steps/s  ETA {eta / 60:.1f} min  ({label})", flush=True)


def save_run(cache_dir, run, data):
    # temp name + rename so an interrupted run never leaves a half file
    path = os.path.join(cache_dir, run["file"])
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, config=np.array(json.dumps(run["config"])), **data)
    os.replace(tmp, path)


# ---- DUT interaction
async def fresh_reset(tb, first):
    # First run: DutTB.reset() (starts clocks, SCLK generator, frame receiver).
    # Later runs: the same reset pulse without start(), so nothing piles up.
    if first:
        await tb.reset()
    else:
        tb.dut.rst_n.value = 0
        await ClockCycles(tb.dut.clk, RESET_CLK)
        tb.dut.rst_n.value = 1
        await ClockCycles(tb.dut.clk, RESET_CLK)      # reset synchronizer
    tb.frames.clear()
    tb.bad_frames = 0
    tb.sent = 0


async def start_run(tb, run, gap):
    tb.gap = gap
    tb.set_ctl(1, run["mu"])                          # adapt on, fixed for the run
    await ClockCycles(tb.dut.clk, CTL_SYNC_CLK)


def frame_lanes(tb):
    # received frames (y, e, w) -> three int16 arrays
    f = np.array(tb.frames, dtype=np.int16).reshape(-1, 3)
    return f[:, 0], f[:, 1], f[:, 2]


async def run_seq(tb, run, prog):
    # one SPI sample after another at 24 clk/sample; frame i answers sample i
    u, r = run["u"], run["r"]
    n = len(u)
    await start_run(tb, run, SEQ_GAP)
    for i in range(n):
        await tb.send(u[i], r[i])
        prog.tick(run["file"])
    await tb.drain(n)
    if len(tb.frames) != n or tb.bad_frames:
        raise AssertionError(f"{run['file']}: {len(tb.frames)} frames for {n} samples, "
                             f"{tb.bad_frames} cut short (expected none at 24 clk/sample)")
    y, e, w = frame_lanes(tb)
    return dict(u=np.array(u, np.int16), r=np.array(r, np.int16), y=y, e=e, w_mon=w,
                sat_flag=tb.sat(), ovf_flag=tb.ovf(), bad_frames=tb.bad_frames)


async def run_rate(tb, run, prog):
    # sample period in clocks -> CS-high gap in SCLK periods (period = 2 * (8 + gap))
    period = run["config"]["period"]
    gap = period // CLK_PER_SCLK - FRAME_SCLK
    assert period % CLK_PER_SCLK == 0 and gap >= MIN_GAP, f"unusable period {period} clk"
    u, r = run["u"], run["r"]
    await start_run(tb, run, gap)
    for i in range(len(u)):
        await tb.send(u[i], r[i])
    prog.done += len(u) - 1                           # count sent samples as steps
    prog.tick(run["file"])
    await tb.drain(len(u))                            # returns after its limit if some were dropped
    y, e, w = frame_lanes(tb)
    return dict(u=np.array(u, np.int16), r=np.array(r, np.int16),
                t=np.arange(len(u), dtype=np.int64) * period, y=y, e=e, w_mon=w,
                sat_flag=tb.sat(), ovf_flag=tb.ovf(), bad_frames=tb.bad_frames)


@cocotb.test()
async def study_top(dut):
    plan = load_runlist()
    tb = DutTB(dut)
    prog = Progress(plan["total_steps"])
    for k, run in enumerate(plan["runs"]):
        await fresh_reset(tb, k == 0)
        runner = run_rate if run["kind"] == "rate" else run_seq
        data = await runner(tb, run, prog)
        save_run(plan["cache_dir"], run, data)
        print(f"[study_top] run {k + 1}/{len(plan['runs'])} saved {run['file']}", flush=True)
