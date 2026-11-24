"""
study_sim.py - cocotb driver for filter_study.py (no analysis, no plotting).

Run through the Makefile's core-only configuration, normally launched by
filter_study.py itself; by hand, from test/:

    STUDY_RUNS=results/cache/_runlist.json \
        make SUITE=core COCOTB_TEST_MODULES=study_sim

STUDY_RUNS holds the run list: either a path to a JSON file (preferred; Windows
caps an environment variable at ~32k characters) or the JSON text itself:

    {"cache_dir": "...", "total_steps": N,
     "runs": [{"file": "rtl_xxx.npz", "kind": "seq" | "rate", "mu": 0..7,
               "u": [...], "r": [...], "times": [...] (rate only),
               "config": {...}}, ...]}

Each run is reset, controlled with tb.set_ctl(1, mu) once, fed through the DUT
and saved as one .npz in cache_dir.  Only the DUT's own outputs and flags are
recorded.
"""

import json
import os
import time

import cocotb
import numpy as np
from cocotb.triggers import FallingEdge

from core_tb import CoreTB

STEP_CLK = 23           # clocks per step (test_filter_core.STEP_CLK)
RESET_CLK = 5           # falling edges of reset between runs (CoreTB.reset default)
PROGRESS_EVERY = 500    # steps between progress lines


def load_runlist():
    # run list from the STUDY_RUNS environment variable (file path or inline JSON)
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
            print(f"[study_sim] {self.done:,}/{self.total:,} steps  "
                  f"{rate:,.0f} steps/s  ETA {eta / 60:.1f} min  ({label})", flush=True)


async def fresh_reset(tb, first):
    # First run: the normal CoreTB.reset() (starts clock + monitor).  Later runs:
    # the same reset pulse without start(), so clocks/monitors don't pile up over
    # hundreds of runs.  rst is asynchronous and clears taps, weights, flags and
    # the monitor-lane counter.
    if first:
        await tb.reset(RESET_CLK)
    else:
        tb.dut.rst.value = 1
        await tb.wait(RESET_CLK)
        tb.dut.rst.value = 0
        await FallingEdge(tb.dut.clk)
    tb.outputs.clear()


def flags(dut):
    return int(dut.sat_flag.value), int(dut.ovf_flag.value)


async def run_seq(tb, run, prog):
    # one tb.step per sample; weights read after each step's update
    u, r = run["u"], run["r"]
    n = len(u)
    y, e, w_mon = np.zeros(n, np.int16), np.zeros(n, np.int16), np.zeros(n, np.int16)
    w_full = np.zeros((n, 4), np.int32)
    tb.set_ctl(1, run["mu"])
    for i in range(n):
        y[i], e[i] = await tb.step(u[i], r[i])
        w_full[i] = tb.weights()
        w_mon[i] = tb.outputs[-1][3]
        prog.tick(run["file"])
    sat, ovf = flags(tb.dut)
    return dict(u=np.array(u, np.int16), r=np.array(r, np.int16), y=y, e=e,
                w_full=w_full, w_mon=w_mon, sat_flag=sat, ovf_flag=ovf)


async def run_rate(tb, run, prog):
    # samples on a cycle schedule, as test_filter_core.run_schedule does
    tb.set_ctl(1, run["mu"])
    samples = list(zip(run["times"], run["u"], run["r"]))
    await tb.send_schedule(samples)
    await tb.wait(STEP_CLK + 5)
    out = list(tb.outputs)
    sat, ovf = flags(tb.dut)
    prog.done += len(samples) - 1          # count sent samples as steps
    prog.tick(run["file"])
    return dict(u=np.array(run["u"], np.int16), r=np.array(run["r"], np.int16),
                t=np.array(run["times"], np.int64),
                y=np.array([o[1] for o in out], np.int16),
                e=np.array([o[2] for o in out], np.int16),
                w_final=np.array(tb.weights(), np.int32), sat_flag=sat, ovf_flag=ovf)


def save_run(cache_dir, run, data):
    # write to a temp name, then rename, so an interrupted run never leaves a half file
    path = os.path.join(cache_dir, run["file"])
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, config=np.array(json.dumps(run["config"])), **data)
    os.replace(tmp, path)


@cocotb.test()
async def study(dut):
    plan = load_runlist()
    tb = CoreTB(dut)
    prog = Progress(plan["total_steps"])
    for k, run in enumerate(plan["runs"]):
        await fresh_reset(tb, k == 0)
        runner = run_rate if run["kind"] == "rate" else run_seq
        data = await runner(tb, run, prog)
        save_run(plan["cache_dir"], run, data)
        print(f"[study_sim] run {k + 1}/{len(plan['runs'])} saved {run['file']}", flush=True)
