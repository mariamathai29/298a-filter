# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

import filter_golden
from dut_tb import DutTB

import math
import random

# e2e tests

# define sw model
class Golden:
    def __init__(self):
        self.f = filter_golden.AdaptiveFilter()
        self.f.adapt_en = False
        self.n = 0 #nth sample

    def set_ctl(self, adapt_en, mu_sel):
        self.f.adapt_en = bool(adapt_en)
        self.f.mu_q = 1 << mu_sel

    def step(self, u, r):
        w = self.f.w[self.n % 4] >> 6 #nth cycle carries readout weight n%4
        self.n +=1
        y, e = self.f.step(u, r)
        return y,e,w

def to_q7(v):
    return filter_golden.to_q(v, 7, 8)

async def set_ctl(tb, g, adapt_en, mu_sel):
    await tb.drain(tb.sent)
    tb.set_ctl(adapt_en, mu_sel) # set dut params
    g.set_ctl(adapt_en, mu_sel) # set model params
    await ClockCycles(tb.dut.clk, 5)

async def stream_match(tb, g, pairs):
    n0 = len(tb.frames)
    exp = [g.step(u,r) for u, r in pairs]
    for u,r in pairs:
        await tb.send(u,r)
    await tb.drain(n0+len(pairs))
    got = tb.frames[n0:] # track dut frames
    assert tb.bad_frames == 0, f"{tb.bad_frames} frames cut short"
    assert len(got) == len(exp), f"{len(got)} frames for {len(exp)} samples"
    for i, (a, b) in enumerate(zip(got, exp)):
        assert a == b, f"frame {n0 + i}: (y, e, w) {a}, expected {b}"
    return got 

def random_pairs(rng, n, lo=-128, hi=127):
    return [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n)]


@cocotb.test()
async def test_reset_pins(dut): # test reset clears all outputs
    tb = DutTB(dut)
    await tb.reset()
    for _ in range(200):
        await ClockCycles(dut.clk, 1)
        assert (int(dut.uo_out.value) & 0xFD) == 0, f"uo_out {dut.uo_out.value} while idle"
        assert ((int(dut.uo_out.value) >> 1) & 1) == 1, f"FS_OUT {dut.uo_out.value} while idle"
        assert dut.uio_oe.value == 0, f"uio_oe {dut.uio_oe.value}"
        assert dut.uio_out.value == 0, f"uio_out {dut.uio_out.value}"
    assert tb.frames == []

@cocotb.test()
async def test_first_samples(dut): # check each y,e,w output for 50 samples against golden model
    tb = DutTB(dut)
    await tb.reset()
    g = Golden()
    tb.set_ctl(1, 7)
    g.set_ctl(1, 7)
    await ClockCycles(dut.clk, 5)         # control synchronizers

    rng = random.Random(1)
    exp = []
    for _ in range(50):
        u, r = rng.randint(-128, 127), rng.randint(-128, 127)
        exp.append(g.step(u, r))
        await tb.send(u, r)
    await tb.drain(len(exp))

    assert tb.bad_frames == 0
    assert len(tb.frames) == len(exp), f"{len(tb.frames)} frames for {len(exp)} samples"
    for i, (got, want) in enumerate(zip(tb.frames, exp)):
        assert got == want, f"frame {i}: (y, e, w) {got}, expected {want}"
    assert tb.ovf() == 0

@cocotb.test()
async def test_partial_frames(dut): # terminating frames early -> no outputs for partial frames
    tb = DutTB(dut)
    await tb.reset()
    g = Golden()
    await set_ctl(tb, g, 1, 6)
    rng = random.Random(2)
    exp = []
    for _ in range(4):
        for bits in range(1, 8):
            await tb.send_partial(rng.randint(-128, 127), rng.randint(-128, 127), bits)
            u, r = rng.randint(-128, 127), rng.randint(-128, 127)
            exp.append(g.step(u, r))
            await tb.send(u, r)
    await tb.drain(len(exp))
    assert tb.bad_frames == 0
    assert len(tb.frames) == len(exp), f"{len(tb.frames)} frames for {len(exp)} complete samples"
    for i, (a, b) in enumerate(zip(tb.frames, exp)):
        assert a == b, f"frame {i}: (y, e, w) {a}, expected {b}"
    assert tb.ovf() == 0

@cocotb.test()
async def test_max_rate(dut): # 12 SCLK cycles, 24 clk
    tb = DutTB(dut, gap=4)
    await tb.reset()
    g = Golden()
    await set_ctl(tb, g, 1, 4)
    await stream_match(tb, g, random_pairs(random.Random(4), 400))
    assert tb.ovf() == 0

# /////////////////
# FLAG TESTS
# /////////////////

@cocotb.test()
async def test_overrun(dut): # 10 SCLK per sample, ovf flag should be set and remain 
    tb = DutTB(dut, gap=2)
    await tb.reset()
    assert tb.ovf() == 0
    rng = random.Random(5)
    for u, r in random_pairs(rng, 100):
        await tb.send(u, r)
    await tb.drain(tb.sent, limit=200)
    assert tb.bad_frames == 0
    assert 0 < len(tb.frames) < tb.sent, f"{len(tb.frames)} frames for {tb.sent} samples"
    assert tb.ovf() == 1, "debug_o1 not set by an overrun"
    await ClockCycles(dut.clk, 200)
    assert tb.ovf() == 1, "debug_o1 did not stay set"

@cocotb.test()
async def test_saturation_flag(dut): # sat not set for small/non saturated frames, set and remain for saturated
    tb = DutTB(dut)
    await tb.reset()
    g = Golden()
    rng = random.Random(7)
    await set_ctl(tb, g, 1, 2)
    await stream_match(tb, g, random_pairs(rng, 100, -20, 20))
    assert not g.f.sat_flag and tb.sat() == 0, "debug_o2 set without saturation"
    await set_ctl(tb, g, 1, 7)
    await stream_match(tb, g, random_pairs(rng, 300))
    assert g.f.sat_flag, "input did not saturate"
    assert tb.sat() == 1, "debug_o2 not set by sat"
    await set_ctl(tb, g, 1, 2)
    await stream_match(tb, g, random_pairs(rng, 50, -20, 20))
    assert tb.sat() == 1, "debug_o2 did not stay set"

# /////////////////
# PERFORMANCE TESTS
# /////////////////

@cocotb.test()
async def test_interference_cancelling(dut): 
    tb = DutTB(dut)
    await tb.reset()
    g = Golden()
    await set_ctl(tb, g, 1, 4)            # 2.12, mu_sel = 4 -> mu_raw = 2^4 -> mu_real = 2^-8
    rng = random.Random(1)
    n, path = 2000, [0.5, -0.3, 0.2, 0.1]
    v = [max(-0.99, min(0.99, rng.gauss(0, 0.3))) for _ in range(n)] # gaussian clipped at |0.99| to avoid unnecessary sat
    sig = [0.25 * math.sin(2 * math.pi * 0.013 * k) for k in range(n)] # sinusoidal signal, desired
    pairs = [(to_q7(v[k]), to_q7(sig[k] + sum(path[j] * v[k - j] for j in range(4) if k >= j)))
             for k in range(n)]
    frames = await stream_match(tb, g, pairs)
    half = range(n // 2, n) # evaluate latter 1000 samples
    p_r = sum((pairs[k][1] - 128 * sig[k]) ** 2 for k in half) # interference : interfered signal - desired
    p_e = sum((frames[k][1] - 128 * sig[k]) ** 2 for k in half) # intereference AFTER cancellation
    reduction = 10 * math.log10(p_r / p_e)
    dut._log.info(f"interference reduction (second half): {reduction:.1f} dB")
    assert reduction >= 10, f"interference reduced by only {reduction:.1f} dB" # at least 10 db reduction


@cocotb.test()
async def test_system_identification(dut):
    tb = DutTB(dut)
    await tb.reset()
    g = Golden()
    rng = random.Random(1)
    h = [0.4375, -0.28125, 0.15625, 0.0625] # plant weights
    n1, n2 = 1000, 500                    # LR scheduling for n samples mu = 2^-7, 2^-10
    v = [max(-0.99, min(0.99, rng.gauss(0, 0.35))) for _ in range(n1 + n2)]
    pairs = []
    for k in range(n1 + n2):
        d = sum(h[j] * v[k - j] for j in range(4) if k >= j) + rng.gauss(0, 0.004)
        pairs.append((to_q7(v[k]), to_q7(max(-0.99, min(0.99, d)))))
    await set_ctl(tb, g, 1, 5) # 2^-7
    f1 = await stream_match(tb, g, pairs[:n1])
    await set_ctl(tb, g, 1, 2) # 2^-12
    f2 = await stream_match(tb, g, pairs[n1:])
    frames = f1 + f2

    w = {n % 4: frames[n][2] / 64 for n in range(len(frames) - 4, len(frames))}
    w = [w[k] for k in range(4)]
    err = max(abs(a - b) for a, b in zip(w, h)) # compare learned weights against plant
    last = range(len(frames) - 500, len(frames))
    p_r = sum(pairs[k][1] ** 2 for k in last) # check residual error of adapted filter vs plant  
    p_e = sum(frames[k][1] ** 2 for k in last)
    reduction = 10 * math.log10(p_r / max(p_e, 1))
    dut._log.info(f"weights {[f'{x:+.3f}' for x in w]} vs h {h}: max error {err:.3f}; "
                  f"error {reduction:.1f} dB below r")
    assert err <= 3 / 64, f"weights {w} not within 3 LSB of {h}"
    assert reduction >= 20, f"error only {reduction:.1f} dB below r" # error should be max 1% of system output power
