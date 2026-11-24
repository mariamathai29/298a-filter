#!/usr/bin/env python3
"""
filter_study.py - characterize the Tiny Tapeout FIR filter (filter_core) and plot
its sign-sign LMS adaptation.  Run from test/:

    python filter_study.py --backend golden --quick     # no simulator needed
    python filter_study.py --quick                      # RTL via cocotb + Icarus
    python filter_study.py                              # full RTL run

    [--backend rtl|golden|top|netlist] [--quick] [--only sysid,signals,anc,tracking,sat]
    [--replot] [--outdir results] [--no-crosscheck]

Backends: rtl     = filter_core.v alone (study_sim.py), full data;
          golden  = Python model, no simulator;
          top     = whole project (tt_um_filter_top + tb.v) through the SPI pins only
                    (study_top.py), RTL sources;
          netlist = same as top but on the gate-level netlist (GATES=yes; needs
                    test/gate_level_netlist.v and PDK_ROOT).
top/netlist use a lighter profile and see weights only through the SPI monitor lane.

Flow: build run configs -> generate inputs (software, fixed seeds) -> simulate
(RTL through study_sim.py / golden model in-process) -> cache one .npz per run in
<outdir>/cache/ -> compute metrics -> write PNGs.  --replot skips simulation and
plots whatever is cached for the chosen --backend/--quick profile.

Needs numpy + matplotlib (not in requirements.txt).  RTL mode also needs make,
cocotb and Icarus on PATH (same as the existing core suite).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

import filter_golden                                 # noqa: E402

filter_golden.N_TAPS = 4                             # as test_filter_core.py does

# ============================ constants ============================
HERE = Path(__file__).resolve().parent
Q_SIG, Q_W, Q_MON = 128.0, 4096.0, 64.0   # Q1.7 signals, Q2.12 weights, Q2.6 monitor
STEP_CLK = 23                             # clocks per step (test_filter_core.STEP_CLK)
SEED_BASE = 20260                         # recorded in every config
H1 = (0.50, -0.30, 0.20, -0.10)           # plant
H2 = (-0.30, 0.50, -0.10, 0.20)           # plant after the tracking switch
U_RMS = 0.25                              # input RMS, FS = 1.0
AR_A = 0.9                                # AR(1) coefficient
SINE_F = 0.11                             # cycles/sample
MULTI_F = (0.05, 0.13, 0.31)
CHIRP_F = (0.01, 0.45)
SIGNALS = ("white", "ar1", "sine", "multitone", "chirp")
SIG_LABEL = {"white": "white Gaussian", "ar1": "AR(1) a=0.9", "sine": "sine 0.11",
             "multitone": "3-tone", "chirp": "chirp 0.01-0.45"}
ANC = dict(f=0.07, rms=0.10)              # interference-cancel sine "s"
MU_SIG = 4                                # mu_sel used for the signal-type study
SAT_GAIN = 2.0                            # sat sweep: plant x2 so rails are reachable (|h|max 1.0 < 2.0 FS weight range)
SMOOTH_WIN = 64                           # trailing moving-average window for e^2
SS_FRAC = 0.20                            # steady state = last 20 % of samples
CONV_MARGIN_DB = 3.0                      # convergence: stays within SS + 3 dB
MIN_DROP_DB = 10.0                        # ...and SS must be >= 10 dB below the initial error peak
ANC_TAIL = 0.25                           # SNR improvement uses the last 25 %
HEADROOM = 0.9                            # |y|,|r| must stay below 0.9 FS
EPS = 1e-12
FLOOR_DB = 10 * np.log10((1 / Q_SIG) ** 2 / 12)   # 8-bit LSB^2/12 in dBFS
RATE_PERIODS = (10, 12, 15, 18, 20, 22, 23, 24, 26, 30, 35, 40)   # clocks/sample
DPI = 160
HDL_BACKENDS = ("rtl", "top", "netlist")          # simulator-driven backends
PIN_LEVEL = ("top", "netlist")                    # SPI-only: no internal signals visible
SEQ_PERIOD_CLK = 24                               # top/netlist: clocks per sample (SPI gap 4)
W_MON_FLOOR_DB = 10 * np.log10(4 * (1 / 64) ** 2 / 12)   # 4 taps, Q2.6 rounding noise
COARSE = False                                    # set in main(): weights come from the monitor lane
NOTE = ""                                         # footnote stamped on every figure
MU_COL = plt.get_cmap("viridis")(np.linspace(0, 1, 8))             # one colour per mu_sel
EXPERIMENTS = ("sysid", "signals", "anc", "tracking", "sat")

PROFILES = {
    "full": dict(mus=tuple(range(8)), n_sys=8000, seeds_sys=3, n_sig=6000, seeds_sig=2,
                 n_anc=6000, seeds_anc=2, n_trk=8000, seeds_trk=2, n_sat=1500,
                 sat_rms=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7), n_rate=400,
                 rate_periods=RATE_PERIODS),
    "quick": dict(mus=tuple(range(2, 8)), n_sys=1500, seeds_sys=1, n_sig=1500, seeds_sig=1,
                  n_anc=1500, seeds_anc=1, n_trk=1500, seeds_trk=1, n_sat=500,
                  sat_rms=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7), n_rate=100, rate_periods=RATE_PERIODS),
    # SPI-only profiles (top / netlist): ~34k / ~9k steps at 24 clk, only a few mu_sel.
    # Sample periods must be even and >= 20 clk (CS-high gap of whole SCLK periods).
    "top": dict(mus=(2, 4, 6), n_sys=3000, seeds_sys=1, n_sig=1500, seeds_sig=1,
                n_anc=2000, seeds_anc=1, n_trk=3000, seeds_trk=1, n_sat=300,
                sat_rms=(0.1, 0.2, 0.3, 0.5, 0.7), n_rate=80, rate_periods=(20, 22, 24, 26, 30, 36, 40)),
    "top_quick": dict(mus=(3, 6), n_sys=800, seeds_sys=1, n_sig=800, seeds_sig=1,
                      n_anc=800, seeds_anc=1, n_trk=800, seeds_trk=1, n_sat=200,
                      sat_rms=(0.1, 0.3, 0.7), n_rate=40, rate_periods=(20, 22, 24, 30)),
}


# ============================ run configs ============================
def mk(scen, n, seed, mu, sig="white", **kw):
    # one run configuration; everything that shapes the inputs is in here (and hashed)
    cfg = dict(kind="seq", scen=scen, sig=sig, n=n, seed=seed, mu=mu, rms=U_RMS,
               h=list(H1), h2=None, switch=None, snr_db=None, anc=None,
               seed_base=SEED_BASE)
    cfg.update(kw)
    return cfg


def plan_configs(P, only):
    # all run configs for the selected experiments, de-duplicated by key
    c = []
    if "sysid" in only:
        c += [mk("sysid", P["n_sys"], s, m) for m in P["mus"] for s in range(P["seeds_sys"])]
    if "signals" in only:
        c += [mk("sysid", P["n_sig"], s, MU_SIG, sig=g)
              for g in SIGNALS for s in range(P["seeds_sig"])]
    if "anc" in only:
        c += [mk("anc", P["n_anc"], s, m, anc=dict(ANC))
              for m in P["mus"] for s in range(P["seeds_anc"])]
    if "tracking" in only:
        c += [mk("tracking", P["n_trk"], s, m, h2=list(H2), switch=P["n_trk"] // 2)
              for m in P["mus"] for s in range(P["seeds_trk"])]
    if "sat" in only:
        c += [mk("sat", P["n_sat"], 0, MU_SIG, rms=float(a), h=[SAT_GAIN * x for x in H1])
              for a in P["sat_rms"]]
        c += [mk("rate", P["n_rate"], 0, MU_SIG, kind="rate", period=p) for p in P["rate_periods"]]
    uniq = {run_key(x): x for x in c}
    return list(uniq.values())


def run_key(cfg):
    # readable prefix + hash of the full config
    tag = f"{cfg['scen']}_{cfg['sig']}_mu{cfg['mu']}_s{cfg['seed']}_n{cfg['n']}"
    if cfg["kind"] == "rate":
        tag += f"_p{cfg['period']}"
    if cfg["scen"] == "sat":
        tag += f"_rms{cfg['rms']}"
    return tag + "_" + hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def cache_path(cache, backend, cfg):
    return cache / f"{backend}_{run_key(cfg)}.npz"


# ============================ signals and plant ============================
def quant(x):
    # round + clip to Q1.7 integers
    return np.clip(np.rint(np.asarray(x) * Q_SIG), -128, 127).astype(np.int64)


def fir(x, h):
    # plant: y[n] = sum h_k x[n-k], zero initial state (like the DUT taps)
    return np.convolve(x, h)[:len(x)]


def ar1(n, rms, rng, burn=500):
    # AR(1) with unit-pole-free variance: var = sigma_w^2 / (1 - a^2) = rms^2
    w = rng.standard_normal(n + burn) * rms * np.sqrt(1 - AR_A ** 2)
    x = np.zeros(n + burn)
    for i in range(1, n + burn):
        x[i] = AR_A * x[i - 1] + w[i]
    return x[burn:]


def gen_signal(sig, n, rms, seed):
    # real-valued input of the given RMS (FS = 1)
    rng = np.random.default_rng([SEED_BASE, seed, 0])
    k = np.arange(n)
    ph = rng.uniform(0, 2 * np.pi, 3)
    if sig == "white":
        return rng.standard_normal(n) * rms
    if sig == "ar1":
        return ar1(n, rms, rng)
    if sig == "sine":
        return rms * np.sqrt(2) * np.sin(2 * np.pi * SINE_F * k + ph[0])
    if sig == "multitone":
        amp = rms * np.sqrt(2 / len(MULTI_F))     # equal tones, total RMS = rms
        return sum(amp * np.sin(2 * np.pi * f * k + p) for f, p in zip(MULTI_F, ph))
    if sig == "chirp":
        f0, f1 = CHIRP_F
        return rms * np.sqrt(2) * np.sin(2 * np.pi * (f0 * k + (f1 - f0) * k ** 2 / (2 * n)))
    raise ValueError(sig)


def anc_signal(n, seed, f, rms):
    # interference-cancel "s": sine with random phase
    phi = np.random.default_rng([SEED_BASE, seed, 2]).uniform(0, 2 * np.pi)
    return rms * np.sqrt(2) * np.sin(2 * np.pi * f * np.arange(n) + phi)


def make_inputs(cfg):
    # (u, r, s) as Q1.7 ints (s is the real-valued sine or None); deterministic
    n = cfg["n"]
    u = quant(gen_signal(cfg["sig"], n, cfg["rms"], cfg["seed"]))
    uq = u / Q_SIG
    r = fir(uq, cfg["h"])
    if cfg["switch"]:
        r[cfg["switch"]:] = fir(uq, cfg["h2"])[cfg["switch"]:]
    if cfg["snr_db"] is not None:
        noise = np.random.default_rng([SEED_BASE, cfg["seed"], 1]).standard_normal(n)
        r = r + noise * np.sqrt(np.mean(r ** 2) / 10 ** (cfg["snr_db"] / 10))
    s = None
    if cfg["anc"]:
        s = anc_signal(n, cfg["seed"], cfg["anc"]["f"], cfg["anc"]["rms"])
        r = r + s
    return u, quant(r), s


def rate_times(cfg):
    return np.arange(cfg["n"]) * cfg["period"]


# ============================ golden backend ============================
def new_model(mu):
    m = filter_golden.AdaptiveFilter(mu_q=1 << mu)
    m.adapt_en = True
    return m


def mon_expected(w_full):
    # monitor lane: output n shows weight n%4 as it was BEFORE step n's update, w >> 6
    prev = np.vstack([np.zeros((1, 4), w_full.dtype), w_full[:-1]])
    n = np.arange(len(w_full))
    return prev[n, n % 4] >> 6


def golden_seq(u, r, mu):
    m = new_model(mu)
    n = len(u)
    y, e, w = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros((n, 4), np.int64)
    for i in range(n):
        y[i], e[i] = m.step(int(u[i]), int(r[i]))
        w[i] = m.w
    return dict(u=u, r=r, y=y, e=e, w_full=w, w_mon=mon_expected(w),
                sat_flag=int(m.sat_flag), ovf_flag=0)


def accepted(times):
    # mirrors test_filter_core.accepted(): a sample is taken only if >= STEP_CLK
    # clocks passed since the last accepted one (otherwise dropped -> ovf_flag)
    last, out = None, []
    for t in times:
        ok = last is None or (t + 1) - last >= STEP_CLK
        out.append(ok)
        if ok:
            last = t + 1
    return out


def golden_rate(u, r, times, mu):
    m = new_model(mu)
    acc = accepted(times)
    outs = []
    for a, b, ok in zip(u, r, acc):
        if ok:
            w = m.w[len(outs) % 4] >> 6           # monitor lane: tap (output index % 4), pre-update
            outs.append(m.step(int(a), int(b)) + (w,))
    return dict(u=u, r=r, t=np.asarray(times), y=np.array([o[0] for o in outs], np.int64),
                e=np.array([o[1] for o in outs], np.int64),
                w_mon=np.array([o[2] for o in outs], np.int64), w_final=np.array(m.w, np.int64),
                sat_flag=int(m.sat_flag), ovf_flag=int(not all(acc)))


def golden_for(cfg):
    u, r, _ = make_inputs(cfg)
    if cfg["kind"] == "rate":
        return golden_rate(u, r, rate_times(cfg), cfg["mu"])
    return golden_seq(u, r, cfg["mu"])


# ============================ cache I/O ============================
def save_npz(path, cfg, data):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, config=np.array(json.dumps(cfg)), **data)
    os.replace(tmp, path)


def weights_from_monitor(w_mon):
    # Pin-level data has no w_full.  Reading m is tap m%4 (Q2.6, floor of w/64, taken
    # before step m's update); hold the latest reading per tap and add half a monitor
    # LSB (32 Q2.12 units) to remove the floor bias.  Taps not yet read stay 0.
    est, cur = np.zeros((len(w_mon), 4), np.int32), np.zeros(4, np.int32)
    for i, v in enumerate(w_mon):
        cur[i % 4] = (int(v) << 6) + 32
        est[i] = cur
    return est


def load_run(path):
    d = np.load(path)
    run = {k: d[k] for k in d.files if k != "config"}
    run["cfg"] = json.loads(str(d["config"]))
    run["sat_flag"], run["ovf_flag"] = int(run["sat_flag"]), int(run["ovf_flag"])
    if "w_full" not in run and "w_mon" in run:        # SPI-only backend
        run["coarse"] = True
        if run["cfg"]["kind"] == "seq":
            run["w_full"] = weights_from_monitor(run["w_mon"])
    return run


def load_all(cfgs, cache, backend):
    # every cached run for these configs: {key: run}; missing ones are just absent
    data = {}
    for c in cfgs:
        p = cache_path(cache, backend, c)
        if p.exists():
            data[run_key(c)] = load_run(p)
    return data


# ============================ simulation ============================
def run_golden_all(todo, cache):
    t0, done = time.time(), 0
    for i, c in enumerate(todo, 1):
        save_npz(cache_path(cache, "golden", c), c, golden_for(c))
        done += c["n"]
        print(f"[golden] run {i}/{len(todo)}  {done:,} steps  "
              f"{done / (time.time() - t0):,.0f} steps/s", flush=True)


def run_item(c, backend):
    # JSON-able entry for study_sim.py / study_top.py
    u, r, _ = make_inputs(c)
    item = dict(file=cache_path(Path(""), backend, c).name, kind=c["kind"], mu=c["mu"],
                u=u.tolist(), r=r.tolist(), config=c)
    if c["kind"] == "rate":
        item["times"] = rate_times(c).tolist()
    return item


HDL_MAKE_ARGS = {     # make arguments per simulator backend (see the Makefile)
    "rtl": ["SUITE=core", "COCOTB_TEST_MODULES=study_sim", "COCOTB_RESULTS_FILE=results_study.xml"],
    "top": ["SUITE=top", "COCOTB_TEST_MODULES=study_top", "COCOTB_RESULTS_FILE=results_study_top.xml"],
    "netlist": ["GATES=yes", "SUITE=top", "COCOTB_TEST_MODULES=study_top",
                "COCOTB_RESULTS_FILE=results_study_netlist.xml"],
}


def preflight(backend):
    # fail early, with a useful message, if the netlist flow is not set up
    if backend == "netlist":
        if not (HERE / "gate_level_netlist.v").exists():
            sys.exit("--backend netlist needs test/gate_level_netlist.v (copied in by the GDS "
                     "workflow); it is not there. Use --backend top for the RTL top level.")
        if not os.environ.get("PDK_ROOT"):
            sys.exit("--backend netlist needs PDK_ROOT set (the Makefile loads the gf180mcu "
                     "cell models from it).")


def run_hdl(todo, cache, backend):
    # one `make` invocation per call; study_sim (core) or study_top (SPI-only)
    if not todo:
        return
    plan = cache / "_runlist.json"
    plan.write_text(json.dumps(dict(cache_dir=str(cache), total_steps=sum(c["n"] for c in todo),
                                    runs=[run_item(c, backend) for c in todo])))
    env = dict(os.environ, STUDY_RUNS=str(plan), PWD=str(HERE))   # Makefile uses $(PWD)
    cmd = ["make"] + HDL_MAKE_ARGS[backend]
    print(f"[{backend}] {' '.join(cmd)}  ({len(todo)} runs)", flush=True)
    rc = subprocess.run(cmd, cwd=HERE, env=env).returncode
    missing = [c for c in todo if not cache_path(cache, backend, c).exists()]
    if rc or missing:
        sys.exit(f"{backend} simulation failed (make exit code {rc}, {len(missing)} runs missing). "
                 f"See the simulator output above and {HDL_MAKE_ARGS[backend][-1].split('=')[1]}.")


# ============================ crosscheck ============================
def first_seed_set(cfgs):
    # keys of the first (lowest) seed of every configuration
    best = {}
    for c in cfgs:
        gid = json.dumps({k: v for k, v in c.items() if k != "seed"}, sort_keys=True)
        if gid not in best or c["seed"] < best[gid]["seed"]:
            best[gid] = c
    return {run_key(c) for c in best.values()}


def first_bad(run, gold):
    # index of the first mismatching sample (or None) between a run and the golden model.
    # Full-weight backends compare w_full; pin-level ones compare y, e and the monitor lane.
    n = min(len(run["y"]), len(gold["y"]))
    bad = (run["y"][:n] != gold["y"][:n]) | (run["e"][:n] != gold["e"][:n])
    if run.get("coarse"):
        bad |= run["w_mon"][:n] != gold["w_mon"][:n]
    elif "w_full" in run:
        bad |= (run["w_full"] != gold["w_full"]).any(axis=1)
    idx = np.nonzero(bad)[0]
    return int(idx[0]) if idx.size else None


def crosscheck(cfg, run):
    # backend vs golden on identical inputs; stop with a report on the first difference
    gold = golden_for(cfg)
    key = run_key(cfg)
    if not (np.array_equal(run["u"], gold["u"]) and np.array_equal(run["r"], gold["r"])):
        sys.exit(f"CROSSCHECK: cached inputs for {key} differ from regenerated inputs "
                 f"(stale cache?). Delete the cache or use another --outdir.")
    flags_ok = run["sat_flag"] == gold["sat_flag"] and run["ovf_flag"] == gold["ovf_flag"]
    if cfg["kind"] == "rate":
        same = len(run["y"]) == len(gold["y"])
        ok = same and run["ovf_flag"] == gold["ovf_flag"]
        if "w_final" in run:
            ok = ok and np.array_equal(run["w_final"], gold["w_final"])
        if run.get("coarse"):
            ok = ok and flags_ok
        i = first_bad(run, gold) if same else 0
        if not ok or i is not None:
            sys.exit(f"CROSSCHECK FAILED: {key}: backend received {len(run['y'])} outputs "
                     f"(sat {run['sat_flag']}, ovf {run['ovf_flag']}), golden {len(gold['y'])} "
                     f"(sat {gold['sat_flag']}, ovf {gold['ovf_flag']}); first bad output index {i}")
        return
    i = first_bad(run, gold)
    if i is None and run.get("coarse") and not flags_ok:
        sys.exit(f"CROSSCHECK FAILED: {key}: flags sat/ovf = {run['sat_flag']}/{run['ovf_flag']}, "
                 f"golden {gold['sat_flag']}/{gold['ovf_flag']}")
    if i is not None:
        got_w = run["w_mon"][i] if run.get("coarse") else run["w_full"][i].tolist()
        want_w = gold["w_mon"][i] if run.get("coarse") else gold["w_full"][i].tolist()
        sys.exit(f"CROSSCHECK FAILED: {key}: first bad sample n={i}: u={run['u'][i]} r={run['r'][i]} "
                 f"got y,e,w=({run['y'][i]},{run['e'][i]},{got_w}) "
                 f"golden ({gold['y'][i]},{gold['e'][i]},{want_w})")


def check_monitor(data):
    # w_mon[n] == w_full[n-1][n % 4] >> 6, with all-zero weights before n=0
    for key, run in data.items():
        if run["cfg"]["kind"] != "seq" or run.get("coarse"):
            continue                                  # pin-level runs have no w_full to check
        bad = np.nonzero(run["w_mon"] != mon_expected(run["w_full"]))[0]
        if bad.size:
            sys.exit(f"MONITOR CHECK FAILED: {key}: first bad n={int(bad[0])} "
                     f"(got {run['w_mon'][bad[0]]}, expected {mon_expected(run['w_full'])[bad[0]]}), "
                     f"{bad.size} bad samples")


def warn_headroom(data):
    # non-saturation experiments should keep |y|,|r| < 0.9 FS and never set sat_flag
    lim, hot, sat = HEADROOM * Q_SIG, [], []
    for key, run in data.items():
        if run["cfg"]["kind"] != "seq" or run["cfg"]["scen"] == "sat":
            continue
        if max(np.abs(run["y"]).max(), np.abs(run["r"]).max()) >= lim:
            hot.append(key)
        if run["sat_flag"]:
            sat.append(key)
    for what, keys in (("|y| or |r| >= 0.9 FS", hot), ("sat_flag set", sat)):
        if keys:
            print(f"WARNING: {what} in {len(keys)} non-saturation run(s), e.g. {keys[0]}")
    cut = [k for k, run in data.items() if int(run.get("bad_frames", 0))]
    if cut:
        print(f"WARNING: SPI frames cut short in {len(cut)} run(s), e.g. {cut[0]}")
    for key, run in data.items():       # rate runs: ovf_flag must agree with drops
        c = run["cfg"]
        if c["kind"] == "rate" and run["ovf_flag"] != int(len(run["y"]) < len(run["u"])):
            print(f"WARNING: {key}: ovf_flag={run['ovf_flag']} but dropped={len(run['u']) - len(run['y'])}")


# ============================ metrics ============================
def to_db(x):
    return 10 * np.log10(np.asarray(x) + EPS)


def smooth(x, win=SMOOTH_WIN):
    # trailing moving average (expanding window for the first win-1 samples)
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(i - win, 0)
    return (c[i] - c[lo]) / (i - lo)


def e2(run):
    return (run["e"] / Q_SIG) ** 2


def wstar(cfg):
    # true plant taps (final plant if it switches) quantized to Q2.12
    h = cfg["h2"] or cfg["h"]
    return np.clip(np.rint(np.array(h) * Q_W), -8192, 8191) / Q_W


def werr2(run):
    return ((run["w_full"] / Q_W - wstar(run["cfg"])) ** 2).sum(axis=1)


def ss_len(n):
    return max(1, int(round(SS_FRAC * n)))


def steady(x):
    return float(np.mean(x[-ss_len(len(x)):]))


def flat_tail(x):
    # the steady-state window must be stationary: its two halves agree within the margin
    t = x[-ss_len(len(x)):]
    h = len(t) // 2
    return h == 0 or abs(to_db(t[:h].mean()) - to_db(t[h:].mean())) <= CONV_MARGIN_DB


def conv_time(sm, thr, start, ss_start, flat=True):
    # first n (relative to `start`) after which sm stays <= thr; None = censored.
    # Also censored if the steady state is not credible (`flat`: window stationary and
    # well below the initial error) or the point found lies inside that window itself.
    if not flat:
        return None
    bad = np.nonzero(sm[start:] > thr)[0]
    t = 0 if bad.size == 0 else int(bad[-1]) + 1
    return None if start + t >= ss_start else t


def run_metrics(run, start=0):
    e = e2(run)
    sm, ss = smooth(e), steady(e)
    thr = ss * 10 ** (CONV_MARGIN_DB / 10)
    return dict(sm=sm, ss_e=float(to_db(ss)), ss_w=float(to_db(steady(werr2(run)))),
                conv=conv_time(sm, thr, start, len(e) - ss_len(len(e)),
                               flat_tail(e) and to_db(ss) <= to_db(sm[start:].max()) - MIN_DROP_DB))


def select(data, **kv):
    # runs whose config matches all key/values, ordered by seed
    ok = [r for r in data.values() if all(r["cfg"].get(k) == v for k, v in kv.items())]
    return sorted(ok, key=lambda r: r["cfg"]["seed"])


def mus_of(runs):
    return sorted({r["cfg"]["mu"] for r in runs})


def of_mu(runs, mu):
    return [r for r in runs if r["cfg"]["mu"] == mu]


def mean_db(runs, fn):
    # average across seeds in the linear domain first, then dB
    return to_db(np.mean([fn(r) for r in runs], axis=0))


def fresp(w, f):
    # |H(f)| of a 4-tap FIR (real weights), f in cycles/sample
    return np.abs(np.exp(-2j * np.pi * np.outer(f, np.arange(len(w)))) @ w)


def welch(x, nper):
    # one-sided Welch PSD (Hann, 50 % overlap), FS^2 per cycle/sample
    w = np.hanning(nper)
    segs = [x[i:i + nper] * w for i in range(0, len(x) - nper + 1, nper // 2)]
    p = np.mean([np.abs(np.fft.rfft(s)) ** 2 for s in segs], axis=0) / (w ** 2).sum()
    p[1:-1] *= 2
    return np.fft.rfftfreq(nper), p


def snr_improvement(run, s):
    # SNR_out - SNR_in over the last 25 %: both use the same signal power, so it
    # reduces to noise-power-in / noise-power-out, noise = (r or e) - s
    n0 = int(len(s) * (1 - ANC_TAIL))
    r, e = run["r"][n0:] / Q_SIG, run["e"][n0:] / Q_SIG
    return float(to_db(np.mean((r - s[n0:]) ** 2)) - to_db(np.mean((e - s[n0:]) ** 2)))


# ============================ plot helpers ============================
def col(mu):
    return MU_COL[mu]


def mu_label(mu):
    return f"mu_sel={mu} (2$^{{{mu - 12}}}$)"


def mu_handles(mus):
    return [Line2D([0], [0], color=col(m), lw=2, label=mu_label(m)) for m in mus]


def mu_axis(ax, mus):
    # x axis = mu_sel with the step size 2^(mu_sel-12) annotated underneath
    ax.set_xticks(mus)
    ax.set_xticklabels([f"{m}\n$2^{{{m - 12}}}$" for m in mus])
    ax.set_xlabel("mu_sel (step size 2^(mu_sel-12) in real weight units, below the tick)")


def errpts(ax, mus, vals, marker):
    # mean marker + min/max bar per mu_sel, coloured by mu_sel, joined by a grey line
    xs, ms = [], []
    for mu in mus:
        v = vals.get(mu)
        if not v:
            continue
        m = float(np.mean(v))
        ax.errorbar(mu, m, yerr=[[m - min(v)], [max(v) - m]], fmt=marker, color=col(mu),
                    capsize=3, ms=7)
        xs.append(mu)
        ms.append(m)
    ax.plot(xs, ms, "-", color="0.65", lw=1, zorder=0)


def conv_panel(ax, mus, metrics, cap, ylabel):
    # convergence time (log y); censored seeds = hollow triangle at the cap, no number
    vals, cens = {}, False
    for mu in mus:
        ts = [m["conv"] for m in metrics[mu]]
        vals[mu] = [max(t, 1) for t in ts if t is not None]
        if any(t is None for t in ts):
            cens = True
            ax.plot(mu, cap, "^", mfc="none", color=col(mu), ms=10)
    errpts(ax, mus, vals, "o")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both")
    if cens:
        ax.text(0.02, 0.03, f"hollow ^ (at {cap}) = not converged in {cap} samples (censored)",
                transform=ax.transAxes, va="bottom", fontsize=8)
    mu_axis(ax, mus)


def seeds_of(runs):
    return len({r["cfg"]["seed"] for r in runs})


def weight_floor(ax):
    # pin-level backends: weight error cannot go below the Q2.6 monitor-lane rounding noise
    if COARSE:
        ax.axhline(W_MON_FLOOR_DB, color="r", ls=":", lw=1)


def save_fig(fig, out, name):
    fig.tight_layout(rect=(0, 0.03 if NOTE else 0, 1, 1))
    if NOTE:
        fig.text(0.995, 0.004, NOTE, ha="right", va="bottom", fontsize=6.5, color="0.3")
    fig.savefig(out / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out / name}")


def clip_db(ax):
    # keep e^2 plots readable: a signal that starts at exactly 0 would plot at -120 dB
    ax.set_ylim(bottom=max(ax.get_ylim()[0], FLOOR_DB - 20))


def skipped(name):
    print(f"skip {name}: no cached data for it")


# ============================ figures ============================
def fig_learning(data, P, out):
    runs = select(data, scen="sysid", sig="white", n=P["n_sys"])
    if not runs:
        return skipped("01_learning_curves")
    mus = mus_of(runs)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for mu in mus:
        rs = of_mu(runs, mu)
        a1.plot(mean_db(rs, lambda r: smooth(e2(r))), color=col(mu), lw=1.2)
        a2.plot(mean_db(rs, werr2), color=col(mu), lw=1.0)
    a1.axhline(FLOOR_DB, color="k", ls="--", lw=1)
    weight_floor(a2)
    a1.legend(handles=mu_handles(mus) + [Line2D([0], [0], color="k", ls="--", label="8-bit LSB$^2$/12 floor")],
              ncol=3, fontsize=8)
    a1.set_ylabel("smoothed e$^2$ (dBFS, 64-sample trailing mean)")
    clip_db(a1)
    a2.set_ylabel("weight error $\\|w-w^*\\|^2$ (dB)")
    a2.set_xlabel("sample n")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.suptitle(f"System ID, white noise RMS {U_RMS} FS, N={P['n_sys']}, seeds={seeds_of(runs)}", fontsize=10)
    save_fig(fig, out, "01_learning_curves.png")


def fig_weights(data, P, out):
    runs = select(data, scen="sysid", sig="white", n=P["n_sys"], seed=0)
    mus = [m for m in (2, 4, 6) if of_mu(runs, m)]
    if not mus:
        return skipped("02_weight_convergence")
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for k, ax in enumerate(axs.flat):
        ax.axhline(H1[k], color="k", ls="--", lw=1)
        for mu in mus:
            run = of_mu(runs, mu)[0]
            n = len(run["u"])
            idx = np.arange(k, n, 4)            # monitor lane shows tap k on these samples
            ax.plot(run["w_full"][:, k] / Q_W, color=col(mu), lw=1)
            ax.plot(idx, run["w_mon"][idx] / Q_MON, ".", ms=2.5, color=col(mu), alpha=0.6)
        ax.set_title(f"tap w[{k}]  (true {H1[k]:+.2f})", fontsize=9)
        ax.set_ylabel("weight (real units)")
        ax.grid(alpha=0.3)
    for ax in axs[1]:
        ax.set_xlabel("sample n")
    extra = [Line2D([0], [0], color="k", ls="--", label="true weight"),
             Line2D([0], [0], color="0.4", marker=".", ls="none", label="monitor lane (Q2.6, pre-update)")]
    axs[0, 0].legend(handles=mu_handles(mus) + extra, fontsize=7, loc="best")
    kind = "held monitor-lane readings" if COARSE else "full-resolution weights"
    fig.suptitle(f"Weight convergence, white noise, N={P['n_sys']}, seed 0 (lines: {kind})", fontsize=10)
    save_fig(fig, out, "02_weight_convergence.png")


def fig_tradeoff(data, P, out):
    runs = select(data, scen="sysid", sig="white", n=P["n_sys"])
    if not runs:
        return skipped("03_mu_tradeoff")
    mus = mus_of(runs)
    M = {mu: [run_metrics(r) for r in of_mu(runs, mu)] for mu in mus}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    conv_panel(a, mus, M, P["n_sys"], "convergence time (samples, log)")
    errpts(b, mus, {m: [x["ss_e"] for x in M[m]] for m in mus}, "o")
    errpts(b, mus, {m: [x["ss_w"] for x in M[m]] for m in mus}, "s")
    b.axhline(FLOOR_DB, color="k", ls="--", lw=1)
    weight_floor(b)
    b.legend(handles=[Line2D([0], [0], marker="o", color="0.4", ls="none", label="steady-state e$^2$ (dBFS)"),
                      Line2D([0], [0], marker="s", color="0.4", ls="none", label="steady-state weight error (dB)"),
                      Line2D([0], [0], color="k", ls="--", label="LSB$^2$/12 floor")], fontsize=8)
    b.set_ylabel("steady state, last 20 % (dB)")
    b.grid(alpha=0.3)
    mu_axis(b, mus)
    fig.suptitle(f"Step-size trade-off: sysid, white noise, N={P['n_sys']}, seeds={seeds_of(runs)} "
                 f"(marker = mean, bars = min/max over seeds)", fontsize=10)
    save_fig(fig, out, "03_mu_tradeoff.png")


def fig_signals(data, P, out):
    sets = {g: select(data, scen="sysid", sig=g, n=P["n_sig"], mu=MU_SIG) for g in SIGNALS}
    sets = {g: rs for g, rs in sets.items() if rs}
    if not sets:
        return skipped("04_signal_types")
    colors = dict(zip(SIGNALS, plt.get_cmap("tab10")(range(5))))   # mu is fixed here, so colour = signal
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2)
    a, b, c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    f = np.linspace(0, 0.5, 512)
    vals = {g: [float(to_db(steady(werr2(r)))) for r in rs] for g, rs in sets.items()}
    base = np.floor(min(min(v) for v in vals.values()) / 10) * 10 - 10   # bars hang from a floor: higher = worse
    c.plot(f, to_db(fresp(np.array(H1), f) ** 2), "k--", lw=2, label="plant")
    for i, (g, rs) in enumerate(sets.items()):
        a.plot(mean_db(rs, lambda r: smooth(e2(r))), color=colors[g], lw=1.2, label=SIG_LABEL[g])
        m = float(np.mean(vals[g]))
        b.bar(i, m - base, bottom=base, color=colors[g])
        b.errorbar(i, m, yerr=[[m - min(vals[g])], [max(vals[g]) - m]], color="k", capsize=3)
        c.plot(f, to_db(fresp(rs[0]["w_full"][-1] / Q_W, f) ** 2), color=colors[g], lw=1.2, label=SIG_LABEL[g])
    a.axhline(FLOOR_DB, color="k", ls="--", lw=1)
    a.set_xlabel("sample n")
    a.set_ylabel("smoothed e$^2$ (dBFS)")
    clip_db(a)
    a.legend(fontsize=8, ncol=3)
    b.set_xticks(range(len(sets)))
    b.set_xticklabels([SIG_LABEL[g] for g in sets], rotation=20, fontsize=8)
    b.set_ylim(base, None)
    b.set_ylabel("final weight error $\\|w-w^*\\|^2$ (dB, last 20 %)")
    c.set_xlabel("frequency (cycles/sample)")
    c.set_ylabel("$|H(f)|$ (dB) of final weights")
    c.legend(fontsize=7)
    for ax in (a, b, c):
        ax.grid(alpha=0.3)
    fig.suptitle(f"Signal types: sysid, mu_sel={MU_SIG}, N={P['n_sig']}, seeds={seeds_of(next(iter(sets.values())))}, "
                 f"RMS {U_RMS} FS; (c) uses seed 0", fontsize=10)
    save_fig(fig, out, "04_signal_types.png")


def pick_mu(mus, pref=MU_SIG):
    return pref if pref in mus else mus[len(mus) // 2]


def fig_anc(data, P, out):
    runs = select(data, scen="anc", n=P["n_anc"])
    if not runs:
        return skipped("05_interference_cancel")
    mus = mus_of(runs)
    mu = pick_mu(mus)
    run = of_mu(runs, mu)[0]
    s = make_inputs(run["cfg"])[2]
    n = len(s)
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2)
    a, b, c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    n0, n1 = int(0.75 * n), min(int(0.75 * n) + 200, n)
    k = np.arange(n0, n1)
    a.plot(k, run["r"][n0:n1] / Q_SIG, color="0.6", lw=1, label="r = s + h*u_ref")
    a.plot(k, run["e"][n0:n1] / Q_SIG, color=col(mu), lw=1.3, label=f"e (mu_sel={mu})")
    a.plot(k, s[n0:n1], "k--", lw=1, label="s (true)")
    a.set_xlabel("sample n")
    a.set_ylabel("amplitude (FS)")
    a.legend(fontsize=8, ncol=3)
    h = n // 2
    nper = min(512, h // 4)
    lows = []
    for x, kw in ((run["r"][h:] / Q_SIG, dict(color="0.6", label="r")),
                  (run["e"][h:] / Q_SIG, dict(color=col(mu), label="e")),
                  (s[h:], dict(color="k", ls="--", label="s"))):
        f, p = welch(x, nper)
        b.plot(f, to_db(p), lw=1.2, **kw)
        if kw["label"] != "s":
            lows.append(float(to_db(p).min()))
    b.set_ylim(bottom=min(lows) - 30)       # s has no energy off its tone; don't plot the EPS floor
    b.set_xlabel("frequency (cycles/sample)")
    b.set_ylabel("PSD (dB FS$^2$ per cycle/sample)")
    b.legend(fontsize=8)
    imp = {m: [snr_improvement(r, make_inputs(r["cfg"])[2]) for r in of_mu(runs, m)] for m in mus}
    errpts(c, mus, imp, "o")
    c.axhline(0, color="k", lw=0.8)
    c.set_ylabel(f"SNR improvement (dB, last {int(ANC_TAIL * 100)} %)")
    mu_axis(c, mus)
    for ax in (a, b, c):
        ax.grid(alpha=0.3)
    fig.suptitle(f"Interference cancel: s=sine {ANC['f']} cyc/smp (RMS {ANC['rms']}), u_ref white "
                 f"sigma {U_RMS}, N={n}, seeds={seeds_of(runs)}; (a),(b) mu_sel={mu} seed 0, "
                 f"PSD over 2nd half", fontsize=10)
    save_fig(fig, out, "05_interference_cancel.png")


def fig_tracking(data, P, out):
    runs = select(data, scen="tracking", n=P["n_trk"])
    if not runs:
        return skipped("06_tracking")
    mus, sw, n = mus_of(runs), runs[0]["cfg"]["switch"], P["n_trk"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    M = {}
    for mu in mus:
        rs = of_mu(runs, mu)
        a.plot(mean_db(rs, lambda r: smooth(e2(r))), color=col(mu), lw=1.1)
        M[mu] = [run_metrics(r, start=sw) for r in rs]
    a.axvline(sw, color="k", ls=":")
    a.legend(handles=mu_handles(mus) + [Line2D([0], [0], color="k", ls=":", label="plant switch h1->h2")],
             fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    a.set_xlabel("sample n")
    a.set_ylabel("smoothed e$^2$ (dBFS, 64-sample trailing mean)")
    clip_db(a)
    a.grid(alpha=0.3)
    conv_panel(b, mus, M, n - sw, "re-convergence time after switch (samples, log)")
    fig.suptitle(f"Tracking: plant h1={list(H1)} -> h2={list(H2)} at n={sw}, N={n}, "
                 f"seeds={seeds_of(runs)}, white noise RMS {U_RMS} FS", fontsize=10)
    save_fig(fig, out, "06_tracking.png")


def at_rail(run):
    # fraction of samples where y or e sits at a rail (-128 / 127)
    hit = np.isin(run["y"], (-128, 127)) | np.isin(run["e"], (-128, 127))
    return float(np.mean(hit))


def fig_sat(data, P, out):
    sat = select(data, kind="seq", scen="sat")
    rate = select(data, kind="rate")
    if not sat and not rate:
        return skipped("07_saturation_and_rate")
    c = col(MU_SIG)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    if sat:
        sat = sorted(sat, key=lambda r: r["cfg"]["rms"])
        x = [r["cfg"]["rms"] for r in sat]
        a.plot(x, [at_rail(r) for r in sat], "-", color=c, lw=1.2)
        for r in sat:
            a.plot(r["cfg"]["rms"], at_rail(r), "o", color=c, mfc=c if r["sat_flag"] else "none", ms=7)
        first = next((r["cfg"]["rms"] for r in sat if r["sat_flag"]), None)
        if first is not None:
            a.axvline(first, color="k", ls="--", lw=1)
            a.text(first, 0.5, f" sat_flag first sets\n at RMS {first} FS", transform=a.get_xaxis_transform(), fontsize=8)
        else:
            a.text(0.03, 0.9, "sat_flag never set in this sweep", transform=a.transAxes, fontsize=8)
        a.set_xlabel("input RMS (FS)")
        a.set_ylabel("fraction of samples with y or e at a rail")
        a.set_title(f"(a) white noise, plant x{SAT_GAIN:g}, mu_sel={MU_SIG}, N={P['n_sat']} (filled = sat_flag set)", fontsize=9)
    if rate:
        rate = sorted(rate, key=lambda r: r["cfg"]["period"])
        p = np.array([r["cfg"]["period"] for r in rate])
        drop = [1 - len(r["y"]) / len(r["u"]) for r in rate]
        b.plot(p, drop, "-", color=c, lw=1.2)
        for r, d in zip(rate, drop):
            b.plot(r["cfg"]["period"], d, "o", color=c, mfc=c if r["ovf_flag"] else "none", ms=7)
        pf = np.linspace(p.min(), p.max(), 400)
        b.plot(pf, 1 - 1 / np.ceil(STEP_CLK / pf), "k:", lw=1, label="ideal 1-1/ceil(23/T)")
        b.axvline(STEP_CLK, color="0.5", ls="--", lw=1, label="step = 23 clk")
        b.legend(fontsize=8)
        b.set_xlabel("sample period (clocks)")
        b.set_ylabel("dropped-sample fraction")
        b.set_title(f"(b) {len(rate[0]['u'])} samples/run, mu_sel={MU_SIG} (filled = ovf_flag set)", fontsize=9)
    for ax in (a, b):
        ax.grid(alpha=0.3)
    fig.suptitle("Saturation and sample-rate limits (single seed)", fontsize=10)
    save_fig(fig, out, "07_saturation_and_rate.png")


FIGS = {"sysid": (fig_learning, fig_weights, fig_tradeoff), "signals": (fig_signals,),
        "anc": (fig_anc,), "tracking": (fig_tracking,), "sat": (fig_sat,)}


# ============================ main ============================
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=("rtl", "golden", "top", "netlist"), default="rtl",
                    help="rtl: filter_core; golden: Python model; top: whole project via SPI pins; "
                         "netlist: top on the gate-level netlist")
    ap.add_argument("--quick", action="store_true",
                    help="N~1500, 1 seed, mu_sel 2-7 (top/netlist: even smaller SPI profile)")
    ap.add_argument("--only", default=",".join(EXPERIMENTS), help="comma list of: " + ",".join(EXPERIMENTS))
    ap.add_argument("--replot", action="store_true", help="plot from cache, no simulation")
    ap.add_argument("--outdir", default="results", help="relative to test/ unless absolute")
    ap.add_argument("--no-crosscheck", action="store_true", help="skip RTL-vs-golden check")
    return ap.parse_args()


def simulate(cfgs, args, cache):
    # run what is not cached; simulators: crosscheck set first so a mismatch stops early
    todo = [c for c in cfgs if not cache_path(cache, args.backend, c).exists()]
    total, tsteps = sum(c["n"] for c in cfgs), sum(c["n"] for c in todo)
    clk = SEQ_PERIOD_CLK if args.backend in PIN_LEVEL else STEP_CLK
    print(f"[plan] {len(cfgs)} runs, {total:,} steps (~{total * clk / 1e6:.1f}M clocks); "
          f"{len(cfgs) - len(todo)} cached, {tsteps:,} steps to simulate")
    if args.backend == "golden":
        return run_golden_all(todo, cache)
    preflight(args.backend)
    if args.no_crosscheck:
        return run_hdl(todo, cache, args.backend)
    keys = first_seed_set(cfgs)
    first = [c for c in cfgs if run_key(c) in keys]
    run_hdl([c for c in todo if run_key(c) in keys], cache, args.backend)
    for c in first:
        crosscheck(c, load_run(cache_path(cache, args.backend, c)))
    print(f"[crosscheck] {len(first)} configurations match the golden model sample-by-sample")
    run_hdl([c for c in todo if run_key(c) not in keys], cache, args.backend)


def profile_name(args):
    base = "top" if args.backend in PIN_LEVEL else ""
    return (base + "_quick" if base else "quick") if args.quick else (base or "full")


def main():
    args = parse_args()
    only = [x for x in args.only.split(",") if x]
    bad = [x for x in only if x not in EXPERIMENTS]
    if bad:
        sys.exit(f"unknown --only entries {bad}; choose from {EXPERIMENTS}")
    global COARSE, NOTE
    pname = profile_name(args)
    P = PROFILES[pname]
    COARSE = args.backend in PIN_LEVEL
    if COARSE:
        NOTE = (f"{args.backend}: SPI pins only; weights rebuilt from the 8-bit monitor lane "
                f"(Q2.6, 1 tap/sample); red dotted = its rounding floor ({W_MON_FLOOR_DB:.0f} dB)")
    outdir = Path(args.outdir)
    outdir = outdir if outdir.is_absolute() else HERE / outdir
    cache = outdir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cfgs = plan_configs(P, only)
    print(f"[plan] backend={args.backend} profile={pname} "
          f"experiments={','.join(only)} outdir={outdir}")
    if not args.replot:
        simulate(cfgs, args, cache)
    data = load_all(cfgs, cache, args.backend)
    check_monitor(data)
    warn_headroom(data)
    for name in only:
        for fn in FIGS[name]:
            fn(data, P, outdir)


if __name__ == "__main__":
    main()