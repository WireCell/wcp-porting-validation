#!/usr/bin/env python3
"""doc qlmatch/30 -- a time-over-threshold (ToT) saturation estimator developed on a PDVD toy MC, judged against the
two-sided repair (doc 11, production OpDecon saturation_repair) separately in simulation and on PDVD data.

    cd pdvd/docs/qlmatch && python3 scripts/saturation_tot_study.py --all      # writes d30/*.txt, pics/saturation_tot_*.png

Stages (each caches into /home/xqian/tmp/sat_tot/):
  --shape    per-channel scintillation model fitted to the median bright-pulse shape of run 039252 (file 0) pulses:
             pulse = SPE template (x) [ff*delta + (1-ff)*(bi*exp(-t/tau_i)/tau_i + (1-bi)*exp(-t/tau_s)/tau_s)]
             (x) gauss(sigma), samples above 0.3 of peak weighted x3; plus the pulse-to-pulse spread of ff.
  --sim      the toy MC (quiet real noise windows of the same channel + A * model shape, rounded, clamped to the 14-bit
             rail) and its validation against held-out data pulses (run 039253, file 1): width-above-level vs depth and
             the doc 11 estimators' bias at matched synthetic depth.
  --develop  ToT calibration from the sim TRAINING set only: PE/R vs ToT per channel (tot_cal), and the model shape
             used to fill a rail run from its measured ToT (tot_fill).
  --bench    Verdict S: held-out sim, real 16383 rail, d = 1.1 .. 20, plus stress arms.
  --data     Verdict D: (a) synthetic clips of the doc 11 pulses (closure to doc 11 + held-out file 1 to d = 6.7);
             (c) real rails whose ganged partner sub-channel did not rail (truth = partner PE x the pair's ratio);
             (c2) both sub-channels railed: pair-ratio consistency; (d) a census of all real cathode rail runs;
             (e) the real-rail tail shape vs the ToT prediction, and the real pileup rate.
Nothing in the toolkit or in doc 11's script is modified; saturation_recovery_study.py is imported read-only.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import saturation_recovery_study as S                               # noqa: E402
from pd_mapping_audit import RAWWF, RAIL, load_cathode_streams, rail_intervals   # noqa: E402

CACHE = "/home/xqian/tmp/sat_tot"
OUTD = os.path.join(os.path.dirname(HERE), "d30")
PICS = os.path.join(os.path.dirname(HERE), "pics")
OFF = np.arange(-30, 700)                     # shape-fit window around the peak
DGRID = (1.11, 1.43, 2.0, 2.86, 4.0, 5.6, 6.7, 10.0, 14.0, 20.0)
DDATA = (1.11, 1.43, 2.0, 2.86, 4.0, 5.6, 6.7)
METHODS = ("raw", "clip", "tail", "twoside", "tmpl", "tot_cal", "tot_fill")
MERGE_GAP = 2                                 # rail runs separated by <= 2 samples are one run
FF_SPREAD = None                              # set by --shape (per-pulse fast-fraction sigma)
rng_master = np.random.default_rng(20260923)
MULTISTART = False                            # --spe v2 (doc 31): best of FIT_STARTS, tight tolerances
FIT_STARTS = ([0.3, 0.3, 8, 80, 1.0], [0.2, 0.4, 5, 100, 0.8], [0.35, 0.5, 6, 70, 0.7], [0.25, 0.2, 10, 120, 1.2],
              [0.3, 0.45, 7, 90, 0.75], [0.15, 0.3, 4, 110, 0.9])
PNG_TAG = ""                                  # "_v2" under --spe v2 (doc 31) so doc 30's figures are never overwritten


def out(fh, *a):
    s = " ".join(str(x) for x in a)
    print(s)
    fh.write(s + "\n")
    fh.flush()


# ------------------------------------------------------------------ shape model
def kernel(ff, bi, ti, ts, sg, n):
    """ff prompt; of the rest a fraction bi decays with tau_i (intermediate), 1-bi with tau_s (LAr triplet)."""
    t = np.arange(n)
    k = np.zeros(n)
    k[0] += ff
    k += (1 - ff) * bi * (np.exp(-t / ti) - np.exp(-(t + 1) / ti))
    k += (1 - ff) * (1 - bi) * (np.exp(-t / ts) - np.exp(-(t + 1) / ts))
    if sg > 0.05:
        g = np.exp(-0.5 * ((t - 4 * sg) / sg) ** 2)
        k = np.convolve(k, g / g.sum())[:n]
    return k


def model(ch, ff, bi, ti, ts, sg, n=1600):
    """Bright-pulse shape (ADC per PE) = template (x) scintillation kernel; area = template area."""
    tpl = np.zeros(n)
    tpl[:len(ch.wave)] = ch.wave
    tpl[len(ch.wave):] = ch.tshape(np.arange(len(ch.wave), n))
    return np.convolve(tpl, kernel(ff, bi, ti, ts, sg, n))[:n]


def shape_at(y):
    ip = int(np.argmax(y))
    idx = ip + OFF
    ok = (idx >= 0) & (idx < len(y))
    o = np.zeros(len(OFF))
    o[ok] = y[idx[ok]]
    return o / y[ip]


def load_pulses():
    p = os.path.join(CACHE, "pulses.npz")
    if not os.path.exists(p):
        C, SEG, PK, B, F = [], [], [], [], []
        files = sorted(glob.glob(os.path.join(RAWWF, "*_rawwf.root")))[:2]
        for k, f in enumerate(files):
            for c, seg, pk, base in S.harvest_pulses(files=[f]):
                C.append(c); SEG.append(seg.astype(np.float32)); PK.append(pk); B.append(base); F.append(k)
        np.savez_compressed(p, chan=np.array(C), seg=np.stack(SEG), pk=np.array(PK), base=np.array(B),
                            fidx=np.array(F))
    return np.load(p)


def shape_fit(fh):
    from scipy.optimize import least_squares
    chd = S.load_channels()
    z = load_pulses()
    seg, pk, b, C, F = z["seg"], z["pk"], z["base"], z["chan"], z["fidx"]
    okw = (pk + OFF.max() < seg.shape[1]) & (pk + OFF.min() >= 0)
    out(fh, f"# shape fit on run 039252 (file 0) pulses only; {len(pk)} pulses harvested (doc 11: 1727), "
            f"{int((okw & (F == 0)).sum())} file-0 pulses with the full [-30, 700) window")
    out(fh, f"{'chan':>6} {'n':>3} {'ff':>6} {'bi':>6} {'tau_i':>6} {'tau_s':>6} {'sigma':>6} {'rms':>7}")
    par = {}
    for c in sorted(chd):
        sel = np.nonzero((C == c) & (F == 0) & okw)[0]
        W = np.stack([(seg[i, pk[i] + OFF] - b[i]) / (seg[i, pk[i]] - b[i]) for i in sel])
        med = np.median(W, 0)
        wt = np.where(med > 0.3, 3.0, 1.0)
        r = None
        for p0 in (FIT_STARTS if MULTISTART else ([0.3, 0.3, 8, 80, 1.0],)):
            ri = least_squares(lambda p: wt * (shape_at(model(chd[c], *p)) - med), p0,
                               bounds=([0, 0, 2, 20, 0], [1, 1, 40, 400, 20]),
                               **(dict(xtol=1e-12, ftol=1e-12, x_scale=[0.1, 0.1, 2, 20, 0.3]) if MULTISTART else {}))
            if r is None or ri.cost < r.cost:
                r = ri
        par[c] = r.x
        rms = np.sqrt(np.mean((shape_at(model(chd[c], *r.x)) - med) ** 2))
        out(fh, f"{c:>6} {len(sel):>3} {r.x[0]:6.3f} {r.x[1]:6.3f} {r.x[2]:6.2f} {r.x[3]:6.1f} {r.x[4]:6.2f} {rms:7.4f}")
    dff = []
    for c in sorted(chd):
        ff0, bi, ti, ts, sg = par[c]
        for i in np.nonzero((C == c) & (F == 0) & okw)[0]:
            w = (seg[i, pk[i] + OFF] - b[i]) / (seg[i, pk[i]] - b[i])
            r = least_squares(lambda p: shape_at(model(chd[c], p[0], bi, ti, ts, sg)) - w, [ff0], bounds=([0], [1]),
                              **(dict(xtol=1e-12, ftol=1e-12, x_scale=[0.1]) if MULTISTART else {}))
            dff.append(r.x[0] - ff0)
    dff = np.array(dff)
    q = np.percentile(dff, [16, 84])
    out(fh, f"per-pulse prompt-fraction spread (other parameters fixed per channel), n {len(dff)}: "
            f"[16,84] = [{q[0]:+.3f}, {q[1]:+.3f}] -> sim sigma_ff {0.5 * (q[1] - q[0]):.3f}")
    np.savez(os.path.join(CACHE, "shape.npz"), chan=np.array(sorted(par)), par=np.array([par[c] for c in sorted(par)]),
             sff=0.5 * (q[1] - q[0]))


def load_shape():
    z = np.load(os.path.join(CACHE, "shape.npz"))
    return {int(c): p for c, p in zip(z["chan"], z["par"])}, float(z["sff"])


# ------------------------------------------------------------------ ToT + estimators
def rail_run(seg, rail_level, lo, hi):
    """The (merged) rail run with the most samples in [lo, hi): (i, j) or None."""
    hit = seg >= rail_level
    if not hit[lo:hi].any():
        return None
    idx = np.flatnonzero(hit)
    br = np.flatnonzero(np.diff(idx) > MERGE_GAP + 1)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br] + 1, idx[-1] + 1]
    best, bn = None, 0
    for a, e in zip(starts, ends):
        n = max(0, min(e, hi) - max(a, lo))
        if n > bn:
            best, bn = (int(a), int(e)), n
    return best


class Shape:
    """Channel model shape s (peak 1) and its width-vs-level table (for tot_fill and the sim)."""

    def __init__(self, ch, p):
        y = model(ch, *p)
        self.ip = int(np.argmax(y))
        self.adc_per_pe_peak = y[self.ip]
        self.s = y / y[self.ip]
        lam = np.logspace(-3, 0, 3000)[:-1]
        u, w = [], []
        for L in lam:
            above = np.flatnonzero(self.s >= L)
            a, e = above[0], above[-1] + 1
            # fractional crossings for a continuous width
            fa = (self.s[a] - L) / max(1e-12, self.s[a] - self.s[a - 1]) if a > 0 else 0.0
            fe = (self.s[e - 1] - L) / max(1e-12, self.s[e - 1] - self.s[e]) if e < len(self.s) else 0.0
            u.append(a)
            w.append((e - a) + fa + fe)
        self.lam, self.u, self.w = lam, np.array(u), np.array(w)

    def level_for(self, tot):
        """lambda such that width(lambda) = tot (width decreases with lambda)."""
        return float(np.interp(tot, self.w[::-1], self.lam[::-1]))


class TotCal:
    """PE/R vs ToT per channel, learned on the sim training set (monotone, log-log interpolation)."""

    def __init__(self, tabs):
        self.tabs = tabs          # chan -> (tot_nodes, pe_over_r_nodes)

    def pe(self, c, tot, R):
        """R * g_c(ToT): log-log interpolation inside the table, log-log linear extrapolation (end 4 nodes) outside."""
        x, y = np.log(self.tabs[c][0]), np.log(self.tabs[c][1])
        lt = np.log(max(tot, 0.5))
        if x[0] <= lt <= x[-1]:
            return float(R * np.exp(np.interp(lt, x, y)))
        sl = slice(-4, None) if lt > x[-1] else slice(0, 4)
        return float(R * np.exp(np.polyval(np.polyfit(x[sl], y[sl], 1), lt)))


def methods_pe(seg, chan, base, pk, rail_level, shp, cal, c):
    """doc 11's five estimators plus tot_cal and tot_fill; window [pk-50, pk+300)."""
    o = S.methods_pe(seg, chan, base, pk, rail_level=rail_level)
    lo, hi = max(0, pk - S.WIN_PRE), min(len(seg), pk + S.WIN_POST)
    rr = rail_run(seg, rail_level, lo, hi)
    if rr is None:
        o["tot_cal"] = o["tot_fill"] = o["clip"]
        return o, None
    i, j = rr
    tot = j - i
    R = rail_level + 0.5 - base                          # rail height over the baseline, ADC
    o["tot_cal"] = cal.pe(c, tot, R) if cal is not None else np.nan
    lam = shp.level_for(tot)
    A = R / lam
    k0 = shp.u[min(len(shp.u) - 1, int(np.searchsorted(shp.lam, lam)))]   # model index of the up-crossing
    w = np.array(seg, np.float64)
    tt = np.arange(i, j)
    kk = k0 + (tt - i)
    fill = A * np.where(kk < len(shp.s), shp.s[np.clip(kk, 0, len(shp.s) - 1)], 0.0)
    w[i:j] = np.maximum(w[i:j], base + fill)
    o["tot_fill"] = S.window_pe(S.decon_replica(w, chan, pedestal=base), pk)
    return o, (tot, R, lam)


# ------------------------------------------------------------------ toy MC
def quiet_windows(file_idx, per_chan=40, thr=300):
    """Quiet SEG-long windows of real cathode streams (baseline + noise + small pulses), per channel."""
    p = os.path.join(CACHE, f"quiet_{file_idx}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return {int(c): z[str(c)] for c in z["chans"]}
    import uproot
    rf = sorted(glob.glob(os.path.join(RAWWF, "*_rawwf.root")))[file_idx]
    t = uproot.open(rf)["rawdump/raw_waveform"]
    evs = sorted(set(t.arrays(["event"], library="np")["event"]))
    got = {}
    for ev in evs:
        (waves, _), _ = load_cathode_streams(rf, ev)
        for c, w in waves.items():
            b = np.median(w)
            L = got.setdefault(c, [])
            for s in range(0, len(w) - S.SEG, S.SEG // 4):
                if len(L) >= per_chan:
                    break
                x = w[s:s + S.SEG]
                if (x - b).max() < thr and (x - b).min() > -thr:
                    L.append(np.array(x, np.float32))
    chans = sorted(got)
    np.savez_compressed(p, chans=np.array(chans), **{str(c): np.stack(got[c]) for c in chans})
    return {c: np.stack(got[c]) for c in chans}


def sim_pulse(noise, shp_fn, ff, A, pos, pile=None):
    """noise window + A * shape(ff) peaking at pos (+ optional pileup (frac, dt)); returns (unclipped, clipped)."""
    s = shp_fn(ff)
    ip = int(np.argmax(s))
    s = s / s[ip]
    w = np.array(noise, np.float64)
    a0 = pos - ip
    n = min(len(s), len(w) - a0)
    w[a0:a0 + n] += A * s[:n]
    if pile is not None:
        fr, dt = pile
        b0 = a0 + dt
        n2 = min(len(s), len(w) - b0)
        w[b0:b0 + n2] += fr * A * s[:n2]
    w = np.round(w)
    return w, np.clip(w, 0, RAIL)


def make_sim(file_idx, n_per, dlist, seed, chd, par, sff, arm="nominal"):
    """Sim sample: rows (chan, d, pe_true, per-method pe, tot, R).  Real rail = 16383."""
    rng = np.random.default_rng(seed)
    noise = quiet_windows(file_idx)
    rows = []
    for c in sorted(chd):
        ff0, bi, ti, ts, sg = par[c]
        if arm == "tau_s+20":
            ts = ts * 1.2
        if arm == "tau_s-20":
            ts = ts * 0.8
        if arm == "spread5":
            sg = np.hypot(sg, 5.0)
        shp_fn = (lambda ff, bi=bi, ti=ti, ts=ts, sg=sg, ch=chd[c]: model(ch, ff, bi, ti, ts, sg))
        nw = noise[c]
        for k in range(n_per):
            nz = nw[rng.integers(len(nw))]
            base = float(np.median(nz))
            ff = np.clip(ff0 + sff * rng.standard_normal(), 0, 1)
            if arm == "ff+0.08":
                ff = min(1.0, ff + 0.08)
            if arm == "ff-0.08":
                ff = max(0.0, ff - 0.08)
            pile = (0.3, int(rng.integers(40, 200))) if arm == "pileup" else None
            d = dlist(rng) if callable(dlist) else dlist[k % len(dlist)]
            A = d * (RAIL - base)
            full, clip = sim_pulse(nz, shp_fn, ff, A, S.SEG // 2, pile)
            pk = int(np.argmax(full[S.SEG // 2 - 60:S.SEG // 2 + 60])) + S.SEG // 2 - 60
            rows.append((c, d, full, clip, pk, base))
    return rows


# ------------------------------------------------------------------ scoring
def summarize(ratios):
    r = np.asarray(ratios, float)
    r = r[np.isfinite(r)]
    if not len(r):
        return (np.nan,) * 3
    return np.median(r), np.percentile(r, 16), np.percentile(r, 84)


def score(med, lo, hi):
    return abs(med - 1) + 0.5 * (hi - lo)


def passes(med, lo, hi):
    return abs(med - 1) < 0.20 and 0.5 * (hi - lo) < 0.30


def table(fh, title, groups, keys, methods, keyfmt="{:>6.2f}"):
    """groups[(key, method)] -> list of ratios. Prints median [16,84] and the per-row winner."""
    out(fh, f"\n{title}")
    out(fh, f"{'d':>6} {'n':>5} |" + "".join(f" {m:>22}" for m in methods) + " | winner (min |bias|+half-band)")
    best_depth = {m: None for m in methods}
    rows = {}
    for k in keys:
        line = keyfmt.format(k) + f" {len(groups.get((k, methods[0]), [])):>5} |"
        sc = {}
        for m in methods:
            med, lo, hi = summarize(groups.get((k, m), []))
            rows[(k, m)] = (med, lo, hi)
            flag = "*" if np.isfinite(med) and passes(med, lo, hi) else " "
            line += f" {med:6.3f} [{lo:5.3f},{hi:6.3f}]{flag}"
            if np.isfinite(med):
                sc[m] = score(med, lo, hi)
                if passes(med, lo, hi):
                    best_depth[m] = k
        win = min(sc, key=sc.get) if sc else "-"
        out(fh, line + f" | {win}")
    out(fh, "  (* = passes |bias| < 20 % and half-band < 30 %)")
    out(fh, "  deepest passing row per method: " + ", ".join(f"{m} {best_depth[m]}" for m in methods))
    return rows


# ------------------------------------------------------------------ stages
def ctx():
    chd = S.load_channels()
    par, sff = load_shape()
    shapes = {c: Shape(chd[c], par[c]) for c in chd}
    return chd, par, sff, shapes


def load_cal():
    p = os.path.join(CACHE, "totcal.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return TotCal({int(c): (z[f"x{c}"], z[f"y{c}"]) for c in z["chans"]})


def run_rows(rows, chd, shapes, cal, synthetic_f=None):
    """Evaluate every sim row; synthetic_f: clip at base + f*(peak-base) (data-like) instead of the real rail."""
    res = []
    for c, d, full, clip, pk, base in rows:
        true = S.methods_pe(full, chd[c], base, pk)["clip"]
        if true <= 0:
            continue
        if synthetic_f is None:
            seg, rl = clip, RAIL - 0.5
        else:
            rl = base + synthetic_f * (full[pk] - base)
            seg, rl = np.minimum(full, rl), rl - 0.5
        o, tr = methods_pe(seg, chd[c], base, pk, rl, shapes[c], cal, c)
        res.append((c, d, true, o, tr))
    return res


def stage_sim(fh):
    chd, par, sff, shapes = ctx()
    out(fh, "# toy MC validation against HELD-OUT data (run 039253, file 1); sim noise from file-1 quiet windows")
    z = load_pulses()
    seg, pk, b, C, F = z["seg"], z["pk"], z["base"], z["chan"], z["fidx"]
    sel = np.nonzero(F == 1)[0]
    fr = (0.9, 0.7, 0.5, 0.35, 0.25, 0.18, 0.15)
    # data-like sim: same channels, amplitudes drawn from the data peak distribution (unrailed, synthetic clip)
    rng = np.random.default_rng(7)
    noise = quiet_windows(1)
    out(fh, f"quiet noise windows per channel (file 1): " + ", ".join(f"{c}:{len(noise[c])}" for c in sorted(noise)))
    simrows = []
    for i in sel:
        c = int(C[i])
        nz = noise[c][rng.integers(len(noise[c]))]
        base = float(np.median(nz))
        ff0, bi, ti, ts, sg = par[c]
        ff = np.clip(ff0 + sff * rng.standard_normal(), 0, 1)
        A = float(seg[i, pk[i]] - b[i])
        full, _ = sim_pulse(nz, lambda f, ch=chd[c], p=(bi, ti, ts, sg): model(ch, f, *p), ff, A, S.SEG // 2)
        p0 = int(np.argmax(full[S.SEG // 2 - 60:S.SEG // 2 + 60])) + S.SEG // 2 - 60
        simrows.append((c, None, full, None, p0, base))

    def widths(w, p, base, f):
        L = base + f * (w[p] - base)
        rr = rail_run(np.asarray(w, float), L, max(0, p - 50), p + 300)
        return (rr[1] - rr[0]) if rr else 0

    out(fh, "\n(i) width above f*peak (= the ToT a clip at depth d = 1/f would show), ticks: median [16,84]")
    out(fh, f"{'d':>6} {'data':>22} {'sim':>22} {'sim/data median':>16}")
    for f in fr:
        wd = [widths(seg[i].astype(float), pk[i], b[i], f) for i in sel]
        ws = [widths(r[2], r[4], r[5], f) for r in simrows]
        qd, qs = np.percentile(wd, [16, 50, 84]), np.percentile(ws, [16, 50, 84])
        out(fh, f"{1 / f:6.2f} {qd[1]:7.1f} [{qd[0]:5.1f},{qd[2]:6.1f}] {qs[1]:7.1f} [{qs[0]:5.1f},{qs[2]:6.1f}] "
                f"{qs[1] / max(qd[1], 1e-9):16.3f}")
    out(fh, "\n(ii) doc 11 estimators at matched synthetic depth: PE_rec/PE_true, data (file 1) vs sim")
    grp_d, grp_s = {}, {}
    for f in fr:
        for i in sel:
            c = int(C[i])
            s_ = seg[i].astype(np.float64)
            t_ = S.methods_pe(s_, chd[c], b[i], pk[i])["clip"]
            rl = b[i] + f * (s_[pk[i]] - b[i])
            o = S.methods_pe(np.minimum(s_, rl), chd[c], b[i], pk[i], rail_level=rl - 0.5)
            for m in ("raw", "clip", "twoside"):
                grp_d.setdefault((f, m), []).append(o[m] / t_)
        for c, _, full, _, p0, base in simrows:
            t_ = S.methods_pe(full, chd[c], base, p0)["clip"]
            rl = base + f * (full[p0] - base)
            o = S.methods_pe(np.minimum(full, rl), chd[c], base, p0, rail_level=rl - 0.5)
            for m in ("raw", "clip", "twoside"):
                grp_s.setdefault((f, m), []).append(o[m] / t_)
    out(fh, f"{'d':>6} |" + "".join(f" {'data ' + m:>22} {'sim ' + m:>22}" for m in ("raw", "clip", "twoside")))
    for f in fr:
        line = f"{1 / f:6.2f} |"
        for m in ("raw", "clip", "twoside"):
            a, s_ = summarize(grp_d[(f, m)]), summarize(grp_s[(f, m)])
            line += f" {a[0]:6.3f} [{a[1]:5.3f},{a[2]:6.3f}] {s_[0]:6.3f} [{s_[1]:5.3f},{s_[2]:6.3f}]"
        out(fh, line)


def stage_develop(fh):
    chd, par, sff, shapes = ctx()
    out(fh, "# ToT calibration from the sim TRAINING set (noise windows of run 039252 = file 0; seed 101); "
            "d log-uniform in [1.05, 25]")
    rows = make_sim(0, 300, lambda r: float(np.exp(r.uniform(np.log(1.05), np.log(25.0)))), 101, chd, par, sff)
    res = run_rows(rows, chd, shapes, None)
    tabs = {}
    out(fh, f"{'chan':>6} {'n':>4} {'ToT nodes (ticks)':>40}  ->  PE/R nodes")
    for c in sorted(chd):
        xs = np.array([r[4][0] for r in res if r[0] == c and r[4] is not None], float)
        ys = np.array([r[2] / r[4][1] for r in res if r[0] == c and r[4] is not None], float)
        qs = np.unique(np.percentile(xs, np.linspace(0, 100, 25)))
        nx, ny = [], []
        for a, e in zip(qs[:-1], qs[1:]):
            m = (xs >= a) & (xs <= e)
            if m.sum() >= 3:
                nx.append(np.median(xs[m])); ny.append(np.median(ys[m]))
        nx, ny = np.array(nx), np.maximum.accumulate(np.array(ny))      # monotone
        keep = np.r_[True, np.diff(nx) > 0]
        tabs[c] = (nx[keep], ny[keep])
        out(fh, f"{c:>6} {len(xs):>4} " + " ".join(f"{x:.0f}" for x in nx[keep]) + "  ->  "
                + " ".join(f"{y:.3f}" for y in ny[keep]))
    np.savez(os.path.join(CACHE, "totcal.npz"), chans=np.array(sorted(tabs)),
             **{f"x{c}": tabs[c][0] for c in tabs}, **{f"y{c}": tabs[c][1] for c in tabs})
    # fill self-check: tot_fill's model reproduces the measured ToT at its solved level
    errs = []
    for c in sorted(chd):
        sh = shapes[c]
        for tot in (3, 10, 30, 100, 250):
            lam = sh.level_for(tot)
            above = np.flatnonzero(sh.s >= lam)
            errs.append(abs((above[-1] + 1 - above[0]) - tot))
    out(fh, f"tot_fill self-check: |integer width at the solved level - ToT| max {max(errs)} tick(s) "
            f"(fractional crossings make the continuous width exact)")


def stage_bench(fh):
    chd, par, sff, shapes = ctx()
    cal = load_cal()
    out(fh, "# Verdict S: held-out sim (noise windows of run 039253 = file 1; seed 202), real rail 16383; "
            "PE_rec/PE_true median [16,84]")
    for arm in ("nominal", "ff+0.08", "ff-0.08", "tau_s+20", "tau_s-20", "spread5", "pileup"):
        rows = make_sim(1, 20 * len(DGRID) if arm == "nominal" else 8 * len(DGRID), DGRID, 202, chd, par, sff, arm)
        res = run_rows(rows, chd, shapes, cal)
        g = {}
        for c, d, true, o, tr in res:
            for m in METHODS:
                g.setdefault((d, m), []).append(o[m] / true)
        table(fh, f"== sim arm {arm}", g, DGRID, METHODS)
        if arm == "nominal":
            rows_nom = g
    plot(rows_nom, DGRID, "sim (held out, real 16383 rail)", "saturation_tot_sim.png")


def plot(g, keys, title, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    col = {"raw": "tab:gray", "clip": "tab:red", "tail": "tab:blue", "twoside": "tab:green", "tmpl": "tab:purple",
           "tot_cal": "tab:orange", "tot_fill": "tab:brown"}
    for m in METHODS:
        s = [summarize(g.get((k, m), [])) for k in keys]
        ax.plot(keys, [x[0] for x in s], "o-", color=col[m], label=m)
        ax.fill_between(keys, [x[1] for x in s], [x[2] for x in s], color=col[m], alpha=0.12)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.axhspan(0.8, 1.2, color="k", alpha=0.05)
    ax.set_xscale("log")
    ax.set_xlabel("clip depth d = true peak / rail")
    ax.set_ylabel("PE_rec / PE_true")
    ax.set_ylim(0, 2.0)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, name.replace(".png", PNG_TAG + ".png")), dpi=130)
    plt.close(fig)


def stage_data(fh):
    chd, par, sff, shapes = ctx()
    cal = load_cal()
    z = load_pulses()
    seg, pk, b, C, F = z["seg"], z["pk"], z["base"], z["chan"], z["fidx"]
    # (a) synthetic clips
    out(fh, "# Verdict D (a): synthetic clips of real unrailed pulses (doc 11's set)")
    g_all, g_1 = {}, {}
    for d in DDATA:
        f = 1.0 / d
        for i in range(len(pk)):
            c = int(C[i])
            s_ = seg[i].astype(np.float64)
            t_ = S.methods_pe(s_, chd[c], b[i], pk[i])["clip"]
            if t_ <= 0:
                continue
            rl = b[i] + f * (s_[pk[i]] - b[i])
            o, _ = methods_pe(np.minimum(s_, rl), chd[c], b[i], pk[i], rl - 0.5, shapes[c], cal, c)
            for m in METHODS:
                g_all.setdefault((d, m), []).append(o[m] / t_)
                if F[i] == 1:
                    g_1.setdefault((d, m), []).append(o[m] / t_)
    table(fh, "closure: all 1727 pulses (doc 11 used d = 1.11 .. 4.00 with f = 0.9 .. 0.25)", g_all, DDATA, METHODS)
    table(fh, "held-out file 1 (run 039253) only -- the shape model was fitted on file 0", g_1, DDATA, METHODS)
    plot(g_1, DDATA, "data: synthetic clips, run 039253 (held out)", "saturation_tot_data_synthetic.png")
    stage_partner(fh, chd, shapes, cal)
    stage_tail(fh, chd, shapes)


def seg_at(w, s):
    s0 = int(max(0, min(s - S.SEG // 2, len(w) - S.SEG)))
    x = np.array(w[s0:s0 + S.SEG], np.float64)
    return x, s0


def stage_partner(fh, chd, shapes, cal, nfiles=5):
    """(c) ganged partners: 1000+10x and 1001+10x read the same X-Arapuca module."""
    import uproot
    out(fh, "\n# Verdict D (c): real rails whose ganged partner did NOT rail; truth = partner PE x pair ratio")
    files = sorted(glob.glob(os.path.join(RAWWF, "*_rawwf.root")))[:nfiles]
    ctrl = {}                          # pair -> list of (PE_a/PE_b, peak_a/peak_b)
    tests = []                         # (pair, a, PE_b, peak_b, method dict, tot, R)
    both = []                          # (pair, (est_a, tot_a), (est_b, tot_b)) -- both sub-channels railed
    cens = {"runs": 0, "edge": 0, "long_gt300": 0}
    tots_all = []
    for rf in files:
        t = uproot.open(rf)["rawdump/raw_waveform"]
        evs = sorted(set(t.arrays(["event"], library="np")["event"]))
        for ev in evs:
            (waves, ts), t0 = load_cathode_streams(rf, ev)
            for x in range(1, 9):
                A_, B_ = 1000 + 10 * x, 1001 + 10 * x
                if A_ not in waves or B_ not in waves:
                    continue
                for a, bch in ((A_, B_), (B_, A_)):
                    wa, wb = waves[a], waves[bch]
                    sh = (ts[a] - ts[bch]) * S.FS                   # sample i of a == sample i+sh of b
                    ra = rail_intervals(wa)
                    rb_hit = wb >= RAIL
                    cens["runs"] += len(ra)
                    # census of a's runs (count each run once: only when a < bch handled both ways)
                    for (s0, s1) in ra:
                        if s0 == 0 or s1 == len(wa):
                            cens["edge"] += 1
                        if s1 - s0 > 300:
                            cens["long_gt300"] += 1
                        tots_all.append(s1 - s0)
                    # tests: rails of a, partner clean within +-400 samples
                    for (s0, s1) in ra:
                        j0 = int(round(s0 + sh))
                        if j0 - 400 < 0 or j0 + 400 >= len(wb) or rb_hit[j0 - 400:j0 + 400].any():
                            continue
                        segb, ob = seg_at(wb, j0)
                        bb = S.seg_baseline(segb)
                        pkb = int(np.argmax(segb[j0 - ob - 60:j0 - ob + 60])) + j0 - ob - 60
                        peb = S.methods_pe(segb, chd[bch], bb, pkb)["clip"]
                        sega, oa = seg_at(wa, s0)
                        ba = S.seg_baseline(sega)
                        pka = pkb + ob - oa - int(round(sh))            # partner's peak mapped into a's segment
                        pka = int(np.clip(pka, S.WIN_PRE, len(sega) - S.WIN_POST - 1))
                        o, tr = methods_pe(sega, chd[a], ba, pka, RAIL - 0.5, shapes[a], cal, a)
                        tests.append(((min(a, bch), max(a, bch)), a, peb, segb[pkb] - bb, RAIL - ba, o, tr))
                    # controls: bright unrailed pulses on a, partner unrailed too (only when a < b to count once)
                    if a > bch:
                        continue
                    # (c2) both sub-channels railed on the same pulse: each estimated on its own rail, window from
                    # its own run start (the true peak is unknown on a real rail)
                    rb = rail_intervals(wb)
                    for (s0, s1) in ra:
                        j0 = s0 + sh
                        mt = [(q0, q1) for (q0, q1) in rb if abs(q0 - j0) < 30]
                        if len(mt) != 1:
                            continue
                        q0, q1 = mt[0]
                        ests = []
                        for ww, cc, r0 in ((wa, a, s0), (wb, bch, q0)):
                            sg_, o_ = seg_at(ww, r0)
                            bs_ = S.seg_baseline(sg_)
                            pk_ = int(np.clip(r0 - o_, S.WIN_PRE, len(sg_) - S.WIN_POST - 1))
                            oo, tr_ = methods_pe(sg_, chd[cc], bs_, pk_, RAIL - 0.5, shapes[cc], cal, cc)
                            ests.append((oo, tr_[0] if tr_ else 0))
                        both.append(((a, bch), ests[0], ests[1]))
                    base_a = np.median(wa)
                    cand = np.flatnonzero((wa - base_a > 3000) & (wa < RAIL))
                    taken = []
                    for i in cand[np.argsort(wa[cand])[::-1]]:
                        if len(taken) >= 6:
                            break
                        if any(abs(i - q) < 2000 for q in taken):
                            continue
                        j = int(round(i + sh))
                        if j - 400 < 0 or j + 400 >= len(wb) or (wa[max(0, i - 400):i + 400] >= RAIL).any() \
                                or rb_hit[j - 400:j + 400].any():
                            continue
                        taken.append(i)
                        sa_, oa = seg_at(wa, i)
                        sb_, ob = seg_at(wb, j)
                        ba_, bb_ = S.seg_baseline(sa_), S.seg_baseline(sb_)
                        pa = int(np.argmax(sa_[i - oa - 30:i - oa + 30])) + i - oa - 30
                        pb = int(np.argmax(sb_[j - ob - 30:j - ob + 30])) + j - ob - 30
                        ea = S.methods_pe(sa_, chd[a], ba_, pa)["clip"]
                        eb = S.methods_pe(sb_, chd[bch], bb_, pb)["clip"]
                        if ea > 0 and eb > 0:
                            ctrl.setdefault((a, bch), []).append((ea / eb, (sa_[pa] - ba_) / (sb_[pb] - bb_)))
    out(fh, f"files {len(files)}; control (both unrailed, bright) pulses per pair: "
            + ", ".join(f"{p[0]}/{p[1]}:{len(v)}" for p, v in sorted(ctrl.items())))
    out(fh, f"{'pair':>11} {'n':>4} {'PE_a/PE_b median [16,84]':>28} {'peak ratio':>11}")
    ratio = {}
    for p, v in sorted(ctrl.items()):
        r = np.array(v)
        q = np.percentile(r[:, 0], [16, 50, 84])
        ratio[p] = (q[1], float(np.median(r[:, 1])), 0.5 * (q[2] - q[0]) / q[1])
        out(fh, f"{p[0]:>5}/{p[1]:<5} {len(r):>4} {q[1]:10.3f} [{q[0]:.3f},{q[2]:.3f}] (+-{100 * ratio[p][2]:.1f}%) "
                f"{ratio[p][1]:11.3f}")
    # tests binned by truth-implied depth d = peak ratio * partner peak / R
    g = {}
    dl = []
    edges = (1.0, 1.25, 1.6, 2.2, 3.2, 5.0, 8.0, 100.0)
    keys = []
    for pair, a, peb, pkh_b, R, o, tr in tests:
        if pair not in ratio or peb <= 0:
            continue
        rpe, rpk, _ = ratio[pair]
        pe_a = rpe * peb if a == pair[0] else peb / rpe
        pk_a = (rpk * pkh_b if a == pair[0] else pkh_b / rpk)
        d = pk_a / R
        dl.append(d)
        k = int(np.searchsorted(edges, d)) - 1
        if k < 0 or k >= len(edges) - 1:
            continue
        key = 0.5 * (edges[k] + edges[k + 1]) if k < len(edges) - 2 else 10.0
        for m in METHODS:
            g.setdefault((key, m), []).append(o[m] / pe_a)
        keys.append(key)
    keys = sorted(set(keys))
    dl = np.array(dl)
    out(fh, f"\nrail runs with a clean partner: {len(dl)}; truth-implied d quantiles 16/50/84/95: "
            + " ".join(f"{q:.2f}" for q in np.percentile(dl, [16, 50, 84, 95])))
    out(fh, f"d bins (edges {edges}); a row key is its bin centre (last bin shown as 10)")
    table(fh, "== real rails vs ganged-partner truth: PE_rec / (ratio x PE_partner)", g, keys, METHODS)
    out(fh, "\n# Verdict D (c2): both sub-channels railed; q = (PE_a/PE_b)_rec / (PE_a/PE_b)_controls -- a method whose "
            "bias grows with depth moves q away from 1 when the two channels sit at different depths")
    out(fh, "  pairs used: peak ratio outside [0.87, 1.15] (the two channels at clearly different depths), 1050/1051 excluded "
            "(1051's template is anomalous: PE ratio 6.5, shape-fit rms 7 %)")
    out(fh, "  bins: ToT of the more-saturated channel (measured, estimator-free); data widths at level (sim stage): "
            "71 ~ d 2, 108 ~ 2.9, 141 ~ 4, 175 ~ 5.6, 194 ~ 6.7 ticks")
    tedges = (0, 40, 71, 108, 141, 194, 260, 2000)
    g2, keys2 = {}, set()
    for pair, (ea, ta), (eb, tb) in both:
        if pair not in ratio or pair == (1050, 1051):
            continue
        rpe, rpk, _ = ratio[pair]
        if 0.87 <= rpk <= 1.15:
            continue
        tmax = max(ta, tb)
        k = int(np.searchsorted(tedges, tmax, side="right")) - 1
        key = tedges[k]
        keys2.add(key)
        for m in METHODS:
            if ea[m] > 0 and eb[m] > 0:
                g2.setdefault((key, m), []).append((ea[m] / eb[m]) / rpe)
    table(fh, "== both-railed pair consistency q (row key = lower ToT-bin edge of the more-saturated channel, ticks)",
          g2, sorted(keys2), METHODS, keyfmt="{:>6.0f}")
    tots_all = np.array(tots_all)
    out(fh, f"\n# (d) census of real cathode rail runs over {len(files)} files (each run seen once per channel): "
            f"{len(tots_all)} runs; at a stream edge {cens['edge']}; longer than 300 ticks {cens['long_gt300']}")
    out(fh, "  run-length (ToT) quantiles 16/50/84/95/99: "
            + " ".join(f"{q:.0f}" for q in np.percentile(tots_all, [16, 50, 84, 95, 99])))
    for tt, dd in ((71, 2.0), (141, 4.0), (194, 6.7), (245, 10.0), (329, 20.0)):          # model widths at 1/d
        out(fh, f"  runs longer than {tt} ticks (~ d > {dd}): {100 * np.mean(tots_all > tt):.1f} %")


TAIL = 150                 # post-run window for the tail-consistency check, ticks
TEDGES = (0, 40, 71, 108, 141, 194, 260, 2000)


def tail_ratio(seg, i, j, base, R, shp):
    """measured / tot_fill-predicted integral over [j, j+TAIL) after a rail run [i, j); None if unusable."""
    if j + TAIL > len(seg):
        return None
    lam = shp.level_for(j - i)
    A = R / lam
    k0 = shp.u[min(len(shp.u) - 1, int(np.searchsorted(shp.lam, lam)))]
    kk = k0 + (j - i) + np.arange(TAIL)
    pred = A * np.where(kk < len(shp.s), shp.s[np.clip(kk, 0, len(shp.s) - 1)], 0.0)
    meas = seg[j:j + TAIL] - base
    if pred.sum() <= 0:
        return None
    return float(meas.sum() / pred.sum())


def stage_tail(fh, chd, shapes, nfiles=5):
    """(e) Tail consistency: does the pulse after a REAL rail look like the single clean pulse ToT assumes?"""
    import uproot
    out(fh, f"\n# Verdict D (e): tail consistency -- r = measured / tot_fill-predicted integral over the {TAIL} ticks after the "
            "rail run (no truth needed)")
    out(fh, "  reference: synthetic clips of held-out run-039253 pulses (isolated, single pulses by construction); "
            "real: every cathode rail run of the census files whose tail window holds no other rail sample")
    z = load_pulses()
    seg, pk, b, C, F = z["seg"], z["pk"], z["base"], z["chan"], z["fidx"]
    ref, real = {}, {}
    for d in DDATA:
        for i in np.nonzero(F == 1)[0]:
            c = int(C[i])
            s_ = seg[i].astype(np.float64)
            rl = b[i] + (s_[pk[i]] - b[i]) / d
            cl = np.minimum(s_, rl)
            rr = rail_run(cl, rl - 0.5, max(0, pk[i] - S.WIN_PRE), pk[i] + S.WIN_POST)
            if rr is None or (cl[rr[1]:rr[1] + TAIL] >= rl - 0.5).any():
                continue
            r = tail_ratio(cl, rr[0], rr[1], b[i], rl - b[i], shapes[c])
            if r is not None:
                k = TEDGES[int(np.searchsorted(TEDGES, rr[1] - rr[0], side="right")) - 1]
                ref.setdefault(k, []).append(r)
    files = sorted(glob.glob(os.path.join(RAWWF, "*_rawwf.root")))[:nfiles]
    n_runs = n_close = n_skip = 0
    beyond = 0
    pile = {"n": 0, "r10": 0, "r30": 0}
    for rf in files:
        t = uproot.open(rf)["rawdump/raw_waveform"]
        for ev in sorted(set(t.arrays(["event"], library="np")["event"])):
            (waves, _), _ = load_cathode_streams(rf, ev)
            for c, w in waves.items():
                if c not in shapes:
                    continue
                # pileup rate: a second rise (4-tick, 3-tick smoothed) of >= 10 % / 30 % of the first pulse's
                # amplitude 40-200 ticks after a bright (> 3000 ADC) unrailed pulse -- the sim pileup arm's geometry
                wf = np.asarray(w, np.float64)
                hh = wf - np.median(wf)
                sm = np.convolve(hh, np.ones(3) / 3, "same")
                rise = sm[4:] - sm[:-4]
                last = -10 ** 9
                for i0 in np.flatnonzero((hh > 3000) & (wf < RAIL)):
                    if i0 - last < 400:
                        continue
                    p0 = i0 + int(np.argmax(hh[i0:i0 + 10]))
                    if (wf[max(0, p0 - 400):p0 + 400] >= RAIL).any() or p0 + 200 > len(rise):
                        continue
                    last = i0
                    mx = rise[p0 + 40:p0 + 200].max()
                    pile["n"] += 1
                    pile["r10"] += int(mx > 0.1 * hh[p0])
                    pile["r30"] += int(mx > 0.3 * hh[p0])
                ivs = rail_intervals(w)
                for q, (a, e) in enumerate(ivs):
                    n_runs += 1
                    nxt = ivs[q + 1][0] if q + 1 < len(ivs) else None
                    if nxt is not None and nxt - e <= 50:
                        n_close += 1              # another rail within 50 ticks: a dip / second pulse
                    if e - a > 345:
                        beyond += 1
                    if nxt is not None and nxt < e + TAIL:
                        n_skip += 1
                        continue
                    sg_, o_ = seg_at(w, a)
                    bs_ = S.seg_baseline(sg_)
                    r = tail_ratio(sg_, a - o_, e - o_, bs_, RAIL - bs_, shapes[c])
                    if r is not None:
                        k = TEDGES[int(np.searchsorted(TEDGES, e - a, side="right")) - 1]
                        real.setdefault(k, []).append(r)
    out(fh, f"  real rail runs {n_runs}; followed by another rail within 50 ticks (dip / second pulse): {n_close} "
            f"({100 * n_close / max(1, n_runs):.1f} %); tail window blocked by a rail (skipped): {n_skip}; "
            f"longer than 345 ticks (beyond the ToT training range, d ~ 25): {beyond} ({100 * beyond / max(1, n_runs):.1f} %)")
    out(fh, f"  pileup rate (real data): of {pile['n']} bright unrailed cathode pulses, a second rise >= 30 % of the peak "
            f"40-200 ticks later in {100 * pile['r30'] / max(1, pile['n']):.2f} %, >= 10 % in "
            f"{100 * pile['r10'] / max(1, pile['n']):.2f} %  (sim pileup arm = 30 % at 40-200 ticks: tot_fill +17 % at d 4, "
            f"+47 % at d >= 6.7)")
    out(fh, "  NOTE (sim): r is blind to pileup INSIDE the run (the pileup arm keeps r ~ 1.00 while tot_fill is +24..+49 %); "
            "r does see a slower tail (tau_s +20 %: r ~ 1.05, tot_fill +5..+17 % at d >= 6.7)")
    out(fh, f"{'ToT>=':>6} | {'reference n':>11} {'r median [16,84]':>24} | {'real n':>7} {'r median [16,84]':>24} | "
            f"{'real/ref median':>15} {'real above ref p84':>18}")
    for k in TEDGES[:-1]:
        a_, b_ = np.array(ref.get(k, [])), np.array(real.get(k, []))
        fa = (f"{len(a_):>11} {np.median(a_):7.3f} [{np.percentile(a_, 16):6.3f},{np.percentile(a_, 84):6.3f}]"
              if len(a_) else f"{0:>11} {'-':>24}")
        fb = (f"{len(b_):>7} {np.median(b_):7.3f} [{np.percentile(b_, 16):6.3f},{np.percentile(b_, 84):6.3f}]"
              if len(b_) else f"{0:>7} {'-':>24}")
        rat = (f"{np.median(b_) / np.median(a_):15.3f} {100 * np.mean(b_ > np.percentile(a_, 84)):17.1f}%"
               if len(a_) and len(b_) else "")
        out(fh, f"{k:>6} | {fa} | {fb} | {rat}")


def main():
    ap = argparse.ArgumentParser()
    for s in ("shape", "sim", "develop", "bench", "data", "all"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--spe", choices=("v1", "v2"), default="v1",
                    help="SPE template set: v1 = doc 30 as committed; v2 = the production templates (doc 31), "
                         "cache /home/xqian/tmp/sat_tot_v2, records d31/")
    a = ap.parse_args()
    if a.spe == "v2":
        global CACHE, OUTD, PNG_TAG, MULTISTART
        MULTISTART = True    # doc 31 sec 2: the single-start fit stops at the start point on some channels
        v1cache = CACHE
        CACHE, OUTD, PNG_TAG = "/home/xqian/tmp/sat_tot_v2", os.path.join(os.path.dirname(HERE), "d31"), "_v2"
        rel = "pgrapher/experiment/protodunevd/pdvd-spe-templates-v2.json"
        tpl = S._find_templates().replace("pdvd-spe-templates.json", "pdvd-spe-templates-v2.json")
        assert tpl.endswith(rel) and os.path.exists(tpl), tpl
        S._find_templates = lambda: tpl     # load_channels() looks the name up at call time
        os.makedirs(CACHE, exist_ok=True)
        for f in ("pulses.npz", "quiet_0.npz", "quiet_1.npz"):   # raw-data caches, template independent
            if not os.path.exists(os.path.join(CACHE, f)) and os.path.exists(os.path.join(v1cache, f)):
                os.symlink(os.path.join(v1cache, f), os.path.join(CACHE, f))
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUTD, exist_ok=True)
    for s, fn in (("shape", shape_fit), ("sim", stage_sim), ("develop", stage_develop), ("bench", stage_bench),
                  ("data", stage_data)):
        if a.all or getattr(a, s):
            with open(os.path.join(OUTD, f"{s}.txt"), "w") as fh:
                fn(fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
