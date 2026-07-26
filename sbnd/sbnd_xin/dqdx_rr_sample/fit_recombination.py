#!/usr/bin/env python3
"""Describe the collected dQ/dx-vs-rr sample with a recombination model.

The measured quantity is dQ/dx(rr) on stopping tracks.  The model is

    dQ/dx = C * < R(dE/dx) * dE/dx >_segment / W_ion

with dE/dx(rr) taken from the same `pion_travel/stopping.root` graphs that
`convert_field.C` uses to build the reference tables.  `R` is the recombination
factor.  Two families:

    Modified Box   R = ln(A + xi) / xi,        xi = (B / (rho E)) dE/dx
    Birks          R = 1 / (1 + kB' dE/dx),    kB' = kB / (rho E)

Every fit carries a free overall scale `C` that absorbs the (uncalibrated)
electronics gain, the mean electron-lifetime attenuation, and -- for Modified
Box -- the undocumented 0.85 fudge factor of `convert_field.C`.  Birks' A_B is
exactly degenerate with `C` and is folded into it, so Birks has one shape
parameter (kB) exactly as Modified Box with A fixed has one (B).

**The average is taken INSIDE R, per point.**  A measured point is dQ integrated
over its own `dx` (~0.65 cm) divided by dx, and `R` is concave, so
`<R(dE/dx) dE/dx>` and `R(<dE/dx>) <dE/dx>` are not the same number.  They
diverge exactly where dE/dx varies fastest across dx -- the high-dE/dx end that
carries the whole shape lever arm -- and Jensen's inequality puts the difference
in the same direction as the excess this script measures.  So the segment average
is done on the correct side of R (`point_model`), and the difference it makes is
reported by `--dumb-average` for anyone who wants to see it.

**Why the fit compares dE/dx-binned medians, not raw points.**  ~76 % of the
sample's points sit in one MIP bin (dE/dx 2.0-2.3 MeV/cm -- the long plateau of
the through-going part of each muon), so a point-weighted fit has almost no
lever arm on the shape of R.  Equal treatment per dE/dx bin gives the
recombination curve the 2 -> 23 MeV/cm lever arm the sample contains.  Muon bins
and proton bins enter as *separate* data points, so a model can only fit both if
one R(dE/dx) really describes both particles -- that is the whole point.

Data per bin is exp(median(ln dQ/dx)) (robust); the model per bin is
exp(mean(ln point_model)) over the same points (smooth in the parameters, and
within a narrow bin indistinguishable from the median).  Bin errors are the
standard error of the mean of ln(dQ/dx) added in quadrature with `--sys-floor`,
because the sparse high-dE/dx bins would otherwise dominate the chi2 on
statistical scatter alone, and because this is uncalibrated data where a
few-percent point-to-point systematic is certain.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 dqdx_rr_sample/fit_recombination.py -o dqdx_rr_sample/recomb_fit.png
  python3 dqdx_rr_sample/fit_recombination.py --rr-max 60      # one robustness arm
  python3 dqdx_rr_sample/fit_recombination.py --dumb-average   # R(<dE/dx>) instead
"""
import argparse
import os

import numpy as np
import uproot
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STOPPING = "/nfs/data/1/xqian/toolkit-dev/energy_loss/pion_travel/stopping.root"

E_FIELD = 0.5        # kV/cm, SBND nominal
RHO = 1.38           # g/cm^3
W_ION = 23.6e-6      # MeV per electron-ion pair

BOX_UB = (0.93, 0.212)     # ArgoNeuT / MicroBooNE eq 3.1 -- what the tables use
BIRKS_ICARUS = 0.0486      # kB, (kV/cm)(g/cm^2)/MeV (ICARUS; A_B = 0.800)

# Fit domain.  Below 0.5 cm the dE/dx graph is an ODE-integration singularity
# (363 MeV/cm at rr -> 0) and the reference tables stop there too; above
# 30 MeV/cm there is no muon coverage to pair the proton against.
RR_MIN = 0.5
DEDX_MAX = 30.0
NSUB = 9             # sub-samples across each point's dx

EDGES = np.array([2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 4.6, 5.4, 6.5, 8.0, 10.5, 14.0,
                  20.0, 30.0])
MIN_IN_BIN = 4


# ---------------------------------------------------------------- data loading
def read_tsv(path):
    lines = [l for l in open(path) if not l.startswith("#")]
    hdr = lines[0].rstrip("\n").split("\t")
    cols = {h: [] for h in hdr}
    for l in lines[1:]:
        for h, v in zip(hdr, l.rstrip("\n").split("\t")):
            cols[h].append(v)
    out = {}
    for h, v in cols.items():
        try:
            out[h] = np.array([float(x) for x in v])
        except ValueError:
            out[h] = np.array(v)
    return out


def dedx_graphs():
    f = uproot.open(STOPPING)
    out = {}
    for n in ("muon", "proton"):
        g = f[n]
        x = np.asarray(g.values("x"), float)
        y = np.asarray(g.values("y"), float)
        o = np.argsort(x)
        out[n] = (x[o], y[o])
    return out


def dedx_samples(graphs, part, rr, dx, nsub=NSUB):
    """(N, nsub) sub-samples of dE/dx across each point's own [rr-dx/2, rr+dx/2]."""
    gx, gy = graphs[part]
    lo = np.maximum(rr - dx / 2.0, RR_MIN / 2.0)
    hi = rr + dx / 2.0
    frac = (np.arange(nsub) + 0.5) / nsub
    return np.interp(lo[:, None] + (hi - lo)[:, None] * frac[None, :], gx, gy)


# ------------------------------------------------------------------- the models
def R_box(dedx, A, B):
    xi = (B / (RHO * E_FIELD)) * dedx
    return np.log(A + xi) / xi


def R_birks(dedx, kB):
    return 1.0 / (1.0 + (kB / (RHO * E_FIELD)) * dedx)


# name -> (param names, p0, lo, hi, R(dedx, *p))
MODELS = {
    "box_fixed": ([], [], [], [], lambda d: R_box(d, *BOX_UB)),
    "box_B":     (["B"], [0.212], [0.02], [2.0],
                  lambda d, B: R_box(d, BOX_UB[0], B)),
    "box_AB":    (["A", "B"], [0.93, 0.212], [0.60, 0.02], [1.20, 2.0],
                  lambda d, A, B: R_box(d, A, B)),
    "birks":     (["kB"], [BIRKS_ICARUS], [0.002], [1.0], R_birks),
}
DUMB = False      # set by --dumb-average: average dE/dx BEFORE applying R


def pretty(name, p):
    names = MODELS[name][0]
    if not names:
        return f"A={BOX_UB[0]} B={BOX_UB[1]} (both fixed)"
    body = "  ".join(f"{n}={v:.4f}" for n, v in zip(names, p))
    return (f"A={BOX_UB[0]} (fixed)  {body}" if name == "box_B" else body)


def model_dqdx(name, p, dedx, C):
    """Model at a single dE/dx -- for drawing curves, not for fitting."""
    return C * MODELS[name][4](dedx, *p) * dedx / W_ION


def point_model(name, p, sub, C):
    """Per-point prediction, segment-integrated on the correct side of R."""
    if DUMB:
        d = np.mean(sub, axis=1)
        return C * MODELS[name][4](d, *p) * d / W_ION
    return C * np.mean(MODELS[name][4](sub, *p) * sub, axis=1) / W_ION


# ------------------------------------------------------------------------- fit
def bin_data(part, dedx, dqdx, drift, sys_floor):
    """One row per (particle, dE/dx bin), carrying the member point indices."""
    rows = []
    for p in ("muon", "proton"):
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = np.where((part == p) & (dedx >= lo) & (dedx < hi))[0]
            if len(s) < MIN_IN_BIN:
                continue
            ln = np.log(dqdx[s])
            rows.append(dict(part=p, lo=float(lo), hi=float(hi), n=len(s), idx=s,
                             dedx=float(np.median(dedx[s])),
                             dqdx=float(np.exp(np.median(ln))),
                             drift=float(np.median(drift[s])),
                             sig=float(np.hypot(np.std(ln) / np.sqrt(len(s)),
                                                sys_floor))))
    return rows


def bin_model(name, p, sub, C, rows):
    """exp(mean(ln point_model)) per bin -- the model counterpart of the data."""
    return np.array([np.exp(np.mean(np.log(point_model(name, p, sub[r["idx"]], C))))
                     for r in rows])


def fit(name, rows, sub, use=("muon", "proton")):
    sel = [r for r in rows if r["part"] in use]
    dq = np.array([r["dqdx"] for r in sel])
    sg = np.array([r["sig"] for r in sel])
    _, p0, plo, phi, _ = MODELS[name]

    def resid(theta):
        C, p = np.exp(theta[0]), list(theta[1:])
        return (np.log(dq) - np.log(bin_model(name, p, sub, C, sel))) / sg

    R0 = MODELS[name][4](np.array([2.1]), *p0)
    C0 = 50e3 / float(R0[0] * 2.1 / W_ION)
    t0 = [np.log(C0)] + list(p0)
    r = least_squares(resid, t0,
                      bounds=([np.log(C0) - 5] + list(plo),
                              [np.log(C0) + 5] + list(phi)),
                      xtol=1e-12, ftol=1e-12)
    C, p = float(np.exp(r.x[0])), list(r.x[1:])
    rs = resid(r.x)
    ndf = max(len(sel) - len(r.x), 1)
    raw = np.log(dq) - np.log(bin_model(name, p, sub, C, sel))
    return dict(name=name, p=p, C=C, sel=sel, chi2=float(np.sum(rs ** 2)),
                ndf=ndf, chi2ndf=float(np.sum(rs ** 2) / ndf),
                rms=float(np.sqrt(np.mean(raw ** 2))))


def ratios(res, rows, sub):
    pred = bin_model(res["name"], res["p"], sub, res["C"], rows)
    return np.array([r["dqdx"] for r in rows]) / pred, pred


def show(res, rows, sub, use, tag):
    print(f"\n--- {res['name']}  (fit on {'+'.join(use)}"
          f"{'; ' + tag if tag else ''}) ---")
    print(f"  {pretty(res['name'], res['p'])}   C={res['C']:.5f}")
    print(f"  chi2/ndf = {res['chi2']:.2f}/{res['ndf']} = {res['chi2ndf']:.2f}"
          f"   rms ln(data/model) = {res['rms']*100:.2f} %")
    ra, pred = ratios(res, rows, sub)
    print(f"  {'part':>7s} {'dE/dx bin':>13s} {'n':>5s} {'<dE/dx>':>8s} "
          f"{'data ke/cm':>10s} {'+-%':>5s} {'model':>9s} {'ratio':>6s}")
    for r, pr, v in zip(rows, pred, ra):
        flag = "" if r["part"] in use else "  (not fitted)"
        print(f"  {r['part']:>7s} {r['lo']:5.1f} - {r['hi']:5.1f} {r['n']:5d} "
              f"{r['dedx']:8.2f} {r['dqdx']/1e3:10.1f} {r['sig']*100:5.1f} "
              f"{pr/1e3:9.1f} {v:6.3f}{flag}")
    for q in ("muon", "proton"):
        m = np.array([r["part"] == q for r in rows])
        if m.any():
            print(f"  {q:>7s}: median ratio {np.median(ra[m]):.3f}, "
                  f"rms ln ratio {np.std(np.log(ra[m]))*100:.2f} %  ({m.sum()} bins)")
    return res


def overlap_test(rows):
    """The decisive test: at the SAME dE/dx, do muon and proton agree?"""
    print("\n=== muon vs proton at matched dE/dx (model-independent) ===")
    mu = {(r["lo"], r["hi"]): r for r in rows if r["part"] == "muon"}
    pr = {(r["lo"], r["hi"]): r for r in rows if r["part"] == "proton"}
    both = sorted(set(mu) & set(pr))
    print(f"  {'dE/dx bin':>13s} {'muon ke/cm':>11s} {'proton ke/cm':>13s} "
          f"{'p/mu':>6s} {'+-%':>5s}")
    rat = []
    for b in both:
        v = pr[b]["dqdx"] / mu[b]["dqdx"]
        rat.append(v)
        print(f"  {b[0]:5.1f} - {b[1]:5.1f} {mu[b]['dqdx']/1e3:11.1f} "
              f"{pr[b]['dqdx']/1e3:13.1f} {v:6.3f} "
              f"{np.hypot(pr[b]['sig'], mu[b]['sig'])*100:5.1f}")
    rat = np.array(rat)
    print(f"  -> median proton/muon at matched dE/dx = {np.median(rat):.3f} "
          f"(spread {np.std(np.log(rat))*100:.1f} %, {len(rat)} bins)")
    print("  A recombination model is a function of dE/dx alone, so it can only\n"
          "  describe both particles if this column is flat at 1.")


def a_scan(rows, sub):
    print("\n=== Modified Box A-B degeneracy (B refit at each fixed A) ===")
    dq = np.array([r["dqdx"] for r in rows])
    sg = np.array([r["sig"] for r in rows])
    print(f"  {'A':>5s} {'B':>7s} {'C':>7s} {'chi2/ndf':>9s} {'rms%':>6s}")
    for A in (0.60, 0.70, 0.80, 0.90, 0.93, 1.00, 1.10, 1.20):
        def resid(t):
            C, B = np.exp(t[0]), t[1]
            m = np.array([np.exp(np.mean(np.log(
                C * np.mean(R_box(sub[r["idx"]], A, B) * sub[r["idx"]], axis=1)
                / W_ION))) for r in rows])
            return (np.log(dq) - np.log(m)) / sg
        r = least_squares(resid, [0.0, 0.212], bounds=([-5, 0.02], [5, 2.0]),
                          xtol=1e-12, ftol=1e-12)
        rs = resid(r.x)
        print(f"  {A:5.2f} {r.x[1]:7.4f} {np.exp(r.x[0]):7.4f} "
              f"{np.sum(rs**2)/(len(rows)-2):9.2f} "
              f"{np.sqrt(np.mean((rs*sg)**2))*100:6.2f}")
    print("  R = ln(A+xi)/xi -> 1 as xi -> 0 requires A = 1, so A well below ~0.9\n"
          "  has no zero-density limit; 0.93 is the published empirical value.")


def robustness(part, de, dq, drift, rr, sub, sys_floor):
    """Does B ~= 0.13 survive the choices that could be driving it?"""
    print("\n=== robustness of the free-B fit ===")
    print("  the MIP bin holds ~76 % of the points and its error is the systematic\n"
          "  floor, so it dominates; these arms vary what could make it dominate")
    print(f"  {'arm':>34s} {'nbins':>6s} {'B':>7s} {'chi2/ndf':>9s} "
          f"{'mu med':>7s} {'p med':>6s}")

    def arm(label, mask=None, floor=None, drop_mip=False):
        m = np.ones(len(de), bool) if mask is None else mask
        rows = bin_data(part[m], de[m], dq[m], drift[m],
                        sys_floor if floor is None else floor)
        # bin_data's idx are into the masked arrays -> use the masked sub too
        sb = sub[m]
        if drop_mip:
            rows = [r for r in rows if not (r["lo"] == 2.0 and r["part"] == "muon")]
        if len(rows) < 4:
            print(f"  {label:>34s}  too few bins")
            return
        res = fit("box_B", rows, sb)
        ra, _ = ratios(res, rows, sb)
        ismu = np.array([r["part"] == "muon" for r in rows])
        print(f"  {label:>34s} {len(rows):6d} {res['p'][0]:7.4f} "
              f"{res['chi2ndf']:9.2f} {np.median(ra[ismu]):7.3f} "
              f"{np.median(ra[~ismu]):6.3f}")

    arm(f"baseline (sys floor {sys_floor*100:.0f} %)")
    arm("sys floor 2 %", floor=0.02)
    arm("sys floor 5 %", floor=0.05)
    arm("sys floor 10 %", floor=0.10)
    arm("rr <= 60 cm (table domain)", mask=rr <= 60)
    arm("rr <= 30 cm", mask=rr <= 30)
    arm("MIP muon bin dropped entirely", drop_mip=True)
    print(f"  reference: the published value is B = {BOX_UB[1]}")


def lifetime(part, de, dq, drift):
    print("\n=== electron-lifetime check: MIP band (dE/dx 2.0-2.5) vs drift ===")
    mip = (de >= 2.0) & (de < 2.5) & (part == "muon")
    tb, tq = [], []
    for lo, hi in [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000),
                   (1000, 1300)]:
        s = mip & (drift >= lo) & (drift < hi)
        if s.sum() < 20:
            continue
        tb.append(float(np.median(drift[s])))
        tq.append(float(np.exp(np.median(np.log(dq[s])))))
        print(f"  drift {lo:4d}-{hi:4d} us: n={s.sum():5d}  "
              f"median dQ/dx = {tq[-1]/1e3:6.2f} ke/cm")
    tb, tq = np.array(tb), np.array(tq)
    for lab, k in (("all bins", 0), ("dropping the 0-200 us bin", 1)):
        if len(tb) - k >= 3:
            sl, _ = np.polyfit(tb[k:], np.log(tq[k:]), 1)
            print(f"  {lab:>28s}: tau = {-1/sl/1e3:5.1f} ms   "
                  f"attenuation over 1290 us = {np.exp(sl*1290)*100:.0f} %")
    print("  The first bin sits above the next two, so the all-bins slope is the\n"
          "  steeper of the two; treat tau as order-10 ms, not a calibration.")
    print("  Uncorrected attenuation cannot explain the proton either: the proton\n"
          "  track sits at drift 1106-1251 us, the MOST attenuated corner of the\n"
          "  sample, so a lifetime correction moves it further UP, not down.")


def fixed_drift_shape(part, de, dq, drift):
    print("\n=== the dE/dx shape at FIXED drift (muons only) ===")
    print("  each cell = median dQ/dx in that (dE/dx, drift) cell divided by the\n"
          "  same column's dE/dx = 2.0-2.3 cell, so a common gain and a common\n"
          "  attenuation both cancel and only the shape of R is left; the last\n"
          "  column is Modified Box A=0.93 B=0.212 at E=0.5 kV/cm")
    debins = [(2.0, 2.3), (2.3, 2.6), (2.6, 3.0), (3.0, 3.5), (3.5, 4.6),
              (4.6, 6.5), (6.5, 10.5)]
    drbins = [(0, 300), (300, 600), (600, 900)]
    print("  " + f"{'dE/dx':>11s} "
          + " ".join(f"{a}-{b} us".rjust(11) for a, b in drbins)
          + f" {'ModBox':>8s}")
    mu = part == "muon"
    above = filled = 0
    for lo, hi in debins:
        cells = []
        s = mu & (de >= lo) & (de < hi)
        ref = model_dqdx("box_fixed", [], np.array([2.15]), 1.0)[0]
        pm = (model_dqdx("box_fixed", [], np.array([np.median(de[s])]), 1.0)[0] / ref
              if s.sum() >= 6 else float("nan"))
        for a, b in drbins:
            t = mu & (de >= lo) & (de < hi) & (drift >= a) & (drift < b)
            t0 = mu & (de >= 2.0) & (de < 2.3) & (drift >= a) & (drift < b)
            if t.sum() >= 6 and t0.sum() >= 6:
                v = np.median(dq[t]) / np.median(dq[t0])
                cells.append(f"{v:11.3f}")
                if lo > 2.0:                     # the 2.0-2.3 row is 1 by construction
                    filled += 1
                    above += v > pm
            else:
                cells.append(f"{'-':>11s}")
        print(f"  {lo:4.1f} - {hi:4.1f} " + " ".join(cells) + f" {pm:8.3f}")
    print(f"  data > ModBox in {above} of the {filled} filled non-normalising cells")


def main():
    global DUMB
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", help="output PNG")
    ap.add_argument("--points", default=os.path.join(HERE, "sample_points.tsv"))
    ap.add_argument("--rr-max", type=float, default=None,
                    help="restrict to rr <= this (cm); 60 = the reference-table domain")
    ap.add_argument("--sys-floor", type=float, default=0.03,
                    help="per-bin systematic floor on ln(dQ/dx) (default %(default)s)")
    ap.add_argument("--dumb-average", action="store_true",
                    help="apply R to the segment-averaged dE/dx instead of "
                         "averaging R*dE/dx across the segment (the WRONG side "
                         "of a concave R; for comparison only)")
    args = ap.parse_args()
    DUMB = args.dumb_average

    d = read_tsv(args.points)
    graphs = dedx_graphs()
    part, rr, dq, dx = d["particle"], d["rr"], d["dqdx"], d["dx"]
    drift = d["drift_us"]
    sub = np.zeros((len(rr), NSUB))
    for p in ("muon", "proton"):
        s = part == p
        sub[s] = dedx_samples(graphs, p, rr[s], dx[s])
    de = np.mean(sub, axis=1)          # the binning coordinate

    keep = (rr >= RR_MIN) & (de <= DEDX_MAX) & (dq > 0)
    if args.rr_max:
        keep &= rr <= args.rr_max
    print(f"E = {E_FIELD} kV/cm, rho = {RHO} g/cm3, W_ion = {W_ION*1e6:.1f} eV, "
          f"{NSUB} sub-samples per dx"
          + ("   [--dumb-average: R applied to <dE/dx>]" if DUMB else ""))
    print(f"points: {len(rr)} in the sample, {keep.sum()} in the fit domain "
          f"(rr >= {RR_MIN} cm"
          + (f", rr <= {args.rr_max} cm" if args.rr_max else "")
          + f", dE/dx <= {DEDX_MAX} MeV/cm)")
    part, de, dq, drift, rr, sub = (part[keep], de[keep], dq[keep], drift[keep],
                                    rr[keep], sub[keep])
    for p in ("muon", "proton"):
        s = part == p
        print(f"  {p:>7s}: n={s.sum():5d}  dE/dx {de[s].min():5.2f} - "
              f"{de[s].max():5.2f} MeV/cm   drift {drift[s].min():.0f} - "
              f"{drift[s].max():.0f} us")

    rows = bin_data(part, de, dq, drift, args.sys_floor)
    nmu = sum(1 for r in rows if r["part"] == "muon")
    print(f"\n{len(rows)} dE/dx bins ({nmu} muon, {len(rows)-nmu} proton), "
          f">= {MIN_IN_BIN} points each, {args.sys_floor*100:.0f} % systematic floor")

    overlap_test(rows)

    both = ("muon", "proton")
    show(fit("box_fixed", rows, sub, ("muon",)), rows, sub, ("muon",),
         "the model the SBND tables use, muons only")
    res = {"box_fixed": show(fit("box_fixed", rows, sub, both), rows, sub, both,
                             "baseline")}
    for name in ("box_B", "box_AB", "birks"):
        res[name] = show(fit(name, rows, sub, both), rows, sub, both, "free shape")

    a_scan(rows, sub)

    print("\n=== summary ===")
    print(f"  {'model':>10s} {'shape params':>32s} {'chi2/ndf':>9s} {'rms%':>6s} "
          f"{'mu med':>7s} {'p med':>6s}")
    for k in ("box_fixed", "box_B", "box_AB", "birks"):
        r = res[k]
        ra, _ = ratios(r, rows, sub)
        ismu = np.array([x["part"] == "muon" for x in rows])
        print(f"  {k:>10s} {pretty(r['name'], r['p']):>32s} {r['chi2ndf']:9.2f} "
              f"{r['rms']*100:6.2f} {np.median(ra[ismu]):7.3f} "
              f"{np.median(ra[~ismu]):6.3f}")

    robustness(part, de, dq, drift, rr, sub, args.sys_floor)
    fixed_drift_shape(part, de, dq, drift)
    lifetime(part, de, dq, drift)

    print("\n=== per-track median data/model ===")
    event, block = d["event"][keep], d["block"][keep]
    print(f"  {'part':>7s} {'event':>7s} {'blk':>4s} {'n':>5s} {'drift us':>9s} "
          f"{'box_fixed':>10s} {'box_B':>7s}")
    for p in ("muon", "proton"):
        for e in sorted(set(event[part == p])):
            for b in sorted(set(block[(part == p) & (event == e)])):
                m = (part == p) & (event == e) & (block == b)
                r0 = np.median(dq[m] / point_model("box_fixed", res["box_fixed"]["p"],
                                                   sub[m], res["box_fixed"]["C"]))
                r1 = np.median(dq[m] / point_model("box_B", res["box_B"]["p"],
                                                  sub[m], res["box_B"]["C"]))
                print(f"  {p:>7s} {int(e):7d} {int(b):4d} {m.sum():5d} "
                      f"{np.mean(drift[m]):9.0f} {r0:10.3f} {r1:7.3f}")

    if not args.out:
        return

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))
    de_all = np.array([r["dedx"] for r in rows])
    grid = np.linspace(2.0, de_all.max() * 1.15, 500)
    style = {"box_fixed": ("#0b0b0b", "-"), "box_B": ("#eb6834", "--"),
             "box_AB": ("#2a78d6", "-"), "birks": ("#1baf7a", "-.")}

    ax = axes[0]
    for k in ("box_fixed", "box_B", "box_AB", "birks"):
        r = res[k]
        col, ls = style[k]
        ax.plot(grid, model_dqdx(r["name"], r["p"], grid, r["C"]) / 1e3,
                color=col, ls=ls, lw=1.8,
                label=f"{k}: {pretty(r['name'], r['p'])}  "
                      f"($\\chi^2$/ndf {r['chi2ndf']:.1f})")
    for q, col, mk in (("muon", "#52514e", "o"), ("proton", "#e34948", "s")):
        s = [r for r in rows if r["part"] == q]
        ax.errorbar([r["dedx"] for r in s], [r["dqdx"] / 1e3 for r in s],
                    yerr=[r["dqdx"] * r["sig"] / 1e3 for r in s], fmt=mk,
                    color=col, ms=7.5, mew=1.3, mec="white", lw=1.3, zorder=5,
                    label=f"{q} ({len(s)} bins)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dE/dx from the stopping-power table  [MeV/cm]")
    ax.set_ylabel("measured dQ/dx  [ke/cm]")
    ax.set_title("Recombination curve: muons and the proton on one axis")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.2, which="both", lw=0.6)

    ax = axes[1]
    ax.axhline(1.0, color="#a3a29b", lw=1.2, ls=":")
    for k in ("box_fixed", "box_B"):
        r = res[k]
        col, ls = style[k]
        ra, _ = ratios(r, rows, sub)
        for q, mk, ms in (("muon", "o", 5.5), ("proton", "s", 8.0)):
            m = np.array([x["part"] == q for x in rows])
            ax.errorbar(de_all[m], ra[m],
                        yerr=ra[m] * np.array([x["sig"] for x in rows])[m],
                        fmt=mk + ls, color=col, ms=ms, mew=1.3, mec="white",
                        lw=1.3, alpha=0.95, label=f"{k}, {q}")
    ax.set_xscale("log")
    ax.set_xlabel("dE/dx  [MeV/cm]")
    ax.set_ylabel("data / model")
    ax.set_ylim(0.78, 1.28)
    ax.set_title("Residual, published vs free-B Modified Box\n"
                 "squares = the proton, circles = muons", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    ax.grid(alpha=0.2, which="both", lw=0.6)

    fig.suptitle("SBND MCP2025C reco1, UNCALIBRATED data, E = 0.5 kV/cm; "
                 "every model carries one free overall normalisation C",
                 fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
