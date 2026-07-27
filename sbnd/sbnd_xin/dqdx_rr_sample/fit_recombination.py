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


def dedx_graphs(names=("muon", "proton")):
    """dE/dx (MeV/cm) vs residual range (cm), x-sorted, from stopping.root.

    Defaults to the two particles this fit has data for; `make_ref_tables.py`
    asks for all five to build the reference tables.
    """
    f = uproot.open(STOPPING)
    out = {}
    for n in names:
        g = f[n]
        x = np.asarray(g.values("x"), float)
        y = np.asarray(g.values("y"), float)
        o = np.argsort(x)
        out[n] = (x[o], y[o])
    return out


def dedx_samples(graphs, part, rr, dx, nsub=NSUB, lo_clip=RR_MIN / 2.0):
    """(N, nsub) sub-samples of dE/dx across each point's own [rr-dx/2, rr+dx/2].

    `lo_clip` keeps the window off the rr -> 0 singularity of the dE/dx graph.
    It is essentially inactive for the fit (the RR_MIN cut already means
    lo >= 0.175 for a 0.65 cm dx), but a caller reproducing convert_field.C's
    1 cm bin average must pass lo_clip = 0 to match that macro's sampling.
    """
    gx, gy = graphs[part]
    lo = np.maximum(rr - dx / 2.0, lo_clip)
    hi = rr + dx / 2.0
    frac = (np.arange(nsub) + 0.5) / nsub
    return np.interp(lo[:, None] + (hi - lo)[:, None] * frac[None, :], gx, gy)


# ------------------------------------------------------------------- the models
def R_box(dedx, A, B):
    xi = (B / (RHO * E_FIELD)) * dedx
    return np.log(A + xi) / xi


def R_birks(dedx, kB):
    return 1.0 / (1.0 + (kB / (RHO * E_FIELD)) * dedx)


# --- the wider zoo.  All of these are written in terms of u = k*(dE/dx / X0)**p
# so that k is dimensionless and O(0.1-1) whatever p does; at p = 1,
# k = B/(rho E) * X0 for the box forms and kB/(rho E) * X0 for Birks.  X0 is the
# MIP anchor, so k is "how much quenching at MIP".
X0 = 2.1        # MeV/cm


def _u(dedx, k, p):
    return k * (dedx / X0) ** p


def R_box_p(dedx, k, p):
    """Modified Box with a free power on dE/dx (A held at the published 0.93)."""
    u = _u(dedx, k, p)
    return np.log(BOX_UB[0] + u) / u


def R_box1_p(dedx, k, p):
    """Thomas-Imel / Box with A = 1 and a free power.  Unlike A = 0.93 this has
    the correct zero-density limit R -> 1 as dE/dx -> 0, so it is the physically
    clean member of the family."""
    u = _u(dedx, k, p)
    return np.log1p(u) / u


def R_box_Akp(dedx, A, k, p):
    """Modified Box with A, the quenching strength and the power all free."""
    u = _u(dedx, k, p)
    return np.log(A + u) / u


def R_birks_p(dedx, k, p):
    """Birks with a free power on dE/dx."""
    return 1.0 / (1.0 + _u(dedx, k, p))


def R_power(dedx, b):
    """Pure power law: R ~ (dE/dx)^-b, so dQ/dx ~ (dE/dx)^(1-b).  Scale into C."""
    return (dedx / X0) ** (-b)


def R_birks_esc(dedx, k, f):
    """Birks plus a dE/dx-independent escape floor f (Doke-Birks in spirit):
    quenching saturates instead of continuing to 0 at large dE/dx."""
    return (1.0 - f) / (1.0 + _u(dedx, k, 1.0)) + f


def R_birks_quad(dedx, k1, k2):
    """Birks with a quadratic term -- quenches harder than 1/(dE/dx) at the top."""
    z = dedx / X0
    return 1.0 / (1.0 + k1 * z + k2 * z * z)


def R_box_birks(dedx, k, w):
    """Convex mix of the two families at a common quenching strength."""
    u = _u(dedx, k, 1.0)
    return w * np.log(BOX_UB[0] + u) / u + (1.0 - w) / (1.0 + u)


# name -> (param names, p0, lo, hi, R(dedx, *p))
MODELS = {
    # --- the four of doc 55 section 7c; parameterisation frozen so those
    # numbers stay reproducible
    "box_fixed": ([], [], [], [], lambda d: R_box(d, *BOX_UB)),
    "box_B":     (["B"], [0.212], [0.02], [2.0],
                  lambda d, B: R_box(d, BOX_UB[0], B)),
    "box_AB":    (["A", "B"], [0.93, 0.212], [0.60, 0.02], [1.20, 2.0],
                  lambda d, A, B: R_box(d, A, B)),
    "birks":     (["kB"], [BIRKS_ICARUS], [0.002], [1.0], R_birks),
    # --- the wider zoo
    "power":      (["b"], [0.25], [0.0], [1.0], R_power),
    "box_p":      (["k", "p"], [0.645, 1.0], [0.02, 0.3], [30.0, 4.0], R_box_p),
    "box1_p":     (["k", "p"], [0.900, 1.0], [0.02, 0.3], [30.0, 4.0], R_box1_p),
    "box_Akp":    (["A", "k", "p"], [0.93, 0.645, 1.0], [0.30, 0.02, 0.3],
                   [1.60, 30.0, 4.0], R_box_Akp),
    "birks_p":    (["k", "p"], [0.102, 1.0], [0.005, 0.3], [30.0, 4.0], R_birks_p),
    "birks_esc":  (["k", "f"], [0.102, 0.05], [0.005, 0.0], [8.0, 0.9],
                   R_birks_esc),
    "birks_quad": (["k1", "k2"], [0.10, 0.01], [0.0, 0.0], [5.0, 5.0],
                   R_birks_quad),
    "box_birks":  (["k", "w"], [0.5, 0.5], [0.02, 0.0], [8.0, 1.0], R_box_birks),
}
ZOO = ["box_fixed", "box_B", "box_AB", "birks", "power", "box_p", "box1_p",
       "box_Akp", "birks_p", "birks_esc", "birks_quad", "box_birks"]
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


RRB = [(0, 1.5), (1.5, 3), (3, 5), (5, 7.5), (7.5, 10), (10, 15), (15, 20),
       (20, 30), (30, 40), (40, 60)]
MIN_PTS_TRACK = 2       # points a track must have in an rr bin to contribute
MIN_TRACKS = 4          # tracks an rr bin needs for the muon average


def bin_data_rr(part, rr, dedx, dqdx, tid, sys_floor):
    """The same row structure, binned in RESIDUAL RANGE instead of dE/dx.

    This is the plane the reference tables are written in, and it weights the
    data differently: rr 10-60 cm is four bins here but sits inside a single
    dE/dx bin.  Which model "wins" depends on that weighting, so both planes are
    fitted and compared (doc 55 section 7g).

    The muon value is the geometric mean over TRACKS of each track's median in
    the bin, so one 400 cm track cannot dominate, and the error is the s.e.m.
    across tracks.  The proton has one track, so its value is that track's
    median and the error is the s.e.m. of its points.
    """
    rows = []
    for p in ("muon", "proton"):
        for lo, hi in RRB:
            per, idx = [], []
            for t in sorted(set(tid[part == p])):
                s = np.where((tid == t) & (rr >= lo) & (rr < hi) & (dqdx > 0))[0]
                if len(s) >= MIN_PTS_TRACK:
                    per.append(float(np.median(dqdx[s])))
                    idx.append(s)
            if len(per) < (MIN_TRACKS if p == "muon" else 1):
                continue
            allidx = np.concatenate(idx)
            per = np.array(per)
            if len(per) > 1:
                sig = float(np.std(np.log(per), ddof=1) / np.sqrt(len(per)))
            else:
                sig = float(np.std(np.log(dqdx[allidx]), ddof=1)
                            / np.sqrt(len(allidx)))
            rows.append(dict(part=p, lo=float(lo), hi=float(hi), n=len(allidx),
                             idx=allidx, ntrk=len(per),
                             # the MEDIAN rr of the member points, not the bin
                             # centre -- the wide outer bins are not uniformly
                             # populated and a curve must be read where the data
                             # actually is
                             rr=float(np.median(rr[allidx])),
                             dedx=float(np.median(dedx[allidx])),
                             dqdx=float(np.exp(np.mean(np.log(per)))),
                             drift=0.0,
                             sig=float(np.hypot(sig, sys_floor))))
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


def zoo(rows, sub):
    """Fit every family jointly, and each particle alone, and report the
    per-particle residual -- because "does it describe the PROTON shape" is a
    different question from "does it minimise the joint chi2"."""
    ismu = np.array([r["part"] == "muon" for r in rows])
    print("\n=== model zoo: joint fit (muon + proton together) ===")
    print("  rms columns are of ln(data/model) over that particle's bins; the\n"
          "  proton column is the one the shape question is about")
    print(f"  {'model':>11s} {'np':>3s} {'shape params':>28s} {'chi2/ndf':>9s} "
          f"{'rms all':>8s} {'rms mu':>7s} {'rms p':>7s} {'p med':>6s}")
    out = {}
    for name in ZOO:
        res = fit(name, rows, sub)
        ra, _ = ratios(res, rows, sub)
        rms = lambda m: float(np.sqrt(np.mean(np.log(ra[m]) ** 2))) * 100
        out[name] = (res, rms(ismu), rms(~ismu))
        print(f"  {name:>11s} {len(res['p']):3d} {pretty(name, res['p']):>28s} "
              f"{res['chi2ndf']:9.2f} {rms(np.ones_like(ismu)):8.2f} "
              f"{rms(ismu):7.2f} {rms(~ismu):7.2f} "
              f"{np.median(ra[~ismu]):6.3f}")

    print("\n=== the same families fitted to ONE particle at a time ===")
    print("  the single-particle rms is the FLOOR: no joint model can beat the\n"
          "  best a family does on that particle by itself")
    print(f"  {'model':>11s} | {'muon-only params':>26s} {'rms mu':>7s} | "
          f"{'proton-only params':>26s} {'rms p':>7s}")
    for name in ZOO:
        cells = []
        for who, m in (("muon", ismu), ("proton", ~ismu)):
            sel = [r for r in rows if r["part"] == who]
            if len(sel) <= len(MODELS[name][1]) + 1:
                cells.append((f"{'too few bins':>26s}", float("nan")))
                continue
            r1 = fit(name, rows, sub, use=(who,))
            ra1, _ = ratios(r1, sel, sub)
            cells.append((f"{pretty(name, r1['p']):>26s}",
                          float(np.sqrt(np.mean(np.log(ra1) ** 2))) * 100))
        print(f"  {name:>11s} | {cells[0][0]} {cells[0][1]:7.2f} | "
              f"{cells[1][0]} {cells[1][1]:7.2f}")

    print("\n  For reference, the intrinsic scatter of the data itself: the "
          "quoted per-bin\n  errors are")
    for who, m in (("muon", ismu), ("proton", ~ismu)):
        e = np.array([r["sig"] for r in rows])[m]
        print(f"    {who:>7s}: {e.min()*100:.1f} - {e.max()*100:.1f} %, "
              f"rms {np.sqrt(np.mean(e**2))*100:.1f} %  ({m.sum()} bins)")
    print("  A model whose per-particle rms is at or below that number is fitting\n"
          "  noise, not shape.")
    return out


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
    # measured, not asserted: with doc 55's single proton this printed
    # 1106-1251 us -- the most attenuated corner -- so a lifetime correction
    # could only move that track further UP.  With doc 62's population the range
    # spans the drift and the argument becomes a spread rather than a corner.
    pr = drift[part == "proton"]
    if len(pr):
        ntrk = "the proton track sits" if len(pr) < 200 else "the protons sit"
        print(f"  Where the protons sit in drift: {ntrk} at "
              f"{pr.min():.0f}-{pr.max():.0f} us (median {np.median(pr):.0f}).\n"
              "  Doc 55 had one proton, at 1106-1251 us -- the MOST attenuated\n"
              "  corner -- so uncorrected attenuation could only move it UP, never\n"
              "  explain an excess.  A population spanning the drift tests that\n"
              "  directly instead (doc 55 sec 11).")


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
    global DUMB, MIN_IN_BIN
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", help="output PNG")
    ap.add_argument("--points", default=os.path.join(HERE, "sample_points.tsv"))
    ap.add_argument("--rr-max", type=float, default=None,
                    help="restrict to rr <= this (cm); 60 = the reference-table domain")
    ap.add_argument("--sys-floor", type=float, default=0.03,
                    help="per-bin systematic floor on ln(dQ/dx) (default %(default)s)")
    ap.add_argument("--min-in-bin", type=int, default=None,
                    help="minimum points per dE/dx bin (default %d; use 3 to let "
                         "the proton's 10.5-14 MeV/cm Bragg-tip bin in)" % MIN_IN_BIN)
    ap.add_argument("--plane", choices=("dedx", "rr"), default="dedx",
                    help="bin (and therefore weight) the fit in dE/dx (default) "
                         "or in residual range -- the two weightings do not "
                         "prefer the same model, see doc 55 section 7g")
    ap.add_argument("--zoo", action="store_true",
                    help="fit the wider model zoo, jointly and per particle")
    ap.add_argument("--dumb-average", action="store_true",
                    help="apply R to the segment-averaged dE/dx instead of "
                         "averaging R*dE/dx across the segment (the WRONG side "
                         "of a concave R; for comparison only)")
    args = ap.parse_args()
    DUMB = args.dumb_average
    if args.min_in_bin:
        globals()["MIN_IN_BIN"] = args.min_in_bin

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

    if args.plane == "rr":
        tid = np.array([f"{int(a)}_{int(b)}" for a, b in
                        zip(d["event"][keep], d["block"][keep])])
        rows = bin_data_rr(part, rr, de, dq, tid, args.sys_floor)
        nmu = sum(1 for r in rows if r["part"] == "muon")
        print(f"\n{len(rows)} residual-range bins ({nmu} muon, "
              f"{len(rows)-nmu} proton), {args.sys_floor*100:.0f} % systematic "
              f"floor")
        print(f"  {'part':>7s} {'rr bin':>12s} {'ntrk':>5s} {'n':>5s} "
              f"{'<dE/dx>':>8s} {'dQ/dx ke/cm':>12s} {'+-%':>5s}")
        for r in rows:
            print(f"  {r['part']:>7s} {r['lo']:5.1f} -{r['hi']:5.1f} "
                  f"{r['ntrk']:5d} {r['n']:5d} {r['dedx']:8.2f} "
                  f"{r['dqdx']/1e3:12.1f} {r['sig']*100:5.1f}")
        if args.zoo:
            zoo(rows, sub)
        both = ("muon", "proton")
        for name in ("box_fixed", "box_B", "box_p"):
            show(fit(name, rows, sub, both), rows, sub, both, "rr-plane fit")
        return

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

    if args.zoo:
        zoo(rows, sub)

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
