#!/usr/bin/env python3
"""Describe the collected dQ/dx-vs-rr sample with a recombination model.

The measured quantity is dQ/dx(rr) on stopping tracks.  The model is

    dQ/dx = C * R(dE/dx) * dE/dx / W_ion  [ * exp(-t_drift/tau) ]

with dE/dx(rr) taken from the same `pion_travel/stopping.root` graphs that
`convert_field.C` uses to build the reference tables, averaged over each
measured point's own `dx` window (the measurement is an average over that
segment, so the model must be too).

`R` is the recombination factor.  Two families:

    Modified Box   R = ln(A + xi) / xi,        xi = (B / (rho E)) dE/dx
    Birks          R = 1 / (1 + kB' dE/dx),    kB' = kB / (rho E)

Every fit carries a free overall scale `C` that absorbs the (uncalibrated)
electronics gain, the mean electron-lifetime attenuation, and -- for Modified
Box -- the undocumented 0.85 fudge factor of `convert_field.C`.  Birks' A_B is
exactly degenerate with `C` and is folded into it, so Birks has one shape
parameter (kB) exactly as Modified Box with A fixed has one (B).

**Why the fit runs on dE/dx-binned medians, not raw points.**  ~76 % of the
sample's points sit in one MIP bin (dE/dx 2.0-2.3 MeV/cm -- the long plateau of
the through-going part of each muon), so a point-weighted fit has almost no
lever arm on the shape of R.  Equal treatment per dE/dx bin gives the
recombination curve the 2 -> 23 MeV/cm lever arm the sample contains.  Muon bins
and proton bins enter as *separate* data points, so a model can only fit both if
one R(dE/dx) really describes both particles -- that is the whole point.

Bin errors are the standard error of the mean of ln(dQ/dx) added in quadrature
with a `SYS_FLOOR` systematic, because the sparse high-dE/dx bins would
otherwise dominate the chi2 on statistical scatter alone, and because this is
uncalibrated data where a few-percent point-to-point systematic is certain.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 dqdx_rr_sample/fit_recombination.py -o dqdx_rr_sample/recomb_fit.png
  python3 dqdx_rr_sample/fit_recombination.py --rr-max 60   # robustness variant
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

EDGES = np.array([2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 4.6, 5.4, 6.5, 8.0, 10.5, 14.0,
                  20.0, 30.0])
MIN_IN_BIN = 4
SYS_FLOOR = 0.03      # per-bin systematic floor on ln(dQ/dx)


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


def dedx_of(graphs, part, rr, dx, nsub=7):
    """dE/dx averaged over [rr-dx/2, rr+dx/2], clipped at the track end."""
    gx, gy = graphs[part]
    lo = np.maximum(rr - dx / 2.0, RR_MIN / 2.0)
    hi = rr + dx / 2.0
    acc = np.zeros_like(rr)
    for j in range(nsub):
        acc += np.interp(lo + (hi - lo) * (j + 0.5) / nsub, gx, gy)
    return acc / nsub


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


def pretty(name, p):
    names = MODELS[name][0]
    if not names:
        return f"A={BOX_UB[0]} B={BOX_UB[1]} (both fixed)"
    body = "  ".join(f"{n}={v:.4f}" for n, v in zip(names, p))
    return (f"A={BOX_UB[0]} (fixed)  {body}" if name == "box_B" else body)


def model_dqdx(name, p, dedx, C, tau=None, drift=None):
    R = MODELS[name][4](dedx, *p)
    out = C * R * dedx / W_ION
    if tau is not None:
        out = out * np.exp(-drift / tau)
    return out


# ------------------------------------------------------------------------- fit
def bin_data(part, dedx, dqdx, drift):
    rows = []
    for p in ("muon", "proton"):
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = (part == p) & (dedx >= lo) & (dedx < hi)
            if s.sum() < MIN_IN_BIN:
                continue
            ln = np.log(dqdx[s])
            rows.append(dict(part=p, lo=float(lo), hi=float(hi), n=int(s.sum()),
                             dedx=float(np.median(dedx[s])),
                             dqdx=float(np.exp(np.median(ln))),
                             drift=float(np.median(drift[s])),
                             sig=float(np.hypot(np.std(ln) / np.sqrt(s.sum()),
                                                SYS_FLOOR))))
    return rows


def fit(name, rows, use=("muon", "proton"), free_tau=False):
    sel = [r for r in rows if r["part"] in use]
    de = np.array([r["dedx"] for r in sel])
    dq = np.array([r["dqdx"] for r in sel])
    sg = np.array([r["sig"] for r in sel])
    dr = np.array([r["drift"] for r in sel])
    _, p0, plo, phi, _ = MODELS[name]

    def unpack(theta):
        lnC = theta[0]
        p = list(theta[1:1 + len(p0)])
        tau = np.exp(theta[-1]) if free_tau else None
        return np.exp(lnC), p, tau

    def resid(theta):
        C, p, tau = unpack(theta)
        m = model_dqdx(name, p, de, C, tau, dr if free_tau else None)
        return (np.log(dq) - np.log(m)) / sg

    R0 = MODELS[name][4](np.array([2.1]), *p0)
    C0 = 50e3 / float(R0[0] * 2.1 / W_ION)
    t0 = [np.log(C0)] + list(p0) + ([np.log(10000.0)] if free_tau else [])
    lo = [np.log(C0) - 5] + list(plo) + ([np.log(300.0)] if free_tau else [])
    hi = [np.log(C0) + 5] + list(phi) + ([np.log(1e6)] if free_tau else [])
    r = least_squares(resid, t0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14)
    C, p, tau = unpack(r.x)
    rs = resid(r.x)
    ndf = max(len(sel) - len(r.x), 1)
    return dict(name=name, p=p, C=C, tau=tau, sel=sel,
                chi2=float(np.sum(rs ** 2)), ndf=ndf,
                chi2ndf=float(np.sum(rs ** 2) / ndf),
                rms=float(np.sqrt(np.mean((np.log(dq) -
                                           np.log(model_dqdx(name, p, de, C, tau,
                                                             dr if free_tau else None))) ** 2))))


def show(res, rows, use, tag, quiet=False):
    name, p, C, tau = res["name"], res["p"], res["C"], res["tau"]
    if quiet:
        return res
    print(f"\n--- {name}{'+tau' if tau else ''}  "
          f"(fit on {'+'.join(use)}{'; ' + tag if tag else ''}) ---")
    print(f"  {pretty(name, p)}   C={C:.5f}" + (f"   tau={tau/1e3:.2f} ms" if tau else ""))
    print(f"  chi2/ndf = {res['chi2']:.2f}/{res['ndf']} = {res['chi2ndf']:.2f}"
          f"   rms ln(data/model) = {res['rms']*100:.2f} %")
    de = np.array([r["dedx"] for r in rows])
    dr = np.array([r["drift"] for r in rows])
    pred = model_dqdx(name, p, de, C, tau, dr if tau else None)
    print(f"  {'part':>7s} {'dE/dx bin':>13s} {'n':>5s} {'<dE/dx>':>8s} "
          f"{'data ke/cm':>10s} {'+-%':>5s} {'model':>9s} {'ratio':>6s}")
    for r, pr in zip(rows, pred):
        flag = "" if r["part"] in use else "  (not fitted)"
        print(f"  {r['part']:>7s} {r['lo']:5.1f} - {r['hi']:5.1f} {r['n']:5d} "
              f"{r['dedx']:8.2f} {r['dqdx']/1e3:10.1f} {r['sig']*100:5.1f} "
              f"{pr/1e3:9.1f} {r['dqdx']/pr:6.3f}{flag}")
    for q in ("muon", "proton"):
        s = [(r, pr) for r, pr in zip(rows, pred) if r["part"] == q]
        if s:
            v = np.array([r["dqdx"] / pr for r, pr in s])
            print(f"  {q:>7s}: median ratio {np.median(v):.3f}, "
                  f"rms ln ratio {np.std(np.log(v))*100:.2f} %  ({len(v)} bins)")
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
        r = pr[b]["dqdx"] / mu[b]["dqdx"]
        e = np.hypot(pr[b]["sig"], mu[b]["sig"])
        rat.append(r)
        print(f"  {b[0]:5.1f} - {b[1]:5.1f} {mu[b]['dqdx']/1e3:11.1f} "
              f"{pr[b]['dqdx']/1e3:13.1f} {r:6.3f} {e*100:5.1f}")
    rat = np.array(rat)
    print(f"  -> median proton/muon at matched dE/dx = {np.median(rat):.3f} "
          f"(spread {np.std(np.log(rat))*100:.1f} %, {len(rat)} bins)")
    print("  A recombination model is a function of dE/dx alone, so it can only\n"
          "  describe both particles if this column is flat at 1.")
    return float(np.median(rat))


def a_scan(rows):
    print("\n=== Modified Box A-B degeneracy (B refit at each fixed A) ===")
    de = np.array([r["dedx"] for r in rows])
    dq = np.array([r["dqdx"] for r in rows])
    sg = np.array([r["sig"] for r in rows])
    print(f"  {'A':>5s} {'B':>7s} {'C':>7s} {'chi2/ndf':>9s} {'rms%':>6s}")
    for A in (0.60, 0.70, 0.80, 0.90, 0.93, 1.00, 1.10, 1.20):
        def resid(t):
            lnC, B = t
            xi = (B / (RHO * E_FIELD)) * de
            return (np.log(dq) - np.log(np.exp(lnC) * np.log(A + xi) / xi
                                        * de / W_ION)) / sg
        r = least_squares(resid, [0.0, 0.212], bounds=([-5, 0.02], [5, 2.0]),
                          xtol=1e-14, ftol=1e-14)
        rs = resid(r.x)
        raw = np.log(dq) - np.log(np.exp(r.x[0])
                                  * np.log(A + (r.x[1] / (RHO * E_FIELD)) * de)
                                  / ((r.x[1] / (RHO * E_FIELD)) * de) * de / W_ION)
        print(f"  {A:5.2f} {r.x[1]:7.4f} {np.exp(r.x[0]):7.4f} "
              f"{np.sum(rs**2)/(len(de)-2):9.2f} {np.sqrt(np.mean(raw**2))*100:6.2f}")
    print("  R = ln(A+xi)/xi -> 1 as xi -> 0 requires A = 1, so A well below ~0.9\n"
          "  has no zero-density limit; 0.93 is the published empirical value.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", help="output PNG")
    ap.add_argument("--points", default=os.path.join(HERE, "sample_points.tsv"))
    ap.add_argument("--rr-max", type=float, default=None,
                    help="restrict to rr <= this (cm); 60 = the reference-table domain")
    args = ap.parse_args()

    d = read_tsv(args.points)
    graphs = dedx_graphs()
    part, rr, dq, dx = d["particle"], d["rr"], d["dqdx"], d["dx"]
    drift = d["drift_us"]
    de = np.zeros_like(rr)
    for p in ("muon", "proton"):
        s = part == p
        de[s] = dedx_of(graphs, p, rr[s], dx[s])

    keep = (rr >= RR_MIN) & (de <= DEDX_MAX) & (dq > 0)
    if args.rr_max:
        keep &= rr <= args.rr_max
    print(f"E = {E_FIELD} kV/cm, rho = {RHO} g/cm3, W_ion = {W_ION*1e6:.1f} eV")
    print(f"points: {len(rr)} in the sample, {keep.sum()} in the fit domain "
          f"(rr >= {RR_MIN} cm"
          + (f", rr <= {args.rr_max} cm" if args.rr_max else "")
          + f", dE/dx <= {DEDX_MAX} MeV/cm)")
    event, block = d["event"][keep], d["block"][keep]
    part, de, dq, drift, rr = part[keep], de[keep], dq[keep], drift[keep], rr[keep]
    for p in ("muon", "proton"):
        s = part == p
        print(f"  {p:>7s}: n={s.sum():5d}  dE/dx {de[s].min():5.2f} - "
              f"{de[s].max():5.2f} MeV/cm   drift {drift[s].min():.0f} - "
              f"{drift[s].max():.0f} us")

    rows = bin_data(part, de, dq, drift)
    nmu = sum(1 for r in rows if r["part"] == "muon")
    print(f"\n{len(rows)} dE/dx bins ({nmu} muon, {len(rows)-nmu} proton), "
          f">= {MIN_IN_BIN} points each, {SYS_FLOOR*100:.0f} % systematic floor")

    overlap_test(rows)

    both = ("muon", "proton")
    show(fit("box_fixed", rows, ("muon",)), rows, ("muon",),
         "the model the SBND tables use, muons only")
    base = show(fit("box_fixed", rows, both), rows, both, "baseline")
    res = {"box_fixed": base}
    for name in ("box_B", "box_AB", "birks"):
        res[name] = show(fit(name, rows, both), rows, both, "free shape")
    a_scan(rows)

    print("\n=== summary ===")
    print(f"  {'model':>14s} {'shape params':>34s} {'chi2/ndf':>9s} {'rms%':>6s} "
          f"{'mu med':>7s} {'p med':>6s}")
    for k in ("box_fixed", "box_B", "box_AB", "birks"):
        r = res[k]
        de_r = np.array([x["dedx"] for x in rows])
        dr_r = np.array([x["drift"] for x in rows])
        pred = model_dqdx(r["name"], r["p"], de_r, r["C"], r["tau"],
                          dr_r if r["tau"] else None)
        ra = np.array([x["dqdx"] for x in rows]) / pred
        ismu = np.array([x["part"] == "muon" for x in rows])
        lbl = pretty(r["name"], r["p"]) + (f"  tau={r['tau']/1e3:.1f}ms" if r["tau"] else "")
        print(f"  {k:>14s} {lbl:>34s} {r['chi2ndf']:9.2f} {r['rms']*100:6.2f} "
              f"{np.median(ra[ismu]):7.3f} {np.median(ra[~ismu]):6.3f}")

    # ---- drift dependence measured AT FIXED dE/dx, so recombination cancels
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
    if len(tb) >= 3:
        # ln Q = ln Q0 - t/tau
        sl, ic = np.polyfit(tb, np.log(tq), 1)
        print(f"  exponential fit: tau = {-1/sl/1e3:.1f} ms "
              f"(attenuation over the full 1290 us drift: "
              f"{np.exp(sl*1290)*100:.0f} %)")
        print("  Uncorrected attenuation cannot explain the proton: the proton "
              "track sits at\n  drift 1106-1251 us, the MOST attenuated corner "
              "of the sample, so applying\n  a lifetime correction moves it "
              "further UP, not down.")

    # ---- is the dE/dx trend a drift artefact?  Repeat it at fixed drift.
    print("\n=== the dE/dx shape at FIXED drift (muons only) ===")
    print("  each cell = median dQ/dx in that (dE/dx, drift) cell divided by the\n"
          "  same column's dE/dx = 2.0-2.3 cell, so a common gain/attenuation\n"
          "  cancels and only the shape of R(dE/dx) is left; last column is what\n"
          "  Modified Box A=0.93 B=0.212 at E=0.5 kV/cm predicts for that ratio")
    debins = [(2.0, 2.3), (2.3, 2.6), (2.6, 3.0), (3.0, 3.5), (3.5, 4.6),
              (4.6, 6.5), (6.5, 10.5)]
    drbins = [(0, 300), (300, 600), (600, 900)]
    hdr = "  " + f"{'dE/dx':>11s} " + " ".join(f"{a}-{b} us".rjust(11)
                                               for a, b in drbins)
    print(hdr + f" {'ModBox':>8s}")
    mu = part == "muon"
    for lo, hi in debins:
        cells = []
        for a, b in drbins:
            s = mu & (de >= lo) & (de < hi) & (drift >= a) & (drift < b)
            s0 = mu & (de >= 2.0) & (de < 2.3) & (drift >= a) & (drift < b)
            cells.append(f"{np.median(dq[s])/np.median(dq[s0]):11.3f}"
                         if s.sum() >= 6 and s0.sum() >= 6 else f"{'-':>11s}")
        s = mu & (de >= lo) & (de < hi)
        ref = model_dqdx("box_fixed", [], np.array([2.15]), 1.0)[0]
        pm = (model_dqdx("box_fixed", [], np.array([np.median(de[s])]), 1.0)[0] / ref
              if s.sum() >= 6 else float("nan"))
        print(f"  {lo:4.1f} - {hi:4.1f} " + " ".join(cells) + f" {pm:8.3f}")

    # ---- per-track scale under the two candidate models
    print("\n=== per-track median data/model ===")
    print(f"  {'part':>7s} {'event':>7s} {'blk':>4s} {'n':>5s} {'drift us':>9s} "
          f"{'box_fixed':>10s} {'box_B':>7s}")
    for p in ("muon", "proton"):
        for e in sorted(set(event[part == p])):
            sel_e = (part == p) & (event == e)
            for b in sorted(set(block[sel_e])):
                m = sel_e & (block == b)
                r0 = np.median(dq[m] / model_dqdx("box_fixed", base["p"], de[m],
                                                  base["C"]))
                r1 = np.median(dq[m] / model_dqdx("box_B", res["box_B"]["p"], de[m],
                                                  res["box_B"]["C"]))
                print(f"  {p:>7s} {int(e):7d} {int(b):4d} {m.sum():5d} "
                      f"{np.mean(drift[m]):9.0f} {r0:10.3f} {r1:7.3f}")

    if not args.out:
        return

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))
    # only draw the models where there is data to constrain them
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
        for q, mk, ms in (("muon", "o", 5.5), ("proton", "s", 8.0)):
            s = [x for x in rows if x["part"] == q]
            de_s = np.array([x["dedx"] for x in s])
            ra = np.array([x["dqdx"] for x in s]) / model_dqdx(r["name"], r["p"],
                                                              de_s, r["C"])
            er = np.array([x["sig"] for x in s])
            ax.errorbar(de_s, ra, yerr=ra * er, fmt=mk + ls, color=col, ms=ms,
                        mew=1.3, mec="white", lw=1.3, alpha=0.95,
                        label=f"{k}, {q}")
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
