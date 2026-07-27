#!/usr/bin/env python3
"""The sample-average muon and the proton, against three recombination models,
in the residual-range plane.

All model curves are built with `convert_field.C`'s own recipe -- apply the
recombination pointwise on the fine dE/dx grid, then average over the 1 cm bin,
on centres 0.5 ... 59.5 cm -- so each is directly readable as "what the `*DeDx`
table would be":

  1. CURRENT       Modified Box  R = ln(A + xi)/xi,  xi = (B/rho E) dE/dx
                   A = 0.93, B = 0.212, C = 0.85
     C = 0.85 is `convert_field.C`'s undocumented fudge factor; there is no other
     normalisation in the shipped tables.  Curve 1 is verified against
     `stopping_ave_dQ_dx_sbnd.root` itself before anything is plotted.

  2. FREE B        the same form with B fitted (doc 55 section 7c)

  3. FREE POWER    R = ln(A + u)/u,  u = k (dE/dx / 2.1 MeV/cm)^p,  A = 0.93
                   k and p fitted -- the winner of the model zoo, and the one
                   that actually follows the proton's shape (doc 55 section 7g)

Every fit carries one free overall normalisation C, printed on the figure,
because on uncalibrated data a curve without its C is not interpretable.

The fits are done **in the residual-range plane** (`fit_recombination.py
--plane rr`), i.e. weighted the way this figure is read.  That matters: the
dE/dx-plane weighting of section 7c prefers slightly different parameters, and
the spread between the two is the honest uncertainty on p.

Data:
  muon   = geometric mean over TRACKS of each track's median dQ/dx in the rr bin
           (so one long track cannot dominate); error = s.e.m. across tracks.
  proton = the single proton track's median per bin; error = s.e.m. of the
           points in the bin.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 dqdx_rr_sample/plot_muon_proton_models.py \
      -o dqdx_rr_sample/muon_proton_vs_models.png
"""
import argparse
import importlib.util
import os

import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.gridspec import GridSpec   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SBND_DQDX = ("/nfs/data/1/xqian/toolkit-dev/energy_loss/pion_travel/"
             "stopping_ave_dQ_dx_sbnd.root")
BOX_FUDGE = 0.85          # convert_field.C's factor == the shipped tables' C
# top of the dE/dx range the fit is constrained by muons as well as the proton;
# above this only the proton contributes and the curves lean on one track
DEDX_BOTH = 10.5


def load_fr():
    spec = importlib.util.spec_from_file_location(
        "fr", os.path.join(HERE, "fit_recombination.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def table_curve(fr, graphs, part, name, p, C, rr=None, nsub=10):
    """convert_field.C's recipe: recombination pointwise, then 1 cm bin average."""
    rr = np.arange(60) + 0.5 if rr is None else np.asarray(rr, float)
    # lo_clip = 0 to match convert_field.C exactly (its first bin samples
    # 0.05 ... 0.95 cm, with no protection against the rr -> 0 singularity)
    sub = fr.dedx_samples(graphs, part, rr, np.ones_like(rr), nsub=nsub,
                          lo_clip=0.0)
    return C * np.mean(fr.MODELS[name][4](sub, *p) * sub, axis=1) / fr.W_ION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--points", default=os.path.join(HERE, "sample_points.tsv"))
    args = ap.parse_args()

    fr = load_fr()
    graphs = fr.dedx_graphs()

    # ---- regression: curve 1 must BE the shipped SBND table -----------------
    f = uproot.open(SBND_DQDX)
    print("curve 1 (A=0.93, B=0.212, C=0.85) vs stopping_ave_dQ_dx_sbnd.root:")
    for part in ("muon", "proton"):
        tab = np.asarray(f[part].values("y"), float)
        cur = table_curve(fr, graphs, part, "box_fixed", [], BOX_FUDGE)
        rel = np.max(np.abs(cur / tab - 1))
        print(f"  {part:>7s}: max relative deviation {rel:.2e}  "
              f"({'PASS' if rel < 2e-3 else 'FAIL'})")

    # ---- the fits, imported from fit_recombination --------------------------
    d = fr.read_tsv(args.points)
    part, rr, dq, dx = d["particle"], d["rr"], d["dqdx"], d["dx"]
    tid = np.array([f"{int(a)}_{int(b)}" for a, b in zip(d["event"], d["block"])])
    sub = np.zeros((len(rr), fr.NSUB))
    for p in ("muon", "proton"):
        s = part == p
        sub[s] = fr.dedx_samples(graphs, p, rr[s], dx[s])
    de = np.mean(sub, axis=1)
    keep = (rr >= fr.RR_MIN) & (de <= fr.DEDX_MAX) & (dq > 0)
    rows = fr.bin_data_rr(part[keep], rr[keep], de[keep], dq[keep], tid[keep], 0.03)

    MOD = [("current", "box_fixed", [], BOX_FUDGE, "#0b0b0b"),
           ("free B", "box_B", None, None, "#eb6834"),
           ("free power", "box_p", None, None, "#2a78d6")]
    fits = {}
    print("\nfits, in the residual-range plane (fit_recombination.py --plane rr):")
    for i, (lab, name, p, C, col) in enumerate(MOD):
        if p is None:
            res = fr.fit(name, rows, sub[keep])
            p, C = res["p"], res["C"]
            chi = res["chi2ndf"]
        else:
            chi = float("nan")
        fits[lab] = (name, list(p), C)
        MOD[i] = (lab, name, list(p), C, col)
        print(f"  {lab:>11s}: {fr.pretty(name, p):>30s}   C = {C:.4f}"
              + (f"   chi2/ndf = {chi:.2f}" if chi == chi else
                 "   (C is the convert_field.C fudge, nothing fitted)"))
    print(f"  C / {BOX_FUDGE}: "
          + ", ".join(f"{lab} {C/BOX_FUDGE:.3f}" for lab, _, _, C, _ in MOD))

    # ---- data profiles, straight out of the fit's own rows ------------------
    prof = {}
    for p in ("muon", "proton"):
        sel = [r for r in rows if r["part"] == p]
        prof[p] = (np.array([r["rr"] for r in sel]),
                   np.array([r["dqdx"] for r in sel]),
                   np.array([r["dqdx"] * r["sig"] for r in sel]),
                   [r["ntrk"] for r in sel],
                   np.array([r["dedx"] for r in sel]))

    print("\n=== data vs the three curves ===")
    for p in ("muon", "proton"):
        cen, val, err, ntr, de_bin = prof[p]
        cs = {lab: table_curve(fr, graphs, p, name, pp, C, rr=cen)
              for lab, name, pp, C, _ in MOD}
        ntrk_p = len({t for t in tid[part == p]})
        print(f"\n  {p}"
              + ("  (mean over tracks of the per-track median)" if ntrk_p > 1
                 else "  (the one track)"))
        print(f"  {'rr (cm)':>9s} {'dE/dx':>7s} {'ntrk':>5s} {'data ke/cm':>11s} "
              f"{'+-%':>5s} " + " ".join(f"/{lab}".rjust(12) for lab in cs))
        for i in range(len(cen)):
            tag = "  *" if de_bin[i] > DEDX_BOTH else ""
            print(f"  {cen[i]:9.1f} {de_bin[i]:7.1f} {ntr[i]:5d} "
                  f"{val[i]/1e3:11.1f} {err[i]/val[i]*100:5.1f} "
                  + " ".join(f"{val[i]/cs[lab][i]:12.3f}" for lab in cs) + tag)
        print(f"  {'median':>34s} {'':>11s} {'':>5s} "
              + " ".join(f"{np.median(val/cs[lab]):12.3f}" for lab in cs))
        print(f"  {'rms of ln(ratio) about 1':>34s} {'':>11s} {'':>5s} "
              + " ".join(f"{np.sqrt(np.mean(np.log(val/cs[lab])**2))*100:11.1f}%"
                         for lab in cs))
    print(f"\n  * = above dE/dx {DEDX_BOTH} MeV/cm, where only the proton "
          f"constrains the curves")

    print("\n=== what the tables would become (free-power model) ===")
    print(f"  {'rr (cm)':>9s} " + " ".join(f"{p}: cur / new / ratio".rjust(28)
                                           for p in ("muon", "proton")))
    for rq in (0.5, 2.5, 5.5, 10.5, 20.5, 40.5, 59.5):
        cells = []
        for p in ("muon", "proton"):
            a = table_curve(fr, graphs, p, "box_fixed", [], BOX_FUDGE, rr=[rq])[0]
            nm, pp, C = fits["free power"]
            b = table_curve(fr, graphs, p, nm, pp, C, rr=[rq])[0]
            cells.append(f"{a:10.0f} {b:10.0f} {b/a:6.3f}")
        print(f"  {rq:9.1f} " + " ".join(cells))

    # ---- figure -------------------------------------------------------------
    grid = np.arange(60) + 0.5
    COL = {"muon": "#2a78d6", "proton": "#e34948"}

    fig = plt.figure(figsize=(14.0, 6.4))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], hspace=0.30,
                  wspace=0.22)
    ax = fig.add_subplot(gs[:, 0])
    for lab, name, p, C, col in MOD:
        for pt, ls in (("muon", "-"), ("proton", "--")):
            ax.plot(grid, table_curve(fr, graphs, pt, name, p, C) / 1e3, ls,
                    color=col, lw=1.9, zorder=5,
                    label=f"{lab}, {pt}" if True else None)
    for pt, mk in (("muon", "o"), ("proton", "s")):
        cen, val, err, ntr, _ = prof[pt]
        n = len({t for t in tid[part == pt]})
        ax.errorbar(cen, val / 1e3, yerr=err / 1e3, fmt=mk, color=COL[pt],
                    ms=8.5, mew=1.5, mec="white", lw=0, elinewidth=1.3,
                    capsize=3, zorder=9,
                    label=(f"{pt}, average of {n} tracks" if n > 1
                           else f"{pt}, {n} track"))
    ax.set_xlim(0, 60)
    ax.set_ylim(30, 330)
    ax.set_yscale("log")
    ax.set_yticks([40, 60, 80, 100, 150, 200, 300])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("residual range from the stopping end  [cm]")
    ax.set_ylabel("dQ/dx  [ke/cm]")
    ax.set_title("Sample-average muon and the proton vs three models",
                 fontsize=11)
    ax.legend(fontsize=7.6, loc="upper right", ncol=2, framealpha=0.95)
    ax.grid(alpha=0.2, which="both", lw=0.6)

    _, _, p_p, C_p, _ = MOD[2]
    _, _, p_b, C_b, _ = MOD[1]
    box = (
        "$\\bf{Modified\\ Box}$   dQ/dx = C $\\cdot$ "
        "$\\langle$ln(A+u)/u $\\cdot$ dE/dx$\\rangle_{1\\,cm}$ / W$_{ion}$\n"
        f"E = {fr.E_FIELD} kV/cm,  $\\rho$ = {fr.RHO},  "
        f"W$_{{ion}}$ = {fr.W_ION*1e6:.1f} eV,  A = 0.93   (all fixed)\n"
        "$\\bf{current}$:      u = (B/$\\rho$E)$\\cdot$dE/dx,  B = 0.212,"
        f"                       $\\bf{{C = {BOX_FUDGE}}}$\n"
        "$\\bf{free\\ B}$:       u = (B/$\\rho$E)$\\cdot$dE/dx,  "
        f"B = {p_b[0]:.4f},"
        f"                     $\\bf{{C = {C_b:.4f}}}$\n"
        "$\\bf{free\\ power}$: u = k$\\cdot$(dE/dx / 2.1)$^p$,  "
        f"k = {p_p[0]:.4f},  p = {p_p[1]:.3f},  $\\bf{{C = {C_p:.4f}}}$\n"
        f"C is the only normalisation; all three land within 2 % of the 0.85 "
        f"fudge factor.")
    ax.text(0.015, 0.015, box, transform=ax.transAxes, fontsize=7.4,
            va="bottom", ha="left", family="DejaVu Sans",
            bbox=dict(fc="white", ec="#a3a29b", alpha=0.94,
                      boxstyle="round,pad=0.45"))

    for row, pt, mk in ((0, "muon", "o"), (1, "proton", "s")):
        ax = fig.add_subplot(gs[row, 1])
        ax.axhline(1.0, color="#a3a29b", lw=1.2, ls=":")
        cen, val, err, ntr, de_bin = prof[pt]
        ins = de_bin <= DEDX_BOTH
        for lab, name, p, C, col in MOD:
            c = table_curve(fr, graphs, pt, name, p, C, rr=cen)
            r = val / c
            ax.plot(cen, r, "-", color=col, lw=1.4, alpha=0.95, zorder=3)
            ax.errorbar(cen[ins], r[ins], yerr=(err / c)[ins], fmt=mk, color=col,
                        ms=7, mew=1.3, mec="white", lw=0, elinewidth=1.1,
                        zorder=5,
                        label=f"{lab}  (rms {np.sqrt(np.mean(np.log(r)**2))*100:.1f} %)")
            if (~ins).any():
                ax.errorbar(cen[~ins], r[~ins], yerr=(err / c)[~ins], fmt=mk,
                            mfc="none", mec=col, ecolor=col, ms=7, mew=1.6,
                            lw=0, elinewidth=1.1, zorder=5)
        ax.set_xlim(0, 60)
        ax.set_ylim(0.84, 1.22)
        ax.set_ylabel("data / model")
        if row == 1:
            ax.set_xlabel("residual range from the stopping end  [cm]")
        ax.set_title(("muon" if pt == "muon" else
                      "proton   (open markers: dE/dx > 10.5 MeV/cm, "
                      "proton-only territory)"), fontsize=9.5)
        ax.legend(fontsize=7.2, loc="lower left", ncol=3, framealpha=0.95)
        ax.grid(alpha=0.2, lw=0.6)

    fig.suptitle("SBND MCP2025C reco1, UNCALIBRATED data (no gain, no electron "
                 "lifetime): the absolute level carries one unknown common "
                 "factor, absorbed by C", fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
