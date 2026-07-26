#!/usr/bin/env python3
"""The sample-average muon and the proton, against the current expectation and
against the best-fit recombination model, in the residual-range plane.

Two model curves, both built with `convert_field.C`'s own recipe -- apply the
recombination pointwise on the fine dE/dx grid, then average over the 1 cm bin,
on centres 0.5 ... 59.5 cm -- so each is directly readable as "what the
`*DeDx` table would be":

  1. CURRENT EXPECTATION   A = 0.93, B = 0.212, C = 0.85
     C = 0.85 is `convert_field.C`'s undocumented fudge factor; there is no
     other normalisation in the shipped tables.  Curve 1 is verified against
     `stopping_ave_dQ_dx_sbnd.root` itself before anything is plotted.

  2. BEST FIT              A = 0.93, B fitted, C fitted
     from `fit_recombination.py` (dE/dx-binned, muons and proton together, one
     free normalisation).  Imported rather than hard-coded so the two scripts
     cannot drift apart.

Both normalisations are printed in the figure, because on uncalibrated data the
absolute scale is not a prediction and a curve without its C is not
interpretable.

Data:
  muon   = mean over TRACKS of each track's median dQ/dx in the rr bin (so one
           long track cannot dominate); error bar = s.e.m. across tracks.
  proton = the single proton track's median per bin; error bar = s.e.m. of the
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

HERE = os.path.dirname(os.path.abspath(__file__))
SBND_DQDX = ("/nfs/data/1/xqian/toolkit-dev/energy_loss/pion_travel/"
             "stopping_ave_dQ_dx_sbnd.root")
BOX_FUDGE = 0.85          # convert_field.C's factor == the shipped tables' C

# rr bins for the data profile (cm)
RRB = [(0, 1.5), (1.5, 3), (3, 5), (5, 7.5), (7.5, 10), (10, 15), (15, 20),
       (20, 30), (30, 40), (40, 60)]
MIN_PTS = 2          # points per track per bin
MIN_TRACKS = 4       # tracks per bin for the muon average
# top of the dE/dx range the recombination fit actually constrained (the highest
# bin with >= 4 points in fit_recombination.py); above this both curves
# extrapolate and the proton's Bragg tip lives there
DEDX_FIT_MAX = 10.5


def load_fr():
    spec = importlib.util.spec_from_file_location(
        "fr", os.path.join(HERE, "fit_recombination.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def table_curve(fr, graphs, part, A, B, C, rr=None, nsub=10):
    """convert_field.C's recipe: recombination pointwise, then 1 cm bin average."""
    rr = np.arange(60) + 0.5 if rr is None else np.asarray(rr, float)
    # lo_clip = 0 to match convert_field.C exactly (its first bin samples
    # 0.05 ... 0.95 cm, with no protection against the rr -> 0 singularity)
    sub = fr.dedx_samples(graphs, part, rr, np.ones_like(rr), nsub=nsub,
                          lo_clip=0.0)
    xi = (B / (fr.RHO * fr.E_FIELD)) * sub
    return C * np.mean(np.log(A + xi) / xi * sub, axis=1) / fr.W_ION


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
        cur = table_curve(fr, graphs, part, 0.93, 0.212, BOX_FUDGE)
        rel = np.max(np.abs(cur / tab - 1))
        print(f"  {part:>7s}: max relative deviation {rel:.2e}  "
              f"({'PASS' if rel < 2e-3 else 'FAIL'})")

    # ---- the best-fit model, imported from fit_recombination ----------------
    d = fr.read_tsv(args.points)
    part, rr, dq, dx = d["particle"], d["rr"], d["dqdx"], d["dx"]
    ev, blk = d["event"], d["block"]
    sub = np.zeros((len(rr), fr.NSUB))
    for p in ("muon", "proton"):
        s = part == p
        sub[s] = fr.dedx_samples(graphs, p, rr[s], dx[s])
    de = np.mean(sub, axis=1)
    keep = (rr >= fr.RR_MIN) & (de <= fr.DEDX_MAX) & (dq > 0)
    rows = fr.bin_data(part[keep], de[keep], dq[keep], d["drift_us"][keep], 0.03)
    best = fr.fit("box_B", rows, sub[keep])
    A_FIT, B_FIT, C_FIT = 0.93, best["p"][0], best["C"]
    print(f"\nbest fit (fit_recombination.py, box_B): A = {A_FIT}, "
          f"B = {B_FIT:.4f}, C = {C_FIT:.4f}   chi2/ndf = {best['chi2ndf']:.2f}")
    print(f"  for scale: the shipped tables' C is the {BOX_FUDGE} fudge factor, "
          f"so C moves by {C_FIT/BOX_FUDGE:.3f}x")
    print(f"  beta' = B/(rho E) = {B_FIT/(fr.RHO*fr.E_FIELD):.4f} cm/MeV "
          f"(published at 0.5 kV/cm: {0.212/(fr.RHO*fr.E_FIELD):.4f})")

    # ---- the data profiles --------------------------------------------------
    prof = {}
    tid = np.array([f"{int(e)}_{int(b)}" for e, b in zip(ev, blk)])
    for p in ("muon", "proton"):
        cen, val, err, ntr, de_bin = [], [], [], [], []
        for lo, hi in RRB:
            per = []
            allpts = []
            for t in sorted(set(tid[part == p])):
                s = (tid == t) & (rr >= lo) & (rr < hi) & (dq > 0)
                if s.sum() >= MIN_PTS:
                    per.append(np.median(dq[s]))
                    allpts.append(dq[s])
            need = MIN_TRACKS if p == "muon" else 1
            if len(per) < need:
                continue
            per = np.array(per)
            cen.append(0.5 * (lo + hi))
            de_bin.append(float(np.median(np.concatenate(
                [np.mean(fr.dedx_samples(graphs, p, np.array([0.5 * (lo + hi)]),
                                         np.array([hi - lo])), axis=1)]))))
            val.append(float(np.mean(per)))
            if len(per) > 1:
                err.append(float(np.std(per, ddof=1) / np.sqrt(len(per))))
            else:
                a = np.concatenate(allpts)
                err.append(float(np.std(a, ddof=1) / np.sqrt(len(a))))
            ntr.append(len(per))
        prof[p] = (np.array(cen), np.array(val), np.array(err), ntr,
                   np.array(de_bin))

    # ---- numbers ------------------------------------------------------------
    print("\n=== data vs the two curves ===")
    for p in ("muon", "proton"):
        cen, val, err, ntr, de_bin = prof[p]
        c1 = table_curve(fr, graphs, p, 0.93, 0.212, BOX_FUDGE, rr=cen)
        c2 = table_curve(fr, graphs, p, A_FIT, B_FIT, C_FIT, rr=cen)
        lab = ("muon: mean over tracks of the per-track median"
               if p == "muon" else "proton: the one track")
        print(f"\n  {lab}")
        print(f"  {'rr (cm)':>9s} {'dE/dx':>7s} {'n_trk':>5s} {'data ke/cm':>11s} "
              f"{'+-':>6s} {'current':>8s} {'ratio':>6s} {'best fit':>9s} {'ratio':>6s}")
        for i in range(len(cen)):
            tag = "  <- beyond the fitted dE/dx range" if de_bin[i] > DEDX_FIT_MAX else ""
            print(f"  {cen[i]:9.1f} {de_bin[i]:7.1f} {ntr[i]:5d} {val[i]/1e3:11.1f} "
                  f"{err[i]/1e3:6.1f} {c1[i]/1e3:8.1f} {val[i]/c1[i]:6.3f} "
                  f"{c2[i]/1e3:9.1f} {val[i]/c2[i]:6.3f}{tag}")
        r1, r2 = val / c1, val / c2
        ins = de_bin <= DEDX_FIT_MAX
        print(f"  median over all bins            "
              f"                            {np.median(r1):6.3f}"
              f"            {np.median(r2):6.3f}")
        print(f"  rms of ln(ratio) about 1, all   "
              f"                            {np.sqrt(np.mean(np.log(r1)**2))*100:5.1f}%"
              f"            {np.sqrt(np.mean(np.log(r2)**2))*100:5.1f}%")
        print(f"  same, dE/dx <= {DEDX_FIT_MAX} only        "
              f"                            "
              f"{np.sqrt(np.mean(np.log(r1[ins])**2))*100:5.1f}%"
              f"            {np.sqrt(np.mean(np.log(r2[ins])**2))*100:5.1f}%")

    print("\n=== what the replacement table would look like ===")
    print(f"  {'rr (cm)':>9s} " + " ".join(f"{p} cur / fit / ratio".rjust(26)
                                           for p in ("muon", "proton")))
    for rq in (0.5, 2.5, 5.5, 10.5, 20.5, 40.5, 59.5):
        cells = []
        for p in ("muon", "proton"):
            a = table_curve(fr, graphs, p, 0.93, 0.212, BOX_FUDGE, rr=[rq])[0]
            b = table_curve(fr, graphs, p, A_FIT, B_FIT, C_FIT, rr=[rq])[0]
            cells.append(f"{a:9.0f} {b:9.0f} {b/a:6.3f}")
        print(f"  {rq:9.1f} " + " ".join(cells))

    # ---- figure -------------------------------------------------------------
    grid = np.arange(60) + 0.5
    COL = {"muon": "#2a78d6", "proton": "#e34948"}
    CUR, FIT = "#0b0b0b", "#eb6834"

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8),
                             gridspec_kw=dict(width_ratios=[1.15, 1.0]))

    ax = axes[0]
    for p, ls in (("muon", "-"), ("proton", "--")):
        ax.plot(grid, table_curve(fr, graphs, p, 0.93, 0.212, BOX_FUDGE) / 1e3,
                ls, color=CUR, lw=2.0, zorder=5,
                label=f"current expectation, {p}")
        ax.plot(grid, table_curve(fr, graphs, p, A_FIT, B_FIT, C_FIT) / 1e3,
                ls, color=FIT, lw=2.0, zorder=5,
                label=f"best-fit model, {p}")
    for p, mk in (("muon", "o"), ("proton", "s")):
        cen, val, err, ntr, de_bin = prof[p]
        n = len({t for t in tid[part == p]})
        ax.errorbar(cen, val / 1e3, yerr=err / 1e3, fmt=mk, color=COL[p],
                    ms=8, mew=1.4, mec="white", lw=1.4, capsize=3, zorder=8,
                    label=(f"muon, sample average of {n} tracks" if p == "muon"
                           else f"proton, {n} track"))
    ax.set_xlim(0, 60)
    ax.set_ylim(30, 320)
    ax.set_yscale("log")
    ax.set_yticks([40, 60, 80, 100, 150, 200, 300])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("residual range from the stopping end  [cm]")
    ax.set_ylabel("dQ/dx  [ke/cm]")
    ax.set_title("Sample-average muon and the proton vs both models", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.95)
    ax.grid(alpha=0.2, which="both", lw=0.6)

    box = (
        "$\\bf{Modified\\ Box}$   dQ/dx = C $\\cdot$ "
        "$\\langle$ln(A+$\\xi$)/$\\xi\\cdot$dE/dx$\\rangle_{1\\,cm}$ / W$_{ion}$,"
        "   $\\xi$ = B/($\\rho$E)$\\cdot$dE/dx\n"
        f"E = {fr.E_FIELD} kV/cm,  $\\rho$ = {fr.RHO},  "
        f"W$_{{ion}}$ = {fr.W_ION*1e6:.1f} eV   (all fixed)\n"
        f"$\\bf{{current}}$:  A = 0.93,  B = 0.212,  "
        f"$\\bf{{C = {BOX_FUDGE}}}$   (the convert_field.C fudge; no other norm.)\n"
        f"$\\bf{{best\\ fit}}$: A = 0.93,  B = {B_FIT:.4f},  "
        f"$\\bf{{C = {C_FIT:.4f}}}$   ($\\chi^2$/ndf {best['chi2ndf']:.2f}, "
        f"C is {C_FIT/BOX_FUDGE:.2f}$\\times$ the current one)")
    ax.text(0.015, 0.015, box, transform=ax.transAxes, fontsize=7.6, va="bottom",
            ha="left", family="DejaVu Sans",
            bbox=dict(fc="white", ec="#a3a29b", alpha=0.94, boxstyle="round,pad=0.45"))

    ax = axes[1]
    ax.axhline(1.0, color="#a3a29b", lw=1.2, ls=":")
    summary = []
    for p, mk, ms in (("muon", "o", 7), ("proton", "s", 8)):
        cen, val, err, ntr, de_bin = prof[p]
        ins = de_bin <= DEDX_FIT_MAX
        for A, B, C, col, nm in ((0.93, 0.212, BOX_FUDGE, CUR, "current"),
                                 (A_FIT, B_FIT, C_FIT, FIT, "best fit")):
            c = table_curve(fr, graphs, p, A, B, C, rr=cen)
            ls = "-" if p == "muon" else "--"
            ax.plot(cen, val / c, ls, color=col, lw=1.4, alpha=0.95, zorder=3)
            ax.errorbar(cen[ins], (val / c)[ins], yerr=(err / c)[ins], fmt=mk,
                        color=col, ms=ms, mew=1.4, mec="white", lw=0,
                        capsize=0, elinewidth=1.1, alpha=0.95, zorder=5,
                        label=f"{p} / {nm}")
            if (~ins).any():   # open markers = extrapolated past the fit domain
                ax.errorbar(cen[~ins], (val / c)[~ins], yerr=(err / c)[~ins],
                            fmt=mk, mfc="none", mec=col, ecolor=col, ms=ms,
                            mew=1.6, lw=0, capsize=0, elinewidth=1.1, zorder=5)
            r = val / c
            summary.append((f"{p} / {nm}", float(np.median(r)),
                            float(np.sqrt(np.mean(np.log(r) ** 2)))))
    ax.set_xlim(0, 60)
    ax.set_ylim(0.80, 1.35)
    ax.set_xlabel("residual range from the stopping end  [cm]")
    ax.set_ylabel("data / model")
    ax.set_title("Ratio to each model — no per-curve rescaling, the C printed\n"
                 f"on the left is all there is.  Open markers: dE/dx > "
                 f"{DEDX_FIT_MAX:.0f} MeV/cm, beyond what the fit saw",
                 fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.95)
    ax.grid(alpha=0.2, lw=0.6)
    txt = "over the 10 rr bins:   median ratio  /  rms of ln(ratio) about 1\n" + \
        "\n".join(f"  {n:<22s} {m:5.3f}   {r*100:4.1f} %" for n, m, r in summary)
    ax.text(0.985, 0.02, txt, transform=ax.transAxes, fontsize=7.6, va="bottom",
            ha="right", family="DejaVu Sans Mono",
            bbox=dict(fc="white", ec="#a3a29b", alpha=0.94,
                      boxstyle="round,pad=0.4"))

    fig.suptitle("SBND MCP2025C reco1, UNCALIBRATED data (no gain, no electron "
                 "lifetime): the absolute level carries one unknown common factor",
                 fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
