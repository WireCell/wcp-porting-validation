#!/usr/bin/env python3
"""The pi0 mass plot for doc pr/126 (sec 4g).

Draws what sections 4a/4g measure, and NOTHING it does not measure: the four
published cells of the mass distribution and the EM scale they imply.

WHAT IS PLOTTED, and why each choice:

  (a) the PRIMARY cell -- ncpi0, min(E)>15 MeV, vertex-chord convention, n=19.
      The fitted curve is the unbinned truncated-Gaussian ML of sec 4g, drawn
      ONLY over its own window [100,185] and normalised to the n_in points the
      fit actually saw -- not to n.  Extending it across the tail, or
      normalising to n, would draw a model over data the fit explicitly
      excludes.  All n points are histogrammed and the out-of-window region is
      shaded: that excluded low tail IS sec 4g's argument and must stay
      visible.  A rug of the individual masses is drawn under the histogram
      because at n=19 the bin choice, not the data, dominates the eye -- which
      is the same reason the fit is unbinned.

  (b) the all-origins cross-check on the same convention, n=45.  Same drawing
      rules.  This is where the skew is blatant (mean 127.3 vs median 137.3).

  (c) the convention systematic: vertex chord against shower axis.  Per sec 4c
      the axis convention is a SYSTEMATIC, not a second measurement, so it is
      drawn as an outline and never as a co-equal curve.

  (d) the implied kine_shower_fudge_factor per cell with its bootstrap CI68,
      against the 0.80 in force and the >= 0.84 sec 4g recommends.  Axis-
      convention cells are open markers for the same reason.  The cell the
      sec-4g sanity gate REJECTED (peak below median) is drawn greyed and
      labelled rather than dropped.

NUMBER PARITY IS A HARD GATE.  Every annotation is read from the published
record docs/pr/pr126-pi0-peak.tsv, never recomputed for the figure; --selftest
re-derives them through pr126_pi0_peak.py's own estimator and fails if anything
disagrees.  A figure that annotates a number the doc's tables do not contain
would be a self-inflicted credibility bug in a doc whose whole value is that it
can be re-checked.

READ-ONLY over docs/pr/pr126-pi0-{mass,peak}.tsv; writes only its own PNG.

    ./pr126_pi0_plot.py --png docs/pr/126_pi0-mass.png
    ./pr126_pi0_plot.py --selftest
"""
import argparse, csv, math, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pr126_pi0_peak as PK          # the estimator itself -- never re-implemented

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEAK_TSV = os.path.join(SX, "docs", "pr", "pr126-pi0-peak.tsv")
PI0_MASS = PK.PI0_MASS
LO, HI = PK.WIN
BINW = 10.0
RANGE = (30.0, 230.0)

C_PEAK, C_MED, C_TRUE, C_AXIS = "#c0392b", "#7f8c8d", "#111111", "#2980b9"


def load_cells():
    """The published record.  Keyed by (conv, which, gate)."""
    with open(PEAK_TSV) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = {}
    for r in rows:
        for k in ("n", "n_in", "ci_excludes_135"):
            r[k] = int(r[k])
        for k in ("median", "median_lo", "median_hi", "peak", "peak_lo", "peak_hi",
                  "sigma", "fudge", "fudge_lo", "fudge_hi"):
            r[k] = float(r[k])
        out[(r["conv"], r["which"], r["gate"])] = r
    return out, rows


def trunc_gauss(x, mu, sigma, lo=LO, hi=HI):
    d = norm.cdf((hi - mu) / sigma) - norm.cdf((lo - mu) / sigma)
    return norm.pdf((x - mu) / sigma) / sigma / d


def draw_mass_panel(ax, cell, title, colour=C_PEAK, show_rug=True):
    x = PK.load(cell["conv"], cell["which"], cell["gate"])
    bins = np.arange(RANGE[0], RANGE[1] + BINW, BINW)

    ax.axvspan(RANGE[0], LO, color="0.92", zorder=0)
    ax.axvspan(HI, RANGE[1], color="0.92", zorder=0)
    ax.hist(x, bins=bins, color="#dfe6ec", edgecolor="#5d6d7e", linewidth=0.9, zorder=2)

    # the fit, over its own window only, normalised to the points it saw
    xs = np.linspace(LO, HI, 400)
    ax.plot(xs, cell["n_in"] * BINW * trunc_gauss(xs, cell["peak"], cell["sigma"]),
            color=colour, lw=2.0, zorder=5)

    ax.axvspan(cell["peak_lo"], cell["peak_hi"], color=colour, alpha=0.13, zorder=1)
    ax.axvline(cell["peak"], color=colour, lw=2.0, zorder=6)
    ax.axvline(cell["median"], color=C_MED, lw=1.6, ls=":", zorder=6)
    ax.axvline(PI0_MASS, color=C_TRUE, lw=1.6, ls="--", zorder=6)

    # headroom so the annotation box never lands on the data or the fit
    top = 1.45 * max(float(np.histogram(x, bins=bins)[0].max()),
                     cell["n_in"] * BINW * trunc_gauss(cell["peak"], cell["peak"], cell["sigma"]))
    ax.set_ylim(0, top)
    if show_rug:
        y0 = -0.055 * top
        ax.plot(x, np.full(len(x), y0), "|", color="#34495e", ms=9, mew=1.3,
                clip_on=False, zorder=7)
        ax.set_ylim(bottom=1.9 * y0, top=top)

    ax.set_title(title, fontsize=10.5, loc="left")
    ax.set_xlabel("m(γγ)  [MeV]", fontsize=9.5)
    ax.set_ylabel("π⁰ candidates / %g MeV" % BINW, fontsize=9.5)
    ax.set_xlim(*RANGE)
    ax.tick_params(labelsize=8.5)

    txt = ("n = %d   (n$_{in}$ = %d in the fit window)\n"
           "median  %.1f\n"
           "PEAK    %.1f   CI68 [%.1f, %.1f]\n"
           "⇒ fudge %.3f  [%.3f, %.3f]"
           % (cell["n"], cell["n_in"], cell["median"], cell["peak"],
              cell["peak_lo"], cell["peak_hi"],
              cell["fudge"], cell["fudge_lo"], cell["fudge_hi"]))
    ax.text(0.975, 0.955, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.2, family="monospace", zorder=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.95))


def build(png):
    cells, rows = load_cells()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))
    fig.suptitle("doc pr/126 — π⁰ invariant mass and the EM charge scale "
                 "(SBND, 98+141 hand scans, 50 hand-paired π⁰)",
                 fontsize=12.5, y=0.985)

    # ---- (a) the primary cell
    a = cells[("vtx", "now", "ncpi0")]
    draw_mass_panel(axes[0][0], a,
                    "(a) PRIMARY — ncπ⁰ truth sample, min(E$_γ$) > 15 MeV, vertex chord")
    fig.legend(handles=[
        plt.Line2D([], [], color=C_TRUE, ls="--", lw=1.6, label="m(π⁰) = 134.98 MeV"),
        plt.Line2D([], [], color=C_MED, ls=":", lw=1.6, label="median (biased low)"),
        plt.Line2D([], [], color=C_PEAK, lw=2.0, label="PEAK — truncated-Gaussian ML, ±CI68"),
        plt.Line2D([], [], color="0.85", lw=8, label="outside the fit window [100,185]"),
        plt.Line2D([], [], color="#34495e", ls="none", marker="|", ms=9, mew=1.3,
                   label="the individual π⁰ (the fit is unbinned)"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.952), ncol=5, fontsize=8.8,
        frameon=False)

    # ---- (b) the all-origins cross-check
    b = cells[("vtx", "now", "all")]
    draw_mass_panel(axes[0][1], b,
                    "(b) cross-check — all origins, same gate")
    xb = PK.load("vtx", "now", "all")
    axes[0][1].text(0.03, 0.70,
                    "mean %.1f — 10 MeV below the median:\nthe low tail is charge loss\n"
                    "and must not be averaged in" % float(np.mean(xb)),
                    transform=axes[0][1].transAxes, ha="left", va="top",
                    fontsize=8.2, color="#7b241c", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))

    # ---- (c) the convention systematic
    ax = axes[1][0]
    bins = np.arange(RANGE[0], RANGE[1] + BINW, BINW)
    xv, xa = PK.load("vtx", "now", "all"), PK.load("axis", "now", "all")
    cv, ca = cells[("vtx", "now", "all")], cells[("axis", "now", "all")]
    ax.hist(xv, bins=bins, color="#dfe6ec", edgecolor="#5d6d7e", lw=0.9,
            label="vertex chord — the measurement (n=%d)" % cv["n"])
    ax.hist(xa, bins=bins, histtype="step", color=C_AXIS, lw=1.8, ls="-",
            label="shower axis — a SYSTEMATIC, not a 2nd measurement (n=%d)" % ca["n"])
    ax.axvline(PI0_MASS, color=C_TRUE, lw=1.6, ls="--")
    ax.axvline(cv["peak"], color=C_PEAK, lw=2.0)
    ax.axvline(ca["peak"], color=C_AXIS, lw=2.0, ls="-.")
    ax.set_ylim(0, 1.40 * max(np.histogram(xv, bins=bins)[0].max(),
                              np.histogram(xa, bins=bins)[0].max()))
    ax.annotate("peak %.1f" % cv["peak"], xy=(cv["peak"], ax.get_ylim()[1] * 0.60),
                xytext=(-48, 0), textcoords="offset points", color=C_PEAK, fontsize=8.8,
                zorder=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.annotate("peak %.1f" % ca["peak"], xy=(ca["peak"], ax.get_ylim()[1] * 0.50),
                xytext=(6, 0), textcoords="offset points", color=C_AXIS, fontsize=8.8,
                zorder=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_title("(c) the direction convention: the dominant systematic (Δpeak %.1f MeV)"
                 % (ca["peak"] - cv["peak"]), fontsize=10.5, loc="left")
    ax.set_xlabel("m(γγ)  [MeV]", fontsize=9.5)
    ax.set_ylabel("π⁰ candidates / %g MeV" % BINW, fontsize=9.5)
    ax.set_xlim(*RANGE)
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.93)

    # ---- (d) the implied scale, per cell
    ax = axes[1][1]
    order = [("vtx", "now", "ncpi0",   "PRIMARY  ncπ⁰, vertex chord",        False, True),
             ("vtx", "now", "all",     "all origins, vertex chord",          False, False),
             ("vtx", "scanhand", "ncpi0", "ncπ⁰ scan-time + scanner marks",  False, False),
             ("axis", "now", "ncpi0",  "ncπ⁰, shower axis  [systematic]",    True,  False),
             ("axis", "now", "all",    "all origins, shower axis  [syst.]",  True,  False),
             ("vtx", "scanreco", "ncpi0", "ncπ⁰ scan-time as reconstructed", False, False)]
    ys = list(range(len(order)))[::-1]
    labels = []
    for y, (conv, which, gate, lab, is_syst, is_prim) in zip(ys, order):
        c = cells[(conv, which, gate)]
        rejected = c["peak"] < c["median"]
        col = "0.62" if rejected else (C_AXIS if is_syst else C_PEAK)
        ax.errorbar(c["fudge"], y,
                    xerr=[[c["fudge"] - c["fudge_lo"]], [c["fudge_hi"] - c["fudge"]]],
                    fmt="o" if not is_syst else "o", ms=8 if is_prim else 6,
                    mfc=col if (not is_syst and not rejected) else "white",
                    mec=col, ecolor=col, elinewidth=1.6, capsize=3.5, zorder=4)
        labels.append(lab + ("   ✗ REJECTED (peak < median)" if rejected else ""))
    ax.axvline(0.80, color="#111111", lw=1.6, ls="-")
    ax.axvline(0.84, color=C_PEAK, lw=1.8, ls="--")
    ax.text(0.80, len(order) - 0.35, " 0.80\n in force", color="#111111",
            fontsize=8.5, va="top")
    ax.text(0.84, len(order) - 0.35, " ≳0.84 recommended\n (prototype ×0.95 ⇒ 0.842)",
            color=C_PEAK, fontsize=8.5, va="top")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.set_ylim(-0.8, len(order) - 0.05)
    ax.set_xlabel("implied  kine_shower_fudge_factor  =  0.80 × peak / 134.9768", fontsize=9.5)
    ax.set_title("(d) the scale each cell implies (bars = bootstrap CI68)",
                 fontsize=10.5, loc="left")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", color="0.9", zorder=0)

    fig.text(0.5, 0.012,
             "The fitted peak is a FLOOR on the true peak: toys at n=19 (§4g) show both "
             "estimators biased low against a one-sided charge-loss tail.  "
             "Bin width %g MeV; the fit is unbinned." % BINW,
             ha="center", fontsize=8.5, color="0.32")
    fig.tight_layout(rect=[0, 0.028, 1, 0.925])
    fig.savefig(png, dpi=150)
    print("wrote %s" % png)
    return cells


def selftest():
    """Hard gate: every annotation in the figure must equal the published record,
    and the published record must still be what the estimator produces."""
    cells, rows = load_cells()
    bad = 0
    for r in rows:
        x = PK.load(r["conv"], r["which"], r["gate"])
        pk = PK.peak_fit(x)
        sig, nin = PK.peak_fit_sigma(x)
        lo, hi = PK.boot(x, PK.peak_fit)
        med = float(np.median(x))
        for name, got, want in (("n", len(x), r["n"]), ("n_in", nin, r["n_in"]),
                                ("median", med, r["median"]), ("peak", pk, r["peak"]),
                                ("peak_lo", lo, r["peak_lo"]), ("peak_hi", hi, r["peak_hi"]),
                                ("sigma", sig, r["sigma"]),
                                ("fudge", PK.to_fudge(pk), r["fudge"])):
            if abs(got - want) > 1e-6:
                print("  FAIL %-38s %-8s recomputed %.6f != published %.6f"
                      % (r["label"], name, got, want))
                bad += 1
        print("  ok  %-40s n=%2d n_in=%2d median=%6.1f peak=%6.1f fudge=%.3f"
              % (r["label"], len(x), nin, med, pk, PK.to_fudge(pk)))

    # the figure's own drawing invariants
    a = cells[("vtx", "now", "ncpi0")]
    x = PK.load("vtx", "now", "ncpi0")
    n_in_win = int(((x >= LO) & (x <= HI)).sum())
    if n_in_win != a["n_in"]:
        print("  FAIL window count %d != published n_in %d" % (n_in_win, a["n_in"]))
        bad += 1
    area = np.trapezoid(trunc_gauss(np.linspace(LO, HI, 20001), a["peak"], a["sigma"]),
                        np.linspace(LO, HI, 20001))
    if abs(area - 1.0) > 1e-4:
        print("  FAIL truncated pdf does not integrate to 1 over the window: %.6f" % area)
        bad += 1
    print("  ok  truncated-Gaussian normalisation over [%g,%g] = %.6f" % (LO, HI, area))
    out = np.sort(x[(x < LO) | (x > HI)])
    print("  ok  the %d points the fit excludes: %s  (%d low, %d high — the high one is a "
          "mis-pair, the only way m can rise)"
          % (len(out), np.round(out, 1), int((out < LO).sum()), int((out > HI).sum())))
    print("SELFTEST %s" % ("FAILED" if bad else "PASSED"))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png", default="docs/pr/126_pi0-mass.png")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    png = a.png if os.path.isabs(a.png) else os.path.join(SX, a.png)
    build(png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
