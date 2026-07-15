#!/usr/bin/env python3
"""Figures for docs/pdvd-crosser-flash-mismatch.md, from the run-wide records
written by crosser_common_velocity.py (crosser_flash_records.csv).

Fig 1  implied-velocity distribution (|t|-weighted), wrong-flash vs pick-ok,
       with nominal 0.1586 and FR 0.153 marked.
Fig 2  dt (true bright flash - picked flash) vs picked-flash time, with the
       velocity-error line (through origin) and constant-offset line -- the
       discriminator between "wrong velocity" and "wrong flash / time offset".

Run from pdvd/:  python3 ql_light_calib/crosser_flash_figures.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(HERE)
CSV = os.path.join(HERE, "crosser_flash_records.csv")
OUT = os.path.join(PDVD, "docs", "pds")
V_NOM = 0.1586
V_FR = 0.153


def load():
    rows = []
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(
                t_true=float(r["t_true"]), dt=float(r["dt"]),
                v_imp=float(r["v_imp"]), measCath=float(r["measCath"]),
                ks=float(r["ks"]), d_nom=float(r["d_nom"]),
                wrong=(r["wrong"] == "True"),
                len0=float(r["len0"]), len4=float(r["len4"])))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    wf = [r for r in rows if r["wrong"]]
    ok = [r for r in rows if not r["wrong"]]

    # Fig 1: implied-velocity histogram (rail hits at +-15% flagged separately)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0.134, 0.184, 26)
    ax.hist([r["v_imp"] for r in wf], bins=bins, alpha=.7, color="C3",
            label=f"wrong-flash (n={len(wf)})")
    ax.hist([r["v_imp"] for r in ok], bins=bins, alpha=.7, color="C0",
            label=f"pick already bright (n={len(ok)})")
    v = np.array([r["v_imp"] for r in rows]); w = np.array([abs(r["t_true"]) for r in rows])
    wmean = (v * w).sum() / w.sum()
    ax.axvline(V_NOM, color="k", ls="-", lw=1.5, label=f"nominal reco {V_NOM}")
    ax.axvline(V_FR, color="C2", ls="--", lw=1.5, label=f"field-response {V_FR}")
    ax.axvline(wmean, color="C1", ls=":", lw=2, label=f"|t|-wtd mean {wmean:.4f}")
    ax.set_xlabel("implied cathode-closure velocity  v = pi*/t_true  [cm/us]")
    ax.set_ylabel("crossers")
    ax.set_title("PDVD run 039252 cathode crossers: implied drift velocity "
                 "(rails at +-15% = short/foul pairs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pdvd-crosser-velocity-hist.png"), dpi=130)

    # Fig 2: dt vs picked-flash time -- velocity vs time-offset discriminator
    good = [r for r in rows if abs(r["v_imp"] / V_NOM - 1) < 0.14]
    tp = np.array([r["t_true"] - r["dt"] for r in good])
    dd = np.array([r["dt"] for r in good])
    A = np.vstack([tp, np.ones_like(tp)]).T
    (slope, inter), *_ = np.linalg.lstsq(A, dd, rcond=None)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter([r["t_true"] - r["dt"] for r in wf], [r["dt"] for r in wf],
               s=[max(8, r["measCath"] / 40) for r in wf], color="C3", alpha=.6,
               label="wrong-flash (area ~ bright cathode PE)")
    ax.scatter([r["t_true"] - r["dt"] for r in ok], [r["dt"] for r in ok],
               s=30, color="C0", marker="s", label="pick already bright (dt=0)")
    xs = np.array([min(tp), max(tp)])
    ax.plot(xs, slope * xs + inter, "C1-", lw=2,
            label=f"fit dt={slope:+.4f}*t{inter:+.0f}  (v={V_NOM/(1+slope):.4f})")
    ax.axhline(np.median([r["dt"] for r in wf]), color="C4", ls=":", lw=1.5,
               label=f"wrong-flash median dt {np.median([r['dt'] for r in wf]):+.0f}us")
    ax.axhline(0, color="gray", lw=.6)
    ax.set_xlabel("picked (display) flash time  [us]")
    ax.set_ylabel("dt = true bright flash - picked flash  [us]")
    ax.set_title("Is it velocity (slope, through 0) or a flash/time mis-pick "
                 "(flat offset)?")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pdvd-crosser-dt-vs-t.png"), dpi=130)
    print("wrote", OUT, "pdvd-crosser-{velocity-hist,dt-vs-t}.png")
    print(f"regression slope {slope:+.5f} intercept {inter:+.1f} "
          f"v_from_slope {V_NOM/(1+slope):.4f}")


if __name__ == "__main__":
    main()
