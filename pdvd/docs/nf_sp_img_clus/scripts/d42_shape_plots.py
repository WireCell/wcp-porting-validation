#!/usr/bin/env python3
"""doc pdvd/42 sec 7 -- figures for the mechanism decomposition of the STM fit's
2-D charge residual, from d42_shape_diag.py outputs.

Usage:
  d42_shape_plots.py --pdvd /tmp/d42diag_pdvd --sbnd /tmp/d42diag_sbnd \
                     --pdvd-clean /tmp/d42clean_pdvd --sbnd-clean /tmp/d42clean_sbnd \
                     --out figs/42
Figures:
  <out>_shape_window.png   B vs window radius (Chebyshev cells and matched mm) -- the windowing control
  <out>_shape_profile.png  transverse charge profile, measured vs predicted, per plane per detector (H1)
  <out>_shape_drift.png    the transverse width the model does NOT reproduce, vs drift band (H1 split)
  <out>_shape_angle.png    theta_P distribution, B and charge_err/charge vs theta_P (H3)
  <out>_shape_h2.png       B vs f_off_far quartile, unfused vs all, residual concentration (H2)
  <out>_shape_sigma.png    the bias B against the model's transverse sigma in WIRE-PITCH units --
                           one monotone curve through all 6 (detector, plane) points
"""
import argparse, csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"pdvd": "tab:red", "sbnd": "tab:blue"}
PL = "UVW"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def prof_rms(rows, key):
    c = np.array([float(r["centre_mm"]) for r in rows])
    w = np.array([float(r[key]) for r in rows])
    if w.sum() <= 0:
        return float("nan")
    mu = np.average(c, weights=w)
    return float(np.sqrt(max(np.average((c - mu) ** 2, weights=w), 0.0)))


def fig_window(A, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for det, pre in A.items():
        w = load(pre + "_window.tsv")
        for k, (kind, xlab) in enumerate((("cheb", "Chebyshev window radius [cells]"),
                                          ("phys_mm", "matched window radius [mm], |Δt| ≤ 1 slice"))):
            for p, ls in zip(PL, ("-", "--", ":")):
                rr = [r for r in w if r["plane"] == p and r["kind"] == kind]
                x = [float(r["radius"]) for r in rr]
                y = [float(r["B"]) for r in rr]
                ax[k].plot(x, y, ls, color=C[det], marker="o", ms=3,
                           label="%s %s" % (det.upper(), p))
            ax[k].set_xlabel(xlab)
    for k in (0, 1):
        ax[k].set_ylabel("B  (Σŷ − Σy)/Σy")
        ax[k].axhline(0, color="k", lw=0.5)
        ax[k].grid(alpha=.3)
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title("the ±1-cell footprint holds only 87–91 % of the U/V prediction\n"
                    "(99.8 % of W's): the plane gap is not a windowing artifact", fontsize=8)
    ax[1].set_title("matched physical windows: PDVD U/V ≈ −0.20 vs W ≈ −0.10,\nSBND all three ≈ −0.08", fontsize=8)
    fig.tight_layout(); fig.savefig(out + "_shape_window.png", dpi=135); plt.close(fig)


def fig_profile(A, out):
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    for i, det in enumerate(("pdvd", "sbnd")):
        p_ = load(A[det] + "_profile.tsv")
        for j, p in enumerate(PL):
            rr = [r for r in p_ if r["plane"] == p and r["band"] == "all" and r["axis"] == "perp"]
            c = np.array([float(r["centre_mm"]) for r in rr])
            y = np.array([float(r["sum_y"]) for r in rr]); h = np.array([float(r["sum_yhat"]) for r in rr])
            a = ax[i, j]
            a.step(c, y / y.max(), where="mid", color="k", lw=1.4, label="measured")
            a.step(c, h / y.max(), where="mid", color=C[det], lw=1.4, label="predicted")
            a.fill_between(c, h / y.max(), y / y.max(), step="mid", color=C[det], alpha=.15)
            ry, rh = prof_rms(rr, "sum_y"), prof_rms(rr, "sum_yhat")
            a.set_title("%s %s   rms %.2f vs %.2f mm\nunmodelled √(Δrms²) = %.2f mm"
                        % (det.upper(), p, ry, rh, np.sqrt(max(ry * ry - rh * rh, 0))), fontsize=8)
            a.set_yscale("log"); a.set_ylim(2e-3, 1.6); a.grid(alpha=.3)
            if j == 0:
                a.set_ylabel("charge / peak measured")
            if i == 1:
                a.set_xlabel("distance from the trajectory, pitch direction [mm]")
            if i == 0 and j == 0:
                a.legend(fontsize=7)
    fig.suptitle("Transverse charge profile about the fitted trajectory: the model is too narrow on every plane,\n"
                 "and far too narrow on PDVD's induction planes (H1)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(out + "_shape_profile.png", dpi=135); plt.close(fig)


def fig_drift(A, Ac, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for k, (src, ttl) in enumerate(((A, "all accepted blocks"), (Ac, "unfused blocks only (f_off < 0.15)"))):
        for det in ("pdvd", "sbnd"):
            if det not in src:
                continue
            p_ = load(src[det] + "_profile.tsv")
            for p, mk in zip(PL, ("o", "s", "^")):
                xs, ys = [], []
                for b, lab in ((f"xband{t}", t) for t in range(3)):
                    rr = [r for r in p_ if r["plane"] == p and r["band"] == b and r["axis"] == "perp"]
                    if not rr or sum(float(r["sum_y"]) for r in rr) <= 0:
                        continue
                    ry, rh = prof_rms(rr, "sum_y"), prof_rms(rr, "sum_yhat")
                    xs.append(lab); ys.append(np.sqrt(max(ry * ry - rh * rh, 0)))
                ax[k].plot(xs, ys, marker=mk, color=C[det], label="%s %s" % (det.upper(), p))
        ax[k].set_title(ttl, fontsize=9)
        ax[k].set_xlabel("drift band  (0 = shortest drift  →  2 = longest)")
        ax[k].set_xticks([0, 1, 2]); ax[k].grid(alpha=.3)
    ax[0].set_ylabel("transverse width the model misses [mm]")
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle("H1 split: the missing width GROWS with drift (a diffusion term) but does not start at zero\n"
                 "(a drift-independent SP-filter term) — and PDVD U/V miss 2–3 mm against SBND's ~1 mm", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .90)); fig.savefig(out + "_shape_drift.png", dpi=135); plt.close(fig)


def fig_angle(A, out):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    ctr = [7.5, 22.5, 37.5, 52.5, 67.5, 82.5]
    for det in A:
        a_ = load(A[det] + "_angle.tsv")
        for p, ls in zip(PL, ("-", "--", ":")):
            rr = [r for r in a_ if r["plane"] == p]
            n = np.array([float(r["N"]) for r in rr]); x = [float(r["theta_lo"]) + 7.5 for r in rr]
            if p == "W":
                ax[0].plot(x, 100 * n / n.sum(), "-o", color=C[det], ms=4, label="%s (W plane)" % det.upper())
            ax[1].plot(x, [float(r["B"]) for r in rr], ls, color=C[det], marker="o", ms=3, label="%s %s" % (det.upper(), p))
            ax[2].plot(x, [float(r["rel_err"]) for r in rr], ls, color=C[det], marker="o", ms=3)
    ax[0].set_ylabel("% of charge"); ax[0].set_title("PDVD is prolonged, SBND is isochronous", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].set_ylabel("B"); ax[1].axhline(0, color="k", lw=.5)
    ax[1].set_title("the bias is FLAT in θ — H3 does not drive it", fontsize=9); ax[1].legend(fontsize=6, ncol=2)
    ax[2].set_ylabel("charge_err / charge"); ax[2].set_title("but the induction uncertainty doubles\nwith prolongedness on PDVD", fontsize=9)
    for k in range(3):
        ax[k].set_xlabel("θ$_P$  [deg]   (0 = isochronous, 90 = prolonged)"); ax[k].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out + "_shape_angle.png", dpi=135); plt.close(fig)


def fig_h2(A, Ac, out):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    width = 0.35
    for i, det in enumerate(A):
        b = load(A[det] + "_block.tsv")
        for j, p in enumerate(PL):
            v = np.array([(float(r["ffar_" + p]), float(r["B_" + p]), float(r["U_" + p]),
                           float(r["Uscaled_" + p]), float(r["top1_" + p]))
                          for r in b if r.get("B_" + p)])
            if len(v) < 10:
                continue
            qs = np.quantile(v[:, 0], [0, .25, .5, .75, 1.0])
            med = [np.median(v[(v[:, 0] >= qs[k]) & (v[:, 0] <= qs[k + 1]), 1]) for k in range(4)]
            ax[0].plot([1, 2, 3, 4], med, marker="os^"[j], color=C[det], ls=("-", "--", ":")[j],
                       label="%s %s" % (det.upper(), p))
            ax[1].bar(i * 3 + j + width * 0, np.median(v[:, 2]), width, color=C[det], alpha=.55,
                      label="U" if (i == 0 and j == 0) else None)
            ax[1].bar(i * 3 + j + width, np.median(v[:, 3]), width, color=C[det],
                      label="after a free per-plane scale" if (i == 0 and j == 0) else None)
            ax[2].bar(i * 3 + j, np.median(v[:, 4]), .6, color=C[det])
    ax[0].set_xticks([1, 2, 3, 4]); ax[0].set_xlabel("f_off_far quartile (fusion / busy-ness)")
    ax[0].set_ylabel("median B"); ax[0].set_title("B does not track fusion — H2 is not the cause", fontsize=9)
    ax[0].legend(fontsize=6, ncol=2); ax[0].grid(alpha=.3)
    for k, ttl, yl in ((1, "a free scale removes almost none of it:\nthe residual is SHAPE, not normalisation", "median U"),
                       (2, "the residual is not concentrated:\ntop 1 % of cells hold < 12 % of Σ|y−ŷ|", "top-1 % share")):
        ax[k].set_xticks(range(6)); ax[k].set_xticklabels(["PDVD U", "PDVD V", "PDVD W", "SBND U", "SBND V", "SBND W"], fontsize=7)
        ax[k].set_title(ttl, fontsize=9); ax[k].set_ylabel(yl); ax[k].grid(alpha=.3, axis="y")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out + "_shape_h2.png", dpi=135); plt.close(fig)


CFG = {"pdvd": dict(DT=7.9135e-7, v=0.148073, ind=(0.259, 0.4316, 0.0575), pitch=(7.65, 7.65, 5.10)),
       "sbnd": dict(DT=8.8e-7, v=0.1563, ind=(0.48359, 0.80599, 0.09403), pitch=(3.0, 3.0, 3.0))}


def fig_sigma(A, out):
    """B vs the model's transverse sigma expressed in wire pitches.

    sigma_model = hypot(sqrt(2*DT*t_drift), ind_sigma_<p>_T) / pitch_<p>, evaluated at
    the sample's median |x| -- exactly TrackFitting.cxx:7310-7312.  The pitch division is
    the point: the same physical spread covers 2.5x fewer PDVD U/V wires than SBND wires.
    """
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for det in A:
        b = load(A[det] + "_block.tsv"); w = load(A[det] + "_window.tsv")
        absx = np.median([float(r["absx_cm"]) for r in b])
        c = CFG[det]
        diff = np.sqrt(2 * c["DT"] * (absx / c["v"] * 1e3))
        for j, p in enumerate(PL):
            sig = np.hypot(diff, c["ind"][j]) / c["pitch"][j]
            rr = [r for r in w if r["plane"] == p and r["kind"] == "cheb" and r["radius"] == "2"][0]
            for k, key in enumerate(("B", "U")):
                v = float(rr[key])
                ax[k].plot(sig, v, marker="os^"[j], color=C[det], ms=9)
                ax[k].annotate("%s %s" % (det.upper(), p), (sig, v), textcoords="offset points",
                               xytext=(7, 4), fontsize=7)
    for k, yl, ttl in ((0, "B   (Σŷ − Σy)/Σy", "the whole plane/detector pattern is one curve in σ/pitch"),
                       (1, "U   Σ|y−ŷ|/Σy", "and so is the unexplained fraction")):
        ax[k].set_xlabel("model transverse σ  [wire pitches]  at the sample's median drift")
        ax[k].set_ylabel(yl); ax[k].grid(alpha=.3); ax[k].set_title(ttl, fontsize=9)
    fig.suptitle("The model spreads charge over too few wires — and PDVD's 7.65 mm U/V pitch makes that worst there\n"
                 "(circle U, square V, triangle W; red PDVD, blue SBND)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .90)); fig.savefig(out + "_shape_sigma.png", dpi=135); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdvd", required=True); ap.add_argument("--sbnd", required=True)
    ap.add_argument("--pdvd-clean", required=True); ap.add_argument("--sbnd-clean", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A = {"pdvd": a.pdvd, "sbnd": a.sbnd}
    Ac = {"pdvd": a.pdvd_clean, "sbnd": a.sbnd_clean}
    fig_window(A, a.out); fig_profile(A, a.out); fig_drift(A, Ac, a.out)
    fig_angle(A, a.out); fig_h2(A, Ac, a.out); fig_sigma(A, a.out)
    print("wrote %s_shape_{window,profile,drift,angle,h2,sigma}.png" % a.out)


if __name__ == "__main__":
    sys.exit(main())
