#!/usr/bin/env python3
"""doc pdvd/42 -- population plots of the STM fit's measured-vs-predicted 2-D
charge, PDVD against SBND, from d42_proj2d_resid.py outputs.

Usage:
  d42_proj2d_plots.py --pdvd figs/42_proj2d_pdvd --sbnd figs/42_proj2d_sbnd --out figs/42 [--status 0]
Figures (PNG):
  <out>_proj2d_dists.png    U_foot, B_foot, chi2/N, uncov_foot per plane, PDVD vs SBND (accepted passes)
  <out>_proj2d_coverage.png f_off (charge the trajectory does not reach) per plane + vs track length
  <out>_proj2d_pulls.png    pull histograms per plane, both detectors, with the robust width
  <out>_proj2d_vs_rr.png    U and B on the footprint vs residual-range bin, per plane
  <out>_proj2d_vs_x.png     PDVD U_foot / B_foot / f_off vs median |x| of the block (drift), all planes
Prints the summary table (median / IQR per detector x plane) as markdown.
"""
import argparse, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLS = {"pdvd": "tab:red", "sbnd": "tab:blue"}
RR_ORDER = ["0-2", "2-5", "5-10", "10-20", "20-40", "40+"]


def read_tsv(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path) if not l.startswith("#")]
    hdr = rows[0]; data = rows[1:]
    out = {h: [] for h in hdr}
    for r in data:
        for h, v in zip(hdr, r): out[h].append(v)
    def col(h, f=float):
        return np.array([f(v) if v not in ("nan", "") else np.nan for v in out[h]]) if f is float else np.array(out[h])
    return out, col


def q(v):
    v = v[np.isfinite(v)]
    if len(v) == 0: return (np.nan, np.nan, np.nan, 0)
    return (np.percentile(v, 25), np.median(v), np.percentile(v, 75), len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdvd", required=True); ap.add_argument("--sbnd", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--status", type=int, default=0)
    a = ap.parse_args()
    D = {}
    for det, pre in (("pdvd", a.pdvd), ("sbnd", a.sbnd)):
        raw, col = read_tsv(pre + "_blocks.tsv")
        st = col("status"); pl = col("plane", str)
        sel = (st == a.status)
        D[det] = dict(plane=pl[sel], U=col("U_foot")[sel], B=col("B_foot")[sel], chi2=col("chi2N_foot")[sel], unc=col("uncov_foot")[sel],
                      foff=col("f_off")[sel], L=col("length_cm")[sel], prms=col("pull_rms")[sel], ptail=col("pull_tail3")[sel],
                      absx=col("absx_med")[sel], Uall=col("U_all")[sel], Nf=col("Nfoot")[sel], ev=col("event", str)[sel], blk=col("block")[sel],
                      pulls=np.load(pre + "_pulls.npz"))
        rr_raw, rc = read_tsv(pre + "_rr.tsv")
        D[det]["rr"] = dict(st=rc("status"), plane=rc("plane", str), bin=rc("rr_bin", str), ab=rc("abs"), sy=rc("sum_y"), sp=rc("sum_yhat"), un=rc("uncov_y"))

    # ---- summary table
    print("| det | plane | blocks | U_foot med [IQR] | B_foot med | chi2/N med | uncov_foot med | f_off med [IQR] | pull rms med | |pull|>3 med |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for det in ("pdvd", "sbnd"):
        d = D[det]
        for P in ("U", "V", "W", "ALL"):
            m = d["plane"] == P
            qu = q(d["U"][m]); qb = q(d["B"][m]); qc = q(d["chi2"][m]); qn = q(d["unc"][m]); qf = q(d["foff"][m]); qp = q(d["prms"][m]); qt = q(d["ptail"][m])
            print("| %s | %s | %d | %.3f [%.3f, %.3f] | %+.3f | %.1f | %.3f | %.3f [%.3f, %.3f] | %.2f | %.3f |" % (
                det, P, qu[3], qu[1], qu[0], qu[2], qb[1], qc[1], qn[1], qf[1], qf[0], qf[2], qp[1], qt[1]))

    # ---- fig 1: distributions per plane
    fig, ax = plt.subplots(4, 3, figsize=(13, 12))
    specs = [("U", "U_foot = sum|y-yhat|/sum y on footprint", (0, 1.2)), ("B", "B_foot = (sum yhat - sum y)/sum y", (-1, 1)),
             ("chi2", "chi2/N on footprint (fit's own sigma)", (0, 60)), ("unc", "uncov_foot = charge with yhat=0 on footprint", (0, 0.6))]
    for i, (key, lab, rng) in enumerate(specs):
        for j, P in enumerate("UVW"):
            for det in ("pdvd", "sbnd"):
                d = D[det]; m = d["plane"] == P; v = d[key][m]; v = v[np.isfinite(v)]
                ax[i, j].hist(np.clip(v, rng[0], rng[1]), bins=40, range=rng, histtype="step", color=COLS[det], lw=1.6,
                              label="%s n=%d med %.3f" % (det.upper(), len(v), np.median(v) if len(v) else np.nan), density=True)
            ax[i, j].set_title("%s plane" % P if i == 0 else ""); ax[i, j].set_xlabel(lab if j == 1 else ""); ax[i, j].legend(fontsize=8)
    fig.suptitle("STM accepted passes (status 0): 2-D charge residual per plane, PDVD vs SBND"); fig.tight_layout()
    fig.savefig(a.out + "_proj2d_dists.png", dpi=110); plt.close(fig)

    # ---- fig 2: coverage
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for j, P in enumerate("UVW"):
        for det in ("pdvd", "sbnd"):
            d = D[det]; m = d["plane"] == P; v = d["foff"][m]; v = v[np.isfinite(v)]
            ax[j].hist(v, bins=40, range=(0, 1), histtype="step", color=COLS[det], lw=1.6, density=True,
                       label="%s n=%d med %.3f" % (det.upper(), len(v), np.median(v) if len(v) else np.nan))
        ax[j].set_title("%s plane: f_off (charge > 1 wire/slice from the trajectory)" % P, fontsize=9); ax[j].legend(fontsize=8)
    for det in ("pdvd", "sbnd"):
        d = D[det]; m = d["plane"] == "ALL"
        ax[3].scatter(d["L"][m], d["foff"][m], s=8, alpha=0.5, color=COLS[det], label=det.upper())
    ax[3].set_xlabel("track length (cm)"); ax[3].set_ylabel("f_off (all planes)"); ax[3].legend(); ax[3].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(a.out + "_proj2d_coverage.png", dpi=110); plt.close(fig)

    # ---- fig 3: pulls
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for j, P in enumerate("UVW"):
        for det in ("pdvd", "sbnd"):
            v = D[det]["pulls"][P]
            if len(v) == 0: continue
            rms = 1.4826 * np.median(np.abs(v - np.median(v)))
            ax[j].hist(np.clip(v, -10, 10), bins=100, range=(-10, 10), histtype="step", color=COLS[det], lw=1.5, density=True,
                       label="%s n=%d med %.2f robust rms %.2f |>3| %.1f%%" % (det.upper(), len(v), np.median(v), rms, 100 * (np.abs(v) > 3).mean()))
        ax[j].set_yscale("log"); ax[j].set_title("%s plane: pull (y - yhat)/sigma, covered footprint cells" % P, fontsize=9); ax[j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(a.out + "_proj2d_pulls.png", dpi=110); plt.close(fig)

    # ---- fig 4: vs rr
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    print("\n| det | plane | " + " | ".join("U %s" % b for b in RR_ORDER) + " | " + " | ".join("B %s" % b for b in RR_ORDER) + " |")
    print("|---|---|" + "---|" * (2 * len(RR_ORDER)))
    for det in ("pdvd", "sbnd"):
        r = D[det]["rr"]; ms = r["st"] == a.status
        for j, P in enumerate("UVW"):
            Us, Bs = [], []
            for b in RR_ORDER:
                m = ms & (r["plane"] == P) & (r["bin"] == b)
                sy = r["sy"][m].sum(); Us.append(r["ab"][m].sum() / sy if sy > 0 else np.nan); Bs.append((r["sp"][m].sum() - sy) / sy if sy > 0 else np.nan)
            xs = np.arange(len(RR_ORDER))
            ax[0, j].plot(xs, Us, "o-", color=COLS[det], label=det.upper()); ax[1, j].plot(xs, Bs, "o-", color=COLS[det], label=det.upper())
            ax[0, j].set_title("%s plane" % P); ax[1, j].set_xticks(xs); ax[1, j].set_xticklabels(RR_ORDER); ax[1, j].set_xlabel("residual range bin (cm)")
            print("| %s | %s | " % (det, P) + " | ".join("%.3f" % u for u in Us) + " | " + " | ".join("%+.3f" % b for b in Bs) + " |")
    ax[0, 0].set_ylabel("U (pooled charge-weighted)"); ax[1, 0].set_ylabel("B (pooled)"); ax[0, 0].legend()
    for k_ in range(3): ax[1, k_].axhline(0, color="k", lw=0.5)
    fig.suptitle("footprint residual vs residual range (accepted passes)"); fig.tight_layout(); fig.savefig(a.out + "_proj2d_vs_rr.png", dpi=110); plt.close(fig)

    # ---- fig 5: PDVD vs |x|
    d = D["pdvd"]; m = d["plane"] == "ALL"
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for k_, (key, lab) in enumerate((("U", "U_foot"), ("B", "B_foot"), ("foff", "f_off"))):
        ax[k_].scatter(d["absx"][m], d[key][m], s=10, alpha=0.6, color=COLS["pdvd"])
        xb = np.arange(0, 340, 40); cen = []; med = []
        for lo in xb:
            mm = m & (d["absx"] >= lo) & (d["absx"] < lo + 40) & np.isfinite(d[key])
            if mm.sum() >= 3: cen.append(lo + 20); med.append(np.median(d[key][mm]))
        ax[k_].plot(cen, med, "k-o", ms=4, label="median per 40 cm")
        ax[k_].set_xlabel("median |x| of the block (cm)"); ax[k_].set_ylabel(lab); ax[k_].legend()
    fig.suptitle("PDVD accepted passes: 2-D residual vs drift position (all planes pooled)"); fig.tight_layout()
    fig.savefig(a.out + "_proj2d_vs_x.png", dpi=110); plt.close(fig)
    print("\nwrote", a.out + "_proj2d_{dists,coverage,pulls,vs_rr,vs_x}.png")


if __name__ == "__main__":
    sys.exit(main())
