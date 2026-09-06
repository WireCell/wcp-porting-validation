#!/usr/bin/env python3
"""doc pdhd/04 sec 8 figure: 3-D image vs 2-D measurement for cluster 128 of
029107 evt 12 (art event 1079).

Inputs are the WCT_TGM_PATH_DUMP CSVs written by TaggerCheckTGM::path_components
(env-gated probe, doc pdhd/04 sec 7, extended in sec 8).  Two arms:

  OFF  the production pctree (pdhd/wct-clustering.jsonnet wrapped_channel_charge
       = false), work/029107_12_d05p
  ON   the same imaging re-clustered with PDHD_CLUS_TLA="-S wrapped_channel_charge=true",
       work/029107_12_d05wc

Usage:
  d05_cluster128_wrapped.py <off_dir> <on_dir> <out.png>
"""
import sys, csv, glob, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# PDHD wrapped-strip stripe edges in the plane's own wire-index order:
# 400 segment-0, 400 segment-1, 348 segment-2 (protodunehd-wires-larsoft-v1);
# V is U reversed, so V's edges sit 52 wires lower.
SEG_EDGES = {"u": (400, 800), "v": (348, 748)}

def segment(pl, w):
    lo, hi = SEG_EDGES[pl]
    return 0 if w < lo else (1 if w < hi else 2)

def load(path):
    return list(csv.DictReader(open(path)))

def occ(rows, key, thr=10.0):
    return np.array([float(r[key]) > thr for r in rows])

def main(off_dir, on_dir, out):
    off = load(os.path.join(off_dir, "tgm_path_cluster128.csv"))
    on  = load(os.path.join(on_dir,  "tgm_path_cluster128.csv"))
    z   = np.array([float(r["z"]) for r in off])
    x   = np.array([float(r["x"]) for r in off])
    ex  = np.array([r["excluded"] == "1" for r in off])
    exn = np.array([r["excluded"] == "1" for r in on])

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.6))

    # ---- A: occupancy vs z -------------------------------------------------
    ax = axes[0][0]
    edges = np.arange(0, 480, 12.0)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(z, edges) - 1
    def prof(mask):
        out = np.full(len(ctr), np.nan)
        for b in range(len(ctr)):
            s = idx == b
            if s.sum() >= 5:
                out[b] = mask[s].mean()
        return out
    # the three 2-D curves sit on top of each other at ~1.0; draw them as one
    # band so the statement "the measurement is there in all three views" is
    # legible instead of three coincident dashes.
    m2 = np.vstack([prof(np.array([r[pl + "hcp"] == "1" for r in off])) for pl in "uvw"])
    ax.fill_between(ctr, np.nanmin(m2, axis=0), 1.055, color="#4c72b0", alpha=0.13, lw=0)
    ax.plot(ctr, np.nanmin(m2, axis=0), color="#33507a", lw=2.4, ls="--",
            label="2-D measurement present (worst of U, V, W)")
    ax.fill_between(ctr, 0, prof(ex), color="0.55", alpha=0.32, lw=0,
                    label="excluded by is_point_good")
    for pl, col in zip("uvw", ["#d62728", "#1f77b4", "#2ca02c"]):
        ax.plot(ctr, prof(occ(off, "q" + pl)), color=col, lw=2,
                label=f"{pl.upper()}  sampled per-point charge")
    ax.set_xlabel("z  [cm]"); ax.set_ylabel("fraction of points with charge")
    ax.set_ylim(-0.03, 1.08); ax.set_xlim(0, 470)
    ax.set_title("A.  Cluster 128: what the 2-D data holds vs what the point carries")
    ax.legend(fontsize=7.8, ncol=2, loc="lower center", framealpha=0.94)
    ax.grid(alpha=0.25)

    # ---- B: wire index vs z, with the segment stripes ----------------------
    ax = axes[0][1]
    for pl, mk, col in (("u", "o", "#d62728"), ("v", "s", "#1f77b4")):
        w = np.array([int(r["w" + pl] if "w" + pl in r else r[pl + "w3"]) for r in off])
        good = occ(off, "q" + pl)
        ax.scatter(z[good], w[good], s=2.0, marker=mk, color=col, alpha=0.55,
                   label=f"{pl.upper()}  sampled charge > 0")
        ax.scatter(z[~good], w[~good], s=2.0, marker=mk, color=col, alpha=0.16)
    lo_u, hi_u = SEG_EDGES["u"]; lo_v, hi_v = SEG_EDGES["v"]
    ax.axhspan(lo_v, hi_u, color="0.75", alpha=0.28, lw=0)
    ax.axhline(lo_u, color="#d62728", ls=":", lw=1.1)
    ax.axhline(hi_u, color="#d62728", ls=":", lw=1.1)
    ax.axhline(lo_v, color="#1f77b4", ls=":", lw=1.1)
    ax.axhline(hi_v, color="#1f77b4", ls=":", lw=1.1)
    ax.text(6, 0.5 * (lo_v + hi_u), "segment-1 stripe\n(wrapped continuations)",
            fontsize=8.4, va="center", color="0.25")
    ax.set_xlabel("z  [cm]"); ax.set_ylabel("wire index in the plane (0-1147)")
    ax.set_xlim(0, 470); ax.set_ylim(0, 1150)
    ax.set_title("B.  The zeros are exactly the segment-1 wires")
    ax.legend(fontsize=8, loc="lower right", markerscale=4, framealpha=0.92)
    ax.grid(alpha=0.2)

    # ---- C: per (volume, plane, segment) --------------------------------
    ax = axes[1][0]
    c = collections.defaultdict(lambda: [0, 0, 0])
    for r in off:
        for pl in "uv":
            k = (f"a{r['apa']}f{r['face']}", pl.upper(), segment(pl, int(r["w3".join(('', ''))] if False else r[pl + "w3"])))
            d = c[k]; d[0] += 1
            d[1] += float(r["q" + pl]) > 10
            d[2] += r[pl + "hcp"] == "1"
    keys = sorted(c, key=lambda k: (k[0], k[1], k[2]))
    xs = np.arange(len(keys))
    samp = [c[k][1] / c[k][0] for k in keys]
    meas = [c[k][2] / c[k][0] for k in keys]
    ax.bar(xs - 0.2, meas, 0.4, color="#7f7f7f", label="2-D measurement present")
    ax.bar(xs + 0.2, samp, 0.4,
           color=["#d62728" if k[2] == 1 else "#2ca02c" for k in keys],
           label="sampled per-point charge > 0")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k[0]}\n{k[1]} seg{k[2]}\nn={c[k][0]}" for k in keys], fontsize=7.2)
    ax.set_ylabel("fraction of points"); ax.set_ylim(0, 1.14)
    ax.set_title("C.  Segment 1 fails in every volume -- APA0 is not special")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    # ---- D: the causal control ------------------------------------------
    ax = axes[1][1]
    ax.scatter(z[~ex], x[~ex], s=1.6, color="#2ca02c", alpha=0.5, label="OFF: walked")
    ax.scatter(z[ex], x[ex], s=1.6, color="#d62728", alpha=0.6, label="OFF: excluded (3542)")
    zo = np.array([float(r["z"]) for r in on]); xo = np.array([float(r["x"]) for r in on])
    ax.scatter(zo[exn], xo[exn] , s=26, facecolor="none", edgecolor="k", lw=1.1,
               label="ON: excluded (14)")
    ax.set_xlabel("z  [cm]"); ax.set_ylabel("x  [cm]  (T0-corrected)")
    ax.set_title("D.  Re-clustered with wrapped_channel_charge = true")
    ax.legend(fontsize=8, loc="upper left", markerscale=2.6, framealpha=0.92)
    ax.grid(alpha=0.25)
    n_off = len(off); n_on = len(on)
    m_off = collections.Counter(sum(float(r["q" + l]) > 10 for l in "uvw") for r in off)
    m_on = collections.Counter(sum(float(r["q" + l]) > 10 for l in "uvw") for r in on)
    ax.text(0.985, 0.03,
            f"planes with charge   OFF -> ON\n"
            f"   3 : {m_off[3]/n_off:5.3f} -> {m_on[3]/n_on:5.3f}\n"
            f"   2 : {m_off[2]/n_off:5.3f} -> {m_on[2]/n_on:5.3f}\n"
            f"   1 : {m_off[1]/n_off:5.3f} -> {m_on[1]/n_on:5.3f}\n"
            f"TGM=false -> TGM=true",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.42", fc="w", ec="0.6", alpha=0.94))

    fig.suptitle("PDHD 029107 evt 12 (art 1079) cluster 128 -- the 3-D image is fully "
                 "supported in 2-D; the per-point charge is not attached",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=135)
    print("wrote", out)

if __name__ == "__main__":
    main(*sys.argv[1:4])
