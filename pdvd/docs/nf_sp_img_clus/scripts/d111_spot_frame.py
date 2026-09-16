#!/usr/bin/env python3
"""doc pdvd/111 A1 -- where the Steiner cloud, the terminals and the STM fit sit relative to the IMAGE, per spot,
in one fit-independent frame, on the 2x2 lever arms (sampler x fit keys) that share one pctree.

Fork by duplication of the loaders of d110_spot_figs.py (untouched).  What is new:
  * the reference is the image only.  The spot frame is the charge-weighted PCA of the cluster's image points within
    R of the click (e1 along the track); the track is cut into slabs of SLAB cm along e1, and in each slab the image
    charge centroid and its transverse rms are the reference.  Nothing here takes a direction from the fit;
  * per slab, the Steiner cloud's transverse centroid offset from the image centroid ("moved") and its transverse rms
    over the image rms ("widened") are reported separately -- d110/earlier distances were containment (nearest image
    point), which a wider cloud improves mechanically;
  * the fit rows and the terminals get the same per-slab offset; plus a local ridge offset per point (image PCA within
    RLOC cm of the point, charge weighted), which follows a bend inside the window.

Inputs are Bee zips (clustering-global, steiner_graph-global, steiner_terminals-global) and tracking-stm.root
T_rec_charge (the persisted round-2 fit), all read-only.  An arm whose zip lacks a Steiner layer for the cluster
reports the cloud as n/a (PDVD production scopes those layers to tagged clusters).

Usage:
  d111_spot_frame.py --out figs/111_spot_frame      # writes <out>.tsv and <out>_<spot>.png
"""
import argparse, json, os, sys, zipfile
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
# spot -> (det, event, cluster, click, R, [(label, arm), ...])
SPOTS = [
    ("h1", "pdhd", "029107_16", 108, (75.3, 438.3, 450.1), 12.0,
     [("A0", "d101hnew"), ("K", "d101hkf"), ("S", "d102hocs"), ("A1", "d108hflip")]),
    ("h2", "pdhd", "029107_16", 106, (182.0, 537.4, 401.8), 24.0,
     [("A0", "d101hnew"), ("K", "d101hkf"), ("S", "d102hocs"), ("A1", "d108hflip")]),
    ("v1", "pdvd", "039349_20", 80, (313.7, 95.9, 124.9), 12.0,
     [("A0", "d103v0"), ("A1", "d103vflip")]),
    ("v2", "pdvd", "039349_20", 80, (94.5, 30.7, 165.0), 12.0,
     [("A0", "d103v0"), ("A1", "d103vflip")]),
    ("v3", "pdvd", "039349_20", 25, (-270.2, -243.7, 36.0), 12.0,
     [("A0", "d103v0"), ("A1", "d103vflip")]),
]
SLAB = 1.0      # cm, slab length along e1
RLOC = 3.0      # cm, local image PCA radius for the ridge offset
MIN_SLAB_PTS = 4


def bee(det, ev, arm, layer):
    z = zipfile.ZipFile(f"{IMG}/{det}/work/{ev}_{arm}/mabc-pr.zip")
    name = f"data/0/0-{layer}.json"
    if name not in z.namelist():
        return None
    d = json.loads(z.read(name))
    if not d.get("x"):
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0)
    q = np.array(d["q"], float) if "q" in d else np.ones(len(d["x"]))
    return np.c_[d["x"], d["y"], d["z"]], np.array(d["cluster_id"]), q


def fit_rows(det, ev, arm, cl):
    t = uproot.open(f"{IMG}/{det}/work/{ev}_{arm}/tracking-stm.root")["T_rec_charge"].arrays(
        ["cluster_id", "x", "y", "z", "q", "pass"], library="np")
    m = t["cluster_id"] == cl
    return np.c_[t["x"][m], t["y"][m], t["z"][m]], t["q"][m], t["pass"][m]


def frame(P, w):
    c = np.average(P, axis=0, weights=w)
    _, _, vt = np.linalg.svd((P - c) * np.sqrt(w / w.sum())[:, None], full_matrices=False)
    e1 = vt[0] / np.linalg.norm(vt[0])
    xh = np.array([1.0, 0, 0])
    e3 = xh - (xh @ e1) * e1
    if np.linalg.norm(e3) < 0.2:
        yh = np.array([0, 1.0, 0]); e3 = yh - (yh @ e1) * e1
    e3 /= np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    return c, e1, e2, e3


def ridge_offset(X, P, w, r=RLOC):
    """transverse distance of each X from the charge-weighted PCA line of the image points within r of X."""
    tree = cKDTree(P); out = np.full(len(X), np.nan)
    for i, x in enumerate(X):
        ii = tree.query_ball_point(x, r)
        if len(ii) < 5:
            continue
        Q, ww = P[ii], w[ii]
        c = np.average(Q, axis=0, weights=ww)
        _, _, vt = np.linalg.svd((Q - c) * np.sqrt(ww / ww.sum())[:, None], full_matrices=False)
        d = vt[0]; v = x - c
        out[i] = np.linalg.norm(v - (v @ d) * d)
    return out


def slab_stats(s_img, T_img, w_img, s_obj, T_obj, w_obj=None):
    """per slab: image centroid (2-D transverse) and rms; object centroid offset from it and object rms."""
    lo, hi = np.floor(s_img.min()), np.ceil(s_img.max())
    rows = []
    for a in np.arange(lo, hi, SLAB):
        mi = (s_img >= a) & (s_img < a + SLAB)
        mo = (s_obj >= a) & (s_obj < a + SLAB)
        if mi.sum() < MIN_SLAB_PTS:
            continue
        ci = np.average(T_img[mi], axis=0, weights=w_img[mi])
        ri = np.sqrt(np.average(np.sum((T_img[mi] - ci) ** 2, axis=1), weights=w_img[mi]))
        if mo.sum() == 0:
            rows.append((a, ri, np.nan, np.nan, 0)); continue
        wo = np.ones(mo.sum()) if w_obj is None else w_obj[mo]
        co = np.average(T_obj[mo], axis=0, weights=wo)
        ro = np.sqrt(np.average(np.sum((T_obj[mo] - co) ** 2, axis=1), weights=wo)) if mo.sum() > 1 else np.nan
        rows.append((a, ri, np.linalg.norm(co - ci), ro, int(mo.sum())))
    return np.array(rows, float).reshape(-1, 5)


def fmt(v, p=2):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cols = ["spot", "det", "event", "cluster", "label", "arm", "n_img", "img_rms_med",
            "n_steiner", "steiner_off_med", "steiner_off_max", "steiner_rms_over_img_med", "steiner_ridge_p90",
            "n_term", "term_off_med", "term_ridge_p90",
            "n_fit", "fit_off_med", "fit_off_max", "fit_ridge_p90", "fit_ridge_max", "fit_q_lt0"]
    lines = ["\t".join(cols)]
    for spot, det, ev, cl, click, R, arms in SPOTS:
        click = np.array(click)
        ref = None
        per_arm = []
        for label, arm in arms:
            img = bee(det, ev, arm, "clustering-global")
            Pi, ci, qi = img
            m = (ci == cl) & (np.linalg.norm(Pi - click, axis=1) < R)
            Pi, wi = Pi[m], np.clip(qi[m], 1, None)
            if ref is None:
                ref = frame(Pi, wi)       # the frame of the first arm's image; every arm shares the pctree/image
            c, e1, e2, e3 = ref
            def proj(X):
                V = X - c
                return V @ e1, np.c_[V @ e2, V @ e3]
            s_i, T_i = proj(Pi)
            rec = {"label": label, "arm": arm, "s_i": s_i, "T_i": T_i, "w_i": wi}
            out = dict(n_img=len(Pi))
            st_i = slab_stats(s_i, T_i, wi, s_i, T_i, wi)
            out["img_rms_med"] = np.nanmedian(st_i[:, 1]) if len(st_i) else np.nan
            for key, layer in (("steiner", "steiner_graph-global"), ("term", "steiner_terminals-global")):
                L = bee(det, ev, arm, layer)
                if L is None or len(L[0]) == 0 or not np.any(L[1] == cl):
                    out[f"n_{key}"] = 0; rec[key] = None
                    continue
                Ps = L[0][(L[1] == cl) & (np.linalg.norm(L[0] - click, axis=1) < R)]
                s_s, T_s = proj(Ps)
                st = slab_stats(s_i, T_i, wi, s_s, T_s)
                out[f"n_{key}"] = len(Ps)
                out[f"{key}_off_med"] = np.nanmedian(st[:, 2]) if len(st) else np.nan
                if key == "steiner":
                    out["steiner_off_max"] = np.nanmax(st[:, 2]) if len(st) else np.nan
                    out["steiner_rms_over_img_med"] = np.nanmedian(st[:, 3] / st[:, 1]) if len(st) else np.nan
                out[f"{key}_ridge_p90"] = np.nanpercentile(ridge_offset(Ps, Pi, wi), 90) if len(Ps) else np.nan
                rec[key] = (s_s, T_s)
            F, qf, pf = fit_rows(det, ev, arm, cl)
            mf = np.linalg.norm(F - click, axis=1) < R
            F, qf = F[mf], qf[mf]
            out["n_fit"] = len(F)
            if len(F):
                s_f, T_f = proj(F)
                st = slab_stats(s_i, T_i, wi, s_f, T_f)
                rf = ridge_offset(F, Pi, wi)
                out.update(fit_off_med=np.nanmedian(st[:, 2]), fit_off_max=np.nanmax(st[:, 2]),
                           fit_ridge_p90=np.nanpercentile(rf, 90), fit_ridge_max=np.nanmax(rf),
                           fit_q_lt0=int((qf < 0).sum()))
                rec["fit"] = (s_f, T_f, qf)
            else:
                rec["fit"] = None
            per_arm.append(rec)
            row = [spot, det, ev, str(cl), label, arm]
            for k in cols[6:]:
                v = out.get(k, np.nan)
                row.append(str(v) if isinstance(v, (int, np.integer)) else fmt(v))
            lines.append("\t".join(row))
            print("\t".join(row), flush=True)
        # figure: transverse coordinate e2 (top) and e3 (bottom) vs s, one column per arm
        n = len(per_arm)
        fig, axs = plt.subplots(2, n, figsize=(4.2 * n, 6.4), sharex=True, sharey="row", squeeze=False)
        for j, rec in enumerate(per_arm):
            for r in range(2):
                ax = axs[r, j]
                ax.scatter(rec["s_i"], rec["T_i"][:, r], s=4, c="0.75", label="image")
                if rec.get("steiner") is not None:
                    ax.scatter(rec["steiner"][0], rec["steiner"][1][:, r], s=5, c="tab:green", alpha=0.6, label="steiner_pc")
                if rec.get("term") is not None:
                    ax.scatter(rec["term"][0], rec["term"][1][:, r], s=22, marker="^", facecolor="none",
                               edgecolor="tab:orange", label="terminals")
                if rec.get("fit") is not None:
                    s_f, T_f, qf = rec["fit"]
                    ax.plot(s_f, T_f[:, r], ".", ms=4, c="tab:blue", label="STM fit (T_rec_charge)")
                    neg = qf < 0
                    ax.plot(s_f[neg], T_f[neg, r], "x", ms=6, c="red", label="fit q<0")
                if r == 0:
                    ax.set_title(f"{spot} {rec['label']} {rec['arm']}")
                ax.set_ylabel(["e2 (transverse, cm)", "e3 (drift-side transverse, cm)"][r])
                if r == 1:
                    ax.set_xlabel("s along image PCA (cm)")
        axs[0, 0].legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(f"{args.out}_{spot}.png", dpi=110)
        plt.close(fig)
    with open(f"{args.out}.tsv", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
