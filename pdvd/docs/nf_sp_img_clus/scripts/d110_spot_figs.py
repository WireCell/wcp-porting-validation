#!/usr/bin/env python3
"""doc pdvd/110 -- figures and metrics for the owner's Bee spots on the stm_fit layer (gaps, zig-zag),
pre-flip arm vs current production, both detectors.

Fork by duplication of d102_spot_figs.py (untouched, so figs/102_* stay reproducible).  Differences:
  * spots and arms for both detectors (SPOTS below), with a per-spot window half-size R;
  * a new <out>_<spot>_bee.png: the persisted STM fit (every T_rec_charge row) beside what Bee actually
    draws from the stm_fit layer, and the clamped track_fit layer as the control, coloured with Bee's own
    charge palette;
  * the Bee drop test is strict q < 0: Bee's sst.js initData skips `data.q[i] < QTHRESH` with QTHRESH = 0
    for every layer but truth/L1, so q == 0 is still drawn (d102 used q <= 0);
  * the profile adds the chord deviation (wiggle), the per-plane pixel separation per step
    hypot(d wire, d slice) -- the degeneracy measure -- and folds (steps that point backward on the local
    +-5-row chord);
  * the zip stm_fit layer is asserted equal to T_rec_charge row for row (q within 0.01, x within 0.001 cm)
    whenever the cluster is in the layer, which ties what Bee shows to the ROOT rows;
  * d102's in-slice grid figure is dropped.

Outputs per spot: <out>_<spot>_{bee,3d,2d,profile}.png; per detector <out>_<det>_spots.tsv.

Facts this relies on (checked for doc 110 on PDVD 039349_20 cl25 and PDHD 029107_16 cl108, both arms):
  * Bee stm_fit / track_fit q == T_rec_charge q = dQ*0.1 - 1000 (Trun dQdx_scale / dQdx_offset);
    track_fit clamps q < 0 to 0 (MultiAlgBlobClustering.cxx:1181), stm_fit does not (:3130);
  * T_proj_data.time_slice and T_rec_charge.pt are both in time-slice units (unique-slice step 1);
  * PDVD has no verified raw -> rank channel map here (DETS["pdvd"]), so its dead channels are not drawn;
    the per-row reg_flag_u/v/w carries the dead-cell information instead.

Usage:
  d110_spot_figs.py --det pdvd --out figs/110
  d110_spot_figs.py --det pdhd --out figs/110
"""
import argparse, colorsys, json, os, sys, zipfile
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
DETS = {
    # raw channel -> rank: base[plane] + apa*nper[plane] + (raw % nch_apa - start[plane])
    "pdhd": dict(base=(0, 3200, 6400), nper=(800, 800, 960), start=(0, 800, 1600), nch_apa=2560),
    "pdvd": dict(base=(0, 3808, 7616), nper=None, start=None, nch_apa=None),
}
# det -> (event, "pre-flip arm,production arm", [name:cluster:x,y,z:R])
SPOTS = {
    "pdvd": ("039349_20", "q29flip,d103vflip",
             ["v1:80:313.7,95.9,124.9:12", "v2:80:94.5,30.7,165.0:12", "v3:25:-270.2,-243.7,36.0:12"]),
    "pdhd": ("029107_16", "d101hnew,d109hstm",
             ["h1:108:75.3,438.3,450.1:12", "h2:106:182.0,537.4,401.8:24"]),
}
R = 12.0        # cm, window half-size around the click (set per spot in main)
PLANES = ("pu", "pv", "pw")
PASS_COLORS = ("tab:blue", "tab:green", "tab:purple", "tab:brown")
BEE_QMAX = 14000 * 2 / 3      # sst.js: getColorAtScalar(q, colorScale^2 * 14000 * 2/3), colorScale 1.0


def bee_rgb(q):
    """Bee's charge colour: hue = (max - q)/max * 2/3 for q <= max, else 0; HSL(hue, 1, 0.5)."""
    q = np.atleast_1d(np.asarray(q, float))
    h = np.where(q <= BEE_QMAX, (BEE_QMAX - q) / BEE_QMAX * 2 / 3, 0.0)
    return np.array([colorsys.hls_to_rgb(v, 0.5, 1.0) for v in h]).reshape(-1, 3)


# ---------------------------------------------------------------- loaders
def bee(work, arm, layer):
    z = zipfile.ZipFile(f"{work}/{arm}/mabc-pr.zip")
    d = json.loads(z.read(f"data/0/0-{layer}.json"))
    return np.c_[d["x"], d["y"], d["z"]], np.array(d["cluster_id"]), np.array(d["q"], float)


def rec(work, arm, kind):
    f = uproot.open(f"{work}/{arm}/tracking-{kind}.root")
    t = f["T_rec_charge"].arrays(library="np")
    r = f["Trun"].arrays(library="np")
    return t, float(r["dQdx_scale"][0]), float(r["dQdx_offset"][0])


def proj_block(work, arm, cl):
    pd = uproot.open(f"{work}/{arm}/tracking-stm.root")["T_proj_data"].arrays(library="np")
    ids = np.array(pd["cluster_id"][0])
    for i, v in enumerate(ids):
        if v // 10 == cl:
            return dict(ch=np.array(pd["channel"][0][i]), ts=np.array(pd["time_slice"][0][i]),
                        q=np.array(pd["charge"][0][i]), qp=np.array(pd["charge_pred"][0][i]))
    return None


def bad_ranks(work, arm, det, verified_planes):
    cfg = DETS[det]
    if cfg["nper"] is None:
        return {}
    b = uproot.open(f"{work}/{arm}/tracking-stm.root")["T_bad_ch"].arrays(library="np")
    out = {}
    for raw, pl, s, e in zip(b["chid"], b["plane"], b["start_time"], b["end_time"]):
        pl = int(pl)
        if pl not in verified_planes:
            continue
        rk = cfg["base"][pl] + (int(raw) // cfg["nch_apa"]) * cfg["nper"][pl] + (int(raw) % cfg["nch_apa"] - cfg["start"][pl])
        out.setdefault(pl, []).append((rk, s, e))
    return out


# ---------------------------------------------------------------- geometry helpers
def local_frame(P, w):
    if len(P) < 5 or np.sum(w) <= 0:
        raise ValueError(f"local_frame: only {len(P)} image points (weight {np.sum(w)}) near the click")
    c = np.average(P, axis=0, weights=w)
    _, _, vt = np.linalg.svd((P - c) * np.sqrt(w / w.sum())[:, None], full_matrices=False)
    e1 = vt[0] / np.linalg.norm(vt[0])
    xh = np.array([1.0, 0, 0])
    e3 = xh - (xh @ e1) * e1
    if np.linalg.norm(e3) < 0.2:             # track along drift: use y as the second axis
        yh = np.array([0, 1.0, 0]); e3 = yh - (yh @ e1) * e1
    e3 /= np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    return c, e1, e2, e3


def to_frame(X, c, e1, e2, e3):
    V = X - c
    return V @ e1, V @ e2, V @ e3


def order_runs(t, sel, maxgap=3.0):
    """consecutive file-order rows of one (cluster,pass) split at jumps > maxgap cm."""
    idx = np.where(sel)[0]
    if len(idx) == 0:
        return []
    P = np.c_[t["x"][idx], t["y"][idx], t["z"][idx]]
    st = np.r_[np.inf, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    brk = np.where((st > maxgap) | (np.r_[1, np.diff(idx)] != 1))[0]
    return [idx[a:b] for a, b in zip(brk, list(brk[1:]) + [len(idx)])]


def local_project(t, sel, key, S, k=12, tol=0.05):
    """project points S to (key, pt) with an affine (x,y,z)->(p*,pt) map fitted on the k fit rows nearest
    to each point.  A PDHD U/V plane wraps, so one global map does not close; a local map on rows that
    stay on one wire segment does.  Returns (w, s, ok): ok = the local map closes to < tol on its rows."""
    idx = np.where(sel)[0]
    n = len(S)
    w = np.full(n, np.nan); s = np.full(n, np.nan); ok = np.zeros(n, bool)
    if len(idx) < k or n == 0:
        return w, s, ok
    X = np.c_[t["x"][idx], t["y"][idx], t["z"][idx]]
    tree = cKDTree(X)
    _, nb = tree.query(S, k=k)
    for i in range(n):
        A = np.c_[X[nb[i]], np.ones(k)]
        cw = np.linalg.lstsq(A, t[key][idx][nb[i]], rcond=None)[0]
        cs = np.linalg.lstsq(A, t["pt"][idx][nb[i]], rcond=None)[0]
        rw = np.max(np.abs(A @ cw - t[key][idx][nb[i]])); rs = np.max(np.abs(A @ cs - t["pt"][idx][nb[i]]))
        w[i] = np.r_[S[i], 1] @ cw; s[i] = np.r_[S[i], 1] @ cs
        ok[i] = (rw < tol) and (rs < tol)
    return w, s, ok


def on_charge(blk, lo, hi, w, s):
    """fraction of projected points (w,s) whose OWN cell (rounded wire, rounded slice) holds measured
    charge (q>0) of this cluster on this plane."""
    if blk is None or len(w) == 0:
        return np.nan
    m = (blk["ch"] >= lo) & (blk["ch"] < hi) & (blk["q"] > 0)
    cells = set(zip(blk["ch"][m].astype(int).tolist(), blk["ts"][m].astype(int).tolist()))
    hit = [(int(np.round(wi)), int(np.round(si))) in cells for wi, si in zip(w, s)]
    return float(np.mean(hit))


def ridge_dist(F, D, P, w, slab=0.5, rad=3.0):
    tree = cKDTree(P); out = np.full(len(F), np.nan)
    for i, (f, d) in enumerate(zip(F, D)):
        ii = tree.query_ball_point(f, rad)
        if not ii:
            continue
        V = P[ii] - f
        m = np.abs(V @ d) < slab
        if m.sum() == 0:
            continue
        c = np.average(V[m], axis=0, weights=w[ii][m])
        out[i] = np.linalg.norm(c - (c @ d) * d)
    return out


def local_dirs(F, k=2):
    D = np.zeros_like(F)
    for i in range(len(F)):
        a, b = max(0, i - k), min(len(F) - 1, i + k)
        v = F[b] - F[a]; n = np.linalg.norm(v)
        D[i] = v / n if n > 0 else np.array([1.0, 0, 0])
    return D


def lattice_lock(t, idx):
    g = [np.abs(2 * t[p][idx] - np.round(2 * t[p][idx])) < 1e-3 for p in PLANES]
    return (np.sum(g, axis=0) >= 2)


def wiggle(F, k=3):
    """distance of row i from the chord F[i-k] -> F[i+k] (cm); nan within k of an end."""
    out = np.full(len(F), np.nan)
    for i in range(k, len(F) - k):
        a, b = F[i - k], F[i + k]; n = np.linalg.norm(b - a)
        if n <= 0:
            continue
        u = (b - a) / n; v = F[i] - a
        out[i] = np.linalg.norm(v - (v @ u) * u)
    return out


def folds(F, k=5):
    """step i -> i+1 points backward on the local chord F[i-k] -> F[i+k+1]."""
    out = np.zeros(len(F), bool)
    for i in range(len(F) - 1):
        a, b = F[max(0, i - k)], F[min(len(F) - 1, i + 1 + k)]
        u = b - a
        if np.linalg.norm(u) > 0:
            out[i] = (F[i + 1] - F[i]) @ u < 0
    return out


def pixel_sep(t, idx):
    """per-plane pixel separation between consecutive rows, hypot(d wire, d slice); shape (n-1, 3)."""
    dT = np.diff(t["pt"][idx])
    return np.c_[[np.hypot(np.diff(t[p][idx]), dT) for p in PLANES]].T


def neg_runs(q):
    """(start, length) of runs of consecutive q < 0 rows."""
    runs, i = [], 0
    while i < len(q):
        if q[i] < 0:
            j = i
            while j < len(q) and q[j] < 0:
                j += 1
            runs.append((i, j - i)); i = j
        else:
            i += 1
    return runs


# ---------------------------------------------------------------- per spot
def spot_data(work, arm, cl, click):
    d = {}
    for layer, key in (("clustering-global", "img"), ("steiner_graph-global", "stg"), ("steiner_terminals-global", "trm"),
                       ("stm_fit-global", "bee_stm"), ("track_fit-global", "bee_trk")):
        P, c, q = bee(work, arm, layer)
        m = (c == cl) & (np.linalg.norm(P - click, axis=1) < 1.6 * R)
        d[key] = (P[m], q[m])
        if layer == "stm_fit-global":
            d["bee_stm_all"] = (P[c == cl], q[c == cl])
    d["stm"], d["stm_scale"], d["stm_off"] = rec(work, arm, "stm")
    d["pr"], _, _ = rec(work, arm, "pr")
    d["proj"] = proj_block(work, arm, cl)
    # the viewer's rows ARE the ROOT rows: assert it whenever the layer carries this cluster
    t = d["stm"]; m = t["cluster_id"] == cl
    Pz, qz = d["bee_stm_all"]
    d["in_layer"] = len(qz) > 0
    if d["in_layer"]:
        if len(qz) != int(m.sum()):
            raise SystemExit(f"{arm} cl{cl}: zip stm_fit rows {len(qz)} != T_rec_charge rows {int(m.sum())}")
        dq = np.abs(qz - t["q"][m]).max(); dx = np.abs(Pz[:, 0] - t["x"][m]).max()
        if dq > 0.01 or dx > 0.001:
            raise SystemExit(f"{arm} cl{cl}: zip stm_fit differs from T_rec_charge (max |dq| {dq}, |dx| {dx})")
        d["zip_eq_root"] = f"{len(qz)} rows, max|dq| {dq:.3g}, max|dx| {dx:.2g}"
    else:
        d["zip_eq_root"] = "cluster not in stm_fit layer"
    return d


def frame_pts(D, arms, cl, click):
    """points that define the local track frame: the cluster's image near the click, or -- when the click
    sits on a fit stretch far from any image (a chord across a gap) -- the STM fit rows near it."""
    P, q = D[arms[-1]]["img"]
    P, q = P[np.linalg.norm(P - click, axis=1) < R], q[np.linalg.norm(P - click, axis=1) < R]
    if len(P) >= 20:
        D[arms[-1]]["frame"] = (P, np.clip(q, 1, None)); D[arms[-1]]["frame_from"] = "image"
        return
    t = D[arms[-1]]["stm"]
    X = np.c_[t["x"], t["y"], t["z"]]
    m = (t["cluster_id"] == cl) & (np.linalg.norm(X - click, axis=1) < R)
    D[arms[-1]]["frame"] = (X[m], np.ones(m.sum())); D[arms[-1]]["frame_from"] = "stm_fit"


def fig_bee(spot, cl, click, arms, D, out):
    """rows: persisted STM rows / Bee stm_fit layer as drawn / Bee track_fit layer as drawn;
    columns: per arm, along vs transverse in the wire plane and along vs transverse toward drift."""
    P0, w0 = D[arms[-1]]["frame"]
    c, e1, e2, e3 = local_frame(P0, w0)
    projs = (("along / transverse in wire plane (cm)", lambda X: to_frame(X, c, e1, e2, e3)[:2]),
             ("along / transverse toward drift (cm)", lambda X: (to_frame(X, c, e1, e2, e3)[0], to_frame(X, c, e1, e2, e3)[2])))
    kinds = ("STM fit: every persisted row (T_rec_charge)", "Bee stm_fit layer as drawn (q >= 0 only)",
             "Bee track_fit layer as drawn (writer clamps q<0 to 0)")
    ncol = 2 * len(arms)
    fig, axs = plt.subplots(3, ncol, figsize=(6.2 * ncol, 13.5), squeeze=False)
    tw = max(6.0, R / 2)
    for a_i, arm in enumerate(arms):
        d = D[arm]; t = d["stm"]
        m = t["cluster_id"] == cl
        X = np.c_[t["x"][m], t["y"][m], t["z"][m]]; q = t["q"][m]
        near = np.linalg.norm(X - click, axis=1) < 1.6 * R
        Pb, qb = d["bee_stm"]; drawn = qb >= 0
        Pt, qt = d["bee_trk"]
        for p_i, (lab, f) in enumerate(projs):
            col = 2 * a_i + p_i
            for r in range(3):
                ax = axs[r][col]
                Pi, _ = d["img"]
                if len(Pi):
                    a, b = f(Pi); ax.scatter(a, b, s=3, color="0.8", zorder=1)
                if r == 0:
                    if near.any():
                        a, b = f(X[near]); qq = q[near]
                        ax.plot(a, b, "-", color="0.55", lw=0.6, zorder=2)
                        ax.scatter(a[qq >= 0], b[qq >= 0], s=14, c=bee_rgb(qq[qq >= 0]), zorder=3)
                        neg = qq < 0
                        ax.scatter(a[neg], b[neg], s=46, marker="x", color="k", zorder=4,
                                   label=f"q<0: Bee skips ({int(neg.sum())} rows)")
                    note = f"{int(near.sum())} rows in window, {int((q[near] < 0).sum())} with q<0"
                elif r == 1:
                    if not d["in_layer"]:
                        ax.text(0.5, 0.5, f"cl{cl} not in this arm's stm_fit layer\n(not tagged STM)",
                                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="tab:red")
                        note = "layer omits the cluster"
                    else:
                        a, b = f(Pb[drawn]); ax.scatter(a, b, s=14, c=bee_rgb(qb[drawn]), zorder=3)
                        note = f"{int(drawn.sum())} of {len(qb)} layer rows drawn"
                else:
                    if len(Pt):
                        a, b = f(Pt); ax.scatter(a, b, s=14, c=bee_rgb(qt), zorder=3)
                    note = f"{len(Pt)} rows, {int((qt == 0).sum())} at q=0 (clamped)"
                a, b = f(np.array([click]))
                ax.scatter(a, b, s=220, marker="*", color="magenta", edgecolor="k", zorder=5)
                ax.set_xlim(-R, R); ax.set_ylim(-tw, tw)
                ax.set_xlabel(lab)
                ax.set_title(f"{spot} cl{cl} [{arm}]\n{kinds[r]}\n{note}", fontsize=8)
                if r == 0 and near.any() and (q[near] < 0).any():
                    ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("colour = Bee's own charge palette (red high, blue ~0); grey = clustering image; "
                 "magenta star = owner's click", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}_{spot}_bee.png", dpi=100); plt.close(fig)


def fig3d(spot, cl, click, arms, D, out):
    P0, w0 = D[arms[-1]]["frame"]
    c, e1, e2, e3 = local_frame(P0, w0)
    fig, axs = plt.subplots(len(arms), 3, figsize=(18, 5.2 * len(arms)), squeeze=False)
    tw = max(6.0, R / 2)
    for r, arm in enumerate(arms):
        d = D[arm]
        for col, (lab, f) in enumerate((("along / transverse in wire plane (cm)", lambda X: to_frame(X, c, e1, e2, e3)[:2]),
                                        ("along / transverse toward drift (cm)", lambda X: (to_frame(X, c, e1, e2, e3)[0], to_frame(X, c, e1, e2, e3)[2])),
                                        ("z / y (cm, top view)", lambda X: (X[:, 2], X[:, 1])))):
            ax = axs[r][col]
            P, q = d["img"]
            if len(P):
                a, b = f(P); ax.scatter(a, b, s=4, c=np.log10(np.clip(q, 100, None)), cmap="Greys", alpha=0.6, label="image")
            P, _ = d["stg"]
            if len(P):
                a, b = f(P); ax.scatter(a, b, s=6, marker="x", color="tab:orange", alpha=0.7, label="Steiner cloud")
            P, _ = d["trm"]
            if len(P):
                a, b = f(P); ax.scatter(a, b, s=30, marker="^", facecolor="none", edgecolor="red", label="terminals")
            t = d["stm"]
            for pa in sorted(set(t["pass"][t["cluster_id"] == cl].tolist())):
                pc = PASS_COLORS[pa % len(PASS_COLORS)]
                sel = (t["cluster_id"] == cl) & (t["pass"] == pa)
                for run in order_runs(t, sel):
                    X = np.c_[t["x"][run], t["y"][run], t["z"][run]]
                    near = np.linalg.norm(X - click, axis=1) < 1.6 * R
                    if not near.any():
                        continue
                    a, b = f(X[near]); ax.plot(a, b, "-", color=pc, lw=1.2)
                    lo = t["q"][run][near] < 0
                    ax.scatter(a[~lo], b[~lo], s=10, color=pc, label=f"STM fit pass {pa}" if col == 0 else None)
                    ax.scatter(a[lo], b[lo], s=14, facecolor="none", edgecolor=pc, label=f"pass {pa}, q<0 (Bee skips)" if col == 0 else None)
            t = d["pr"]; sel = (t["cluster_id"] == cl)
            cols = plt.cm.tab10(np.linspace(0, 1, 10))
            for j, sc in enumerate(sorted(set(t["sub_cluster_id"][sel].tolist()))):
                ii = np.where(sel & (t["sub_cluster_id"] == sc))[0]
                X = np.c_[t["x"][ii], t["y"][ii], t["z"][ii]]
                near = np.linalg.norm(X - click, axis=1) < 1.6 * R
                if not near.any():
                    continue
                a, b = f(X[near])
                if sc == -1:
                    ax.scatter(a, b, s=40, marker="s", color="k", label="PR vertex")
                else:
                    ax.plot(a, b, "--", color=cols[j % 10], lw=1)
                    ax.scatter(a, b, s=6, color=cols[j % 10])
            a, b = f(np.array([click]))
            ax.scatter(a, b, s=220, marker="*", color="magenta", edgecolor="k", zorder=5, label="owner's click")
            if col < 2:
                ax.set_xlim(-R, R); ax.set_ylim(-tw, tw)
            else:
                ax.set_xlim(click[2] - R, click[2] + R); ax.set_ylim(click[1] - R, click[1] + R)
            ax.set_xlabel(lab); ax.set_title(f"{spot} cl{cl} [{arm}]")
            if r == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{out}_{spot}_3d.png", dpi=100); plt.close(fig)


def fig2d(spot, cl, click, arms, D, out, det, work, verified_planes):
    """returns {arm: {metric: value}} with per-plane on-charge fractions (cells of THIS cluster only)."""
    base = DETS[det]["base"]
    fig, axs = plt.subplots(len(arms), 3, figsize=(18, 5.2 * len(arms)), squeeze=False)
    metrics = {}
    for r, arm in enumerate(arms):
        d = D[arm]; t = d["stm"]; tp = d["pr"]
        sel_all = t["cluster_id"] == cl
        X = np.c_[t["x"], t["y"], t["z"]]
        near = sel_all & (np.linalg.norm(X - click, axis=1) < R)
        far = sel_all & (np.linalg.norm(X - click, axis=1) < 3 * R)
        blk = d["proj"]
        bad = bad_ranks(work, arm, det, verified_planes)
        S, _ = d["stg"]
        S = S[np.linalg.norm(S - click, axis=1) < R] if len(S) else S
        mt = metrics.setdefault(arm, {})
        for pl, key in enumerate(PLANES):
            ax = axs[r][pl]
            lo, hi = base[pl], (base[pl + 1] if pl < 2 else 10 ** 7)
            cw, cs, cok = local_project(t, far, key, np.array([click]))
            if near.any():
                vals = t[key][near]; med = np.median(vals)
                vals = vals[np.abs(vals - med) < 40]          # one wire segment across a U/V wrap
                w0, w1 = vals.min() - 8, vals.max() + 8
                s0, s1 = t["pt"][near].min() - 12, t["pt"][near].max() + 12
            elif np.isfinite(cw[0]):
                w0, w1, s0, s1 = cw[0] - 20, cw[0] + 20, cs[0] - 30, cs[0] + 30
            else:
                ax.set_title(f"{arm} {key}: no fit rows near the click"); continue
            ii = np.where(near)[0]
            mt[f"stm_onq_{key}"] = round(on_charge(blk, lo, hi, t[key][ii], t["pt"][ii]), 2) if len(ii) else np.nan
            # +-1 wire on the row's own slice: the exact own-cell test reads low on PDVD U/V (d110_gap_census sec 2c)
            mt[f"stm_onq1_{key}"] = round(float(np.mean([max(on_charge(blk, lo, hi, [t[key][i] + dw], [t["pt"][i]]) for dw in (-1, 0, 1))
                                                         for i in ii])), 2) if len(ii) and blk is not None else np.nan
            jn = np.where((tp["cluster_id"] == cl) & (tp["flag_vertex"] == 0) & (np.linalg.norm(np.c_[tp["x"], tp["y"], tp["z"]] - click, axis=1) < R))[0]
            mt[f"pr_onq_{key}"] = round(on_charge(blk, lo, hi, tp[key][jn], tp["pt"][jn]), 2) if len(jn) else np.nan
            if blk is not None:
                m = (blk["ch"] >= lo) & (blk["ch"] < hi) & (blk["ch"] >= w0) & (blk["ch"] <= w1) & (blk["ts"] >= s0) & (blk["ts"] <= s1)
                ax.scatter(blk["ts"][m], blk["ch"][m], c=np.log10(np.clip(blk["q"][m], 100, None)), marker="s", s=22, cmap="viridis", vmin=2.5, vmax=4.8)
                pm = m & (blk["qp"] > 0)
                ax.scatter(blk["ts"][pm], blk["ch"][pm], marker="s", s=22, facecolor="none", edgecolor="w", lw=0.4)
            for rk, s, e in bad.get(pl, []):
                if w0 <= rk <= w1:
                    ax.axhspan(rk - 0.5, rk + 0.5, color="red", alpha=0.15)
            ii = np.where(far)[0]
            for pa in sorted(set(t["pass"][far].tolist())):
                for run in order_runs(t, far & (t["pass"] == pa)):
                    ax.plot(t["pt"][run], t[key][run], ".-", color=PASS_COLORS[pa % len(PASS_COLORS)], ms=3, lw=0.8,
                            label=f"STM fit pass {pa}")
                    ng = run[t["q"][run] < 0]
                    ax.scatter(t["pt"][ng], t[key][ng], s=40, marker="x", color="k", zorder=4,
                               label="q<0 (Bee skips)")
            rg = ii[t["reg_flag_" + key[1]][ii] == 1]
            ax.scatter(t["pt"][rg], t[key][rg], s=26, facecolor="none", edgecolor="red", label=f"reg_flag_{key[1]}")
            jj = np.where((tp["cluster_id"] == cl) & (np.linalg.norm(np.c_[tp["x"], tp["y"], tp["z"]] - click, axis=1) < 3 * R))[0]
            ax.scatter(tp["pt"][jj], tp[key][jj], s=8, marker="+", color="tab:pink", label="PR fit")
            if cok[0]:
                ax.scatter(cs, cw, s=200, marker="*", color="magenta", edgecolor="k", zorder=5)
            note = f"on-charge STM {mt[f'stm_onq_{key}']} PR {mt[f'pr_onq_{key}']}"
            ax.set_xlim(s0, s1); ax.set_ylim(w0, w1)
            ax.set_xlabel("time slice"); ax.set_ylabel(f"{key} (base+rank)")
            ax.set_title(f"{spot} cl{cl} [{arm}] {key}\n{note}", fontsize=8)
            if r == 0 and pl == 0:
                h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
                ax.legend(u.values(), u.keys(), fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{out}_{spot}_2d.png", dpi=100); plt.close(fig)
    return metrics


def figprofile(spot, cl, click, arms, D, out):
    fig, axs = plt.subplots(4, len(arms), figsize=(8 * len(arms), 13), squeeze=False, sharex="col")
    rows = []
    for c_, arm in enumerate(arms):
        d = D[arm]; t = d["stm"]; scale, off = d["stm_scale"], d["stm_off"]
        sel = t["cluster_id"] == cl
        runs = order_runs(t, sel)
        best = None
        for run in runs:
            X = np.c_[t["x"][run], t["y"][run], t["z"][run]]
            dd = np.linalg.norm(X - click, axis=1)
            if best is None or dd.min() < best[1]:
                best = (run, dd.min(), int(np.argmin(dd)))
        img_P, img_q = d["img"]; stg_P, _ = d["stg"]
        if best is None:
            continue
        run, dmin, k0 = best
        X = np.c_[t["x"][run], t["y"][run], t["z"][run]]
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]; L = L - L[k0]
        wig_all = wiggle(X); fold_all = folds(X)
        win = np.abs(L) < R
        Xw = X[win]; idx = run[win]; Lw = L[win]
        qw = t["q"][idx]
        dq = (qw - off) / scale
        with np.errstate(divide="ignore", invalid="ignore"):
            dqdx = dq / t["nq"][idx]
        Dd = local_dirs(Xw)
        rd = ridge_dist(Xw, Dd, img_P, np.clip(img_q, 1, None)) if len(img_P) else np.full(len(Xw), np.nan)
        sd = cKDTree(stg_P).query(Xw)[0] if len(stg_P) else np.full(len(Xw), np.nan)
        lock = lattice_lock(t, idx)
        regs = {p: t["reg_flag_" + p][idx] for p in "uvw"}
        sep = pixel_sep(t, idx) if len(idx) > 1 else np.zeros((0, 3))
        wg = wig_all[win]; fd = fold_all[win]
        neg = qw < 0
        ax = axs[0][c_]
        ax.plot(Lw, dqdx / 1e3, ".-", label="dQ/dx (ke/cm, doc 101 ruler)")
        ax.plot(Lw, dq / 1e3, ".-", alpha=0.5, label="dQ (ke)")
        ax.scatter(Lw[neg], dq[neg] / 1e3, s=40, facecolor="none", edgecolor="red", label="q<0: dQ<10 ke, Bee skips")
        ax.axhline(10, color="red", ls=":", lw=0.8); ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{spot} cl{cl} [{arm}] STM fit, nearest to click {dmin:.1f} cm"); ax.legend(fontsize=7)
        ax = axs[1][c_]
        ax.plot(Lw, rd, ".-", label="to local image ridge (cm)")
        ax.plot(Lw, sd, ".-", label="to nearest Steiner point (cm)")
        ax.plot(Lw, wg, ".-", label="wiggle: distance from the +-3-row chord (cm)")
        ax.set_ylim(0, None); ax.legend(fontsize=7)
        ax = axs[2][c_]
        Lm = 0.5 * (Lw[1:] + Lw[:-1])
        for j, p in enumerate("uvw"):
            ax.plot(Lm, sep[:, j], ".-", label=f"{p}: hypot(d wire, d slice) per step")
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(0.02, 5); ax.legend(fontsize=7)
        ax.set_ylabel("pixel separation (<0.5: rows share cells)")
        ax = axs[3][c_]
        for j, p in enumerate("uvw"):
            ax.scatter(Lw[regs[p] == 1], np.full((regs[p] == 1).sum(), j), marker="|", s=200, color="red")
        ax.scatter(Lw[lock], np.full(lock.sum(), 3), marker="|", s=200, color="k")
        ax.scatter(Lw[fd], np.full(fd.sum(), 4), marker="|", s=200, color="tab:orange")
        ax.scatter(Lw[neg], np.full(neg.sum(), 5), marker="|", s=200, color="tab:blue")
        ax.set_yticks([0, 1, 2, 3, 4, 5]); ax.set_yticklabels(["reg u", "reg v", "reg w", "lattice lock", "fold (back step)", "q<0 (Bee skips)"])
        ax.set_xlabel("arc length along STM fit from the row nearest the click (cm)")
        gaps = np.diff(L)
        nr = neg_runs(qw)
        longest = max(nr, key=lambda r: r[1]) if nr else (0, 0)
        hole_cm = float(np.sum(np.linalg.norm(np.diff(Xw[longest[0]:longest[0] + longest[1]], axis=0), axis=1))) if longest[1] > 1 else 0.0
        ndeg = (sep < 0.5).sum(1) if len(sep) else np.zeros(0)
        rows.append(dict(spot=spot, cl=cl, arm=arm, window_cm=R, in_stm_fit_layer=int(d["in_layer"]),
                         zip_eq_root=d["zip_eq_root"], stm_nearest_cm=round(float(dmin), 2),
                         n_win=int(win.sum()), bee_skipped_q_lt0=int(neg.sum()),
                         longest_hole_rows=int(longest[1]), longest_hole_cm=round(hole_cm, 2),
                         wiggle_max_cm=round(float(np.nanmax(wg)), 2) if np.isfinite(wg).any() else np.nan,
                         folds=int(fd.sum()),
                         steps_2plus_degenerate=round(float(np.mean(ndeg >= 2)), 2) if len(ndeg) else np.nan,
                         ridge_med=round(float(np.nanmedian(rd)), 2), ridge_max=round(float(np.nanmax(rd)), 2),
                         steiner_med=round(float(np.nanmedian(sd)), 2),
                         reg_any=round(float(np.mean((regs["u"] + regs["v"] + regs["w"]) > 0)), 2),
                         lattice_lock=round(float(np.mean(lock)), 2),
                         dqdx_med_ke=round(float(np.nanmedian(dqdx)) / 1e3, 1),
                         max_step_cm=round(float(gaps[np.abs(L[1:]) < R].max()) if (np.abs(L[1:]) < R).any() else 0.0, 2)))
    fig.tight_layout(); fig.savefig(f"{out}_{spot}_profile.png", dpi=100); plt.close(fig)
    return rows


def pr_rows(cl, click, arms, D):
    rows = []
    for arm in arms:
        t = D[arm]["pr"]
        sel = t["cluster_id"] == cl
        X = np.c_[t["x"], t["y"], t["z"]]
        dd = np.linalg.norm(X - click, axis=1)
        near = sel & (dd < R) & (t["flag_vertex"] == 0)
        rows.append(dict(arm=arm, pr_nearest_cm=round(float(dd[sel].min()), 2) if sel.any() else np.nan,
                         pr_rows_win=int(near.sum()), pr_segments_win=len(set(t["sub_cluster_id"][near].tolist())),
                         pr_q_lt0_win=int(np.sum(t["q"][near] < 0))))
    return rows


def main():
    global R
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=sorted(SPOTS))
    ap.add_argument("--spot", action="append", default=[], help="name:cluster:x,y,z:R (overrides SPOTS)")
    ap.add_argument("--uv-rank-verified", action="store_true",
                    help="draw U/V dead channels too (only after the raw->rank rule is verified for U/V)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = f"{IMG}/{a.det}/work"
    event, armlist, spots = SPOTS[a.det]
    arms = [f"{event}_{x}" for x in armlist.split(",")]
    spots = a.spot or spots
    verified = {0, 1, 2} if a.uv_rank_verified else {2}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    allrows = []
    for sp in spots:
        name, cl, xyz, rr = sp.split(":")
        R = float(rr)
        cl = int(cl); click = np.array([float(v) for v in xyz.split(",")])
        D = {arm: spot_data(work, arm, cl, click) for arm in arms}
        frame_pts(D, arms, cl, click)
        fig_bee(name, cl, click, arms, D, a.out)
        fig3d(name, cl, click, arms, D, a.out)
        m2d = fig2d(name, cl, click, arms, D, a.out, a.det, work, verified)
        prow = {r["arm"]: r for r in pr_rows(cl, click, arms, D)}
        P, q = D[arms[-1]]["frame"]
        c, e1, e2, e3 = local_frame(P, q)
        ang = float(np.degrees(np.arccos(min(1.0, abs(e1[0])))))
        for r in figprofile(name, cl, click, arms, D, a.out):
            r.update({k: v for k, v in prow[r["arm"]].items() if k != "arm"})
            r["angle_to_drift_deg"] = round(ang, 1)
            r.update(m2d.get(r["arm"], {}))
            r["frame_from"] = D[arms[-1]]["frame_from"]
            allrows.append(r)
    keys = list(allrows[0].keys())
    for r in allrows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(f"{a.out}_{a.det}_spots.tsv", "w") as fo:
        fo.write("\t".join(keys) + "\n")
        for r in allrows:
            fo.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    for r in allrows:
        print({k: r.get(k) for k in keys})


if __name__ == "__main__":
    main()
