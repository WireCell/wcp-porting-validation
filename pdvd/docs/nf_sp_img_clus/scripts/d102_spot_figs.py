#!/usr/bin/env python3
"""doc pdvd/102 sec 2 -- figures and metrics for the owner's Bee spots, knob-off vs knob-on.

For each spot (event, cluster, clicked Bee point) and each arm, three figures:
  <out>_<spot>_3d.png       local 3-D view in the track frame: clustering image, retiled Steiner cloud and
                            terminals (Bee layers), the STM fit (tracking-stm.root) and the PR multi-track fit
                            by segment with vertex rows (tracking-pr.root); the click as a star
  <out>_<spot>_2d.png       wire x slice per plane: T_proj_data measured charge of the cluster's block, dead
                            channels (T_bad_ch, drawn only on planes whose raw->rank rule is verified), the
                            STM-fit and PR-fit projections (pu/pv/pw vs pt) and the Steiner points projected
                            by an affine (x,y,z)->(p*,pt) map fitted on the cluster's own fit rows (drawn only
                            if the map closes to < 0.05 wire / slice)
  <out>_<spot>_profile.png  along the STM fit: dQ and dQ/dx (doc 101 ruler: ((q-offset)/scale)/nq), the Bee
                            display zero (q <= 0), distance to the local image ridge and to the nearest Steiner
                            point, reg_flag u/v/w and the lattice-lock flag
and one row per (spot, arm) in <out>_spots.tsv.

Facts this relies on (checked on 028084_21, doc 102 sec 2.1):
  * Bee stm_fit / track_fit q == T_rec_charge q (= dQ*0.1 - 1000); track_fit clamps q < 0 to 0
    (MultiAlgBlobClustering.cxx:1180-1181, :3130).
  * T_proj_data.channel and pu/pv/pw are base[plane] + rank; T_bad_ch.chid is the raw channel id.

Usage:
  d102_spot_figs.py --det pdhd --event 028084_21 --arms d101smoff,d101smkf --out figs/102 \
      [--spot spot1:128:326.1,457.4,314.4 ...]
"""
import argparse, json, os, sys, zipfile
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
DEFAULT_SPOTS = {
    "pdhd": ["spot1:128:326.1,457.4,314.4", "spot2:55:-67.1,569.9,398.9",
             "spot3:55:-78.0,571.8,418.7", "spot4:26:-146.7,401.6,92.5"],
}
R = 12.0        # cm, window half-size around the click
PLANES = ("pu", "pv", "pw")
PASS_COLORS = ("tab:blue", "tab:green", "tab:purple", "tab:brown")


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
    charge (q>0) of this cluster on this plane.  A +-1 tolerance is not used: on a near-isochronous
    stretch the band is several wires wide per slice, and +-1 hides a 3-8 wire miss."""
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


def inslice_nn(P, xdec=2):
    xs = np.round(P[:, 0], xdec); out = []
    for u in np.unique(xs):
        S = P[xs == u][:, 1:]
        if len(S) > 1:
            D = np.linalg.norm(S[:, None] - S[None], axis=2); D[D == 0] = np.inf
            out += list(D.min(1))
    return np.array(out)


# ---------------------------------------------------------------- per spot
def spot_data(work, arm, cl, click):
    d = {}
    for layer, key in (("clustering-global", "img"), ("steiner_graph-global", "stg"), ("steiner_terminals-global", "trm")):
        P, c, q = bee(work, arm, layer)
        m = (c == cl) & (np.linalg.norm(P - click, axis=1) < 1.6 * R)
        d[key] = (P[m], q[m])
    d["stm"], d["stm_scale"], d["stm_off"] = rec(work, arm, "stm")
    d["pr"], _, _ = rec(work, arm, "pr")
    d["proj"] = proj_block(work, arm, cl)
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


def fig3d(spot, cl, click, arms, D, out):
    P0, w0 = D[arms[-1]]["frame"]
    c, e1, e2, e3 = local_frame(P0, w0)
    fig, axs = plt.subplots(len(arms), 3, figsize=(18, 5.2 * len(arms)), squeeze=False)
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
                    lo = t["q"][run][near] <= 0
                    ax.scatter(a[~lo], b[~lo], s=10, color=pc, label=f"STM fit pass {pa}" if col == 0 else None)
                    ax.scatter(a[lo], b[lo], s=14, facecolor="none", edgecolor=pc, label=f"pass {pa}, Bee q<=0" if col == 0 else None)
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
                ax.set_xlim(-R, R); ax.set_ylim(-6, 6)
            else:
                ax.set_xlim(click[2] - R, click[2] + R); ax.set_ylim(click[1] - R, click[1] + R)
            ax.set_xlabel(lab); ax.set_title(f"{spot} cl{cl} [{arm}]")
            if r == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{out}_{spot}_3d.png", dpi=110); plt.close(fig)


def fig2d(spot, cl, click, arms, D, out, det, work, verified_planes):
    """returns {arm: {metric: value}} with per-plane on-charge fractions (cells of THIS cluster only:
    PdvdMagnifyTrackingVisitor.cxx:462-472 keeps the fitted cluster's own cells + owner-less dead fillers)."""
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
            # on-charge fractions, |dwire| <= 1 and |dslice| <= 1
            ii = np.where(near)[0]
            mt[f"stm_onq_{key}"] = round(on_charge(blk, lo, hi, t[key][ii], t["pt"][ii]), 2) if len(ii) else np.nan
            jn = np.where((tp["cluster_id"] == cl) & (tp["flag_vertex"] == 0) & (np.linalg.norm(np.c_[tp["x"], tp["y"], tp["z"]] - click, axis=1) < R))[0]
            mt[f"pr_onq_{key}"] = round(on_charge(blk, lo, hi, tp[key][jn], tp["pt"][jn]), 2) if len(jn) else np.nan
            sw, ss, sok = local_project(t, far, key, S)
            mt[f"steiner_onq_{key}"] = round(on_charge(blk, lo, hi, sw[sok], ss[sok]), 2) if sok.any() else np.nan
            mt[f"steiner_proj_ok_{key}"] = f"{int(sok.sum())}/{len(S)}"
            if blk is not None:
                m = (blk["ch"] >= lo) & (blk["ch"] < hi) & (blk["ch"] >= w0) & (blk["ch"] <= w1) & (blk["ts"] >= s0) & (blk["ts"] <= s1)
                sc = ax.scatter(blk["ts"][m], blk["ch"][m], c=np.log10(np.clip(blk["q"][m], 100, None)), marker="s", s=22, cmap="viridis", vmin=2.5, vmax=4.8)
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
            rg = ii[t["reg_flag_" + key[1]][ii] == 1]
            ax.scatter(t["pt"][rg], t[key][rg], s=26, facecolor="none", edgecolor="red", label=f"reg_flag_{key[1]}")
            jj = np.where((tp["cluster_id"] == cl) & (np.linalg.norm(np.c_[tp["x"], tp["y"], tp["z"]] - click, axis=1) < 3 * R))[0]
            ax.scatter(tp["pt"][jj], tp[key][jj], s=8, marker="+", color="tab:pink", label="PR fit")
            if sok.any():
                ax.scatter(ss[sok], sw[sok], s=10, marker="x", color="tab:orange", label="Steiner (local affine)")
            if cok[0]:
                ax.scatter(cs, cw, s=200, marker="*", color="magenta", edgecolor="k", zorder=5)
            note = (f"on-charge STM {mt[f'stm_onq_{key}']} PR {mt[f'pr_onq_{key}']} Steiner {mt[f'steiner_onq_{key}']}"
                    f" (proj {mt[f'steiner_proj_ok_{key}']})")
            ax.set_xlim(s0, s1); ax.set_ylim(w0, w1)
            ax.set_xlabel("time slice"); ax.set_ylabel(f"{key} (base+rank)")
            ax.set_title(f"{spot} cl{cl} [{arm}] {key}\n{note}", fontsize=8)
            if r == 0 and pl == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{out}_{spot}_2d.png", dpi=110); plt.close(fig)
    return metrics


def figprofile(spot, cl, click, arms, D, out):
    fig, axs = plt.subplots(3, len(arms), figsize=(8 * len(arms), 10), squeeze=False, sharex="col")
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
        win = np.abs(L) < R
        Xw = X[win]; idx = run[win]; Lw = L[win]
        dq = (t["q"][idx] - off) / scale
        with np.errstate(divide="ignore", invalid="ignore"):
            dqdx = dq / t["nq"][idx]
        Dd = local_dirs(Xw)
        rd = ridge_dist(Xw, Dd, img_P, np.clip(img_q, 1, None)) if len(img_P) else np.full(len(Xw), np.nan)
        sd = cKDTree(stg_P).query(Xw)[0] if len(stg_P) else np.full(len(Xw), np.nan)
        lock = lattice_lock(t, idx)
        regs = {p: t["reg_flag_" + p][idx] for p in "uvw"}
        ax = axs[0][c_]
        ax.plot(Lw, dqdx / 1e3, ".-", label="dQ/dx (ke/cm, doc 101 ruler)")
        ax.plot(Lw, dq / 1e3, ".-", alpha=0.5, label="dQ (ke)")
        lo = t["q"][idx] <= 0
        ax.scatter(Lw[lo], dq[lo] / 1e3, s=40, facecolor="none", edgecolor="red", label="Bee q<=0 (dQ<10 ke)")
        ax.axhline(10, color="red", ls=":", lw=0.8)
        ax.set_title(f"{spot} cl{cl} [{arm}] STM fit, nearest to click {dmin:.1f} cm"); ax.legend(fontsize=7)
        ax = axs[1][c_]
        ax.plot(Lw, rd, ".-", label="to local image ridge (cm)")
        ax.plot(Lw, sd, ".-", label="to nearest Steiner point (cm)")
        ax.set_ylim(0, None); ax.legend(fontsize=7)
        ax = axs[2][c_]
        for j, p in enumerate("uvw"):
            ax.scatter(Lw[regs[p] == 1], np.full((regs[p] == 1).sum(), j), marker="|", s=200, color="red")
        ax.scatter(Lw[lock], np.full(lock.sum(), 3), marker="|", s=200, color="k")
        ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(["reg u", "reg v", "reg w", "lattice lock"])
        ax.set_xlabel("arc length along STM fit from the point nearest the click (cm)")
        gaps = np.diff(L)
        rows.append(dict(spot=spot, cl=cl, arm=arm, stm_nearest_cm=round(float(dmin), 2),
                         n_win=int(win.sum()), ridge_med=round(float(np.nanmedian(rd)), 2), ridge_max=round(float(np.nanmax(rd)), 2),
                         steiner_med=round(float(np.nanmedian(sd)), 2),
                         reg_any=round(float(np.mean((regs["u"] + regs["v"] + regs["w"]) > 0)), 2),
                         reg_v=round(float(np.mean(regs["v"])), 2), lattice_lock=round(float(np.mean(lock)), 2),
                         bee_q_le0=int(lo.sum()), dqdx_med_ke=round(float(np.nanmedian(dqdx)) / 1e3, 1),
                         max_step_cm=round(float(gaps[np.abs(L[1:]) < R].max()) if (np.abs(L[1:]) < R).any() else 0.0, 2)))
    fig.tight_layout(); fig.savefig(f"{out}_{spot}_profile.png", dpi=110); plt.close(fig)
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
                         pr_bee_zero_win=int(np.sum(t["q"][near] <= 0))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="pdhd")
    ap.add_argument("--event", default="028084_21")
    ap.add_argument("--arms", default="d101smoff,d101smkf")
    ap.add_argument("--spot", action="append", default=[])
    ap.add_argument("--uv-rank-verified", action="store_true",
                    help="draw U/V dead channels too (only after the raw->rank rule is verified for U/V)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = f"{IMG}/{a.det}/work"
    arms = [f"{a.event}_{x}" for x in a.arms.split(",")]
    spots = a.spot or DEFAULT_SPOTS[a.det]
    verified = {0, 1, 2} if a.uv_rank_verified else {2}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    allrows = []; grid = []
    for sp in spots:
        name, cl, xyz = sp.split(":")
        cl = int(cl); click = np.array([float(v) for v in xyz.split(",")])
        D = {arm: spot_data(work, arm, cl, click) for arm in arms}
        frame_pts(D, arms, cl, click)
        fig3d(name, cl, click, arms, D, a.out)
        m2d = fig2d(name, cl, click, arms, D, a.out, a.det, work, verified)
        prow = {r["arm"]: r for r in pr_rows(cl, click, arms, D)}
        P, q = D[arms[-1]]["frame"]
        c, e1, e2, e3 = local_frame(P, q)
        ang = float(np.degrees(np.arccos(min(1.0, abs(e1[0])))))
        img_nn = inslice_nn(D[arms[-1]]["img"][0][np.linalg.norm(D[arms[-1]]["img"][0] - click, axis=1) < 6])
        stg_nn = inslice_nn(D[arms[-1]]["stg"][0][np.linalg.norm(D[arms[-1]]["stg"][0] - click, axis=1) < 6])
        grid.append((name, img_nn, stg_nn))
        for r in figprofile(name, cl, click, arms, D, a.out):
            r.update({k: v for k, v in prow[r["arm"]].items() if k != "arm"})
            r["angle_to_drift_deg"] = round(ang, 1)
            r["img_inslice_nn_med"] = round(float(np.median(img_nn)), 2) if len(img_nn) else np.nan
            r["steiner_inslice_nn_med"] = round(float(np.median(stg_nn)), 2) if len(stg_nn) else np.nan
            r.update(m2d.get(r["arm"], {}))
            r["frame_from"] = D[arms[-1]]["frame_from"]
            allrows.append(r)
    # grid figure: in-slice nearest-neighbour spacing, image vs Steiner cloud
    fig, axs = plt.subplots(1, len(grid), figsize=(4.5 * len(grid), 3.6), squeeze=False)
    for ax, (name, a1, a2) in zip(axs[0], grid):
        bins = np.linspace(0, 3, 31)
        ax.hist(a1, bins, histtype="step", label=f"image (med {np.median(a1):.2f})" if len(a1) else "image")
        ax.hist(a2, bins, histtype="step", label=f"Steiner cloud (med {np.median(a2):.2f})" if len(a2) else "Steiner")
        ax.set_title(name); ax.set_xlabel("in-slice nearest-neighbour spacing (cm)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{a.out}_grid_spots.png", dpi=110); plt.close(fig)
    keys = list(allrows[0].keys())
    with open(f"{a.out}_spots.tsv", "w") as fo:
        fo.write("\t".join(keys) + "\n")
        for r in allrows:
            fo.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    for r in allrows:
        print({k: r[k] for k in keys})


if __name__ == "__main__":
    main()
