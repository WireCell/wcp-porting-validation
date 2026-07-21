#!/usr/bin/env python3
"""120-event census of cathode-crossing pairs: is the evt298567 in-cathode tail
(doc 16 &10.2/10.3) an accident or systematic?

For every top(apa4)/bottom(apa0) cluster pair in the _keep calib dumps:
  - junk-robust cathode endpoints (clipped 3D line fit; on-axis tube keeps the
    crossing pile at 6-9 cm perp, rejects merged-in junk at >=23 cm),
  - T0-free sum-test residual  R = X_top_end + X_bot_end  (trigger-offset diff
    cancels; ideal crosser: +3 + -3 = 0; R = pen_bot - pen_top),
  - pair acceptance: end-to-end (y,z) proximity + 3D collinearity + |R| bound,
  - when a trustworthy T0 exists (xtpc-pinned or co-selected flash): absolute
    per-side face penetration and past-face pile shape (npts, charge frac,
    (y,z) rms, drift extent).

Writes one TSV row per accepted pair.
"""
import json
import os
import sys
import glob
import numpy as np
from multiprocessing import Pool

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
OUT = os.environ.get("CENSUS_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cathode_tail_pairs.tsv"))

MIN_PTS = 50          # cluster size floor
TUBE = 12.0           # cm, on-axis tube (pile 6-9 cm in, junk >=23 cm out)
DEEP_SLAB = 3.0       # cm of u used to define the end (y,z)
D_YZ_MAX = 35.0       # cm, end-to-end (y,z) distance (along-track gap allowed:
                      # a short side retreats along the track for shallow dips)
D_PERP_MAX = 8.0      # cm, end-to-other-line perpendicular offset (the real
                      # same-line discriminator)
ANG_MAX = 25.0        # deg, 3D collinearity of the two halves
R_MAX = 25.0          # cm, |sum-test residual| acceptance
QFRAC = 0.02          # charge-quantile "body end"


def robust_yz_line(y, z):
    """Iteratively clipped 2D PCA line in the drift-free (y,z) projection
    (doc 16 &2: any axis statistic on junk-loaded clusters must be clipped).
    Returns center(2), unit dir(2), perp dist(N), along-track coord(N), extent."""
    P = np.stack([y, z], axis=1)
    keep = np.ones(len(P), bool)
    for thr in (1e9, 30.0, 15.0, 10.0):
        c = np.median(P[keep], axis=0)
        Q = P[keep] - c
        cov = Q.T @ Q / max(len(Q), 1)
        val, vec = np.linalg.eigh(cov)
        e = vec[:, -1]
        perp = np.abs((P - c) @ np.array([-e[1], e[0]]))
        rms = np.sqrt(np.mean(perp[keep] ** 2))
        nkeep = perp < min(thr, max(3.0 * rms, 6.0))
        if nkeep.sum() < 10:
            break
        keep = nkeep
    c = np.median(P[keep], axis=0)
    Q = P[keep] - c
    cov = Q.T @ Q / max(len(Q), 1)
    val, vec = np.linalg.eigh(cov)
    e = vec[:, -1]
    perp = np.abs((P - c) @ np.array([-e[1], e[0]]))
    w = (P - c) @ e
    ext = np.percentile(w[keep], 98) - np.percentile(w[keep], 2)
    return c, e, perp, w, ext


def cluster_info(c, geom):
    """Per-cluster endpoint/direction summary in the T0-free drift frame."""
    x = np.array(c["x"], float)
    y = np.array(c["y"], float)
    z = np.array(c["z"], float)
    q = np.clip(np.array(c["q"], float), 0, None)
    s, ax = geom["s"], geom["anode_x"]
    u = s * (x - ax)                       # anode 0 -> cathode 336.91
    ctr2, e2, perp2, wtrk, yz_ext = robust_yz_line(y, z)
    if yz_ext < 10.0:
        # steep (drift-aligned) track: (y,z) line degenerate; tube = points
        # near the (y,z) centroid
        perp = np.hypot(y - ctr2[0], z - ctr2[1])
        steep = True
    else:
        perp = perp2
        steep = False
    tube = perp < TUBE
    if tube.sum() < 10:
        return None
    # u along the track: fit u = a + b*w on tube points, clipped; then require
    # 3D consistency so junk sitting on the (y,z)-line EXTENSION beyond the
    # track end (doc 16 clump anatomy) is excluded, while the drift-aligned
    # crossing pile at the end (few cm u-residual) survives.
    if steep:
        b_slope = 0.0
        drift_cos = 1.0
        dir3 = np.array([-1.0, 0.0, 0.0])   # anode-ward in u
    else:
        ut0, wt = u[tube], wtrk[tube]
        keep = np.ones(len(wt), bool)
        for _ in range(3):
            A = np.vstack([wt[keep], np.ones(keep.sum())]).T
            bfit, afit = np.linalg.lstsq(A, ut0[keep], rcond=None)[0]
            res = ut0 - (bfit * wt + afit)
            sres = max(np.std(res[keep]), 1.0)
            keep = np.abs(res) < 3 * sres
            if keep.sum() < 10:
                keep = np.ones(len(wt), bool)
                break
        b_slope = bfit
        resid_all = u - (bfit * wtrk + afit)
        tube = tube & (np.abs(resid_all) < 10.0)
        if tube.sum() < 10:
            return None
        d3 = np.array([b_slope, e2[0], e2[1]])
        d3 /= np.linalg.norm(d3)
        # orient anode-ward (decreasing u)
        if d3[0] > 0:
            d3 = -d3
        dir3 = d3
        drift_cos = abs(dir3[0])
    ut, yt, zt, qt = u[tube], y[tube], z[tube], q[tube]
    # deep (cathode) end
    u_end = ut.max()
    dm = ut > u_end - DEEP_SLAB
    end_y, end_z = np.median(yt[dm]), np.median(zt[dm])
    # charge-quantile body end (walk from deep, discard first QFRAC of charge)
    o = np.argsort(-ut)
    cum = np.cumsum(qt[o])
    tot = max(cum[-1], 1e-9)
    iq = np.searchsorted(cum, QFRAC * tot)
    u_end_q = ut[o][min(iq, len(o) - 1)]
    length = float(np.percentile(ut, 99) - np.percentile(ut, 1)) if steep else \
        float(np.linalg.norm([b_slope, 1.0]) * yz_ext)
    return dict(uid=c["uid"], apa=c["apa"], n=len(x), ntube=int(tube.sum()),
                q_tot=float(q.sum()),
                u=u, y=y, z=z, q=q, tube=tube, perp=perp,
                x_raw=x, steep=steep, line_c=ctr2, line_e=e2,
                dirn=dir3, u_end=float(u_end), u_end_q=float(u_end_q),
                end_y=float(end_y), end_z=float(end_z), length=length,
                drift_cos=float(drift_cos))


def perp_to_line(ci_line, ey, ez):
    """Perp distance of the point (ey,ez) to a cluster's (y,z) line."""
    c, e = ci_line["line_c"], ci_line["line_e"]
    if ci_line["steep"]:
        return float(np.hypot(ey - c[0], ez - c[1]))
    return float(abs((ey - c[0]) * (-e[1]) + (ez - c[1]) * e[0]))


def pile_metrics(ci, X, face, sgn):
    """Past-face material of one half at a given T0.

    X: physical x of all points; face: own cathode face; sgn: +1 if the volume
    occupies x > face (top), -1 if x < face (bottom).
    Penetration = signed depth past the face toward/through the cathode.
    """
    tube = ci["tube"]
    past = tube & (sgn * (X - face) < 0)
    Xt = X[tube]
    pen_end = float(sgn * (face - Xt[np.argmax(sgn * (face - Xt))])) if len(Xt) else 0.0
    # equivalently: deepest tube point's penetration
    pen_end = float(np.max(sgn * (face - Xt))) if len(Xt) else 0.0
    if not past.any():
        return pen_end, 0, 0.0, 0.0, 0.0
    yz = np.stack([ci["y"][past], ci["z"][past]], axis=1)
    yz_rms = float(np.sqrt(((yz - yz.mean(0)) ** 2).sum(1).mean()))
    qfrac = float(ci["q"][past].sum() / max(ci["q_tot"], 1e-9))
    ext = float(np.ptp(X[past])) if past.sum() > 1 else 0.0
    return pen_end, int(past.sum()), qfrac, yz_rms, ext


def process(job):
    run, idx, dump = job
    try:
        d = json.load(open(dump))
    except Exception as e:
        return f"# FAIL {run} {idx} {dump}: {e}\n", None
    ds = d["drift_speed"]
    dclk = d["trigger_offsets_us"][1] - d["trigger_offsets_us"][0]
    gt, gb = d["geometry"]["4"], d["geometry"]["0"]
    evt = None
    for f in glob.glob(os.path.dirname(dump) + "/calib-evt*.json"):
        evt = os.path.basename(f)[9:-5]
    tops, bots = [], []
    for c in d["clusters"]:
        if c["npoints"] < MIN_PTS or c["apa"] not in (0, 4):
            continue
        ci = cluster_info(c, gt if c["apa"] == 4 else gb)
        if ci is None:
            continue
        (tops if c["apa"] == 4 else bots).append(ci)
    # bundle lookup
    pin = {}
    sel = {}
    for b in d["bundles"]:
        k = b["main_cluster"]
        if b.get("xtpc_pin"):
            pin.setdefault(k, []).append(b["flash_gid"])
        if b.get("auto_selected"):
            sel.setdefault(k, []).append(b["flash_gid"])
    fby = {f["gid"]: f for f in d["flashes"]}

    rows = []
    for t in tops:
        # T0-free physical end of top at t0=0 bottom clock: X_t = x_raw + dclk*ds... see R
        for b in bots:
            dyz = np.hypot(t["end_y"] - b["end_y"], t["end_z"] - b["end_z"])
            if dyz > D_YZ_MAX:
                continue
            dperp = 0.5 * (perp_to_line(t, b["end_y"], b["end_z"]) +
                           perp_to_line(b, t["end_y"], t["end_z"]))
            if dperp > D_PERP_MAX:
                continue
            # sum-test residual, T0-free:
            # X_t(t0) = x_t + (t0 + dclk)*ds ; X_b(t0) = x_b - t0*ds
            xte = gt["anode_x"] - t["u_end"] / gt["s"] * 1.0  # invert u -> raw x
            # u = s*(x - ax) => x = ax + u/s
            xte = gt["anode_x"] + t["u_end"] / gt["s"]
            xbe = gb["anode_x"] + b["u_end"] / gb["s"]
            R = xte + xbe + dclk * ds
            if abs(R) > R_MAX:
                continue
            # collinearity: both dirs point anode-ward in own frame; compare in
            # common frame by flipping the bottom's drift component sign
            dt_, db_ = t["dirn"].copy(), b["dirn"].copy()
            # common frame: x_c = -u_top (top), +u_bot ... both anode-ward flips
            dcom_t = np.array([-dt_[0], dt_[1], dt_[2]])
            dcom_b = np.array([db_[0], db_[1], db_[2]])
            cosang = abs(np.dot(dcom_t, dcom_b))
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            if ang > ANG_MAX:
                continue
            # T0 source
            t0src, gid = "none", -1
            common_pin = set(pin.get(t["uid"], [])) & set(pin.get(b["uid"], []))
            common_sel = set(sel.get(t["uid"], [])) & set(sel.get(b["uid"], []))
            if common_pin:
                t0src, gid = "pin", sorted(common_pin)[0]
            elif common_sel:
                t0src, gid = "cosel", sorted(common_sel)[0]
            else:
                # QL missed the pair: bracket the T0 from the two face
                # conditions (top end at +face when t1=(face_t - xte)/v, bottom
                # end at -face when t0=(xbe - face_b)/v) and take the brightest
                # flash inside the bracket -- light-anchored, QL-independent.
                t1_top = (gt["cathode_x"] - xte) / (gt["sign_offset"] * ds)
                t0_top = t1_top - dclk
                t0_bot = (gb["cathode_x"] - xbe) / (gb["sign_offset"] * ds)
                lo, hi = min(t0_top, t0_bot) - 15.0, max(t0_top, t0_bot) + 15.0
                cand = [f for f in fby.values()
                        if lo <= f["time"] <= hi and f["total_PE"] > 1000]
                if cand:
                    fbest = max(cand, key=lambda f: f["total_PE"])
                    t0src, gid = "geo", fbest["gid"]
            row = dict(run=run, idx=idx, evt=evt, uid_t=t["uid"], uid_b=b["uid"],
                       n_t=t["n"], n_b=b["n"], len_t=round(t["length"], 1),
                       len_b=round(b["length"], 1),
                       drift_cos_t=round(t["drift_cos"], 3),
                       drift_cos_b=round(b["drift_cos"], 3),
                       end_y=round(0.5 * (t["end_y"] + b["end_y"]), 1),
                       end_z=round(0.5 * (t["end_z"] + b["end_z"]), 1),
                       d_yz=round(float(dyz), 2), d_perp=round(float(dperp), 2),
                       ang=round(float(ang), 2),
                       R=round(float(R), 2),
                       R_q=round(float((gt["anode_x"] + t["u_end_q"] / gt["s"]) +
                                       (gb["anode_x"] + b["u_end_q"] / gb["s"]) +
                                       dclk * ds), 2),
                       t0src=t0src, gid=gid)
            if gid >= 0:
                f = fby[gid]
                Xt = t["x_raw"] + gt["sign_offset"] * f["time1"] * ds
                Xb = b["x_raw"] + gb["sign_offset"] * f["time"] * ds
                pen_t, np_t, qf_t, yzr_t, ext_t = pile_metrics(t, Xt, gt["cathode_x"], +1)
                pen_b, np_b, qf_b, yzr_b, ext_b = pile_metrics(b, Xb, gb["cathode_x"], -1)
                row.update(pe=round(f["total_PE"], 0), t_us=round(f["time"], 2),
                           pen_t=round(pen_t, 2), pen_b=round(pen_b, 2),
                           npast_t=np_t, npast_b=np_b,
                           qfpast_t=round(qf_t, 4), qfpast_b=round(qf_b, 4),
                           yzrms_t=round(yzr_t, 2), yzrms_b=round(yzr_b, 2),
                           driftext_t=round(ext_t, 2), driftext_b=round(ext_b, 2))
            else:
                row.update(pe=-1, t_us=np.nan, pen_t=np.nan, pen_b=np.nan,
                           npast_t=-1, npast_b=-1, qfpast_t=np.nan,
                           qfpast_b=np.nan, yzrms_t=np.nan, yzrms_b=np.nan,
                           driftext_t=np.nan, driftext_b=np.nan)
            rows.append(row)
    return None, rows


def main():
    jobs = []
    for run, n in (("039252", 18), ("039253", 18), ("039349", 84)):
        for i in range(n):
            dd = f"{BASE}/{run}_{i}_keep"
            fs = glob.glob(dd + "/calib-evt*.json")
            if fs:
                jobs.append((run, i, fs[0]))
    print(f"{len(jobs)} dumps", flush=True)
    cols = None
    nrows = 0
    with Pool(6) as pool, open(OUT, "w") as fo:
        for err, rows in pool.imap_unordered(process, jobs):
            if err:
                print(err.strip(), flush=True)
                continue
            for r in rows:
                if cols is None:
                    cols = list(r.keys())
                    fo.write("\t".join(cols) + "\n")
                fo.write("\t".join(str(r[c]) for c in cols) + "\n")
                nrows += 1
    print(f"wrote {nrows} pairs -> {OUT}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:      # single-dump test mode
        err, rows = process(("test", 0, sys.argv[1]))
        for r in (rows or []):
            print(r)
    else:
        main()
