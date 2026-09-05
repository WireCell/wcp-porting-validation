#!/usr/bin/env python3
# PDHD fork of the doc pdvd/42 script of the same name, by DUPLICATION;
# the PDVD file is untouched.  Only the detector tables change: a "pdhd"
# entry whose plane thresholds are (3200, 6400) -- the CUMULATIVE per-plane
# channel counts (3200 U, 3200 V, 3840 W), because PdvdMagnifyTrackingVisitor
# writes pu/pv/pw and T_proj_data.channel as base[plane] + the wire's RANK among
# that plane's channels, not as raw LArSoft channel ids.
# See pdhd/docs/stm-tagger-chain.md sec 8.
"""doc pdvd/42 sec 7 -- WHY the STM fit under-explains the 2-D charge it reaches.

Decomposes the doc-42 `U_foot` / `B_foot` residual against the three mechanisms
the owner named (2026-09-05):
  H1  the smearing model (SP software filter + diffusion) used by the fit
  H2  busy events -- charge from other activity overlapping the trajectory
  H3  PDVD's prolonged (drift-parallel) topology giving long induction ROIs

Everything is re-analysis of the doc-42 arms already on disk; no wire-cell is run.

Reference frame.  Every cell is assigned to its nearest fitted point by the
Chebyshev distance in (rounded wire, rounded slice) -- the doc-42 footprint
metric -- and additionally carries the CONTINUOUS offsets
    dperp = (channel - p<plane>[nearest]) * pitch_mm      [mm, pitch direction]
    dpar  = (time_slice - pt[nearest])   * slice_mm       [mm, drift direction]
so that windows can be made physically matched across planes and detectors
(a +-1 cell window is +-7.65 mm on PDVD U/V but +-3.00 mm on SBND: NOT
comparable, doc 42 sec 2.1).

Local topology.  Points are ordered along the track by residual range; the local
direction is the central difference of the neighbours, and
    theta_P = atan2(|dt|*slice_mm, |dwire|*pitch_mm)   [deg, per plane]
is 0 for an isochronous segment (moves along wires, constant time) and 90 for a
prolonged segment (moves along the drift, one wire, many ticks).  This is the
H3 variable and it is dimensionally fair across planes and detectors.

Outputs (<out>_*.tsv):
  window   the window ladder: Sy, Syhat, B, U vs Chebyshev radius and vs
           physical radius -- shows how much of each is a windowing artifact
  balance  per-block per-plane measured/predicted charge in matched physical
           windows: the induction-vs-collection charge scale
  profile  charge-weighted transverse (dperp) and longitudinal (dpar) profiles,
           measured vs predicted, split by drift-time tercile (H1: a diffusion
           mismatch grows as sqrt(t_drift), a filter-constant mismatch does not)
  angle    B, U, charge_err/charge and cell counts vs theta_P (H3)
  block    per-block: B and U before/after a free per-plane scale (scale vs
           shape), residual concentration (top 1% of cells' share of sum|y-yh|,
           the H2 discriminator), f_off, dead fraction, busy-ness proxies
"""
import argparse, os, sys
import numpy as np
import uproot
from scipy.spatial import cKDTree

# (u_lo, v_lo) channel bounds; pitch per plane [mm]; slice width [mm]
DET = {
    "pdvd": dict(bounds=(3808, 7616), pitch=(7.65, 7.65, 5.10), slice_mm=4 * 0.5 * 1.48073),
    "sbnd": dict(bounds=(3968, 7936), pitch=(3.00, 3.00, 3.00), slice_mm=4 * 0.5 * 1.563),
    "pdhd": dict(bounds=(3200, 6400), pitch=(4.6693, 4.6693, 4.7920), slice_mm=4 * 0.5 * 1.576),
}
CHEB = [0, 1, 2, 3, 5, 10]
RPHYS = [4.0, 8.0, 12.0, 16.0, 24.0]          # mm, pitch direction; |dpar| <= 1 slice
THETA_EDGES = [0, 15, 30, 45, 60, 75, 90.001]
PERP_EDGES = np.arange(-24, 24.01, 1.5)        # mm
PAR_EDGES = np.arange(-15, 15.01, 1.0)         # mm


def dead_mask(bad, ch, ts, tps):
    m = np.zeros(len(ch), bool)
    if bad is None or len(bad["chid"]) == 0:
        return m
    by = {}
    for c, s, e in zip(bad["chid"], bad["start_time"], bad["end_time"]):
        by.setdefault(int(c), []).append((s / tps - 0.5, e / tps + 0.5))
    for j in np.where(np.isin(ch, np.fromiter(by.keys(), dtype=np.int64)))[0]:
        for s, e in by[int(ch[j])]:
            if s <= ts[j] <= e:
                m[j] = True
                break
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--det", required=True, choices=DET)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticks-per-slice", type=int, default=4)
    ap.add_argument("--status", type=int, default=0, help="STM status to keep (-1 = all)")
    ap.add_argument("--max-foff", type=float, default=1.1,
                    help="keep only blocks whose off-trajectory charge fraction (Chebyshev>2, W plane) is below this "
                         "-- the unfused subset, to separate H2 (overlapping activity) from H1 (smearing)")
    a = ap.parse_args()
    D = DET[a.det]
    bounds, PITCH, SMM = D["bounds"], D["pitch"], D["slice_mm"]

    # accumulators
    win = {p: {"cheb_y": np.zeros(len(CHEB)), "cheb_h": np.zeros(len(CHEB)),
               "cheb_a": np.zeros(len(CHEB)), "phys_y": np.zeros(len(RPHYS)),
               "phys_h": np.zeros(len(RPHYS)), "phys_a": np.zeros(len(RPHYS)),
               "tot_y": 0.0, "tot_h": 0.0} for p in "UVW"}
    prof = {p: {k: np.zeros(len(PERP_EDGES) - 1) for k in ("y", "h")} for p in "UVW"}
    prof3 = {p: {t: {k: np.zeros(len(PERP_EDGES) - 1) for k in ("y", "h")} for t in range(3)} for p in "UVW"}
    lprof = {p: {k: np.zeros(len(PAR_EDGES) - 1) for k in ("y", "h")} for p in "UVW"}
    ang = {p: {k: np.zeros(len(THETA_EDGES) - 1) for k in ("y", "h", "abs", "n", "err", "q")} for p in "UVW"}
    blocks, balrows = [], []
    drift_all = []

    for path in a.roots:
        try:
            f = uproot.open(path)
            ks = {k.split(";")[0] for k in f.keys()}
            if "T_proj_data" not in ks or "T_rec_charge" not in ks:
                continue
            r = f["T_rec_charge"].arrays(["pu", "pv", "pw", "pt", "rr", "ndf", "status", "x"], library="np")
            d = f["T_proj_data"].arrays(library="np")
            bad = f["T_bad_ch"].arrays(["chid", "start_time", "end_time"], library="np") if "T_bad_ch" in ks else None
        except Exception as ex:
            print("skip", path, ex, file=sys.stderr)
            continue
        ev = os.path.basename(os.path.dirname(path))
        ids = [int(c) for c in d["cluster_id"][0]]
        nblk_ev = sum(1 for j, b in enumerate(ids)
                      if (r["ndf"] == b).sum() >= 5 and (a.status < 0 or int(r["status"][r["ndf"] == b][0]) == a.status))
        for i, blk in enumerate(ids):
            m = r["ndf"] == blk
            if m.sum() < 5:
                continue
            st = int(r["status"][m][0])
            if a.status >= 0 and st != a.status:
                continue
            ch = np.asarray(list(d["channel"][0][i]), dtype=np.int64)
            ts = np.asarray(list(d["time_slice"][0][i]), dtype=np.int64)
            q = np.asarray(list(d["charge"][0][i]), float)
            qe = np.asarray(list(d["charge_err"][0][i]), float)
            qp = np.asarray(list(d["charge_pred"][0][i]), float)
            pl = np.digitize(ch, bounds)
            dead = dead_mask(bad, ch, ts, a.ticks_per_slice)
            live = (q > 0) & ~dead
            # trajectory, ordered along the track by residual range
            order = np.argsort(-r["rr"][m])
            pts = {k: r[k][m][order] for k in ("pu", "pv", "pw", "pt", "rr", "x")}
            npts = len(order)
            # local direction by central difference (per plane, in cells)
            drow = {}
            for P, key in enumerate(("pu", "pv", "pw")):
                w = pts[key]; t = pts["pt"]
                iw = np.gradient(w) if npts > 2 else np.full(npts, w[-1] - w[0])
                it = np.gradient(t) if npts > 2 else np.full(npts, t[-1] - t[0])
                drow[P] = np.degrees(np.arctan2(np.abs(it) * SMM, np.abs(iw) * PITCH[P] + 1e-9))
            # doc 42 sec 7: gate on the block's own off-trajectory fraction (W plane,
            # Chebyshev > 2) so the profile/angle accumulators can be restricted to
            # unfused blocks without changing the metric definitions.
            if a.max_foff < 1.0:
                mpW = pl == 2
                if mpW.sum() == 0:
                    continue
                twr = cKDTree(np.column_stack([np.round(pts["pw"]), np.round(pts["pt"])]).astype(float))
                dW, _ = twr.query(np.column_stack([ch[mpW], ts[mpW]]).astype(float), p=np.inf)
                lvW = live[mpW]
                qW = q[mpW]
                if qW[lvW].sum() <= 0 or 1 - qW[lvW & (dW <= 2)].sum() / qW[lvW].sum() > a.max_foff:
                    continue
            brow = {"det": a.det, "event": ev, "block": blk, "status": st, "npts": npts,
                    "len_cm": float(pts["rr"].max()), "absx_cm": float(np.median(np.abs(pts["x"]))),
                    "nblk_ev": nblk_ev}
            bal = {}
            for P, key in enumerate(("pu", "pv", "pw")):
                p_ = "UVW"[P]
                mp = pl == P
                if mp.sum() == 0:
                    continue
                pcr = np.round(pts[key]).astype(float)
                ptr = np.round(pts["pt"]).astype(float)
                tree = cKDTree(np.column_stack([pcr, ptr]))
                cw = ch[mp].astype(float); ct = ts[mp].astype(float)
                Q = q[mp]; QP = qp[mp]; QE = qe[mp]; lv = live[mp]
                dist, idx = tree.query(np.column_stack([cw, ct]), p=np.inf)
                dperp = (cw - pts[key][idx]) * PITCH[P]
                dpar = (ct - pts["pt"][idx]) * SMM
                th = drow[P][idx]
                # drift-time tercile by |x| of the nearest point
                ax = np.abs(pts["x"][idx])
                drift_all.append(np.abs(pts["x"]))
                W = win[p_]
                for j, k in enumerate(CHEB):
                    s = lv & (dist <= k)
                    W["cheb_y"][j] += Q[s].sum(); W["cheb_h"][j] += QP[s].sum()
                    W["cheb_a"][j] += np.abs(Q[s] - QP[s]).sum()
                for j, R in enumerate(RPHYS):
                    s = lv & (np.abs(dperp) <= R) & (np.abs(dpar) <= 1.01 * SMM)
                    W["phys_y"][j] += Q[s].sum(); W["phys_h"][j] += QP[s].sum()
                    W["phys_a"][j] += np.abs(Q[s] - QP[s]).sum()
                W["tot_y"] += Q[lv].sum(); W["tot_h"] += QP.sum()
                # profiles (|dpar| <= 1 slice for transverse, |dperp| <= 1 pitch for longitudinal)
                st_ = lv & (np.abs(dpar) <= 1.01 * SMM)
                prof[p_]["y"] += np.histogram(dperp[st_], PERP_EDGES, weights=Q[st_])[0]
                prof[p_]["h"] += np.histogram(dperp[st_], PERP_EDGES, weights=QP[st_])[0]
                sl_ = lv & (np.abs(dperp) <= 1.01 * PITCH[P])
                lprof[p_]["y"] += np.histogram(dpar[sl_], PAR_EDGES, weights=Q[sl_])[0]
                lprof[p_]["h"] += np.histogram(dpar[sl_], PAR_EDGES, weights=QP[sl_])[0]
                # H1: same transverse profile in three |x| (drift-time) bands
                xb = np.digitize(ax, XBAND[a.det])
                for t3 in range(3):
                    s3 = st_ & (xb == t3)
                    if s3.sum():
                        prof3[p_][t3]["y"] += np.histogram(dperp[s3], PERP_EDGES, weights=Q[s3])[0]
                        prof3[p_][t3]["h"] += np.histogram(dperp[s3], PERP_EDGES, weights=QP[s3])[0]
                # H3: angle bins, inside the physical 12 mm window
                sA = lv & (np.abs(dperp) <= 12.0) & (np.abs(dpar) <= 1.01 * SMM)
                ib = np.digitize(th[sA], THETA_EDGES) - 1
                for b in range(len(THETA_EDGES) - 1):
                    s = ib == b
                    if not s.any():
                        continue
                    A = ang[p_]
                    A["y"][b] += Q[sA][s].sum(); A["h"][b] += QP[sA][s].sum()
                    A["abs"][b] += np.abs(Q[sA][s] - QP[sA][s]).sum()
                    A["n"][b] += s.sum(); A["err"][b] += QE[sA][s].sum(); A["q"][b] += Q[sA][s].sum()
                # per-block numbers, in the Chebyshev<=2 window (100% of the prediction)
                s2 = lv & (dist <= 2)
                sy = Q[s2].sum(); sh = QP[s2].sum()
                sy12 = Q[lv & (np.abs(dperp) <= 12.0) & (np.abs(dpar) <= 1.01 * SMM)].sum()
                sh12 = QP[(np.abs(dperp) <= 12.0) & (np.abs(dpar) <= 1.01 * SMM)].sum()
                bal[p_] = (sy, sh, sy12, sh12)
                if sy > 0 and s2.sum() > 10:
                    resid = np.abs(Q[s2] - QP[s2])
                    k1 = max(1, int(0.01 * len(resid)))
                    top1 = np.sort(resid)[-k1:].sum() / max(resid.sum(), 1e-9)
                    scale = (Q[s2] * QP[s2]).sum() / max((QP[s2] ** 2).sum(), 1e-9)
                    Us = np.abs(Q[s2] - scale * QP[s2]).sum() / sy
                    brow.update({f"B_{p_}": sh / sy - 1, f"U_{p_}": resid.sum() / sy,
                                 f"Uscaled_{p_}": Us, f"scale_{p_}": scale, f"top1_{p_}": top1,
                                 f"foff_{p_}": 1 - sy / max(Q[lv].sum(), 1e-9),
                                 f"ffar_{p_}": Q[lv & (dist > 5)].sum() / max(Q[lv].sum(), 1e-9),
                                 f"dead_{p_}": (dead & mp).sum() / max(mp.sum(), 1),
                                 f"theta_{p_}": float(np.median(drow[P]))})
            if len(bal) == 3 and all(v[0] > 0 for v in bal.values()):
                balrows.append({"det": a.det, "event": ev, "block": blk,
                                "npts": npts, "nblk_ev": nblk_ev,
                                "foff": brow.get("foff_W", float("nan")),
                                "mUW": bal["U"][0] / bal["W"][0], "mVW": bal["V"][0] / bal["W"][0],
                                "pUW": bal["U"][1] / max(bal["W"][1], 1e-9), "pVW": bal["V"][1] / max(bal["W"][1], 1e-9),
                                "mUW12": bal["U"][2] / max(bal["W"][2], 1e-9), "mVW12": bal["V"][2] / max(bal["W"][2], 1e-9),
                                "pUW12": bal["U"][3] / max(bal["W"][3], 1e-9), "pVW12": bal["V"][3] / max(bal["W"][3], 1e-9)})
            if len(brow) > 8:
                blocks.append(brow)

    def dump(name, rows):
        if not rows:
            return
        cols = sorted({k for r_ in rows for k in r_})
        head = [c for c in ("det", "event", "block", "status", "npts", "len_cm", "absx_cm", "nblk_ev") if c in cols]
        cols = head + [c for c in cols if c not in head]
        with open(a.out + "_" + name + ".tsv", "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r_ in rows:
                fh.write("\t".join(("%.5g" % r_[c] if isinstance(r_.get(c), float) else str(r_.get(c, ""))) for c in cols) + "\n")

    dump("block", blocks)
    dump("balance", balrows)
    with open(a.out + "_window.tsv", "w") as fh:
        fh.write("plane\tkind\tradius\tsum_y\tsum_yhat\tsum_abs\tB\tU\tfrac_y\tfrac_yhat\n")
        for p in "UVW":
            W = win[p]
            for j, k in enumerate(CHEB):
                fh.write("%s\tcheb\t%d\t%.6g\t%.6g\t%.6g\t%+.4f\t%.4f\t%.4f\t%.4f\n" % (
                    p, k, W["cheb_y"][j], W["cheb_h"][j], W["cheb_a"][j],
                    W["cheb_h"][j] / max(W["cheb_y"][j], 1e-9) - 1, W["cheb_a"][j] / max(W["cheb_y"][j], 1e-9),
                    W["cheb_y"][j] / max(W["tot_y"], 1e-9), W["cheb_h"][j] / max(W["tot_h"], 1e-9)))
            for j, R in enumerate(RPHYS):
                fh.write("%s\tphys_mm\t%.1f\t%.6g\t%.6g\t%.6g\t%+.4f\t%.4f\t%.4f\t%.4f\n" % (
                    p, R, W["phys_y"][j], W["phys_h"][j], W["phys_a"][j],
                    W["phys_h"][j] / max(W["phys_y"][j], 1e-9) - 1, W["phys_a"][j] / max(W["phys_y"][j], 1e-9),
                    W["phys_y"][j] / max(W["tot_y"], 1e-9), W["phys_h"][j] / max(W["tot_h"], 1e-9)))
    with open(a.out + "_profile.tsv", "w") as fh:
        fh.write("plane\tband\taxis\tcentre_mm\tsum_y\tsum_yhat\n")
        c = 0.5 * (PERP_EDGES[1:] + PERP_EDGES[:-1])
        cl = 0.5 * (PAR_EDGES[1:] + PAR_EDGES[:-1])
        for p in "UVW":
            for j in range(len(c)):
                fh.write("%s\tall\tperp\t%.3f\t%.6g\t%.6g\n" % (p, c[j], prof[p]["y"][j], prof[p]["h"][j]))
            for j in range(len(cl)):
                fh.write("%s\tall\tpar\t%.3f\t%.6g\t%.6g\n" % (p, cl[j], lprof[p]["y"][j], lprof[p]["h"][j]))
            for t3 in range(3):
                for j in range(len(c)):
                    fh.write("%s\txband%d\tperp\t%.3f\t%.6g\t%.6g\n" % (p, t3, c[j], prof3[p][t3]["y"][j], prof3[p][t3]["h"][j]))
    with open(a.out + "_angle.tsv", "w") as fh:
        fh.write("plane\ttheta_lo\ttheta_hi\tN\tsum_y\tsum_yhat\tsum_abs\tB\tU\trel_err\n")
        for p in "UVW":
            A = ang[p]
            for b in range(len(THETA_EDGES) - 1):
                if A["n"][b] == 0:
                    continue
                fh.write("%s\t%g\t%g\t%d\t%.6g\t%.6g\t%.6g\t%+.4f\t%.4f\t%.4f\n" % (
                    p, THETA_EDGES[b], THETA_EDGES[b + 1], A["n"][b], A["y"][b], A["h"][b], A["abs"][b],
                    A["h"][b] / max(A["y"][b], 1e-9) - 1, A["abs"][b] / max(A["y"][b], 1e-9),
                    A["err"][b] / max(A["q"][b], 1e-9)))
    print("%s: %d blocks, %d balance rows -> %s_{block,balance,window,profile,angle}.tsv"
          % (a.det, len(blocks), len(balrows), a.out))


XBAND = {"pdvd": [113.0, 226.0], "sbnd": [66.7, 133.3],
         "pdhd": [117.7, 235.5]}   # |x| cm terciles of the drift (PDHD anode |x| 353.1 cm)
#   (T_rec_charge x, rr, nq are all in cm; the drift half-width is 338 cm PDVD, 200 cm SBND)

if __name__ == "__main__":
    sys.exit(main())
