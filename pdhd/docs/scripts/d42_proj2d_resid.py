#!/usr/bin/env python3
# PDHD fork of the doc pdvd/42 script of the same name, by DUPLICATION;
# the PDVD file is untouched.  Only the detector tables change: a "pdhd"
# entry whose plane thresholds are (3200, 6400) -- the CUMULATIVE per-plane
# channel counts (3200 U, 3200 V, 3840 W), because PdvdMagnifyTrackingVisitor
# writes pu/pv/pw and T_proj_data.channel as base[plane] + the wire's RANK among
# that plane's channels, not as raw LArSoft channel ids.
# See pdhd/docs/stm-tagger-chain.md sec 8.
"""doc pdvd/42 -- measured vs predicted 2-D pixel charge of the STM fit, per
(event, cluster, pass) block and per plane, for PDVD and SBND alike.

Port by duplication of sbnd_xin/scripts/pr109_2d_resid.py (near-vertex boxes,
PR stage) to whole STM blocks.  Reads the standard -stm-fit outputs:
  T_proj_data   cluster_id(*10+pass) / channel / time_slice / charge / charge_err / charge_pred
  T_rec_charge  x y z q nq pu pv pw pt rr ndf(=block) status pass reduced_chi2
  T_stm_pass    kink_num exit_L left_L per (cluster, pass)
  T_bad_ch      chid start_time end_time (ticks)

Cells of a block (owned by the fitted cluster, doc-42 writer) are split by
their Chebyshev distance in (channel, slice) to the nearest fitted point of the
same plane:
  footprint   d <= 1   the fit passed here; the dQ/dx fit is asked to describe it
  off         d >  1   the trajectory never came -- the Steiner-path / coverage question
              (split further: near 1 < d <= 5, far d > 5 wires/slices)
Metrics (pr/109 definitions), per plane and pooled ('ALL'):
  Q_all, Q_foot, f_off = 1 - Q_foot/Q_all          charge the trajectory does not reach
  U_all   = sum|y-yhat| / sum y over ALL live cells   the naive headline
  on the footprint:  N, U, B = (sum yhat - sum y)/sum y, chi2/N with the fit's own
      sigma = sqrt(err^2 + (q*rel)^2 + add^2) (rel 0.075/0.075/0.05, add 0/0/300,
      identical in pdvd_track_fitting.json and sbnd_track_fitting.json),
      uncov = sum y[yhat==0] / sum y, pull rms (1.4826*MAD) and |pull|>3 fraction
  per residual-range bin of the nearest fitted point (footprint cells only):
      sum|y-yhat|, sum y, sum yhat  -> the coupling to the dQ/dx-vs-rr check
Dead cells (T_bad_ch) and cells with charge <= 0 are excluded everywhere.

Usage:
  d42_proj2d_resid.py --det pdvd --out figs/42_proj2d_pdvd  work/*_d42fit/tracking-stm.root
  d42_proj2d_resid.py --det sbnd --out figs/42_proj2d_sbnd  work-stmcamp-d42fit/nusel_evt*/tracking-stm.root
Writes <out>_blocks.tsv (one row per block x plane), <out>_rr.tsv (block x plane x rr bin)
and <out>_pulls.npz (per-plane pull arrays, capped).
"""
import argparse, os, sys
import numpy as np
import uproot
from scipy.spatial import cKDTree

PLANES = {"pdvd": (3808, 7616), "sbnd": (3968, 7936), "pdhd": (3200, 6400)}
REL = (0.075, 0.075, 0.05)
ADD = (0.0, 0.0, 300.0)
RR_EDGES = [0, 2, 5, 10, 20, 40, 1e9]
RR_NAMES = ["0-2", "2-5", "5-10", "10-20", "20-40", "40+"]
PULL_CAP = 400000


def event_label(path):
    d = os.path.basename(os.path.dirname(path))
    return d


def dead_mask(bad, ch, ts, ticks_per_slice):
    """cells on a dead (channel, slice window)."""
    m = np.zeros(len(ch), bool)
    if bad is None or len(bad["chid"]) == 0:
        return m
    by = {}
    for c, s, e in zip(bad["chid"], bad["start_time"], bad["end_time"]):
        by.setdefault(int(c), []).append((s / ticks_per_slice - 0.5, e / ticks_per_slice + 0.5))
    for j in np.where(np.isin(ch, np.fromiter(by.keys(), dtype=np.int64)))[0]:
        for s, e in by[int(ch[j])]:
            if s <= ts[j] <= e:
                m[j] = True; break
    return m


def robust_rms(v):
    if len(v) == 0: return float("nan")
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--det", required=True, choices=PLANES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticks-per-slice", type=int, default=4)
    ap.add_argument("--min-npts", type=int, default=5)
    a = ap.parse_args()
    bounds = PLANES[a.det]

    cols = ("det event block cluster pass status npts length_cm kink_num exit_L left_L plane "
            "Ncells Nlive Ndead Nfoot Q_all Q_foot f_off U_all U_foot B_foot chi2N_foot uncov_foot "
            "pull_rms pull_tail3 Nfoot_pred absx_med fit_chi2_med f_off_near f_off_far").split()
    rows = []
    rr_rows = []
    pulls = {p: [] for p in "UVW"}

    for path in a.roots:
        try:
            f = uproot.open(path)
            r = f["T_rec_charge"].arrays(["pu", "pv", "pw", "pt", "rr", "ndf", "status", "pass", "x", "reduced_chi2"], library="np")
            d = f["T_proj_data"].arrays(library="np")
            sp = f["T_stm_pass"].arrays(library="np")
            bad = f["T_bad_ch"].arrays(["chid", "start_time", "end_time"], library="np") if "T_bad_ch" in {k.split(";")[0] for k in f.keys()} else None
        except Exception as ex:
            print("skip", path, ex, file=sys.stderr); continue
        ev = event_label(path)
        blocks = [int(c) for c in d["cluster_id"][0]]
        pass_info = {(int(c) * 10 + int(p)): (int(k), float(eL), float(lL))
                     for c, p, k, eL, lL in zip(sp["cluster_id"], sp["pass"], sp["kink_num"], sp["exit_L"], sp["left_L"])}
        for i, blk in enumerate(blocks):
            m = r["ndf"] == blk
            npts = int(m.sum())
            if npts < a.min_npts: continue
            st = int(r["status"][m][0]); pas = int(r["pass"][m][0])
            length = float(r["rr"][m].max())
            absx = "%.1f" % float(np.median(np.abs(r["x"][m]))); fchi2 = "%.3f" % float(np.nanmedian(r["reduced_chi2"][m]))
            kink, exL, lfL = pass_info.get(blk, (-1, float("nan"), float("nan")))
            ch = np.asarray(list(d["channel"][0][i]), dtype=np.int64)
            ts = np.asarray(list(d["time_slice"][0][i]), dtype=np.int64)
            q = np.asarray(list(d["charge"][0][i]), dtype=float)
            qe = np.asarray(list(d["charge_err"][0][i]), dtype=float)
            qp = np.asarray(list(d["charge_pred"][0][i]), dtype=float)
            pl = np.digitize(ch, bounds)
            dead = dead_mask(bad, ch, ts, a.ticks_per_slice)
            live = (q > 0) & ~dead
            tot = {}
            for P, key in enumerate(("pu", "pv", "pw")):
                mp = pl == P
                pc = np.round(r[key][m]).astype(np.int64); pt = np.round(r["pt"][m]).astype(np.int64)
                rr = r["rr"][m]
                if mp.sum() == 0 or len(pc) == 0:
                    continue
                tree = cKDTree(np.column_stack([pc, pt]).astype(float))
                dist, idx = tree.query(np.column_stack([ch[mp], ts[mp]]).astype(float), p=np.inf)
                sel = np.where(mp)[0]
                lv = live[sel]
                foot = lv & (dist <= 1.0)
                y = q[sel]; yh = qp[sel]; ye = qe[sel]
                Q_all = float(y[lv].sum()); Q_foot = float(y[foot].sum())
                Q_near = float(y[lv & (dist > 1.0) & (dist <= 5.0)].sum()); Q_far = float(y[lv & (dist > 5.0)].sum())
                sig = np.sqrt(ye ** 2 + (y * REL[P]) ** 2 + ADD[P] ** 2)
                pull = (y - yh) / np.maximum(sig, 1e-6)
                Nf = int(foot.sum())
                U_all = float(np.abs(y[lv] - yh[lv]).sum() / Q_all) if Q_all > 0 else float("nan")
                U_f = float(np.abs(y[foot] - yh[foot]).sum() / Q_foot) if Q_foot > 0 else float("nan")
                B_f = float((yh[foot].sum() - Q_foot) / Q_foot) if Q_foot > 0 else float("nan")
                chi2N = float((pull[foot] ** 2).sum() / Nf) if Nf else float("nan")
                uncov = float(y[foot & (yh == 0)].sum() / Q_foot) if Q_foot > 0 else float("nan")
                cov = foot & (yh > 0)
                prms = robust_rms(pull[cov]); ptail = float((np.abs(pull[cov]) > 3).mean()) if cov.sum() else float("nan")
                Npred = int(cov.sum())
                if cov.sum():
                    pulls["UVW"[P]].append(pull[cov])
                rows.append([a.det, ev, blk, blk // 10, pas, st, npts, "%.1f" % length, kink, "%.1f" % exL, "%.1f" % lfL, "UVW"[P],
                             int(mp.sum()), int(lv.sum()), int((dead & mp).sum()), Nf, "%.0f" % Q_all, "%.0f" % Q_foot,
                             "%.4f" % (1 - Q_foot / Q_all) if Q_all > 0 else "nan", "%.4f" % U_all, "%.4f" % U_f, "%+.4f" % B_f,
                             "%.3f" % chi2N, "%.4f" % uncov, "%.3f" % prms, "%.4f" % ptail, Npred, absx, fchi2,
                             "%.4f" % (Q_near / Q_all) if Q_all > 0 else "nan", "%.4f" % (Q_far / Q_all) if Q_all > 0 else "nan"])
                tot.setdefault("Q_all", 0.0); tot["Q_all"] += Q_all
                tot.setdefault("Q_foot", 0.0); tot["Q_foot"] += Q_foot
                tot.setdefault("Q_near", 0.0); tot["Q_near"] += Q_near
                tot.setdefault("Q_far", 0.0); tot["Q_far"] += Q_far
                tot.setdefault("abs_all", 0.0); tot["abs_all"] += float(np.abs(y[lv] - yh[lv]).sum())
                tot.setdefault("abs_f", 0.0); tot["abs_f"] += float(np.abs(y[foot] - yh[foot]).sum())
                tot.setdefault("yh_f", 0.0); tot["yh_f"] += float(yh[foot].sum())
                tot.setdefault("chi2", 0.0); tot["chi2"] += float((pull[foot] ** 2).sum())
                tot.setdefault("Nf", 0); tot["Nf"] += Nf
                tot.setdefault("unc", 0.0); tot["unc"] += float(y[foot & (yh == 0)].sum())
                tot.setdefault("Nc", 0); tot["Nc"] += int(mp.sum())
                tot.setdefault("Nl", 0); tot["Nl"] += int(lv.sum())
                tot.setdefault("Nd", 0); tot["Nd"] += int((dead & mp).sum())
                tot.setdefault("Np", 0); tot["Np"] += Npred
                tot.setdefault("pull", []).append(pull[cov])
                # rr bins on the footprint
                rrc = rr[idx]
                for b in range(len(RR_NAMES)):
                    sb = foot & (rrc >= RR_EDGES[b]) & (rrc < RR_EDGES[b + 1])
                    if sb.sum() == 0: continue
                    rr_rows.append([a.det, ev, blk, st, "UVW"[P], RR_NAMES[b], int(sb.sum()),
                                    "%.0f" % np.abs(y[sb] - yh[sb]).sum(), "%.0f" % y[sb].sum(), "%.0f" % yh[sb].sum(),
                                    "%.0f" % y[sb & (yh == 0)].sum()])
            if tot:
                pa = np.concatenate(tot["pull"]) if tot["pull"] else np.array([])
                rows.append([a.det, ev, blk, blk // 10, pas, st, npts, "%.1f" % length, kink, "%.1f" % exL, "%.1f" % lfL, "ALL",
                             tot["Nc"], tot["Nl"], tot["Nd"], tot["Nf"], "%.0f" % tot["Q_all"], "%.0f" % tot["Q_foot"],
                             "%.4f" % (1 - tot["Q_foot"] / tot["Q_all"]) if tot["Q_all"] > 0 else "nan",
                             "%.4f" % (tot["abs_all"] / tot["Q_all"]) if tot["Q_all"] > 0 else "nan",
                             "%.4f" % (tot["abs_f"] / tot["Q_foot"]) if tot["Q_foot"] > 0 else "nan",
                             "%+.4f" % ((tot["yh_f"] - tot["Q_foot"]) / tot["Q_foot"]) if tot["Q_foot"] > 0 else "nan",
                             "%.3f" % (tot["chi2"] / tot["Nf"]) if tot["Nf"] else "nan",
                             "%.4f" % (tot["unc"] / tot["Q_foot"]) if tot["Q_foot"] > 0 else "nan",
                             "%.3f" % robust_rms(pa), "%.4f" % (np.abs(pa) > 3).mean() if len(pa) else "nan", tot["Np"], absx, fchi2,
                             "%.4f" % (tot["Q_near"] / tot["Q_all"]) if tot["Q_all"] > 0 else "nan", "%.4f" % (tot["Q_far"] / tot["Q_all"]) if tot["Q_all"] > 0 else "nan"])

    with open(a.out + "_blocks.tsv", "w") as fh:
        fh.write("# doc pdvd/42 d42_proj2d_resid.py det=%s files=%d\n" % (a.det, len(a.roots)))
        fh.write("\t".join(cols) + "\n")
        for r_ in rows: fh.write("\t".join(str(v) for v in r_) + "\n")
    with open(a.out + "_rr.tsv", "w") as fh:
        fh.write("det\tevent\tblock\tstatus\tplane\trr_bin\tN\tabs\tsum_y\tsum_yhat\tuncov_y\n")
        for r_ in rr_rows: fh.write("\t".join(str(v) for v in r_) + "\n")
    npz = {}
    for p, lst in pulls.items():
        v = np.concatenate(lst) if lst else np.array([])
        if len(v) > PULL_CAP:
            v = np.random.default_rng(42).choice(v, PULL_CAP, replace=False)
        npz[p] = v
    np.savez_compressed(a.out + "_pulls.npz", **npz)
    nb = len({(r_[1], r_[2]) for r_ in rows})
    print("%s: %d files, %d blocks, %d rows -> %s_blocks.tsv, _rr.tsv, _pulls.npz" % (a.det, len(a.roots), nb, len(rows), a.out))


if __name__ == "__main__":
    sys.exit(main())
