#!/usr/bin/env python3
"""doc pdvd/102 sec 5 -- census of the owner's three defects over whole data arms, by angle to drift.

Per STM-fit row (tracking-stm.root T_rec_charge, one run = consecutive rows of one (cluster, pass)
with steps < 3 cm), with the local direction from rows i-3..i+3:
  reg      any reg_flag_u/v/w (TrackFitting.cxx:7842-7930: the row's projection lands on a flag-0 cell
           or on no cell of the fit's charge set)
  lock     >= 2 of pu/pv/pw on an integer or half-integer (a lattice / seed position, not fitted)
  onq_p    the row's OWN cell (round wire, round slice) holds measured charge of this cluster on plane p
           (T_proj_data block cluster*10+pass: PdvdMagnifyTrackingVisitor.cxx:462-472 keeps the fitted
           cluster's own cells); 98.5 % on regularisation-clean W rows (doc 102 sec 2.1)
  offimg   the row is > 3 cm from every clustering-image point of its own cluster (Bee clustering layer)
Per run:  chord  = a stretch of >= 10 consecutive offimg rows (a trajectory crossing a region the image
           does not have), with its arc length and max distance.
Per cluster in the Bee steiner_graph layer: the in-slice nearest-neighbour spacing of the retiled cloud
(points grouped by x to 0.01 cm) -> the 3-wire lattice share.

Usage:
  d102_census.py --out DIR  pdhd:d101hkf pdhd:d101hnew pdvd:d101vkf pdvd:d101vnew [...]
Writes DIR/census_rows.tsv (angle bin x arm), DIR/census_chords.tsv, DIR/census_steiner.tsv, and prints a
summary.
"""
import argparse, glob, json, os, zipfile
import numpy as np
import uproot
from scipy.spatial import cKDTree

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
BASE = {"pdhd": (0, 3200, 6400, 10 ** 7), "pdvd": (0, 3808, 7616, 10 ** 7)}
PITCH_W = {"pdhd": 0.4792, "pdvd": 0.5100}
BINS = [0, 30, 50, 65, 75, 85, 90.01]
PL = ("pu", "pv", "pw")


def load_event(det, work):
    stm = uproot.open(f"{work}/tracking-stm.root")
    t = stm["T_rec_charge"].arrays(["x", "y", "z", "q", "pu", "pv", "pw", "pt", "cluster_id", "pass",
                                    "reg_flag_u", "reg_flag_v", "reg_flag_w"], library="np")
    pd = stm["T_proj_data"].arrays(library="np")
    cells = {}
    if len(pd["cluster_id"]):
        for bi, blk in enumerate(pd["cluster_id"][0]):
            ch = np.asarray(pd["channel"][0][bi]); ts = np.asarray(pd["time_slice"][0][bi]); q = np.asarray(pd["charge"][0][bi])
            per = []
            for p in range(3):
                m = (ch >= BASE[det][p]) & (ch < BASE[det][p + 1]) & (q > 0)
                per.append(set(zip(ch[m].astype(int).tolist(), ts[m].astype(int).tolist())))
            cells[int(blk)] = per
    z = zipfile.ZipFile(f"{work}/mabc-pr.zip")
    img = json.loads(z.read("data/0/0-clustering-global.json"))
    I = np.c_[img["x"], img["y"], img["z"]]; ic = np.array(img["cluster_id"])
    stg = json.loads(z.read("data/0/0-steiner_graph-global.json"))
    S = np.c_[stg["x"], stg["y"], stg["z"]]; sc = np.array(stg["cluster_id"])
    return t, cells, (I, ic), (S, sc)


def runs_of(t, cl, pa):
    idx = np.where((t["cluster_id"] == cl) & (t["pass"] == pa))[0]
    if len(idx) == 0:
        return []
    X = np.c_[t["x"][idx], t["y"][idx], t["z"][idx]]
    st = np.r_[np.inf, np.linalg.norm(np.diff(X, axis=0), axis=1)]
    brk = np.where(st > 3.0)[0]
    return [idx[a:b] for a, b in zip(brk, list(brk[1:]) + [len(idx)])]


def inslice_nn(P):
    xs = np.round(P[:, 0], 2); out = []
    for u in np.unique(xs):
        Q = P[xs == u][:, 1:]
        if len(Q) > 1:
            D = np.linalg.norm(Q[:, None] - Q[None], axis=2); D[D == 0] = np.inf
            out += list(D.min(1))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--exclude", default="",
                    help="comma list of <run6>_<evt> to drop from EVERY arm, so arms are compared on one event set "
                         "(an event whose Bee zip lacks the steiner_graph layer in any arm)")
    ap.add_argument("arms", nargs="+", help="det:ARMTAG")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    excl = set(x for x in a.exclude.split(",") if x)
    rows_out, chord_out, stg_out = [], [], []
    for spec in a.arms:
        det, tag = spec.split(":")
        works = sorted(glob.glob(f"{IMG}/{det}/work/*_{tag}"))
        works = [w for w in works if os.path.exists(f"{w}/tracking-stm.root") and os.path.exists(f"{w}/mabc-pr.zip")
                 and os.path.basename(w)[:-(len(tag) + 1)] not in excl]
        if a.max_events:
            works = works[:a.max_events]
        acc = {k: [] for k in ("ang", "reg", "lock", "onq_pu", "onq_pv", "onq_pw", "offimg")}
        nch = 0; chord_len = []; stg_nn = []
        for w in works:
            ev = os.path.basename(w)[:-(len(tag) + 1)]
            try:
                t, cells, (I, ic), (S, sc) = load_event(det, w)
            except Exception as e:
                print(f"[{tag}] {ev} skipped: {e}"); continue
            X = np.c_[t["x"], t["y"], t["z"]]
            trees = {}
            for cl in np.unique(t["cluster_id"]):
                P = I[ic == cl]
                trees[int(cl)] = cKDTree(P) if len(P) else None
                SP = S[sc == cl]
                if len(SP) > 20:
                    nn = inslice_nn(SP)
                    if len(nn):
                        stg_nn.append(nn)
                for pa in np.unique(t["pass"][t["cluster_id"] == cl]):
                    blk = cells.get(int(cl) * 10 + int(pa))
                    for run in runs_of(t, cl, pa):
                        if len(run) < 20:
                            continue
                        P = X[run]
                        dimg = trees[int(cl)].query(P)[0] if trees[int(cl)] is not None else np.full(len(run), np.inf)
                        off = dimg > 3.0
                        # chords: >= 10 consecutive off-image rows
                        j = 0
                        while j < len(run):
                            if off[j]:
                                k = j
                                while k < len(run) and off[k]:
                                    k += 1
                                if k - j >= 10:
                                    arc = float(np.sum(np.linalg.norm(np.diff(P[j:k], axis=0), axis=1)))
                                    chord_out.append(dict(det=det, arm=tag, event=ev, cluster=int(cl), pass_=int(pa),
                                                          nrows=k - j, arc_cm=round(arc, 1), max_dist_cm=round(float(dimg[j:k].max()), 1)))
                                    nch += 1; chord_len.append(arc)
                                j = k
                            else:
                                j += 1
                        for jj in range(3, len(run) - 3):
                            d = P[jj + 3] - P[jj - 3]; L = np.linalg.norm(d)
                            if L < 1e-6 or L > 6:
                                continue
                            i = run[jj]
                            acc["ang"].append(np.degrees(np.arccos(min(1.0, abs(d[0]) / L))))
                            acc["reg"].append(int(t["reg_flag_u"][i] or t["reg_flag_v"][i] or t["reg_flag_w"][i]))
                            g = [abs(2 * t[p][i] - np.round(2 * t[p][i])) < 1e-3 for p in PL]
                            acc["lock"].append(int(sum(g) >= 2))
                            for p_i, p in enumerate(PL):
                                if blk is None:
                                    acc["onq_" + p].append(np.nan)
                                else:
                                    acc["onq_" + p].append(float((int(np.round(t[p][i])), int(np.round(t["pt"][i]))) in blk[p_i]))
                            acc["offimg"].append(int(off[jj]))
        A = {k: np.array(v, float) for k, v in acc.items()}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (A["ang"] >= lo) & (A["ang"] < hi)
            r = dict(det=det, arm=tag, events=len(works), angle=f"{lo}-{int(hi)}", nrows=int(m.sum()))
            for k in ("reg", "lock", "onq_pu", "onq_pv", "onq_pw", "offimg"):
                r[k] = round(float(np.nanmean(A[k][m])), 4) if m.sum() else np.nan
            rows_out.append(r)
        nn = np.concatenate(stg_nn) if stg_nn else np.array([])
        lat = 3 * PITCH_W[det]
        stg_out.append(dict(det=det, arm=tag, npts=len(nn), nn_med=round(float(np.median(nn)), 3) if len(nn) else np.nan,
                            share_at_3wire=round(float(np.mean(np.abs(nn - lat) < 0.15)), 3) if len(nn) else np.nan,
                            share_below_1wire=round(float(np.mean(nn < 1.2 * PITCH_W[det])), 3) if len(nn) else np.nan))
        print(f"[{det}:{tag}] events {len(works)} rows {len(A['ang'])} chords {nch} (arc med {np.median(chord_len) if chord_len else 0:.1f} cm, "
              f"sum {np.sum(chord_len):.0f} cm) steiner in-slice NN med {stg_out[-1]['nn_med']} 3-wire share {stg_out[-1]['share_at_3wire']}")
    for name, rows in (("census_rows", rows_out), ("census_chords", chord_out), ("census_steiner", stg_out)):
        if not rows:
            continue
        keys = list(rows[0].keys())
        with open(f"{a.out}/{name}.tsv", "w") as fo:
            fo.write("\t".join(keys) + "\n")
            for r in rows:
                fo.write("\t".join(str(r[k]) for k in keys) + "\n")
    for r in rows_out:
        print("\t".join(str(r[k]) for k in r))


if __name__ == "__main__":
    main()
