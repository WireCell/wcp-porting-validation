#!/usr/bin/env python3
"""doc pdvd/100 -- the QtoL crosser fit split into its two halves (measured only; production keeps QtoL 0.094).

    python3 d100_qtol_halves.py --tag p99wflip --tag p99rwon > d100/qtol_halves.txt

pdvd/ql_light_calib/fit_qtol_crossers.py sums the two halves' predictions (pr = pb + pt), so a top-only charge scale is
diluted by the bottom half.  This keeps pb and pt apart: the anchor harvest is DUPLICATED from that script (its strict
geometric cathode-crosser triplets, same cuts, same keep mask), not imported, so the production calibration script stays
untouched.  Per anchor with both halves bundled, a non-negative least-squares fit over the kept channels

    meas_ch = a_b * pb_ch + a_t * pt_ch

gives the bottom and top light-to-prediction factors; the medians over anchors are reported per arm, with the pooled
estimator of the production script (median Sum meas / Sum (pb+pt)) as its reproduction check.
Prediction (doc 100 prereg sec 3): a_t(gain ON) / a_t(gain OFF) ~ 1/1.125; a_b unchanged.
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.optimize import nnls

HERE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/ql_light_calib"
sys.path.insert(0, HERE)
import fit_qtol_crossers as F  # noqa: E402  (dump_paths, cluster_geom, U_CATH, med only)


def harvest(paths):
    out = []
    for path in paths:
        d = json.load(open(path))
        v = d["drift_speed"]
        geom = d["geometry"]
        qtol = d["quality_params"]["QtoL"]
        keep_static = np.array([od["active"] and not od["auto_masked"] for od in d["opdets"]], bool)
        pred = {(b["flash_gid"], b["main_cluster"]): b["pred_pe"] for b in d["bundles"]}
        bots, tops = [], []
        for c in d["clusters"]:
            if c["npoints"] < 400:
                continue
            apa = 0 if c["uid"] < 4000000 else 4
            umax0, y, z, span = F.cluster_geom(c, apa, geom)
            if span < 40.0:
                continue
            (bots if apa == 0 else tops).append((c["uid"], umax0, y, z))
        flashes = d["flashes"]
        for f in flashes:
            if f["total_PE"] < 1500:
                continue
            if any(g is not f and abs(g["time"] - f["time"]) < 30 and g["total_PE"] > 0.25 * f["total_PE"] for g in flashes):
                continue
            for ub, umb, yb, zb in bots:
                if abs(umb - v * f["time"] - F.U_CATH) > 3.0:
                    continue
                for ut, umt, yt, zt in tops:
                    if abs(umt - v * f["time1"] - F.U_CATH) > 3.0:
                        continue
                    if np.hypot(yb - yt, zb - zt) > 20.0:
                        continue
                    pb, pt = pred.get((f["gid"], ub)), pred.get((f["gid"], ut))
                    if pb is None and pt is None:
                        continue
                    sat = np.asarray(f.get("sat", [0] * 40), bool)
                    cov = np.asarray(f.get("cov", [1.0] * 40), float)
                    out.append(dict(evt=d["charge_ident"], gid=f["gid"], meas=np.asarray(f["pe"], float),
                                    pb=None if pb is None else np.asarray(pb, float),
                                    pt=None if pt is None else np.asarray(pt, float),
                                    keep=keep_static & ~sat & (cov >= 1.0), qtol=qtol))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    a = ap.parse_args()
    print("# doc pdvd/100 -- QtoL crosser anchors split by half (d100_qtol_halves.py); measured only")
    res = {}
    for tag in a.tag:
        anchors = harvest(F.dump_paths(tag))
        pooled, ab, at, fb = [], [], [], []
        for x in anchors:
            k = x["keep"]
            pr = (x["pb"] if x["pb"] is not None else 0) + (x["pt"] if x["pt"] is not None else 0)
            sp = pr[k].sum()
            if sp > 0 and x["meas"][k].sum() > 0:
                pooled.append(x["meas"][k].sum() / sp)
            if x["pb"] is None or x["pt"] is None:
                continue
            A = np.vstack([x["pb"][k], x["pt"][k]]).T
            if A.shape[0] < 4 or A[:, 0].sum() <= 0 or A[:, 1].sum() <= 0:
                continue
            coef, _ = nnls(A, x["meas"][k])
            ab.append(coef[0]); at.append(coef[1])
            fb.append(A[:, 0].sum() / A.sum())
        qt = {x["qtol"] for x in anchors}
        m = F.med(np.array(pooled))
        print(f"\n== {tag}: dumps {len(F.dump_paths(tag))}, anchors {len(anchors)}, dump QtoL {sorted(qt)}")
        print(f"  pooled (the production script's estimator): median {m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}] n={m[3]}")
        for name, arr in (("a_b (bottom half)", ab), ("a_t (top half)", at), ("bottom share of pred", fb)):
            mm = F.med(np.array(arr))
            print(f"  {name:22s}: median {mm[0]:.3f} [{mm[1]:.3f}, {mm[2]:.3f}] n={mm[3]}")
        res[tag] = dict(pooled=np.median(pooled), ab=np.median(ab), at=np.median(at))
    if len(a.tag) == 2:
        t0, t1 = a.tag
        print(f"\n{t1} / {t0}: pooled {res[t1]['pooled'] / res[t0]['pooled']:.3f}; "
              f"a_b {res[t1]['ab'] / res[t0]['ab']:.3f}; a_t {res[t1]['at'] / res[t0]['at']:.3f} "
              f"(prediction a_t 1/1.125 = {1 / 1.125:.3f}, a_b 1.000)")


if __name__ == "__main__":
    main()
