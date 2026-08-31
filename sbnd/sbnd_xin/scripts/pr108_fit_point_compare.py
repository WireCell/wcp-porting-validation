#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/108 -- per-point comparison of T_rec_charge (fitted
trajectory points + fitted charge) between two arms: WCP prototype
(nue_5384_<sr>_<ev>.root) and/or WCT toolkit (track_com_5384_<ev>.root).

Row order is not reproducible (pr/7 sec 5), so points are matched by
(cluster_id, sub_cluster_id) + nearest 3-D position; a point with no partner
within --match cm counts as "unmatched" (a point-set change).  Frames: x,y,z
are cm on both sides; WCP pu/pv/pw/pt carry the display +0.5 and pt is in
time slices (WCT: ticks, x4) -- neither is used for matching here.  WCP rescales
U-plane charge by 1/0.7 on uBooNE channel ranges (PR3DCluster_multi_dQ_dx_fit.h
~870-885); --undo-u07 divides those points' q back when the arm is WCP.
q is inverted from the stored feature (dQ*0.1-1000, identical on both sides) to raw dQ.

Junction region: the vertices of the main cluster (flag_vertex==1 rows) with
>= 3 distinct sub_cluster_ids within 1.5 cm; a point is "near junction" when
within --rj cm of any such vertex (taken from arm A).

Usage:
  pr108_fit_point_compare.py A.root B.root [--label-a WCP-on --label-b WCT-on]
      [--kind-a wcp|wct] [--kind-b wcp|wct] [--match 0.5] [--rj 3] [--tsv out]
Prints: n points, matched fraction, |dpos| quantiles, dq/q quantiles at
|dpos| < 0.1 cm, all and near-junction.
"""
import argparse, sys
import numpy as np
import uproot

U07_RANGES = [(296, 327), (336, 337), (343, 351), (376, 400), (410, 484), (501, 524), (536, 671)]


def load(path, kind, undo_u07):
    t = uproot.open(path)["T_rec_charge"]
    keys = ["x", "y", "z", "q", "flag_vertex", "cluster_id", "sub_cluster_id", "pu"]
    a = t.arrays(keys, library="np")
    # Both sides write q = dQ*0.1 - 1000 (the DL-net feature: WCP NeutrinoID.cxx:1883/1980,
    # WCT UbooneMagnifyTrackingVisitor.cxx:387/476); invert to raw dQ.
    q = (a["q"].astype(float) + 1000.0) * 10.0
    if kind == "wcp" and undo_u07:
        pu = np.floor(a["pu"]).astype(int)  # display +0.5 -> wire index
        m = np.zeros(len(pu), bool)
        for lo, hi in U07_RANGES:
            m |= (pu >= lo) & (pu <= hi)
        q[m] *= 0.7
    return dict(X=np.stack([a["x"], a["y"], a["z"]], 1).astype(float), q=q,
                fv=a["flag_vertex"] == 1, cl=a["cluster_id"].astype(int), sc=a["sub_cluster_id"].astype(int))


def junctions(d, r=1.5):
    if len(d["X"]) == 0:
        return np.zeros((0, 3))
    vals, cnt = np.unique(d["cl"], return_counts=True)
    main = vals[cnt.argmax()]
    J = []
    for i in np.where(d["fv"] & (d["cl"] == main))[0]:
        dd = np.linalg.norm(d["X"] - d["X"][i], axis=1)
        near = (dd < r) & (d["sc"] >= 0) & (d["cl"] == main)
        if len(set(d["sc"][near])) >= 3:
            J.append(d["X"][i])
    return np.array(J) if J else np.zeros((0, 3))


def match(A, B, tol, by="pos"):
    """For every point of A find the nearest B point (by="pos": any point;
    by="id": same (cluster_id, sub_cluster_id) -- ids are NOT stable across
    implementations or across topology changes, so "pos" is the default);
    returns index array (-1 = unmatched) and distance."""
    idx = np.full(len(A["X"]), -1); dist = np.full(len(A["X"]), np.inf)
    if len(B["X"]) == 0:
        return idx, dist
    key_b = {}
    for j, (c, s) in enumerate(zip(B["cl"], B["sc"])):
        key_b.setdefault((c, s), []).append(j)
    allb = np.arange(len(B["X"]))
    for i, (c, s) in enumerate(zip(A["cl"], A["sc"])):
        cand = allb if by == "pos" else key_b.get((c, s))
        if cand is None or len(cand) == 0:
            continue
        cand = np.asarray(cand)
        dd = np.linalg.norm(B["X"][cand] - A["X"][i], axis=1)
        k = dd.argmin()
        if dd[k] <= tol:
            idx[i] = cand[k]; dist[i] = dd[k]
    return idx, dist


def qtile(v):
    if len(v) == 0:
        return "n=0"
    p = np.percentile(v, [50, 90, 100])
    return f"n={len(v)} med={p[0]:.3g} p90={p[1]:.3g} max={p[2]:.3g}"


def report(A, B, la, lb, tol, rj, tsv=None, tag="", by="pos"):
    idx, dist = match(A, B, tol, by)
    J = junctions(A)
    nearJ = np.zeros(len(A["X"]), bool)
    for j in J:
        nearJ |= np.linalg.norm(A["X"] - j, axis=1) <= rj
    ok = idx >= 0
    print(f"[{tag}] {la} vs {lb}: A {len(A['X'])} pts, B {len(B['X'])} pts, matched {ok.sum()}/{len(ok)} "
          f"(unmatched A {(~ok).sum()}, B-only {len(B['X']) - len(set(idx[ok]))}), junctions {len(J)}, near-J pts {nearJ.sum()}")
    for name, sel in (("all", np.ones(len(ok), bool)), ("near-junction", nearJ)):
        m = ok & sel
        dpos = dist[m]
        qa = A["q"][m]; qb = B["q"][idx[m]]
        close = dpos < 0.1
        with np.errstate(divide="ignore", invalid="ignore"):
            dq = np.where(qa != 0, (qb - qa) / qa, np.nan)
        dqc = dq[close]; dqc = dqc[np.isfinite(dqc)]
        dv = B["X"][idx[m]] - A["X"][m] if m.any() else np.zeros((0, 3))
        axes = " ".join(f"d{c}={np.median(dv[:, k]) if len(dv) else float('nan'):+.3f}" for k, c in enumerate("xyz"))
        print(f"    {name:14s} |dpos| cm: {qtile(dpos)} [median signed {axes}] ; dq/q at |dpos|<0.1: {qtile(np.abs(dqc))} (signed med {np.median(dqc) if len(dqc) else float('nan'):+.3g}) ; unmatched {(~ok & sel).sum()}")
    if tsv:
        with open(tsv, "a") as f:
            for i in range(len(A["X"])):
                f.write(f"{tag}\t{la}\t{lb}\t{A['cl'][i]}\t{A['sc'][i]}\t{A['X'][i][0]:.4f}\t{A['X'][i][1]:.4f}\t{A['X'][i][2]:.4f}\t{A['q'][i]:.2f}\t"
                        f"{idx[i]}\t{dist[i] if idx[i] >= 0 else -1:.4f}\t{B['q'][idx[i]] if idx[i] >= 0 else float('nan'):.2f}\t{int(nearJ[i])}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--kind-a", default="wcp"); ap.add_argument("--kind-b", default="wct")
    ap.add_argument("--match", type=float, default=0.5); ap.add_argument("--rj", type=float, default=3.0)
    ap.add_argument("--undo-u07", action="store_true"); ap.add_argument("--tsv"); ap.add_argument("--tag", default="")
    ap.add_argument("--by", default="pos", choices=["pos", "id"])
    a = ap.parse_args()
    A = load(a.a, a.kind_a, a.undo_u07); B = load(a.b, a.kind_b, a.undo_u07)
    report(A, B, a.label_a, a.label_b, a.match, a.rj, a.tsv, a.tag, a.by)


if __name__ == "__main__":
    main()
