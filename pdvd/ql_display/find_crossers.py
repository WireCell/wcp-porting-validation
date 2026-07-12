#!/usr/bin/env python3
"""Find cathode-crossing track pairs ("standard candles") in a PDVD calib dump.

A cathode crosser is one physical track reconstructed as two clusters, one per
drift volume (apa 0 = bottom/BDE, apa 4 = top/TDE), that at the TRUE flash T0
meet at the cathode plane (x ~ 0) with aligned directions. Such a pair fixes
the T0 unambiguously (both volumes constrained) and is the best probe of the
light-detector model. Method + thresholds derived from the 4 hand-picked pairs
of run 039252 evt298567 (gids 24/47/61/83): see
docs/ql-cathode-crosser-recipe.md.

Selection (pair level, then flash level):
  1. candidate pair = one apa-0 + one apa-4 cluster sharing >=1 flash gid in
     the bundle set, both with 3-D span >= --min-len (default 50 cm);
  2. transverse continuity: min point-pair (y,z) distance <= --dyz-cut (25 cm)
     [T0-independent, cheap prefilter];
  3. track-axis collinearity: min(global-PCA angle, local-end angle) <=
     --angle-cut (10 deg), folded to [0,90]. Connector angles are NOT used
     (transverse inter-volume shift makes them noise on true crossers), and
     the local direction alone is not trusted (end-curl) -- same lesson as the
     C++ xtpc_joint_pin gate;
  4. meeting point: exact full-cluster 3-D closest approach d in the
     T0-corrected frame (x_corr = x + sign_offset*t_flash*drift_speed).
     Only flashes where the meeting MIDPOINT sits at the cathode,
     |x_mid| <= --xmid-cut (10 cm), are eligible: the 4 hand-picked pairs
     all meet at |x_mid| <= 2 cm, while wrong-T0 overlapping halves meet
     deep inside a volume (evt298567 c109+c4000131: d=8 cm but
     x_mid=-48 cm). Pass if min eligible d <= --d-cut (25 cm, matching the
     C++ xtpc_dmax chosen for PDVD; true pairs sit at 10-22 cm: the two
     active volumes each end ~3 cm from x=0, plus top/bottom timing skew
     and reco -- evt298581 c9+c4000151, a 4 m + 3.6 m crosser at 1.3 deg,
     meets at d=21.8 cm);
  5. flash pick = the eligible shared flash minimizing d; near-ties
     (within --d-tie 3 cm) go to the brighter flash (evt298567 gid61 vs
     gid60 lesson).

Near-misses (passing 1-3 but d-cut < d <= 60 cm, or angle in (10,25] deg) are
reported for eye review but not selected.

Output: a human-readable table (stdout) and, with --emit, a decisions JSONL
usable by make_labels.py in which the crosser bundles are keep/add and every
other auto bundle is rejected (crossers-only tag).

Usage:
  python3 find_crossers.py work/039252_0/calib-evt298567.json
  python3 find_crossers.py work/039252_0/calib-evt298567.json \
      --emit decisions-crossers/decisions-evt298567.jsonl
  # self-check against the 4 hand-picked evt298567 pairs:
  python3 find_crossers.py work/039252_0/calib-evt298567.json --self-check
"""

import argparse
import json
import math
import os

import numpy as np
from scipy.spatial import cKDTree

MIN_CLUS_LEN = 5.0   # viewer length filter (make_labels contract)

KNOWN_298567 = {24: (21, 4000097), 47: (8, 4000069),
                61: (134, 4000062), 83: (96, 4000076)}


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    for g, f in enumerate(sorted(d["flashes"], key=lambda f: f["time"]), start=1):
        f["group"] = g
    d["flash_by_gid"] = {f["gid"]: f for f in d["flashes"]}
    d["cluster_by_uid"] = {c["uid"]: c for c in d["clusters"]}
    d["geometry"] = {int(k): g for k, g in d["geometry"].items()}
    return d


class Clus:
    """Per-cluster cached geometry (raw frame; the T0 correction is a rigid
    x shift, so trees/axes are built once on raw coordinates)."""

    def __init__(self, d, uid):
        c = d["cluster_by_uid"][uid]
        self.uid = uid
        self.apa = c["apa"]
        g = d["geometry"][self.apa]
        self.sign_offset = g["sign_offset"]
        self.P = np.column_stack([np.asarray(c["x"], float),
                                  np.asarray(c["y"], float),
                                  np.asarray(c["z"], float)])
        self.length = float(np.linalg.norm(self.P.max(0) - self.P.min(0))) \
            if len(self.P) else 0.0
        self.tree3 = cKDTree(self.P) if len(self.P) else None
        self.tree_yz = cKDTree(self.P[:, 1:]) if len(self.P) else None
        self.gax = self._pca(self.P)

    @staticmethod
    def _pca(P):
        if len(P) < 3:
            return None
        C = P - P.mean(0)
        _, _, vt = np.linalg.svd(C, full_matrices=False)
        return vt[0] / np.linalg.norm(vt[0])

    def local_axis(self, pt, r=15.0):
        idx = self.tree3.query_ball_point(pt, r)
        return self._pca(self.P[idx])


def fold_angle(u, v):
    if u is None or v is None:
        return float("nan")
    c = abs(float(np.dot(u, v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("--min-len", type=float, default=50.0)
    ap.add_argument("--dyz-cut", type=float, default=25.0)
    ap.add_argument("--angle-cut", type=float, default=10.0)
    ap.add_argument("--d-cut", type=float, default=25.0)
    ap.add_argument("--xmid-cut", type=float, default=10.0)
    ap.add_argument("--d-tie", type=float, default=3.0)
    ap.add_argument("--near-d", type=float, default=60.0)
    ap.add_argument("--near-angle", type=float, default=25.0)
    ap.add_argument("--emit", metavar="JSONL",
                    help="write a crossers-only decisions JSONL for make_labels.py")
    ap.add_argument("--self-check", action="store_true",
                    help="assert the 4 hand-picked evt298567 pairs are found")
    args = ap.parse_args()

    d = load(args.calib)
    event = os.path.basename(args.calib)[len("calib-"):-len(".json")]
    drift = d["drift_speed"]
    trig = d.get("trigger_offset", 0.0)

    # best-ks bundle index per (gid, uid); never the ident=-1 pseudo-cluster
    best = {}
    for j, b in enumerate(d["bundles"]):
        uid = b["main_cluster"]
        if uid == 3999999 or uid not in d["cluster_by_uid"]:
            continue
        k = (b["flash_gid"], uid)
        if k not in best or b["ks_dis"] < d["bundles"][best[k]]["ks_dis"]:
            best[k] = j

    # shared flashes per (bottom uid, top uid) pair
    by_gid = {}
    for (gid, uid) in best:
        by_gid.setdefault(gid, {0: [], 4: []})[d["cluster_by_uid"][uid]["apa"]].append(uid)
    pair_gids = {}
    for gid, sides in by_gid.items():
        for u0 in sides[0]:
            for u4 in sides[4]:
                pair_gids.setdefault((u0, u4), []).append(gid)

    clus = {}

    def C(uid):
        if uid not in clus:
            clus[uid] = Clus(d, uid)
        return clus[uid]

    found, near = [], []
    for (u0, u4), gids in sorted(pair_gids.items()):
        c0, c4 = C(u0), C(u4)
        if c0.length < args.min_len or c4.length < args.min_len:
            continue
        # (2) transverse continuity, T0-independent
        dyz_min, _ = c4.tree_yz.query(c0.P[:, 1:], k=1)
        if dyz_min.min() > args.dyz_cut:
            continue
        # (4) exact 3-D closest approach per shared flash (rigid relative x
        # shift: query c4's raw tree with c0 points shifted by dx0 - dx4)
        per_gid = []
        for gid in gids:
            t = d["flash_by_gid"][gid]["time"] + trig
            rel = (c0.sign_offset - c4.sign_offset) * t * drift
            Q = c0.P.copy()
            Q[:, 0] += rel
            dist, jn = c4.tree3.query(Q, k=1)
            i = int(np.argmin(dist))
            # closest points in each cluster's own T0-corrected frame
            dx0 = c0.sign_offset * t * drift
            p0 = c0.P[i] + np.array([dx0, 0.0, 0.0])
            p4 = c4.P[jn[i]] + np.array([c4.sign_offset * t * drift, 0.0, 0.0])
            per_gid.append(dict(gid=gid, d=float(dist[i]),
                                pe=d["flash_by_gid"][gid]["total_PE"],
                                t_us=d["flash_by_gid"][gid]["time"],
                                x_mid=0.5 * (p0[0] + p4[0]),
                                i0=i, j4=int(jn[i]), p0=p0, p4=p4))
        per_gid.sort(key=lambda r: r["d"])
        eligible = [r for r in per_gid if abs(r["x_mid"]) <= args.xmid_cut]
        if not eligible:
            continue
        bestg = eligible[0]
        for r in eligible[1:]:
            if r["d"] - eligible[0]["d"] <= args.d_tie and r["pe"] > bestg["pe"]:
                bestg = r
        # (3) collinearity at the best flash's closest points
        a_glob = fold_angle(c0.gax, c4.gax)
        l0 = c0.local_axis(c0.P[bestg["i0"]])
        l4 = c4.local_axis(c4.P[bestg["j4"]])
        a_loc = fold_angle(l0, l4)
        a_min = min(x for x in (a_glob, a_loc) if not math.isnan(x)) \
            if not (math.isnan(a_glob) and math.isnan(a_loc)) else float("nan")
        rec = dict(u0=u0, u4=u4, len0=c0.length, len4=c4.length,
                   gid=bestg["gid"], t_us=bestg["t_us"], pe=bestg["pe"],
                   d=bestg["d"], x_mid=0.5 * (bestg["p0"][0] + bestg["p4"][0]),
                   dyz=float(np.hypot(bestg["p0"][1] - bestg["p4"][1],
                                      bestg["p0"][2] - bestg["p4"][2])),
                   a_glob=a_glob, a_loc=a_loc, a_min=a_min,
                   n_flashes=len(gids),
                   runner=[(r["gid"], round(r["d"], 1), int(r["pe"]))
                           for r in sorted(per_gid, key=lambda r: r["d"])[:4]])
        if math.isnan(a_min):
            continue
        if bestg["d"] <= args.d_cut and a_min <= args.angle_cut:
            found.append(rec)
        elif bestg["d"] <= args.near_d and a_min <= args.near_angle:
            near.append(rec)

    def fmt(r):
        return (f"gid{r['gid']:4d} t={r['t_us']:9.2f} pe={r['pe']:7.0f} "
                f"c{r['u0']}+c{r['u4']} len {r['len0']:.0f}/{r['len4']:.0f} "
                f"d {r['d']:5.1f} x_mid {r['x_mid']:6.1f} dyz {r['dyz']:5.1f} "
                f"aG {r['a_glob']:5.1f} aL {r['a_loc']:5.1f} "
                f"nfl {r['n_flashes']:2d} alt {r['runner']}")

    found.sort(key=lambda r: r["d"])
    near.sort(key=lambda r: r["d"])
    print(f"# {args.calib}\n# cuts: len>={args.min_len} dyz<={args.dyz_cut} "
          f"angle<={args.angle_cut} d<={args.d_cut}\n")
    print(f"== CROSSERS ({len(found)}) ==")
    for r in found:
        print(fmt(r))
    print(f"\n== NEAR-MISSES ({len(near)}) [not selected; eye-review] ==")
    for r in near:
        print(fmt(r))

    if args.self_check:
        got = {(r["gid"], r["u0"], r["u4"]) for r in found}
        want = {(g, u0, u4) for g, (u0, u4) in KNOWN_298567.items()}
        missing = want - got
        extra = got - want
        print(f"\nself-check vs evt298567 hand scan: "
              f"{len(want & got)}/4 recovered, {len(extra)} extra")
        for m in sorted(missing):
            print("  MISSING", m)
        for e in sorted(extra):
            print("  EXTRA  ", e)
        if missing:
            raise SystemExit(1)

    if args.emit:
        os.makedirs(os.path.dirname(args.emit) or ".", exist_ok=True)
        keep_idx = {}
        for r in found:
            for uid in (r["u0"], r["u4"]):
                j = best[(r["gid"], uid)]
                keep_idx[j] = r
        lines = []
        # crosser bundles: keep (auto) or add (non-auto)
        for j, r in sorted(keep_idx.items()):
            b = d["bundles"][j]
            lines.append({
                "event": event, "group": d["flash_by_gid"][b["flash_gid"]]["group"],
                "flash_gid": b["flash_gid"], "flash_time_us": r["t_us"],
                "apa": b["apa"], "main_cluster_uid": b["main_cluster"],
                "main_cluster_ident": d["cluster_by_uid"][b["main_cluster"]]["ident"],
                "bundle_idx": j,
                "verdict": "keep" if b["auto_selected"] else "add",
                "auto_selected": bool(b["auto_selected"]),
                "confidence": "high",
                "reason": (f"cathode-crosser half (pair c{r['u0']}+c{r['u4']}): "
                           f"d={r['d']:.1f}cm at x_mid={r['x_mid']:.1f}, "
                           f"dyz={r['dyz']:.1f}cm, aG={r['a_glob']:.1f} "
                           f"aL={r['a_loc']:.1f} deg; flash {r['gid']} "
                           f"({r['pe']:.0f} PE) minimizes the meeting distance."),
            })
        # every other auto bundle: reject (crossers-only tag). Short (<=5 cm)
        # autos are rejected too -- make_labels would otherwise implicit-keep
        # them and pollute the crossers-only scan state.
        for j, b in enumerate(d["bundles"]):
            if not b["auto_selected"] or j in keep_idx:
                continue
            uid = b["main_cluster"]
            lines.append({
                "event": event, "group": d["flash_by_gid"][b["flash_gid"]]["group"],
                "flash_gid": b["flash_gid"],
                "flash_time_us": d["flash_by_gid"][b["flash_gid"]]["time"],
                "apa": b["apa"], "main_cluster_uid": uid,
                "main_cluster_ident": d["cluster_by_uid"].get(uid, {}).get("ident"),
                "bundle_idx": j, "verdict": "reject",
                "auto_selected": True, "confidence": "high",
                "reason": "not a cathode-crosser pair half (crossers-only tag).",
            })
        with open(args.emit, "w") as fh:
            for r in lines:
                fh.write(json.dumps(r) + "\n")
        nk = sum(1 for r in lines if r["verdict"] in ("keep", "add"))
        print(f"\nwrote {args.emit}: {nk} crosser halves, "
              f"{len(lines) - nk} rejects")


if __name__ == "__main__":
    main()
