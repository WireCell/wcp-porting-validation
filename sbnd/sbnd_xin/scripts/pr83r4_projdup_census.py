#!/usr/bin/env python3
"""pr/83 round 4 census -- projective duplicates at the main vertex.

A 1-track-1-shower stem reported as TWO main-vertex tracks that overlap in
>= 2 of the 3 wire views while separating in 3D; the charge-starved member
reads far-below-MIP stem dQ/dx (owner report 2026-08-17; exemplars 138009
12094/12095, 168596 14168/14172, 74544 12105/12107).  Round 3's 3D corridor
metric reads 0.14-0.58 on these -- below every op1 gate -- so they are
invisible to pr83r2_census.py.

Metric (calibrated on the three exemplars vs all other vertex pairs, see
doc 83 sec 11): segments with an endpoint within --vtx-tol of the reco
vertex, pairs with chord angle < --angle, 2nd-best per-view 2D overlap
(views (x, cos(a)z - sin(a)y), a in {0,+60,-60} deg; SBND wire angles)
>= --frac at --tol cm, stem dQ/dx ratio (first --stem cm from the vertex)
< --ratio.  Exemplars: angle 12.7-16.5, 2nd-view 0.82-1.00 @1.4, ratio
0.08-0.28; every non-target pair fails BOTH angle (70-90) and overlap
(<= 0.16).

Usage:
  pr83r4_projdup_census.py <arm> [<arm2> ...] [--tsv out.tsv]
      [--frac 0.7] [--tol 1.4] [--angle 20] [--ratio 0.4]
      [--stem 8] [--min-len 5] [--vtx-tol 1.5]

Vertices come from pr_scores_table.py --root <arm> (per-arm, cached).
Exit 0 always (reporting tool, not a gate).
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)

VIEW_DEG = (0.0, 60.0, -60.0)  # SBND wire angles from Y in the YZ plane


def arm_vertices(arm):
    out = subprocess.run(
        [sys.executable, os.path.join(SX, "pr_scores_table.py"), "--root", arm],
        capture_output=True, text=True, check=True).stdout
    lines = out.strip().split("\n")
    idx = {c: i for i, c in enumerate(lines[0].split("\t"))}
    vtx = {}
    for ln in lines[1:]:
        f = ln.split("\t")
        try:
            evt = int(f[idx["event"]])
            vtx[evt] = np.array([float(f[idx["nu_x_cm"]]),
                                 float(f[idx["nu_y_cm"]]),
                                 float(f[idx["nu_z_cm"]])])
        except (ValueError, KeyError, IndexError):
            continue
    return vtx


def bee_segments(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith("track_fit-global.json")]
        if not names:
            return None
        d = json.loads(z.read(names[0]))
    P = np.array([d["x"], d["y"], d["z"]], float).T
    q = np.asarray(d["q"], float)
    rid = np.asarray(d["real_cluster_id"])
    out = {}
    for r in set(rid.tolist()):
        if r < 0:
            continue
        m = rid == r
        if m.sum() < 2:
            continue
        out[int(r)] = (P[m], q[m])
    return out


def seg_len(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def view(P, deg):
    s, c = math.sin(math.radians(deg)), math.cos(math.radians(deg))
    return np.column_stack([P[:, 0], c * P[:, 2] - s * P[:, 1]])


def frac_within(A, B, tol):
    return float(np.mean([np.linalg.norm(B - a, axis=1).min() < tol for a in A]))


def stem_dqdx(P, q, v, stem):
    if np.linalg.norm(P[-1] - v) < np.linalg.norm(P[0] - v):
        P, q = P[::-1], q[::-1]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    m = s < stem
    L = s[m][-1] if m.sum() > 1 else 0
    return q[m].sum() / L if L > 0.5 else -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--tol", type=float, default=1.4)
    ap.add_argument("--angle", type=float, default=20.0)
    ap.add_argument("--ratio", type=float, default=0.4)
    ap.add_argument("--stem", type=float, default=8.0)
    ap.add_argument("--min-len", type=float, default=5.0)
    ap.add_argument("--vtx-tol", type=float, default=1.5)
    a = ap.parse_args()

    rows = []
    total_evts = 0
    for arm in a.arms:
        vtxs = arm_vertices(arm)
        n_arm = 0
        for d in sorted(glob.glob(os.path.join(arm, "pr_evt*"))):
            evt = int(os.path.basename(d).replace("pr_evt", ""))
            if evt not in vtxs:
                continue
            zp = os.path.join(d, "mabc-pr.zip")
            if not os.path.exists(zp):
                continue
            segs = bee_segments(zp)
            if not segs:
                continue
            total_evts += 1
            v = vtxs[evt]
            vc = [r for r, (P, _) in segs.items()
                  if min(np.linalg.norm(P[0] - v), np.linalg.norm(P[-1] - v)) < a.vtx_tol]
            for i in range(len(vc)):
                for j in range(i + 1, len(vc)):
                    (A, qA), (B, qB) = segs[vc[i]], segs[vc[j]]
                    la, lb = seg_len(A), seg_len(B)
                    if min(la, lb) < a.min_len:
                        continue
                    ca, cb = A[-1] - A[0], B[-1] - B[0]
                    den = np.linalg.norm(ca) * np.linalg.norm(cb)
                    if den <= 0:
                        continue
                    ang = math.degrees(math.acos(
                        min(1.0, abs(float(np.dot(ca, cb))) / den)))
                    if ang > a.angle:
                        continue
                    S, L = (A, B) if la <= lb else (B, A)
                    ov = sorted((frac_within(view(S, th), view(L, th), a.tol)
                                 for th in VIEW_DEG), reverse=True)
                    if ov[1] < a.frac:
                        continue
                    da, db = stem_dqdx(A, qA, v, a.stem), stem_dqdx(B, qB, v, a.stem)
                    if da <= 0 or db <= 0:
                        continue
                    ratio = min(da, db) / max(da, db)
                    if ratio >= a.ratio:
                        continue
                    o3 = max(frac_within(S, L, a.tol), frac_within(L, S, a.tol))
                    n_arm += 1
                    rows.append((arm, evt, vc[i], vc[j], la, lb, ang,
                                 ov[0], ov[1], ov[2], o3, da, db, ratio))
                    print(f"{arm} evt {evt}: pair {vc[i]}/{vc[j]} "
                          f"len {la:.1f}/{lb:.1f}cm ang={ang:.1f} "
                          f"views={ov[0]:.2f}/{ov[1]:.2f}/{ov[2]:.2f} o3D={o3:.2f} "
                          f"stemdQdx {da:.0f}/{db:.0f} ratio={ratio:.2f}")
        print(f"# {arm}: {n_arm} projective-dup findings")
    print(f"# TOTAL: {len(rows)} findings over {total_evts} events")

    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("arm\tevent\tseg_a\tseg_b\tlen_a_cm\tlen_b_cm\tangle_deg\t"
                    "view1\tview2\tview3\toverlap3d\tstem_dqdx_a\tstem_dqdx_b\tratio\n")
            for r in rows:
                f.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.3f}"
                                  for x in r) + "\n")
        print(f"# wrote {a.tsv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
