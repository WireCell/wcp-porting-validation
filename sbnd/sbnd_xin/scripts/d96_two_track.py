#!/usr/bin/env python3
"""doc 96 -- what is the over-clustered in-beam main actually made of?

Sequential RANSAC over the main's blob-sample points (the Bee clustering
layer), then merge lines whose directions agree within 12 deg into one
"direction group".  Reports per group: point count, far-pair length, both
endpoints and the distance from each endpoint to the nearest detector face;
then the contact point between the two largest groups, the opening angle
there, and how much charge each group carries on either side of it.

This is a DESCRIPTION of the picture, not a proposed algorithm.  It exists so
the doc can quote endpoints, contact distance and opening angle instead of
eyeballing the projections, and so the crossing-vs-kink question is decided by
counts rather than by impression.  Near a shallow crossing the direction
assignment is ambiguous within roughly +-50 cm of the contact -- read the
arm lengths with that in mind.

Repro:  python3 scripts/d96_two_track.py
"""
import json
import zipfile

import numpy as np
from scipy.spatial import cKDTree

TOL = 4.0                      # cm, RANSAC inlier radius
MERGE_ANG = 12.0               # deg, lines closer than this are one object
rng = np.random.default_rng(20260902)

# SBND active volume, cm.  From the AnodePlane sensvol banner in every QL log:
#   apa0 face0 [(-201.45,-199.965,0) -> (-0.45,199.965,501)]
#   apa1 face1 [(0.45,-199.965,0)    -> (201.45,199.965,501)]
FACES = {"x-anode0": lambda P: P[:, 0] + 201.45, "x-anode1": lambda P: 201.45 - P[:, 0],
         "y-bot": lambda P: P[:, 1] + 199.965,   "y-top": lambda P: 199.965 - P[:, 1],
         "z-up": lambda P: P[:, 2] - 0.0,        "z-down": lambda P: 501.0 - P[:, 2]}

CASES = ((30, 5, "272-2-30"), (21, 15, "105-23-21"))


def ransac(P, nmax=6, iters=4000, minpts=60):
    out, rest, idx = [], P.copy(), np.arange(len(P))
    while len(out) < nmax and len(rest) >= minpts:
        best = (0, None, None)
        for _ in range(iters):
            i, j = rng.integers(0, len(rest), 2)
            if i == j:
                continue
            d = rest[j] - rest[i]
            n = np.linalg.norm(d)
            if n < 40:
                continue
            d = d / n
            k = int((np.linalg.norm(np.cross(rest - rest[i], d), axis=1) < TOL).sum())
            if k > best[0]:
                best = (k, rest[i], d)
        if best[1] is None or best[0] < minpts:
            break
        p0, d = best[1], best[2]
        m = np.linalg.norm(np.cross(rest - p0, d), axis=1) < TOL
        out.append((idx[m], d))
        rest, idx = rest[~m], idx[~m]
    return out, idx


def groups(P):
    lines, leftover = ransac(P)
    objs = []
    for ii, dd in lines:
        for o in objs:
            if abs(float(np.dot(dd, o[1]))) > np.cos(np.radians(MERGE_ANG)):
                o[0].append(ii)
                break
        else:
            objs.append(([ii], dd))
    return [(np.concatenate(ii), dd) for ii, dd in objs], leftover


def main_points(evt, mid):
    with zipfile.ZipFile(f"work-dbg25a-ql/ql_evt{evt}/mabc-all-apa.zip") as z:
        d = json.loads(z.read("data/0/0-clustering-global.json"))
    cid = np.array(d["cluster_id"], int)
    return np.c_[d["x"], d["y"], d["z"]].astype(float)[cid == mid]


for evt, mid, rse in CASES:
    P = main_points(evt, mid)
    objs, leftover = groups(P)
    print(f"=== {rse}  in-beam main cid={mid}  {len(P)} pts -> "
          f"{len(objs)} direction group(s), {len(leftover)} unassigned")
    for k, (I, dd) in enumerate(objs):
        Q = P[I]
        a = int(np.argmax(((Q - Q[0]) ** 2).sum(1)))
        b = int(np.argmax(((Q - Q[a]) ** 2).sum(1)))
        print(f"  object {k}: {len(I):6d} pts  far-pair {np.linalg.norm(Q[a]-Q[b]):6.1f} cm  "
              f"dir {dd.round(3)}")
        for tag, E in (("end1", Q[a]), ("end2", Q[b])):
            dists = {f: float(fn(E[None, :])[0]) for f, fn in FACES.items()}
            f, v = min(dists.items(), key=lambda t: t[1])
            print(f"      {tag} ({E[0]:7.1f},{E[1]:7.1f},{E[2]:7.1f})  "
                  f"nearest face {f} at {v:5.1f} cm")
    A, B = P[objs[0][0]], P[objs[1][0]]
    dmin, _ = cKDTree(A).query(B)
    C = B[int(np.argmin(dmin))]
    ang = np.degrees(np.arccos(min(1, abs(float(np.dot(objs[0][1], objs[1][1]))))))
    print(f"  contact ({C[0]:.1f},{C[1]:.1f},{C[2]:.1f})  closest approach {dmin.min():.2f} cm  "
          f"opening angle {ang:.1f} deg  "
          f"({(dmin < 3).sum()} of {len(B)} object-1 points within 3 cm of object 0)")
    for k, (I, dd) in enumerate(objs[:2]):
        s = (P[I] - C) @ dd
        print(f"    object {k} about the contact: arm- {abs(s.min()):6.1f} cm / arm+ {s.max():6.1f} cm "
              f"({int((s < -5).sum())} / {int((s > 5).sum())} pts either side)")
