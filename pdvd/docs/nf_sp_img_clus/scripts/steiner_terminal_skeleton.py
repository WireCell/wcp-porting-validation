#!/usr/bin/env python3
"""doc pdvd/37: the two Steiner-terminal metrics doc 31 sec 6.1 owes, plus the
vertex-region test the owner's fourth criterion asks for.

Doc 31 built two of the four grading rows (largest terminal-free run; terminals
per cm).  The third -- "on the track / vertex", i.e. transverse distance to the
FITTED skeleton -- was called "still owed ... needs a PR product the phase dump
does not carry".  It does carry it: `calib-pr-evt*.json` has
`segments[].points[]` (the fitted trajectory), `vertices[].fit` and
`main_vertex`, in the SAME frame and numbering as `steiner[]`.  Verified on
039349/14: a cluster's terminals NN-match its own polyline at median 0.70 cm
while its whole steiner CLOUD matches at 0.65 cm -- so the frames agree and the
only honest form of the metric is terminal-vs-cloud, the cloud being the
matched control.

Two metrics.

C1  LOCALIZATION.  Distance from each terminal to the nearest point OF A LINE
    SEGMENT of its own cluster's fitted polylines (not to the polyline's
    vertices -- most PDVD segments carry 2 points spanning tens of cm, so a
    vertex-only distance would measure segment sampling, not localization).
    Reported as the EXCESS over the same quantity for the whole steiner cloud.
    Only clusters PR actually fitted have a skeleton at all, so the number of
    clusters entering C1 is printed next to every median: this grades
    localization on a selected, non-random subset.

C2  THE VERTEX REGION.  For every PR vertex of degree >= 2:
      * terminals within --core cm of the vertex fit point are VERTEX-CORE
        terminals.  They are counted separately and NOT branch-assigned:
        incident polylines converge there, so nearest-polyline assignment is a
        coin flip exactly where the measurement matters, and "a branch dropped
        to zero" would be a tie-breaking artefact rather than a result.
      * every other terminal within --rv cm is assigned to its nearest INCIDENT
        segment = its branch.
      * the decisive number: how many incident branches drop to ZERO terminals
        inside --rv after greedy thinning at radius R.

    The dump carries no charge (PrDisplayDump.cxx:1113 stores x/y/z/flag only),
    so the true greedy order is not reproducible offline.  Four columns bracket
    it: two orders doc 31 sec 12.6 used (principal axis, seeded shuffle), an
    ADVERSARIAL order (farthest-from-vertex first, so vertex terminals are the
    ones suppressed), and an ORDER-FREE upper bound that needs no charge at all:

        a branch CAN be emptied under some greedy order only if every one of its
        terminals inside R_v has a terminal of a DIFFERENT branch within R.

    That predicate is geometric, so it bounds the failure exactly.  If the bound
    is zero, no ordering can empty a branch and the question is closed.

C3  --exempt X repeats C2 with terminals within X cm of ANY PR vertex exempted
    from thinning, and reports what the exemption costs on the density axis.

Usage:
  python3 steiner_terminal_skeleton.py \
      PDVD:'/path/work/*_d34base/calib-pr-evt*.json' \
      SBND:'/path/work-mcp2k-d97fvpr2/pr_evt*/calib-pr-evt*.json' \
      UB:'/path/sweep/d37ub/*/calib-pr-evt*.json'
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- geometry ---
def seg_point_distance(P, A, B):
    """Distance from each point of P (n,3) to the polyline with vertices V.

    A, B are (m,3) arrays of segment endpoints.  Returns (n,) minima.  Done in
    chunks so a 10^5-point cloud against a 10^3-piece polyline stays in memory.
    """
    if len(A) == 0:
        return np.full(len(P), np.inf)
    out = np.empty(len(P))
    D = B - A                                    # (m,3)
    LL = np.einsum("ij,ij->i", D, D)             # (m,)
    LL_safe = np.where(LL > 0, LL, 1.0)
    step = max(1, int(2_000_000 / max(1, len(A))))
    for lo in range(0, len(P), step):
        Q = P[lo:lo + step]                      # (k,3)
        W = Q[:, None, :] - A[None, :, :]        # (k,m,3)
        t = np.einsum("kmj,mj->km", W, D) / LL_safe
        t = np.clip(t, 0.0, 1.0)
        C = A[None, :, :] + t[:, :, None] * D[None, :, :]
        out[lo:lo + step] = np.linalg.norm(Q[:, None, :] - C, axis=-1).min(1)
    return out


def polyline_pieces(pts):
    """(A, B) endpoint arrays for a list of dump point dicts."""
    V = np.array([[p["x"], p["y"], p["z"]] for p in pts], float)
    if len(V) < 2:
        return V, V                              # a single point is its own piece
    return V[:-1], V[1:]


def greedy_nms(P, order, R, protected=None):
    """Indices kept by greedy suppression at radius R.

    `protected` is a boolean mask of points that are never suppressed and never
    suppress others -- the C3 vertex exemption.
    """
    keep_forced = np.zeros(len(P), bool) if protected is None else protected.copy()
    tree = cKDTree(P)
    dead = np.zeros(len(P), bool)
    kept = []
    for i in order:
        if keep_forced[i] or dead[i]:
            continue
        kept.append(i)
        for j in tree.query_ball_point(P[i], R):
            if j != i and not keep_forced[j]:
                dead[j] = True
    kept.extend(np.nonzero(keep_forced)[0].tolist())
    return np.asarray(sorted(set(kept)), int)


# ------------------------------------------------------------------- input ---
def load(path):
    try:
        return json.load(open(path))
    except Exception as exc:
        print(f"  skip {os.path.basename(path)}: {exc}", file=sys.stderr)
        return None


def cluster_skeleton(d):
    """cluster_id -> (A, B) polyline pieces over all its segments."""
    per = defaultdict(list)
    for s in d.get("segments", []):
        per[s["cluster_id"]].append(polyline_pieces(s["points"]))
    out = {}
    for cid, lst in per.items():
        A = np.vstack([a for a, b in lst if len(a)])
        B = np.vstack([b for a, b in lst if len(b)])
        out[cid] = (A, B)
    return out


def steiner_clusters(d):
    for e in d.get("steiner", []):
        if "flag_terminal" not in e:
            continue
        P = np.stack([e["x"], e["y"], e["z"]], 1).astype(float)
        t = np.asarray(e["flag_terminal"], bool)
        if len(t) != len(P):
            continue
        yield e["cluster_id"], bool(e["is_main_cluster"]), P, t


# ---------------------------------------------------------------- metric 1 ---
def c1_records(d, args):
    skel = cluster_skeleton(d)
    recs = []
    for cid, main, P, t in steiner_clusters(d):
        if cid not in skel or t.sum() < 2 or len(P) < args.minpts:
            continue
        A, B = skel[cid]
        d_cloud = seg_point_distance(P, A, B)
        d_term = d_cloud[t]
        # the principal axis, for the drift-alignment bin
        C = P - P.mean(0)
        lam, vec = np.linalg.eigh(C.T @ C / len(C))
        axis = vec[:, np.argsort(lam)[::-1][0]]
        rec = dict(cid=cid, main=main, npts=len(P), nterm=int(t.sum()),
                   nseg=len(A), cosx=abs(float(axis[0])),
                   term=float(np.median(d_term)), cloud=float(np.median(d_cloud)),
                   term_p90=float(np.percentile(d_term, 90)))
        rec["excess"] = rec["term"] - rec["cloud"]
        # after thinning: does the surviving set drift OFF the skeleton?
        T = P[t]
        tT = (T - P.mean(0)) @ axis
        for R in args.nms:
            kept = greedy_nms(T, np.argsort(tT), R)
            rec[f"term{R}"] = float(np.median(d_term[kept]))
        recs.append(rec)
    return recs


# ---------------------------------------------------------------- metric 2 ---
def c2_records(d, args):
    """One record per PR vertex of degree >= 2.

    The greedy suppression depends only on the CLUSTER's terminal set, not on
    which vertex we are looking at, so every kept-set is computed once per
    cluster and reused for all its vertices.  Recomputing it per vertex made
    this O(vertices x R x orders) NMS passes over the same 10^4-point set.
    """
    segs = d.get("segments", [])
    verts = d.get("vertices", [])
    terms = {cid: P[t] for cid, main, P, t in steiner_clusters(d)}
    seg_by_id = {s["id"]: s for s in segs}
    incident = defaultdict(list)
    for s in segs:
        for vid in (s.get("start_vertex_id"), s.get("end_vertex_id")):
            if vid is not None and vid >= 0:
                incident[vid].append(s["id"])

    # every vertex fit point in this event -- the C3 exemption set
    vpts = np.array([[v["fit"]["x"], v["fit"]["y"], v["fit"]["z"]]
                     for v in verts if "fit" in v], float)

    # ---- per-cluster kept sets, computed once -----------------------------
    want = sorted({v["cluster_id"] for v in verts
                   if "fit" in v and v.get("degree", 0) >= 2
                   and len(incident.get(v["id"], [])) >= 2
                   and v["cluster_id"] in terms and len(terms[v["cluster_id"]]) >= 2})
    kept_cache = {}
    prot_cache = {}
    for cid in want:
        T = terms[cid]
        rng = np.random.default_rng(20260904)
        orders = {
            "ax": np.argsort((T - T.mean(0)) @ np.linalg.eigh(
                    ((T - T.mean(0)).T @ (T - T.mean(0))) / len(T))[1][:, -1]),
            "rnd": rng.permutation(len(T)),
        }
        prot = (np.linalg.norm(T[:, None, :] - vpts[None, :, :], axis=-1).min(1) < args.exempt
                if (args.exempt > 0 and len(vpts)) else np.zeros(len(T), bool))
        prot_cache[cid] = prot
        kept_cache[cid] = {}
        for R in args.nms:
            for tag, order in orders.items():
                kept_cache[cid][(R, tag)] = set(greedy_nms(T, order, R).tolist())
        # the "adv" order and the exemption are vertex-specific only through dV;
        # they are filled in the vertex loop below and cached per (cid, vid).

    recs = []
    for v in verts:
        if "fit" not in v or v.get("degree", 0) < 2:
            continue
        cid = v["cluster_id"]
        inc = incident.get(v["id"], [])
        if len(inc) < 2 or cid not in terms:
            continue
        T = terms[cid]
        if len(T) < 2:
            continue
        V = np.array([v["fit"]["x"], v["fit"]["y"], v["fit"]["z"]], float)
        dV = np.linalg.norm(T - V, axis=1)

        core = dV < args.core
        near = (dV < args.rv) & ~core
        if near.sum() < 1:
            continue

        pieces = [polyline_pieces(seg_by_id[sid]["points"]) for sid in inc]
        DD = np.stack([seg_point_distance(T[near], A, B) for A, B in pieces], 1)
        branch = DD.argmin(1)
        idx_near = np.nonzero(near)[0]
        core_idx = np.nonzero(core)[0]

        rec = dict(vid=v["id"], cid=cid, is_main=bool(v.get("is_main")),
                   degree=int(v["degree"]), nbranch=len(inc),
                   ncore=int(core.sum()), nnear=int(near.sum()), nterm=len(T))
        occ0 = np.array([(branch == k).sum() for k in range(len(inc))])
        rec["branches_occupied"] = int((occ0 > 0).sum())

        adv_ord = np.argsort(-dV)          # farthest from the vertex FIRST
        prot = prot_cache[cid]

        def emptied(kept):
            alive = np.zeros(len(inc), int)
            for j, gi in enumerate(idx_near):
                if gi in kept:
                    alive[branch[j]] += 1
            return int(((occ0 > 0) & (alive == 0)).sum())

        for R in args.nms:
            for tag in ("ax", "rnd"):
                kept = kept_cache[cid][(R, tag)]
                rec[f"lost{R}_{tag}"] = emptied(kept)
                rec[f"core{R}_{tag}"] = int(sum(1 for gi in core_idx if gi in kept))
            kept_adv = set(greedy_nms(T, adv_ord, R).tolist())
            rec[f"lost{R}_adv"] = emptied(kept_adv)
            rec[f"core{R}_adv"] = int(sum(1 for gi in core_idx if gi in kept_adv))
            if args.exempt > 0:
                kept_ex = set(greedy_nms(T, adv_ord, R, protected=prot).tolist())
                rec[f"lost{R}_exempt"] = emptied(kept_ex)
                rec[f"core{R}_exempt"] = int(sum(1 for gi in core_idx if gi in kept_ex))
                rec[f"keep{R}_exempt"] = len(kept_ex) / len(T)
                rec[f"keep{R}_plain"] = len(kept_adv) / len(T)
            # ORDER-FREE BOUND
            atrisk = 0
            for k in range(len(inc)):
                mine = idx_near[branch == k]
                other = idx_near[branch != k]
                if len(mine) == 0 or len(other) == 0:
                    continue
                if np.all(cKDTree(T[other]).query(T[mine])[0] < R):
                    atrisk += 1
            rec[f"bound{R}"] = atrisk
        recs.append(rec)
    return recs


# ----------------------------------------------------------------- report ---
def q(v, fmt="{:5.2f}"):
    v = np.asarray(v, float)
    if not len(v):
        return "  --"
    a, b, c = np.percentile(v, 25), np.median(v), np.percentile(v, 75)
    return (fmt + " [" + fmt + "," + fmt + "]").format(b, a, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="LABEL:glob")
    ap.add_argument("--minpts", type=int, default=50)
    ap.add_argument("--core", type=float, default=1.0, help="vertex-core radius [cm]")
    ap.add_argument("--rv", type=float, default=3.0, help="vertex neighbourhood [cm]")
    ap.add_argument("--exempt", type=float, default=3.0, help="C3 exemption radius [cm], 0=off")
    ap.add_argument("--nms", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    per = {}
    for spec in args.specs:
        label, pattern = spec.split(":", 1)
        paths = sorted(glob.glob(pattern))
        if args.max_files:
            paths = paths[:args.max_files]
        c1, c2 = [], []
        for p in paths:
            d = load(p)
            if d is None:
                continue
            c1 += c1_records(d, args)
            c2 += c2_records(d, args)
        per[label] = (len(paths), c1, c2)
        print(f"== {label}: {len(paths)} dump(s), {len(c1)} cluster(s) with a fitted "
              f"skeleton, {len(c2)} vertex(es) of degree>=2 with terminals")

    print(f"\n-- C1  terminals vs the FITTED skeleton (median [IQR] over clusters; "
          f"cloud = matched control)")
    print(f"  {'arm':10s} {'nclus':>6s} {'terminal':>18s} {'cloud':>18s} {'EXCESS':>18s} "
          f"{'term p90':>18s}")
    for label, (_, c1, _) in per.items():
        if not c1:
            print(f"  {label:10s} (none)")
            continue
        print(f"  {label:10s} {len(c1):6d} {q([r['term'] for r in c1]):>18s} "
              f"{q([r['cloud'] for r in c1]):>18s} {q([r['excess'] for r in c1]):>18s} "
              f"{q([r['term_p90'] for r in c1]):>18s}")

    print(f"\n-- C1  does thinning move the survivors OFF the skeleton?  "
          f"(median terminal->skeleton, cm)")
    print(f"  {'arm':10s} {'production':>18s} " +
          " ".join(f"{'R='+str(R):>18s}" for R in args.nms))
    for label, (_, c1, _) in per.items():
        if not c1:
            continue
        print(f"  {label:10s} {q([r['term'] for r in c1]):>18s} " +
              " ".join(f"{q([r['term'+str(R)] for r in c1]):>18s}" for R in args.nms))

    print(f"\n-- C2  the vertex region: core radius {args.core} cm, neighbourhood {args.rv} cm")
    print(f"  {'arm':10s} {'nvtx':>5s} {'branches':>16s} {'core terminals':>18s} "
          f"{'near terminals':>18s}")
    for label, (_, _, c2) in per.items():
        if not c2:
            print(f"  {label:10s} (none)")
            continue
        print(f"  {label:10s} {len(c2):5d} {q([r['branches_occupied'] for r in c2],'{:4.1f}'):>16s} "
              f"{q([r['ncore'] for r in c2],'{:5.1f}'):>18s} "
              f"{q([r['nnear'] for r in c2],'{:5.1f}'):>18s}")

    print(f"\n-- C2  BRANCHES EMPTIED inside {args.rv} cm by thinning at R "
          f"(total over all vertices; 'bound' is the order-free upper bound)")
    print(f"  {'arm':10s} {'R':>5s} {'bound':>8s} {'axis':>8s} {'shuffled':>9s} "
          f"{'adversarial':>12s} {'of branches':>12s}"
          + (f" {'exempt':>8s}" if args.exempt > 0 else ""))
    for label, (_, _, c2) in per.items():
        if not c2:
            continue
        tot = sum(r["branches_occupied"] for r in c2)
        for R in args.nms:
            row = (f"  {label:10s} {R:5.2f} {sum(r[f'bound{R}'] for r in c2):8d} "
                   f"{sum(r[f'lost{R}_ax'] for r in c2):8d} "
                   f"{sum(r[f'lost{R}_rnd'] for r in c2):9d} "
                   f"{sum(r[f'lost{R}_adv'] for r in c2):12d} {tot:12d}")
            if args.exempt > 0:
                row += f" {sum(r[f'lost{R}_exempt'] for r in c2):8d}"
            print(row)

    if args.exempt > 0:
        print(f"\n-- C3  what the {args.exempt} cm vertex exemption costs "
              f"(fraction of terminals kept, median over vertices' clusters)")
        print(f"  {'arm':10s} {'R':>5s} {'plain':>18s} {'exempt':>18s}")
        for label, (_, _, c2) in per.items():
            if not c2:
                continue
            for R in args.nms:
                print(f"  {label:10s} {R:5.2f} {q([r[f'keep{R}_plain'] for r in c2]):>18s} "
                      f"{q([r[f'keep{R}_exempt'] for r in c2]):>18s}")

    print(f"\n-- C2  vertex-CORE terminals (within {args.core} cm of the vertex fit point) "
          f"surviving, TOTAL over vertices")
    hdr = f"  {'arm':10s} {'nvtx':>6s} {'production':>12s}"
    for R in args.nms:
        hdr += f" {'R='+str(R)+' adv':>12s}"
        if args.exempt > 0:
            hdr += f" {'R='+str(R)+' exm':>12s}"
    print(hdr)
    for label, (_, _, c2) in per.items():
        if not c2:
            continue
        row = f"  {label:10s} {len(c2):6d} {sum(r['ncore'] for r in c2):12d}"
        for R in args.nms:
            row += f" {sum(r[f'core{R}_adv'] for r in c2):12d}"
            if args.exempt > 0:
                row += f" {sum(r[f'core{R}_exempt'] for r in c2):12d}"
        print(row)
    print(f"\n-- C2  vertices that lose EVERY vertex-core terminal (of those that had >=1),"
          f" under all three orders")
    print(f"  {'arm':10s} {'had core':>9s} " +
          " ".join(f"{'R='+str(R)+' '+t:>12s}" for R in args.nms for t in ("ax", "rnd", "adv")))
    for label, (_, _, c2) in per.items():
        if not c2:
            continue
        had = [r for r in c2 if r["ncore"] > 0]
        print(f"  {label:10s} {len(had):9d} " +
              " ".join(f"{sum(1 for r in had if r[f'core{R}_{t}']==0):12d}"
                       for R in args.nms for t in ("ax", "rnd", "adv")))


if __name__ == "__main__":
    main()
