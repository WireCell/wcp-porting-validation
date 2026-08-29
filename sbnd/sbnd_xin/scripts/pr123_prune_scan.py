#!/usr/bin/env python3
"""doc pr/123 -- FINAL-body detachment scan (the post-pass prune candidate).

The per-absorb census (pr123_pass4_census.py) measured that no at-absorb-time
threshold separates: legitimate EM growth also proceeds by far fragment
chains from a tiny pass-entry stem (TARGET snap_dis med 29.5 vs OUT 34.0).
The owner's "detached from the contiguous body" must therefore be read
against the FINAL shower: single-linkage components of the final membership
at gap G; a component not containing the start segment is detached.

For every labeled marked shower (dump membership via segment.shower_id --
lossy on overlapped showers, doc pr/91 caveat, acceptable for a scan):

  - build the segment graph with edges where min point-pair distance < G,
  - the KEEP component = the one holding the start segment (shower id),
  - every member outside it is PRUNED at that G.

Report, per G: OUT members pruned (want-high) vs TARGET members pruned
(collateral), plus per-event detail for the OUT-marked showers.

Repro:
  ./scripts/pr123_prune_scan.py 'work-pr123r1-dbgA2-*' 'work-pr123r1-dbg141v2-*'
"""
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
LABEL_DIRS = [os.path.join(SX, "em_labels", t)
              for t in ("emscan-0827", "emscan-0828-agent5")]
GS = [15.0, 20.0, 25.0, 30.0, 40.0]

try:
    import numpy as np
except ImportError:
    np = None


def load_labels(ev):
    for ld in LABEL_DIRS:
        p = os.path.join(ld, "labels-evt%d.json" % ev)
        if os.path.exists(p):
            em = json.load(open(p)).get("em") or {}
            marks = em.get("marks_by_shower") or {}
            out = {}
            for shw, mm in marks.items():
                ins = {int(s) for s, v in mm.items() if v == "in"}
                outs = {int(s) for s, v in mm.items() if v == "out"}
                out[int(shw)] = (ins, outs)
            return {"marks": out, "tag": os.path.basename(ld)}
    return None


def seg_pts(seg):
    return np.array([[p["x"], p["y"], p["z"]] for p in seg["points"]], dtype=float)


def min_dist(a, b):
    # min pairwise distance between two point sets
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return math.sqrt(d2.min())


def components(ids, dmat, gap):
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in ids:
        for j in ids:
            if i < j and dmat[(i, j)] < gap:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj
    comp = {}
    for i in ids:
        comp.setdefault(find(i), set()).add(i)
    return list(comp.values())


def main():
    roots = [r for g in sys.argv[1:] for r in sorted(glob.glob(g))]
    tot = {g: dict(out_pruned=0, out_tot=0, tgt_pruned=0, tgt_tot=0,
                   unl_pruned=0, unl_tot=0, showers_cut=0) for g in GS}
    detail = []
    for root in roots:
        for dj in sorted(glob.glob(os.path.join(root, "pr_evt*", "calib-pr-evt*.json"))):
            ev = int(os.path.basename(os.path.dirname(dj))[len("pr_evt"):])
            lab = load_labels(ev)
            if not lab:
                continue
            j = json.load(open(dj))
            segs = {s["id"]: s for s in j.get("segments", [])}
            for shw_key, (ins, outs) in lab["marks"].items():
                members = [sid for sid, s in segs.items() if s.get("shower_id") == shw_key]
                if shw_key not in segs or len(members) < 2:
                    continue
                pts = {sid: seg_pts(segs[sid]) for sid in members if segs[sid].get("points")}
                ids = [sid for sid in members if sid in pts and len(pts[sid])]
                if shw_key not in ids or len(ids) < 2:
                    continue
                dmat = {}
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        i, jd = ids[a], ids[b]
                        dmat[(min(i, jd), max(i, jd))] = min_dist(pts[i], pts[jd])
                for g in GS:
                    comps = components(sorted(ids), dmat, g)
                    keep = next(c for c in comps if shw_key in c)
                    pruned = set(ids) - keep
                    t = tot[g]
                    for sid in ids:
                        if sid == shw_key:
                            continue
                        is_out = sid in outs
                        is_tgt = (not is_out) and (sid not in outs)
                        klass = "out" if is_out else ("tgt" if sid in (set(members) - outs) else "unl")
                        # members not OUT-marked count as scanner-approved holds
                        if is_out:
                            t["out_tot"] += 1
                            if sid in pruned:
                                t["out_pruned"] += 1
                        else:
                            t["tgt_tot"] += 1
                            if sid in pruned:
                                t["tgt_pruned"] += 1
                    if pruned:
                        t["showers_cut"] += 1
                    if g == 25.0 and (pruned or outs):
                        detail.append((ev, shw_key, len(ids), sorted(pruned),
                                       sorted(o for o in outs if o in ids)))
    print("FINAL-body prune scan (single-linkage components, keep = start-seg component)")
    print("  G_cm  OUT pruned/tot   nonOUT pruned/tot (collateral)  showers touched")
    for g in GS:
        t = tot[g]
        print("  %4.0f  %5d/%-5d      %5d/%-5d                     %d"
              % (g, t["out_pruned"], t["out_tot"], t["tgt_pruned"], t["tgt_tot"],
                 t["showers_cut"]))
    print("\nG=25 detail (event, shower, nseg, pruned members, OUT marks present):")
    for ev, shw, n, pruned, outs_in in detail:
        print("  evt%-7d shw=%-7d n=%-3d pruned=%s outs=%s"
              % (ev, shw, n, pruned, outs_in))


if __name__ == "__main__":
    main()
