#!/usr/bin/env python3
"""Kink / zigzag census behind doc pr/86 sec 15 (round 2).

Round 1 shipped a graph-level repair whose census (pr86_merged_prong_census)
could not see the two shapes the owner actually pointed at:

  KINK    (evt38856)  the round-1 splice concatenates wcpt chains, so a
          30-40 deg corner at the dissolved junction now lives INSIDE a
          segment where vertex-degree metrics cannot see it.  Metric (i):
          max in-segment turn within KINK_R of the anchor, chord windows of
          WIN cm each side.
  ZIGZAG  (evt349945)  a long prong reaches the anchor only through a
          multi-segment polyline (14 cm of path over a 5.8 cm straight
          line).  The round-1 Class-B census required aim <= 25 deg
          (pr86_merged_prong_census.py AIM_ANGLE), which this shape fails
          by construction (349945 aims 49.9 deg away).  Metric (ii):
          for every long segment ending GAP_MIN..GAP_MAX from the anchor,
          the graph-path length vs the straight gap (ratio), plus the
          end-aim angle -- no aim cut.

Read-only: opens calib dumps and hand-scan labels, writes nothing unless
--json/--tsv is given.  Arms are selected exactly like the round-1 censuses:
set PR86_DUMP_ARMS=<arm>[:<arm>...] (unset falls back to the round-1
pr85ion2 ARMS list -- for round 2 you almost always want the env set).

    PR86_DUMP_ARMS=work-mcp1k-pr86ion:work-nuecc48-pr86ion:work-ncpi0-pr86ion \\
        python3 pr86_kink_census.py --top 15
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vtx_rules"))
import vtx_io                                        # noqa: E402

sys.path.insert(0, HERE)
import pr85_near_vertex_census as P85                # noqa: E402
import pr86_orphan_census as C86                     # noqa: E402

# ---------------------------------------------------------------- constants
KINK_R = 6.0     # cm, in-segment turns are scored this close to the anchor
WIN = 2.0        # cm, chord window on each side of a candidate turn point
KINK_MIN_LEN = 5.0   # cm, ignore micro-segments (their "turns" are noise)
GAP_MIN = 2.0    # cm, zigzag candidates end at least this far from the anchor
GAP_MAX = 10.0   # cm, ... and at most this far (the Class-B band)
LONG = 15.0      # cm, "a real prong" for the zigzag metric
AIM_WIN = 3.0    # cm, end-direction window for the aim angle
ANCHOR_TOL = 1.0  # cm, main_vertex -> nearest dump vertex (C86 convention)
CLICK_TOL = 1.0   # cm, click -> nearest dump vertex (C86 convention)


def pdist(a, b):
    return math.dist(a, b)


def seg_xyz(seg):
    return [P85.pxyz(p) for p in P85.seg_pts(seg)]


# ---------------------------------------------------------------- metric (i)
def max_kink(seg, anchor_xyz):
    """Max turn (deg, 0 = straight) at fit points within KINK_R of the anchor.

    Direction on each side of point i is the chord over the first WIN cm of
    arc walking away from i.  Chords, not adjacent steps: the fit point pitch
    is ~0.6-1.2 cm and adjacent-step angles are dominated by sampling noise.
    Returns (turn_deg, dist_to_anchor) of the worst point, or (None, None).
    """
    pts = seg_xyz(seg)
    if len(pts) < 3:
        return None, None
    best, best_d = None, None
    for i in range(1, len(pts) - 1):
        d_anchor = pdist(pts[i], anchor_xyz)
        if d_anchor > KINK_R:
            continue
        # backward chord
        arc = 0.0
        j = i
        while j > 0 and arc < WIN:
            arc += pdist(pts[j], pts[j - 1])
            j -= 1
        if arc < 0.5 * WIN:
            continue
        va = tuple(pts[i][k] - pts[j][k] for k in range(3))
        # forward chord
        arc = 0.0
        k2 = i
        while k2 < len(pts) - 1 and arc < WIN:
            arc += pdist(pts[k2], pts[k2 + 1])
            k2 += 1
        if arc < 0.5 * WIN:
            continue
        vb = tuple(pts[k2][k] - pts[i][k] for k in range(3))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        if na == 0 or nb == 0:
            continue
        cosv = sum(va[k] * vb[k] for k in range(3)) / (na * nb)
        turn = math.degrees(math.acos(max(-1.0, min(1.0, cosv))))
        if best is None or turn > best:
            best, best_d = turn, d_anchor
    if best is None:
        return None, None
    return round(best, 1), round(best_d, 2)


# --------------------------------------------------------------- metric (ii)
def end_aim(seg, end_idx, anchor_xyz):
    """Angle (deg) between the segment's outgoing end direction and the line
    end -> anchor.  0 = the extension points straight at the anchor."""
    pts = seg_xyz(seg)
    if len(pts) < 2:
        return None
    if end_idx != 0:
        pts = pts[::-1]
    # chord over the first AIM_WIN cm inward from the end
    arc, j = 0.0, 0
    while j < len(pts) - 1 and arc < AIM_WIN:
        arc += pdist(pts[j], pts[j + 1])
        j += 1
    v_end = tuple(pts[0][k] - pts[j][k] for k in range(3))    # inward->end
    v_to = tuple(anchor_xyz[k] - pts[0][k] for k in range(3))
    ne = math.sqrt(sum(x * x for x in v_end))
    nt = math.sqrt(sum(x * x for x in v_to))
    if ne == 0 or nt == 0:
        return None
    cosv = sum(v_end[k] * v_to[k] for k in range(3)) / (ne * nt)
    return round(math.degrees(math.acos(max(-1.0, min(1.0, cosv)))), 1)


def zigzags(dump, seg_of, anchor, anchor_xyz):
    """Long segments ending GAP_MIN..GAP_MAX from the anchor: gap, aim,
    graph-path length to the anchor, and path/gap ratio."""
    here = {s["id"] for s in seg_of.get(anchor["id"], [])}
    out = []
    for s in dump.get("segments", []):
        if s.get("length", 0.0) < LONG:
            continue
        if s["id"] in here:
            continue
        pts = seg_xyz(s)
        if len(pts) < 2:
            continue
        d0 = pdist(pts[0], anchor_xyz)
        d1 = pdist(pts[-1], anchor_xyz)
        end_idx = 0 if d0 <= d1 else len(pts) - 1
        gap = min(d0, d1)
        if not (GAP_MIN <= gap <= GAP_MAX):
            continue
        near_vid = s.get("start_vertex_id" if end_idx == 0 else "end_vertex_id")
        path = P85.graph_path(dump, seg_of, near_vid, anchor["id"])
        path_len = round(sum(t.get("length", 0.0) for t in path), 2) if path else None
        ratio = round(path_len / gap, 2) if path_len else None
        out.append({
            "seg": s["id"], "len": round(s.get("length", 0.0), 2),
            "gap": round(gap, 2), "aim": end_aim(s, end_idx, anchor_xyz),
            "path_hops": len(path) if path else None,
            "path_len": path_len, "ratio": ratio,
            "cluster": s.get("cluster_id"),
        })
    out.sort(key=lambda z: -z["len"])
    return out


# --------------------------------------------------------------------- rows
def score_anchor(dump, anchor, anchor_xyz):
    seg_of = vtx_io.segments_of_vertex(dump)
    kinks = []
    for s in dump.get("segments", []):
        if s.get("length", 0.0) < KINK_MIN_LEN:
            continue
        turn, d = max_kink(s, anchor_xyz)
        if turn is not None:
            kinks.append({"seg": s["id"], "turn": turn, "d": d,
                          "len": round(s.get("length", 0.0), 2)})
    kinks.sort(key=lambda k: -k["turn"])
    return {
        "vid": anchor["id"],
        "max_kink": kinks[0]["turn"] if kinks else None,
        "kinks": kinks[:5],
        "zigzags": zigzags(dump, seg_of, anchor, anchor_xyz),
    }


def scan_one(ev, root, arm, path, label):
    with open(path) as fh:
        dump = json.load(fh)
    row = {"ev": ev, "root": root, "arm": arm}
    mv = dump.get("main_vertex")
    if mv:
        a, gap = C86.anchor_near(dump, vtx_io.xyz(mv), ANCHOR_TOL)
        if a is not None:
            row["reco"] = score_anchor(dump, a, vtx_io.xyz(mv))
    if label is not None:
        a, gap = C86.anchor_near(dump, label["truth"], CLICK_TOL)
        if a is not None:
            row["click"] = score_anchor(dump, a, label["truth"])
    return row


def rows():
    dumps = C86.find_dumps()
    labs = C86.labels_by_event()
    out = []
    for ev in sorted(dumps):
        root, arm, path = dumps[ev]
        try:
            out.append(scan_one(ev, root, arm, path, labs.get(ev)))
        except Exception as exc:
            out.append({"ev": ev, "root": root, "arm": arm, "error": str(exc)})
    return out


# ------------------------------------------------------------------ reports
def census(rs):
    good = [r for r in rs if "error" not in r]
    print("dumps read %d, unreadable %d, reco-anchored %d"
          % (len(rs), len(rs) - len(good),
             sum(1 for r in good if "reco" in r)))
    hist = {}
    for r in good:
        mk = r.get("reco", {}).get("max_kink")
        b = ("none" if mk is None else
             "<20" if mk < 20 else "20-40" if mk < 40 else
             "40-60" if mk < 60 else ">=60")
        hist[b] = hist.get(b, 0) + 1
    print("max in-segment kink within %.0f cm of the reco anchor:" % KINK_R)
    for b in ("none", "<20", "20-40", "40-60", ">=60"):
        if b in hist:
            print("   %-6s : %4d events" % (b, hist[b]))
    nz = sum(1 for r in good for z in r.get("reco", {}).get("zigzags", []))
    bad = sum(1 for r in good for z in r.get("reco", {}).get("zigzags", [])
              if z["ratio"] is not None and z["ratio"] >= 1.5)
    nopath = sum(1 for r in good for z in r.get("reco", {}).get("zigzags", [])
                 if z["ratio"] is None)
    print("zigzag candidates (len>=%.0f, gap %.0f-%.0f cm): %d   "
          "ratio>=1.5: %d   no path: %d" % (LONG, GAP_MIN, GAP_MAX,
                                            nz, bad, nopath))
    print()


def top(rs, n):
    good = [r for r in rs if "error" not in r and "reco" in r]
    def key(r):
        mk = r["reco"].get("max_kink") or 0.0
        zr = max((z["ratio"] or 0.0 for z in r["reco"]["zigzags"]), default=0.0)
        return (-(mk >= 30) - (zr >= 1.5), -mk, -zr)
    good.sort(key=lambda r: key(r) + (r["ev"],))
    print("TOP %d by (kink >= 30 deg | zigzag ratio >= 1.5) at the reco anchor" % n)
    for r in good[:n]:
        a = r["reco"]
        zz = ", ".join("%s:len%.0f gap%.1f aim%s path%s r%s"
                       % (z["seg"], z["len"], z["gap"], z["aim"],
                          z["path_len"], z["ratio"]) for z in a["zigzags"])
        kk = ", ".join("%s:%.0fdeg@%.1fcm" % (k["seg"], k["turn"], k["d"])
                       for k in a["kinks"][:3])
        print("  evt%-7d max_kink %-5s  kinks[%s]  zigzag[%s]"
              % (r["ev"], a["max_kink"], kk, zz))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write all rows to this path")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--events", nargs="*", type=int,
                    help="print full rows for these events")
    args = ap.parse_args()

    rs = rows()
    census(rs)
    top(rs, args.top)
    if args.events:
        by = {r["ev"]: r for r in rs}
        for ev in args.events:
            print(json.dumps(by.get(ev, {"ev": ev, "missing": True}), indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rs, fh, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
