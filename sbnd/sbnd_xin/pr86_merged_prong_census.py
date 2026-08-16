#!/usr/bin/env python3
"""Census behind doc pr/86 sec 10 -- the prong whose LAST STRETCH went missing.

    cd sbnd_xin && python3 pr86_merged_prong_census.py [--top N] [--json OUT]

pr86_orphan_census.py counts ORPHANS: segments whose charge reaches the vertex
but which are not incident on it.  The owner named a second defect that is not
the same thing:

    "a track connecting to the vertex did not go through the last track
     segment, but merge to another track"

i.e. the prong's own final stretch -- the piece that should run from the prong
down to the neutrino vertex -- is not the prong's any more.  Two shapes, and
the owner asked for both:

CLASS A -- the collinear hand-off.  The last stretch DOES exist as an object,
    incident on the vertex, but it was made a segment in its own right and the
    prong hangs off its far end:

        V0 (vertex) --- S --- V1 --- P
                                 turn(S,P) at V1 ~ 180 deg

    S is geometrically P's own continuation, so the break at V1 is spurious.
    Measured by the turn angle at V1: 180 deg means P continues straight
    through S, 90 deg means P is a genuine daughter of a different track S.
    Every Class-A prong IS an orphan, so this class is a REFINEMENT of the
    orphan census, not a new population.

CLASS B -- the absorbed last stretch.  The prong's final charge was merged into
    a DIFFERENT track's segment, so near the vertex the prong has no object of
    its own at all.  It therefore stops SHORT of the vertex -- outside the 3 cm
    TOUCH radius -- and pr86_orphan_census.py cannot see it: there is no
    non-incident segment at the vertex to count, and these events score clean.

        V0 (vertex)
         |\\
         | \\  Q -- one segment, carrying its own charge AND P's last stretch
         |  \\
        V1   \\
         P starts only here, short of V0

    Detected geometrically, with no truth and no charge model: P ends near the
    vertex but not at it, P AIMS at the vertex, and the gap between them is
    already covered by another segment Q that does reach the vertex.  Q is the
    track P was merged into.

Both classes are scored at BOTH anchors (reco main_vertex and the hand click),
the same convention pr86_orphan_census.py uses and for the same reason.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vtx_rules"))
import vtx_io                                        # noqa: E402
import pr86_orphan_census as C86                     # noqa: E402
P85 = C86.P85

# ------------------------------------------------------------ class A
STRAIGHT = 150.0   # deg of turn at V1 above which P is P's own continuation.
                   # Same number as mvga_interposed_angle (NeutrinoPatternBase.h
                   # m_mvga_reseat_angle/interposed default), on purpose: op3's
                   # interposed branch uses exactly this test, so a Class-A case
                   # above it is one op3 would accept on collinearity.
REAL_SEG = 2.5     # cm, mvga_stub in SBND production -- above this the
                   # interposed object is a track, not clutter op3 can absorb.

# ------------------------------------------------------------ class B
MIN_PRONG = 5.0    # cm, "this is a real prong, not a fragment"
GAP_MIN = C86.TOUCH   # 3.0 cm -- BELOW this the prong is an orphan and class A
                      # or the orphan census already owns it.  Class B starts
                      # exactly where the orphan census stops seeing.
GAP_MAX = 10.0     # cm, beyond this a missing stretch is not "the last bit"
END_ARC = 2.0      # cm of arclength: the prong must END here, not pass by
                   # (pr/85 STRADDLE_ARC -- a hairpin must not read as an end)
AIM_ANGLE = 25.0   # deg between (V0 - end) and the prong's own extension.
                   # A prong that stops short but points elsewhere lost nothing.
COVER = 1.5        # cm, how close another segment must run to the gap for the
                   # gap to count as "already drawn by somebody else"
COVER_FRAC = 0.7   # fraction of gap samples that must be covered
STEP = 0.5         # cm, gap sampling pitch


# ------------------------------------------------------------ class C
# The unifying measurement, and the one that reproduces the owner's own hand
# judgement.  Classes A and B each describe one way the graph can be wrong; this
# asks the question underneath both -- in how many distinct directions does the
# CHARGE leave the vertex, and does the graph give it that many arms?
#
#   owner, evt38856: "one of the track has a very large angle turn, which is not
#   right.  It should be a 3-track vertex."
#   measured:        graph degree 2, charge arms 3.
#
# It catches shapes A and B cannot: two prongs drawn as ONE segment passing
# through the vertex shows up as 2 arms on 1 edge.  It is NOT a superset --
# evt268067 has degree 4 and 4 arms yet is a clean Class B (its two prongs are
# handed off through a 9 cm segment along one of those arms) -- so the three
# measurements are reported side by side, not merged.
ARM_R = 6.0      # cm, radius around the vertex in which arms are counted
ARM_SEP = 25.0   # deg, two charge directions closer than this are one arm
ARM_MINPTS = 3   # fitted points before a direction counts as an arm


def arm_census(dump, c):
    """Distinct directions the charge leaves the vertex in.

    Greedy angular clustering of the unit vectors from the vertex to every
    fitted point in the shell [1 cm, ARM_R].  The inner 1 cm is excluded
    because directions there are numerically unstable, not because nothing is
    there.  Greedy order follows the dump's segment order, which is stable.
    """
    clusters = []
    for s in dump.get("segments", []):
        for p in P85.seg_pts(s):
            q = P85.pxyz(p)
            d = math.dist(q, c)
            if not (1.0 <= d <= ARM_R):
                continue
            v = [(q[k] - c[k]) / d for k in range(3)]
            for cl in clusters:
                if P85.angle_deg(v, cl["dir"]) <= ARM_SEP:
                    cl["n"] += 1
                    cl["segs"].add(s["id"])
                    n = cl["n"]
                    cl["dir"] = [(cl["dir"][k] * (n - 1) + v[k]) / n
                                 for k in range(3)]
                    mag = math.sqrt(sum(t * t for t in cl["dir"])) or 1.0
                    cl["dir"] = [t / mag for t in cl["dir"]]
                    break
            else:
                clusters.append({"dir": list(v), "n": 1, "segs": {s["id"]}})
    keep = [cl for cl in clusters if cl["n"] >= ARM_MINPTS]
    return [{"n": cl["n"], "segs": sorted(cl["segs"])} for cl in keep]


def seg_ends(s):
    pts = P85.seg_pts(s)
    return (P85.pxyz(pts[0]), P85.pxyz(pts[-1])) if pts else (None, None)


def turn_at(S, P, v1):
    """Turn angle at the shared vertex v1 between segments S and P.

    180 deg = S and P leave v1 in opposite directions, i.e. they form one
    straight track through v1.
    """
    ds = P85.unit_dir(S, "start" if S.get("start_vertex_id") == v1 else "end")
    dp = P85.unit_dir(P, "start" if P.get("start_vertex_id") == v1 else "end")
    return P85.angle_deg(ds, dp)


def class_a(dump, row_anchor, segs, seg_of):
    """Refine the anchor's orphans into collinear hand-offs."""
    out = []
    for o in row_anchor["orphans"]:
        if not o["path"] or len(o["path"]) != 1:
            continue
        S, P = segs.get(o["path"][0][0]), segs.get(o["seg"])
        if S is None or P is None:
            continue
        common = ({S.get("start_vertex_id"), S.get("end_vertex_id")}
                  & {P.get("start_vertex_id"), P.get("end_vertex_id")})
        if len(common) != 1:
            continue
        v1 = common.pop()
        ang = turn_at(S, P, v1)
        if ang is None:
            continue
        # Who else is at V1?  "merged to ANOTHER track" needs another track.
        others = [t["id"] for t in seg_of.get(v1, [])
                  if t["id"] not in (S["id"], P["id"])]
        out.append({
            "prong": P["id"], "prong_len": round(P.get("length", 0.0), 2),
            "handoff": S["id"], "handoff_len": o["path"][0][1],
            "v1": v1, "turn": round(ang, 1),
            "others_at_v1": others,
            "straight": ang >= STRAIGHT,
            "real": o["path"][0][1] >= REAL_SEG,
        })
    return out


def class_b(dump, anchor_xyz, anchor_id, segs, seg_of):
    """Prongs that stop short of the vertex, aim at it, and whose gap is
    already drawn by another segment that does reach the vertex."""
    incident = {s["id"] for s in seg_of.get(anchor_id, [])}
    # Segments that reach the anchor at all -- only these can be an absorber,
    # otherwise the "cover" is some unrelated track crossing the gap.
    reaching = []
    for s in dump.get("segments", []):
        pts = P85.seg_pts(s)
        if not pts:
            continue
        d, _ = P85.nearest_point(s, anchor_xyz)
        if d is not None and d <= C86.TOUCH:
            reaching.append(s)

    found = []
    for P in dump.get("segments", []):
        pts = P85.seg_pts(P)
        if len(pts) < 2 or P.get("length", 0.0) < MIN_PRONG:
            continue
        if P["id"] in incident:
            continue
        dmin, imin = P85.nearest_point(P, anchor_xyz)
        if dmin is None or not (GAP_MIN < dmin <= GAP_MAX):
            continue
        if P85.arc_to_end(P, imin) > END_ARC:
            continue                      # passes by, does not end here
        end = P85.pxyz(pts[0] if imin * 2 < len(pts) else pts[-1])
        which = "start" if imin * 2 < len(pts) else "end"
        d_in = P85.unit_dir(P, which)     # points INTO the segment
        if d_in is None:
            continue
        toward = tuple(anchor_xyz[k] - end[k] for k in range(3))
        n = math.sqrt(sum(t * t for t in toward))
        if n <= 0:
            continue
        toward = tuple(t / n for t in toward)
        # P's own extension is -d_in; require it to point at the anchor.
        aim = P85.angle_deg(tuple(-t for t in d_in), toward)
        if aim is None or aim > AIM_ANGLE:
            continue
        # Is the gap already drawn by somebody else that reaches the anchor?
        nsamp = max(2, int(n / STEP))
        best_id, best_hits = None, 0
        for Q in reaching:
            if Q["id"] == P["id"]:
                continue
            qp = [P85.pxyz(q) for q in P85.seg_pts(Q)]
            if not qp:
                continue
            hits = 0
            for i in range(1, nsamp + 1):
                t = i / (nsamp + 1.0)
                x = tuple(end[k] + toward[k] * n * t for k in range(3))
                if min(math.dist(x, q) for q in qp) <= COVER:
                    hits += 1
            if hits > best_hits:
                best_id, best_hits = Q["id"], hits
        if best_id is None or best_hits < COVER_FRAC * nsamp:
            continue
        Q = segs[best_id]
        found.append({
            "prong": P["id"], "prong_len": round(P.get("length", 0.0), 2),
            "gap": round(dmin, 2), "aim_deg": round(aim, 1),
            "absorber": best_id,
            "absorber_len": round(Q.get("length", 0.0), 2),
            "absorber_incident": best_id in incident,
            "cover_frac": round(best_hits / nsamp, 2),
            "same_cluster": P.get("cluster_id") == Q.get("cluster_id"),
        })
    found.sort(key=lambda f: -f["prong_len"])
    return found


def scan_one(ev, root, arm, path, label):
    with open(path) as fh:
        dump = json.load(fh)
    segs = {s["id"]: s for s in dump.get("segments", [])}
    seg_of = vtx_io.segments_of_vertex(dump)
    base = C86.scan_one(ev, root, arm, path, label)
    out = {"ev": ev, "root": root, "b1": base.get("b1"),
           "owned": C86.OWNED.get(ev)}
    for key, target in (("reco", dump.get("main_vertex")),
                        ("click", label["truth"] if label else None)):
        a = base.get(key)
        if not a or target is None:
            continue
        c = vtx_io.vertex_xyz({"id": a["vid"], "fit": None}) if False else None
        # the anchor's own coordinates, from the dump vertex the base row used
        vtx = next(v for v in dump["vertices"] if v["id"] == a["vid"])
        c = vtx_io.vertex_xyz(vtx)
        arms = arm_census(dump, c)
        out[key] = {
            "vid": a["vid"], "deg": a["deg"], "touch": a["touch"],
            "n_orphan": a["n_orphan"],
            "A": class_a(dump, a, segs, seg_of),
            "B": class_b(dump, c, a["vid"], segs, seg_of),
            "arms": len(arms), "arm_detail": arms,
        }
    return out


# The owner named two events that show Class B.  The census must reproduce both
# before any other number is read off it -- 268067 in particular, because the
# ORPHAN census scores it clean (degree 4, zero orphans) and doc pr/86 sec 1.1
# wrongly concluded from that it had been repaired by the pr/85 flip.
GATE = {
    281214: {"prong": 60029, "absorber": 60043},   # owner: "281241 plot clearly shows this"
    268067: {"prong": 15009, "absorber": 15082},   # owner: "same thing happened in 268067"
}
# owner, evt38856: "It should be a 3-track vertex."  The arm census must say so
# without being told: degree 2, arms 3.
ARM_GATE = {38856: {"vid": 12106, "deg": 2, "arms": 3}}


def check_gate(rs):
    by = {r["ev"]: r for r in rs}
    ok = True
    print("gate -- the two events the owner named must come out as Class B")
    for ev, want in sorted(GATE.items()):
        r = by.get(ev)
        hits = [b for k in ("reco", "click") if r and k in r
                for b in r[k]["B"]
                if b["prong"] == want["prong"] and b["absorber"] == want["absorber"]]
        print("  evt%-7d prong %-7s absorbed into %-7s : %s"
              % (ev, want["prong"], want["absorber"],
                 "found (gap %.2f cm, cover %.2f)" % (hits[0]["gap"], hits[0]["cover_frac"])
                 if hits else "NOT FOUND"))
        ok = ok and bool(hits)
    for ev, want in sorted(ARM_GATE.items()):
        r = by.get(ev)
        got = r["reco"] if r and "reco" in r else None
        hit = bool(got and got["vid"] == want["vid"]
                   and got["deg"] == want["deg"] and got["arms"] == want["arms"])
        print("  evt%-7d vertex %-6s degree %d, charge arms %d : %s"
              % (ev, want["vid"], want["deg"], want["arms"],
                 "reproduced" if hit
                 else "NOT reproduced (got deg %s arms %s)"
                      % (got and got["deg"], got and got["arms"])))
        ok = ok and hit
    print("  gate:", "PASS" if ok else "FAIL -- do not use these numbers")
    print()
    return ok


def rows():
    dumps = C86.find_dumps()
    labs = C86.labels_by_event()
    out = []
    for ev in sorted(dumps):
        root, arm, path = dumps[ev]
        try:
            out.append(scan_one(ev, root, arm, path, labs.get(ev)))
        except Exception as exc:
            out.append({"ev": ev, "root": root, "error": str(exc)})
    return out


def report(rs, topn):
    good = [r for r in rs if "error" not in r]
    print("dumps scored : %d   (errors %d)"
          % (len(good), len(rs) - len(good)))
    print()

    # ---- class A
    allA = [(r, k, a) for r in good for k in ("reco", "click")
            if k in r for a in r[k]["A"]]
    straight = [x for x in allA if x[2]["straight"]]
    real = [x for x in straight if x[2]["real"]]
    other = [x for x in straight if x[2]["others_at_v1"]]
    print("CLASS A -- collinear hand-off (the prong hangs off its own last stretch)")
    print("  one-hop orphans measured            : %d" % len(allA))
    print("  turn >= %.0f deg (P continues through S): %d" % (STRAIGHT, len(straight)))
    print("    ... through a REAL segment >= %.1f cm : %d" % (REAL_SEG, len(real)))
    print("    ... with another track also at V1   : %d" % len(other))
    print("  turn histogram (deg):")
    h = {}
    for x in allA:
        h[min(int(x[2]["turn"] // 30) * 30, 150)] = \
            h.get(min(int(x[2]["turn"] // 30) * 30, 150), 0) + 1
    for k in sorted(h):
        print("     [%3d,%3d) : %3d %s" % (k, k + 30, h[k], "#" * h[k]))
    print()
    print("  top Class-A cases (real hand-off segment), by prong length:")
    for r, k, a in sorted(real, key=lambda x: -x[2]["prong_len"])[:topn]:
        print("    evt%-7d %-5s prong %-7s %6.2f cm  <- hand-off %-7s %5.2f cm"
              "  turn %5.1f  others@V1 %s%s"
              % (r["ev"], k, a["prong"], a["prong_len"], a["handoff"],
                 a["handoff_len"], a["turn"], a["others_at_v1"] or "none",
                 "   [%s]" % r["owned"] if r["owned"] else ""))
    print()

    # ---- class B
    allB = [(r, k, b) for r in good for k in ("reco", "click")
            if k in r for b in r[k]["B"]]
    evsB = {r["ev"] for r, k, b in allB}
    invisible = {r["ev"] for r, k, b in allB
                 if r[k]["n_orphan"] == 0}
    print("CLASS B -- absorbed last stretch (prong stops short, gap drawn by another track)")
    print("  cases                                : %d" % len(allB))
    print("  events                               : %d" % len(evsB))
    print("  ... events the ORPHAN census scores CLEAN at that anchor: %d"
          % len(invisible))
    if allB:
        gaps = sorted(b["gap"] for _, _, b in allB)
        print("  gap (prong end -> vertex): median %.2f cm, range %.2f-%.2f"
              % (gaps[len(gaps) // 2], gaps[0], gaps[-1]))
        aab = sorted(b["absorber_len"] for _, _, b in allB)
        print("  absorber length          : median %.2f cm, range %.2f-%.2f"
              % (aab[len(aab) // 2], aab[0], aab[-1]))
        print("  absorber incident on the vertex   : %d of %d"
              % (sum(1 for _, _, b in allB if b["absorber_incident"]), len(allB)))
        print("  absorber in a DIFFERENT cluster   : %d of %d"
              % (sum(1 for _, _, b in allB if not b["same_cluster"]), len(allB)))
    print()
    print("  top Class-B cases, by prong length:")
    for r, k, b in sorted(allB, key=lambda x: -x[2]["prong_len"])[:topn]:
        print("    evt%-7d %-5s prong %-7s %6.2f cm  gap %4.2f cm  aim %4.1f deg"
              "  absorbed into %-7s %6.2f cm  cover %.2f%s%s"
              % (r["ev"], k, b["prong"], b["prong_len"], b["gap"], b["aim_deg"],
                 b["absorber"], b["absorber_len"], b["cover_frac"],
                 "" if b["absorber_incident"] else "  (absorber not incident)",
                 "   [%s]" % r["owned"] if r["owned"] else ""))
    print()


def report_arms(rs, topn):
    good = [r for r in rs if "error" not in r]
    rowsC = [(r, k, r[k]) for r in good for k in ("reco", "click") if k in r]
    print("CLASS C -- charge arms vs graph degree at the vertex")
    print("  anchors measured : %d" % len(rowsC))
    over = [x for x in rowsC if x[2]["arms"] > x[2]["deg"]]
    same = [x for x in rowsC if x[2]["arms"] == x[2]["deg"]]
    under = [x for x in rowsC if x[2]["arms"] < x[2]["deg"]]
    print("  arms == degree   : %d  (%.0f%%)  -- the graph agrees with the charge"
          % (len(same), 100.0 * len(same) / max(1, len(rowsC))))
    print("  arms >  degree   : %d  (%.0f%%)  -- the graph is MISSING arms"
          % (len(over), 100.0 * len(over) / max(1, len(rowsC))))
    print("  arms <  degree   : %d  (%.0f%%)"
          % (len(under), 100.0 * len(under) / max(1, len(rowsC))))
    h = {}
    for _, _, x in rowsC:
        h[x["arms"] - x["deg"]] = h.get(x["arms"] - x["deg"], 0) + 1
    print("  (arms - degree) histogram: %s"
          % {k: h[k] for k in sorted(h)})
    print()
    print("  worst (arms - degree), excluding events owned by another round:")
    for r, k, x in sorted([y for y in over if not y[0]["owned"]],
                          key=lambda y: -(y[2]["arms"] - y[2]["deg"]))[:topn]:
        print("    evt%-7d %-5s vertex %-6s degree %d  charge arms %d  (+%d)"
              % (r["ev"], k, x["vid"], x["deg"], x["arms"],
                 x["arms"] - x["deg"]))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json")
    args = ap.parse_args()
    rs = rows()
    check_gate(rs)
    report(rs, args.top)
    report_arms(rs, args.top)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rs, fh, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
