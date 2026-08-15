#!/usr/bin/env python3
"""Census behind doc pr/85 -- near-vertex PR quality.

Every number in docs/pr/85_near-vertex-pr-quality.md comes from one run of this
script.  Read-only: it opens the deployed-arm calib dumps and the hand-scan
labels and writes nothing.

    cd sbnd_xin && python3 pr85_near_vertex_census.py

The owner named two failure modes while hand-scanning neutrino vertices:

  1. "one track aiming at the vertex got merged to a nearby track first, and
     then missing the last part of connecting to neutrino vertex"
  2. "multiple tracks go zig and zag, and multiple small segments are showing
     up" -- where the right answer is one vertex with a few clean prongs.

THE ONE DECISION THAT MAKES THESE NUMBERS MEAN ANYTHING: every metric is scored
at the owner's CLICK (`vtx_io.load_labels()['truth']`, the rank-1 pick), not at
the reconstructed `main_vertex`.  Scoring at the reco vertex would mix "the
topology near the vertex is ugly" with "the wrong vertex was picked", and the
second is doc pr/80's subject, already measured.  `b1` (click -> reco vertex) is
carried as a column so events where the two disagree can be segregated rather
than silently counted as topology failures.

Mode 1 is reported as THREE populations, because they are different defects
with different fixes and the obvious framing conflates them.  First split by
where on the prong the click falls:

  1b STRADDLE the prong's INTERIOR passes the click -- two prongs were
              reconstructed as one object and never broken at the vertex.
  1a          the prong's own END sits at the click.

The split is by arclength from the nearest fitted point to the nearer segment
end (STRADDLE_ARC below), not by 3-D distance, so a hairpin does not read as an
end.  Then 1a splits again, on the graph question that actually decides what a
fix would have to do -- walk the PR graph from the prong's near-end vertex to
the clicked vertex:

  1a-VIA      a path exists, but every segment on it is a stub.  The prong IS
              connected; it reaches the vertex through clutter.  This is the
              owner's "merged to a nearby track first" seen in the graph, and
              it is the same object as doc pr/84 sec 5's hierarchy inversion.
  1a-CUT      no path.  The prong is genuinely disconnected from the vertex --
              often because it is in a different cluster entirely.

Reporting 1a as one number would have hidden that the majority are CONNECTED,
which changes the fix from "add an edge" to "collapse the interposed stub".

Isochronous events are counted but held out of the candidate list: the owner
said "things can be complicated when isochronous", and on a segment lying in
the drift-perpendicular plane the simple topology is not obviously achievable.
The measure is the segment's angle out of that plane, the doc pr/73 sec 4.5
quantity, with the same 10-degree bin edge.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "vtx_rules"))
import baselines                                   # noqa: E402
import vtx_io                                      # noqa: E402

TOUCH = 3.0        # cm, "this segment's charge is at the vertex"
STUB = 3.0         # cm, a segment this short is clutter, not a prong
LONG = 10.0        # cm, a prong this long is a real object and should be attached
CLICK_TOL = 1.0    # cm, the pr/78/79/80 tolerance -- do not redefine it here
ISO = 10.0         # deg out of the drift-perpendicular plane (doc pr/73 sec 4.5)
STRADDLE_ARC = 2.0  # cm of arclength from an end; beyond this the pass is interior
NEAR_VTX = 3.0     # cm, radius for counting rival vertices around the click

# Events that belong to another round.  Named, not silently dropped: each is a
# real instance of one of these modes, and the doc says so.
OWNED = {
    283713: "pr/84", 287517: "pr/84", 284794: "pr/84", 65289: "pr/84",
    283040: "pr/83", 59899: "pr/83", 72586: "pr/83",
}

# doc pr/84 sec 5's micro-parent inversions, for the cross-flag.  A sub-2 cm PF
# node parenting a >10 cm child more than 5x longer than itself.
PR84_MICRO = {
    283040, 394796, 314838, 353223, 277276, 168596, 64717, 174114, 283091,
    400474, 286353, 352123, 284145, 287517, 282204, 59335, 57883, 423981,
    54351, 57575, 285567,
}


# ------------------------------------------------------------------ geometry
def pxyz(p):
    return (p["x"], p["y"], p["z"])


def seg_pts(s):
    return s.get("points") or []


def iso_deg(s):
    """Angle of the segment's chord out of the drift-perpendicular plane.

    0 deg = fully isochronous (the chord lies in a plane of constant drift
    coordinate).  Chord, not path: a bow inside an isochronous stretch must not
    read as non-isochronous.
    """
    pts = seg_pts(s)
    if len(pts) < 2:
        return None
    a, b = pts[0], pts[-1]
    chord = math.dist(pxyz(a), pxyz(b))
    if chord <= 0:
        return None
    return math.degrees(math.asin(min(1.0, abs(b["x"] - a["x"]) / chord)))


def nearest_point(s, c):
    """(distance, index) of the fitted point closest to c."""
    best, bi = None, None
    for i, q in enumerate(seg_pts(s)):
        d = math.dist(pxyz(q), c)
        if best is None or d < best:
            best, bi = d, i
    return best, bi


def arc_to_end(s, i):
    """Arclength from fitted point i to the NEARER end of the segment."""
    pts = seg_pts(s)
    fwd = sum(math.dist(pxyz(pts[k]), pxyz(pts[k + 1])) for k in range(i))
    bwd = sum(math.dist(pxyz(pts[k]), pxyz(pts[k + 1]))
              for k in range(i, len(pts) - 1))
    return min(fwd, bwd)


def unit_dir(s, at_end, span=10.0):
    """Unit vector pointing AWAY from the given end, over up to `span` cm.

    Same convention as vtx_geom.seg_direction; reimplemented on the raw point
    list so the two mode tables use one code path.
    """
    pts = seg_pts(s)
    if len(pts) < 2:
        return None
    if at_end == "start":
        a, rest = pts[0], pts[1:]
    else:
        a, rest = pts[-1], list(reversed(pts[:-1]))
    b = rest[-1]
    for q in rest:
        if math.dist(pxyz(q), pxyz(a)) >= span:
            b = q
            break
    v = tuple(b[k] - a[k] for k in ("x", "y", "z"))
    n = math.sqrt(sum(t * t for t in v))
    return tuple(t / n for t in v) if n > 0 else None


def angle_deg(u, v):
    if u is None or v is None:
        return None
    d = max(-1.0, min(1.0, sum(a * b for a, b in zip(u, v))))
    return math.degrees(math.acos(d))


def drift_angle(s):
    """Angle of the segment chord to the drift axis, in degrees.

    examine_vertices_4's second clause tests |angle - 90| < 10 on exactly this
    quantity (NeutrinoStructureExaminer.cxx:2074-2085).
    """
    pts = seg_pts(s)
    if len(pts) < 2:
        return None
    v = tuple(pts[0][k] - pts[-1][k] for k in ("x", "y", "z"))
    n = math.sqrt(sum(t * t for t in v))
    if n <= 0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, v[0] / n))))


def direct_length(s):
    pts = seg_pts(s)
    if len(pts) < 2:
        return 0.0
    return math.dist(pxyz(pts[0]), pxyz(pts[-1]))


def graph_path(dump, seg_of, src_vid, dst_vid, max_hops=6):
    """Shortest vertex path src->dst through the PR graph, or None.

    Vertices are keyed by id, which is stable and sortable, so this never
    iterates a pointer-keyed container (CLAUDE.md sec 2, determinism).
    """
    if src_vid is None or dst_vid is None:
        return None
    if src_vid == dst_vid:
        return []
    seen = {src_vid}
    front = [(src_vid, [])]
    for _ in range(max_hops):
        nxt = []
        for vid, path in front:
            for s in sorted(seg_of.get(vid, []), key=lambda t: t["id"]):
                other = (s["end_vertex_id"] if s.get("start_vertex_id") == vid
                         else s.get("start_vertex_id"))
                if other is None or other in seen:
                    continue
                if other == dst_vid:
                    return path + [s]
                seen.add(other)
                nxt.append((other, path + [s]))
        front = nxt
        if not front:
            break
    return None


# --------------------------------------------------------------------- rows
def scan_one(label):
    """Every near-vertex measurement for one labelled event, or None."""
    dump_path = baselines.deployed_dump_path(label)
    with open(dump_path) as fh:
        dump = json.load(fh)
    click = label["truth"]

    vtx = None
    vd = None
    for v in dump.get("vertices", []):
        q = vtx_io.vertex_xyz(v)
        if q is None:
            continue
        d = math.dist(q, click)
        if vd is None or d < vd:
            vtx, vd = v, d
    if vtx is None or vd > CLICK_TOL:
        return {"ev": label["eventNo"], "tag": label["tag"], "unmatched": True,
                "click_gap": vd}

    seg_of = vtx_io.segments_of_vertex(dump)
    incident = {s["id"] for s in seg_of.get(vtx["id"], [])}

    # `benign` collects prongs that reach the click through a real (non-stub)
    # segment.  That is an ordinary grand-daughter topology, not a defect, and
    # counting it as one would inflate mode 1 by the commonest shape there is.
    stubs, via, cut, straddle, benign, isos = [], [], [], [], [], []
    for s in dump.get("segments", []):
        if not seg_pts(s):
            continue
        dmin, imin = nearest_point(s, click)
        if dmin > TOUCH:
            continue
        io = iso_deg(s)
        if io is not None:
            isos.append(io)
        length = s.get("length", 0.0)
        if length < STUB:
            stubs.append((s["id"], round(length, 2), round(dmin, 2),
                          s["id"] in incident))
        if length < LONG or s["id"] in incident:
            continue
        arc = arc_to_end(s, imin)
        pts = seg_pts(s)
        end = ("start" if math.dist(pxyz(pts[0]), click)
               <= math.dist(pxyz(pts[-1]), click) else "end")
        near_vid = s.get("start_vertex_id" if end == "start" else "end_vertex_id")
        rec = {
            "seg": s["id"], "len": round(length, 1), "charge_to_click": round(dmin, 2),
            "arc_from_end": round(arc, 2), "near_end_vertex": near_vid,
            "same_cluster": s.get("cluster_id") == vtx["cluster_id"],
            "cluster": s.get("cluster_id"),
        }
        if arc > STRADDLE_ARC:
            straddle.append(rec)
            continue
        gpath = graph_path(dump, seg_of, near_vid, vtx["id"])
        if gpath is None:
            rec["path"] = None
            cut.append(rec)
        else:
            rec["path"] = [(t["id"], round(t.get("length", 0.0), 2))
                           for t in gpath]
            rec["path_len"] = round(sum(t.get("length", 0.0) for t in gpath), 2)
            rec["all_stubs"] = all(t.get("length", 0.0) < STUB for t in gpath)
            (via if rec["all_stubs"] else benign).append(rec)

    rivals = sum(1 for v in dump.get("vertices", [])
                 if vtx_io.vertex_xyz(v)
                 and math.dist(vtx_io.vertex_xyz(v), click) <= NEAR_VTX)

    return {
        "ev": label["eventNo"], "tag": label["tag"], "unmatched": False,
        "path": dump_path, "vid": vtx["id"], "cid": vtx["cluster_id"],
        "click_gap": round(vd, 3), "deg": len(incident),
        "deg_field": vtx.get("degree"), "rivals": rivals,
        "stubs": stubs, "via": via, "cut": cut, "straddle": straddle,
        "benign": benign,
        "iso": (min(isos) if isos else None),
        "b1": label["b1"], "owned": OWNED.get(label["eventNo"]),
        "micro84": label["eventNo"] in PR84_MICRO,
    }


def rows():
    labs = [L for L in vtx_io.load_labels() if baselines.deployed_dump_path(L)]
    return labs, [scan_one(L) for L in labs]


# ------------------------------------------------------------------ reports
def fmt_iso(r):
    return "%5.1f" % r["iso"] if r["iso"] is not None else "    -"


def census(good):
    m1 = [r for r in good if r["via"] or r["cut"] or r["straddle"]]
    m2 = [r for r in good if len(r["stubs"]) >= 2]
    n = len(good)
    print("=== the two modes, scored at the owner's click")
    print("scorable events (a dump vertex within %.1f cm of the click): %d"
          % (CLICK_TOL, n))
    print("  mode 1  a >=%.0f cm prong with charge within %.0f cm of the click"
          % (LONG, TOUCH))
    print("          that carries NO edge to the clicked vertex ....... %3d  (%.1f%%)"
          % (len(m1), 100.0 * len(m1) / n))
    print("            1a-VIA   connected, but only through stubs .... %3d"
          % len([r for r in good if r["via"]]))
    print("            1a-CUT   no path to the vertex at all ......... %3d"
          % len([r for r in good if r["cut"]]))
    print("            1b STRADDLE its interior passes the vertex ..... %3d"
          % len([r for r in good if r["straddle"]]))
    print("          (excluded: prong reaching the click through a real")
    print("           segment, an ordinary grand-daughter) ........... %3d"
          % len([r for r in good if r["benign"]]))
    print("          of the 1a-CUT prongs, in a DIFFERENT cluster:  %d of %d"
          % (len([r for r in good if any(not d["same_cluster"] for d in r["cut"])]),
             len([r for r in good if r["cut"]])))
    print("  mode 2  >=2 segments under %.0f cm with charge within %.0f cm  %3d  (%.1f%%)"
          % (STUB, TOUCH, len(m2), 100.0 * len(m2) / n))
    print("  both modes in one event ................................. %3d"
          % len([r for r in m1 if len(r["stubs"]) >= 2]))
    print()
    print("  isochronous (a near-vertex segment within %.0f deg of the" % ISO)
    print("  drift-perpendicular plane, doc pr/73 sec 4.5):")
    for name, pop in (("mode 1", m1), ("mode 2", m2)):
        k = len([r for r in pop if r["iso"] is not None and r["iso"] < ISO])
        print("    %s: %d of %d  (held out of the candidate list)"
              % (name, k, len(pop)))
    print()
    print("  degree of the clicked vertex, all scorable events:")
    hist = {}
    for r in good:
        hist[r["deg"]] = hist.get(r["deg"], 0) + 1
    print("    " + "  ".join("deg%d=%d" % (k, hist[k]) for k in sorted(hist)))
    print("  rival vertices within %.0f cm of the click:" % NEAR_VTX)
    hist = {}
    for r in good:
        hist[r["rivals"]] = hist.get(r["rivals"], 0) + 1
    print("    " + "  ".join("n%d=%d" % (k, hist[k]) for k in sorted(hist)))
    return m1, m2


def control(good, m1, m2):
    """Is any of this a discriminator, or does it fire everywhere?

    doc pr/80 sec 10.8 is the standing lesson: a flag measured only on the
    events it was designed for says nothing.  The denominator here is every
    scorable event, and the interesting comparison is against the events where
    the owner's click and the reconstruction agree (b1 <= 1 cm), because those
    are the ones where the topology is the only thing that can be wrong.
    """
    agree = [r for r in good if r["b1"] is not None and r["b1"] <= CLICK_TOL]
    print("=== is the defect concentrated where the vertex was picked WELL?")
    print("  events where click and reco main_vertex agree (b1 <= %.1f cm): "
          "%d of %d" % (CLICK_TOL, len(agree), len(good)))
    for name, pop in (("mode 1", m1), ("mode 2", m2)):
        k = len([r for r in pop if r["b1"] is not None and r["b1"] <= CLICK_TOL])
        base = 100.0 * len(pop) / len(good)
        here = 100.0 * k / max(1, len(agree))
        print("    %s: %d of %d agreeing events (%.1f%%) vs %.1f%% overall "
              "-- x%.2f" % (name, k, len(agree), here, base, here / base))
    print("  A ratio near 1 means the defect is NOT an artefact of a")
    print("  mis-picked vertex: it is there when the vertex is right.")
    print()


def candidates(m1, m2):
    """The non-isochronous shortlists the 6 events are drawn from."""
    def ok(r):
        return (r["iso"] is not None and r["iso"] >= ISO
                and r["owned"] is None
                and r["b1"] is not None and r["b1"] <= CLICK_TOL)

    print("=== candidate shortlist: non-isochronous, vertex agreed, not owned")
    print("    by another round (%s)"
          % ", ".join("%d=%s" % (k, v) for k, v in sorted(OWNED.items())))
    print()
    def show(title, pop, key, col):
        print("--- %s" % title)
        for r in pop[:12]:
            print("    evt%-7d iso%s deg=%d riv=%d b1=%.2f  %s%s"
                  % (r["ev"], fmt_iso(r), r["deg"], r["rivals"], r["b1"],
                     json.dumps(r[col]),
                     "  [pr84 micro]" if r["micro84"] else ""))
        print()

    c1 = sorted([r for r in m1 if ok(r) and r["via"]],
                key=lambda r: -max(x["len"] for x in r["via"]))
    show("mode 1a-VIA: prong connected only through stubs, by prong length",
         c1, None, "via")
    c1c = sorted([r for r in m1 if ok(r) and r["cut"]],
                 key=lambda r: -max(x["len"] for x in r["cut"]))
    show("mode 1a-CUT: prong with no path to the vertex, by prong length",
         c1c, None, "cut")
    c1b = sorted([r for r in m1 if ok(r) and r["straddle"]],
                 key=lambda r: -max(x["len"] for x in r["straddle"]))
    show("mode 1b STRADDLE: segment interior passes the vertex unbroken",
         c1b, None, "straddle")
    c2 = sorted([r for r in m2 if ok(r)],
                key=lambda r: (-len(r["stubs"]), -r["rivals"]))
    show("mode 2 STUB CHAIN, by stub count then rival vertices "
         "(id, len cm, charge->click cm, attached?)", c2, None, "stubs")
    return c1, c1c, c1b, c2


# --------------------------------------------------- examine_* replay tables
#
# The predicates below are transcribed from the production source and evaluated
# on the dump geometry.  They say what each pass WOULD see; they cannot see the
# kProtectedBreak flag, which the dump does not carry.  Wherever a gate depends
# on it the table prints "pb?" and the doc resolves it from the ES3CENSUS log
# (NeutrinoStructureExaminer.cxx:518, env WCT_ES3_MERGE_CENSUS) instead of
# guessing.

EV4_LEN = 2.0        # cm   NeutrinoStructureExaminer.cxx:2084
EV4_MAG = 3.5        # cm   :2085
EV4_BAND = 10.0      # deg from 90 to the drift axis, :2085
ES3_A10 = 18.0       # deg  :565-575 (angle_10cm)
ES3_A3 = 27.0        # deg  :575-585 (angle_3cm)
ESF2_DIS = 2.0       # cm   :3420 examine_structure_final_2
ESF3_DIS = 2.5       # cm   :3611 examine_structure_final_3
MVGA_STUB = 2.0      # cm   NeutrinoPatternBase.h:477
MVGA_RADIUS = 15.0   # cm   NeutrinoPatternBase.h:471
MVGA_SATELLITE = 0.0  # cm  NeutrinoPatternBase.h:480 -- main vertex only


def replay(row):
    """Per-defect table: which pass owns it, its threshold, this event's value."""
    with open(row["path"]) as fh:
        dump = json.load(fh)
    by_id = {s["id"]: s for s in dump.get("segments", [])}
    seg_of = vtx_io.segments_of_vertex(dump)
    vby = vtx_io.vertices_by_id(dump)
    click = None
    for v in dump.get("vertices", []):
        if v["id"] == row["vid"]:
            click = vtx_io.vertex_xyz(v)
    out = []

    for d in row["via"] + row["cut"]:
        sid = d["seg"]
        s = by_id[sid]
        # Which end is the one near the click, and what vertex holds it?
        pts = seg_pts(s)
        e0 = math.dist(pxyz(pts[0]), click)
        e1 = math.dist(pxyz(pts[-1]), click)
        end = "start" if e0 <= e1 else "end"
        vid = d["near_end_vertex"]
        v = vby.get(vid)
        gap = math.dist(vtx_io.vertex_xyz(v), click) if v and vtx_io.vertex_xyz(v) else None
        sibs = [t for t in seg_of.get(vid, []) if t["id"] != sid]
        # examine_structure_3 would have merged this prong into a sibling only
        # if the junction is degree 2 and the two arms are collinear.
        ang10 = None
        if len(sibs) == 1:
            other = sibs[0]
            oend = ("start" if other.get("start_vertex_id") == vid else "end")
            ang10 = angle_deg(unit_dir(s, end), unit_dir(other, oend))
            ang10 = None if ang10 is None else 180.0 - ang10
        out.append({
            "kind": "1a-VIA" if d["path"] else "1a-CUT", "seg": sid,
            "len": d["len"], "charge_to_click": d["charge_to_click"],
            "same_cluster": d["same_cluster"], "path": d["path"],
            "near_end_vertex": vid, "vertex_gap": None if gap is None else round(gap, 2),
            "vertex_degree": len(seg_of.get(vid, [])),
            "siblings": [(t["id"], round(t.get("length", 0), 2)) for t in sibs],
            "es3_angle_10cm": None if ang10 is None else round(ang10, 1),
            "es3_would_merge": (ang10 is not None and ang10 < ES3_A10),
            "esf2_in_range": gap is not None and gap < ESF2_DIS,
            "esf3_in_range": gap is not None and gap < ESF3_DIS,
        })

    for d in row["straddle"]:
        out.append(dict(d, kind="1b straddle",
                        note="no examine_* pass breaks a segment at a vertex "
                             "it does not own; the break decision is "
                             "segment_search_kink (PRSegmentFunctions.cxx:676), "
                             "upstream of every pass in this table"))

    for sid, length, dmin, attached in row["stubs"]:
        s = by_id[sid]
        dl = direct_length(s)
        da = drift_angle(s)
        v1, v2 = s.get("start_vertex_id"), s.get("end_vertex_id")
        d1, d2 = len(seg_of.get(v1, [])), len(seg_of.get(v2, []))
        fires = (dl < EV4_LEN
                 or (dl < EV4_MAG and da is not None
                     and abs(da - 90.0) < EV4_BAND))
        # examine_vertices_4 needs a dying endpoint that is degree>=2, not the
        # main vertex, and not protected.  The clicked vertex is the main
        # vertex on these events, so an endpoint that IS the click is barred.
        cand = [(v, dg) for v, dg in ((v1, d1), (v2, d2))
                if dg >= 2 and v != row["vid"]]
        out.append({
            "kind": "2 stub", "seg": sid, "len": length,
            "direct_length": round(dl, 2),
            "drift_angle": None if da is None else round(da, 1),
            "ev4_predicate": fires,
            "endpoints": [(v1, d1), (v2, d2)],
            "at_click": row["vid"] in (v1, v2),
            "ev4_eligible_endpoint": cand or "NONE (deg<2, or the only "
                                             "candidate IS the main vertex)",
            "mvga_op3_len_ok": dl < MVGA_STUB,
            "mvga_op3_terminal": (d1 == 1 or d2 == 1),
        })
    return out


def mvga_op3_ledger(row):
    """Why main_vertex_graph_audit op3 does or does not absorb each stub.

    Transcribed from NeutrinoGraphAudit.cxx:391-415.  op3 walks the segments
    incident on an ANCHOR (the main vertex; also, when mvga_satellite > 0, any
    main-cluster vertex within that radius) and absorbs one when:

        len < m_mvga_stub (2 cm)                      :395
        the far vertex is not the main vertex          :397
        degree(far vertex) == 1        -- TERMINAL     :398
        far vertex not kProtectedBreak                 :399
        and (corridor overlap >= 0.7 OR <= 4 fits)     :413-416
        and the anchor has >= 2 incident segments      :391

    The `degree(far) == 1` line is the one that matters for this doc: an
    interposed stub -- one whose far end carries the long prong -- has
    degree >= 2 there by construction, so op3 can never absorb it.  That is
    mode 1a-VIA, and it is not a tuning question.
    """
    with open(row["path"]) as fh:
        dump = json.load(fh)
    seg_of = vtx_io.segments_of_vertex(dump)
    by_id = {s["id"]: s for s in dump.get("segments", [])}
    mv = row["vid"]
    n_at_main = len(seg_of.get(mv, []))
    out = []
    for sid, length, dmin, attached in row["stubs"]:
        s = by_id[sid]
        v1, v2 = s.get("start_vertex_id"), s.get("end_vertex_id")
        if mv not in (v1, v2):
            out.append((sid, length, "not incident on the main vertex -- "
                        "invisible to op3 unless mvga_satellite > 0"))
            continue
        far = v2 if v1 == mv else v1
        dfar = len(seg_of.get(far, []))
        if n_at_main < 2:
            why = ("the main vertex has %d incident segment(s); op3 needs >= 2 "
                   "to shed one" % n_at_main)
        elif length >= MVGA_STUB:
            why = ("len %.2f >= mvga_stub %.1f cm" % (length, MVGA_STUB))
        elif dfar != 1:
            why = ("far vertex %s has degree %d, not 1 -- op3 is TERMINAL-only; "
                   "this is an INTERPOSED stub" % (far, dfar))
        else:
            why = "eligible (subject to the overlap / point-degeneracy gate)"
        out.append((sid, length, why))
    return out


def main():
    labs, rs = rows()
    good = [r for r in rs if not r["unmatched"]]
    bad = [r for r in rs if r["unmatched"]]
    print(__doc__.split("\n\n")[0])
    print()
    print("labels with a deployed dump: %d   scorable: %d   click further than "
          "%.1f cm from any vertex: %d" % (len(labs), len(good), CLICK_TOL,
                                           len(bad)))
    print()
    m1, m2 = census(good)
    print()
    control(good, m1, m2)
    c1, c1c, c1b, c2 = candidates(m1, m2)

    # Three mode-1 and three mode-2, no event used twice.  Mode 1 is drawn
    # VIA-first because that is the population the owner described; CUT and
    # STRADDLE backfill so all three shapes are represented if they can be.
    pick = []
    for pool in (c1, c1c, c1b, c1):
        for r in pool:
            if len(pick) >= 3:
                break
            if r["ev"] not in pick:
                pick.append(r["ev"])
    for r in c2:
        if len(pick) >= 6:
            break
        if r["ev"] not in pick:
            pick.append(r["ev"])
    print("=== the six events (3 x mode 1, 3 x mode 2, no event twice)")
    print("    %s" % ", ".join("evt%d" % e for e in pick))
    print()
    byev = {r["ev"]: r for r in good}
    for ev in pick:
        r = byev[ev]
        print("--- evt%d  %s  cluster %s vertex %s  deg=%d rivals=%d iso=%s "
              "b1=%.2f" % (ev, r["tag"], r["cid"], r["vid"], r["deg"],
                           r["rivals"], fmt_iso(r).strip(), r["b1"]))
        for d in replay(r):
            print("    %s" % json.dumps(d, sort_keys=True))
        for sid, length, why in mvga_op3_ledger(r):
            print("    mvga op3  seg %-7s %5.2f cm  %s" % (sid, length, why))
        print()

    print("=== mvga op3 decline ledger over the whole mode-2 population")
    reasons = {}
    for r in good:
        if len(r["stubs"]) < 2:
            continue
        for _, _, why in mvga_op3_ledger(r):
            key = why.split(" -- ")[0].split(";")[0]
            key = "far vertex degree != 1 (INTERPOSED)" if "not 1" in key else key
            key = "len >= mvga_stub 2.0 cm" if key.startswith("len ") else key
            reasons[key] = reasons.get(key, 0) + 1
    for key in sorted(reasons, key=lambda k: -reasons[k]):
        print("  %4d  %s" % (reasons[key], key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
