#!/usr/bin/env python3
"""Census behind doc pr/86 -- orphan segments at the neutrino vertex.

Every number in docs/pr/86_orphan-segments-at-the-vertex.md comes from one run
of this script.  Read-only: it opens post-flip calib dumps and the hand-scan
labels and writes nothing.

    cd sbnd_xin && python3 pr86_orphan_census.py            # summary + top 10
    cd sbnd_xin && python3 pr86_orphan_census.py --json OUT # full rows

THE OBJECT.  An **orphan** is a segment whose charge comes within TOUCH of an
anchor vertex but which is NOT incident on it in the PR graph -- at ANY length.
That last clause is the whole reason this script exists rather than a flag added
to pr85_near_vertex_census.py: pr/85 counted a segment as clutter only below
STUB = 3 cm and as a prong only above LONG = 10 cm, so a non-incident 8 cm
segment was counted as neither.  doc pr/85 sec 11.1 measured 20 of 462 events
lost to that band; evt349945, one of the owner's three exemplars, is one of them.

TWO ANCHORS, RANKED BY THE WORSE.  The owner's three exemplars are each
defective at a different anchor, which is why neither alone is the right
population:

    evt268067  5 orphans at the click,          0 at the reco main_vertex
    evt38856   0 orphans at the click,          4 at the reco main_vertex
    evt349945  1 orphan  at the click,          3 at the reco main_vertex

  A_reco   the dump vertex nearest the dump's own `main_vertex`.  Needs no hand
           label, so it covers the whole 1000-event sample -- and it is where
           the reconstruction actually believes the neutrino is, which is what
           every downstream tagger and BDT sees.
  A_click  the dump vertex nearest the owner's rank-1 pick, where a label
           exists.  This is what the owner is looking at when they say "the PR
           near the vertex is not ideal".

An event whose click has NO PR vertex within CLICK_TOL is recorded as its own
class, never silently dropped: post-flip evt268067 is exactly that case (nearest
vertex 1.52 cm away) and dropping it would have hidden a real regression.

WHY EACH ORPHAN IS AN ORPHAN.  Walking the PR graph from the orphan's own
near-end vertex to the anchor splits them the way doc pr/85 sec 2.2 did, because
the three need different fixes:

  VIA     a path exists and every segment on it is under MVGA_STUB.  The
          segment IS connected, through clutter -- pr/85's interposed stub.
  BENIGN  a path exists through at least one real segment.  Ordinary
          grand-daughter topology, not a defect.
  CUT     no path.  Often a different cluster entirely, in which case the
          missing connection is an upstream clustering decision, not a
          PR-graph one.

All of them are counted in the ranking: the owner's complaint is what the
picture looks like, and a BENIGN orphan is just as much charge sitting on the
vertex unattached.  The split is the diagnosis, not the selection.

Isochronicity is REPORTED, not excluded.  pr/85 held isochronous events out on
the owner's instruction; this round the owner chose evt349945 (0.65 deg out of
the drift-perpendicular plane) as an exemplar, so they are in scope.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vtx_rules"))
import vtx_io                                        # noqa: E402

# Reused rather than copied: iso_deg / nearest_point / arc_to_end / graph_path /
# pxyz / seg_pts all carry pr/85's conventions (chord not path for iso, nearer
# end for arclength, BFS keyed on vertex id so nothing iterates a pointer-keyed
# container).  Importing keeps the two censuses from drifting apart.
sys.path.insert(0, HERE)
import pr85_near_vertex_census as P85                # noqa: E402

# ---------------------------------------------------------------- constants
TOUCH = 3.0        # cm, "this segment's charge is at the vertex" (pr/85 TOUCH)
CLICK_TOL = 1.0    # cm, the pr/78/79/80 tolerance -- do not redefine it here
ANCHOR_TOL = 1.0   # cm, main_vertex -> nearest dump vertex; a bigger gap means
                   # the dump's vertex list and its main_vertex disagree, which
                   # is a different bug and is reported separately
MVGA_STUB = 2.5    # cm, SBND PRODUCTION value since doc pr/85 sec 10.5
                   # (wct-pr-perevt.jsonnet mvga_stub = 2.5).  This is the
                   # ceiling op3 will absorb below, so it is the right knife
                   # for "clutter vs real segment" in the VIA/BENIGN split.
NEAR_VTX = 3.0     # cm, radius for counting rival vertices around the anchor
ISO = 10.0         # deg out of the drift-perpendicular plane (doc pr/73 4.5)

# The post-flip production arms.  work-*-pr85ion2 is the doc pr/85 sec 10.7
# flip-config arm, proven byte-identical to a bare production run by the
# sec 10.7 post-flip smoke; work-pr86-dumps adds the 555 mcp1k events that were
# run without PR_EXTRA_STAGES=pr_display and so had no calib dump.  Order
# matters: the reference arm wins for the overlap-control events.
ARMS = [
    ("mcp1k",   "work-mcp1k-pr85ion2"),
    ("nuecc48", "work-nuecc48-pr85ion2"),
    ("ncpi0",   "work-ncpi0-pr85ion2"),
    ("mcp1k",   "work-pr86-dumps"),
]

# Events that belong to another round.  Named, not silently dropped -- each is a
# real instance of this defect and the doc says which round owns it.
OWNED = {
    283713: "pr/84", 287517: "pr/84", 284794: "pr/84", 65289: "pr/84",
    283040: "pr/83", 59899: "pr/83", 72586: "pr/83",
    59685: "pr/85", 280972: "pr/85", 360535: "pr/85",
    278785: "pr/85", 437699: "pr/85", 314838: "pr/85",
}

# The owner's three exemplars.  The script must reproduce their rows before any
# new claim is read off it; see check_exemplars().
EXEMPLARS = {268067: (0, 5), 38856: (4, 0), 349945: (3, 1)}   # (reco, click)


# ------------------------------------------------------------------ dumps
def find_dumps():
    """{eventNo: (root_name, arm, path)} over the post-flip arms.

    Event ids do not collide across the three ql roots (verified), so a flat
    map keyed on the id is unambiguous.
    """
    out = {}
    for root, arm in ARMS:
        for p in sorted(glob.glob(os.path.join(HERE, arm,
                                               "pr_evt*", "calib-pr-evt*.json"))):
            ev = int(re.search(r"calib-pr-evt(\d+)", p).group(1))
            if ev not in out:                 # first arm listed wins
                out[ev] = (root, arm, p)
    return out


def labels_by_event():
    return {L["eventNo"]: L for L in vtx_io.load_labels()}


# ---------------------------------------------------------------- geometry
def anchor_near(dump, target, tol):
    """(vertex, gap) for the dump vertex nearest `target`, or (None, gap)."""
    best, bd = None, None
    for v in dump.get("vertices", []):
        q = vtx_io.vertex_xyz(v)
        if q is None:
            continue
        d = math.dist(q, target)
        if bd is None or d < bd:
            best, bd = v, d
    if best is None:
        return None, None
    return (best if bd <= tol else None), bd


def classify(dump, seg_of, seg, anchor, click_or_reco_xyz):
    """VIA / BENIGN / CUT for one orphan, with the path that decided it."""
    pts = P85.seg_pts(seg)
    end = ("start" if math.dist(P85.pxyz(pts[0]), click_or_reco_xyz)
           <= math.dist(P85.pxyz(pts[-1]), click_or_reco_xyz) else "end")
    near_vid = seg.get("start_vertex_id" if end == "start" else "end_vertex_id")
    path = P85.graph_path(dump, seg_of, near_vid, anchor["id"])
    if path is None:
        return "CUT", None, None
    trail = [(t["id"], round(t.get("length", 0.0), 2)) for t in path]
    # path_max is carried alongside the label because the VIA/BENIGN cut is a
    # threshold choice, not a fact: a "BENIGN" path whose longest hop is 3.0 cm
    # is visually the same clutter as a VIA path whose longest hop is 2.4 cm.
    # The doc bins on path_max; the label is only the mvga-reachability answer.
    path_max = max((t.get("length", 0.0) for t in path), default=0.0)
    kind = "VIA" if path_max < MVGA_STUB else "BENIGN"
    return kind, trail, round(path_max, 2)


def anchor_hop(orphan):
    """Length of the path segment that is INCIDENT on the anchor, or None.

    This is the number that decides whether mvga op3 can reach an orphan at
    all.  op3 iterates `incident` = the anchor's own out_edges
    (NeutrinoGraphAudit.cxx:384-393) and skips anything with
    `len >= m_mvga_stub` (:395), so the only orphans it can ever attach are
    those separated from the anchor by a stub UNDER the ceiling.  graph_path
    returns segments ordered from the orphan's near-end vertex toward the
    anchor, so the anchor-incident hop is the LAST element.
    """
    p = orphan.get("path")
    return p[-1][1] if p else None


def score_anchor(dump, anchor, target):
    """Every measurement at one anchor vertex."""
    c = vtx_io.vertex_xyz(anchor)
    seg_of = vtx_io.segments_of_vertex(dump)
    here = {s["id"] for s in seg_of.get(anchor["id"], [])}
    touch, orphans, isos = 0, [], []
    for s in dump.get("segments", []):
        pts = P85.seg_pts(s)
        if not pts:
            continue
        dmin, imin = P85.nearest_point(s, c)
        if dmin is None or dmin > TOUCH:
            continue
        touch += 1
        io = P85.iso_deg(s)
        if io is not None:
            isos.append(io)
        if s["id"] in here:
            continue
        kind, path, path_max = classify(dump, seg_of, s, anchor, c)
        orphans.append({
            "seg": s["id"], "len": round(s.get("length", 0.0), 2),
            "dist": round(dmin, 2), "kind": kind, "path": path,
            "path_max": path_max, "hops": len(path) if path else None,
            "anchor_hop": path[-1][1] if path else None,
            "cluster": s.get("cluster_id"),
            "same_cluster": s.get("cluster_id") == anchor.get("cluster_id"),
            "arc_from_end": round(P85.arc_to_end(s, imin), 2),
            "iso": round(io, 1) if io is not None else None,
        })
    rivals = sum(1 for v in dump.get("vertices", [])
                 if vtx_io.vertex_xyz(v)
                 and math.dist(vtx_io.vertex_xyz(v), c) <= NEAR_VTX)
    orphans.sort(key=lambda o: -o["len"])
    return {
        "vid": anchor["id"], "cluster": anchor.get("cluster_id"),
        "gap": None, "deg": len(here), "touch": touch,
        "n_orphan": len(orphans), "orphans": orphans, "rivals": rivals,
        "iso_min": round(min(isos), 2) if isos else None,
    }


# --------------------------------------------------------------------- rows
def scan_one(ev, root, arm, path, label):
    with open(path) as fh:
        dump = json.load(fh)
    row = {"ev": ev, "root": root, "arm": arm, "path": path,
           "owned": OWNED.get(ev), "labelled": label is not None,
           "b1": round(label["b1"], 2) if label and label.get("b1") is not None
                 else None}

    mv = dump.get("main_vertex")
    if mv:
        a, gap = anchor_near(dump, vtx_io.xyz(mv), ANCHOR_TOL)
        if a is not None:
            row["reco"] = score_anchor(dump, a, vtx_io.xyz(mv))
            row["reco"]["gap"] = round(gap, 3)
        else:
            row["reco_unmatched"] = round(gap, 3) if gap is not None else None
    else:
        row["reco_unmatched"] = "no main_vertex in dump"

    if label is not None:
        a, gap = anchor_near(dump, label["truth"], CLICK_TOL)
        if a is not None:
            row["click"] = score_anchor(dump, a, label["truth"])
            row["click"]["gap"] = round(gap, 3)
        else:
            row["click_unmatched"] = round(gap, 3) if gap is not None else None
    return row


def worst(row):
    """(n_orphan, touch, longest orphan) over whichever anchor is worse."""
    best = (0, 0, 0.0, None)
    for key in ("reco", "click"):
        a = row.get(key)
        if not a:
            continue
        t = (a["n_orphan"], a["touch"],
             max((o["len"] for o in a["orphans"]), default=0.0), key)
        if t[:3] > best[:3]:
            best = t
    return best


def rows():
    dumps = find_dumps()
    labs = labels_by_event()
    out = []
    for ev in sorted(dumps):
        root, arm, path = dumps[ev]
        try:
            out.append(scan_one(ev, root, arm, path, labs.get(ev)))
        except Exception as exc:                       # a truncated dump
            out.append({"ev": ev, "root": root, "arm": arm, "error": str(exc)})
    return out


# ------------------------------------------------------------------ reports
def check_exemplars(rs):
    """Gate: the census must reproduce the owner's three events first."""
    by = {r["ev"]: r for r in rs}
    ok = True
    print("exemplar gate (expected from the plan's post-flip table)")
    for ev, (want_r, want_c) in sorted(EXEMPLARS.items()):
        r = by.get(ev)
        if r is None:
            print("  evt%-7d MISSING from the population" % ev)
            ok = False
            continue
        gr = r.get("reco", {}).get("n_orphan")
        gc = (r.get("click", {}).get("n_orphan")
              if "click" in r else "unmatched(%s)" % r.get("click_unmatched"))
        mark = "ok " if (gr == want_r and (gc == want_c or want_c == 5
                                           and str(gc).startswith("unmatched"))) \
            else "DIFF"
        if mark == "DIFF":
            ok = False
        print("  evt%-7d reco %s (want %s)   click %s (want %s)   %s"
              % (ev, gr, want_r, gc, want_c, mark))
    print("  gate:", "PASS" if ok else "REVIEW -- numbers moved, explain before using")
    print()
    return ok


def census(rs):
    good = [r for r in rs if "error" not in r]
    print("dumps read              : %d  (%s)"
          % (len(rs), ", ".join("%s=%d" % (a, sum(1 for r in rs if r["arm"] == a))
                                for _, a in ARMS)))
    print("unreadable              : %d" % sum(1 for r in rs if "error" in r))
    print("with a reco anchor      : %d   (unmatched %d)"
          % (sum(1 for r in good if "reco" in r),
             sum(1 for r in good if "reco_unmatched" in r)))
    print("with a hand label       : %d   (click anchored %d, click unmatched %d)"
          % (sum(1 for r in good if r["labelled"]),
             sum(1 for r in good if "click" in r),
             sum(1 for r in good if "click_unmatched" in r)))
    print()
    hist = {}
    for r in good:
        hist[worst(r)[0]] = hist.get(worst(r)[0], 0) + 1
    print("orphan count at the WORSE anchor:")
    for k in sorted(hist):
        print("   %d orphan(s) : %4d events" % (k, hist[k]))
    print()
    kinds = {}
    for r in good:
        for key in ("reco", "click"):
            for o in r.get(key, {}).get("orphans", []):
                kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
    print("orphan classification (both anchors pooled):", kinds)
    band = sum(1 for r in good for key in ("reco", "click")
               for o in r.get(key, {}).get("orphans", [])
               if MVGA_STUB <= o["len"] < 10.0)
    over = sum(1 for r in good for key in ("reco", "click")
               for o in r.get(key, {}).get("orphans", []) if o["len"] >= 10.0)
    under = sum(1 for r in good for key in ("reco", "click")
                for o in r.get(key, {}).get("orphans", []) if o["len"] < MVGA_STUB)
    print("orphan length vs the mvga_stub=%.1f ceiling: under %d, %.1f-10 cm %d, "
          ">=10 cm %d" % (MVGA_STUB, under, MVGA_STUB, band, over))

    # The op3-reachability ledger.  op3 only ever looks at segments INCIDENT on
    # the anchor (NeutrinoGraphAudit.cxx:384-395), so an orphan is reachable
    # only through an incident stub under the ceiling.  Everything else is
    # architecturally out of mvga's reach, at ANY knob value.
    reach = {"no path (CUT)": 0, "anchor hop >= ceiling": 0,
             "anchor hop < ceiling": 0}
    for r in good:
        for key in ("reco", "click"):
            for o in r.get(key, {}).get("orphans", []):
                h = o.get("anchor_hop")
                if h is None:
                    reach["no path (CUT)"] += 1
                elif h >= MVGA_STUB:
                    reach["anchor hop >= ceiling"] += 1
                else:
                    reach["anchor hop < ceiling"] += 1
    print("op3 reachability (anchor-incident hop vs mvga_stub=%.1f):" % MVGA_STUB)
    for k in ("anchor hop < ceiling", "anchor hop >= ceiling", "no path (CUT)"):
        print("   %-24s %4d" % (k, reach[k]))
    print()


def top(rs, n=10, skip_owned=True):
    good = [r for r in rs if "error" not in r and worst(r)[0] >= 2]
    if skip_owned:
        good = [r for r in good if not r["owned"]]
    good.sort(key=lambda r: (-worst(r)[0], -worst(r)[1], -worst(r)[2], r["ev"]))
    print("TOP %d by orphan count at the worse anchor%s"
          % (n, "  (events owned by pr/83, pr/84, pr/85 excluded)"
             if skip_owned else ""))
    print("  %-9s %-8s %-6s %-28s %-28s %-6s %s"
          % ("event", "root", "worse", "reco: deg/touch/orph",
             "click: deg/touch/orph", "iso", "orphan lengths (cm)"))
    for r in good[:n]:
        w = worst(r)
        def cell(key):
            a = r.get(key)
            if not a:
                um = r.get(key + "_unmatched")
                return "unmatched (%s cm)" % um if um is not None else "-"
            return "%d / %d / %d" % (a["deg"], a["touch"], a["n_orphan"])
        a = r.get(w[3]) or {}
        print("  evt%-6d %-8s %-6s %-28s %-28s %-6s %s"
              % (r["ev"], r["root"], w[3], cell("reco"), cell("click"),
                 ("%.1f" % a["iso_min"]) if a.get("iso_min") is not None else "-",
                 ", ".join("%s:%.1f(%s)" % (o["seg"], o["len"], o["kind"])
                           for o in a.get("orphans", []))))
    print()
    return good[:n]


def detail(rs, evs):
    by = {r["ev"]: r for r in rs}
    for ev in evs:
        r = by.get(ev)
        if r is None:
            continue
        print("--- evt%d  (%s, %s)%s" % (ev, r["root"], r["arm"],
                                         "  OWNED BY %s" % r["owned"]
                                         if r["owned"] else ""))
        print("    b1 (click -> reco) : %s cm" % r["b1"])
        for key in ("reco", "click"):
            a = r.get(key)
            if not a:
                print("    %-6s : unmatched (%s cm)"
                      % (key, r.get(key + "_unmatched")))
                continue
            print("    %-6s : vertex %s clus %s  deg %d  touch %d  rivals %d  "
                  "iso_min %s" % (key, a["vid"], a["cluster"], a["deg"],
                                  a["touch"], a["rivals"], a["iso_min"]))
            for o in a["orphans"]:
                print("        orphan %-7s %6.2f cm  d=%4.2f  %-6s clus %-4s "
                      "%s arc %.2f  iso %s  path_max %s  path %s"
                      % (o["seg"], o["len"], o["dist"], o["kind"], o["cluster"],
                         "same" if o["same_cluster"] else "OTHER",
                         o["arc_from_end"], o["iso"], o["path_max"], o["path"]))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write all rows to this path")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--detail", nargs="*", type=int,
                    help="print the full per-orphan table for these events")
    args = ap.parse_args()

    rs = rows()
    check_exemplars(rs)
    census(rs)
    picked = top(rs, args.top)
    if args.detail is not None:
        detail(rs, args.detail or [r["ev"] for r in picked])
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rs, fh, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
