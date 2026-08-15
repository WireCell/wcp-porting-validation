"""The polarity control -- run BEFORE any rule is fitted (doc pr/80).

Two questions, both of which can silently invert the whole track branch:

  Q1  Does points[0] sit at start_vertex_id and points[-1] at end_vertex_id?
      vtx_geom.endpoint_vertices() assumes so; this measures it against the
      vertices' own fitted positions.

  Q2  Is "the vertex is at the END OPPOSITE the Bragg rise" actually true?
      Measured only on DEV events where production already agrees with the
      owner, so the truth vertex is not in doubt.  Correct polarity shows as a
      high agreement fraction; an inverted one shows as ~0%, which is a clean
      signal that no amount of threshold tuning could disguise.

Repro:
  cd sbnd_xin && python3 vtx_rules/polarity_control.py
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_split                                                # noqa: E402
import vtx_geom as G                                             # noqa: E402
import vtx_io                                                    # noqa: E402

MIN_TRACK_LEN = 10.0      # cm; a Bragg test needs a track worth testing


def deployed_dump_path(label):
    """The min_accept=10 arm's dump for this label, or None if it has none."""
    dep = label["arm"].replace("-prod0813", "-ma10")
    p = os.path.join(vtx_io.BASE, dep, "pr_evt%d" % label["eventNo"],
                     "calib-pr-evt%d.json" % label["eventNo"])
    return p if os.path.exists(p) else None


def main():
    halves = make_split.load_split()
    dev = halves["dev"]
    labels = [L for L in vtx_io.load_labels() if L["key"] in dev]

    q1 = collections.Counter()
    q2 = collections.Counter()
    q2w = collections.Counter()      # same, restricted to not dir_weak
    dirsign_map = collections.Counter()
    n_ev = 0

    for L in labels:
        path = deployed_dump_path(L)
        if not path:
            continue
        with open(path) as fh:
            dump = json.load(fh)
        mv = vtx_io.xyz(dump.get("main_vertex"))
        d_prod = vtx_io.dist(L["truth"], mv)
        if not vtx_io.correct(d_prod):
            continue                  # truth vertex must not be in doubt
        n_ev += 1

        verts = vtx_io.vertices_by_id(dump)
        truth_vid = L["truth_vid"]

        for seg in dump.get("segments", []):
            pts = G.seg_points(seg)
            if len(pts) < 2:
                continue

            # ---- Q1: point order vs the two named vertices -----------------
            a, b = G.endpoint_vertices(seg)
            va, vb = verts.get(a), verts.get(b)
            if va and vb:
                pa, pb = vtx_io.vertex_xyz(va), vtx_io.vertex_xyz(vb)
                p0 = (pts[0]["x"], pts[0]["y"], pts[0]["z"])
                p1 = (pts[-1]["x"], pts[-1]["y"], pts[-1]["z"])
                if pa is not None and pb is not None:
                    fwd = vtx_io.dist(p0, pa) + vtx_io.dist(p1, pb)
                    rev = vtx_io.dist(p0, pb) + vtx_io.dist(p1, pa)
                    q1["as_assumed" if fwd <= rev else "REVERSED"] += 1

            # ---- Q2: Bragg end vs the truth vertex's end -------------------
            if not G.is_track(seg) or seg.get("length", 0) < MIN_TRACK_LEN:
                continue
            at = G.end_name_of_vertex(seg, truth_vid)
            if at is None:
                continue              # segment not attached to the truth vertex
            be = G.bragg_end(seg)
            key = "no_opinion" if be is None else (
                "OPPOSITE (rule holds)" if be != at else "SAME (rule inverted)")
            q2[key] += 1
            if not seg.get("dir_weak"):
                q2w[key] += 1
            if be is not None:
                dirsign_map[(seg.get("dirsign"), be)] += 1

    def show(title, c):
        tot = sum(c.values())
        print("\n%s  (n=%d)" % (title, tot))
        for k in sorted(c, key=lambda k: -c[k]):
            print("   %-24s %5d  %5.1f%%" % (k, c[k], 100.0 * c[k] / max(tot, 1)))
        if "OPPOSITE (rule holds)" in c or "SAME (rule inverted)" in c:
            dec = tot - c.get("no_opinion", 0)
            hold = c.get("OPPOSITE (rule holds)", 0)
            if dec:
                print("   -> of %d decisive: %.1f%% support the rule" %
                      (dec, 100.0 * hold / dec))

    print("dev events with production already correct: %d" % n_ev)
    show("Q1  points[0] <-> start_vertex_id", q1)
    show("Q2  Bragg end vs truth-vertex end", q2)
    # MEASURED 2026-08-15: this comes out EMPTY, because dir_weak is set on
    # 1335/1377 = 96.9% of segments in a 20-event nuecc48 sample.  It is not a
    # usable quality gate -- gating R1/R3/R5 on `not dir_weak` would discard the
    # rule rather than sharpen it.  Kept in the control so the fact stays
    # measured rather than remembered.
    show("Q2  same, dir_weak segments excluded", q2w)
    print("\nQ2 cross-check, dirsign vs Bragg end (NOT an input to any rule):")
    for k in sorted(dirsign_map, key=lambda k: (str(k[0]), str(k[1]))):
        print("   dirsign=%-4s bragg=%-6s %5d" % (k[0], k[1], dirsign_map[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
