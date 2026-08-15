"""Baselines the hand-scan rules must be shown against (doc pr/80).

A rule set that only beats "do nothing" has not been shown to carry physics.
Five reference points, in increasing order of how much they already know:

  B0   production `main_vertex` on the DEPLOYED arm (min_accept = 10.0).
       The number that matters.  Reproduces doc pr/79's 358/473.
  B1   production `main_vertex` as recorded IN THE LABEL, i.e. the pre-flip
       min_accept = 4.0 arm the scan was actually taken on.
  B2   pure geometry: the lowest-z vertex of the main cluster.
  B3   the non-Bragg endpoint of the longest track in the main cluster
       -- owner rule 5 alone, as a one-line control.
  B3b  the LOWER-Z endpoint of that same track.  B3 embeds R1's Bragg
       polarity, so if that polarity were wrong B3 would inherit the bug and
       stop being an independent control; B3b cannot.

The deployed arm is identified from each dump's own
`vertex_scoreboard.dl_min_accept_score`, never from a directory name -- there
are five `work-*-ma10*` variants and only the field distinguishes them.

Repro:
  cd sbnd_xin && python3 vtx_rules/baselines.py            # all 481
  cd sbnd_xin && python3 vtx_rules/baselines.py --half dev
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_split                                                # noqa: E402
import vtx_geom as G                                             # noqa: E402
import vtx_io                                                    # noqa: E402


def deployed_dump_path(label):
    """The min_accept=10 arm's dump for this label, or None if it has none.

    Only the four `-prod0813` arms have an `-ma10` counterpart; the 8
    `vtxscan-prod0813-mc` labels (r1qlmc / r2mc) do not, which is exactly why
    the deployed-arm sample is 473 and not 481 -- the same 473 doc pr/79 used.
    """
    dep = label["arm"].replace("-prod0813", "-ma10")
    p = os.path.join(vtx_io.BASE, dep, "pr_evt%d" % label["eventNo"],
                     "calib-pr-evt%d.json" % label["eventNo"])
    return p if os.path.exists(p) else None


def check_deployed(dump):
    """Assert this dump really is the deployed operating point."""
    b = dump.get("vertex_scoreboard") or {}
    return b.get("dl_min_accept_score"), b.get("dl_top_k")


def b2_lowest_z(dump):
    cid = vtx_io.main_cluster_id(dump)
    best = None
    for v in dump.get("vertices", []):
        if v.get("cluster_id") != cid:
            continue
        p = vtx_io.vertex_xyz(v)
        if p is None:
            continue
        if best is None or p[2] < best[2]:
            best = p
    return best


def longest_main_track(dump):
    cid = vtx_io.main_cluster_id(dump)
    best = None
    for s in dump.get("segments", []):
        if s.get("cluster_id") != cid or not G.is_track(s):
            continue
        if best is None or s.get("length", 0) > best.get("length", 0):
            best = s
    return best


def b3_track_end(dump, mode):
    """mode "bragg": the end opposite the Bragg rise.  mode "z": the lower-z end."""
    seg = longest_main_track(dump)
    if seg is None:
        return None
    verts = vtx_io.vertices_by_id(dump)
    a, b = G.endpoint_vertices(seg)
    pa = vtx_io.vertex_xyz(verts.get(a, {}))
    pb = vtx_io.vertex_xyz(verts.get(b, {}))
    if pa is None or pb is None:
        return None
    if mode == "z":
        return pa if pa[2] <= pb[2] else pb
    be = G.bragg_end(seg)
    if be is None:
        return None                   # no opinion is not a silent coin flip
    return pb if be == "start" else pa


def evaluate(half=None):
    halves = make_split.load_split()
    keep = halves.get(half) if half else None
    labels = [L for L in vtx_io.load_labels()
              if keep is None or L["key"] in keep]

    names = ["B0 deployed main_vertex", "B1 scanned-arm main_vertex",
             "B2 lowest-z vertex, main cluster",
             "B3 non-Bragg end, longest main track",
             "B3b lower-z end, longest main track"]
    hit1 = {n: 0 for n in names}
    hit3 = {n: 0 for n in names}
    answered = {n: 0 for n in names}
    n_dep = 0
    opoints = set()

    for L in labels:
        # B1 needs no dump at all -- it is carried inside the label.
        answered[names[1]] += L["b1"] is not None
        hit1[names[1]] += vtx_io.correct(L["b1"])
        hit3[names[1]] += vtx_io.correct(L["b1"], vtx_io.TOL_LOOSE)

        path = deployed_dump_path(L)
        if not path:
            continue
        with open(path) as fh:
            dump = json.load(fh)
        opoints.add(check_deployed(dump))
        n_dep += 1

        for name, got in ((names[0], vtx_io.xyz(dump.get("main_vertex"))),
                          (names[2], b2_lowest_z(dump)),
                          (names[3], b3_track_end(dump, "bragg")),
                          (names[4], b3_track_end(dump, "z"))):
            if got is None:
                continue
            answered[name] += 1
            d = vtx_io.dist(L["truth"], got)
            hit1[name] += vtx_io.correct(d)
            hit3[name] += vtx_io.correct(d, vtx_io.TOL_LOOSE)

    print("half=%s   labels=%d   with a deployed arm=%d" %
          (half or "all", len(labels), n_dep))
    print("deployed operating points seen (min_accept, top_k): %s" %
          sorted(opoints, key=str))
    print("\n%-38s %8s %14s %14s" % ("baseline", "answered", "correct@1cm",
                                     "correct@3cm"))
    for n in names:
        a = answered[n]
        print("%-38s %8d %7d %6.1f%% %7d %6.1f%%" %
              (n, a, hit1[n], 100.0 * hit1[n] / max(a, 1),
               hit3[n], 100.0 * hit3[n] / max(a, 1)))
    print("\nNote: percentages are over ANSWERED events.  B3 declines to answer "
          "when\nthe Bragg test has no opinion; B2/B3b always answer when a main "
          "cluster exists.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", choices=["dev", "test"],
                    help="restrict to one half of the frozen split")
    return evaluate(ap.parse_args().half)


if __name__ == "__main__":
    sys.exit(main())
