#!/usr/bin/env python3
"""doc pdvd/39 round 2 -- compare two PR arms whose CLUSTER IDS ARE NOT COMPARABLE.

The unmerge_assoc stage splits one cluster into several, so the base arm's id N
and the arm's id N are different objects and an id-keyed set diff is meaningless
(feedback_match_objects_across_layers_before_comparing).  This script matches
objects GEOMETRICALLY through the 'clustering' Bee layer, which both arms dump
from the same underlying blobs:

    for each base cluster, find every arm cluster that holds any of its points;
    the arm cluster inheriting the LARGEST share is the base cluster's heir.

A verdict (STM / stm_fit candidacy) is then scored base-cluster vs heir, and the
split multiplicity is reported alongside so a "verdict moved" line can never be
confused with "the object was cut in two".

Usage:
    d39_unmerge_census.py <base_dir> <arm_dir> [<arm_dir> ...]

Each dir is a work/<run6>_<evt>_<tag>/ holding mabc-pr.zip.

Repro (doc pdvd/39 sec 12):
    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
    docs/nf_sp_img_clus/scripts/d39_unmerge_census.py \
        work/039252_2_d39r2base work/039252_2_d39r2unm
"""
import collections
import glob
import json
import os
import re
import sys
import zipfile

import numpy as np
from scipy.spatial import cKDTree

# Points are matched on the T0-corrected coords the clustering layer dumps.  The
# two arms run the same clus stage output, so identical blobs give identical
# coordinates; the tolerance only guards float formatting in the JSON.
MATCH_TOL = 1e-3
# Connected-component link distance for the satellite census, in cm.  6 cm is
# well past PDVD's ~5 mm point spacing along a track, so a component boundary
# here means real empty space, not sampling.
LINK_CM = 6.0
# A Steiner point further than this (cm) from ANY live 3D point in the whole
# event is "void": fabricated by the retiler, not reconstructed charge.
VOID_CM = 3.0

LAYERS = ("clustering", "stm", "stm_fit", "stm_tagged",
          "steiner_graph", "steiner_terminals")


def tagger_sets(d):
    """{tag: set(cluster id)} from the arm's PR log, for the heir lookup.

    A base STM tag that disappears has NOT necessarily become untagged -- on
    PDVD it usually becomes TGM or FC, i.e. the cosmic verdict survives and only
    its KIND changed.  Reporting the loss without the heir's other verdicts
    would read as a regression when it is a reclassification.
    """
    sets = {"TGM": set(), "STM": set(), "FC": set()}
    logs = glob.glob(os.path.join(d, "wct_pr_*.log"))
    if not logs:
        return sets
    pat = re.compile(r"TaggerCheck(TGM|FC): cluster (\d+) \S+ \1=(true|false)")
    pat_stm = re.compile(r"TaggerCheckSTM: cluster (\d+) \S+ STM=([01])")
    for line in open(logs[0], errors="replace"):
        m = pat_stm.search(line)
        if m:
            if m.group(2) == "1":
                sets["STM"].add(int(m.group(1)))
            continue
        m = pat.search(line)
        if m and m.group(3) == "true":
            sets[m.group(1)].add(int(m.group(2)))
    return sets


def load(d):
    """Return {layer: dict} for one arm dir."""
    zp = os.path.join(d, "mabc-pr.zip")
    z = zipfile.ZipFile(zp)
    out = {}
    for name in LAYERS:
        member = "data/0/0-%s-global.json" % name
        try:
            out[name] = json.loads(z.read(member))
        except KeyError:
            out[name] = None
    return out


def points(layer):
    return np.array([layer["x"], layer["y"], layer["z"]]).T


def ids_of(layer):
    return sorted(set(layer["cluster_id"])) if layer else []


def by_cluster(layer):
    o = collections.defaultdict(list)
    for i, c in enumerate(layer["cluster_id"]):
        o[c].append((layer["x"][i], layer["y"][i], layer["z"][i]))
    return {k: np.array(v) for k, v in o.items()}


def components(pts, link=LINK_CM):
    """Sizes of the connected components of a point set, largest first."""
    if len(pts) < 2:
        return [len(pts)]
    tree = cKDTree(pts)
    par = list(range(len(pts)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for a, b in tree.query_pairs(link):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    return sorted(collections.Counter(find(i) for i in range(len(pts))).values(),
                  reverse=True)


def void_fraction(steiner_pts, live_tree):
    """Fraction of steiner points with no live 3D point within VOID_CM."""
    if steiner_pts is None or not len(steiner_pts):
        return 0.0, 0, 0
    d, _ = live_tree.query(steiner_pts)
    nv = int((d >= VOID_CM).sum())
    return nv / float(len(steiner_pts)), nv, len(steiner_pts)


def heirs(base_cl, arm_cl, arm_tree, arm_ids_flat):
    """{base id: [(arm id, npoints shared), ...] sorted by share desc}."""
    out = {}
    for k, pts in base_cl.items():
        d, j = arm_tree.query(pts)
        owned = arm_ids_flat[j][d < MATCH_TOL]
        tally = collections.Counter(owned.tolist())
        out[k] = sorted(tally.items(), key=lambda kv: -kv[1])
    return out


def report_arm(base_dir, base, arm_dir, arm):
    print()
    print("=" * 72)
    print("ARM %s   vs BASE %s" % (os.path.basename(arm_dir),
                                   os.path.basename(base_dir)))
    print("=" * 72)

    bcl, acl = base["clustering"], arm["clustering"]
    print("  clustering objects: base %d -> arm %d"
          % (len(ids_of(bcl)), len(ids_of(acl))))
    if len(bcl["x"]) != len(acl["x"]):
        print("  NOTE: point counts differ (%d vs %d) -- the two arms did NOT "
              "read the same pctree; matching is unreliable"
              % (len(bcl["x"]), len(acl["x"])))

    # layer-scope consistency: the fit-scoped layers must cover one object set
    print()
    print("  layer object sets (the doc pdvd/39 round-2 scope fix):")
    for tag, d in (("base", base), ("arm", arm)):
        sets = {n: ids_of(d[n]) for n in LAYERS if d[n]}
        scoped = [n for n in ("stm", "stm_fit", "steiner_graph",
                              "steiner_terminals") if n in sets]
        same = len({tuple(sets[n]) for n in scoped}) == 1
        print("    %-5s %s  fit-scoped layers agree: %s   stm_tagged=%d"
              % (tag, "  ".join("%s=%d" % (n, len(sets[n])) for n in scoped),
                 "YES" if same else "NO", len(sets.get("stm_tagged", []))))

    # geometric matching
    apts = points(acl)
    atree = cKDTree(apts)
    aids = np.array(acl["cluster_id"])
    bmap = by_cluster(bcl)
    h = heirs(bmap, acl, atree, aids)

    arm_sets = tagger_sets(arm_dir)
    base_tag = set(ids_of(base["stm_tagged"]))
    arm_tag = set(ids_of(arm["stm_tagged"]))
    base_fit = set(ids_of(base["stm_fit"]))
    arm_fit = set(ids_of(arm["stm_fit"]))

    print()
    print("  STM VERDICT, matched object by object "
          "(heir = arm cluster inheriting the most points):")
    print("    %-6s %8s %6s %8s %-9s %s"
          % ("base", "npts", "split", "heir", "verdict", "note"))
    moved = kept = 0
    for k in sorted(base_fit | base_tag):
        hh = h.get(k, [])
        if not hh:
            print("    %-6d %8d %6s %8s %-9s no geometric heir" %
                  (k, len(bmap[k]), "-", "-", "-"))
            continue
        heir, share = hh[0]
        was = k in base_tag
        now = heir in arm_tag
        # a base cluster that is no longer even a candidate is a verdict change
        cand = heir in arm_fit
        verdict = "%s->%s" % ("STM" if was else "cand",
                              "STM" if now else ("cand" if cand else "dropped"))
        note = ""
        if len(hh) > 1:
            note = "split into %d (heir keeps %d%%)" % (
                len(hh), round(100.0 * share / len(bmap[k])))
        if was != now:
            moved += 1
            where = ""
            if was and not now:
                other = [t for t in ("TGM", "FC") if heir in arm_sets[t]]
                where = (" -> heir is " + "+".join(other)) if other else \
                        " -> heir carries NO cosmic tag"
            note = ("VERDICT MOVED%s; " % where + note) if note \
                else "VERDICT MOVED" + where
        elif was:
            kept += 1
        print("    %-6d %8d %6d %8d %-9s %s"
              % (k, len(bmap[k]), len(hh), heir, verdict, note))
    lost = [k for k in base_tag if h.get(k) and h[k][0][0] not in arm_tag]
    reclass = [k for k in lost
               if h[k][0][0] in arm_sets["TGM"] or h[k][0][0] in arm_sets["FC"]]
    print("    STM tagged: base %d -> arm %d   (%d kept, %d moved)"
          % (len(base_tag), len(arm_tag), kept, moved))
    print("    of the %d base STM tags whose heir is not STM, %d are TGM or FC "
          "in the arm (reclassified, still cosmic) and %d carry no cosmic tag"
          % (len(lost), len(reclass), len(lost) - len(reclass)))

    # satellite + void census over the fit-scoped population of each arm
    print()
    print("  SATELLITE / FABRICATED-POINT CENSUS over the fit-scoped clusters:")
    for tag, d in (("base", base), ("arm", arm)):
        live = cKDTree(points(d["clustering"]))
        stm_by = by_cluster(d["stm"]) if d["stm"] else {}
        sg_by = by_cluster(d["steiner_graph"]) if d["steiner_graph"] else {}
        multi = 0
        tot_v = tot_s = 0
        worst = (0.0, None)
        for k, pts in stm_by.items():
            cs = components(pts)
            if len(cs) > 1:
                multi += 1
            f, nv, ns = void_fraction(sg_by.get(k), live)
            tot_v += nv
            tot_s += ns
            if f > worst[0]:
                worst = (f, k)
        n = len(stm_by)
        print("    %-5s %2d clusters; %d multi-component (%.0f%%); "
              "steiner void %d/%d = %.1f%%; worst cluster %.0f%% (id %s)"
              % (tag, n, multi, 100.0 * multi / n if n else 0,
                 tot_v, tot_s, 100.0 * tot_v / tot_s if tot_s else 0,
                 100.0 * worst[0], worst[1]))


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    base_dir = argv[1]
    base = load(base_dir)
    for arm_dir in argv[2:]:
        report_arm(base_dir, base, arm_dir, load(arm_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
