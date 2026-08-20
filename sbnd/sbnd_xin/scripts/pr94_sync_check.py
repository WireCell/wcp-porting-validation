#!/usr/bin/env python3
"""doc pr/94 §10.1 -- prove T_tagger[i], T_kine[i], the TaggerCheckNeutrino log
and the Bee "mc" layer all refer to the SAME bundle, for every row of an arm.

This is the check that turns "I wrote N rows" into "the rows are the right
rows".  Four independent producers are cross-joined per row i:

  A. T_tagger[i]  -> (cluster_id, matched_flash_gid, nu_index, nu_x/y/z)
  B. T_kine[i]    -> (cluster_id, matched_flash_gid, nu_index, kine_nu_*_corr)
  C. the wct_pr log's per-row publish sentinel
       "TaggerCheckNeutrino: [nu_per_bundle] ROW i gid G cluster C vertex (...)"
     -- emitted at the point the row's TaggerInfo/KineInfo are stashed, NOT at
     selection time: main_cluster can be repointed in between by
     swap_main_cluster, so the selection line's cluster id is not necessarily
     the id the row carries.
  D. mabc-pr.zip's data/0/0-mc.json roots, whose synthetic per-bundle root is
     labelled "nu <i> (gid G, cluster C)"
  E. mabc-pr.zip's POINT layers (track_fit / shower_track / vertices), which
     are produced by a different function than D and were missed by the first
     version of this check -- the owner's 2026-08-19 Bee scan of NCpi0 evt
     18625 found the second candidate rendered as a bare particle-flow root
     with no points behind it.  Root cause was the same unnamed-slot read
     already fixed once in fill_bee_pf_tree; D passing while E failed is
     exactly why the two are now checked separately.

Checks per row: A.identity == B.identity (exact); A.nu_* == B.kine_nu_*_corr
(exact -- T_tagger's nu_x/y/z ARE KineInfo's corrected vertex, so any drift
means the two trees were filled from different fitters); A.identity == C's
(gid, cluster) at the same ordinal; A.identity == D's parsed label; and D's
root position within POS_TOL cm of the vertex (they are the same point, but
Bee rounds to 6 significant digits).

Usage: pr94_sync_check.py <arm> [--verbose]
Exit 0 = every row of every event passes.
"""
import argparse
import json
import os
import re
import zipfile

import uproot

POS_TOL = 0.01  # cm; Bee JSON rounds, the ROOT branches do not

# NOT anchored on the "[nu_per_bundle]" prefix, and searched over the whole
# file rather than line by line, ON PURPOSE.  WCT's logger is written from
# several threads into one fd, so a record's PREFIX can be destroyed by an
# interleaved write while its body survives -- e.g. mcp2k evt 90751 holds a
# line that begins bare at "ROW 0 gid 1000002 ...", with its neighbours'
# timestamps running backwards (18:02:37.474 then 18:02:34.207).  A
# prefix-anchored, per-line parse reported 7 sync FAILURES across mcp1k+mcp2k
# that were purely its own fragility: every one of those tree rows matched its
# surviving log record exactly on gid, cluster and vertex.  A gate that fails
# on log corruption when the DATA is right is a broken gate.
RE_ROW = re.compile(
    r"ROW (\d+) gid (-?\d+) cluster (-?\d+) "
    r"vertex \((-?[\d.]+), (-?[\d.]+), (-?[\d.]+)\) cm")
RE_ROOT = re.compile(r"^nu (\d+) \(gid (-?\d+), cluster (-?\d+)\)$")


def read_root(path):
    """-> (T_tagger, T_kine) arrays, or (None, None) when the event produced
    no tagger output at all.

    TaggerCheckNeutrino selecting no candidate anywhere leaves the grouping
    with no TrackFitting, so UbooneTaggerOutputVisitor returns before booking
    either tree and the file carries only Trun/T_proj/T_bad_ch.  rc is still 0
    and this is common (539 of 1000 mcp1k events with the knob off), so it is
    a state to recognise, not an error.
    """
    with uproot.open(path) as f:
        names = [k.split(";")[0] for k in f.keys()]
        if "T_tagger" not in names or "T_kine" not in names:
            return None, None
        tt = f["T_tagger"].arrays(library="np")
        tk = f["T_kine"].arrays(library="np")
    return tt, tk


def read_log(path):
    """Collect ROW records, keyed by row index and de-duplicated.

    Both `wct_pr_evt<ID>.log` and its sibling `stdout.log` carry the same
    records, so a record lost to interleaved writes in one file is usually
    intact in the other -- read both and take the union.  Keyed by row index
    rather than appended, so the same record appearing in both files counts
    once and order is by index, not by which file was read first.
    """
    recs = {}
    for p in (path, os.path.join(os.path.dirname(path), "stdout.log")):
        if not os.path.exists(p):
            continue
        for m in RE_ROW.finditer(open(p, errors="replace").read()):
            recs.setdefault(int(m.group(1)),
                            (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                             [float(m.group(4)), float(m.group(5)), float(m.group(6))]))
    return [recs[k] for k in sorted(recs)]


POINT_LAYERS = ("track_fit", "shower_track", "vertices")

# cm.  Every row that reconstructed a vertex must have SOME point of each layer
# near that vertex.  Measured on the fixed arms: track_fit and vertices land
# exactly on it (0.00 cm -- the vertex's own fit point is appended to both) and
# shower_track's nearest associate point is 0.2-0.5 cm away.  A candidate that
# was NOT rendered misses by the distance between the two bundles: 352.7 cm on
# the pre-fix NCpi0 evt 18625.  10 cm therefore separates the two populations
# by more than an order of magnitude at both ends.
LAYER_TOL = 10.0


def read_bee_layers(path):
    """{layer: [(x,y,z), ...]} for the PR point layers, or None if absent.

    NOTE the join is POSITIONAL, not by cluster_id.  cluster ids are re-issued
    by `enumerate_idents` after every visitor in MultiAlgBlobClustering's main
    loop, so the id T_tagger recorded at TaggerCheckNeutrino time is a
    DIFFERENT EPOCH from the id the Bee dump writes (SBND 18255/10550: selected
    cluster 7 in T_tagger, the same activity carries 62 by the time the magnify
    visitor sees it).  An id-keyed check reports that as a missing render and
    is simply wrong -- the vertex position is the epoch-independent identity.
    """
    if not os.path.exists(path):
        return None
    out = {}
    with zipfile.ZipFile(path) as z:
        for layer in POINT_LAYERS:
            names = [n for n in z.namelist()
                     if n.endswith("-%s-global.json" % layer)]
            if not names:
                continue
            d = json.loads(z.read(sorted(names)[0]))
            out[layer] = list(zip(d.get("x", []), d.get("y", []), d.get("z", [])))
    return out or None


def nearest(pts, v):
    best = None
    for p in pts:
        d = ((p[0] - v[0]) ** 2 + (p[1] - v[1]) ** 2 + (p[2] - v[2]) ** 2) ** 0.5
        if best is None or d < best:
            best = d
    return best


def read_bee(path):
    if not os.path.exists(path):
        return None
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("-mc.json")]
        if not names:
            return None
        roots = json.loads(z.read(sorted(names)[0]))
    out = []
    for r in roots:
        m = RE_ROOT.match(r.get("text", ""))
        if m:
            out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                        r.get("data", {}).get("start")))
    return out


def check_event(prdir, evt, verbose):
    root = os.path.join(prdir, "tracking-pr.root")
    log = os.path.join(prdir, "wct_pr_evt%d.log" % evt)
    bee = os.path.join(prdir, "mabc-pr.zip")
    if not os.path.exists(root):
        return 0, ["evt %d: no tracking-pr.root" % evt]

    tt, tk = read_root(root)
    if tt is None:
        return 0, []          # no candidate selected anywhere; nothing to sync
    n = len(tt["nu_index"])
    if n != len(tk["nu_index"]):
        return 0, ["evt %d: T_tagger has %d rows, T_kine has %d"
                   % (evt, n, len(tk["nu_index"]))]
    if n and tt["nu_index"][0] < 0:
        return 0, []          # legacy single-candidate row, nothing to sync

    cands = read_log(log) if os.path.exists(log) else None
    beeroots = read_bee(bee)
    beelayers = read_bee_layers(bee)

    bad = []
    for i in range(n):
        a = (int(tt["cluster_id"][i]), int(tt["matched_flash_gid"][i]),
             int(tt["nu_index"][i]))
        b = (int(tk["cluster_id"][i]), int(tk["matched_flash_gid"][i]),
             int(tk["nu_index"][i]))
        if a != b:
            bad.append("evt %d row %d: T_tagger id %s != T_kine id %s" % (evt, i, a, b))
            continue
        if a[2] != i:
            bad.append("evt %d row %d: nu_index=%d out of order" % (evt, i, a[2]))
        for ta, tb in (("nu_x", "kine_nu_x_corr"), ("nu_y", "kine_nu_y_corr"),
                       ("nu_z", "kine_nu_z_corr")):
            if tt[ta][i] != tk[tb][i]:
                bad.append("evt %d row %d: %s %r != T_kine %s %r"
                           % (evt, i, ta, tt[ta][i], tb, tk[tb][i]))
        if cands is not None:
            if i >= len(cands):
                bad.append("evt %d row %d: no ROW sentinel in the log (log has %d)"
                           % (evt, i, len(cands)))
            elif (cands[i][0], cands[i][1], cands[i][2]) != (a[2], a[1], a[0]):
                bad.append("evt %d row %d: log ROW %s, tree %s"
                           % (evt, i, cands[i][:3], (a[2], a[1], a[0])))
            else:
                dv = max(abs(cands[i][3][j] - float(tt[c][i]))
                         for j, c in enumerate(("nu_x", "nu_y", "nu_z")))
                if dv > POS_TOL:
                    bad.append("evt %d row %d: log vertex %.4f cm from the tree vertex"
                               % (evt, i, dv))

        # A row with no vertex reconstructed (an in-beam bundle that was opened
        # and yielded nothing) has no particle flow, so fill_bee_pf_tree
        # legitimately emits no root for it -- see its "no main vertex" early
        # return.  Only rows that DO carry a vertex must appear in Bee.
        has_vtx = (float(tt["nu_x"][i]), float(tt["nu_y"][i]),
                   float(tt["nu_z"][i])) != (0.0, 0.0, 0.0)
        if beeroots is not None and has_vtx:
            hit = [r for r in beeroots if r[0] == i]
            if not hit:
                bad.append("evt %d row %d: no Bee root node for nu_index %d" % (evt, i, i))
            else:
                _, gid, cid, start = hit[0]
                if (cid, gid) != (a[0], a[1]):
                    bad.append("evt %d row %d: Bee root (gid %d, cluster %d) != tree (gid %d, cluster %d)"
                               % (evt, i, gid, cid, a[1], a[0]))
                if start:
                    dv = max(abs(start[0] - float(tt["nu_x"][i])),
                             abs(start[1] - float(tt["nu_y"][i])),
                             abs(start[2] - float(tt["nu_z"][i])))
                    if dv > POS_TOL:
                        bad.append("evt %d row %d: Bee root %.4f cm from the tree vertex"
                                   % (evt, i, dv))

        # E. the point layers.  Same has_vtx gate as D: a row that
        # reconstructed nothing legitimately contributes no points either.
        if beelayers is not None and has_vtx:
            v = (float(tt["nu_x"][i]), float(tt["nu_y"][i]), float(tt["nu_z"][i]))
            for layer in POINT_LAYERS:
                pts = beelayers.get(layer)
                if not pts:
                    continue
                dmin = nearest(pts, v)
                if dmin is None or dmin > LAYER_TOL:
                    bad.append("evt %d row %d: nearest Bee '%s' point is %.1f cm from "
                               "this candidate's vertex -- it was not rendered"
                               % (evt, i, layer, -1 if dmin is None else dmin))
    if verbose and not bad:
        print("evt %d: %d row(s) OK" % (evt, n))
    return n, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    evts = sorted(int(m.group(1)) for d in os.listdir(args.arm)
                  if (m := re.match(r"pr_evt(\d+)$", d)))
    nrows = nbad = 0
    for e in evts:
        n, bad = check_event(os.path.join(args.arm, "pr_evt%d" % e), e, args.verbose)
        nrows += n
        for b in bad:
            print("FAIL " + b)
        nbad += len(bad)
    print("# events: %d  rows checked: %d  failures: %d" % (len(evts), nrows, nbad))
    print("PASS" if nbad == 0 else "FAIL")
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
