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

RE_ROW = re.compile(
    r"\[nu_per_bundle\] ROW (\d+) gid (-?\d+) cluster (-?\d+) "
    r"vertex \((-?[\d.]+), (-?[\d.]+), (-?[\d.]+)\) cm")
RE_ROOT = re.compile(r"^nu (\d+) \(gid (-?\d+), cluster (-?\d+)\)$")


def read_root(path):
    with uproot.open(path) as f:
        tt = f["T_tagger"].arrays(library="np")
        tk = f["T_kine"].arrays(library="np")
    return tt, tk


def read_log(path):
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = RE_ROW.search(line)
            if m:
                out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                            [float(m.group(4)), float(m.group(5)), float(m.group(6))]))
    return out


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
    n = len(tt["nu_index"])
    if n != len(tk["nu_index"]):
        return 0, ["evt %d: T_tagger has %d rows, T_kine has %d"
                   % (evt, n, len(tk["nu_index"]))]
    if n and tt["nu_index"][0] < 0:
        return 0, []          # legacy single-candidate row, nothing to sync

    cands = read_log(log) if os.path.exists(log) else None
    beeroots = read_bee(bee)

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
