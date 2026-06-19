#!/usr/bin/env python3
"""Round-by-round Q/L chain validation against the run-29107 4-event hand scan.

Reports the winner-diff between a BASELINE dump set and a NEW dump set, BUCKETED by
the user's 3 rules for treating the hand scan (it is a *reasonable* reference, not
absolute truth):

  1. xTPC samples are all correct and MUST be kept       -> HARD GATE (pass/fail)
  2. auto-matches NOT in the hand scan are OK            -> new winners are NEUTRAL
  3. some small hand-scan picks may be sub-optimal       -> non-xTPC accept loss is SOFT

Everything is keyed by (apa, round(flash_time,2), cluster_ident) so it survives the
flash-gid renumbering any measured-PE / selection change causes.  The xTPC GT subset
is the hand-scan-ACCEPTED matches whose BASELINE dump bundle carries xtpc_consistent
(read from the dump, since the label `flags` block has no xtpc field).

Usage:
  validate_chain.py BASELINE_DIR NEW_DIR
    DIR is either flat (DIR/calib-evt<E>.json) or work-structured
    (DIR/029107_<idx>/calib-evt<E>.json).  Pass the same dir twice to self-check.
"""
import json
import os
import sys

PD = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd"
EVTS = {983: 0, 991: 1, 999: 2, 1007: 3}
LAB = PD + "/work/ql_labels/labels-evt%d.json"


def dump_path(base, evt, idx):
    for p in ("%s/calib-evt%d.json" % (base, evt),
              "%s/029107_%d/calib-evt%d.json" % (base, idx, evt)):
        if os.path.exists(p):
            return p
    raise SystemExit("no calib dump for evt%d under %s" % (evt, base))


def bundle_index(d):
    """(apa, round(time,2), cluster_ident) -> bundle."""
    fb = {f["gid"]: f for f in d["flashes"]}
    idx = {}
    for b in d["bundles"]:
        if b["flash_gid"] in fb:
            f = fb[b["flash_gid"]]
            idx[(b["apa"], round(f["time"], 2), b["main_cluster"] % 1000000)] = b
    return idx


def winners(idx):
    return {k for k, b in idx.items() if b.get("auto_selected")}


def acc(entries):
    s = set()
    for m in entries:
        for ci in m["cluster_idents"]:
            s.add((m["apa"], round(m["flash_time_us"], 2), ci))
    return s


def resolve(keys, idx):
    """A hand-scan match lists several cluster_idents on one (apa,time); the bundle
    lives under whichever ident is the main_cluster.  Collapse to the keys that
    actually resolve to a dump bundle (so the metric counts matches, not idents)."""
    by_at = {}
    for (apa, t, c) in keys:
        by_at.setdefault((apa, t), []).append(c)
    out = set()
    for (apa, t), cs in by_at.items():
        hit = next((c for c in cs if (apa, t, c) in idx), None)
        out.add((apa, t, hit if hit is not None else cs[0]))
    return out


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    base_dir, new_dir = sys.argv[1], sys.argv[2]

    # aggregates
    g = dict(xt_tot=0, xt_old=0, xt_new=0,         # xTPC accept preserved (HARD)
             nx_tot=0, nx_old=0, nx_new=0,         # non-xTPC accept preserved (soft)
             rj_tot=0, rj_old=0, rj_new=0,         # reject re-selected (bad)
             neu=0, conflict=0,                    # new winners (neutral / conflicting)
             w_old=0, w_new=0)                     # raw winner totals (informational)
    print("per-event (BASELINE -> NEW), keyed by (apa,time,cluster):")
    print("  %-7s %-18s %-18s %-16s %-14s" %
          ("evt", "xTPC-accept", "nonxTPC-accept", "reject-reselect", "new-win(confl)"))
    for evt, ix in EVTS.items():
        old = json.load(open(dump_path(base_dir, evt, ix)))
        new = json.load(open(dump_path(new_dir, evt, ix)))
        lab = json.load(open(LAB % evt))
        iO, iN = bundle_index(old), bundle_index(new)
        wO, wN = winners(iO), winners(iN)
        accept = resolve(acc(lab["matches"]), iO)
        reject = resolve(acc(lab.get("rejected_auto", [])), iO)

        # xTPC GT subset = accepted matches whose BASELINE bundle is xtpc_consistent
        xt = {k for k in accept if iO.get(k, {}).get("xtpc_consistent")}
        nx = accept - xt

        xt_old, xt_new = len(xt & wO), len(xt & wN)
        nx_old, nx_new = len(nx & wO), len(nx & wN)
        rj_old, rj_new = len(reject & wO), len(reject & wN)

        # new winners not in the hand scan: neutral unless they re-select a reject
        # key or land on a GT-accept (apa,time) with a different cluster (a steal).
        gt_at = {(a, t) for (a, t, c) in accept}
        new_only = wN - wO
        conflict = sum(1 for (a, t, c) in new_only
                       if (a, t) in gt_at and (a, t, c) not in accept)
        neutral = len(new_only) - conflict

        g["xt_tot"] += len(xt); g["xt_old"] += xt_old; g["xt_new"] += xt_new
        g["nx_tot"] += len(nx); g["nx_old"] += nx_old; g["nx_new"] += nx_new
        g["rj_tot"] += len(reject); g["rj_old"] += rj_old; g["rj_new"] += rj_new
        g["neu"] += neutral; g["conflict"] += conflict
        g["w_old"] += len(wO); g["w_new"] += len(wN)
        print("  evt%-4d %2d/%2d -> %2d/%2d      %2d/%2d -> %2d/%2d        %2d -> %2d           %3d (%d)" %
              (evt, xt_old, len(xt), xt_new, len(xt),
               nx_old, len(nx), nx_new, len(nx),
               rj_old, rj_new, neutral, conflict))

    print("\n=== AGGREGATE (4 events) ===")
    hard = "PASS" if g["xt_new"] >= g["xt_old"] else "*** FAIL ***"
    print("  [HARD] xTPC accept preserved: %d/%d  (baseline %d/%d)   %s"
          % (g["xt_new"], g["xt_tot"], g["xt_old"], g["xt_tot"], hard))
    print("  [soft] non-xTPC accept:       %d/%d  (baseline %d/%d)"
          % (g["nx_new"], g["nx_tot"], g["nx_old"], g["nx_tot"]))
    rjflag = "" if g["rj_new"] <= g["rj_old"] else "   <-- REGRESSION (more rejects re-selected)"
    print("  [bad]  reject re-selected:    %d/%d  (baseline %d/%d)%s"
          % (g["rj_new"], g["rj_tot"], g["rj_old"], g["rj_tot"], rjflag))
    print("  [neutral] new winners not in hand scan: %d   (conflicting steals: %d)"
          % (g["neu"], g["conflict"]))
    print("  [info] total auto_selected winners: %d -> %d" % (g["w_old"], g["w_new"]))


if __name__ == "__main__":
    main()
