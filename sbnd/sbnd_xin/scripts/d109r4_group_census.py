#!/usr/bin/env python3
"""sbnd_xin/docs/109 rev 4: what nu_bundle_flash_group did, event by event.

Compares two arms that differ ONLY in the nu_bundle_flash_group knob and
reports, for every event the knob is ELIGIBLE to touch (a candidate whose flash
group holds a second in-window gid with clusters, scripts/d109r4_eligible_census.py):

  * the contact decision the toolkit logged for each same-group gid pair
    ("[nu_bundle_flash_group] gids A (tpc) and B (tpc) share flash group G:
    closest clusters a / b at d cm, x / x cm -> MERGED | kept apart"), with d;
  * the T_tagger rows before and after, keyed by the selected activity
    (sel_cluster_id -- nu_index cannot key them, a merge renumbers): rows
    dropped (expected: the other half of a merged pair), rows added (a merged
    bundle whose pick changed), and for every surviving row whether it changed
    and how (cluster_id, has_vertex, vertex_moved_cluster, n_companion,
    kine_reco_Enu, vertex);
  * the partner mains admitted as companions, from the log.

The local samples are DATA (no truth), so this is a reconstruction census: it
says what moved and by how much, and it asserts what must NOT move -- every
row of a non-eligible event in both arms is bit-identical (the rev 3 lesson:
a knob that perturbs its no-op path reads as a clean result otherwise).

Usage:
  scripts/d109r4_group_census.py <labelOff> <labelOn> [--samples ...] [--jobs 24]
"""
import argparse
import collections
import glob
import math
import os
import re
from multiprocessing import Pool

import uproot

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COLS = ["nu_index", "cluster_id", "sel_cluster_id", "matched_flash_gid", "flash_group", "flash_tpc",
        "has_vertex", "vertex_moved_cluster", "numu_score", "nue_score", "nu_x", "nu_y", "nu_z"]
RE_PAIR = re.compile(r"\[nu_bundle_flash_group\] gids (\d+) \(tpc (-?\d+)\) and (\d+) \(tpc (-?\d+)\) share flash group (-?\d+): "
                     r"closest clusters (-?\d+) / (-?\d+) at d ([-\d.na]+) cm, x ([-\d.na]+) / ([-\d.na]+) cm -> (.*)$")
RE_PARTNER = re.compile(r"\[nu_bundle_flash_group\] gid (\d+): partner main cluster (\d+) \(gid (\d+), L ([\d.]+) cm, TGM=(\w+) STM=(\w+)\) admitted as companion of cluster (\d+)")


def rows(path):
    f = uproot.open(path)
    names = {k.split(";")[0] for k in f.keys()}
    out = {}
    if "T_tagger" in names and f["T_tagger"].num_entries:
        T = f["T_tagger"].arrays(COLS, library="np")
        K = f["T_kine"].arrays(["kine_reco_Enu"], library="np")
        B = f["T_bundle"].arrays(["nu_index", "n_companion", "n_companion_dropped", "n_main", "n_demoted"], library="np") \
            if "T_bundle" in names else None
        ncomp = {}
        if B is not None:
            for i in range(len(B["nu_index"])):
                if B["nu_index"][i] >= 0:
                    ncomp[int(B["nu_index"][i])] = (int(B["n_companion"][i]), int(B["n_companion_dropped"][i]),
                                                    int(B["n_main"][i]), int(B["n_demoted"][i]))
        for i in range(len(T["nu_index"])):
            r = {c: T[c][i].item() for c in COLS}
            r["Enu"] = float(K["kine_reco_Enu"][i])
            r["bundle"] = ncomp.get(int(T["nu_index"][i]), (-1, -1, -1, -1))
            out[r["sel_cluster_id"]] = r
    nb = f["T_bundle"].num_entries if "T_bundle" in names else -1
    return out, nb


def log_lines(evtdir):
    evt = os.path.basename(evtdir)[6:]
    pairs, partners = [], []
    try:
        for ln in open(os.path.join(evtdir, "wct_pr_evt%s.log" % evt), errors="replace"):
            m = RE_PAIR.search(ln)
            if m:
                g = m.groups()
                pairs.append(dict(ga=int(g[0]), ta=int(g[1]), gb=int(g[2]), tb=int(g[3]), group=int(g[4]),
                                  ca=int(g[5]), cb=int(g[6]), d=float(g[7]) if g[7] != "nan" else math.nan,
                                  xa=g[8], xb=g[9], merged=g[10].startswith("in contact")))
                continue
            m = RE_PARTNER.search(ln)
            if m:
                g = m.groups()
                partners.append(dict(gid=int(g[0]), cluster=int(g[1]), pgid=int(g[2]), L=float(g[3]),
                                     tgm=g[4], stm=g[5], main=int(g[6])))
    except FileNotFoundError:
        pass
    return pairs, partners


def one(args):
    off, on, eligible = args
    a, nba = rows(off)
    b, nbb = rows(on)
    pairs, partners = log_lines(os.path.dirname(on))
    dropped = [a[k] for k in a if k not in b]
    added = [b[k] for k in b if k not in a]
    changed = []
    for k in a:
        if k in b:
            diff = {}
            for c in ("cluster_id", "has_vertex", "vertex_moved_cluster", "Enu", "numu_score", "nue_score",
                      "matched_flash_gid", "bundle"):
                if a[k][c] != b[k][c]:
                    diff[c] = (a[k][c], b[k][c])
            dv = math.sqrt(sum((a[k][x] - b[k][x]) ** 2 for x in ("nu_x", "nu_y", "nu_z")))
            if dv > 0:
                diff["vtx_move_cm"] = round(dv, 2)
            if diff:
                changed.append((k, diff))
    return os.path.basename(os.path.dirname(off)), eligible, a, b, nba, nbb, dropped, added, changed, pairs, partners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labelOff")
    ap.add_argument("labelOn")
    ap.add_argument("--samples", nargs="+", default=["nuecc48", "ncpi0", "mcp1k", "mcp2k"])
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()

    stats = collections.Counter()
    dvals = []
    for s in a.samples:
        elig = set()
        ef = os.path.join(SX, "docs", "109_logs", "r4", "events_eligible_%s.txt" % s)
        if os.path.isfile(ef):
            elig = {ln.strip() for ln in open(ef) if ln.strip()}
        jobs = []
        for off in sorted(glob.glob("%s/work-%s-%s/pr_evt*/tracking-pr.root" % (SX, s, a.labelOff))):
            on = off.replace("-%s/" % a.labelOff, "-%s/" % a.labelOn)
            if os.path.isfile(on):
                evt = os.path.basename(os.path.dirname(off))[6:]
                jobs.append((off, on, evt in elig))
        if not jobs:
            continue
        print("== %s: %d events in both arms, %d eligible" % (s, len(jobs), sum(1 for j in jobs if j[2])))
        with Pool(a.jobs) as p:
            for evt, eligible, ra, rb, nba, nbb, dropped, added, changed, pairs, partners in p.imap(one, jobs):
                stats["events compared"] += 1
                if not eligible:
                    stats["non-eligible events"] += 1
                    if dropped or added or changed or nba != nbb or pairs or partners:
                        stats["!! NON-ELIGIBLE EVENT CHANGED"] += 1
                        print("  !! %s is not eligible but changed: dropped %d added %d changed %s bundles %d->%d log pairs %d"
                              % (evt, len(dropped), len(added), changed[:2], nba, nbb, len(pairs)))
                    continue
                stats["eligible events"] += 1
                nm = sum(1 for p_ in pairs if p_["merged"])
                stats["gid pairs tested"] += len(pairs)
                stats["gid pairs merged"] += nm
                stats["gid pairs kept apart"] += len(pairs) - nm
                for p_ in pairs:
                    dvals.append((p_["d"], p_["merged"], evt))
                if nm:
                    stats["events with a merge"] += 1
                if partners:
                    stats["events with a partner main admitted"] += 1
                    stats["partner mains admitted"] += len(partners)
                if dropped:
                    stats["events with a dropped row"] += 1
                    stats["rows dropped"] += len(dropped)
                    for r in dropped:
                        stats["dropped rows with NO vertex (placeholder)" if r["has_vertex"] != 1
                              else "dropped rows WITH a vertex"] += 1
                if added:
                    stats["events with an added row (pick changed)"] += 1
                if changed:
                    stats["events with a changed surviving row"] += 1
                    for k, diff in changed:
                        for c in diff:
                            stats["  surviving-row field changed: %s" % c] += 1
                if not (nm or dropped or added or changed):
                    stats["eligible events unchanged (all pairs kept apart)"] += 1
                    continue
                print("  %s: %d -> %d rows; bundles %d -> %d" % (evt, len(ra), len(rb), nba, nbb))
                for p_ in pairs:
                    print("     pair gid %d (tpc %d) / %d (tpc %d) group %d: clusters %d / %d d %.1f cm x %s / %s -> %s"
                          % (p_["ga"], p_["ta"], p_["gb"], p_["tb"], p_["group"], p_["ca"], p_["cb"], p_["d"],
                             p_["xa"], p_["xb"], "MERGED" if p_["merged"] else "apart"))
                for q in partners:
                    print("     partner main cluster %d (gid %d, L %.1f cm, TGM=%s STM=%s) -> companion of cluster %d (row gid %d)"
                          % (q["cluster"], q["pgid"], q["L"], q["tgm"], q["stm"], q["main"], q["gid"]))
                for r in dropped:
                    print("     dropped  sel %d gid %d tpc %d has_vertex %d Enu %.1f numu %.3f vtx (%.1f, %.1f, %.1f)"
                          % (r["sel_cluster_id"], r["matched_flash_gid"], r["flash_tpc"], r["has_vertex"], r["Enu"],
                             r["numu_score"], r["nu_x"], r["nu_y"], r["nu_z"]))
                for r in added:
                    print("     added    sel %d gid %d tpc %d has_vertex %d Enu %.1f numu %.3f"
                          % (r["sel_cluster_id"], r["matched_flash_gid"], r["flash_tpc"], r["has_vertex"], r["Enu"], r["numu_score"]))
                for k, diff in changed:
                    print("     survivor sel %d: %s" % (k, diff))
    print("== totals")
    for k in sorted(stats):
        print("  %-52s %d" % (k, stats[k]))
    if dvals:
        print("== closest-approach distances of every tested pair (cm), ascending")
        for d, m, evt in sorted(dvals, key=lambda t: (t[0] != t[0], t[0])):
            print("  %8.1f  %s  %s" % (d, "MERGED" if m else "apart ", evt))


if __name__ == "__main__":
    main()
