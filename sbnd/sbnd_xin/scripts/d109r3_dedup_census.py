#!/usr/bin/env python3
"""sbnd_xin/docs/109 rev 3: what nu_dedup_flash_group actually removes.

Compares two arms that differ ONLY in the nu_dedup_flash_group knob and reports,
per event, which T_tagger rows disappeared and what they were.  The question the
owner has to answer before any flip is not "does it dedup" but "is the row it
drops ever the true neutrino", so every dropped row is printed with the fields
that decide that: its flash group, its selected cluster and length, whether it
had a vertex at all, its reco Enu and its numu score.

Usage:
  scripts/d109r3_dedup_census.py <labelOff> <labelOn> [--samples ...] [--jobs 24]
"""
import argparse
import collections
import glob
import os
from multiprocessing import Pool

import uproot

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COLS = ["nu_index", "cluster_id", "sel_cluster_id", "matched_flash_gid", "flash_group",
        "flash_tpc", "flash_time_us", "has_vertex", "numu_score", "nue_score",
        "nu_x", "nu_y", "nu_z"]


def rows(path):
    """One dict per T_tagger row, keyed for comparison by (gid, sel_cluster_id)."""
    f = uproot.open(path)
    names = {k.split(";")[0] for k in f.keys()}
    if "T_tagger" not in names:
        return {}
    T = f["T_tagger"].arrays(COLS, library="np")
    K = f["T_kine"].arrays(["kine_reco_Enu"], library="np")
    out = {}
    for i in range(len(T["nu_index"])):
        r = {c: T[c][i].item() for c in COLS}
        r["Enu"] = float(K["kine_reco_Enu"][i])
        # The bundle gid + the activity the selection chose identify the row
        # across arms; nu_index cannot, because dropping a row renumbers it.
        out[(r["matched_flash_gid"], r["sel_cluster_id"])] = r
    return out


def one(args):
    off, on = args
    a, b = rows(off), rows(on)
    dropped = [a[k] for k in a if k not in b]
    added = [b[k] for k in b if k not in a]
    changed = []
    for k in a:
        if k in b:
            for c in ("cluster_id", "has_vertex", "Enu", "numu_score"):
                if a[k][c] != b[k][c]:
                    changed.append((k, c, a[k][c], b[k][c]))
    return os.path.basename(os.path.dirname(off)), len(a), len(b), dropped, added, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labelOff")
    ap.add_argument("labelOn")
    ap.add_argument("--samples", nargs="+", default=["nuecc48", "ncpi0", "mcp1k", "mcp2k"])
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()

    stats = collections.Counter()
    for s in a.samples:
        pairs = []
        for off in sorted(glob.glob(f"{SX}/work-{s}-{a.labelOff}/pr_evt*/tracking-pr.root")):
            on = off.replace(f"-{a.labelOff}/", f"-{a.labelOn}/")
            if os.path.isfile(on):
                pairs.append((off, on))
        if not pairs:
            continue
        print(f"== {s}: {len(pairs)} events in both arms")
        with Pool(a.jobs) as p:
            for evt, na, nb, dropped, added, changed in p.imap(one, pairs):
                stats["events compared"] += 1
                stats["rows off"] += na
                stats["rows on"] += nb
                if added:
                    stats["EVENTS WITH ADDED ROWS (unexpected)"] += 1
                    print(f"  !! {evt}: {len(added)} row(s) ADDED by the knob")
                if changed:
                    stats["EVENTS WITH CHANGED SURVIVING ROWS (unexpected)"] += 1
                    print(f"  !! {evt}: surviving row changed: {changed[:3]}")
                if not dropped:
                    continue
                stats["events with a dropped row"] += 1
                stats["rows dropped"] += len(dropped)
                print(f"  {evt}: {na} -> {nb} row(s)")
                for r in dropped:
                    stats["dropped rows with a vertex"] += int(r["has_vertex"] == 1)
                    stats["dropped rows with NO vertex (placeholder)"] += int(r["has_vertex"] != 1)
                    print(f"     dropped gid {r['matched_flash_gid']} (tpc {r['flash_tpc']}, group "
                          f"{r['flash_group']}, t0 {r['flash_time_us']:.4f} us) sel cluster "
                          f"{r['sel_cluster_id']} has_vertex {r['has_vertex']} Enu {r['Enu']:.1f} MeV "
                          f"numu {r['numu_score']:.3f} vtx ({r['nu_x']:.1f}, {r['nu_y']:.1f}, {r['nu_z']:.1f})")
    print("== totals")
    for k in sorted(stats):
        print(f"  {k:46s} {stats[k]}")


if __name__ == "__main__":
    main()
