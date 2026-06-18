#!/usr/bin/env python3
"""Re-key a saved PDHD Q/L hand-scan to a freshly reprocessed calib dump.

A scan_state's "selected" entries are [flash_gid, main_cluster] pairs. flash_gid is
a POSITIONAL enumeration index (anode_ident*stride + index-in-run.flashes), so any
reprocess that changes which flashes pass flash_minPE (e.g. the measured_pe_scale
optical retune, which raises APA0 total_PE) renumbers the flashes and breaks the
saved gids -- the clusters are unchanged, only the flash index moved.

This re-keys each pick by the STABLE pair (main_cluster, flash_time): the old gid's
flash time is recovered from a labels file saved against the OLD dump (labels carry
flash_time_us per match), then matched to the bundle in the NEW dump with the same
main_cluster whose flash time is closest. main_cluster is untouched (clustering is
upstream of matching). The original scan_state is preserved as <name>.prereprocess.

Usage:  ./remap_scan_after_reprocess.py <scan_state.json> <labels.json> <new_calib.json>
"""
import json
import os
import sys

TOL_US = 0.5  # flash-time match tolerance (flash times are exact across a reprocess)


def main(scan_path, labels_path, dump_path):
    # Idempotent: re-key always from the ORIGINAL (pre-reprocess) scan, so re-running
    # against a newer dump does not compound. The backup holds that original.
    bak = scan_path + ".prereprocess"
    src = bak if os.path.exists(bak) else scan_path
    sel = json.load(open(src))
    state = sel if isinstance(sel, dict) else {"selected": sel}
    picks = state["selected"]
    g2t = {m["flash_gid"]: m["flash_time_us"] for m in json.load(open(labels_path))["matches"]}
    dump = json.load(open(dump_path))

    # Reproduce the viewer's phantom-flash collapse (Event.__init__): each per-side
    # run dumps the full flash list, so every physical flash appears twice (canonical
    # apa==lit + phantom apa!=lit). The viewer re-points every bundle's flash_gid to
    # the CANONICAL copy (keyed by (lit side, time, total_PE)) and load_state matches
    # the saved (flash_gid, main_cluster) against THOSE collapsed gids -- so we must
    # store collapsed gids, not the raw dump gids.
    flashes = dump["flashes"]
    _lit = lambda f: 0 if sum(f["pe"][80:]) >= sum(f["pe"][:80]) else 1
    _sig = lambda f: (_lit(f), round(f["time"], 4), round(f["total_PE"], 3))
    canon_gid = {_sig(f): f["gid"] for f in flashes if f["apa"] == _lit(f)}
    remap_flash = {f["gid"]: canon_gid[_sig(f)]
                   for f in flashes if f["apa"] != _lit(f) and _sig(f) in canon_gid}
    ftime = {f["gid"]: f["time"] for f in flashes}

    by_mc = {}
    for b in dump["bundles"]:
        cgid = remap_flash.get(b["flash_gid"], b["flash_gid"])   # collapsed gid
        by_mc.setdefault(b["main_cluster"], []).append(cgid)

    out, nfix, nmiss = [], 0, 0
    for old_gid, mc in picks:
        if old_gid not in g2t or mc not in by_mc:
            out.append([old_gid, mc]); nmiss += 1; continue
        t = g2t[old_gid]
        cands = sorted(set(g for g in by_mc[mc] if abs(ftime.get(g, 1e18) - t) <= TOL_US))
        if not cands:
            out.append([old_gid, mc]); nmiss += 1; continue
        new_gid = cands[0]
        out.append([new_gid, mc])
        if new_gid != old_gid:
            nfix += 1
    state["selected"] = out

    if not os.path.exists(bak):
        os.rename(scan_path, bak)   # first run: preserve the original
    with open(scan_path, "w") as fh:
        json.dump(state, fh)
    print("re-keyed %d picks: %d gid-shifted, %d unresolved; backup %s"
          % (len(out), nfix, nmiss, os.path.basename(bak)))
    if nmiss:
        print("  WARNING: %d picks could not be matched (left as-is)" % nmiss)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
