#!/usr/bin/env python3
"""doc 75 -- nu_selected_as_main_snapshot_all knob census.

Finding A: the DL/SCN vertex path (determine_overall_main_vertex_DL ->
swap_main_cluster) can leave Flags::main_cluster permanently set on a
cluster it swapped ONTO, because the narrow SelectedMainFlagGuard
(nu_selected_as_main) only restores the candidate's own pointer.  This
script measures the observable consequence on the enriched (promoted-main /
multi-candidate) manifest, two ways:

  1. Sentinel count: greps every pr_evt<ID>/wct_pr_evt<ID>.log in the ON
     arm for the "[nu_selected_as_main_snapshot_all] restoring cluster ...
     flag N -> M" line the guard's destructor emits ONLY when the live
     value it is about to restore differs from the pre-pass snapshot --
     i.e. a swap actually moved the flag during that candidate's pass.
  2. Persisted-state diff: compares calib-pr-evt<ID>.json's "steiner" block
     (each entry's is_main_cluster, written from the live post-event
     Flags::main_cluster -- PrDisplayDump::dump_steiner) between the OFF
     (narrow-guard) and ON (wide-guard) arms.  A count of main-flagged
     clusters > 1 in the OFF arm, corrected to <= 1 in the ON arm, is the
     leak made visible and then closed.

Usage: d75_mainflag_census.py <armOFF> <armON> [--label name]
Exit 0 always (census, not a gate).
"""
import argparse
import glob
import json
import os
import re
import sys


def events_of(arm):
    out = {}
    for p in glob.glob(os.path.join(arm, "pr_evt*")):
        m = re.match(r".*pr_evt(\d+)$", p)
        if m:
            out[int(m.group(1))] = p
    return out


def sentinel_count(evt_dir, evt_id):
    log_path = os.path.join(evt_dir, f"wct_pr_evt{evt_id}.log")
    if not os.path.exists(log_path):
        return 0
    n = 0
    with open(log_path, errors="replace") as f:
        for line in f:
            if "[nu_selected_as_main_snapshot_all] restoring cluster" in line:
                n += 1
    return n


def main_flagged_ids(evt_dir, evt_id):
    p = os.path.join(evt_dir, f"calib-pr-evt{evt_id}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    st = d.get("steiner", [])
    return sorted(c.get("cluster_id") for c in st if c.get("is_main_cluster"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armA", help="OFF arm (narrow guard)")
    ap.add_argument("armB", help="ON arm (wide guard)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ea, eb = events_of(args.armA), events_of(args.armB)
    common = sorted(set(ea) & set(eb))
    print(f"# {args.label} events: A={len(ea)} B={len(eb)} common={len(common)}")

    total_sentinels = 0
    n_swap_events = 0
    n_multi_main_off = 0
    n_multi_main_on = 0
    for evt in common:
        n_sent = sentinel_count(eb[evt], evt)
        total_sentinels += n_sent
        if n_sent:
            n_swap_events += 1
        ma, mb = main_flagged_ids(ea[evt], evt), main_flagged_ids(eb[evt], evt)
        if ma is not None and len(ma) > 1:
            n_multi_main_off += 1
        if mb is not None and len(mb) > 1:
            n_multi_main_on += 1
        if n_sent or ma != mb:
            print(f"evt {evt}: sentinels={n_sent} main_flagged OFF={ma} ON={mb}")
    print(f"# summary {args.label}: {n_swap_events}/{len(common)} events show >=1 DL-swap "
          f"during a candidate pass ({total_sentinels} total restores logged); "
          f"multi-main-flagged clusters (a >1 count is the leak, visible): "
          f"OFF {n_multi_main_off}/{len(common)}, ON {n_multi_main_on}/{len(common)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
