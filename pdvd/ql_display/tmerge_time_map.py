#!/usr/bin/env python
"""Build the old->new flash-time translation map across the flash_tail_merge
reprocess (doc 26 step 4).

The tail merge absorbs a late flash member into its seed and the merged flash
keeps the SEED time, so scan verdicts recorded at a LATE member's time (e.g.
doc 23 track C, matched at raw 2801.95) no longer join any flash within the
standard 0.5 us tolerance.  This script compares an OLD (pre-merge light) and
NEW (merge-ON light) calib-dump tag pair -- same QL op point, same trigger
offsets, so times share one frame -- and records, per event, every OLD flash
time that vanished together with the surviving earlier seed it merged into:

  gone flash g maps to seed s iff  0 < t_g - t_s <= --window (3.5 us default,
  tail_window 3.0 + slack) and s survives in BOTH dumps.

Anything gone without such a seed is REPORTED (QL flash admission can also cut
flashes; those are not merges and are not mapped).  Output JSON:
  {"window": w, "old_tag": ..., "new_tag": ...,
   "events": {"<evt>": [[t_old_us, t_new_us], ...]}}

Consumers: ql_agree_score.py --truth-time-map, remap_scan_state.py --time-map.

Usage (from pdvd/):
  python ql_display/tmerge_time_map.py --old-tag tm0k --new-tag tm0 \
      --out work/ql_scores/tm0/time_map.json
"""

import argparse
import json
import os

RUN = "039252"
EVT0, EVT_STEP, NEVT = 298567, 14, 18
EPS = 1e-3  # us: flash times are bit-stable within one frame


def flash_times(calib_path):
    with open(calib_path) as fh:
        d = json.load(fh)
    return sorted({round(f["time"], 4) for f in d["flashes"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-tag", required=True)
    ap.add_argument("--new-tag", required=True)
    ap.add_argument("--work-root", default="work")
    ap.add_argument("--window", type=float, default=3.5,
                    help="max seed lookback (us) = tail_window + slack")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    events, n_map, n_unmapped = {}, 0, 0
    for idx in range(NEVT):
        evt = EVT0 + EVT_STEP * idx
        paths = [os.path.join(args.work_root, f"{RUN}_{idx}_{tag}",
                              f"calib-evt{evt}.json")
                 for tag in (args.old_tag, args.new_tag)]
        if not all(os.path.exists(p) for p in paths):
            print(f"[warn] evt{evt}: missing dump, skipped")
            continue
        told, tnew = (flash_times(p) for p in paths)
        set_new = set(tnew)
        both = [t for t in told if t in set_new]  # seeds must survive in both
        pairs = []
        for t in told:
            if t in set_new:
                continue  # survived: not merged (or exactly reproduced)
            seeds = [s for s in both if 0.0 < t - s <= args.window]
            if not seeds:
                n_unmapped += 1
                print(f"  [unmapped] evt{evt} t={t:.3f} us gone with no seed "
                      f"in window (admission cut, not a merge)")
                continue
            pairs.append([t, max(seeds)])  # nearest earlier survivor
        events[str(evt)] = pairs
        n_map += len(pairs)
        print(f"evt{evt}: {len(told)} -> {len(tnew)} flashes, "
              f"{len(pairs)} merge pairs mapped")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(dict(window=args.window, old_tag=args.old_tag,
                       new_tag=args.new_tag, events=events), fh, indent=1)
    print(f"TOTAL {n_map} pairs mapped, {n_unmapped} unmapped -> {args.out}")


if __name__ == "__main__":
    main()
