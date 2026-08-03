#!/usr/bin/env python3
"""doc pr/11 sec 5: per-process component-configuration cost.

run_pr_chain_batch.sh runs ONE wire-cell process PER EVENT, so every one-time
per-process cost is paid 1071 times.  The two BDT scorers load TMVA/xgboost
weight XML at configure time; this measures that from the `-L debug` lines

  D [  main  ] configuring component: "X":"pr"
  D [  main  ] configured component:  "X":"pr"

and compares it against the same event's WCT `Timer: Total` (which covers only
graph execution, NOT configuration) -- that gap is why timecmd's wall_s is
several seconds larger than the Timer's wall on every event.
"""
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timedelta

ROOT = sys.argv[1]
WANT = ('UbooneNueBDTScorer', 'UbooneNumuBDTScorer', 'TaggerCheckNeutrino',
        'MultiAlgBlobClustering')

TS = re.compile(r'^\[(\d\d):(\d\d):(\d\d)\.(\d\d\d)\]')
ING = re.compile(r'configuring component: "([A-Za-z]+)"')
ED = re.compile(r'configured component:  "([A-Za-z]+)"')
TIMER = re.compile(r'Timer: Total ([0-9.]+) wall-sec, ([0-9.]+) core-sec')


def t_of(line):
    m = TS.match(line)
    if not m:
        return None
    h, mi, s, ms = (int(x) for x in m.groups())
    return timedelta(hours=h, minutes=mi, seconds=s, milliseconds=ms)


cfg = defaultdict(list)
timer_wall = []
total_cfg = []

logs = sorted(glob.glob(os.path.join(ROOT, 'pr_evt*', 'wct_pr_evt*.log')))
for path in logs:
    start = {}
    per_event = {}
    tw = None
    with open(path, errors='replace') as fh:
        for line in fh:
            t = t_of(line)
            if t is not None:
                m = ING.search(line)
                if m:
                    start[m.group(1)] = t
                    continue
                m = ED.search(line)
                if m and m.group(1) in start:
                    d = (t - start[m.group(1)]).total_seconds()
                    if d < 0:
                        d += 86400.0
                    per_event[m.group(1)] = d
                    continue
            m = TIMER.search(line)
            if m:
                tw = float(m.group(1))
    for k, v in per_event.items():
        cfg[k].append(v)
    if per_event:
        total_cfg.append(sum(per_event.values()))
    if tw is not None:
        timer_wall.append(tw)


def d(v, nd=3):
    v = sorted(v)
    if not v:
        return 'n=0'
    return (f"n={len(v)} min={v[0]:.{nd}f} p50={v[len(v)//2]:.{nd}f} "
            f"p90={v[int(.9*len(v))]:.{nd}f} max={v[-1]:.{nd}f} "
            f"mean={st.mean(v):.{nd}f} TOTAL={sum(v):.1f}s")


print(f"logs parsed: {len(logs)}\n")
print("--- per-component configure cost (seconds, once per process) ---")
for k in sorted(cfg, key=lambda k: -sum(cfg[k])):
    if k in WANT or sum(cfg[k]) > 5:
        print(f"  {k:<28} {d(cfg[k])}")
print("\n--- ALL components summed, per process ---")
print(f"  {'configure total':<28} {d(total_cfg)}")
print("\n--- WCT Timer wall (graph execution only, EXCLUDES configure) ---")
print(f"  {'Timer wall':<28} {d(timer_wall, 2)}")
if total_cfg and timer_wall:
    print(f"\n  median configure / median Timer wall = "
          f"{sorted(total_cfg)[len(total_cfg)//2]:.2f} / "
          f"{sorted(timer_wall)[len(timer_wall)//2]:.2f} = "
          f"{sorted(total_cfg)[len(total_cfg)//2]/sorted(timer_wall)[len(timer_wall)//2]:.2f}x")
    print(f"  configure total across the sample: {sum(total_cfg):.0f} s")
