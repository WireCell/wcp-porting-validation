#!/usr/bin/env python3
"""doc pr/11: per-substage cost breakdown for the full PR chain, across a
whole run_pr_chain_batch.sh out_root (or several).

Aggregates two log idioms, both free at the log level run_pr_chain_batch.sh
already uses (-L debug; perf=true is already the SBND default for MABC and
TaggerCheckNeutrino -- cfg/pgrapher/experiment/sbnd/clus.jsonnet:1106,1240 --
so no extra config or rerun is needed to get this):
  "MABC timing: <step> took <ms> ms"                  -- clus/src/MultiAlgBlobClustering.cxx
  "TaggerCheckNeutrino timing: <substage> took <ms> ms" -- clus/src/TaggerCheckNeutrino.cxx

Modelled on mabc_step_totals.py (doc 54/55) but path-driven (glob over
pr_evt*/wct_pr_evt*.log under one or more out_roots) rather than hardcoded to
the three doc-54 sample roots, and reports per-event median/max alongside the
total so a single pathological event does not read as "the bottleneck" for
every event.

Usage: pr_stage_totals.py <out_root> [<out_root> ...]
"""
import glob
import os
import re
import statistics
import sys

RE_MABC = re.compile(r'MABC timing: (.+?) took ([\d.]+) ms')
RE_NU = re.compile(r'TaggerCheckNeutrino timing: (.+?) took ([\d.]+) ms')


def main():
    roots = sys.argv[1:]
    if not roots:
        sys.exit(__doc__)

    per_event = {}   # step -> [ms, ms, ...] one entry per event that had the step
    nev = 0
    for root in roots:
        for log in sorted(glob.glob(os.path.join(root, 'pr_evt*', 'wct_pr_evt*.log'))):
            nev += 1
            seen_this_event = {}
            with open(log, errors='replace') as f:
                for line in f:
                    for prefix, rx in (('MABC', RE_MABC), ('TaggerCheckNeutrino', RE_NU)):
                        m = rx.search(line)
                        if m:
                            step = f'{prefix}:{m.group(1)}'
                            seen_this_event[step] = seen_this_event.get(step, 0.0) + float(m.group(2))
            for step, ms in seen_this_event.items():
                per_event.setdefault(step, []).append(ms)

    if nev == 0:
        sys.exit(f'ERROR: no pr_evt*/wct_pr_evt*.log under {roots}')

    rows = []
    for step, vals in per_event.items():
        total_s = sum(vals) / 1000.0
        rows.append((step, total_s, len(vals), statistics.median(vals),
                      max(vals), min(vals)))
    rows.sort(key=lambda r: -r[1])

    w = max(len(r[0]) for r in rows) + 2
    print(f'events: {nev}  (steps below appear only in the events that ran them)')
    print(f'{"step":<{w}}{"total_s":>10}{"n_evt":>7}{"median_ms":>11}{"max_ms":>10}{"min_ms":>10}')
    for step, total_s, n, med, mx, mn in rows:
        if total_s < 0.05:
            continue
        print(f'{step:<{w}}{total_s:10.2f}{n:7d}{med:11.2f}{mx:10.2f}{mn:10.2f}')


if __name__ == '__main__':
    main()
