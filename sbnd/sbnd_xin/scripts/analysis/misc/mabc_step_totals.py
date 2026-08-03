#!/usr/bin/env python3
"""Sum the PR job's per-step 'MABC timing' lines over a 30-event arm.

Usage: python3 mabc_step_totals.py <tag> [<tag> ...]
       (tags name work-{mcp10,mcp1000,mcp1000b}-<tag> roots)

Prints one row per pipeline step with the total seconds per arm, so a perf or
scope change can be attributed to a step rather than to whole-job wall (which
also carries config compile + tensor IO).  Used by docs/54 and docs/55.
"""
import glob
import os
import re
import sys

ROOTS = ["mcp10", "mcp1000", "mcp1000b"]
RE = re.compile(r"MABC timing: (.+?) took ([\d.]+) ms")
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

tags = sys.argv[1:]
if not tags:
    sys.exit(__doc__)

tot = {t: {} for t in tags}
nev = {t: 0 for t in tags}
for t in tags:
    for r in ROOTS:
        for log in sorted(glob.glob(f"{HERE}/work-{r}-{t}/nusel_evt*/wct_nusel_evt*.log")):
            nev[t] += 1
            with open(log, errors="replace") as f:
                for line in f:
                    m = RE.search(line)
                    if m:
                        tot[t][m.group(1)] = tot[t].get(m.group(1), 0.0) + float(m.group(2)) / 1000.0

steps = sorted({s for t in tags for s in tot[t]},
               key=lambda s: -max(tot[t].get(s, 0) for t in tags))
w = max(len(s) for s in steps) + 2
print(f"events: " + "  ".join(f"{t}={nev[t]}" for t in tags))
print("step".ljust(w) + "".join(f"{t:>12}" for t in tags))
for s in steps:
    if max(tot[t].get(s, 0) for t in tags) < 0.05:
        continue
    print(s.ljust(w) + "".join(f"{tot[t].get(s, 0):12.2f}" for t in tags))
print("TOTAL".ljust(w) + "".join(f"{sum(tot[t].values()):12.2f}" for t in tags))
