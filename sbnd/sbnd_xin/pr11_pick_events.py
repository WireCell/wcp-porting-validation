#!/usr/bin/env python3
"""doc pr/11: derive the event subsets for the DL-acceptance arm and Phase 4.

Ranking key is core_s, NOT wall_s: wall under 20-way concurrency is
contention-dominated (median-wall events span core 0.7-3.6 s), so wall is the
wrong key for picking events whose latency we want to measure.
"""
import csv
import sys

PATH = sys.argv[1]
WHICH = sys.argv[2]

rows = []
with open(PATH) as fh:
    for r in csv.DictReader(fh, delimiter='\t'):
        rows.append(r)


def f(r, k):
    v = r.get(k, '')
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


mcp = [r for r in rows if r['sample'] == 'mcp1k' and f(r, 'core_s') is not None]
mcp.sort(key=lambda r: f(r, 'core_s'))
nue = [r for r in rows if r['sample'] == 'nuecc48']

if WHICH == 'phase4':
    heavy = mcp[-10:][::-1]
    mid = len(mcp) // 2
    med = mcp[mid - 5:mid + 5]
    nue_h = sorted([r for r in nue if f(r, 'core_s') is not None],
                   key=lambda r: -f(r, 'core_s'))[:10]
    for tag, rs in (('HEAVY_MCP1K', heavy), ('MEDIAN_MCP1K', med),
                    ('HEAVY_NUECC48', nue_h)):
        print(f"# {tag}")
        for r in rs:
            print(f"#   evt {r['event']:<8} core={f(r,'core_s'):>7.2f}s "
                  f"wall={f(r,'wall_s'):>7.2f}s {r['event_label']}")
        print(f"{tag}=\"{' '.join(r['event'] for r in rs)}\"")
elif WHICH == 'dlrate':
    # ~200 mcp1k stratified evenly over the core_s ordering (deterministic
    # every-Nth pick, no RNG) + every nuecc48 event.
    n = 200
    last = len(mcp) - 1
    idx = sorted({round(i * last / (n - 1)) for i in range(n)})
    sub = [mcp[i] for i in idx]
    print(f"# mcp1k stratified over core_s: {len(sub)} events "
          f"(core {f(sub[0],'core_s'):.2f} .. {f(sub[-1],'core_s'):.2f} s)")
    print(f"MCP1K_DL=\"{' '.join(r['event'] for r in sub)}\"")
    print(f"# nuecc48: all {len(nue)}")
    print(f"NUECC_DL=\"{' '.join(r['event'] for r in nue)}\"")
