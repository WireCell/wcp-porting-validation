#!/usr/bin/env python3
"""Parse MABC per-stage timing from standalone reprocess logs.
Attributes each 'MABC timing: <Stage>:<scope> took <ms> ms' line to the MABC
node instance (apa0-0 / apa1-0 / all) and the event ident currently loaded in
that node. Emits per-stage and per-event aggregates."""
import re, sys, glob
from collections import defaultdict

LOAD = re.compile(r'<MultiAlgBlobClustering:([a-z0-9_-]+)> loading tensor set ident=(\d+)')
TIME = re.compile(r'<MultiAlgBlobClustering:([a-z0-9_-]+)> MABC timing: ([A-Za-z0-9_]+):([a-z0-9_-]+) took ([0-9.]+) ms')

def node_of(scope):
    if scope.startswith('all'): return 'all'
    if scope.startswith('apa0-0'): return 'apa0-0'
    if scope.startswith('apa1-0'): return 'apa1-0'
    return scope

# (node) -> current ident
cur = {}
# (ident, node, stage) -> ms   ; stage aggregated (drop scope suffix variants)
ev_stage = defaultdict(float)
stage_tot = defaultdict(float)     # stage -> total ms
stage_max = defaultdict(float)     # stage -> max single ms
stage_max_ev = {}                  # stage -> (ident,node) of max
ev_tot = defaultdict(float)        # ident -> total clustering ms (all nodes/stages)
ev_node = defaultdict(float)       # (ident,node) -> ms
events = set()
files = sys.argv[1:]
for fn in files:
    for line in open(fn, errors='ignore'):
        m = LOAD.search(line)
        if m:
            cur[m.group(1)] = m.group(2); events.add(m.group(2)); continue
        m = TIME.search(line)
        if not m: continue
        node_scope, stage, tscope, ms = m.group(1), m.group(2), m.group(3), float(m.group(4))
        node = node_of(node_scope)
        ident = cur.get(node_scope, '?')
        stage_tot[stage] += ms
        ev_tot[ident] += ms
        ev_node[(ident, node)] += ms
        ev_stage[(ident, stage)] += ms
        if ms > stage_max[stage]:
            stage_max[stage] = ms; stage_max_ev[stage] = (ident, node)

print(f"# events with timing: {len(events)}   files: {len(files)}")
gtot = sum(stage_tot.values())
print(f"# total clustering wall summed over all events/nodes/stages: {gtot/1000:.1f} s\n")

print("== STAGE RANKING (by total ms across all events) ==")
print(f"{'stage':34s} {'total_s':>9s} {'%':>6s} {'max_ms':>10s} {'max_event':>12s}")
for stage, tot in sorted(stage_tot.items(), key=lambda x:-x[1]):
    me = stage_max_ev.get(stage, ('?','?'))
    print(f"{stage:34s} {tot/1000:9.1f} {100*tot/gtot:6.1f} {stage_max[stage]:10.0f} {me[0]+'/'+me[1]:>12s}")

print("\n== SLOWEST EVENTS (by total clustering ms, all nodes) ==")
for ident, tot in sorted(ev_tot.items(), key=lambda x:-x[1])[:15]:
    # dominant stage for this event
    stages = [(s, ev_stage[(ident,s)]) for s in stage_tot if (ident,s) in ev_stage]
    ds, dm = max(stages, key=lambda x:x[1]) if stages else ('?',0)
    print(f"  ident={ident:>8s}  total={tot/1000:7.1f}s   dominant: {ds} ({dm/1000:.1f}s)")

print("\n== PER-NODE SHARE (sum over events) ==")
node_tot = defaultdict(float)
for (ident,node),ms in ev_node.items(): node_tot[node]+=ms
for node,ms in sorted(node_tot.items(), key=lambda x:-x[1]):
    print(f"  {node:10s} {ms/1000:8.1f}s  {100*ms/gtot:5.1f}%")
