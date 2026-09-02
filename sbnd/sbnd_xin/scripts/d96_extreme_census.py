#!/usr/bin/env python3
"""doc 96 sec 6 -- is TaggerCheckTGM's extreme-group count a flag for over-clustering?

`component_extreme_wcps` already runs on every in-beam main in the PR chain and
logs "-> N extreme group(s)".  A single track should give 2.  This censuses N
against the TGM/STM verdict over doc 95's 25-event sample to decide whether
"N >= 3 on a TGM-tagged in-beam main" is rare enough to be a usable flag.

Answer (see the doc): NO.  Kept as the measured negative.

Repro:  python3 scripts/d96_extreme_census.py
"""
import collections
import os
import re

MAN = "bee/dbg25/dbg25.manifest.tsv"
RE_EXT = re.compile(r'component_extreme_wcps: cluster (\d+) .*-> (\d+) extreme group')
RE_VER = re.compile(r'visit: TaggerCheck(TGM|STM): cluster (\d+) . (?:TGM=)?(\w+)')

rows = []
for line in open(MAN):
    if line.startswith("#"):
        continue
    f = [x.strip() for x in line.split("\t")]
    if f[0] == "bee_idx":
        continue
    idx, _, run, sub, evt, _ql, pr = f
    log = f"{pr}/pr_evt{evt}/wct_pr_evt{evt}.log"
    if not os.path.isfile(log):
        continue
    ext, verd = {}, {}
    for ln in open(log, errors="replace"):
        m = RE_EXT.search(ln)
        if m:
            ext[int(m.group(1))] = int(m.group(2))
        m = RE_VER.search(ln)
        if m:
            verd.setdefault(int(m.group(2)), {})[m.group(1)] = m.group(3)
    for c, n in sorted(ext.items()):
        v = verd.get(c, {})
        rows.append((int(idx), f"{run}-{sub}-{evt}", c, n, v.get("TGM", "-"), v.get("STM", "-")))

print(f"{'idx':>3} {'RSE':<12} {'cid':>4} {'extreme_groups':>15} {'TGM':>6} {'STM':>5}")
for r in rows:
    print(f"{r[0]:>3} {r[1]:<12} {r[2]:>4} {r[3]:>15} {r[4]:>6} {r[5]:>5}")
hist = dict(sorted(collections.Counter(r[3] for r in rows).items()))
tgm = [r for r in rows if r[4] in ("true", "1")]
hit = [r for r in tgm if r[3] >= 3]
print(f"\nmains evaluated: {len(rows)}   histogram of extreme groups: {hist}")
print(f"TGM-tagged mains: {len(tgm)};  of those with >=3 extreme groups: {len(hit)} "
      f"({100.0*len(hit)/max(1,len(tgm)):.0f}%)")
print("  ", [(r[1], r[2], r[3]) for r in hit])
