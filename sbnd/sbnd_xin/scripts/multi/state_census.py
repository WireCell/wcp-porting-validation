#!/usr/bin/env python3
"""doc 82 round 2 -- outcome-state census.

Each draw of an event gets ONE state id: a digest over the member-content
hashes of every Q/L product it wrote (never archive bytes -- CLAUDE.md M2).
Counting distinct ids per event answers "is the outcome still binary".
"""
import hashlib, sys, tarfile, zipfile
from pathlib import Path

def members(p):
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            for n in sorted(z.namelist()):
                yield n, hashlib.sha256(z.read(n)).hexdigest()
    else:
        with tarfile.open(p) as t:
            for m in sorted(t.getmembers(), key=lambda m: m.name):
                if not m.isfile():
                    continue
                f = t.extractfile(m)
                yield m.name, hashlib.sha256(f.read()).hexdigest()

def state(evtdir):
    h = hashlib.sha256()
    for p in sorted(evtdir.iterdir()):
        if p.suffix not in (".zip", ".gz"):
            continue
        h.update(p.name.encode())
        for n, d in members(p):
            h.update(n.encode()); h.update(d.encode())
    return h.hexdigest()[:10]

root = Path(sys.argv[1])
per_evt = {}
for arm in sorted(root.iterdir()):
    if not arm.is_dir():
        continue
    for draw in sorted(arm.glob("draw*")):
        if not (draw / ".done").exists():
            continue
        for ed in sorted(draw.glob("ql_evt*")):
            evt = ed.name[len("ql_evt"):]
            try:
                s = state(ed)
            except Exception as e:
                print(f"  SKIP {ed}: {e}"); continue
            per_evt.setdefault(evt, {}).setdefault(s, []).append(f"{arm.name}/{draw.name}")

for evt in sorted(per_evt):
    st = per_evt[evt]
    n = sum(len(v) for v in st.values())
    print(f"evt {evt}: {n} draws -> {len(st)} distinct state(s)")
    for s, who in sorted(st.items(), key=lambda kv: -len(kv[1])):
        print(f"    {s}  x{len(who):3d}   e.g. {who[0]}")
