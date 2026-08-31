#!/usr/bin/env python3
"""pr124 flip-equivalence by decomposition: each flipchk event must be
byte-identical to the single-knob arm that owns it (onA / onC) or to the
OFF arm (dbg); events where BOTH knobs fire may match neither and are
listed for combined-effect verification."""
import hashlib, os, re, sys
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest")
import hash_archive

def rollup(p):
    r = hashlib.sha256()
    for name, payload in hash_archive.members(p):
        r.update(hashlib.sha256(name.encode()+payload).hexdigest().encode())
    return r.hexdigest()

def evhash(arm, ev):
    hs = []
    for n in ("mabc-pr.zip", "pctree-pr-evt%d.tar.gz" % ev):
        p = os.path.join(arm, "pr_evt%d" % ev, n)
        if os.path.exists(p): hs.append(rollup(p))
    return tuple(hs)

flip, onA, onC, off = sys.argv[1:5]
evs = sorted(int(m.group(1)) for d in os.listdir(flip) if (m := re.match(r"pr_evt(\d+)$", d)))
counts = {"dbg":0,"onA":0,"onC":0,"NONE":[]}
for ev in evs:
    h = evhash(flip, ev)
    if h == evhash(off, ev): counts["dbg"] += 1
    elif h == evhash(onA, ev): counts["onA"] += 1
    elif h == evhash(onC, ev): counts["onC"] += 1
    else: counts["NONE"].append(ev)
print("%s: %d events -> dbg %d, onA %d, onC %d, NONE %s" % (
    os.path.basename(flip), len(evs), counts["dbg"], counts["onA"], counts["onC"], counts["NONE"]))
