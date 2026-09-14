#!/usr/bin/env python3
"""doc 109 rev 2: classify calib-pr-evt<ID>.json differences between two arms.

Usage: calib_classes.py <labelA> <labelB> [sample ...]   (run from sbnd_xin/)
Classes: meta.runNo, meta.subRunNo, shower_id (the showers[].shower_id values),
and OTHER:<key> for anything else, so a difference outside the two known
group-mode defects is named instead of absorbed.
"""
import collections, glob, json, os, re, sys

def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{p}.{k}" if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, f"{p}[{i}]")
    else:
        yield p, o

def cls(key):
    if key in ("meta.runNo", "meta.subRunNo"):
        return key
    # wall-clock duration of the DL off-chain (TaggerCheckNeutrino.cxx:4339/4404,
    # MS(Clock::now() - t_total)): differs between ANY two runs, not a physics value
    if re.search(r"(^|\.)vertex_scoreboard\.dual_chain\.off_ms$", key):
        return "wallclock off_ms"
    # top-level showers[] and the per-candidate candidates[i].showers[] layout
    if re.search(r"(^|\.)showers\[\d+\]\.shower_id$", key):
        return "shower_id"
    return "OTHER:" + key

A, B = sys.argv[1], sys.argv[2]
samples = sys.argv[3:] or ["nuecc48", "ncpi0", "mcp1k"]
bad = 0
for s in samples:
    c = collections.Counter(); other = collections.Counter()
    for fb in sorted(glob.glob(f"work-{s}-{B}/pr_evt*/calib-pr-evt*.json")):
        fa = fb.replace(f"work-{s}-{B}/", f"work-{s}-{A}/")
        if not os.path.exists(fa):
            c["only in B"] += 1; continue
        a = dict(flat(json.load(open(fa)))); b = dict(flat(json.load(open(fb))))
        ks = {cls(k) for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
        c[" + ".join(sorted(ks)) or "identical"] += 1
        for k in ks:
            if k.startswith("OTHER:"):
                other[k] += 1
    print(f"== {A} vs {B} {s}: " + ", ".join(f"{k}: {v}" for k, v in sorted(c.items())))
    for k, v in other.most_common(10):
        print(f"   {k}  ({v} files)")
    bad += sum(other.values())
print("=== OTHER differences:", bad)
sys.exit(1 if bad else 0)
