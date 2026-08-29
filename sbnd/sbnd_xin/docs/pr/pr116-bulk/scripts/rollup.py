#!/usr/bin/env python3
"""Roll-up of the agent scan tag, including the category the bucket table hides.

`on_save` stores ONE em.verdict -- the shower selected at save time -- while
marks_by_shower holds many.  So an event whose leading shower is fine but whose
SECOND shower is broken files as `correct` and its finding lives only in the
marks.  That set is listed separately here; it is not a bucket in
em114_categorize.py and would otherwise be invisible.

    ./rollup.py [--tag emscan-0828-agent5]
"""
import argparse, glob, json, os
from collections import Counter

SX = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
PILOT = {"evt400504", "evt280159", "evt284791", "evt175896", "evt284200"}

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="emscan-0828-agent5")
a = ap.parse_args()

lab = os.path.join(SX, "em_labels", a.tag)
verd, conf, prov = Counter(), Counter(), Counter()
marked_correct, start_axis, pio_stored = [], [], []
n_in_tot = n_out_tot = n_showers_marked = 0
marked_events, out_events, q_events = set(), set(), set()
for p in sorted(glob.glob(lab + "/labels-*.json")):
    e = os.path.basename(p)[7:-5]
    r = json.load(open(p))
    em = r.get("em", {}) or {}
    note = (r.get("note") or "")
    v = em.get("verdict")
    verd[v] += 1
    conf[r.get("confidence")] += 1
    prov["pilot" if e in PILOT else "bulk"] += 1
    mbs = em.get("marks_by_shower") or {}
    n = sum(1 for s in mbs.values() for m in s.values() if m in ("in", "out"))
    ni = sum(1 for s in mbs.values() for m in s.values() if m == "in")
    no = sum(1 for s in mbs.values() for m in s.values() if m == "out")
    nq = sum(1 for s in mbs.values() for m in s.values() if m not in ("in", "out"))
    n_in_tot += ni; n_out_tot += no
    if n:
        marked_events.add(e); n_showers_marked += len(mbs)
    if no:
        out_events.add(e)
    if nq:
        q_events.add(e)
    if v == "correct" and n:
        others = [s for s in mbs if str(s) != str(em.get("shower"))]
        marked_correct.append((e, n, ",".join(map(str, others)) or "same shower"))
    if "start/axis wrong" in note:
        start_axis.append(e)
    # the pairing lives in pio.candidates[].gammas["1"|"2"], NOT a flat g1/g2
    for c in ((r.get("pio") or {}).get("candidates") or []):
        g = c.get("gammas") or {}
        if g.get("1") and g.get("2"):
            pio_stored.append(e)
            break

print("tag %s -- %d labels (%s)" % (a.tag, sum(verd.values()), dict(prov)))
print("\nverdicts:")
for k, v in verd.most_common():
    print("   %-32s %3d" % (k, v))
print("\nconfidence:")
for k, v in conf.most_common():
    print("   %-32s %3d" % (k, v))
print("\n'correct' but carrying real IN/OUT marks -- the leading shower is fine "
      "and another one is not.\nNot a bucket in em114_categorize.py:")
for e, n, o in marked_correct:
    print("   %-11s %d mark(s), on shower %s" % (e, n, o))
print("   -> %d event(s)" % len(marked_correct))
print("\nnote carries 'start/axis wrong' (no EM_VERDICTS entry exists): %d"
      % len(start_axis))
print("   " + " ".join(start_axis))
# The pr/119 probe now in the toolkit tree counts hand-scan OUT marks, so make
# this tag's marks countable the same way.
print("\nmarks -- the quantity a shower-expel census consumes:")
print("   %d IN and %d OUT across %d event(s) (%d shower(s))"
      % (n_in_tot, n_out_tot, len(marked_events), n_showers_marked))
print("   events with >=1 OUT: " + " ".join(sorted(out_events)))
if q_events:
    print("   NOTE: %d event(s) also carry '?' marks, which pr/115's IN/OUT rule"
          % len(q_events))
    print("   does not define: " + " ".join(sorted(q_events)))
print("\npi0 pairing stored: %d" % len(pio_stored))
print("   " + " ".join(pio_stored))
