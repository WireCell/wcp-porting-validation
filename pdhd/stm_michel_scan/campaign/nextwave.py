#!/usr/bin/env python3
"""Compute the next wave's chunks from what is actually on disk.

    nextwave.py ROUND_DIR ITEMS_FILE WAVE [--agents 5] [--per 12] [--seed S]

doc pdhd/18.  Ported from the doc pdvd/55 tranche-2 scratch copy (never
committed); the round dir, item list and wave name are arguments now.

The remaining list is recomputed by diffing ITEMS_FILE against every record in
ROUND_DIR/v_parts/*/ every time, never tracked in a variable.  That is what
makes a scanner dying mid-chunk harmless: whatever it wrote is banked, whatever
it did not is simply still missing and gets handed out again.

Scanners write into a private dir each (v_parts/<wave>_a<i>/) so that no
scanner can see another's verdicts -- on PDVD a shared out dir let one scanner
`ls` 49 finished calls, an anchor an independent scan must not have.

RULE (doc pdvd/55 sec 16): never run this while any agent of the previous wave
is still writing.  A wave computed over a list that still holds items a live
scanner has not banked yet hands them out twice -- that race cost PDVD 8
duplicate scans.  The caller waits for every agent's report first.

--double N (with --from-waves) instead draws N already-scanned items, seeded,
for an independent re-scan by agents that did not write them.
"""
import argparse, json, os, random, sys

ap = argparse.ArgumentParser()
ap.add_argument("round"); ap.add_argument("items"); ap.add_argument("wave")
ap.add_argument("--agents", type=int, default=5)
ap.add_argument("--per", type=int, default=12)
ap.add_argument("--double", type=int, default=0,
                help="draw this many ALREADY-scanned items for a blind re-scan")
ap.add_argument("--seed", type=int, default=20260911)
a = ap.parse_args()

PARTS = os.path.join(a.round, "v_parts")
os.makedirs(PARTS, exist_ok=True)
want = [l.strip() for l in open(a.items) if l.strip() and not l.startswith("#")]

done, by_writer, broken = set(), {}, []
for dd in sorted(os.listdir(PARTS)):
    p = os.path.join(PARTS, dd)
    if not os.path.isdir(p):
        continue
    for fn in sorted(os.listdir(p)):
        if not fn.endswith(".json"):
            continue
        try:
            r = json.load(open(os.path.join(p, fn)))
            done.add(r["key"])
            by_writer.setdefault(r["key"], set()).add(dd)
        except Exception as e:                   # a killed writer leaves a short file
            broken.append((dd + "/" + fn, str(e)[:60]))
if broken:
    print("!! %d unreadable record(s) -- treated as NOT done:" % len(broken))
    for fn, e in broken:
        print("   %s  %s" % (fn, e))

wdir = os.path.join(a.round, "waves")
os.makedirs(wdir, exist_ok=True)
if any(f.startswith(a.wave + "_a") for f in os.listdir(wdir)):
    sys.exit("wave %r already exists in %s -- pick a new wave name" % (a.wave, wdir))

if a.double:
    pool = sorted(k for k in want if k in done)
    rnd = random.Random(a.seed)
    todo = rnd.sample(pool, min(a.double, len(pool)))
    print("double scan: %d of %d scanned items drawn (seed %d)" % (len(todo), len(pool), a.seed))
else:
    todo = [k for k in want if k not in done]
    print("scanned %d / %d ; remaining %d" % (len([k for k in want if k in done]),
                                               len(want), len(todo)))
    if not todo:
        print("\nWAVES COMPLETE")
        raise SystemExit(0)

chunks = [todo[i::a.agents] for i in range(a.agents)] if a.double else \
         [todo[i * a.per:(i + 1) * a.per] for i in range(a.agents)]
for i, ch in enumerate(chunks):
    if not ch:
        break
    p = os.path.join(wdir, "%s_a%d.txt" % (a.wave, i))
    open(p, "w").write("\n".join(ch) + "\n")
    out = os.path.join(PARTS, "%s_a%d" % (a.wave, i))
    if a.double:
        # a re-scan agent must not re-scan its own item; the writer dirs are
        # named by wave+slot, so this can only be checked after assignment
        clash = [k for k in ch if out.split("/")[-1] in by_writer.get(k, set())]
        if clash:
            sys.exit("double-scan agent %d would re-scan its own items %s" % (i, clash))
    print("\nagent %d: %d items" % (i, len(ch)))
    print("  items file: %s" % p)
    print("  out dir   : %s" % out)
