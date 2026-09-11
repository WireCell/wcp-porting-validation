#!/usr/bin/env python3
"""Turn the resolved verdict records into N disjoint apply specs.

    mkspec.py ROUND_DIR ITEMS_FILE NCHUNKS [BATCH]

BATCH (default "all") names the specs spec_<BATCH>_<i>.json, so the apply can be
pipelined between scan waves: each batch's ITEMS_FILE is the items of the waves
that have finished, and no key may appear in two batches (asserted against the
specs already on disk).

doc pdhd/18.  Ported from the doc pdvd/55 tranche-2 scratch copy; reads ONLY
ROUND_DIR/v_resolved (one record per key, by construction), never v_parts --
the PDVD copy walked v_parts with last-writer-wins, which is how a declined pin
got applied (doc pdvd/55 sec 17.6).

Each apply process must own a disjoint key set so the N private labels.json
files can be merged by a plain dict union.  Disjointness is asserted here.
Writes ROUND_DIR/spec/spec_<i>.json.
"""
import collections, json, os, sys

R, items, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
batch = sys.argv[4] if len(sys.argv) > 4 else "all"
V = os.path.join(R, "v_resolved")
want = [l.strip() for l in open(items) if l.strip() and not l.startswith("#")]
recs = {}
for k in want:
    p = os.path.join(V, k.replace("/", "_") + ".json")
    if not os.path.exists(p):
        sys.exit("no resolved record for %s -- run resolve.py first" % k)
    recs[k] = json.load(open(p))

out = os.path.join(R, "spec")
os.makedirs(out, exist_ok=True)
earlier = set()
for fn in os.listdir(out):
    if fn.startswith("spec_") and fn.endswith(".json") and not fn.startswith("spec_%s_" % batch):
        earlier |= {it["key"] for it in json.load(open(os.path.join(out, fn)))}
clash = sorted(earlier & set(want))
if clash:
    sys.exit("%d key(s) already in another batch's spec: %s" % (len(clash), clash[:10]))
sets = []
for i in range(n):
    ch = want[i::n]
    spec = []
    for k in ch:
        r = recs[k]
        it = dict(key=k, verdict=r["verdict"], michel_kind=r["michel_kind"], tags=r["tags"])
        if r.get("notes"):
            it["notes"] = r["notes"]
        if r.get("pin_rr") is not None:
            it["pin_rr"] = r["pin_rr"]
        spec.append(it)
    json.dump(spec, open(os.path.join(out, "spec_%s_%d.json" % (batch, i)), "w"),
              indent=1, ensure_ascii=False)
    sets.append(set(ch))
    print("spec_%s_%d.json  %d items" % (batch, i, len(spec)))
for i in range(n):
    for j in range(i + 1, n):
        if sets[i] & sets[j]:
            sys.exit("chunks %d and %d overlap: %s" % (i, j, sets[i] & sets[j]))
print("chunks disjoint; union %d of %d" % (len(set().union(*sets)), len(want)))
print(dict(collections.Counter(r["verdict"] for r in recs.values())))
