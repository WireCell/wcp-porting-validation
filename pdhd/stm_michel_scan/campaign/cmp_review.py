#!/usr/bin/env python3
"""The review round against the round it reviews (doc pdhd/19).

    cmp_review.py ROUND OLD_RECORD TIERS_JSON [--owner OWNER_LABELS] [--disagree OUT.tsv]

ROUND/v_parts/rv*_a*/*.json are the blind re-scan's records (mkv.py); the OLD
record is the committed smx18 record; TIERS_JSON maps key -> queue tier (A-E),
which the reviewers never saw.  Tier A is the owner's own 30: the owner's smx1
label is the truth there and is never overridden, so those items are scored
against BOTH the old blind call and the owner, and are kept out of the pooled
agreement number.

Prints agreement on the verdict, on stopper-or-not, and on michel_kind, by
tier and by the two scanners' confidence, the direction-rule (DIRECTION:)
and grey-continuation (CONTINUES:) uses, and writes the items whose
stopper-or-not call changed (the adjudication list) to --disagree.
"""
import argparse, collections, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("round"); ap.add_argument("old"); ap.add_argument("tiers")
ap.add_argument("--owner", default=None)
ap.add_argument("--disagree", default=None)
ap.add_argument("--nonblind", default=None,
                help="file of keys the reviewers could see an answer for (e.g. keys the rubric "
                     "itself names with the owner's verdict); kept out of the pooled numbers "
                     "and shown on their own line")
a = ap.parse_args()
NONBLIND = set(l.strip() for l in open(a.nonblind) if l.strip()) if a.nonblind else set()


def base(v):
    return v[5:] if v and v.startswith("FRAG_") else v


def stop(v):
    return base(v) in ("STM_MICHEL", "STM_ONLY")


def klass(v):
    b = base(v)
    return "stop" if b in ("STM_MICHEL", "STM_ONLY") else ("thru" if b == "THRU" else "unscored")


old = {r["key"]: r for r in json.load(open(a.old))}
tiers = json.load(open(a.tiers))
new = {}
for f in sorted(glob.glob(os.path.join(a.round, "v_parts", "rv*_a*", "*.json"))):
    r = json.load(open(f))
    if r["key"] in new:
        raise SystemExit("two review records for %s" % r["key"])
    new[r["key"]] = r
owner = {}
if a.owner:
    L = json.load(open(a.owner))
    L = L.get("labels", L)
    owner = {k: (v.get("choice") or v.get("label")) for k, v in L.items()}

missing = sorted(set(tiers) - set(new))
print("review records %d of %d queued items%s" % (len(new), len(tiers),
      ("; MISSING %s" % " ".join(missing)) if missing else ""))


def table(keys, title):
    n = len(keys)
    if not n:
        return
    v = sum(base(new[k]["verdict"]) == base(old[k]["verdict"]) for k in keys)
    sc = [k for k in keys if klass(new[k]["verdict"]) != "unscored" and klass(old[k]["verdict"]) != "unscored"]
    s = sum(stop(new[k]["verdict"]) == stop(old[k]["verdict"]) for k in sc)
    mk = [k for k in keys if base(new[k]["verdict"]) == base(old[k]["verdict"]) and stop(new[k]["verdict"])]
    m = sum(new[k]["michel_kind"] == old[k]["michel_kind"] for k in mk)
    print("  %-34s n=%3d  verdict %3d/%-3d  stopper-or-not %3d/%-3d  kind (same verdict) %d/%d"
          % (title, n, v, n, s, len(sc), m, len(mk)))


keys = sorted(new)
print("\nold (smx18, v1-v3, old display) vs review (smx19, v4, fixed display):")
for t in "ABCDE":
    table([k for k in keys if tiers.get(k) == t], "tier %s" % t)
pooled = [k for k in keys if tiers.get(k) in "BCDE" and k not in NONBLIND]
table(pooled, "POOLED B-E (tier A and non-blind out)")
if NONBLIND:
    table([k for k in keys if k in NONBLIND and tiers.get(k) != "A"],
          "non-blind, tiers B-E (not pooled)")
print("\nby confidence, pooled B-E:")
for co in ("high", "medium", "low"):
    table([k for k in pooled if new[k]["confidence"] == co], "review %s" % co)
for co in ("high", "medium", "low"):
    table([k for k in pooled if old[k]["confidence"] == co], "old %s" % co)
table([k for k in pooled if new[k]["confidence"] == "high" and old[k]["confidence"] == "high"],
      "both high")

print("\nverdict transitions, pooled B-E (old -> review):")
tr = collections.Counter((base(old[k]["verdict"]), base(new[k]["verdict"])) for k in pooled)
for (o, n_), c in sorted(tr.items(), key=lambda x: -x[1]):
    print("  %-11s -> %-11s %3d" % (o, n_, c))

if owner:
    A = [k for k in keys if tiers.get(k) == "A"]
    print("\ntier A against the owner's smx1 label (the truth there):")
    for k in A:
        ov = owner.get(k)
        print("  %-15s owner %-11s old-blind %-11s review %-11s (%s)%s"
              % (k, ov, old[k]["verdict"], new[k]["verdict"], new[k]["confidence"],
                 "  = owner" if ov and base(ov) == base(new[k]["verdict"]) else ""))

for pfx in ("DIRECTION:", "CONTINUES:", "FLAT_STOP:", "ANODE:", "OVERSHOOT:", "UNDERSHOOT:"):
    ks = [k for k in keys if pfx in (new[k].get("notes") or "")]
    ko = [k for k in keys if pfx in (old[k].get("notes") or "")]
    print("%-12s review %2d  old %2d" % (pfx, len(ks), len(ko)))

if a.disagree:
    with open(a.disagree, "w") as fh:
        fh.write("key\ttier\told_verdict\told_conf\treview_verdict\treview_conf\n")
        for k in keys:
            if tiers.get(k) == "A":
                continue
            if klass(new[k]["verdict"]) != klass(old[k]["verdict"]):
                fh.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (k, tiers.get(k), old[k]["verdict"],
                         old[k]["confidence"], new[k]["verdict"], new[k]["confidence"]))
    print("\nadjudication list (stop / thru / unscored class changed, tier A excluded) -> %s" % a.disagree)
