#!/usr/bin/env python3
"""Adjudication packets for a review round (doc pdhd/19).

    mkadj.py ROUND OLD_RECORD DISAGREE_TSV NAGENTS [--seed S]

For every item of DISAGREE_TSV (cmp_review.py --disagree: the stop / thru /
unscored class changed between the old record and the review), writes one
markdown packet per adjudicator, ROUND/adj/adj_items_a<i>.md, with the two
scans as "scan A" and "scan B" in a seeded random order, so the adjudicator is
not told which is older.  Only verdict, michel_kind, confidence, pin, evidence
and notes are copied -- never the chain's verdict, the owner's label, the
scanner id, the wave or the rubric sha (any of which says which scan is which).
The key -> (A, B) mapping is written to --map, which must lie OUTSIDE ROUND (the
adjudicators may read ROUND/adj/ for their own packet).
"""
import argparse, csv, glob, json, os, random

ap = argparse.ArgumentParser()
ap.add_argument("round"); ap.add_argument("old"); ap.add_argument("disagree")
ap.add_argument("nagents", type=int)
ap.add_argument("--map", required=True, help="where the private A/B map goes (outside ROUND)")
ap.add_argument("--seed", type=int, default=20260911)
a = ap.parse_args()

if os.path.abspath(a.map).startswith(os.path.abspath(a.round) + os.sep):
    raise SystemExit("--map must lie outside ROUND")
old = {r["key"]: r for r in json.load(open(a.old))}
new = {}
for f in glob.glob(os.path.join(a.round, "v_parts", "rv*_a*", "*.json")):
    r = json.load(open(f)); new[r["key"]] = r
keys = [r["key"] for r in csv.DictReader(open(a.disagree), delimiter="\t")]
rnd = random.Random(a.seed)
rnd.shuffle(keys)
OUT = os.path.join(a.round, "adj")
os.makedirs(OUT, exist_ok=True)
FIELDS = ("verdict", "michel_kind", "confidence", "pin_rr", "evidence", "notes")


def block(name, r):
    s = ["### scan %s" % name]
    for f in FIELDS:
        if r.get(f) not in (None, ""):
            s.append("* **%s**: %s" % (f, r[f]))
    return "\n".join(s)


ab = {}
chunks = [keys[i::a.nagents] for i in range(a.nagents)]
for i, ch in enumerate(chunks):
    lines = ["# Adjudication packet a%d — %d items (doc pdhd/19)\n" % (i, len(ch)),
             "Frames: `/home/xqian/tmp/h19/shots/<event>_<cluster>/`. Brief: "
             "`/home/xqian/tmp/h19/ADJ_TASK.md`. Rubric: `/home/xqian/tmp/h19/RUBRIC.md`.\n"]
    for k in ch:
        pair = [("old", old[k]), ("review", new[k])]
        rnd.shuffle(pair)
        ab[k] = {"A": pair[0][0], "B": pair[1][0], "agent": i}
        lines.append("## `%s`\n" % k)
        lines.append(block("A", pair[0][1]) + "\n")
        lines.append(block("B", pair[1][1]) + "\n")
    open(os.path.join(OUT, "adj_items_a%d.md" % i), "w").write("\n".join(lines))
    open(os.path.join(OUT, "adj_items_a%d.txt" % i), "w").write("".join(k + "\n" for k in ch))
    print("a%d: %d items" % (i, len(ch)))
json.dump(ab, open(a.map, "w"), indent=1)
