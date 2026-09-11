#!/usr/bin/env python3
"""List the applied rows that disagree with their resolved record, per labeldir.

    fixup_spec.py ROUND        -> ROUND/spec/fixup_<lbldir>.json for every lbl_* that needs one

doc pdhd/18.  `scan_harness.py apply` stops with SystemExit when an item's tags
will not land after four passes (the object table re-sorts under the clicks on
big tables, worse on a loaded machine -- doc pdvd/55 sec 17.5), and everything
it saved before is kept.  But the failing item is ALREADY a row in the labeldir
(saved with short tags), so apply_parallel.sh's RESUME, which re-applies only
keys not on disk, skips it.  This compares every row with its resolved record
on verdict (`choice`), kind, every tag and the pin, and writes the mismatches
plus the spec items never applied, one spec per labeldir.  Re-apply each into
ITS OWN labeldir (the viewer rewrites that key's row), never two processes on one
labeldir.
"""
import glob, json, os, sys

R = sys.argv[1]
tot = 0
for L in sorted(glob.glob(os.path.join(R, "lbl_*", "labels.json"))):
    name = os.path.basename(os.path.dirname(L))              # lbl_<batch>_<i>
    spec_f = os.path.join(R, "spec", "spec_%s.json" % name[4:])
    if not os.path.exists(spec_f):
        continue
    rows = json.load(open(L))["labels"]
    todo = []
    for it in json.load(open(spec_f)):
        r = rows.get(it["key"])
        bad = None
        if r is None:
            bad = "not applied"
        elif r.get("choice") != it["verdict"] or r.get("michel_kind") != it["michel_kind"]:
            bad = "verdict/kind"
        elif (r.get("pf_segments") or {}) != it["tags"]:
            bad = "tags"
        elif bool((r.get("pin") or {}).get("placed")) != (it.get("pin_rr") is not None):
            bad = "pin"
        if bad:
            todo.append(dict(it, _why=bad))
    out = os.path.join(R, "spec", "fixup_%s.json" % name[4:])
    if todo:
        json.dump([{k: v for k, v in t.items() if k != "_why"} for t in todo],
                  open(out, "w"), indent=1, ensure_ascii=False)
        print("%s: %d to re-apply  %s" % (name, len(todo), [(t["key"], t["_why"]) for t in todo]))
    elif os.path.exists(out):
        os.remove(out)
    tot += len(todo)
print("total to re-apply: %d" % tot)
