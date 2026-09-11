#!/usr/bin/env python3
"""Merge the N private, app-written labels.json into a NEW scan tag.

    merge.py ROUND DET TAG ITEMS_FILE --check    # gate only, write nothing
    merge.py ROUND DET TAG ITEMS_FILE --write    # gate, then write the tag

Unions every ROUND/lbl_*/labels.json (apply_parallel.sh writes one per batch and
process, so the apply can be pipelined between scan waves).

doc pdhd/18.  Ported from the doc pdvd/55 tranche-2 scratch copy.  There the
merge went INTO a live tag that already held 60 rows (smx1a), so its gate was
"the 60 rows stay byte-identical".  Here the tag is NEW (smx18): the gate is that
it does not exist yet, or -- on a re-run -- that it still has the sha this script
recorded when it wrote it (ROUND/<tag>_post_merge.sha), so an owner edit made
through the app since is never overwritten (M13; the stale-tab guard).

Every row was produced by clicking the real widgets; the N processes only ran
against private label dirs.  Key sets are disjoint by mkspec.py, so this is a
dict union; an intersecting key is refused, not reconciled.  Every item of
ITEMS_FILE must be present, and every row must agree with its resolved record
on verdict, kind and tag count.
"""
import argparse, collections, hashlib, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("round"); ap.add_argument("det"); ap.add_argument("tag")
ap.add_argument("items")
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--check", action="store_true"); g.add_argument("--write", action="store_true")
a = ap.parse_args()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.dirname(os.path.dirname(HERE))
LIVE = os.path.join(IMG, a.det, "work", "stm_michel_labels", a.tag, "labels.json")
SHAF = os.path.join(a.round, "%s_post_merge.sha" % a.tag)

if os.path.exists(LIVE):
    live_sha = hashlib.sha256(open(LIVE, "rb").read()).hexdigest()
    mine = open(SHAF).read().split()[0] if os.path.exists(SHAF) else None
    if live_sha != mine:
        sys.exit("!! %s exists and is not the file this script wrote (sha %s vs %s) -- "
                 "someone has edited it; refusing (M13)" % (LIVE, live_sha[:12], (mine or "none")[:12]))
    print("live %s is this script's own earlier write (%s)" % (a.tag, live_sha[:12]))

merged, seen, head = {}, collections.Counter(), None
import glob
lbls = sorted(glob.glob(os.path.join(a.round, "lbl_*", "labels.json")))
if not lbls:
    sys.exit("no lbl_*/labels.json under %s" % a.round)
for p in lbls:
    whole = json.load(open(p))
    if head is None:     # the viewer's own header (scan/doc/det/tag/sheet)
        head = {k: v for k, v in whole.items() if k != "labels"}
    d = whole["labels"]
    print("  %s: %d rows" % (os.path.basename(os.path.dirname(p)), len(d)))
    for k, v in d.items():
        seen[k] += 1
        merged[k] = v
dupes = [k for k, c in seen.items() if c > 1]
if dupes:
    sys.exit("keys written by more than one apply process: %s" % dupes[:10])

want = [l.strip() for l in open(a.items) if l.strip() and not l.startswith("#")]
missing = [k for k in want if k not in merged]
extra = [k for k in merged if k not in want]
print("merged rows %d; items %d; missing %d; extra %d" % (len(merged), len(want), len(missing), len(extra)))
bad = []
for k in want:
    if k not in merged:
        continue
    r = json.load(open(os.path.join(a.round, "v_resolved", k.replace("/", "_") + ".json")))
    L = merged[k]
    if (L.get("choice") != r["verdict"] or L.get("michel_kind") != r["michel_kind"]
            or L.get("pf_tagged") != len(r["tags"])):
        bad.append("%s: label %s/%s/%s vs record %s/%s/%d" % (
            k, L.get("choice"), L.get("michel_kind"), L.get("pf_tagged"),
            r["verdict"], r["michel_kind"], len(r["tags"])))
for b in bad[:20]:
    print("  !! " + b)
if missing or extra or bad:
    sys.exit("refusing: %d missing, %d extra, %d disagreeing" % (len(missing), len(extra), len(bad)))
print("every row agrees with its resolved record")

if a.check:
    print("--check only, nothing written")
    raise SystemExit(0)

os.makedirs(os.path.dirname(LIVE), exist_ok=True)
tmp = LIVE + ".merge.tmp"
out = dict(head or {})
out["tag"] = a.tag           # the apply processes ran under a throwaway tag
out["labels"] = merged
json.dump(out, open(tmp, "w"), indent=1)
os.replace(tmp, LIVE)
sha = hashlib.sha256(open(LIVE, "rb").read()).hexdigest()
open(SHAF, "w").write(sha + "  " + LIVE + "\n")
print("WROTE %s\n  rows %d\n  sha  %s  (recorded in %s)" % (LIVE, len(merged), sha, SHAF))
