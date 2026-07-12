#!/usr/bin/env python3
"""Merge crossers + boundary decisions into a combined 'candles' decisions JSONL
per event: union of all keep/add lines from both finders (dedup by bundle_idx),
plus a reject for every auto bundle selected by neither.

Produces one ql_scan tag (`candles`) that shows BOTH cathode-crosser pairs and
anode boundary tracks together, so a single viewer/port covers all the PDVD
standard candles. Run from pdvd/ql_display/ after find_crossers.py --emit and
find_boundary.py --emit have populated decisions-crossers/ and decisions-boundary/:

  python3 merge_candles.py
  for f in ../work/039252_*/calib-evt*.json; do
    id=$(basename $f | sed 's/calib-evt//;s/.json//')
    python3 make_labels.py $f decisions-candles/decisions-evt$id.jsonl --tag candles
  done
  ../ql_scan/serve_ql_scan.sh 5017 --tag candles \
    ../work/039252_{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}/calib-evt*.json

A cluster that is a crosser half at one flash and a boundary track's best-T0 at a
neighbouring flash is selected on BOTH (make_labels prints a non-fatal WARNING) --
that surfaces where the two candle definitions disagree on T0 by a flash or two."""
import json, os, sys, glob

XDIR = "decisions-crossers"
BDIR = "decisions-boundary"
ODIR = "decisions-candles"
os.makedirs(ODIR, exist_ok=True)

CAL = {}
for f in glob.glob("../work/039252_*/calib-evt*.json"):
    if "prextpc" in f or "light" in f:
        continue
    ev = os.path.basename(f)[len("calib-"):-len(".json")]
    CAL[ev] = f

def read(path):
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out

n_events = 0
for ev, cal in sorted(CAL.items()):
    xs = read(os.path.join(XDIR, "decisions-%s.jsonl" % ev))
    bs = read(os.path.join(BDIR, "decisions-%s.jsonl" % ev))
    d = json.load(open(cal))
    autos = {j for j, b in enumerate(d["bundles"]) if b["auto_selected"]}

    keepadd = {}   # bundle_idx -> line (keep/add), union of both finders
    for r in xs + bs:
        if r["verdict"] in ("keep", "add"):
            j = r["bundle_idx"]
            if j not in keepadd:
                keepadd[j] = r
    # rejects: every auto bundle not selected by either finder
    lines = list(keepadd.values())
    fby = {f["gid"]: f for f in d["flashes"]}
    for gi, f in enumerate(sorted(d["flashes"], key=lambda f: f["time"]), 1):
        f["group"] = gi
    cby = {c["uid"]: c for c in d["clusters"]}
    for j in sorted(autos):
        if j in keepadd:
            continue
        b = d["bundles"][j]
        uid = b["main_cluster"]
        lines.append({
            "event": ev, "group": fby[b["flash_gid"]]["group"],
            "flash_gid": b["flash_gid"], "flash_time_us": fby[b["flash_gid"]]["time"],
            "apa": b["apa"], "main_cluster_uid": uid,
            "main_cluster_ident": cby.get(uid, {}).get("ident"),
            "bundle_idx": j, "verdict": "reject",
            "auto_selected": True, "confidence": "high",
            "reason": "neither a cathode-crosser half nor an anode boundary track.",
        })
    with open(os.path.join(ODIR, "decisions-%s.jsonl" % ev), "w") as fh:
        for r in sorted(lines, key=lambda r: r["bundle_idx"]):
            fh.write(json.dumps(r) + "\n")
    nx = sum(1 for r in xs if r["verdict"] in ("keep", "add"))
    nb = sum(1 for r in bs if r["verdict"] in ("keep", "add"))
    print("%s: crossers=%d boundary=%d union=%d" % (ev, nx, nb, len(keepadd)))
    n_events += 1
print("events merged:", n_events)
