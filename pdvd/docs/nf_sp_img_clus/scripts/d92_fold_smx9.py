#!/usr/bin/env python3
"""doc pdvd/92 -- fold the owner's smx9 blind tie-break into a NEW merged record (M13: the smx8
record is never rewritten), and report the three-way verdict on the contested item.

smx9 asked one question: 039349_71/37, which the owner judged STM_MICHEL in smx7 and THRU in smx8,
served blind a third time with 6 fresh controls.  The fold is PROVENANCE, not a re-grade: the third
call agreed with smx8, so no stopper/Michel/judged class moves and every census in doc 92 §6 stands
unchanged.  The script asserts exactly that and fails loudly if it is not true.

Merge rules are d92_score_smx8.py's owner_row, with source smx9.

Fork by duplication (CLAUDE.md M10) of d92_score_smx8.py's fold section.

Usage: STM_SCAN_RECORD=<smx1a..smx8 record> d92_fold_smx9.py --labels L.json --key K.tsv
           [--write-merged OUT]
"""
import argparse, csv, json, os, sys
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--write-merged", default=None)
a = ap.parse_args()
if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_smx5_smx6_smx7_smx8_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the merged smx1a..smx8 record (doc 92)")
if a.write_merged and os.path.exists(a.write_merged):
    sys.exit("REFUSING: %s exists" % a.write_merged)

rec0 = C.load_record()
L = json.load(open(a.labels))["labels"]
K = {r["key"]: r for r in csv.DictReader([l for l in open(a.key) if not l.startswith("#")], delimiter="\t")}
assert set(L) == set(K), "labels and key differ: %s" % sorted(set(L) ^ set(K))

cls = lambda v: "THRU" if C.bare(v) == "THRU" else ("STM" if C.bare(v) in ("STM_MICHEL", "STM_ONLY") else v)

print("=== 1. the three-way verdict on the contested item ===")
for k in sorted(K):
    if K[k]["group"] != "tiebreak":
        continue
    votes = [K[k]["prior_verdict"], K[k]["record_verdict"], L[k]["choice"]]
    stop = sum(1 for v in votes if cls(v) == "STM"); thru = sum(1 for v in votes if cls(v) == "THRU")
    pin = (L[k].get("pin") or {})
    print("  %-14s smx7 %-11s smx8 %-11s smx9 %-11s -> stopper %d / THRU %d%s" % (
        k, votes[0], votes[1], votes[2], stop, thru,
        ("; smx9 pin rr %.1f" % pin["rr"]) if pin.get("placed") and pin.get("rr") is not None else ""))
    print("  majority: %s" % ("STOPPER" if stop > thru else "THRU" if thru > stop else "SPLIT"))

ctl = sorted(k for k in K if K[k]["group"] == "control")
held = [k for k in ctl if cls(K[k]["record_verdict"]) == cls(L[k]["choice"])]
print("\n=== 2. the controls (the null) ===")
print("  %d of %d held the stopper-vs-THRU call" % (len(held), len(ctl)))
for k in ctl:
    if k not in held:
        print("   OVERTURNED %-14s %s -> %s" % (k, K[k]["record_verdict"], L[k]["choice"]))


def owner_row(k, v):
    old = rec0[k]
    tags = dict(old.get("tags") or {})
    tags.update(v.get("pf_segments") or {})
    pin = v.get("pin")
    kept = False
    if not (pin or {}).get("placed") and (old.get("pin") or {}).get("placed"):
        pin, kept = old["pin"], True
    row = dict(key=k, verdict=v["choice"], michel_kind=v.get("michel_kind"), confidence="owner",
               tranche=old.get("tranche"), scan_id=old.get("scan_id"), evidence=(v.get("notes") or "").strip(),
               tags=tags, source="smx9 (owner, 2026-09-11)", pin=pin)
    if old.get("pin_rr") is not None:
        row["pin_rr"] = old["pin_rr"]
        row["pin_rr_source"] = old.get("source", "smx1a")
    return row, kept


print("\n=== 3. the fold ===")
rec1 = dict(rec0)
for k in sorted(L):
    row, kept = owner_row(k, L[k])
    old = rec0[k]
    diff = ["%s %s -> %s" % (f, old.get(f), row.get(f)) for f in ("verdict", "michel_kind", "confidence") if old.get(f) != row.get(f)]
    if diff or kept:
        print("  %-14s %s%s" % (k, "; ".join(diff), "  [the old placed pin kept]" if kept else ""))
    rec1[k] = row
moved = [k for k in rec0 if (C.is_stopper(rec1[k]), C.is_michel(rec1[k]), C.judged(rec1[k])) !=
         (C.is_stopper(rec0[k]), C.is_michel(rec0[k]), C.judged(rec0[k]))]
print("  stopper / Michel / judged class changed on %d of %d records: %s" % (len(moved), len(rec0), " ".join(moved) or "none"))
js0 = sum(1 for k in rec0 if C.judged(rec0[k]) and C.is_stopper(rec0[k]))
js1 = sum(1 for k in rec1 if C.judged(rec1[k]) and C.is_stopper(rec1[k]))
print("  judged stoppers: %d -> %d" % (js0, js1))
if moved or js0 != js1:
    sys.exit("UNEXPECTED: smx9 was a provenance fold; doc 92 sec 6's censuses would have to be redone")
print("  => every census in doc 92 sec 6 stands unchanged (this fold records the third call, it does not re-grade)")

if a.write_merged:
    with open(a.write_merged, "w") as fh:
        json.dump([rec1[k] for k in sorted(rec1)], fh, indent=1)
    print("\nwrote %d merged records -> %s (%d from smx9)" % (len(rec1), a.write_merged, len(L)))
