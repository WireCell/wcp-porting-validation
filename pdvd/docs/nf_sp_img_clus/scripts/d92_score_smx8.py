#!/usr/bin/env python3
"""doc pdvd/92 -- score the owner's smx8 blind re-judge against its key, fold it into a NEW merged
record (the old one is never rewritten, M13), and re-grade production and the two operating points
on both records.

Fork by duplication (CLAUDE.md M10) of d90_score_smx7.py; the tag, the record names, the group names
and the grading source changed.  d90's script stays as doc 90's record.

smx8 = the 11-item DECISION SET of the wide Bragg-peak read (every candidate that any of the four
candidate operating points would turn into a stopper) plus 8 controls that no grid point moves.
Arms are not needed here: the twin is exact (585/585), so the movers come from d92_twin.py.

WHAT THIS MAY AND MAY NOT CONCLUDE.  The tranche was chosen BY the quantity being measured --
distance to the R / D boundary.  So:
  * folding the owner's verdicts into the record is legitimate (the record is the ground truth, and
    smx7 did exactly this to smx1a calls);
  * RE-SELECTING W / R / D on the corrected record would be circular, and this script does not do
    it.  The operating point stays doc 91's pre-registered W 8 / R 1.3 / D 0.8.
  * the corrected census is NOT comparable to a record-wide re-judge: only boundary items were
    re-examined, so corrections concentrate exactly where the rule acts.  The 8 controls are the
    only null this round has, and section 1 reports how many of them held.

Merge rules (smx4's owner_row, doc 70, as in smx5 / smx6 / smx7):
  * verdict / michel_kind / confidence "owner" / evidence = notes / source smx8;
  * tags: the record's, with any pf_segments override on top;
  * pin: the smx8 pin, EXCEPT that an unplaced smx8 pin never replaces a placed one; a scanner
    pin_rr on the old row is carried with pin_rr_source.

Usage: STM_SCAN_RECORD=<smx1a..smx7 record> d92_score_smx8.py --labels L.json --key K.tsv
           --prep PREP --twin TWIN.json [--write-merged OUT]
"""
import argparse, collections, csv, json, os, sys
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--prep", required=True, help="the production prep (p90vprod)")
ap.add_argument("--twin", required=True, help="d92_twin.py's JSON: the exact movers per operating point")
ap.add_argument("--write-merged", default=None)
a = ap.parse_args()
if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_smx5_smx6_smx7_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the smx1a..smx7 record (doc 90)")
if a.write_merged and os.path.exists(a.write_merged):
    sys.exit("REFUSING: %s exists" % a.write_merged)

rec0 = C.load_record()
L = json.load(open(a.labels))["labels"]
K = {r["key"]: r for r in csv.DictReader([l for l in open(a.key) if not l.startswith("#")], delimiter="\t")}
assert set(L) == set(K), "labels and key differ: %s" % sorted(set(L) ^ set(K))
P = {}
for f in sorted(os.listdir(a.prep)):
    if f.startswith("smprep-") and f.endswith(".json"):
        P[f[7:-5].replace("-c", "/")] = json.load(open(os.path.join(a.prep, f)))["verdict"]
TW = json.load(open(a.twin))

cls = lambda v: "THRU" if C.bare(v) == "THRU" else ("STM" if C.bare(v) in ("STM_MICHEL", "STM_ONLY") else v)

print("=== 1. the owner's smx8 call against the record, by group ===")
for grp in ("decision", "control"):
    ks = sorted((k for k in K if K[k]["group"] == grp), key=lambda t: int(K[t]["scan_id"]))
    held = [k for k in ks if cls(K[k]["record_verdict"]) == cls(L[k]["choice"])]
    over = [k for k in ks if cls(K[k]["record_verdict"]) != cls(L[k]["choice"])]
    print("  -- %-8s (%2d): stopper-vs-THRU call held on %d, overturned on %d" % (grp, len(ks), len(held), len(over)))
    for k in over:
        print("       OVERTURNED %-14s %s/%s -> %s" % (k, K[k]["record_verdict"], K[k]["record_confidence"], L[k]["choice"]))
print("  the controls are this round's only null: a high overturn rate there would mean the re-judge")
print("  itself moved, not the boundary items.")

print("\n=== 2. every item ===")
print("  %-2s %-14s %-8s %-18s %-12s %-16s %s" % ("id", "item", "group", "record", "owner smx8", "owner pin", "moved at"))
for k in sorted(K, key=lambda t: int(K[t]["scan_id"])):
    v, kr = L[k], K[k]
    pin = v.get("pin") or {}
    ps = "rr %.1f (moved %.1f)" % (pin["rr"], pin.get("moved_cm", 0)) if pin.get("placed") else "fit end"
    print("  %-2s %-14s %-8s %-18s %-12s %-16s %s" % (
        kr["scan_id"], k, kr["group"], "%s/%s" % (kr["record_verdict"], kr["record_confidence"]),
        v["choice"], ps, kr["moved_at"] or "-"))


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
               tags=tags, source="smx8 (owner, 2026-09-11)", pin=pin)
    if old.get("pin_rr") is not None:
        row["pin_rr"] = old["pin_rr"]
        row["pin_rr_source"] = old.get("source", "smx1a")
    return row, kept


print("\n=== 3. the fold, field by field ===")
rec1 = dict(rec0)
for k in sorted(L, key=lambda t: int(K[t]["scan_id"])):
    row, kept = owner_row(k, L[k])
    old = rec0[k]
    diff = ["%s %s -> %s" % (f, old.get(f), row.get(f)) for f in ("verdict", "michel_kind", "confidence") if old.get(f) != row.get(f)]
    if diff or kept:
        print("  %-14s %s%s" % (k, "; ".join(diff), "  [the old placed pin kept]" if kept else ""))
    rec1[k] = row
moved = [k for k in rec0 if (C.is_stopper(rec1[k]), C.is_michel(rec1[k]), C.judged(rec1[k])) !=
         (C.is_stopper(rec0[k]), C.is_michel(rec0[k]), C.judged(rec0[k]))]
print("  stopper / Michel / judged class changed on %d of %d records: %s" % (len(moved), len(rec0), " ".join(moved)))
print("  judged stoppers: smx7 record %d -> smx8 record %d" % (
    sum(1 for k in rec0 if C.judged(rec0[k]) and C.is_stopper(rec0[k])),
    sum(1 for k in rec1 if C.judged(rec1[k]) and C.is_stopper(rec1[k]))))


def census(R, movers):
    J = [k for k in R if C.judged(R[k])]
    stm = lambda k: (k in P and int(P[k]["is_stm"]) == 1) or k in movers
    tp = sum(1 for k in J if stm(k) and C.is_stopper(R[k])); fp = sum(1 for k in J if stm(k) and not C.is_stopper(R[k]))
    fn = sum(1 for k in J if not stm(k) and C.is_stopper(R[k]))
    pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn, 1)
    return "%3d / %2d / %3d  eff %.3f pur %.3f" % (tp, fp, fn, ef, pu)


print("\n=== 4. is_stm (TP / FP / FN) before and after the fold ===")
POINTS = [("production", set())] + [(k, set(TW["arms"][k]["movers"])) for k in sorted(TW["arms"])]
for lab, mov in POINTS:
    print("  %-11s | smx7 record %s | smx8 record %s" % (lab, census(rec0, mov), census(rec1, mov)))

print("\n=== 5. what the rule gains on the CORRECTED record, item by item ===")
for lab, mov in POINTS[1:]:
    g = sorted(mov)
    st = [k for k in g if C.is_stopper(rec1[k])]; th = [k for k in g if C.judged(rec1[k]) and not C.is_stopper(rec1[k])]
    un = [k for k in g if not C.judged(rec1[k])]
    print("  %-8s (W %.1f R %.1f D %.1f): +%d stopper(s) %s | +%d THRU %s | unjudged %d" % (
        lab, TW["arms"][lab]["W"], TW["arms"][lab]["R"], TW["arms"][lab]["D"],
        len(st), " ".join(st), len(th), " ".join(th) or "-", len(un)))
print("\n  The operating point is NOT re-selected here: doc 91 pre-registered W 8 / R 1.3 / D 0.8 and")
print("  this tranche was drawn from the boundary, so re-choosing on it would be circular.")

if a.write_merged:
    with open(a.write_merged, "w") as fh:
        json.dump([rec1[k] for k in sorted(rec1)], fh, indent=1)
    print("\nwrote %d merged records -> %s (%d from smx8)" % (len(rec1), a.write_merged, len(L)))
