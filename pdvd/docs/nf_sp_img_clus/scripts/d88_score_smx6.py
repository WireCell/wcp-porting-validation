#!/usr/bin/env python3
"""doc pdvd/88 sec 9 -- score the owner's smx6 answer against the key, and fold it into
a NEW merged record (the old one is never rewritten, M13).

Fork by duplication (CLAUDE.md M10) of d86_score_smx5.py; the tag, the record names and
the MECH mapping changed.

The owner answered in free text (the notes field, plus the message that returned the
scan), not in the panel's `MECH: ...; DEAD: ...;` line.  MECH below is my mapping of
those words onto the vocabulary, quoted beside it so it can be checked; DEAD was not
answered and is not scored.

Merge rules (smx4's owner_row, doc 70, plus one belt):
  * verdict / michel_kind / confidence "owner" / evidence = notes / source smx5;
  * tags: the record's, with any pf_segments override on top (none were set);
  * pin: the smx5 pin, EXCEPT that an unplaced smx5 pin never replaces a placed one
    (stm_michel_viewer's own no-downgrade rule); a scanner pin_rr on the old row is
    carried with pin_rr_source.

Usage: STM_SCAN_RECORD=<smx1a+smx3+smx4+smx5 record> d88_score_smx6.py --labels L.json --key K.tsv
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
if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_smx5_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the smx1a+smx3+smx4+smx5 record (doc 86)")
if a.write_merged and os.path.exists(a.write_merged):
    sys.exit("REFUSING: %s exists" % a.write_merged)

rec0 = C.load_record()
L = json.load(open(a.labels))["labels"]
K = {r["key"]: r for r in csv.DictReader([l for l in open(a.key) if not l.startswith("#")], delimiter="\t")}

# the owner's words -> MECH (the smx6 notes, 2026-09-11: "Yes, there are some clear image
# results near the end of the STM. They were not identified as Michel, so no track
# trajectory fit, but clear in image.")  -> imaged, no fitted segment: `unfitted`.
MECH = {"039253_15/36": "unfitted"}

print("=== 1. MECH, verdict and Michel kind against the key ===")
print("%-13s %-10s %-10s %-11s %-11s %-9s %s" % ("item", "verdict", "record", "pred MECH", "owner MECH", "kind", "notes"))
agree = 0
for k in sorted(K, key=lambda t: int(K[t]["scan_id"])):
    v, kr = L[k], K[k]
    ok = MECH[k] == kr["pred_mech"]
    agree += ok
    print("%-13s %-10s %-10s %-11s %-11s %-9s \"%s\"" % (k, v["choice"], kr["record_verdict"], kr["pred_mech"],
                                                         MECH[k] + ("" if ok else " (!)"), v.get("michel_kind"),
                                                         (v.get("notes") or "").strip()))
print("MECH agrees with the prediction on %d of %d; DEAD not answered (the free-text notes carry no DEAD line)" % (agree, len(K)))
print("verdict: %d of %d STM_MICHEL (record: %d of %d)" % (
    sum(L[k]["choice"] == "STM_MICHEL" for k in K), len(K), sum(K[k]["record_verdict"] == "STM_MICHEL" for k in K), len(K)))


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
               tags=tags, source="smx6 (owner, 2026-09-11)", pin=pin, mech=MECH[k])
    if old.get("pin_rr") is not None:
        row["pin_rr"] = old["pin_rr"]
        row["pin_rr_source"] = old.get("source", "smx1a")
    return row, kept


print("\n=== 2. what the fold changes, field by field ===")
rec1 = dict(rec0)
for k, v in L.items():
    row, kept = owner_row(k, v)
    old = rec0[k]
    diff = ["%s %s -> %s" % (f, old.get(f), row.get(f)) for f in ("verdict", "michel_kind", "confidence")
            if old.get(f) != row.get(f)]
    print("  %-13s %s%s" % (k, "; ".join(diff) or "no field change",
                           "; pin: the placed %s pin (rr %.1f) kept over an unplaced smx6 pin" % (
                               old.get("source"), old["pin"]["rr"]) if kept else ""))
    rec1[k] = row
same = all(C.is_stopper(rec1[k]) == C.is_stopper(rec0[k]) and C.is_michel(rec1[k]) == C.is_michel(rec0[k])
           and C.judged(rec1[k]) == C.judged(rec0[k]) for k in rec0)
print("  stopper / Michel / judged class identical on all %d records: %s" % (len(rec0), same))
for k in L:
    print("  %s: stopper %s -> %s, Michel %s -> %s" % (k, C.is_stopper(rec0[k]), C.is_stopper(rec1[k]),
                                                      C.is_michel(rec0[k]), C.is_michel(rec1[k])))
if a.write_merged:
    with open(a.write_merged, "w") as fh:
        json.dump([rec1[k] for k in sorted(rec1)], fh, indent=1)
    print("\nwrote %d merged records -> %s (%d from smx6)" % (len(rec1), a.write_merged, len(L)))
