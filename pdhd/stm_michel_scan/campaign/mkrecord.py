#!/usr/bin/env python3
"""Build the committed scan record from the resolved verdicts.

    mkrecord.py ROUND SHEET OWNER_LABELS OUT_RECORD OUT_PROVENANCE [--calib-items F]

doc pdhd/18.  Ported from the doc pdvd/55 tranche-2 scratch mkrecord.py.

OUT_RECORD is a LIST, one record per item, in the shape
verify_scan_record.py already reads (the PDVD record's shape), with added fields:
  arm, rubric_sha, scanner, wave     -- which dump, which rubric text, who, when:
                                        a label file never names its arm, and that
                                        nearly cost the PDVD scan its basis in a
                                        cleanup round (feedback_scan_labels_need_
                                        their_source_arm), so every record does;
  owner_smx1                         -- on the items the owner labelled: the
                                        owner's choice / michel_kind / pin / notes,
                                        which is the GRADING truth for those items
                                        (the agent record there is the blind
                                        calibration, doc pdhd/18 sec 5);
  calibration: true                  -- on those same items.
OUT_PROVENANCE is the round header: arms, TLAs, pin, rubric shas, gates.
"""
import argparse, csv, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("round"); ap.add_argument("sheet"); ap.add_argument("owner")
ap.add_argument("out_record"); ap.add_argument("out_prov")
ap.add_argument("--calib-items", default=None)
ap.add_argument("--arm", default="h18s")
ap.add_argument("--provenance-json", default=None,
                help="a JSON object merged into OUT_PROVENANCE (arms, pins, gates)")
a = ap.parse_args()

rows = list(csv.DictReader([l for l in open(a.sheet) if not l.startswith("#")], delimiter="\t"))
owner = json.load(open(a.owner))["labels"]
calib = set(l.strip() for l in open(a.calib_items)) if a.calib_items else set()

out, missing = [], []
for r in sorted(rows, key=lambda r: (int(r["tranche"]), int(r["scan_id"]))):
    k = "%s/%s" % (r["event"], r["cluster"])
    p = os.path.join(a.round, "v_resolved", k.replace("/", "_") + ".json")
    if not os.path.exists(p):
        missing.append(k)
        continue
    v = json.load(open(p))
    rec = dict(key=k, scan_id=int(r["scan_id"]), tranche=int(r["tranche"]),
               verdict=v["verdict"], michel_kind=v["michel_kind"],
               confidence=v["confidence"])
    if v.get("pin_rr") is not None:
        rec["pin_rr"] = v["pin_rr"]
    if v.get("notes"):
        rec["notes"] = v["notes"]
    rec.update(evidence=v["evidence"], tags=v["tags"], arm=a.arm,
               rubric_sha=v.get("rubric_sha"), scanner=v.get("scanner"),
               wave=v.get("wave"), written=v.get("written"))
    if k in owner:
        o = owner[k]
        pin = o.get("pin") or {}
        rec["owner_smx1"] = dict(
            choice=o.get("choice"), label=o.get("label"), partial=o.get("partial"),
            michel_kind=o.get("michel_kind"), notes=o.get("notes"),
            pin_placed=bool(pin.get("placed")), pin_rr=pin.get("rr") if pin.get("placed") else None,
            arm="d53h")
    if k in calib:
        rec["calibration"] = True
    out.append(rec)
if missing:
    sys.exit("%d sheet item(s) have no resolved record: %s" % (len(missing), missing[:10]))

with open(a.out_record + ".tmp", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
os.replace(a.out_record + ".tmp", a.out_record)
prov = json.load(open(a.provenance_json)) if a.provenance_json else {}
prov.update(n_records=len(out), n_owner=sum(1 for r in out if "owner_smx1" in r),
            rubric_shas=sorted({r["rubric_sha"] for r in out if r.get("rubric_sha")}))
json.dump(prov, open(a.out_prov, "w"), indent=1, ensure_ascii=False)
print("wrote %s (%d records, %d with owner_smx1) and %s"
      % (a.out_record, len(out), prov["n_owner"], a.out_prov))
