#!/usr/bin/env python3
"""The committed record after a review round (doc pdhd/19).

    mkreview_record.py ROUND OLD_RECORD OUT_RECORD OUT_PROVENANCE [--provenance-json F]

One record per item of OLD_RECORD, in its order:
  * an item the review round resolved (ROUND/v_resolved/<key>.json) gets a NEW
    record in mkrecord.py's shape, built from the resolved record (the
    adjudication where there was one, the blind review otherwise), plus a
    `review` block naming what it replaced and how it was reached:
        review.previous   the OLD record's verdict / kind / confidence / rubric
                          sha / scanner / wave
        review.blind      the blind re-scan's verdict / kind / confidence /
                          scanner / notes (kept even when an adjudication
                          superseded it)
        review.adjudicated  true when the final call is an adjudication
    `owner_smx1` and `calibration` are carried over unchanged: on the owner's
    items the owner's label stays the grading truth (doc pdhd/18 sec 5);
  * every other item is copied from OLD_RECORD verbatim.
OLD_RECORD itself is never written (M13).
"""
import argparse, glob, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("round"); ap.add_argument("old")
ap.add_argument("out_record"); ap.add_argument("out_prov")
ap.add_argument("--provenance-json", default=None)
ap.add_argument("--arm", default="h18s")
a = ap.parse_args()
if os.path.abspath(a.out_record) == os.path.abspath(a.old):
    sys.exit("refusing to write over the old record (M13)")

old = json.load(open(a.old))
blind = {}
for f in glob.glob(os.path.join(a.round, "v_parts", "rv*_a*", "*.json")):
    r = json.load(open(f))
    if r["key"] in blind:
        sys.exit("two blind review records for %s" % r["key"])
    blind[r["key"]] = r

out, n_new, n_adj = [], 0, 0
for o in old:
    k = o["key"]
    p = os.path.join(a.round, "v_resolved", k.replace("/", "_") + ".json")
    if not os.path.exists(p):
        out.append(o)
        continue
    v = json.load(open(p))
    rec = dict(key=k, scan_id=o["scan_id"], tranche=o["tranche"],
               verdict=v["verdict"], michel_kind=v["michel_kind"],
               confidence=v["confidence"])
    if v.get("pin_rr") is not None:
        rec["pin_rr"] = v["pin_rr"]
    if v.get("notes"):
        rec["notes"] = v["notes"]
    rec.update(evidence=v["evidence"], tags=v["tags"], arm=a.arm,
               rubric_sha=v.get("rubric_sha"), scanner=v.get("scanner"),
               wave=v.get("wave"), written=v.get("written"))
    b = blind.get(k)
    if b is None:
        sys.exit("%s is resolved but has no blind review record" % k)
    adj = (v.get("scanner") or "").startswith("adj")
    rec["review"] = dict(
        previous={f: o.get(f) for f in ("verdict", "michel_kind", "confidence",
                                        "rubric_sha", "scanner", "wave")},
        blind={f: b.get(f) for f in ("verdict", "michel_kind", "confidence",
                                     "scanner", "notes")},
        adjudicated=adj)
    for f in ("owner_smx1", "calibration"):
        if f in o:
            rec[f] = o[f]
    out.append(rec)
    n_new += 1; n_adj += adj

with open(a.out_record + ".tmp", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
os.replace(a.out_record + ".tmp", a.out_record)
prov = json.load(open(a.provenance_json)) if a.provenance_json else {}
prov.update(n_records=len(out), n_reviewed=n_new, n_adjudicated=n_adj,
            n_owner=sum(1 for r in out if "owner_smx1" in r),
            rubric_shas=sorted({r["rubric_sha"] for r in out if r.get("rubric_sha")}))
json.dump(prov, open(a.out_prov, "w"), indent=1, ensure_ascii=False)
print("wrote %s (%d records: %d reviewed, %d of them adjudicated; %d carried verbatim) and %s"
      % (a.out_record, len(out), n_new, n_adj, len(out) - n_new, a.out_prov))
