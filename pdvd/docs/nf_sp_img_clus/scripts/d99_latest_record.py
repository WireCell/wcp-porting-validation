#!/usr/bin/env python3
"""doc pdvd/99 sec 6.4 / doc pdvd/98 sec 11 -- the hand record of the latest configuration: the smx record carried onto
p98vonq (sec 5) plus the blind swap scan's verdicts on the clusters p98vonq tags that production does not (sw99).

    python3 d99_latest_record.py --carried ../../scan/pdvd_stm_michel_smx9_carried_p98vonq.json \
        --sw99 ../../scan/pdvd_stm_michel_sw99_verdicts.json --out ../../scan/pdvd_stm_michel_p98vonq_carried_sw99_verdicts.json

A key present in both keeps the carried verdict (the record's scanner saw the object first; the blind call is the
calibration, sec 6.4 item 6) and is counted.  Only sw99 items displayed on p98vonq are added: the prod_only controls are
p96vprod keys and do not name p98vonq clusters.  Never writes over an existing file.
"""
import argparse, collections, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--carried", required=True)
ap.add_argument("--sw99", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if os.path.exists(a.out):
    sys.exit(f"REFUSING: {a.out} exists")
car = json.load(open(a.carried))
keys = {r["key"] for r in car}
out = [dict(r, latest_source="carried") for r in car]
added, both = [], []
for r in json.load(open(a.sw99)):
    if r.get("display_arm") != "p98vonq":
        continue
    if r["key"] in keys:
        both.append(r["key"])
        continue
    out.append(dict(key=r["key"], verdict=r["verdict"], michel_kind=r["michel_kind"], confidence=r["confidence"],
                    source="sw99", tags=r.get("tags", {}), pin_rr=r.get("pin_rr"), notes=r.get("notes", ""),
                    evidence=r.get("evidence", ""), latest_source="sw99"))
    added.append(r)
json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
print(f"carried {len(car)} + sw99 added {len(added)} (already carried, kept carried: {len(both)}) -> {len(out)} -> {a.out}")
print("added verdicts:", dict(collections.Counter(r["verdict"] for r in added)))
print("added STM_MICHEL kinds:", dict(collections.Counter(r["michel_kind"] for r in added if r["verdict"] == "STM_MICHEL")))
