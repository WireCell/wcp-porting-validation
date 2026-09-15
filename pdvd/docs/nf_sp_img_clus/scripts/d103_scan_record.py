#!/usr/bin/env python3
"""doc pdvd/103 sec 5 -- turn the blind agent scan of one detector into a new verdicts record.  Records-only route
(the doc pdvd/99 sw99 / doc pdhd/25 smx25 precedent: no labels.json tag; the record is the artifact).

Fork by duplication of d99_swap_scan_score.py (untouched), reduced to what doc 103 needs:
  * every mkv.py record under ROUND/v_parts/*/ (a duplicate is reported; the earliest `written` is kept, and the
    later one is kept aside as a double scan);
  * the private item list ITEMS (key, display_arm, role new|calibration) -- never shown to a scanner;
  * V2 of figs/103_pred.txt: stopper-or-not agreement of the calibration items with the existing record (smx22 with
    the owner precedence on PDHD; the merged smx1a..smx9 record on PDVD), with the > 25 % stop rule;
  * the new record (refuses to overwrite): each mkv.py record plus display_arm, calibration (bool), tag, source_round.

Usage: d103_scan_record.py --det pdhd --round /home/xqian/tmp/d103/round_pdhd --items /home/xqian/tmp/d103/items/pdhd_items.tsv
          --tag smx28 [--record-out $IMG/pdhd/docs/scan/pdhd_stm_michel_smx28_verdicts.json]
"""
import argparse, collections, csv, glob, json, os, sys

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
STOP = {"STM_MICHEL", "STM_ONLY", "FRAG_STM_MICHEL", "FRAG_STM_ONLY"}
NON = {"THRU", "FRAG_THRU"}


def cls(v):
    return "stop" if v in STOP else "non" if v in NON else "excl"


def strip(v):
    return v[5:] if v and v.startswith("FRAG_") else v


def old_truth(det):
    if det == "pdhd":
        out = {}
        for r in json.load(open(f"{IMG}/pdhd/docs/scan/pdhd_stm_michel_smx22_verdicts.json")):
            if r.get("owner_review"):
                v = r["owner_review"]["verdict"]
            elif r.get("owner_smx1"):
                v = r["owner_smx1"].get("choice") or r["owner_smx1"].get("label")
            else:
                v = r["verdict"]
            out[r["key"]] = (v, r.get("confidence", ""))
        return out
    rec = json.load(open(f"{IMG}/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json"))
    return {r["key"]: (r["verdict"], r.get("confidence", "")) for r in rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--round", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--record-out", default=None)
    a = ap.parse_args()
    items = {r["key"]: r for r in csv.DictReader((l for l in open(a.items) if not l.startswith("#")), delimiter="\t")}
    recs, doubles = {}, []
    for f in sorted(glob.glob(os.path.join(a.round, "v_parts", "*", "*.json"))):
        r = json.load(open(f))
        if r["key"] in recs:
            first, second = sorted([recs[r["key"]], r], key=lambda x: x["written"])
            recs[r["key"]] = first
            doubles.append(second)
            continue
        recs[r["key"]] = r
    print(f"# doc pdvd/103 blind scan record ({a.det}, tag {a.tag}); round {a.round}")
    missing = sorted(k for k in items if k not in recs)
    extra = sorted(k for k in recs if k not in items)
    print(f"items {len(items)}; records {len(recs)}; missing {len(missing)} {missing}; not in the item list {extra}")
    print(f"double scans {len(doubles)}: " + ", ".join(f"{d['key']} {d['scanner']} {d['verdict']} vs {recs[d['key']]['verdict']}"
                                                     for d in doubles))
    print(f"rubric sha on the records: {dict(collections.Counter(r.get('rubric_sha') for r in recs.values()))}")
    print("verdicts (new items): " + str(dict(collections.Counter(recs[k]["verdict"] for k in recs
                                                                 if items.get(k, {}).get("role") == "new"))))
    print("confidence (new items): " + str(dict(collections.Counter(recs[k]["confidence"] for k in recs
                                                                   if items.get(k, {}).get("role") == "new"))))

    old = old_truth(a.det)
    cal = [k for k, r in items.items() if r["role"] == "calibration" and k in recs]
    judged = [k for k in cal if cls(recs[k]["verdict"]) != "excl" and cls(strip(old[k][0])) != "excl"]
    dis = [k for k in judged if cls(recs[k]["verdict"]) != cls(old[k][0])]
    frac = len(dis) / max(1, len(judged))
    print(f"\nV2 calibration: {len(cal)} scanned, {len(judged)} judged on both sides; stopper-or-not disagreements {len(dis)} "
          f"({frac:.0%}) -> {'FAIL (> 25 %): labels not usable for grading' if frac > 0.25 else 'PASS'}")
    for k in dis:
        print(f"   {k}: existing {old[k][0]} ({old[k][1]}) vs blind {recs[k]['verdict']} ({recs[k]['confidence']})")
    for k in cal:
        if k not in judged:
            print(f"   {k}: not judged on one side: existing {old[k][0]} vs blind {recs[k]['verdict']}")

    if a.record_out:
        if os.path.exists(a.record_out):
            sys.exit(f"REFUSING: {a.record_out} exists")
        out = []
        for k in sorted(recs):
            if k not in items:
                continue
            v = dict(recs[k])
            v.update(display_arm=items[k]["display_arm"], calibration=items[k]["role"] == "calibration", tag=a.tag,
                     source_round="doc pdvd/103 round 1 verdict-blind agent scan")
            out.append(v)
        json.dump(out, open(a.record_out, "w"), indent=1, ensure_ascii=False)
        print(f"\nwrote {a.record_out} ({len(out)} records)")


if __name__ == "__main__":
    main()
