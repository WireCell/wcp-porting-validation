#!/usr/bin/env python3
"""The committed record after the owner's rulings and the v5 double scan (doc pdhd/19 sec 8).

    mkowner_record.py BASE_RECORD RULINGS_JSON ROUND OUT_RECORD OUT_PROV
                      [--provenance-json F] [--pick KEY=SCANNER ...] [--skip-v5] [--arm h18s]

One record per item of BASE_RECORD (the smx19 record), in its order:
  * an item in RULINGS_JSON's `items` gets an `owner_review` block (verdict,
    michel_kind, the owner's words, the question, the date).  Nothing else in the
    record changes.  The agent verdict stays; the census (d18_census.py) takes
    owner_review first, then owner_smx1, then the agent.  A ruling with no
    michel_kind stays None and is never filled in from the chain;
  * an item scanned in the v5 pass (ROUND/v_parts/rv5_a*/<key>.json) must have
    exactly two records, from two scanners.  The pass is judged on the
    stop / thru / unscored class:
        both scans agree with the base record's class  -> base kept, "confirmed"
        both agree on another class                    -> "adopted": the record
            is rebuilt from one scan (the lower-numbered scanner when the two
            verdicts are identical, else the one --pick names), and that scan is
            copied to ROUND/v_resolved/<key>.json and listed in
            ROUND/items_adopted.txt for mkspec / apply / merge
        the two scans disagree                          -> base kept, "split",
            owner_queue true
    Every v5 item carries a `review_v5` block with both scans and the outcome.
    The base record's `review` block (the smx19 round) is kept as it was;
  * every other item is copied from BASE_RECORD verbatim.
BASE_RECORD is never written (M13).  --skip-v5 ignores the v5 pass, which gives
the record with the owner's rulings alone, so their census effect can be stated
apart from the pass.
"""
import argparse, glob, json, os, shutil, sys

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("rulings"); ap.add_argument("round")
ap.add_argument("out_record"); ap.add_argument("out_prov")
ap.add_argument("--provenance-json", default=None)
ap.add_argument("--pick", action="append", default=[], help="KEY=SCANNER, for an adopted item whose two verdicts differ within one class")
ap.add_argument("--skip-v5", action="store_true")
ap.add_argument("--arm", default="h18s")
a = ap.parse_args()
if os.path.abspath(a.out_record) == os.path.abspath(a.base):
    sys.exit("refusing to write over the base record (M13)")


def base_v(v):
    return v[5:] if v and v.startswith("FRAG_") else v


def klass(v):
    b = base_v(v)
    return "stop" if b in ("STM_MICHEL", "STM_ONLY") else ("thru" if b == "THRU" else "unscored")


base = json.load(open(a.base))
keys = {o["key"] for o in base}
R = json.load(open(a.rulings))
rul = R["items"]
miss = sorted(set(rul) - keys)
if miss:
    sys.exit("rulings name keys not in the base record: %s" % miss)
pick = dict(p.split("=", 1) for p in a.pick)

v5 = {}
if not a.skip_v5:
    for f in sorted(glob.glob(os.path.join(a.round, "v_parts", "rv5_a*", "*.json"))):
        r = json.load(open(f))
        v5.setdefault(r["key"], []).append(r)
    for k, rs in v5.items():
        if k not in keys:
            sys.exit("v5 record for %s, which is not in the base record" % k)
        if len(rs) != 2 or len({r.get("scanner") for r in rs}) != 2:
            sys.exit("%s: need exactly two v5 records from two scanners, have %d" % (k, len(rs)))
        if k in rul:
            sys.exit("%s is both owner-ruled and in the v5 pass" % k)
        rs.sort(key=lambda r: r.get("scanner") or "")

out, adopted, n = [], [], dict(owner_review=0, confirmed=0, adopted=0, split=0)
for o in base:
    k = o["key"]
    rec = o
    if k in v5:
        s = v5[k]
        cls = [klass(r["verdict"]) for r in s]
        if cls[0] != cls[1]:
            outcome, chosen = "split", None
        elif cls[0] == klass(o["verdict"]):
            outcome, chosen = "confirmed", None
        else:
            outcome = "adopted"
            if s[0]["verdict"] == s[1]["verdict"]:
                chosen = pick.get(k, s[0].get("scanner"))
            elif k in pick:
                chosen = pick[k]
            else:
                sys.exit("%s: both scans say %s but the verdicts differ (%s / %s): name one with --pick"
                         % (k, cls[0], s[0]["verdict"], s[1]["verdict"]))
        if outcome == "adopted":
            v = next((r for r in s if r.get("scanner") == chosen), None)
            if v is None:
                sys.exit("%s: --pick names %s, which did not scan it" % (k, chosen))
            rec = dict(key=k, scan_id=o["scan_id"], tranche=o["tranche"],
                       verdict=v["verdict"], michel_kind=v["michel_kind"], confidence=v["confidence"])
            if v.get("pin_rr") is not None:
                rec["pin_rr"] = v["pin_rr"]
            if v.get("notes"):
                rec["notes"] = v["notes"]
            rec.update(evidence=v["evidence"], tags=v["tags"], arm=a.arm,
                       rubric_sha=v.get("rubric_sha"), scanner=v.get("scanner"),
                       wave=v.get("wave"), written=v.get("written"))
            for f in ("review", "owner_smx1", "calibration"):
                if f in o:
                    rec[f] = o[f]
            os.makedirs(os.path.join(a.round, "v_resolved"), exist_ok=True)
            src = next(f for f in glob.glob(os.path.join(a.round, "v_parts", "rv5_a*", k.replace("/", "_") + ".json"))
                       if json.load(open(f)).get("scanner") == chosen)
            shutil.copyfile(src, os.path.join(a.round, "v_resolved", k.replace("/", "_") + ".json"))
            adopted.append(k)
        else:
            rec = dict(o)
            if outcome == "split":
                rec["owner_queue"] = True
        rec["review_v5"] = dict(
            previous={f: o.get(f) for f in ("verdict", "michel_kind", "confidence", "rubric_sha", "scanner")},
            scans=[{f: r.get(f) for f in ("scanner", "verdict", "michel_kind", "confidence", "rubric_sha", "notes")} for r in s],
            outcome=outcome, adopted=chosen)
        n[outcome] += 1
    if k in rul:
        u = rul[k]
        rec = dict(rec)
        rec["owner_review"] = dict(verdict=u["owner_verdict"], michel_kind=u.get("michel_kind"),
                                   date=R["date"], question=u.get("question"), words=u.get("words"),
                                   scan_id=u.get("scan_id"), note=u.get("note"),
                                   source=os.path.relpath(os.path.abspath(a.rulings),
                                                          os.path.dirname(os.path.abspath(a.out_record))))
        if u.get("app_edit"):
            # an edit the owner's viewer session saved without a stated ruling: kept as provenance, never used
            rec["owner_review"]["app_edit"] = u["app_edit"]
        n["owner_review"] += 1
    out.append(rec)

if not a.skip_v5:
    with open(os.path.join(a.round, "items_adopted.txt"), "w") as fh:
        fh.write("".join(k + "\n" for k in adopted))
with open(a.out_record + ".tmp", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
os.replace(a.out_record + ".tmp", a.out_record)
prov = json.load(open(a.provenance_json)) if a.provenance_json else {}
prov.update(n_records=len(out), n_owner_review=n["owner_review"], n_v5=len(v5),
            v5_outcomes={x: n[x] for x in ("confirmed", "adopted", "split")}, adopted=adopted,
            rubric_shas=sorted({r["rubric_sha"] for r in out if r.get("rubric_sha")}))
json.dump(prov, open(a.out_prov, "w"), indent=1, ensure_ascii=False)
print("wrote %s (%d records: %d owner_review; v5 %d = confirmed %d, adopted %d, split %d) and %s"
      % (a.out_record, len(out), n["owner_review"], len(v5), n["confirmed"], n["adopted"], n["split"], a.out_prov))
