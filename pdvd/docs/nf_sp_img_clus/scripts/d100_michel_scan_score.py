#!/usr/bin/env python3
"""doc pdvd/100 round 2 -- fold the owner's own100m look (what the gain flip moves) and re-grade BOTH arms on the corrected
record, as pre-registered in d100/prereg_round2.md sec 1.

    python3 d100_michel_scan_score.py --set /home/xqian/tmp/p100/mscan \
        --record-out ../../scan/pdvd_stm_michel_own100m_verdicts.json --carry-out /home/xqian/tmp/p100/carry_r2 \
        > ../d100/michel_scan.txt

1. each object: record verdict -> owner verdict, both chain answers.
2. how the owner's verdicts split each mover kind (descriptive only: the set is selected by the disputed metric, so no
   purity is computed on it).
3. writes the NEW record (one item per arm key: p100c = p99rwon key, and the p99wflip key) and the corrected records
   (latest_on_p99wflip / latest_on_p99rwon with every own100m or own100 key's verdict replaced; new files, the carried
   records are never edited).
4. the re-grade (d99_grade's items + census on the corrected records) and the pre-registered criterion: p100c Michel
   purity within 1.5 sigma (binomial, both arms in quadrature) of p99wflip's.  is_stm beside it.
Refuses an existing record or carry dir (M13).  Unlabelled items keep their carried verdict and are counted.
"""
import argparse, collections, csv, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d99_grade as G

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
LABELS = IMG + "/pdvd/work/stm_michel_labels/own100m/labels.json"
OWN100 = IMG + "/pdvd/docs/scan/pdvd_stm_michel_own100_verdicts.json"
P100 = "/home/xqian/tmp/p100"
CARRIED = {"p99wflip": P100 + "/carry/latest_on_p99wflip.json", "p100c": P100 + "/carry/latest_on_p99rwon.json"}
KEYCOL = {"p99wflip": "key_p99wflip", "p100c": "key_p100c"}
STOP = ("STM_MICHEL", "STM_ONLY")


def base(v):
    return (v or "").split(" ")[0].replace("FRAG_", "")


def cls(v):
    b = base(v)
    if not b:
        return "unjudged"
    return "stopper" if b in STOP else ("excluded" if b in ("MESSY", "UNCLEAR") else "non-stopper")


def purity(tp, fp):
    n = tp + fp
    p = tp / n if n else float("nan")
    return p, (math.sqrt(p * (1 - p) / n) if n else float("nan")), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--record-out", required=True)
    ap.add_argument("--carry-out", required=True)
    a = ap.parse_args()
    for p in (a.record_out, a.carry_out):
        if os.path.exists(p):
            sys.exit(f"REFUSING: {p} exists")
    L = json.load(open(LABELS))["labels"]
    rows = {r["key_p100c"]: r for r in csv.DictReader(open(a.set + "/items.tsv"), delimiter="\t")}
    extra = set(L) - set(rows)
    if extra:
        sys.exit(f"labels name keys outside the set: {sorted(extra)}")
    print(f"# doc pdvd/100 round 2 -- own100m (owner, 2026-09-13, not blind): {len(L)} of {len(rows)} objects labelled "
          f"(tranche 1: {sum(1 for k in L if rows[k]['tranche'] == '1')} of {sum(1 for r in rows.values() if r['tranche'] == '1')})")
    print(f"# labels {LABELS}; set {a.set}\n")

    print("== 1. each object (shown on p100c)")
    for k, r in sorted(rows.items(), key=lambda kv: int(kv[1]["scan_id"])):
        x = L.get(k)
        ov = x["label"] if x else "-"
        mk = (x.get("michel_kind") or "") if x and base(ov) == "STM_MICHEL" else ""
        chg = "" if not x else ("  same" if base(ov) == r["record_verdict"] else
                                ("  CLASS" if cls(ov) != cls(r["record_verdict"]) else "  kind"))
        print(f"  {r['scan_id']:>2s} t{r['tranche']} {k:14s} wflip {r['key_p99wflip']:14s} {r['vol']:6s} "
              f"p99wflip [{r['p99wflip']}] p100c [{r['p100c']}] | record {r['record_verdict']} ({r['record_confidence']}) "
              f"-> OWNER {ov} {mk}{chg}")

    print("\n== 2. the owner's verdicts by mover kind (descriptive; NOT a purity -- the set is selected by the metric)")
    by = collections.defaultdict(collections.Counter)
    for k, r in rows.items():
        x = L.get(k)
        kind = f"t{r['tranche']} {r['vol']:6s} {r['kind']}"
        by[kind][base(x["label"]) if x else "unlabelled"] += 1
    for kind in sorted(by):
        print(f"  {kind:62s} {dict(by[kind])}")

    rec = []
    for k, x in L.items():
        r = rows[k]
        for arm, col, arms in (("p100c", "key_p100c", "p99rwon p100c"), ("p99wflip", "key_p99wflip", "p99wflip")):
            rec.append(dict(key=r[col], verdict=x["label"],
                            michel_kind=x.get("michel_kind") if base(x["label"]) == "STM_MICHEL" else None,
                            confidence="owner", source="own100m (owner, 2026-09-13)", scan_id=int(x.get("scan_id") or r["scan_id"]),
                            tranche=int(r["tranche"]), evidence=x.get("notes") or "",
                            notes="doc pdvd/100 round 2: p99wflip -> p100c chain mover, owner look on p100c (not blind)",
                            arms=arms, shown_key=k, shown_arm="p100c", record_key=r["record_key"],
                            pin=x.get("pin") if (x.get("pin") or {}).get("placed") else None,
                            pf_segments=x.get("pf_segments"), tags=x.get("pf_segments"),
                            prior_record=f"{r['record_verdict']} ({r['record_confidence']})"))
    json.dump(rec, open(a.record_out, "w"), indent=1)
    print(f"\n== 3. wrote {a.record_out}: {len(rec)} items (one per arm key)")

    own100 = {}
    if os.path.exists(OWN100):
        for it in json.load(open(OWN100)):
            own100[it["key"]] = it
    os.makedirs(a.carry_out)
    corrected = {}
    for arm, path in CARRIED.items():
        items = json.load(open(path))
        o_m = {it["key"]: it for it in rec if (arm == "p100c") == ("p100c" in it["arms"])}
        n_m = n_o = 0
        for it in items:
            src = o_m.get(it["key"])
            tag = "own100m"
            if src is None and it["key"] in own100 and arm in own100[it["key"]].get("arms", ""):
                src, tag = own100[it["key"]], "own100"
            if src is None:
                continue
            it["prior_verdict"] = it.get("verdict"); it["prior_confidence"] = it.get("confidence")
            it["verdict"] = src["verdict"]; it["michel_kind"] = src.get("michel_kind")
            it["confidence"] = "owner"; it["corrected_by"] = tag
            n_m += tag == "own100m"; n_o += tag == "own100"
        out = f"{a.carry_out}/latest_on_{'p99rwon' if arm == 'p100c' else arm}_corrected.json"
        json.dump(items, open(out, "w"), indent=1)
        corrected[arm] = out
        print(f"  {arm}: {out}: {len(items)} items; replaced by own100m {n_m} (of {len(o_m)}), by own100 {n_o}")

    print("\n== 4. re-grade on the corrected records (pre-registered criterion: prereg.md sec 2 / prereg_round2.md sec 1)")
    res = {}
    for arm in ("p99wflip", "p100c"):
        its, miss, _, n = G.items_on(arm, corrected[arm])
        (TP, FP, FN, TN), _, (mTP, mFP, mFN, mTN) = G.BM.census(its)
        res[arm] = dict(stm=purity(TP, FP), stm_eff=TP / max(1, TP + FN), mich=purity(mTP, mFP), mich_eff=mTP / max(1, mTP + mFN),
                        counts=(TP, FP, FN, TN, mTP, mFP, mFN, mTN))
        s, m = res[arm]["stm"], res[arm]["mich"]
        print(f"  {arm:8s} is_stm {TP}/{FP}/{FN}/{TN} purity {s[0]:.3f} +- {s[1]:.3f} eff {res[arm]['stm_eff']:.3f}   "
              f"michel {mTP}/{mFP}/{mFN}/{mTN} purity {m[0]:.3f} +- {m[1]:.3f} eff {res[arm]['mich_eff']:.3f}   "
              f"(record {n} items, no candidate {miss})")
        for vol in ("top", "bottom"):
            sub = [x for x in its if (float(x[4]["stop_x"]) > 0) == (vol == "top")]
            (TP, FP, FN, TN), _, (mTP, mFP, mFN, mTN) = G.BM.census(sub)
            print(f"      {vol:6s} is_stm {TP}/{FP}/{FN}/{TN} purity {TP / max(1, TP + FP):.3f}   "
                  f"michel {mTP}/{mFP}/{mFN}/{mTN} purity {mTP / max(1, mTP + mFP):.3f} eff {mTP / max(1, mTP + mFN):.3f}")
    for what in ("mich", "stm"):
        (pw, sw, _), (pc, sc, _) = res["p99wflip"][what], res["p100c"][what]
        z = (pc - pw) / math.sqrt(sw ** 2 + sc ** 2)
        name = "Michel purity" if what == "mich" else "is_stm purity"
        verdict = ("PASS" if abs(z) <= 1.5 or pc >= pw else "FAIL") if what == "mich" else "(reported)"
        print(f"  {name}: p99wflip {pw:.3f} -> p100c {pc:.3f}, difference {pc - pw:+.3f} = {z:+.2f} sigma  {verdict}")
    print(f"\n  unlabelled objects keep their carried verdict: {len(rows) - len(L)}")


if __name__ == "__main__":
    main()
