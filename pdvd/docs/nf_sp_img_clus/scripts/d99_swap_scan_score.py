#!/usr/bin/env python3
"""doc pdvd/99 sec 6.4 -- score the blind swap scan with the readout pre-registered in d99/swap_scan_prereg.md.

    python3 d99_swap_scan_score.py --round /home/xqian/tmp/p99scan --key d99/swap_scan_key.tsv \
        [--record-out ../../scan/pdvd_stm_michel_sw99_verdicts.json]

Reads every scanner record under ROUND/v_parts/*/ (one per item; a duplicate is reported and the earliest kept), joins the
key (group, display arm, production-side status, record verdict), and prints:
  1-3  purity per side (stoppers / (stoppers + non-stoppers), Wilson 68 %) and the difference on_only - prod_only with a
       bootstrap over items (10 000, seed 20260915), 68 % and 95 %;
  4    by volume, by production-side status, high confidence only, MESSY+UNCLEAR rate;
  5    Michel: STM_MICHEL fraction of hand stoppers; chain michel_found on the display arm vs the hand Michel;
  6    stopper-or-not agreement with the existing record, by scanner confidence;
  7    derived whole-arm is_stm purity with the shared clusters from the record (labelled as mixing instruments).
"""
import argparse, collections, csv, glob, json, os, sys
import numpy as np
import d99_match as M

STOP = {"STM_MICHEL", "STM_ONLY", "FRAG_STM_MICHEL", "FRAG_STM_ONLY"}
NON = {"THRU", "FRAG_THRU"}
EXCL = {"MESSY", "UNCLEAR"}
SCAN = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/scan"
REC = SCAN + "/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json"


def cls(v):
    return "stop" if v in STOP else "non" if v in NON else "excl" if v in EXCL else None


def wilson(k, n, z=1.0):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def purity_line(label, rows):
    k = sum(r["cls"] == "stop" for r in rows)
    n = sum(r["cls"] in ("stop", "non") for r in rows)
    x = sum(r["cls"] == "excl" for r in rows)
    p, lo, hi = wilson(k, n)
    return f"  {label:44s} items {len(rows):3d}  stoppers {k:3d} / judged {n:3d} = {p:.3f} [{lo:.3f}, {hi:.3f}]  MESSY+UNCLEAR {x}", p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--record-out", default=None)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--exclude-keys", default=None,
                    help="POST-HOC sensitivity (not pre-registered): a file of keys dropped before scoring, e.g. the items "
                         "whose scanner named the readout-window edge (d99/swap_scan_window_flagged.tsv)")
    a = ap.parse_args()
    drop = set()
    if a.exclude_keys:
        drop = {l.split("\t")[0].strip() for l in open(a.exclude_keys) if l.strip() and not l.startswith("#")}
        print(f"POST-HOC: {len(drop)} keys excluded from {a.exclude_keys}")

    key = {r["key"]: r for r in csv.DictReader((l for l in open(a.key) if not l.startswith("#")), delimiter="\t")}
    recs, dup = {}, []
    for f in sorted(glob.glob(os.path.join(a.round, "v_parts", "*", "*.json"))):
        r = json.load(open(f))
        if r["key"] in recs:
            dup.append((r["key"], recs[r["key"]]["scanner"], r["scanner"]))
            if r["written"] >= recs[r["key"]]["written"]:
                continue
        recs[r["key"]] = r
    print(f"# doc pdvd/99 sec 6.4 -- blind swap scan score (d99_swap_scan_score.py); key {a.key}")
    print(f"records {len(recs)} over {len(key)} key items; duplicates {len(dup)} {dup}")
    missing = sorted(k for k, r in key.items() if int(r["in_prep"]) and k not in recs)
    extra = sorted(k for k in recs if k not in key)
    print(f"key items with no record: {len(missing)} {missing}; records not in the key: {extra}")
    shas = collections.Counter(r.get("rubric_sha") for r in recs.values())
    print(f"rubric sha on the records: {dict(shas)}")

    rows = []
    for k, r in recs.items():
        if k not in key or k in drop:
            continue
        kr = key[k]
        evt, cid = k.split("/")
        stm, _ = M.load_stm(evt, kr["arm"])
        d = stm.get(int(cid))
        rows.append(dict(key=k, group=kr["group"], arm=kr["arm"], vol=kr["vol"], status=kr["status"],
                         verdict=r["verdict"], cls=cls(r["verdict"]), conf=r["confidence"], kind=r["michel_kind"],
                         notes=r.get("notes", ""), rec=kr["record_verdict"], rec_conf=kr["record_confidence"],
                         mf=(int(d["michel_found"]) if d else -1), is_stm=(int(d["is_stm"]) if d else -1)))
    assert all(r["cls"] for r in rows), [r for r in rows if not r["cls"]]
    bad = [r["key"] for r in rows if r["is_stm"] != 1]
    print(f"display-arm is_stm != 1 on a scanned item (must be empty): {bad}")
    G = {g: [r for r in rows if r["group"] == g] for g in ("on_only", "prod_only")}

    print("\n== 1-2. purity per side (Wilson 68 %)")
    P = {}
    for g in G:
        line, P[g] = purity_line(g, G[g]); print(line)
        print("     verdicts: " + "  ".join(f"{v} {n}" for v, n in sorted(collections.Counter(r["verdict"] for r in G[g]).items())))

    print(f"\n== 3. purity(on_only) - purity(prod_only), bootstrap over items (N {a.nboot}, seed {a.seed})")
    rng = np.random.default_rng(a.seed)
    arr = {g: np.array([1 if r["cls"] == "stop" else 0 for r in G[g] if r["cls"] in ("stop", "non")]) for g in G}
    diffs = []
    for _ in range(a.nboot):
        s = {g: arr[g][rng.integers(0, len(arr[g]), len(arr[g]))].mean() for g in arr}
        diffs.append(s["on_only"] - s["prod_only"])
    diffs = np.array(diffs)
    d0 = arr["on_only"].mean() - arr["prod_only"].mean()
    q68 = np.quantile(diffs, [0.16, 0.84]); q95 = np.quantile(diffs, [0.025, 0.975])
    verdict = "LOSS" if q95[1] < 0 else "GAIN" if q95[0] > 0 else "purity-neutral within this sample"
    print(f"  difference {d0:+.3f}  68 % [{q68[0]:+.3f}, {q68[1]:+.3f}]  95 % [{q95[0]:+.3f}, {q95[1]:+.3f}]  -> {verdict}")

    print("\n== 4. splits")
    for g in G:
        for vol in ("top", "bottom"):
            print(purity_line(f"{g} {vol}", [r for r in G[g] if r["vol"] == vol])[0])
    for st in ("not_candidate", "candidate_not_stm"):
        for g in G:
            print(purity_line(f"{g} production-side status {st}", [r for r in G[g] if r["status"] == st])[0])
    for g in G:
        print(purity_line(f"{g} object not matched (object_*)", [r for r in G[g] if r["status"].startswith("object_")])[0])
    for g in G:
        print(purity_line(f"{g} high confidence only", [r for r in G[g] if r["conf"] == "high"])[0])
    for g in G:
        c = collections.Counter(r["conf"] for r in G[g])
        print(f"  {g:44s} confidence {dict(c)}")
    for g in G:
        pre = collections.Counter(p.split(":")[0].strip() for r in G[g] for p in r["notes"].split(";") if ":" in p)
        print(f"  {g:44s} notes prefixes {dict(pre)}")

    print("\n== 5. Michel")
    for g in G:
        st = [r for r in G[g] if r["cls"] == "stop"]
        k = sum(r["verdict"] in ("STM_MICHEL", "FRAG_STM_MICHEL") for r in st)
        p, lo, hi = wilson(k, len(st))
        judged = [r for r in G[g] if r["cls"] in ("stop", "non")]
        hm = lambda r: r["verdict"] in ("STM_MICHEL", "FRAG_STM_MICHEL")
        tp = sum(r["mf"] == 1 and hm(r) for r in judged); fp = sum(r["mf"] == 1 and not hm(r) for r in judged)
        fn = sum(r["mf"] != 1 and hm(r) for r in judged)
        print(f"  {g:10s} hand STM_MICHEL / hand stoppers {k}/{len(st)} = {p:.3f} [{lo:.3f}, {hi:.3f}];  chain michel_found "
              f"vs hand Michel (judged): TP {tp} FP {fp} FN {fn}  purity {tp / max(tp + fp, 1):.3f}  eff {tp / max(tp + fn, 1):.3f}")

    print("\n== 6. stopper-or-not agreement with the existing record (carried on p98vonq / smx on p96vprod)")
    for g in G:
        for conf in ("high", "medium", "low", None):
            sel = [r for r in G[g] if r["rec"] and cls(r["rec"]) in ("stop", "non") and r["cls"] in ("stop", "non")
                   and (conf is None or r["conf"] == conf)]
            agree = sum(cls(r["rec"]) == r["cls"] for r in sel)
            print(f"  {g:10s} scanner {conf or 'all':6s} agree {agree}/{len(sel)}")
        dis = [(r["key"], r["rec"], r["verdict"], r["conf"]) for r in G[g]
               if r["rec"] and cls(r["rec"]) in ("stop", "non") and r["cls"] in ("stop", "non") and cls(r["rec"]) != r["cls"]]
        print(f"    disagreements (key, record, scan, scan conf): {dis}")

    print("\n== 7. derived whole-arm is_stm purity, shared clusters from the record (MIXES INSTRUMENTS)")
    import d99_swap_scan_set as W
    import d99_population as Pp
    from concurrent.futures import ProcessPoolExecutor
    carried = {r["key"]: r for r in json.load(open(f"{SCAN}/pdvd_stm_michel_smx9_carried_p98vonq.json"))}
    record = {r["key"]: r for r in json.load(open(REC))}
    ev = Pp.events()
    for arm, other, src, g in (("p98vonq", "p96vprod", carried, "on_only"), ("p96vprod", "p98vonq", record, "prod_only")):
        with ProcessPoolExecutor(32) as ex:
            allr = [x for xx in ex.map(W.rows_of, [(e, arm, other) for e in ev]) for x in xx]
        shared = [x for x in allr if x["status"] == "is_stm"]
        sv = [cls(src[x["key"]]["verdict"]) for x in shared if x["key"] in src]
        ks, ns = sum(c == "stop" for c in sv), sum(c in ("stop", "non") for c in sv)
        n_side = len(allr) - len(shared)
        ps = ks / max(ns, 1)
        pa = (len(shared) * ps + n_side * P[g]) / len(allr)
        print(f"  {arm:9s} is_stm {len(allr)} = shared {len(shared)} (record judged {ns}, purity {ps:.3f}) + {g} {n_side} "
              f"(scan purity {P[g]:.3f}) -> {pa:.3f}")

    if a.record_out:
        if os.path.exists(a.record_out):
            sys.exit(f"REFUSING: {a.record_out} exists")
        out = []
        for r in sorted(rows, key=lambda r: r["key"]):
            v = dict(recs[r["key"]])
            v.update(group=r["group"], display_arm=r["arm"], production_side_status=r["status"], tag="sw99",
                     source_round="doc pdvd/99 sec 6.4 blind swap scan")
            out.append(v)
        json.dump(out, open(a.record_out, "w"), indent=1, ensure_ascii=False)
        print(f"\nwrote {a.record_out} ({len(out)} records)")


if __name__ == "__main__":
    main()
