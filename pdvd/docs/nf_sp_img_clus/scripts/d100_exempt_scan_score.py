#!/usr/bin/env python3
"""doc pdvd/100 round 2 -- fold the owner's own100x look (what the readout-edge exemption brings back) and apply the adoption
rule pre-registered in d100/prereg_round2.md sec 2.

    python3 d100_exempt_scan_score.py --set /home/xqian/tmp/p100/xscan \
        --record-out ../../scan/pdvd_stm_michel_own100x_verdicts.json > ../d100/exempt_scan.txt

The gained sets are re-derived from d100r2_exempt.py's outputs (d100/exempt_p100bx.txt, d100/exempt_p100bxp.txt sec 5):
each gained object carries its owner verdict from own100 (by geometry, already resolved there) or from own100x (this scan;
an object shown once stands for both arms' keys).  Rule: the exemption is recommended for production only if the
owner-judged gained set's purity (stoppers / (stoppers + non-stoppers); MESSY / UNCLEAR left out) is at least the arm's
is_stm purity on the corrected record of sec 7.2 (p100c 0.963; for the production side, p99wflip 0.965).  Below it: reported
as "N stoppers for M non-stoppers", left to the owner.  Split by edge, because own100 measured only 039349's late edge.
Refuses an existing record (M13).
"""
import argparse, csv, json, os, re, sys

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
LABELS = IMG + "/pdvd/work/stm_michel_labels/own100x/labels.json"
D100 = IMG + "/pdvd/docs/nf_sp_img_clus/d100"
ARMS = {"p100bx": ("exempt_p100bx.txt", 0.963, "p100c"), "p100bxp": ("exempt_p100bxp.txt", 0.965, "p99wflip")}
STOP = ("STM_MICHEL", "STM_ONLY")


def cls(v):
    b = (v or "").split(" ")[0].replace("FRAG_", "")
    if not b:
        return "unjudged"
    return "stopper" if b in STOP else ("excluded" if b in ("MESSY", "UNCLEAR") else "non-stopper")


def gained(fn):
    out, on = [], False
    for line in open(os.path.join(D100, fn)):
        if line.startswith("== 5."):
            on = True; continue
        if on:
            m = re.match(r"\s+(\d{6}_\d+/\d+)\s+(\w+)\s+tick\s+([0-9.]+)\s+michel_ke\s+([0-9.]+)\s+\|\s+(\S+)\s+(\S+)", line)
            if m:
                out.append(dict(key=m.group(1), vol=m.group(2), tick=float(m.group(3)), ke=float(m.group(4)),
                                src=m.group(5), verdict=None if m.group(6) == "-" else m.group(6)))
    return out


def edge(key, tick):
    nt = 6400 if key.startswith("039349") else 10000
    return "early" if tick < 60 else f"late{nt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--record-out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.record_out):
        sys.exit(f"REFUSING: {a.record_out} exists")
    L = json.load(open(LABELS))["labels"]
    items = {r["key"]: r for r in csv.DictReader(open(a.set + "/items.tsv"), delimiter="\t")}
    also = {r["also"].split(" ")[1]: k for k, r in items.items() if r["also"]}   # other-arm key -> shown key
    print(f"# doc pdvd/100 round 2 -- own100x (owner, 2026-09-13, not blind): {len(L)} of {len(items)} objects labelled")
    print(f"# labels {LABELS}; set {a.set}\n")

    print("== 1. each object")
    for k, r in sorted(items.items(), key=lambda kv: int(kv[1]["scan_id"])):
        x = L.get(k)
        print(f"  {r['scan_id']:>2s} {k:15s} {r['arm']:8s} {r['vol']:6s} tick {float(r['stop_tick']):7.1f} {r['edge']:32s} "
              f"michel {float(r['michel_ke']):5.1f} MeV | OWNER {x['label'] if x else '-'}{'  also ' + r['also'] if r['also'] else ''}")

    rec = []
    for k, x in L.items():
        r = items[k]
        keys = [(k, r["arm"])] + ([(r["also"].split(" ")[1], r["also"].split(" ")[0])] if r["also"] else [])
        for kk, arm in keys:
            rec.append(dict(key=kk, verdict=x["label"], michel_kind=x.get("michel_kind") if cls(x["label"]) == "stopper" else None,
                            confidence="owner", source="own100x (owner, 2026-09-13)", scan_id=int(r["scan_id"]),
                            evidence=x.get("notes") or "", arms=arm, shown_key=k, shown_arm=r["arm"],
                            notes=f"doc pdvd/100 round 2: readout-edge exemption gain, {r['edge']}, owner look (not blind)",
                            pin=x.get("pin") if (x.get("pin") or {}).get("placed") else None))
    json.dump(rec, open(a.record_out, "w"), indent=1)
    print(f"\nwrote {a.record_out}: {len(rec)} items (one per arm key)")

    print("\n== 2. the adoption rule (prereg_round2.md sec 2), per arm and by edge")
    for arm, (fn, bar, base) in ARMS.items():
        g = gained(fn)
        for it in g:
            if it["verdict"] is None:
                shown = it["key"] if it["key"] in items else also.get(it["key"])
                it["verdict"] = (L.get(shown) or {}).get("label") if shown else None
                it["src"] = "own100x" if it["verdict"] else "unjudged"
            it["edge"] = edge(it["key"], it["tick"])
        st = sum(cls(it["verdict"]) == "stopper" for it in g)
        ns = sum(cls(it["verdict"]) == "non-stopper" for it in g)
        ex = sum(cls(it["verdict"]) == "excluded" for it in g)
        un = sum(cls(it["verdict"]) == "unjudged" for it in g)
        pur = st / (st + ns) if st + ns else float("nan")
        ok = pur >= bar if st + ns else False
        print(f"  {arm} (vs {base}): gained {len(g)}: stoppers {st}, non-stoppers {ns}, excluded {ex}, unjudged {un}; "
              f"purity {pur:.3f} vs bar {bar:.3f} -> {'RECOMMEND' if ok and un == 0 else ('below the bar: ' + str(st) + ' stoppers for ' + str(ns) + ' non-stoppers, owner decides' if st + ns else 'nothing judged')}")
        for e in ("early", "late6400", "late10000"):
            s = [it for it in g if it["edge"] == e]
            if s:
                print(f"      {e:9s} n {len(s)}: " + ", ".join(f"{it['key']} {it['verdict'] or '-'} ({it['src']})" for it in s))


if __name__ == "__main__":
    main()
