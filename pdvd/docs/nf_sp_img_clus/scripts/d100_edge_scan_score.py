#!/usr/bin/env python3
"""doc pdvd/100 step A -- fold and score the owner's look (own100, 2026-09-13, not blind) at the 19 run-039349 objects the
real readout window untagged.

    python3 d100_edge_scan_score.py --set /home/xqian/tmp/p100/scan \
        --record-out ../../scan/pdvd_stm_michel_own100_verdicts.json > ../d100/edge_scan.txt

Inputs: the viewer's labels (pdvd/work/stm_michel_labels/own100/labels.json), the set's items.tsv (both arm keys, the
guard's own log line, the existing record verdicts; d100_edge_scan_set.py) and the payloads the owner saw (the source
arm's chain answer).  Writes a NEW record (refuses an existing file; M13) with one item per ARM KEY: an object in both the
production and the latest list gives two items with the same verdict (key on p96vprod/p99wflip and on p98vonq/p99rwon).

Sections: 1 each object; 2 counts by side and volume; 3 record verdicts the owner changed; 4 would a smaller guard window
separate (per tick threshold, stoppers vs non-stoppers still removed); 5 the chain's own Michel finding as a separator;
6 what the guard costs on production's graded record (its 11 objects).
"""
import argparse, collections, csv, json, os, re, sys

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
LABELS = IMG + "/pdvd/work/stm_michel_labels/own100/labels.json"
STOP = ("STM_MICHEL", "STM_ONLY")


def cls(v):
    if not v:
        return "unjudged"
    b = v.split(" ")[0].replace("FRAG_", "")
    return "stopper" if b in STOP else ("excluded" if b in ("MESSY", "UNCLEAR") else "non-stopper")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--record-out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.record_out):
        sys.exit(f"REFUSING: {a.record_out} exists")
    lab = json.load(open(LABELS))
    L = lab["labels"]
    rows = {r["key"]: r for r in csv.DictReader(open(a.set + "/items.tsv"), delimiter="\t")}
    if set(L) != set(rows):
        sys.exit(f"labels {len(L)} vs set {len(rows)}: not the same items")
    print(f"# doc pdvd/100 step A -- own100 (owner, 2026-09-13, not blind): {len(L)} of {len(rows)} objects labelled")
    print(f"# labels {LABELS}; set {a.set}\n")

    objs = []
    for k in sorted(L, key=lambda s: (int(s.split("_")[1].split("/")[0]), int(s.split("/")[1]))):
        x, r = L[k], rows[k]
        pay = json.load(open(f"{a.set}/prep/smprep-{k.replace('/', '-c')}.json"))["verdict"]
        ticks = {}
        for part in r["guard"].split(" | "):
            m = re.match(r"(\w+) .*?([0-9.]+) ticks before", part)
            if m:
                ticks[m.group(1)] = float(m.group(2))
        recv = {}
        for part in filter(None, r["record_verdicts"].split(" | ")):
            m = re.match(r"(\w+) (\S+): (\w+) \((\w+)\)", part)
            if m:
                recv[m.group(1)] = (m.group(3), m.group(4))
        pin = x.get("pin") or {}
        objs.append(dict(key=k, x=x, r=r, pay=pay, ticks=ticks, recv=recv, verdict=x["label"], cls=cls(x["label"]),
                         pin=pin if pin.get("placed") else None))

    print("== 1. each object (shown arm; the guard's ticks before the 6400-tick frame end; the chain's Michel on the shown arm)")
    for o in objs:
        p = o["pay"]
        mk = o["x"].get("michel_kind") if o["verdict"] == "STM_MICHEL" else ""
        print(f"  {o['key']:14s} {o['r']['shown_arm']:8s} {o['r']['vol']:6s} ticks {','.join('%s %.1f' % kv for kv in o['ticks'].items()) or 'no guard line':28s} "
              f"chain michel_found {p.get('michel_found')} ke {p.get('michel_ke_best') or 0:5.1f} | OWNER {o['verdict']:10s} {mk:9s}"
              f"{' pin rr %.1f' % o['pin']['rr'] if o['pin'] else ''} | record {'; '.join('%s %s (%s)' % (s, v, c) for s, (v, c) in o['recv'].items()) or 'none'}")

    print("\n== 2. counts (an object in both lists counts on both sides)")
    c = collections.Counter(o["verdict"] for o in objs)
    print(f"  owner verdicts: {dict(c)}; stoppers {sum(o['cls'] == 'stopper' for o in objs)} / non-stoppers {sum(o['cls'] == 'non-stopper' for o in objs)}")
    for side, col in (("production (p96vprod -> p99wflip)", "prod_key"), ("latest (p98vonq -> p99rwon)", "latest_key")):
        s = [o for o in objs if o["r"][col]]
        print(f"  {side}: {len(s)} objects: stoppers {sum(o['cls'] == 'stopper' for o in s)}, non-stoppers {sum(o['cls'] == 'non-stopper' for o in s)}")
    for vol in ("top", "bottom"):
        s = [o for o in objs if o["r"]["vol"] == vol]
        print(f"  {vol}: {len(s)}: stoppers {sum(o['cls'] == 'stopper' for o in s)}, non-stoppers {sum(o['cls'] == 'non-stopper' for o in s)}")

    print("\n== 3. the owner against the existing records")
    for o in objs:
        for s, (v, conf) in o["recv"].items():
            if v != o["verdict"]:
                flip = "CLASS " if cls(v) != o["cls"] else "kind  "
                print(f"  {flip}{o['key']:14s} {s} {v} ({conf}) -> owner {o['verdict']}")
        if not o["recv"]:
            print(f"  new    {o['key']:14s} unjudged -> owner {o['verdict']}")

    print("\n== 4. a smaller guard window (objects the guard removed; smallest ticks-before over the sides that name it)")
    g = [o for o in objs if o["ticks"]]
    print(f"  guard-removed objects: {len(g)} (stoppers {sum(o['cls'] == 'stopper' for o in g)}, non-stoppers {sum(o['cls'] == 'non-stopper' for o in g)}); "
          f"not guard-removed: {[o['key'] for o in objs if not o['ticks']]}")
    for T in (10, 20, 30, 40, 50, 60):
        rm = [o for o in g if min(o["ticks"].values()) <= T]
        print(f"  guard {T:2d} ticks: still removes stoppers {sum(o['cls'] == 'stopper' for o in rm):2d}, non-stoppers {sum(o['cls'] == 'non-stopper' for o in rm):2d}")
    st = sorted(min(o["ticks"].values()) for o in g if o["cls"] == "stopper")
    nt = sorted(min(o["ticks"].values()) for o in g if o["cls"] == "non-stopper")
    print(f"  ticks before the end -- stoppers {st}; non-stoppers {nt}")

    print("\n== 5. the chain's own Michel as a separator (on the guard-removed objects)")
    for mf in (1, 0):
        s = [o for o in g if int(o["pay"].get("michel_found") or 0) == mf]
        print(f"  chain michel_found {mf}: {len(s)}: stoppers {sum(o['cls'] == 'stopper' for o in s)} "
              f"({[o['key'] for o in s if o['cls'] == 'stopper']}), non-stoppers {sum(o['cls'] == 'non-stopper' for o in s)} "
              f"({[o['key'] for o in s if o['cls'] == 'non-stopper']})")

    print("\n== 6. production's 11 objects (the change shipped in wcp f52a374e)")
    s = [o for o in objs if o["r"]["prod_key"]]
    print(f"  before the window (p96vprod) all 11 were is_stm; with it (p99wflip) none is.  Owner: stoppers "
          f"{sum(o['cls'] == 'stopper' for o in s)} (lost), non-stoppers {sum(o['cls'] == 'non-stopper' for o in s)} (removed correctly).")
    s1 = [o for o in s if int(o["pay"].get("michel_found") or 0) == 1 and o["ticks"]]
    print(f"  of them with a chain Michel: {len(s1)}: stoppers {sum(o['cls'] == 'stopper' for o in s1)}, non-stoppers {sum(o['cls'] == 'non-stopper' for o in s1)}")

    rec = []
    for o in objs:
        for side, col, arms in (("latest", "latest_key", "p98vonq p99rwon"), ("production", "prod_key", "p96vprod p99wflip")):
            kk = o["r"][col]
            if not kk:
                continue
            rec.append(dict(key=kk, verdict=o["verdict"],
                            michel_kind=o["x"].get("michel_kind") if o["verdict"] == "STM_MICHEL" else None,
                            confidence="owner", source="own100 (owner, 2026-09-13)", scan_id=int(o["x"]["scan_id"]),
                            tranche=int(o["x"]["tranche"]), evidence=o["x"].get("notes") or "",
                            notes="doc pdvd/100 step A: run 039349 readout-window edge, owner look (not blind)",
                            side=side, arms=arms, shown_key=o["key"], shown_arm=o["r"]["shown_arm"],
                            pin=o["pin"], pf_segments=o["x"].get("pf_segments"), tags=o["x"].get("pf_segments"),
                            guard=o["r"]["guard"], prior_record=o["r"]["record_verdicts"]))
    json.dump(rec, open(a.record_out, "w"), indent=1)
    print(f"\nwrote {a.record_out}: {len(rec)} items (one per arm key)")


if __name__ == "__main__":
    main()
