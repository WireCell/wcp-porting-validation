#!/usr/bin/env python3
"""doc pdvd/99 sec 6.4 -- the blind scan set for the two sides of the is_stm swap between production and the latest
configuration (owner 2026-09-13: "scan the clusters the latest configuration tags but production doesn't").

    python3 d99_swap_scan_set.py --on p98vonq --prod p96vprod --n-controls 45 --seed 20260913 --out DIR

Groups (matched with d99_population's matcher and thresholds, so the sets are exactly sec 6.3's rows):
  on_only    every p98vonq is_stm cluster that is NOT is_stm on p96vprod: candidate_not_stm, not_candidate, or its object
             did not match (object_*).  ALL of them are scanned.
  prod_only  every p96vprod is_stm cluster that is not is_stm on p98vonq.  A seeded sample of --n-controls is scanned in the
             same waves, by the same scanners, under the same rubric: the instrument-matched purity of the side that leaves.
Each item is displayed on the arm that tags it (on_only from the p98vonq prep, prod_only from the p96vprod prep).  An
event/cluster key present in both groups (the two arms number clusters independently) is excluded from the control draw so
one shot directory cannot hold two objects.

Writes (never over an existing file):
  DIR/items_all.txt          shuffled (seed) event/cluster list -- what the scanners are handed, no group
  DIR/sheet_<arm>.tsv        the harness manifest per display arm (rows copied from that arm's prep sheet)
  DIR/scan_key.tsv           key: group, display arm, production-side status, carried/record verdict (NEVER shown)
"""
import argparse, collections, csv, json, os, random, sys
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
import d99_match as M
import d99_population as P

SCAN = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/scan"
REC = SCAN + "/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json"


def rows_of(job):
    evt, A, B = job
    sa, _ = M.load_stm(evt, A)
    sb, _ = M.load_stm(evt, B)
    ba, bb = M.load_bee(evt, A), M.load_bee(evt, B)
    out = []
    if ba is None or bb is None:
        return out
    tree = cKDTree(bb[0])
    for cid, d in sorted(sa.items()):
        if int(d["is_stm"]) != 1:
            continue
        vol = "top" if d["stop_x"] > 0 else "bottom"
        pts = ba[0][ba[1] == cid]
        if len(pts) == 0:
            out.append(dict(key=f"{evt}/{cid}", vol=vol, status="no_points", other=""))
            continue
        m = M.match_cluster(pts, bb[0], bb[1], tree)
        st = M.status_of(m)
        if st != "ok":
            out.append(dict(key=f"{evt}/{cid}", vol=vol, status="object_" + st, other=f"{evt}/{m.get('best')}"))
            continue
        e = sb.get(m["best"])
        s = "not_candidate" if e is None else ("candidate_not_stm" if int(e["is_stm"]) != 1 else "is_stm")
        out.append(dict(key=f"{evt}/{cid}", vol=vol, status=s, other=f"{evt}/{m['best']}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", default="p98vonq")
    ap.add_argument("--prod", default="p96vprod")
    ap.add_argument("--n-controls", type=int, default=45)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--prep-sheet", nargs=2, metavar=("ARM", "SHEET"), action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    for f in ("items_all.txt", "scan_key.tsv"):
        if os.path.exists(os.path.join(a.out, f)):
            sys.exit(f"REFUSING: {a.out}/{f} exists (a scan set is never rebuilt in place)")
    os.makedirs(a.out, exist_ok=True)
    ev = P.events()
    with ProcessPoolExecutor(a.jobs) as ex:
        on = [r for rr in ex.map(rows_of, [(e, a.on, a.prod) for e in ev]) for r in rr]
        pr = [r for rr in ex.map(rows_of, [(e, a.prod, a.on) for e in ev]) for r in rr]
    on_only = [dict(r, group="on_only", arm=a.on) for r in on if r["status"] != "is_stm"]
    prod_only = [dict(r, group="prod_only", arm=a.prod) for r in pr if r["status"] != "is_stm"]
    print(f"{a.on} is_stm {len(on)}: not is_stm on {a.prod} {len(on_only)} "
          f"{dict(collections.Counter(r['status'] for r in on_only))}")
    print(f"{a.prod} is_stm {len(pr)}: not is_stm on {a.on} {len(prod_only)} "
          f"{dict(collections.Counter(r['status'] for r in prod_only))}")
    clash = {r["key"] for r in on_only} & {r["key"] for r in prod_only}
    pool = sorted((r for r in prod_only if r["key"] not in clash), key=lambda r: r["key"])
    ctl = sorted(random.Random(a.seed).sample(pool, min(a.n_controls, len(pool))), key=lambda r: r["key"])
    print(f"key clashes excluded from the control draw: {sorted(clash)}; controls {len(ctl)} of {len(pool)} (seed {a.seed})")

    carried = {r["key"]: r for r in json.load(open(f"{SCAN}/pdvd_stm_michel_smx9_carried_{a.on}.json"))}
    record = {r["key"]: r for r in json.load(open(REC))}
    sheets = {arm: sheet for arm, sheet in a.prep_sheet}
    items = on_only + ctl
    for r in items:
        src = carried if r["arm"] == a.on else record
        v = src.get(r["key"])
        r["record_verdict"] = v["verdict"] if v else ""
        r["record_confidence"] = v.get("confidence", "") if v else ""
        r["record_source"] = v.get("source", "") if v else ""
    order = list(range(len(items)))
    random.Random(a.seed + 1).shuffle(order)
    items = [items[i] for i in order]
    for n, r in enumerate(items, start=1):
        r["scan_id"] = n

    # harness manifests: the prep sheet's own rows for this arm's items, re-numbered by scan_id
    for arm, sheet in sheets.items():
        lines = [l for l in open(sheet) if not l.startswith("#")]
        rd = list(csv.DictReader(lines, delimiter="\t"))
        by = {f"{x['event']}/{x['cluster']}": x for x in rd}
        cols = list(rd[0].keys())
        mine = [r for r in items if r["arm"] == arm]
        miss = [r["key"] for r in mine if r["key"] not in by]
        if miss:
            print(f"!! {arm}: {len(miss)} item(s) absent from the prep sheet (prep filter): {miss}")
        p = os.path.join(a.out, f"sheet_{arm}.tsv")
        with open(p, "w") as fh:
            fh.write(f"# doc pdvd/99 sec 6.4 -- blind swap scan, display arm {arm}; rows from {sheet}\n")
            fh.write("\t".join(cols) + "\n")
            for r in mine:
                if r["key"] in by:
                    x = dict(by[r["key"]], scan_id=str(r["scan_id"]), tranche="1")
                    fh.write("\t".join(x[c] for c in cols) + "\n")
        for r in mine:
            r["in_prep"] = int(r["key"] in by)
    with open(os.path.join(a.out, "items_all.txt"), "w") as fh:
        fh.write("".join(r["key"] + "\n" for r in items if r["in_prep"]))
    cols = ["scan_id", "key", "group", "arm", "vol", "status", "other", "in_prep", "record_verdict", "record_confidence",
            "record_source"]
    with open(os.path.join(a.out, "scan_key.tsv"), "w") as fh:
        fh.write(f"# doc pdvd/99 sec 6.4 -- KEY (never shown to a scanner). on={a.on} prod={a.prod} seed={a.seed} "
                 f"n_controls={a.n_controls}; record_verdict = carried record on {a.on} / smx record on {a.prod}\n")
        fh.write("\t".join(cols) + "\n")
        for r in sorted(items, key=lambda r: r["scan_id"]):
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"items {sum(r['in_prep'] for r in items)} -> {a.out}/items_all.txt")


if __name__ == "__main__":
    main()
