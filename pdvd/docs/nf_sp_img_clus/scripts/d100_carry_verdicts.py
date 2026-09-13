#!/usr/bin/env python3
"""doc pdvd/100 -- carry a hand-scan record's VERDICTS (not its tags) from the arm it is keyed on to another arm by geometry.

    python3 d100_carry_verdicts.py --record R --base-arm p98vonq --arm p99rwon --out J > log

d99_match.py carries tags as well and needs every item's source prep; the latest record mixes sources it has no prep map
for (sw99).  The doc 100 threshold twin reads only verdicts (census_lib.judged / is_stopper), so this carries the item as
is, re-keyed: where the two arms' pctree files are byte-identical for the event the key is kept ("same pctree");
otherwise the key's cluster is matched with d99_match.match_cluster / status_of on the Bee clustering-global points (its
pre-registered thresholds) and carried only on status ok.  Each carried item gets carried_from, carry and target_arm.
Refuses an existing --out.
"""
import argparse, collections, glob, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
import d99_match as M

WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def one(job):
    evt, base, arm, cids = job
    pb = sorted(glob.glob(f"{WORK}/{evt}_{base}/pctree-evt*.tar.gz"))
    pa = sorted(glob.glob(f"{WORK}/{evt}_{arm}/pctree-evt*.tar.gz"))
    if pb and pa and sha(pb[0]) == sha(pa[0]):
        return {c: (c, "same pctree") for c in cids}
    ba, bn = M.load_bee(evt, base), M.load_bee(evt, arm)
    if ba is None or bn is None:
        return {c: (None, "no Bee dump") for c in cids}
    tree = cKDTree(bn[0])
    out = {}
    for c in cids:
        pts = ba[0][ba[1] == c]
        if len(pts) == 0:
            out[c] = (None, "no base points"); continue
        m = M.match_cluster(pts, bn[0], bn[1], tree)
        st = M.status_of(m)
        out[c] = (int(m["best"]), "geometry ok") if st == "ok" else (None, f"geometry {st}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--base-arm", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists")
    rec = json.load(open(a.record))
    by_evt = collections.defaultdict(list)
    for it in rec:
        evt, c = it["key"].split("/")
        by_evt[evt].append(int(c))
    with ProcessPoolExecutor(a.jobs) as ex:
        res = dict(zip(by_evt, ex.map(one, [(e, a.base_arm, a.arm, cs) for e, cs in by_evt.items()])))
    out, how, seen = [], collections.Counter(), collections.defaultdict(list)
    for it in rec:
        evt, c = it["key"].split("/")
        new, h = res[evt][int(c)]
        how[h] += 1
        if new is None:
            continue
        nk = f"{evt}/{new}"
        seen[nk].append(it["key"])
        o = dict(it)
        o.update(key=nk, carried_from=it["key"], carry=h, target_arm=a.arm, base_arm=a.base_arm,
                 source_record=os.path.basename(a.record))
        out.append(o)
    coll = {k: v for k, v in seen.items() if len(v) > 1}
    out = [o for o in out if o["key"] not in coll]
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"record {a.record}: {len(rec)} items; base {a.base_arm} -> {a.arm}")
    print("carry:", dict(how))
    print(f"collisions (two keys -> one cluster, dropped): {len(coll)} {sorted(coll.items())[:10]}")
    moved = [(o['carried_from'], o['key']) for o in out if o['carried_from'] != o['key']]
    print(f"carried {len(out)}; re-keyed {len(moved)}: {moved}")


if __name__ == "__main__":
    main()
