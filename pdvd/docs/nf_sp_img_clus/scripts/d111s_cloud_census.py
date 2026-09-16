#!/usr/bin/env python3
"""doc pdvd/111 round 2, A1 -- the Steiner cloud at every seed-born deviation, on EVERY cluster.

Round 1's sub-classes (d111_stage_attrib.py) read the Steiner cloud from the Bee `steiner_graph-global` layer, which PDHD
and PDVD scope to STM-tagged clusters (doc 109), so a third of the near rows were `cloud_na`.  The calib dump
(calib-pr-evt*.json, PrDisplayDump) carries `steiner` -- steiner_pc x/y/z and flag_terminal -- for every cluster that has
one.  No new run: inputs are the doc-111 trace arm (STMPATH seed/final), its tracking-stm.root T_rec_charge, its Bee image,
its calib dump, and the round-1 deviated-row table figs/111_attrib_<det>_rows.tsv.gz (origin seed or org1).

Per deviated row (the seed location Q = closest point of the persisted round-2 seed to the final row):
  near (seed 1-5 cm off the ridge) / far (> 5 cm);
  no_steiner_within_1.5cm | steiner_on_ridge_nearby (a Steiner point within 1.5 cm of Q lies < 1 cm from the ridge;
    split < 0.5 / 0.5-1 cm) | steiner_off_ridge_image_backed (the nearby Steiner points are all >= 1 cm off the ridge
    but one lies < 1 cm from an image point) | steiner_off_ridge_no_image;
  the carrying seed segment's two vertices: terminal or path interior, and whether either is > 1 cm off the ridge;
  round 1's own sub-class rule (long_edge > 3 cm first; cloud_has_ridge = a Steiner point < 0.5 cm from the ridge within
    1.5 cm of Q, else cloud_off; crawl_reroute from the round-1 table) re-applied with the full cloud;
  null floors: ridge offsets of the image points, the terminals and the path interiors of the same clusters.

Usage: d111s_cloud_census.py <pdhd|pdvd> <arm: d111htr|d111vtr> > figs/111s_cloud_<det>.txt
"""
import collections, glob, gzip, json, os, sys, zipfile
from multiprocessing import Pool
import numpy as np, uproot
from scipy.spatial import cKDTree
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts")
sys.dont_write_bytecode = True
import d111_stage_attrib as S

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
F = IMG + "/pdvd/docs/nf_sp_img_clus/figs"

def parse(path):
    blocks, cur, fill = [], {}, None
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("STMPATHN "):
                f = line.split(); tag, stage = f[1], f[2]
                if stage not in ("seed", "final"):
                    fill = None; continue
                if stage == "seed" or tag not in cur:
                    p = tag.split("/")
                    b = dict(cl=int(p[0]) if p[0].lstrip("-").isdigit() else -1, dir=p[1] if len(p)>1 else "?", rnd=p[2] if len(p)>2 else "?", st={}, order=len(blocks)); blocks.append(b); cur[tag] = b
                b = cur[tag]; b["st"][stage] = []; fill = (b, stage, tag)
            elif line.startswith("STMPATH ") and fill is not None:
                f = line.split()
                if f[1] == fill[2] and f[2] == fill[1]:
                    fill[0]["st"][fill[1]].append((float(f[4]), float(f[5]), float(f[6])))
    for b in blocks:
        b["st"] = {k: np.array(v).reshape(-1, 3) for k, v in b["st"].items()}
    return blocks

def one(args):
    det, arm, pre, rows = args
    d = f"{IMG}/{det}/work/{pre}_{arm}"
    acc = collections.Counter(); pop = collections.defaultdict(list)
    try:
        blocks = parse(f"/home/xqian/tmp/d111/arm_{arm}/evt_{pre}.log.gz")
        t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(["x", "y", "z", "cluster_id", "pass"], library="np")
        img = S.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
        cal = json.load(open(glob.glob(f"{d}/calib-pr-evt*.json")[0]))
    except Exception as e:
        acc["event_fail"] += 1; return acc, pop
    st = collections.defaultdict(list); term = collections.defaultdict(list)
    for s in cal.get("steiner", []):
        st[s["cluster_id"]].append(np.c_[s["x"], s["y"], s["z"]]); term[s["cluster_id"]].append(np.array(s["flag_terminal"], bool))
    r2 = collections.defaultdict(list)
    for b in blocks:
        if b["rnd"] == "r2" and "final" in b["st"]:
            r2[b["cl"]].append(b)
    want = collections.defaultdict(dict)
    for r in rows:
        want[(r[1], r[2])][r[3]] = r
    for (cl, ps), rr in want.items():
        m = (t["cluster_id"] == cl) & (t["pass"] == ps)
        Fr = np.c_[t["x"][m], t["y"][m], t["z"][m]]
        match = None
        for b in r2.get(cl, []):
            if len(b["st"]["final"]) == len(Fr) and np.max(np.abs(b["st"]["final"] - Fr)) < 0.01:
                match = b
        if match is None or cl not in st:
            acc["rec_unmatched_or_nocloud"] += len(rr); continue
        mi = img[1] == cl
        R = S.Ridge(img[0][mi], img[2][mi]); itree = cKDTree(img[0][mi])
        SP = np.concatenate(st[cl]); TF = np.concatenate(term[cl]); stree = cKDTree(SP)
        seed = match["st"]["seed"]
        so = R.offset(SP)
        for i, r in rr.items():
            Q, sl = S.closest_on_polyline(seed, Fr[i:i+1]); Q = Q[0]; sl = sl[0]
            dseed = R.offset(Q[None])[0]
            dist = "far" if dseed > 5 else "near"
            nb = stree.query_ball_point(Q, 1.5)
            if len(nb) == 0:
                cls = "no_steiner_within_1.5cm"
                acc[("nost_img", dist, bool(itree.query(Q)[0] < 1.0))] += 1
            elif np.min(so[nb]) < 1.0:
                acc[("onridge_lt05", dist, bool(np.min(so[nb]) < 0.5))] += 1
                cls = "steiner_on_ridge_nearby"
            else:
                dimg = itree.query(SP[nb])[0]
                cls = "steiner_off_ridge_image_backed" if np.min(dimg) < 1.0 else "steiner_off_ridge_no_image"
            le = "edge>3" if sl > 3 else "edge<=3"
            # bracketing seed vertices
            j = int(np.argmin(np.linalg.norm(seed - Q, axis=1)))
            cand = [k for k in (j - 1, j, j + 1) if 0 <= k < len(seed)]
            # the segment containing Q
            best = None
            for k in range(max(0, j - 1), min(len(seed) - 1, j + 1)):
                A, B = seed[k], seed[k + 1]; AB = B - A; L2 = AB @ AB
                tt = np.clip(((Q - A) @ AB) / L2, 0, 1) if L2 > 0 else 0
                dd = np.linalg.norm(A + tt * AB - Q)
                if best is None or dd < best[0]:
                    best = (dd, k)
            k = best[1]
            ends = []
            for v in (seed[k], seed[min(k + 1, len(seed) - 1)]):
                dv, jv = stree.query(v)
                isterm = bool(TF[jv]) if dv < 0.01 else None
                ends.append((isterm, R.offset(v[None])[0]))
            offend = [e for e in ends if e[1] > 1.0]
            if not offend:
                endc = "both_ends_on_ridge"
            elif any(e[0] for e in offend):
                endc = "an_off_ridge_end_is_TERMINAL"
            elif any(e[0] is None for e in offend):
                endc = "off_ridge_end_unmatched"
            else:
                endc = "off_ridge_end_is_interior"
            sub = r[5]
            if sub == "crawl_reroute":
                r1c = "crawl_reroute"
            elif sl > 3:
                r1c = "long_edge"
            elif len(nb) and np.min(so[nb]) < 0.5:
                r1c = "cloud_has_ridge"
            else:
                r1c = "cloud_off"
            acc[("r1rule", dist, r1c)] += 1
            key = (dist, sub == "crawl_reroute")
            acc[("cls", dist, cls)] += 1
            acc[("edge", dist, le)] += 1
            acc[("cls_edge", dist, cls, le)] += 1
            acc[("ends", dist, endc)] += 1
            acc[("tot", dist)] += 1
            acc[("crawl", dist, sub == "crawl_reroute")] += 1
        # population: steiner offsets for this cluster
        pop["term"].extend(so[TF].tolist()); pop["inter"].extend(so[~TF].tolist())
        pop["image"].extend(R.offset(img[0][mi]).tolist())
    return acc, pop

def main(det, arm):
    rows = collections.defaultdict(list)
    with gzip.open(f"{F}/111_attrib_{det}_rows.tsv.gz", "rt") as g:
        next(g)
        for line in g:
            f = line.rstrip("\n").split("\t")
            if f[17] in ("seed", "org1"):
                rows[f[0]].append((f[0], int(f[1]), int(f[2]), int(f[3]), f[17], f[18]))
    jobs = [(det, arm, pre, rr) for pre, rr in sorted(rows.items())]
    tot = collections.Counter(); pop = collections.defaultdict(list)
    with Pool(16) as p:
        for a, pp in p.imap_unordered(one, jobs):
            tot.update(a)
            for k, v in pp.items(): pop[k].extend(v)
    print(f"=== {det} {arm}: events {len(jobs)}, event_fail {tot['event_fail']}, rows unmatched/nocloud {tot['rec_unmatched_or_nocloud']}")
    for dist in ("near", "far"):
        n = tot[("tot", dist)]
        print(f"-- {dist}: rows {n}; crawl_reroute {tot[('crawl', dist, True)]}")
        for c in ("no_steiner_within_1.5cm", "steiner_on_ridge_nearby", "steiner_off_ridge_image_backed", "steiner_off_ridge_no_image"):
            v = tot[("cls", dist, c)]
            print(f"   {c:34s} {v:7d} {100*v/max(n,1):5.1f} %   edge>3: {tot[('cls_edge', dist, c, 'edge>3')]:6d}  edge<=3: {tot[('cls_edge', dist, c, 'edge<=3')]:6d}")
        for c in ("both_ends_on_ridge", "an_off_ridge_end_is_TERMINAL", "off_ridge_end_is_interior", "off_ridge_end_unmatched"):
            v = tot[("ends", dist, c)]
            print(f"   ends: {c:30s} {v:7d} {100*v/max(n,1):5.1f} %")
        print(f"   no_steiner rows with image within 1 cm of the seed: {tot[('nost_img', dist, True)]} / without {tot[('nost_img', dist, False)]}")
        print(f"   steiner_on_ridge (<1 cm) rows where the nearest on-ridge Steiner is < 0.5 cm: {tot[('onridge_lt05', dist, True)]} / 0.5-1 cm {tot[('onridge_lt05', dist, False)]}")
    print("-- round 1's sub-class rule re-applied on the full calib cloud (near rows; round 1 had cloud_na for untagged clusters)")
    n = tot[("tot", "near")]
    for c in ("long_edge", "crawl_reroute", "cloud_has_ridge", "cloud_off"):
        v = tot[("r1rule", "near", c)]
        print(f"   {c:20s} {v:7d} {100*v/max(n,1):5.1f} %")
    for k in ("image", "term", "inter"):
        x = np.array(pop[k])
        x = x[np.isfinite(x)]
        print(f"population steiner {k}: n {len(x)}, offset p50 {np.median(x):.2f} p90 {np.percentile(x,90):.2f} >1cm {100*np.mean(x>1):.1f} % >3cm {100*np.mean(x>3):.1f} %")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
