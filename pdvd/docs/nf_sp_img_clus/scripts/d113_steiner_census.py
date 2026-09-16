#!/usr/bin/env python3
"""doc pdvd/113 -- the Steiner-level census of ONE dump arm (d111_run_arms.sh with WCT_STM_PATH_DEBUG=1
WCT_STEINER_GRAPH_DUMP=1), the same numbers for every retile_mode arm.

Per event (read-only): the trace $LOGD/evt_<run6>_<evt>.log.gz, tracking-stm.root T_rec_charge (which clusters carry an
STM fit), mabc-pr.zip clustering-global (the image), wct_pr_*.log (CreateSteinerGraph WARN lines, RETILEMODE census).

  S1 seed  -- every distinct STM rough walk (STMRP "rough", once per (cluster, from, to)) of a cluster with an STM fit is
              replayed with scipy Dijkstra on that cluster's dumped steiner_graph with the dumped weights -- the C++
              seed (doc 111 r2 A3 replay check: cost equal on 1151/1151 walks) -- and scored against the image ridge
              with d111s_replay.score: seed length, length > 1 cm off the ridge ("off"), > 5 cm ("far"), the
              wiggle share and the image-charge coverage.  Exactly the "base:0" row of d111s_replay.py.
  spots    -- the seed's max ridge offset inside the doc-111 spot windows (h1, h2, v1-v3).
  graph    -- for the Steiner block each such walk ran on (one per cluster, its first walk): retiled cloud size,
              steiner_pc vertices / edges, non-extreme terminals by d112 plane class (3live / 1blank / 1dead / 2blank+)
              and how many sit > 1 cm off the ridge, bridge edges (base ctpc or mst: the gap jumps) count and length.
  whole    -- over every STGC block of the event (all clusters the stage built): blocks, retiled points, vertices.
  warn     -- CreateSteinerGraph "has no points" / "produced no steiner_graph" / "empty steiner_pc" lines.

Usage: d113_steiner_census.py --det pdhd --arm d113hbase [--logd DIR] [--jobs 12] --out figs/113_steiner_d113hbase
       -> <out>.txt and <out>.json (per-event totals, so a comparison can restrict to common events)
"""
import argparse, collections, glob, json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np
import uproot
from scipy.spatial import cKDTree
import d111s_common as C
from d111s_replay import score, SPOTS
from d112_terminal_planes import vertex_class, vertex_role, CLASSES
A = C.A

WARNS = {"no_points": "has no points, skipping", "no_graph": "produced no steiner_graph",
         "empty_pc": "empty steiner_pc"}


def one(args):
    det, arm, logd, pre = args
    acc = collections.Counter()
    spots = collections.defaultdict(list)
    d = f"{C.IMG}/{det}/work/{pre}_{arm}"
    lg = f"{logd}/evt_{pre}.log.gz"
    if not (os.path.exists(lg) and os.path.exists(f"{d}/tracking-stm.root")):
        return pre, None, None
    for p in glob.glob(f"{d}/wct_pr_*.log"):
        with open(p, errors="replace") as fh:
            for line in fh:
                for k, pat in WARNS.items():
                    if pat in line:
                        acc[f"warn_{k}"] += 1
    tr = C.parse_trace(lg)
    for sb in tr["stg"]:
        acc["whole_blocks"] += 1
        acc["whole_nret"] += sb.header.get("nret", 0)
        acc["whole_nsv"] += sb.nv
        acc["whole_nse"] += len(sb.E_s)
    t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(["cluster_id"], library="np")
    have = set(t["cluster_id"].tolist())
    img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    seen, cache, graphed = set(), {}, set()
    for rp in tr["rp"]:
        order, cl, kind, src, dst, npath = rp
        if kind != "rough" or cl not in have or (cl, src, dst) in seen:
            continue
        seen.add((cl, src, dst))
        sb = C.graph_for(tr, cl, order)
        mi = img[1] == cl
        if sb is None or mi.sum() < 5 or max(src, dst) >= sb.nv:
            acc["walks_skipped"] += 1
            continue
        if cl not in cache:
            R = A.Ridge(img[0][mi], img[2][mi])
            q = np.clip(img[2][mi], 0, None)
            cache[cl] = (R, cKDTree(img[0][mi]), q, float(q.sum()))
        R, itree, qimg, qtot = cache[cl]
        if not R.ok:
            acc["walks_skipped"] += 1
            continue
        if cl not in graphed:                     # the graph census: once per cluster, on its first walk's block
            graphed.add(cl)
            acc["g_clusters"] += 1
            acc["g_nret"] += sb.header.get("nret", 0)
            acc["g_nsv"] += sb.nv
            acc["g_nse"] += len(sb.E_s)
            cls, _, _ = vertex_class(sb)
            role = vertex_role(sb)
            off = R.offset(sb.V) > 1.0
            for ci, cname in enumerate(CLASSES):
                m = (role == 0) & (cls == ci)
                acc[f"term_{cname}"] += int(m.sum())
                acc[f"term_{cname}_off"] += int((m & off).sum())
            acc["vert_off"] += int(off.sum())
            br = (sb.E_base == C.BASE_CODE["ctpc"]) | (sb.E_base == C.BASE_CODE["mst"])
            acc["bridge_edges"] += int(br.sum())
            acc["bridge_len"] += float(sb.E_len[br].sum())
            acc["long_edges_5cm"] += int((sb.E_len >= 5.0).sum())
        p, c = sb.walk(src, dst)
        acc["walks"] += 1
        if p is None:
            acc["unreach"] += 1
            continue
        s = score(sb.V[p], R, itree, qimg, qtot)
        for kk in ("L", "off", "far", "wig", "nint", "cov", "qtot"):
            acc[kk] += s[kk]
        for name, (sdet, sev, scl, click, win) in SPOTS.items():
            if sdet == det and sev == pre and scl == cl:
                X, _ = C.polyline_samples(sb.V[p], 0.3)
                if len(X):
                    m = np.linalg.norm(X - np.array(click), axis=1) <= win
                    if m.any():
                        spots[name].append(float(np.max(R.offset(X[m]))))
    return pre, dict(acc), dict(spots)


def summary(E, spots, events=None):
    tot = collections.Counter()
    for pre, acc in E.items():
        if events is None or pre in events:
            tot.update(acc)
    sh = lambda a, b: tot[a] / tot[b] if tot[b] else float("nan")
    nterm = sum(tot[f"term_{c}"] for c in CLASSES)
    nterm_off = sum(tot[f"term_{c}_off"] for c in CLASSES)
    return dict(
        events=len(E) if events is None else len([e for e in E if e in events]),
        walks=tot["walks"], walks_skipped=tot["walks_skipped"], unreach=tot["unreach"],
        seed_len_m=tot["L"] / 100.0, off=sh("off", "L"), far=sh("far", "L"), wig=sh("wig", "nint"), cov=sh("cov", "qtot"),
        g_clusters=tot["g_clusters"], g_nret=tot["g_nret"], g_nsv=tot["g_nsv"], g_nse=tot["g_nse"],
        terminals=nterm, terminals_off=nterm_off,
        term_class={c: tot[f"term_{c}"] for c in CLASSES}, term_class_off={c: tot[f"term_{c}_off"] for c in CLASSES},
        bridge_edges=tot["bridge_edges"], bridge_len_m=tot["bridge_len"] / 100.0, long_edges_5cm=tot["long_edges_5cm"],
        whole_blocks=tot["whole_blocks"], whole_nret=tot["whole_nret"], whole_nsv=tot["whole_nsv"], whole_nse=tot["whole_nse"],
        warn={k: tot[f"warn_{k}"] for k in WARNS}, spots=spots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--logd")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logd = a.logd or f"/home/xqian/tmp/d113/arm_{a.arm}"
    evs = sorted(os.path.basename(p)[:-len(a.arm) - 1] for p in glob.glob(f"{C.IMG}/{a.det}/work/*_{a.arm}"))
    E, S, missing = {}, collections.defaultdict(list), []
    with Pool(a.jobs) as pool:
        for pre, acc, sp in pool.imap_unordered(one, [(a.det, a.arm, logd, e) for e in evs]):
            if acc is None:
                missing.append(pre)
                continue
            E[pre] = acc
            for k, v in sp.items():
                S[k].extend(v)
    s = summary(E, dict(S))
    L = [f"# doc pdvd/113 Steiner census: det={a.det} arm={a.arm} events {len(E)} (missing {len(missing)}: {sorted(missing)})",
         f"S1 seed (rough walks replayed on the dumped graph): walks {s['walks']} (skipped {s['walks_skipped']}, unreachable "
         f"{s['unreach']}), length {s['seed_len_m']:.1f} m, > 1 cm off {100*s['off']:.2f} %, > 5 cm off {100*s['far']:.2f} %, "
         f"wiggle {100*s['wig']:.2f} %, image coverage {100*s['cov']:.2f} %",
         f"graphs of STM clusters: {s['g_clusters']}; retiled points {s['g_nret']}; vertices {s['g_nsv']}; edges {s['g_nse']}; "
         f"bridge edges (ctpc/mst) {s['bridge_edges']} ({s['bridge_len_m']:.1f} m); edges >= 5 cm {s['long_edges_5cm']}",
         f"non-extreme terminals {s['terminals']}, > 1 cm off {s['terminals_off']} "
         f"({100*s['terminals_off']/max(1,s['terminals']):.1f} %)"]
    for c in CLASSES:
        n, o = s["term_class"][c], s["term_class_off"][c]
        L.append(f"  {c:8s} {n:8d} ({100*n/max(1,s['terminals']):5.1f} % of terminals), off {o:7d} "
                 f"({100*o/max(1,n):5.1f} % of the class; {100*o/max(1,s['terminals_off']):5.1f} % of off terminals)")
    L.append(f"whole event: Steiner blocks {s['whole_blocks']}, retiled points {s['whole_nret']}, vertices {s['whole_nsv']}, "
             f"edges {s['whole_nse']}")
    L.append(f"CreateSteinerGraph WARN lines: {s['warn']}")
    L.append("spots, seed max ridge offset in the window per walk (cm): " +
             "; ".join(f"{k} " + "/".join(f"{v:.2f}" for v in sorted(vals)) for k, vals in sorted(S.items())))
    txt = "\n".join(L) + "\n"
    open(a.out + ".txt", "w").write(txt)
    json.dump(dict(det=a.det, arm=a.arm, missing=sorted(missing), events=E, spots=dict(S), summary=s),
              open(a.out + ".json", "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
