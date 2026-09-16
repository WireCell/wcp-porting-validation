#!/usr/bin/env python3
"""doc pdvd/111 round 2, A3 + A4 -- where in the Steiner stage a seed-born deviation is made.

Inputs, per event of a dump arm (d111_run_arms.sh with WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1; same production
output as the round-1 trace arm, gated separately):
  * the trace $LOGD/evt_<run6>_<evt>.log.gz: the Steiner dump (retiled cloud STGR, steiner_pc STGV, steiner_graph STGE),
    the rough-path walks STMRP, the STM path stages STMPATH (seed, final);
  * tracking-stm.root T_rec_charge (the persisted fits) and mabc-pr.zip clustering-global (the image);
  * the round-1 deviated-row table figs/111_attrib_<det>_rows.tsv.gz (origin seed / org1 = seed-born).

A3 checks (printed first):
  join    -- every seed point of every STM pass is a dumped steiner_pc vertex of its cluster (6-digit print precision);
  replay  -- a scipy Dijkstra on the dumped graph between the STMRP rough-walk ends has the same cost as the C++ round-1
             seed (relative 1e-9); path identity is reported (equal-cost ties may differ).

A4, per seed-born deviated row (seed location Q = closest point of the persisted round-2 seed to the final row):
  * layer at the ridge foot of Q (the image axis point nearest Q), in order:
      no_image_at_foot   no image point within 1 cm of the foot (the ridge metric itself is thin there)
      retile_gap         the image is there, no RETILED point within 1 cm  -> the retiler lost it
      tree_omission      a retiled point is there, no Steiner vertex within 1 cm  -> create_steiner_tree left it out
      path_choice        a Steiner vertex is there  -> the graph holds the ridge, the Dijkstra walk did not take it
  * the carrying steiner_graph edge (the seed segment containing Q): source (path / connect / sbt), base provenance
    (closely_same / closely_other / ctpc / mst), length, and its 2-D support (share of samples ok in 3 planes, the sgp
    unsupported share) at both dump radii;
  * its two vertices: terminal / extreme / path interior, and whether each lies > 1 cm off the ridge or > 1 cm from any
    image point (a fabricated vertex);
  * the image oracle: Dijkstra between the same walk ends with every edge re-weighted w * (1 + 100 * f_off), f_off = the
    share of the edge's straight segment > 1 cm off the ridge.  alt_on = the oracle route is within 1 cm of the ridge at
    this location; the walk-level cost ratio oracle/chosen under the ORIGINAL weights says how much the graph charges
    for staying on the image.
  * painted vertices: a retiled point with a zero charge in some plane sits on a cell the retiler PAINTED
    (ImproveCluster_1::hack_activity_improved writes 1e-3 around its own Dijkstra paths) or on a dead channel (same
    sentinel); the carrying edge's ends are classed painted / real, and so is the whole retiled cloud by ridge offset.
Null floors: ridge offsets of the image points, retiled points, terminals and path interiors of the same clusters, and
the share of on-ridge terminals (< 0.5 cm) failing each 2-D support test at the vertex.
Edge discrimination: all round-2 seed edges >= 0.5 cm, split by f_off (0 vs >= 0.5): each support test's bad share.

Usage: d111s_graph_census.py --det pdhd --arm d111hst [--events a,b] [--jobs 12] --out figs/111s_graph_pdhd
       -> <out>.txt, <out>_rows.tsv.gz
"""
import argparse, collections, glob, gzip, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np
import uproot
from scipy.spatial import cKDTree
import d111s_common as C
A = C.A

THR = 1.0
NEAR_MAX = 5.0
FOOT_R = 1.0


def ridge_foot(R, X):
    """the ridge point nearest X: its nearest voxel's local axis, kept within +-1 cm of that voxel's local centroid."""
    _, j = R.tree.query(X)
    V = X - R.cen[j]
    t = np.clip(np.sum(V * R.ax[j], axis=1), -1.0, 1.0)
    return R.cen[j] + t[:, None] * R.ax[j]


def near_share(tree, a, b, r=1.0, step=0.5):
    """share of samples of segment a-b within r cm of a point of tree."""
    if tree is None:
        return np.nan
    L = np.linalg.norm(b - a)
    n = max(1, int(round(L / step)))
    f = (np.arange(n) + 0.5) / n
    P = a + f[:, None] * (b - a)
    return float(np.mean(tree.query(P)[0] <= r))


def edge_offfrac(sb, R, step=0.5):
    """share of each edge's straight segment > THR off the ridge."""
    a = sb.V[sb.E_s]; b = sb.V[sb.E_t]
    L = np.linalg.norm(b - a, axis=1)
    n = np.maximum(1, np.round(L / step)).astype(int)
    idx = np.repeat(np.arange(len(L)), n)
    start = np.r_[0, np.cumsum(n)[:-1]]
    k = np.arange(n.sum()) - np.repeat(start, n)
    f = (k + 0.5) / n[idx]
    P = a[idx] + f[:, None] * (b[idx] - a[idx])
    off = R.offset(P) > THR
    return np.bincount(idx, weights=off, minlength=len(L)) / n


def mask_ok3(m):
    """vertex mask -> all three planes live or dead."""
    ok = [((m >> p) & 1) | ((m >> (p + 3)) & 1) for p in range(3)]
    return (m >= 0) & (ok[0] > 0) & (ok[1] > 0) & (ok[2] > 0)


def support_bad(S, mode):
    n = np.maximum(S[:, 0], 1)
    if mode == "3pl":
        return 1.0 - S[:, 1] / n
    if mode == "2pl":
        return 1.0 - S[:, 2] / n
    # sgp: (unsupported + 0.25 dead-only) / n; unsupported = no plane ok, dead-only = ok but no live plane
    return ((S[:, 0] - S[:, 3]) + 0.25 * (S[:, 3] - S[:, 4])) / n


def dedupe(j):
    out = [int(j[0])]
    for x in j[1:]:
        if int(x) != out[-1]:
            out.append(int(x))
    return out


def one(args):
    det, arm, logd, pre, rows = args
    acc = collections.Counter()
    lists = collections.defaultdict(list)
    out_rows = []
    d = f"{C.IMG}/{det}/work/{pre}_{arm}"
    lg = f"{logd}/evt_{pre}.log.gz"
    if not (os.path.exists(lg) and os.path.exists(f"{d}/tracking-stm.root")):
        acc["events_missing"] += 1
        return acc, lists, out_rows
    tr = C.parse_trace(lg)
    acc["events"] += 1
    acc["lines_repaired"] += tr["repaired"]
    if not tr["stg"]:
        acc["events_without_dump"] += 1
        return acc, lists, out_rows
    t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(["x", "y", "z", "cluster_id", "pass"], library="np")
    img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    blocks = [b for b in tr["blocks"] if b["cl"] >= 0 and "seed" in b["st"]]

    # ---- A3 join over every block
    for b in blocks:
        sb = C.graph_for(tr, b["cl"], b["order"])
        if sb is None:
            acc["join_blocks_nograph"] += 1
            continue
        j = C.join_vertices(sb, b["st"]["seed"])
        acc["join_points"] += len(j); acc["join_missing"] += int((j < 0).sum())

    want = collections.defaultdict(dict)
    for r in rows:
        want[(r["cl"], r["pass"])][r["row"]] = r
    r2 = collections.defaultdict(list)
    for b in blocks:
        if b["rnd"] == "r2" and "final" in b["st"]:
            r2[b["cl"]].append(b)
    done_clusters = set()
    for (cl, ps) in sorted(set(zip(t["cluster_id"].tolist(), t["pass"].tolist()))):
        m = (t["cluster_id"] == cl) & (t["pass"] == ps)
        F = np.c_[t["x"][m], t["y"][m], t["z"][m]]
        match = None
        for b in r2.get(cl, []):
            if len(b["st"]["final"]) == len(F) and np.max(np.abs(b["st"]["final"] - F)) < 0.01:
                match = b
        acc["records"] += 1
        if match is None:
            acc["records_unmatched"] += 1
            continue
        sb = C.graph_for(tr, cl, match["order"])
        mi = img[1] == cl
        if sb is None or mi.sum() < 5:
            acc["records_nograph_or_noimage"] += 1
            continue
        R = A.Ridge(img[0][mi], img[2][mi])
        if not R.ok:
            acc["records_noridge"] += 1
            continue
        r1s = [b for b in blocks if b["cl"] == cl and b["dir"] == match["dir"] and b["rnd"] == "r1" and b["order"] < match["order"]]
        r1 = r1s[-1] if r1s else None
        prev_order = r1["order"] if r1 is not None else -1
        rough = [r for r in C.walks_for(tr, cl, prev_order) if r[2] == "rough"]
        crawl = [r for r in C.walks_for(tr, cl, match["order"], after=prev_order) if r[2] in ("crawl1", "crawl2")]
        # ---- A3 replay on the rough walk behind round 1
        w_or = None
        if r1 is not None and rough:
            rw = rough[-1]
            j1 = C.join_vertices(sb, r1["st"]["seed"])
            if (j1 >= 0).all() and len(j1) >= 2:
                p, c = sb.walk(rw[3], rw[4])
                cc = sb.path_cost(dedupe(j1))
                acc["replay_walks"] += 1
                if p is None:
                    acc["replay_unreachable"] += 1
                else:
                    acc["replay_cost_equal"] += int(abs(cc - c) <= 1e-9 * max(1.0, c))
                    acc["replay_path_identical"] += int(dedupe(j1) == p)
        # ---- image oracle for the persisted seed's walks
        off_e = edge_offfrac(sb, R)
        w_or = sb.E_w * (1.0 + 100.0 * off_e)
        walks = crawl[-2:] if len(crawl) >= 2 else (rough[-1:] if rough else [])
        orP, chosen_c, oracle_c, unreach = [], 0.0, 0.0, False
        for wk in walks:
            p_ch, c_ch = sb.walk(wk[3], wk[4])
            p_or, _ = sb.walk(wk[3], wk[4], w_or)
            if p_ch is None or p_or is None:
                unreach = True
                continue
            chosen_c += c_ch; oracle_c += sb.path_cost(p_or)
            orP.append(sb.V[p_or])
        ratio = oracle_c / chosen_c if chosen_c > 0 else np.nan
        orPoly = np.concatenate(orP) if orP else np.zeros((0, 3))
        acc["records_used"] += 1
        # ---- per-cluster null floors (once per cluster)
        vo = R.offset(sb.V)
        if cl not in done_clusters:
            done_clusters.add(cl)
            lists["null_image"].extend(R.offset(img[0][mi]).tolist())
            lists["null_retiled"].extend(R.offset(sb.R).tolist())
            lists["null_term"].extend(vo[sb.V_term].tolist())
            lists["null_inter"].extend(vo[~sb.V_term & ~sb.V_ext].tolist())
            rq0 = (sb.R_q < 1.0).any(axis=1)
            ro = R.offset(sb.R)
            for lab, mm in (("on", ro <= THR), ("off", (ro > THR) & (ro <= 5)), ("far", ro > 5)):
                acc[("painted_ret", lab)] += int((rq0 & mm).sum()); acc[("n_ret", lab)] += int(mm.sum())
            vq0 = (sb.R_q[sb.V_old] < 1.0).any(axis=1)
            for lab, mm in (("on", vo <= THR), ("off", (vo > THR) & (vo <= 5)), ("far", vo > 5)):
                acc[("painted_v", lab)] += int((vq0 & mm).sum()); acc[("n_v", lab)] += int(mm.sum())
            on = sb.V_term & (vo < 0.5)
            acc["null_term_on"] += int(on.sum())
            acc["null_term_on_fail3_02"] += int((on & ~mask_ok3(sb.V_m02)).sum())
            acc["null_term_on_fail3_61"] += int((on & ~mask_ok3(sb.V_m61)).sum())
        # ---- edge discrimination on this record's persisted seed edges
        j2 = C.join_vertices(sb, match["st"]["seed"])
        if (j2 < 0).any() or len(j2) < 2:
            acc["records_seed_unjoined"] += 1
            continue
        seed = match["st"]["seed"]
        for a_, b_ in zip(j2[:-1], j2[1:]):
            if a_ == b_:
                continue
            k = sb.edge_index.get((min(a_, b_), max(a_, b_)))
            if k is None or sb.E_len[k] < 0.5:
                continue
            cls_ = "off" if off_e[k] >= 0.5 else ("on" if off_e[k] == 0 else None)
            if cls_ is None:
                continue
            acc[("disc_n", cls_)] += 1
            for mode in ("3pl", "2pl", "sgp"):
                for rad, S in (("02", sb.S02), ("61", sb.S61)):
                    bad = support_bad(S[k:k + 1], mode)[0]
                    acc[("disc_bad", cls_, mode, rad)] += bad
                    acc[("disc_bad20", cls_, mode, rad)] += int(bad >= 0.2)
            acc[("disc_len", cls_)] += sb.E_len[k]
        # ---- per deviated row
        rr = want.get((cl, ps), {})
        if not rr:
            continue
        itree = cKDTree(img[0][mi]); rtree = cKDTree(sb.R) if len(sb.R) else None; vtree = cKDTree(sb.V)
        for i, r in sorted(rr.items()):
            if i >= len(F):
                continue
            Q, _ = A.closest_on_polyline(seed, F[i:i + 1])
            Q = Q[0]
            dseed = float(R.offset(Q[None])[0])
            dist = "far" if dseed > NEAR_MAX else "near"
            foot = ridge_foot(R, Q[None])[0]
            if itree.query(foot)[0] > FOOT_R:
                layer = "no_image_at_foot"
            elif rtree is None or rtree.query(foot)[0] > FOOT_R:
                layer = "retile_gap"
            elif vtree.query(foot)[0] > FOOT_R:
                layer = "tree_omission"
            else:
                layer = "path_choice"
            # the retiled cloud around the foot: is it centred on the image ridge?
            ret_cen = "n/a"
            if rtree is not None:
                nbr = rtree.query_ball_point(foot, 2.0)
                if len(nbr) >= 3:
                    wq = np.clip(sb.R_q[nbr].mean(axis=1), 1, None)
                    cen = np.average(sb.R[nbr], axis=0, weights=wq)
                    ret_cen = "off" if R.offset(cen[None])[0] > THR else "on"
            jj = int(np.argmin(np.linalg.norm(seed - Q, axis=1)))
            best = None
            for kk in range(max(0, jj - 1), min(len(seed) - 1, jj + 1)):
                A_, B_ = seed[kk], seed[kk + 1]; AB = B_ - A_; L2 = AB @ AB
                tt = np.clip(((Q - A_) @ AB) / L2, 0, 1) if L2 > 0 else 0.0
                dq = np.linalg.norm(A_ + tt * AB - Q)
                if best is None or dq < best[0]:
                    best = (dq, kk)
            kk = best[1] if best else 0
            va, vb = int(j2[kk]), int(j2[min(kk + 1, len(j2) - 1)])
            k = sb.edge_index.get((min(va, vb), max(va, vb)))
            if k is None:
                src = base = "no_edge"; elen = np.nan; b3 = bsg = np.nan
            else:
                src = C.SRC_NAME[sb.E_src[k]]; base = C.BASE_NAME[sb.E_base[k]]; elen = sb.E_len[k]
                b3 = support_bad(sb.S61[k:k + 1], "3pl")[0] if sb.S61[k, 0] > 0 else np.nan
                bsg = support_bad(sb.S02[k:k + 1], "sgp")[0] if sb.S02[k, 0] > 0 else np.nan
            painted_ends = int(sum(bool((sb.R_q[sb.V_old[v]] < 1.0).any()) for v in (va, vb)))
            ends = []
            for v in (va, vb):
                kind = "extreme" if sb.V_ext[v] else ("terminal" if sb.V_term[v] else "interior")
                offr = vo[v] > THR
                fab = itree.query(sb.V[v])[0] > THR
                ends.append((kind, offr, fab))
            offends = [e for e in ends if e[1]]
            if not offends:
                endc = "both_on_ridge"
            else:
                kinds = set(e[0] for e in offends)
                endc = "off_" + "+".join(sorted(kinds))
            fabc = any(e[2] for e in ends)
            chord_img = near_share(itree, sb.V[va], sb.V[vb])
            chord_ret = near_share(rtree, sb.V[va], sb.V[vb])
            if len(orPoly):
                Qo, _ = A.closest_on_polyline(orPoly, F[i:i + 1])
                alt_on = bool(R.offset(Qo)[0] <= THR)
            else:
                alt_on = None
            if unreach or not np.isfinite(ratio):
                rb = "unreach"
            elif not alt_on:
                rb = "no_on_image_route"
            elif ratio <= 1.05:
                rb = "alt<=1.05"
            elif ratio <= 1.2:
                rb = "alt<=1.2"
            elif ratio <= 1.5:
                rb = "alt<=1.5"
            else:
                rb = "alt>1.5"
            crawl_row = r["sub"] == "crawl_reroute"
            iso = r["angle"] >= 75
            acc[("n", dist)] += 1
            acc[("layer", dist, layer)] += 1
            acc[("src", dist, src)] += 1
            acc[("base", dist, base)] += 1
            acc[("srcbase", dist, src, base)] += 1
            acc[("elen", dist, "<=3" if not (elen > 3) else ">3")] += 1
            acc[("ends", dist, endc)] += 1
            acc[("fab", dist, fabc)] += 1
            acc[("alt", dist, rb)] += 1
            acc[("layer_alt", dist, layer, rb)] += 1
            acc[("crawl", dist, crawl_row)] += 1
            acc[("iso_layer", dist, iso, layer)] += 1
            acc[("iso_n", dist, iso)] += 1
            acc[("retcen", dist, layer, ret_cen)] += 1
            acc[("painted_ends", dist, painted_ends)] += 1
            acc[("painted_ends_layer", dist, layer, painted_ends)] += 1
            cimg = "img>=0.5" if chord_img >= 0.5 else "img<0.5"
            cret = "ret>=0.5" if chord_ret >= 0.5 else "ret<0.5"
            acc[("chord", dist, cimg, cret)] += 1
            lists[("b3", dist, layer)].append(b3); lists[("bsg", dist, layer)].append(bsg)
            out_rows.append(f"{pre}\t{cl}\t{ps}\t{i}\t{dseed:.2f}\t{dist}\t{layer}\t{src}\t{base}\t{elen:.2f}\t"
                            f"{b3:.3f}\t{bsg:.3f}\t{endc}\t{int(fabc)}\t{rb}\t{ratio:.3f}\t{int(crawl_row)}\t{r['angle']:.1f}\t"
                            f"{ret_cen}\t{chord_img:.2f}\t{chord_ret:.2f}\t{painted_ends}")
    return acc, lists, out_rows


def pct(a, b):
    return f"{100.0 * a / b:.1f} %" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--logd")
    ap.add_argument("--events", default="")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logd = a.logd or f"/home/xqian/tmp/d111/arm_{a.arm}"
    rows = collections.defaultdict(list)
    with gzip.open(f"{C.DOC}/figs/111_attrib_{a.det}_rows.tsv.gz", "rt") as g:
        hdr = next(g).rstrip("\n").split("\t")
        ix = {h: k for k, h in enumerate(hdr)}
        for line in g:
            f = line.rstrip("\n").split("\t")
            if f[ix["origin"]] in ("seed", "org1"):
                rows[f[0]].append(dict(cl=int(f[1]), pass_=int(f[2]), row=int(f[3]), angle=float(f[ix["angle"]]),
                                       sub=f[ix["sub"]]))
    for pre in rows:
        for r in rows[pre]:
            r["pass"] = r.pop("pass_")
    evs = sorted(os.path.basename(p)[:-len(a.arm) - 1] for p in glob.glob(f"{C.IMG}/{a.det}/work/*_{a.arm}"))
    if a.events:
        evs = [e for e in evs if e in a.events.split(",")]
    jobs = [(a.det, a.arm, logd, pre, rows.get(pre, [])) for pre in evs]
    acc = collections.Counter(); lists = collections.defaultdict(list); out_rows = []
    with Pool(a.jobs) as pool:
        for ac, ls, orows in pool.imap_unordered(one, jobs):
            acc.update(ac); out_rows.extend(orows)
            for k, v in ls.items():
                lists[k].extend(v)
    L = []
    P = L.append
    P(f"# doc pdvd/111 round 2 Steiner-level census: det={a.det} arm={a.arm} events={acc['events']} (missing {acc['events_missing']}, without dump {acc['events_without_dump']})")
    P(f"# THR={THR} cm, near <= {NEAR_MAX} cm, foot radius {FOOT_R} cm; rows = round-1 table origin seed/org1")
    P("")
    P("## A3 checks")
    P(f"dump lines rejoined after a spliced stderr line: {acc['lines_repaired']}")
    P(f"join: seed points {acc['join_points']}, not a dumped vertex {acc['join_missing']}; blocks without a dump {acc['join_blocks_nograph']}")
    P(f"replay (round-1 rough walks): {acc['replay_walks']} walks, cost equal {acc['replay_cost_equal']}, path identical {acc['replay_path_identical']}, unreachable {acc['replay_unreachable']}")
    P(f"records {acc['records']}, unmatched {acc['records_unmatched']}, no graph/image {acc['records_nograph_or_noimage']}, no ridge {acc['records_noridge']}, seed unjoined {acc['records_seed_unjoined']}, used {acc['records_used']}")
    P("")
    P("## null floors: ridge offset > 1 cm / p50 (cm)")
    for k in ("null_image", "null_retiled", "null_term", "null_inter"):
        x = np.array(lists[k]); x = x[np.isfinite(x)]
        if len(x):
            P(f"| {k[5:]} | n {len(x)} | > 1 cm {100*np.mean(x > 1):.1f} % | p50 {np.median(x):.2f} | p90 {np.percentile(x, 90):.2f} |")
    P(f"on-ridge terminals (< 0.5 cm) failing the 3-plane test at the vertex: r0.2/ch0 {pct(acc['null_term_on_fail3_02'], acc['null_term_on'])}, "
      f"r0.6/ch1 {pct(acc['null_term_on_fail3_61'], acc['null_term_on'])} (n {acc['null_term_on']})")
    P("")
    P("## painted or dead cells (a retiled point with a zero-charge plane), by ridge offset: <= 1 / 1-5 / > 5 cm")
    P(f"retiled points: {pct(acc[('painted_ret','on')], acc[('n_ret','on')])} / {pct(acc[('painted_ret','off')], acc[('n_ret','off')])} / {pct(acc[('painted_ret','far')], acc[('n_ret','far')])}"
      f" (n {acc[('n_ret','on')]} / {acc[('n_ret','off')]} / {acc[('n_ret','far')]})")
    P(f"Steiner vertices: {pct(acc[('painted_v','on')], acc[('n_v','on')])} / {pct(acc[('painted_v','off')], acc[('n_v','off')])} / {pct(acc[('painted_v','far')], acc[('n_v','far')])}"
      f" (n {acc[('n_v','on')]} / {acc[('n_v','off')]} / {acc[('n_v','far')]})")
    P("")
    P("## edge discrimination: persisted-seed edges >= 0.5 cm, on the ridge (f_off = 0) vs off (f_off >= 0.5)")
    P(f"edges on {acc[('disc_n','on')]} ({acc[('disc_len','on')]:.0f} cm), off {acc[('disc_n','off')]} ({acc[('disc_len','off')]:.0f} cm)")
    P("| test | radius | mean bad on | mean bad off | bad >= 0.2 on | bad >= 0.2 off |")
    P("|---|---|---|---|---|---|")
    for mode in ("3pl", "2pl", "sgp"):
        for rad in ("02", "61"):
            no, nf = acc[("disc_n", "on")], acc[("disc_n", "off")]
            P(f"| {mode} | {rad} | {acc[('disc_bad','on',mode,rad)]/max(no,1):.3f} | {acc[('disc_bad','off',mode,rad)]/max(nf,1):.3f} | "
              f"{pct(acc[('disc_bad20','on',mode,rad)], no)} | {pct(acc[('disc_bad20','off',mode,rad)], nf)} |")
    for dist in ("near", "far"):
        n = acc[("n", dist)]
        P("")
        P(f"## {dist} seed-born deviated rows: {n}")
        P("| layer at the ridge foot | rows | share | iso (>= 75 deg) share of iso rows |")
        P("|---|---|---|---|")
        niso = acc[("iso_n", dist, True)]
        for lay in ("no_image_at_foot", "retile_gap", "tree_omission", "path_choice"):
            P(f"| {lay} | {acc[('layer', dist, lay)]} | {pct(acc[('layer', dist, lay)], n)} | {pct(acc[('iso_layer', dist, True, lay)], niso)} |")
        P("")
        P("| carrying edge source x base | rows | share |")
        P("|---|---|---|")
        keys = sorted([k for k in acc if k[0] == "srcbase" and k[1] == dist], key=lambda k: -acc[k])
        for k in keys:
            P(f"| {k[2]} / {k[3]} | {acc[k]} | {pct(acc[k], n)} |")
        P(f"carrying edge > 3 cm: {pct(acc[('elen', dist, '>3')], n)}; crawl re-route rows: {pct(acc[('crawl', dist, True)], n)}")
        P("")
        P("| carrying edge interior: image within 1 cm / retiled within 1 cm (share of samples >= 0.5) | rows | share |")
        P("|---|---|---|")
        for ci in ("img>=0.5", "img<0.5"):
            for cr in ("ret>=0.5", "ret<0.5"):
                P(f"| {ci} / {cr} | {acc[('chord', dist, ci, cr)]} | {pct(acc[('chord', dist, ci, cr)], n)} |")
        P("")
        P("| layer | retiled centroid within 2 cm of the foot on the ridge | off the ridge (> 1 cm) | n/a |")
        P("|---|---|---|---|")
        for lay in ("no_image_at_foot", "retile_gap", "tree_omission", "path_choice"):
            P(f"| {lay} | {acc[('retcen', dist, lay, 'on')]} | {acc[('retcen', dist, lay, 'off')]} | {acc[('retcen', dist, lay, 'n/a')]} |")
        P("")
        P("| carrying edge ends | rows | share |")
        P("|---|---|---|")
        for k in sorted([k for k in acc if k[0] == "ends" and k[1] == dist], key=lambda k: -acc[k]):
            P(f"| {k[2]} | {acc[k]} | {pct(acc[k], n)} |")
        P(f"an end vertex > 1 cm from every image point (fabricated): {pct(acc[('fab', dist, True)], n)}")
        P(f"carrying edge ends on painted-or-dead cells: 0 ends {pct(acc[('painted_ends', dist, 0)], n)}, 1 end {pct(acc[('painted_ends', dist, 1)], n)}, 2 ends {pct(acc[('painted_ends', dist, 2)], n)}")
        P("| layer | 0 painted ends | 1 | 2 |")
        P("|---|---|---|---|")
        for lay in ("no_image_at_foot", "retile_gap", "tree_omission", "path_choice"):
            P(f"| {lay} | " + " | ".join(str(acc[('painted_ends_layer', dist, lay, k)]) for k in (0, 1, 2)) + " |")
        P("")
        P("| image oracle at the location | rows | share |")
        P("|---|---|---|")
        for rb in ("alt<=1.05", "alt<=1.2", "alt<=1.5", "alt>1.5", "no_on_image_route", "unreach"):
            P(f"| {rb} | {acc[('alt', dist, rb)]} | {pct(acc[('alt', dist, rb)], n)} |")
        P("")
        P("| layer x oracle | " + " | ".join(("alt<=1.05", "alt<=1.2", "alt<=1.5", "alt>1.5", "no_on_image_route")) + " |")
        P("|---|---|---|---|---|---|")
        for lay in ("no_image_at_foot", "retile_gap", "tree_omission", "path_choice"):
            P(f"| {lay} | " + " | ".join(str(acc[("layer_alt", dist, lay, rb)]) for rb in ("alt<=1.05", "alt<=1.2", "alt<=1.5", "alt>1.5", "no_on_image_route")) + " |")
        P("")
        P("| layer | carrying-edge 3-plane bad (r0.6/ch1) p50 | sgp bad (r0.2/ch0) p50 |")
        P("|---|---|---|")
        for lay in ("no_image_at_foot", "retile_gap", "tree_omission", "path_choice"):
            b3 = np.array(lists[("b3", dist, lay)]); bs = np.array(lists[("bsg", dist, lay)])
            b3 = b3[np.isfinite(b3)]; bs = bs[np.isfinite(bs)]
            P(f"| {lay} | {np.median(b3):.2f} (n {len(b3)}) | {np.median(bs):.2f} |" if len(b3) and len(bs) else f"| {lay} | n/a | n/a |")
    txt = "\n".join(L) + "\n"
    open(a.out + ".txt", "w").write(txt)
    with gzip.open(a.out + "_rows.tsv.gz", "wt") as g:
        g.write("event\tcluster\tpass\trow\td_seed\tdist\tlayer\tedge_src\tedge_base\tedge_len\tedge_bad3_r061\tedge_badsgp_r020\t"
                "ends\tfab_end\toracle\toracle_cost_ratio\tcrawl\tangle\tretiled_centroid\tchord_image_share\tchord_retiled_share\tpainted_ends\n")
        g.write("\n".join(sorted(out_rows)) + ("\n" if out_rows else ""))
    print(txt)


if __name__ == "__main__":
    main()
