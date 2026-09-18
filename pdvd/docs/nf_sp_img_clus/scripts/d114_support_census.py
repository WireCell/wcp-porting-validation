#!/usr/bin/env python3
"""doc pdvd/114 sec 2-3 -- where is the fitted STM trajectory NOT supported by the image, and is each such stretch a
detour (the image is continuous, the trajectory left it) or a real gap jump (the image is absent)?

Population: every persisted STM fit record (T_rec_charge, one (cluster, pass)) of a dump arm (d111_run_arms.sh with
WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1).  Read-only.

Per fitted row, three views of "support":
  * 2-D, per plane: the cluster has a measured charge cell (T_proj_data, q > 0) within +-1 wire of the row's own 2-D
    coordinate (pu/pv/pw, pt) at its rounded slice -- the metric-free test doc 111 sec 4.1 validated the ridge metric
    against.  A plane is DEAD at the row when the nearest Steiner vertex within DEAD_R carries that plane's dead bit in
    the dumped test_good_point mask (0.6 cm / ch_range 1).  A row is UNSUPPORTED when it is off-charge in >= 1 live
    (not dead) plane.  Rows on charge in every plane but > THR off the ridge are WIDE (width or branch, not a ghost).
  * 3-D: distance to the nearest image point of the cluster (Bee clustering-global) and the canonical ridge offset
    (d111_stage_attrib.Ridge).
  * the seed under the row: the closest point of the persisted round-2 seed polyline, the vertices of the carrying seed
    segment (joined to the dumped steiner_pc), their class (3live / 1blank / 1dead / 2blank+), role and the carrying
    edge's source / base provenance; whether the round-2 seed was re-routed there by adjust_rough_path's crawl (> THR
    from the round-1 seed).

Stretches: maximal runs of >= MIN_ROWS consecutive unsupported rows of one record, closed by the nearest supported
rows before and after (the anchors).  Each stretch is classified:
  GAP      no chain of own-cluster image points (hops <= HOP cm, inside a ball of radius L/2 + BALL_PAD around the
           stretch) joins the anchors' image -> the image really is absent between them.
             GAP-dead      >= DEAD_SHARE of its rows have a dead plane
             GAP-short     image (any id) lies within NEAR of >= COVER_SHARE of the chord (a break shorter than the
                           stretch, or the trajectory beside a broken image)
             GAP-live      neither (live planes, empty: SP / imaging inefficiency)
  XID      an image chain joins the anchors but it belongs to ANOTHER Bee cluster id (the chord is covered by other-id
           image and not by own-id image): the fit is supported; the Bee layer re-labelled the fragments of a
           separated cluster.  A reference artefact, listed apart from GAP and DETOUR.
  DETOUR   an image chain joins the anchors (the image is continuous, any id) and the trajectory left it.
             D-crawl       >= 50 % of its rows sit on the crawl re-route
             D-bridge      >= 50 % of its rows are carried by a ctpc / mst base edge (the graph bridged a continuous image)
             D-blank-term  a 1blank (one zero, not dead, plane) TERMINAL is among its seed vertices
             D-blank-int   else a 1blank / 2blank+ interior is among them
             D-sbt         else >= 50 % carried by same-blob terminal edges
             D-3live       else (every seed vertex sees all three planes)
  FIT      the seed under >= FIT_SHARE of its rows is itself supported (its carrying vertices are 3live / 1dead and the
           seed point is within THR of the image): the fit made the deviation.
For DETOUR stretches the seed's own sub-route between the anchors' vertices is compared with the cheapest route on the
dumped steiner_graph that never leaves the ridge (d112 walk_masked / route_stats), when both anchors join a vertex.

Outputs:
  <out>.txt            the census
  <out>_rows.tsv.gz    every unsupported or WIDE row with its views
  <out>_stretches.tsv  one line per stretch (the case catalogue's source)

Usage:
  d114_support_census.py --det pdhd --arm d114hbase [--logd /home/xqian/tmp/d114/arm_d114hbase] [--jobs 12] \
      --out figs/114_support_pdhd
"""
import argparse, collections, glob, gzip, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import uproot
from multiprocessing import Pool
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import d111s_common as C
from d111s_graph_census import edge_offfrac
from d112_terminal_planes import vertex_class, vertex_role, CLASSES, ROLES
from d112_case_anatomy import walk_masked, route_stats
A = C.A

THR = 1.0          # cm, ridge / image distance
DEAD_R = 1.5       # cm, nearest vertex for the dead bits
MIN_ROWS = 2
HOP = 2.0          # cm, image chain hop (the blob-sampled image has nearest-neighbour gaps up to ~2 cm)
BALL_PAD = 5.0     # cm
NEAR = 1.5         # cm, chord coverage
COVER_SHARE = 0.5
DEAD_SHARE = 0.5
FIT_SHARE = 0.8
PLANE = "UVW"
CLASS_ORDER = ["GAP-dead", "GAP-short", "GAP-live", "XID", "D-crawl", "D-bridge", "D-blank-term", "D-blank-int", "D-sbt",
               "D-3live", "FIT"]


def load_cells(f, det):
    pd = f["T_proj_data"].arrays(library="np")
    cells = collections.defaultdict(lambda: [set(), set(), set()])
    for cid, ch, ts, q in zip(pd["cluster_id"][0], pd["channel"][0], pd["time_slice"][0], pd["charge"][0]):
        ch = np.asarray(ch).astype(int); ts = np.asarray(ts).astype(int); m = np.asarray(q) > 0
        pl = np.searchsorted(np.array(A.BASE[det][1:]), ch, side="right")
        for p in range(3):
            mm = m & (pl == p)
            cells[int(cid) // 10][p].update(zip(ch[mm].tolist(), ts[mm].tolist()))
    return cells


def image_chain(P, a, b, L):
    """is there a chain of points of P with hops <= HOP joining the point nearest a to the point nearest b, using only
    points within L/2 + BALL_PAD of the chord midpoint?"""
    if len(P) < 2:
        return False
    mid = 0.5 * (a + b)
    sel = np.where(np.linalg.norm(P - mid, axis=1) <= 0.5 * L + BALL_PAD)[0]
    if len(sel) < 2:
        return False
    Q = P[sel]
    tree = cKDTree(Q)
    ia = int(np.argmin(np.linalg.norm(Q - a, axis=1))); ib = int(np.argmin(np.linalg.norm(Q - b, axis=1)))
    if np.linalg.norm(Q[ia] - a) > NEAR or np.linalg.norm(Q[ib] - b) > NEAR:
        return False
    pairs = tree.query_pairs(HOP, output_type="ndarray")
    if len(pairs) == 0:
        return ia == ib
    M = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(Q), len(Q)))
    _, lab = connected_components(M, directed=False)
    return lab[ia] == lab[ib]


def chord_cover(tree, a, b, step=0.5):
    """share of chord samples with a tree point within NEAR."""
    L = np.linalg.norm(b - a)
    n = max(1, int(round(L / step)))
    f = (np.arange(n) + 0.5) / n
    X = a + f[:, None] * (b - a)
    if tree is None:
        return 0.0
    d, _ = tree.query(X)
    return float((d <= NEAR).mean())


def one(args):
    det, arm, pre, logd = args
    d = f"{C.IMG}/{det}/work/{pre}_{arm}"
    lg = f"{logd}/evt_{pre}.log.gz"
    acc = collections.Counter()
    rows_out, str_out = [], []
    if not (os.path.exists(lg) and os.path.exists(f"{d}/tracking-stm.root")):
        acc["events_missing"] += 1
        return pre, acc, rows_out, str_out
    tr = C.parse_trace(lg)
    f = uproot.open(f"{d}/tracking-stm.root")
    t = f["T_rec_charge"].arrays(library="np")
    cells = load_cells(f, det)
    img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    if img is None:
        acc["events_no_image"] += 1
        return pre, acc, rows_out, str_out
    Pall, cid_all, q_all = img
    tree_all = cKDTree(Pall) if len(Pall) else None
    acc["events"] += 1
    blocks = [b for b in tr["blocks"] if "seed" in b["st"]]
    cache = {}
    for (cl, ps) in sorted(set(zip(t["cluster_id"].tolist(), t["pass"].tolist()))):
        m = (t["cluster_id"] == cl) & (t["pass"] == ps)
        idx = np.where(m)[0]
        F = np.c_[t["x"][m], t["y"][m], t["z"][m]]
        n = len(F)
        acc["records"] += 1
        acc["rows"] += n
        b2 = None
        for b in blocks:
            if b["cl"] == cl and b["rnd"] == "r2" and "final" in b["st"] and len(b["st"]["final"]) == n \
                    and np.max(np.abs(b["st"]["final"] - F)) < 0.01:
                b2 = b
        if b2 is None:
            acc["records_unmatched"] += 1
            continue
        if cl not in cache:
            mi = cid_all == cl
            Pc = Pall[mi]
            R = A.Ridge(Pc, q_all[mi])
            sb = C.graph_for(tr, cl, b2["order"])
            info = None
            if sb is not None and sb.nv:
                cls, zero, dead = vertex_class(sb)
                info = dict(cls=cls, zero=zero, dead=dead, role=vertex_role(sb), vtree=cKDTree(sb.V),
                            voff=R.offset(sb.V) if R.ok else np.full(sb.nv, np.nan), offfrac=None)
                dm61 = np.c_[[(sb.V_m61 >= 0) & (((sb.V_m61 >> (3 + p)) & 1) > 0) for p in range(3)]].T
                info["dead61"] = dm61
            cache[cl] = (Pc, cKDTree(Pc) if len(Pc) else None, cKDTree(Pall[~mi]) if (~mi).sum() else None, R, sb, info)
        Pc, tree_c, tree_o, R, sb, info = cache[cl]
        if not R.ok or tree_c is None:
            acc["records_no_image"] += 1
            continue
        acc["records_used"] += 1
        # ---- 2-D support
        on = np.zeros((n, 3), bool)
        for p, key in enumerate(("pu", "pv", "pw")):
            on[:, p] = A.onq_pm1(cells[cl][p], t[key][m], t["pt"][m])
        deadp = np.zeros((n, 3), bool)
        if info is not None:
            dv, jv = info["vtree"].query(F)
            near = dv <= DEAD_R
            deadp[near] = info["dead61"][jv[near]]
        offq = (~on & ~deadp).any(axis=1)
        noff = (~on & ~deadp).sum(axis=1)
        # ---- 3-D
        d3, _ = tree_c.query(F)                      # own-id image
        d3a, _ = tree_all.query(F) if tree_all is not None else (np.full(n, np.inf), None)   # any cluster's image
        dr = R.offset(F)
        # the ridge is the own-id image's; where that image is farther than RLOC the offset is just the distance to
        # it, which is not a ridge measurement (Bee re-labels the fragments of a separated cluster under new ids)
        dr_eff = np.where(d3 <= A.RLOC, dr, np.nan)
        # unsupported = off a live plane's charge AND away from the image in 3-D (> THR from the own ridge where it is
        # defined, or > THR from every image point of ANY cluster); off-charge rows within THR of both are sub-cm
        # wander (JITTER), reported but not catalogued
        unsup = offq & ((np.nan_to_num(dr_eff, nan=0.0) > THR) | (d3a > THR))
        jitter = offq & ~unsup
        wide = ~offq & (np.nan_to_num(dr_eff, nan=0.0) > THR)
        # ---- seed under the row
        seed = b2["st"]["seed"]
        Q, seglen = A.closest_on_polyline(seed, F)
        sd3, _ = tree_c.query(Q)
        # carrying segment: index of the seed point starting the segment holding Q
        segi = np.zeros(n, int)
        if len(seed) > 1:
            st = cKDTree(seed)
            _, js = st.query(Q, k=2)
            segi = np.min(js, axis=1)
            segi = np.clip(segi, 0, len(seed) - 2)
        r1s = [b for b in blocks if b["cl"] == cl and b["dir"] == b2["dir"] and b["rnd"] == "r1" and b["order"] < b2["order"]]
        crawl = np.zeros(n, bool)
        if r1s:
            Q1, _ = A.closest_on_polyline(r1s[-1]["st"]["seed"], Q)
            crawl = np.linalg.norm(Q1 - Q, axis=1) > THR
        j = C.join_vertices(sb, seed) if sb is not None else np.full(len(seed), -1)
        vcls = np.full((n, 2), -1); vrole = np.full((n, 2), -1); esrc = np.full(n, -1); ebase = np.full(n, -1)
        seed_sup = np.zeros(n, bool)
        if info is not None and len(seed) > 1:
            va, vb = j[segi], j[np.minimum(segi + 1, len(seed) - 1)]
            for k in range(n):
                a_, b_ = int(va[k]), int(vb[k])
                if a_ >= 0:
                    vcls[k, 0] = info["cls"][a_]; vrole[k, 0] = info["role"][a_]
                if b_ >= 0:
                    vcls[k, 1] = info["cls"][b_]; vrole[k, 1] = info["role"][b_]
                if a_ >= 0 and b_ >= 0 and a_ != b_:
                    e = sb.edge_index.get((min(a_, b_), max(a_, b_)))
                    if e is not None:
                        esrc[k] = sb.E_src[e]; ebase[k] = sb.E_base[e]
            seed_sup = (sd3 <= THR) & np.all((vcls == 0) | (vcls == 2) | (vcls == -1), axis=1) & (vcls >= 0).any(axis=1)
        # ---- counters
        step = np.r_[np.linalg.norm(np.diff(F, axis=0), axis=1), 0]
        acc["rows_used"] += n; acc["len"] += float(step.sum())
        acc["rows_unsup"] += int(unsup.sum()); acc["len_unsup"] += float(step[unsup].sum())
        acc["rows_wide"] += int(wide.sum()); acc["rows_jitter"] += int(jitter.sum())
        dev = dr > THR
        acc["rows_dev"] += int(dev.sum()); acc["dev_unsup"] += int((dev & unsup).sum()); acc["nondev_unsup"] += int((~dev & unsup).sum())
        acc["rows_d3gt1"] += int((d3a > THR).sum()); acc["d3gt1_unsup"] += int(((d3a > THR) & unsup).sum())
        acc["rows_xid"] += int(((d3 > THR) & (d3a <= THR)).sum())
        for p in range(3):
            acc[f"plane{p}_off"] += int((~on[:, p]).sum()); acc[f"plane{p}_dead"] += int(deadp[:, p].sum())
        # ---- rows out
        for k in np.where(unsup | wide)[0]:
            rows_out.append((pre, cl, ps, int(idx[k]), f"{F[k,0]:.2f}", f"{F[k,1]:.2f}", f"{F[k,2]:.2f}",
                             "".join("1" if on[k, p] else ("D" if deadp[k, p] else "0") for p in range(3)),
                             int(noff[k]), int(unsup[k]), int(wide[k]), f"{d3[k]:.2f}", f"{d3a[k]:.2f}", f"{dr[k]:.2f}", f"{sd3[k]:.2f}",
                             f"{seglen[k]:.2f}", int(crawl[k]), CLASSES[vcls[k, 0]] if vcls[k, 0] >= 0 else "-",
                             CLASSES[vcls[k, 1]] if vcls[k, 1] >= 0 else "-", ROLES[vrole[k, 0]] if vrole[k, 0] >= 0 else "-",
                             ROLES[vrole[k, 1]] if vrole[k, 1] >= 0 else "-",
                             C.SRC_NAME[esrc[k]] if esrc[k] >= 0 else "-", C.BASE_NAME[ebase[k]] if ebase[k] >= 0 else "-",
                             int(seed_sup[k])))
        # ---- stretches
        runs, cur = [], []
        for k in range(n):
            if unsup[k]:
                cur.append(k)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        for run in runs:
            if len(run) < MIN_ROWS:
                acc["stretch_short"] += 1
                continue
            ia = max([k for k in range(run[0]) if not unsup[k]], default=run[0])
            ib = min([k for k in range(run[-1] + 1, n) if not unsup[k]], default=run[-1])
            a, b = F[ia], F[ib]
            L = float(np.linalg.norm(b - a))
            length = float(step[run[0]:run[-1] + 1].sum())
            rr = np.array(run)
            chain = image_chain(Pall, a, b, L) if ia != run[0] and ib != run[-1] else False
            cov_own = chord_cover(tree_c, a, b)
            cov_oth = chord_cover(tree_o, a, b)
            cov_any = chord_cover(tree_all, a, b)
            dead_sh = float(deadp[rr].any(axis=1).mean())
            crawl_sh = float(crawl[rr].mean())
            bridge_sh = float(np.isin(ebase[rr], (C.BASE_CODE["ctpc"], C.BASE_CODE["mst"])).mean())
            sbt_sh = float((esrc[rr] == C.SRC_CODE["sbt"]).mean())
            fit_sh = float(seed_sup[rr].mean())
            vs = set()
            for k in rr:
                for kk in (segi[k], min(segi[k] + 1, len(seed) - 1)):
                    if j[kk] >= 0:
                        vs.add(int(j[kk]))
            vs = sorted(vs)
            has_bt = any(info["cls"][v] == 1 and info["role"][v] == 0 for v in vs) if info else False
            has_bi = any(info["cls"][v] in (1, 3) and info["role"][v] != 0 for v in vs) if info else False
            has_blank_any = any(info["cls"][v] in (1, 3) for v in vs) if info else False
            open_ended = ia == run[0] or ib == run[-1]
            if fit_sh >= FIT_SHARE:
                klass = "FIT"
            elif not chain:
                klass = "GAP-dead" if dead_sh >= DEAD_SHARE else ("GAP-short" if cov_any >= COVER_SHARE else "GAP-live")
            elif cov_own < COVER_SHARE and cov_oth >= COVER_SHARE:
                klass = "XID"
            else:
                if crawl_sh >= 0.5:
                    klass = "D-crawl"
                elif bridge_sh >= 0.5:
                    klass = "D-bridge"
                elif has_bt:
                    klass = "D-blank-term"
                elif has_bi:
                    klass = "D-blank-int"
                elif sbt_sh >= 0.5:
                    klass = "D-sbt"
                else:
                    klass = "D-3live"
            acc[("stretch", klass)] += 1; acc[("stretch_rows", klass)] += len(run); acc[("stretch_len", klass)] += length
            acc[("stretch_blank_any", klass)] += int(has_blank_any)
            acc[("stretch_open", klass)] += int(open_ended)
            # route comparison for detours
            ratio = ""; onimg_len = ""; seed_len = ""
            if klass.startswith("D-") and info is not None and j[segi[run[0]]] >= 0 and j[min(segi[run[-1]] + 1, len(seed) - 1)] >= 0:
                if info["offfrac"] is None:
                    info["offfrac"] = edge_offfrac(sb, R)
                va_ = int(j[max(0, segi[run[0]])]); vb_ = int(j[min(segi[run[-1]] + 1, len(seed) - 1)])
                # walk back to the anchors' vertices
                ka = max(0, segi[ia]) if ia != run[0] else max(0, segi[run[0]])
                kb = min(segi[ib] + 1, len(seed) - 1) if ib != run[-1] else min(segi[run[-1]] + 1, len(seed) - 1)
                if j[ka] >= 0 and j[kb] >= 0 and j[ka] != j[kb]:
                    va_, vb_ = int(j[ka]), int(j[kb])
                    sub = []
                    for kk in range(ka, kb + 1):
                        if j[kk] >= 0 and (not sub or sub[-1] != j[kk]):
                            sub.append(int(j[kk]))
                    s_seed = route_stats(sb, R, sub) if len(sub) > 1 else None
                    s_on = route_stats(sb, R, walk_masked(sb, va_, vb_, sb.E_w, info["offfrac"] == 0))
                    if s_seed is not None:
                        seed_len = f"{s_seed['length']:.2f}"
                        acc[("route_n", klass)] += 1
                        if s_on is not None:
                            ratio = f"{s_on['cost'] / max(s_seed['cost'], 1e-9):.3f}"
                            onimg_len = f"{s_on['length']:.2f}"
                            acc[("route_onimg", klass)] += 1
                            acc[("route_ratio_sum", klass)] += s_on['cost'] / max(s_seed['cost'], 1e-9)
                        else:
                            acc[("route_noalt", klass)] += 1
            mid = F[rr].mean(axis=0)
            str_out.append((det, pre, cl, ps, int(idx[run[0]]), int(idx[run[-1]]), len(run), f"{length:.2f}", f"{L:.2f}",
                            f"{mid[0]:.1f}", f"{mid[1]:.1f}", f"{mid[2]:.1f}", f"{d3a[rr].max():.2f}", f"{np.nanmax(dr_eff[rr]) if np.isfinite(dr_eff[rr]).any() else np.nan:.2f}",
                            f"{noff[rr].max()}", klass, int(chain), f"{cov_own:.2f}", f"{cov_oth:.2f}", f"{cov_any:.2f}", f"{dead_sh:.2f}",
                            f"{crawl_sh:.2f}", f"{bridge_sh:.2f}", f"{sbt_sh:.2f}", f"{fit_sh:.2f}",
                            "/".join(f"{CLASSES[info['cls'][v]]}:{ROLES[info['role'][v]][0]}" for v in vs) if info else "-",
                            int(open_ended), seed_len, onimg_len, ratio))
    return pre, acc, rows_out, str_out


ROW_HDR = ("event cluster pass row x y z on_uvw noff unsup wide d_img d_img_any d_ridge seed_d_img seed_seglen crawl vcls_a vcls_b "
           "vrole_a vrole_b edge_src edge_base seed_sup").split()
STR_HDR = ("det event cluster pass row_a row_b nrows length_cm chord_cm mx my mz max_d_img max_d_ridge max_noff class chain "
           "cov_own cov_other cov_any dead_share crawl_share bridge_share sbt_share fit_share seed_vertices open_ended "
           "seed_route_cm onimage_route_cm onimage_cost_ratio").split()


def pct(a, b):
    return f"{100.0 * a / b:.1f} %" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--logd"); ap.add_argument("--events"); ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logd = a.logd or f"/home/xqian/tmp/d114/arm_{a.arm}"
    evs = a.events.split(",") if a.events else sorted(os.path.basename(p)[:-len(f"_{a.arm}")]
                                                      for p in glob.glob(f"{C.IMG}/{a.det}/work/*_{a.arm}"))
    with Pool(a.jobs) as pool:
        res = pool.map(one, [(a.det, a.arm, e, logd) for e in evs], chunksize=1)
    acc = collections.Counter()
    rows, strs = [], []
    for pre, ac, r, s in res:
        acc.update(ac); rows.extend(r); strs.extend(s)
    with gzip.open(f"{a.out}_rows.tsv.gz", "wt") as fh:
        fh.write("\t".join(ROW_HDR) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    with open(f"{a.out}_stretches.tsv", "w") as fh:
        fh.write("\t".join(STR_HDR) + "\n")
        for r in strs:
            fh.write("\t".join(str(x) for x in r) + "\n")
    L = [f"# doc pdvd/114 support census: det={a.det} arm={a.arm} events={acc['events']} (missing {acc['events_missing']}, "
         f"no image {acc['events_no_image']}); records {acc['records']} (unmatched {acc['records_unmatched']}, no image "
         f"{acc['records_no_image']}, used {acc['records_used']})",
         f"# thresholds: THR {THR} cm, HOP {HOP}, BALL_PAD {BALL_PAD}, NEAR {NEAR}, COVER_SHARE {COVER_SHARE}, DEAD_SHARE {DEAD_SHARE}, "
         f"FIT_SHARE {FIT_SHARE}, MIN_ROWS {MIN_ROWS}, DEAD_R {DEAD_R}", ""]
    L.append("## rows")
    L.append(f"- rows {acc['rows_used']} ({acc['len'] / 100:.1f} m); UNSUPPORTED (off-charge in >= 1 live plane and > {THR} cm "
             f"from the ridge or from every image point) {acc['rows_unsup']} ({pct(acc['rows_unsup'], acc['rows_used'])}, "
             f"{acc['len_unsup'] / 100:.1f} m); JITTER (off-charge but within {THR} cm of both) {acc['rows_jitter']} "
             f"({pct(acc['rows_jitter'], acc['rows_used'])}); WIDE (on charge everywhere, > {THR} cm off the ridge) "
             f"{acc['rows_wide']} ({pct(acc['rows_wide'], acc['rows_used'])})")
    L.append(f"- per plane U/V/W: off-charge {acc['plane0_off']} / {acc['plane1_off']} / {acc['plane2_off']}; dead at the row "
             f"{acc['plane0_dead']} / {acc['plane1_dead']} / {acc['plane2_dead']}")
    L.append(f"- check (doc 111 sec 4.1): rows > {THR} cm off the ridge {acc['rows_dev']} ({pct(acc['rows_dev'], acc['rows_used'])}); "
             f"unsupported among them {pct(acc['dev_unsup'], acc['rows_dev'])}, among the rest {pct(acc['nondev_unsup'], acc['rows_used'] - acc['rows_dev'])}")
    L.append(f"- rows > {THR} cm from every image point of ANY cluster {acc['rows_d3gt1']} ({pct(acc['rows_d3gt1'], acc['rows_used'])}); "
             f"unsupported among them {pct(acc['d3gt1_unsup'], acc['rows_d3gt1'])}; rows > {THR} cm from the own-id image but "
             f"within {THR} cm of another id's {acc['rows_xid']} ({pct(acc['rows_xid'], acc['rows_used'])})")
    L.append("")
    L.append("## stretches (>= 2 consecutive unsupported rows)")
    tot = sum(acc[("stretch", k)] for k in CLASS_ORDER)
    totr = sum(acc[("stretch_rows", k)] for k in CLASS_ORDER)
    totl = sum(acc[("stretch_len", k)] for k in CLASS_ORDER)
    L.append(f"- stretches {tot} (single-row runs not counted: {acc['stretch_short']}); rows {totr}; length {totl / 100:.1f} m")
    L.append("")
    L.append("| class | stretches | share | rows | length (m) | length share | with a blank-plane seed vertex | open-ended | detour routes with an on-image alternative | mean on-image / seed cost |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k in CLASS_ORDER:
        ns = acc[("stretch", k)]
        if ns == 0:
            continue
        rn = acc[("route_n", k)]; ro = acc[("route_onimg", k)]
        rr_ = f"{acc[('route_ratio_sum', k)] / ro:.2f}" if ro else "-"
        ra_ = f"{ro} of {rn}" if rn else "-"
        L.append(f"| {k} | {ns} | {pct(ns, tot)} | {acc[('stretch_rows', k)]} | {acc[('stretch_len', k)] / 100:.1f} | "
                 f"{pct(acc[('stretch_len', k)], totl)} | {pct(acc[('stretch_blank_any', k)], ns)} | {acc[('stretch_open', k)]} | "
                 f"{ra_} | {rr_} |")
    grp = collections.Counter()
    for k in CLASS_ORDER:
        g = "GAP" if k.startswith("GAP") else ("DETOUR" if k.startswith("D-") else k)
        grp[(g, "n")] += acc[("stretch", k)]; grp[(g, "len")] += acc[("stretch_len", k)]
    L.append("")
    L.append("| group | stretches | share | length (m) | length share |")
    L.append("|---|---|---|---|---|")
    for g in ("GAP", "DETOUR", "FIT", "XID"):
        L.append(f"| {g} | {grp[(g, 'n')]} | {pct(grp[(g, 'n')], tot)} | {grp[(g, 'len')] / 100:.1f} | {pct(grp[(g, 'len')], totl)} |")
    open(f"{a.out}.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
