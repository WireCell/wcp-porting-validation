#!/usr/bin/env python3
"""doc pdvd/115 sec 2 -- offline sizing of the charge-aware base-graph pricing on the base-graph dump.

Reads the WCT_STEINER_GRAPH_DUMP + WCT_STEINER_BASE_DUMP logs of a knob-OFF arm (STGA: every base-graph edge with its
geometric weight; STGR: every retiled point with its charges and blob; STGV/STGE: the C++ tree; STMRP: the STM rough
walks) and, per cluster with a persisted STM record and per distinct rough walk, replays create_enhanced_steiner_graph
(SteinerGrapher.cxx) from the dumped terminal set under a family of base-graph pricings:

    w' = w * (1 + alpha * 0.5 * (nz(s) + nz(t)))      nz = planes at charge 0 at the endpoint (STGR %.0f => q < 0.5)

  1. Voronoi   scipy multi-source Dijkstra from the terminals on the priced graph (min_only, predecessors);
  2. bridges   per unordered terminal pair the FIRST minimum of dist[s] + w' + dist[t] over the base edges in the
               dumped (boost) order -- the C++ keeps the first strict minimum;
  3. back-walk both endpoints of every selected bridge along the predecessors to their terminals: the tree edges;
  4. reduced graph: tree edges with weight (geometric length | priced length, per --scope) x the production charge
               factor 0.8 + 0.4 * mean Q0/(Q+Q0), Q0 = 10000, Q = calc_charge_wcp(disable_dead_mix_cell = false:
               RMS over the nonzero planes, 0 with fewer than two) from STGR; plus the same-blob post-pass edges
               (raw distance between reduced vertices of one STGR blob when either is a terminal);
  5. the rough walk (STMRP from/to, steiner_pc indices mapped to points; a point no longer in the tree is replaced by
               the nearest tree vertex) by Dijkstra on the reduced graph, scored against the image ridge with
               d111s_replay.score exactly as d113_steiner_census.py scores the C++ seed.

CHECK at alpha = 0: the replayed tree-edge set against the dumped STGE path/connect edges, and the replayed walk
cost against the dumped graph's own walk (d111s_common.SteinerBlock.walk).

Per level: S1 (> 1 cm off), far (> 5 cm), wig, cov, the interiors by zero-plane class and their off-ridge share, the
h1 / v3 window max, and the doc-114 catalogue GAP stretches whose in-ball seed length changes by > 20 % (or whose walk
becomes unreachable) -- the sizing bar of figs/115_sizing_rule.txt.

Usage:
  d115_base_replay.py --det pdhd --arm d115hbd [--logd /home/xqian/tmp/d115/arm_d115hbd] [--jobs 4] \
      --stretches figs/114_support_pdhd_stretches.tsv --levels 0:tree,0.25:tree,0.5:tree,1:tree,2:tree,0.5:tree+path,1:tree+path,2:tree+path \
      --out figs/115_replay_pdhd
Writes <out>.txt and <out>.json.
"""
import argparse, collections, csv, glob, json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import uproot
from multiprocessing import Pool
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import d111s_common as C
from d111s_replay import score, SPOTS
A = C.A

Q0 = 10000.0
F1, F2 = 0.8, 0.4
BALL_PAD = 5.0
GAP_CLASSES = ("GAP-dead", "GAP-short", "GAP-live")


def parse_levels(s):
    out = []
    for tok in s.split(","):
        a, sc = tok.split(":")
        out.append((float(a), sc))
    return out


def wcp_charge(q):
    nz = q != 0
    n = nz.sum(axis=1)
    ss = (q * q * nz).sum(axis=1)
    return np.where(n > 1, np.sqrt(ss / np.maximum(n, 1)), 0.0)


class Replay:
    """one cluster's dumped base graph + terminals; trees per level are cached."""

    def __init__(self, sb):
        self.sb = sb
        self.N = len(sb.R)
        self.pos = sb.R
        self.nz = (sb.R_q < 0.5).sum(axis=1)
        self.Q = wcp_charge(sb.R_q)
        self.T = np.unique(sb.V_old[sb.V_term])
        self.A_s, self.A_t, self.A_w = sb.A_s, sb.A_t, sb.A_w
        self.trees = {}
        # dumped tree pairs in point indices
        m = (sb.E_src == C.SRC_CODE["path"]) | (sb.E_src == C.SRC_CODE["connect"])
        a = sb.V_old[sb.E_s[m]]; b = sb.V_old[sb.E_t[m]]
        self.dumped_pairs = set(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))

    def tree(self, alpha, scope):
        key = (alpha, scope)
        if key in self.trees:
            return self.trees[key]
        s, t, w = self.A_s, self.A_t, self.A_w
        wr = w * (1.0 + alpha * 0.5 * (self.nz[s] + self.nz[t])) if alpha > 0 else w.copy()
        G = csr_matrix((np.r_[wr, wr], (np.r_[s, t], np.r_[t, s])), shape=(self.N, self.N))
        T = self.T
        if len(T) < 2:
            return None
        dist, pred, src = dijkstra(G, directed=False, indices=T, min_only=True, return_predecessors=True)
        # bridges: first minimum per unordered terminal pair in edge order
        sa, sb_ = src[s], src[t]
        m = (sa != sb_) & np.isfinite(dist[s]) & np.isfinite(dist[t])
        idx = np.where(m)[0]
        if len(idx) == 0:
            return None
        tot = dist[s[idx]] + wr[idx] + dist[t[idx]]
        ka = np.minimum(sa[idx], sb_[idx]); kb = np.maximum(sa[idx], sb_[idx])
        order = np.lexsort((idx, tot, kb, ka))       # by pair, then cost, then edge order
        ka_o, kb_o = ka[order], kb[order]
        first = np.r_[True, (ka_o[1:] != ka_o[:-1]) | (kb_o[1:] != kb_o[:-1])]
        sel = idx[order[first]]
        pairs = set()
        for e in sel:
            a, b = int(s[e]), int(t[e])
            pairs.add((min(a, b), max(a, b)))
            for v in (a, b):
                cur = v
                while src[cur] != cur:
                    u = int(pred[cur])
                    if u < 0:
                        break
                    pairs.add((min(u, cur), max(u, cur)))
                    cur = u
        V = np.array(sorted({x for p in pairs for x in p}))
        new = {int(v): i for i, v in enumerate(V)}
        term = np.isin(V, T)
        # reduced weights
        ea = np.array([p[0] for p in pairs]); eb = np.array([p[1] for p in pairs])
        # geometric or priced length of each tree pair: look up in the base edge list
        key_all = self.A_s.astype(np.int64) * self.N + self.A_t
        key_rev = self.A_t.astype(np.int64) * self.N + self.A_s
        lut = {}
        for k1, k2, ww, wp in zip(key_all.tolist(), key_rev.tolist(), w.tolist(), wr.tolist()):
            lut[k1] = (ww, wp); lut[k2] = (ww, wp)
        L = np.array([lut.get(int(a) * self.N + int(b), (np.nan, np.nan)) for a, b in zip(ea, eb)])
        geo = L[:, 1] if scope == "tree+path" else L[:, 0]
        Qs, Qt = self.Q[ea], self.Q[eb]
        fac = F1 + F2 * (0.5 * Q0 / (Qs + Q0) + 0.5 * Q0 / (Qt + Q0))
        rw = geo * fac
        es = np.array([new[int(a)] for a in ea]); et = np.array([new[int(b)] for b in eb])
        # same-blob post-pass
        blob = self.sb.R_blob[V]
        add_s, add_t, add_w = [], [], []
        have = set(zip(es.tolist(), et.tolist())) | set(zip(et.tolist(), es.tolist()))
        by_blob = collections.defaultdict(list)
        for i, b in enumerate(blob.tolist()):
            by_blob[b].append(i)
        for b, mem in by_blob.items():
            if len(mem) < 2:
                continue
            for i in range(len(mem)):
                for j in range(i + 1, len(mem)):
                    x, y = mem[i], mem[j]
                    if (term[x] or term[y]) and (x, y) not in have:
                        add_s.append(x); add_t.append(y); add_w.append(float(np.linalg.norm(self.pos[V[x]] - self.pos[V[y]])))
                        have.add((x, y)); have.add((y, x))
        es2 = np.r_[es, add_s].astype(int); et2 = np.r_[et, add_t].astype(int); w2 = np.r_[rw, add_w]
        n = len(V)
        R = csr_matrix((np.r_[w2, w2], (np.r_[es2, et2], np.r_[et2, es2])), shape=(n, n))
        out = dict(V=V, new=new, term=term, pairs=pairs, R=R, ntree=len(pairs), nsbt=len(add_s), vtree=cKDTree(self.pos[V]))
        self.trees[key] = out
        return out

    def walk(self, tr, src_pt, dst_pt):
        """Dijkstra on the reduced graph between the tree vertices nearest to two points -> (point index path, cost)."""
        a = tr["new"].get(int(src_pt)); b = tr["new"].get(int(dst_pt))
        if a is None:
            a = int(tr["vtree"].query(self.pos[src_pt])[1])
        if b is None:
            b = int(tr["vtree"].query(self.pos[dst_pt])[1])
        dist, pred = dijkstra(tr["R"], directed=False, indices=a, return_predecessors=True)
        if not np.isfinite(dist[b]):
            return None, np.inf
        path = [b]
        while path[-1] != a:
            path.append(int(pred[path[-1]]))
        return tr["V"][np.array(path[::-1])], float(dist[b])


def ball_length(P, mid, r):
    X, wl = C.polyline_samples(P, 0.3)
    if len(X) == 0:
        return 0.0
    m = np.linalg.norm(X - mid, axis=1) <= r
    return float(wl[m].sum())


def one(args):
    det, arm, logd, pre, levels, gaps = args
    d = f"{C.IMG}/{det}/work/{pre}_{arm}"
    lg = f"{logd}/evt_{pre}.log.gz"
    acc = collections.defaultdict(collections.Counter)
    spots = collections.defaultdict(lambda: collections.defaultdict(list))
    chk = collections.Counter()
    if not (os.path.exists(lg) and os.path.exists(f"{d}/tracking-stm.root")):
        return pre, None, None, None
    tr = C.parse_trace(lg, want_base=True)
    t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(["cluster_id"], library="np")
    have = set(t["cluster_id"].tolist())
    img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    seen, cache = set(), {}
    for rp in tr["rp"]:
        order, cl, kind, src, dst, npath = rp
        if kind != "rough" or cl not in have or (cl, src, dst) in seen:
            continue
        seen.add((cl, src, dst))
        sb = C.graph_for(tr, cl, order)
        mi = img[1] == cl
        if sb is None or sb.A_s is None or mi.sum() < 5 or max(src, dst) >= sb.nv:
            chk["walks_skipped"] += 1
            continue
        if id(sb) not in cache:
            R = A.Ridge(img[0][mi], img[2][mi])
            q = np.clip(img[2][mi], 0, None)
            cache[id(sb)] = (Replay(sb), R, cKDTree(img[0][mi]), q, float(q.sum()))
        rep, R, itree, qimg, qtot = cache[id(sb)]
        if not R.ok:
            chk["walks_skipped"] += 1
            continue
        # the C++ walk on the dumped graph
        p0, c0 = sb.walk(src, dst)
        chk["walks"] += 1
        base_len = {}
        for (alpha, scope) in levels:
            lab = f"{alpha:g}:{scope}"
            trr = rep.tree(alpha, scope)
            a_ = acc[lab]
            if trr is None:
                a_["unreach"] += 1
                continue
            if alpha == 0 and scope == "tree" and "checked" not in trr:
                trr["checked"] = True
                dp, rp_ = rep.dumped_pairs, trr["pairs"]
                chk["dumped_edges"] += len(dp); chk["replayed_edges"] += len(rp_)
                chk["common_edges"] += len(dp & rp_)
                chk["clusters"] += 1
                if dp == rp_:
                    chk["clusters_exact"] += 1
                # interiors census on the replayed tree
            if "interiors" not in trr:
                trr["interiors"] = True
                V = trr["V"]; interior = ~trr["term"]
                nz = rep.nz[V[interior]]
                off = R.offset(rep.pos[V[interior]]) > 1.0
                a_["nv"] += len(V); a_["ne"] += trr["ntree"]; a_["nsbt"] += trr["nsbt"]
                for c in (0, 1, 2):
                    m = (nz == c) if c < 2 else (nz >= 2)
                    a_[f"int_{c}"] += int(m.sum()); a_[f"int_{c}_off"] += int((m & off).sum())
            path, cost = rep.walk(trr, sb.V_old[src], sb.V_old[dst])
            a_["walks"] += 1
            if path is None:
                a_["unreach"] += 1
                continue
            if alpha == 0 and scope == "tree" and p0 is not None:
                if abs(cost - c0) <= 1e-6 * max(1.0, c0):
                    chk["walk_cost_equal"] += 1
                else:
                    chk["walk_cost_diff"] += 1
                if abs(cost - c0) <= 5e-3 * max(1.0, c0):      # amendment 1: within 0.5 %
                    chk["walk_cost_close"] += 1
            P = rep.pos[path]
            s = score(P, R, itree, qimg, qtot)
            for kk in ("L", "off", "far", "wig", "nint", "cov", "qtot"):
                a_[kk] += s[kk]
            for name, (sdet, sev, scl, click, win) in SPOTS.items():
                if sdet == det and sev == pre and scl == cl:
                    X, _ = C.polyline_samples(P, 0.3)
                    if len(X):
                        m = np.linalg.norm(X - np.array(click), axis=1) <= win
                        if m.any():
                            spots[lab][name].append(float(np.max(R.offset(X[m]))))
            # GAP stretches of this (event, cluster): in-ball seed length vs the alpha = 0 level
            for gi, (mid, r) in enumerate(gaps.get((pre, cl), [])):
                Lb = ball_length(P, mid, r)
                if alpha == 0 and scope == "tree":
                    base_len[(gi, src, dst)] = Lb
                else:
                    L0 = base_len.get((gi, src, dst))
                    if L0 is None or L0 < 1.0:
                        continue
                    a_["gap_n"] += 1
                    if abs(Lb - L0) > 0.2 * L0:
                        a_["gap_changed"] += 1
        # unreachable walks on a level also count against its GAP stretches
        for (alpha, scope) in levels:
            pass
    return pre, {k: dict(v) for k, v in acc.items()}, {k: {n: v for n, v in d_.items()} for k, d_ in spots.items()}, dict(chk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--logd"); ap.add_argument("--events"); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--stretches"); ap.add_argument("--levels", default="0:tree,0.25:tree,0.5:tree,1:tree,2:tree,0.5:tree+path,1:tree+path,2:tree+path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logd = a.logd or f"/home/xqian/tmp/d115/arm_{a.arm}"
    levels = parse_levels(a.levels)
    if levels[0] != (0.0, "tree"):
        levels = [(0.0, "tree")] + [l for l in levels if l != (0.0, "tree")]
    gaps = collections.defaultdict(list)
    if a.stretches:
        for row in csv.DictReader(open(a.stretches), delimiter="\t"):
            if row["class"] in GAP_CLASSES:
                L = float(row["length_cm"])
                gaps[(row["event"], int(row["cluster"]))].append((np.array((float(row["mx"]), float(row["my"]), float(row["mz"]))), 0.5 * L + BALL_PAD))
    evs = sorted(os.path.basename(p)[:-len(a.arm) - 1] for p in glob.glob(f"{C.IMG}/{a.det}/work/*_{a.arm}"))
    if a.events:
        evs = [e for e in evs if e in set(a.events.split(","))]
    with Pool(a.jobs) as pool:
        res = pool.map(one, [(a.det, a.arm, logd, pre, levels, dict(gaps)) for pre in evs])
    tot = collections.defaultdict(collections.Counter); spots = collections.defaultdict(lambda: collections.defaultdict(list)); chk = collections.Counter()
    nev = 0
    for pre, acc, sp, ck in res:
        if acc is None:
            continue
        nev += 1
        for k, v in acc.items():
            tot[k].update(v)
        for k, d_ in sp.items():
            for n, v in d_.items():
                spots[k][n].extend(v)
        chk.update(ck)
    labs = [f"{al:g}:{sc}" for al, sc in levels]
    base = tot[labs[0]]
    sh = lambda c, x, y: (c[x] / c[y]) if c[y] else float("nan")
    rows = {}
    for lab in labs:
        c = tot[lab]
        rows[lab] = dict(walks=c["walks"], unreach=c["unreach"], L_m=c["L"] / 100.0, off=sh(c, "off", "L"), far=sh(c, "far", "L"),
                         wig=sh(c, "wig", "nint"), cov=sh(c, "cov", "qtot"), nv=c["nv"], ne=c["ne"], nsbt=c["nsbt"],
                         interiors={f"nz{k}": c[f"int_{k}"] for k in (0, 1, 2)}, interiors_off={f"nz{k}": c[f"int_{k}_off"] for k in (0, 1, 2)},
                         gap_n=c["gap_n"], gap_changed=c["gap_changed"],
                         spots={n: (max(v) if v else float("nan")) for n, v in spots[lab].items()})
    with open(a.out + ".txt", "w") as fh:
        P = lambda s="": fh.write(s + "\n")
        P(f"# doc pdvd/115 base-graph replay: det={a.det} arm={a.arm} events={nev} levels={a.levels}")
        P(f"# CHECK alpha=0: clusters {chk['clusters']} (tree edge set exact on {chk['clusters_exact']}); dumped tree edges {chk['dumped_edges']}, "
          f"replayed {chk['replayed_edges']}, common {chk['common_edges']} "
          f"({100*chk['common_edges']/max(1,chk['dumped_edges']):.2f} % of dumped, {100*chk['common_edges']/max(1,chk['replayed_edges']):.2f} % of replayed); "
          f"walks {chk['walks']} (skipped {chk['walks_skipped']}), walk cost equal {chk['walk_cost_equal']}, differs {chk['walk_cost_diff']}, "
          f"within 0.5 % {chk['walk_cost_close']} ({100*chk['walk_cost_close']/max(1,chk['walks']):.1f} %)")
        P()
        P("| level | walks | unreach | seed m | S1 > 1 cm off | rel | far > 5 cm | wig | cov | tree v / e / sbt | interiors nz0 / nz1 / nz2+ (off share) | GAP stretches changed | spots |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for lab in labs:
            r = rows[lab]
            rel = (r["off"] - base["off"] / base["L"]) / (base["off"] / base["L"]) if base["L"] and base["off"] else float("nan")
            rel = (r["off"] / rows[labs[0]]["off"] - 1) if rows[labs[0]]["off"] else float("nan")
            io = r["interiors"]; ioff = r["interiors_off"]
            ints = " / ".join(f"{io[k]} ({100*ioff[k]/max(1,io[k]):.0f} %)" for k in ("nz0", "nz1", "nz2"))
            gap = f"{r['gap_changed']} / {r['gap_n']} ({100*r['gap_changed']/max(1,r['gap_n']):.1f} %)" if lab != labs[0] else "-"
            sp = ", ".join(f"{n} {v:.2f}" for n, v in sorted(r["spots"].items()))
            P(f"| {lab} | {r['walks']} | {r['unreach']} | {r['L_m']:.1f} | {100*r['off']:.2f} % | {100*rel:+.1f} % | {100*r['far']:.2f} % | "
              f"{100*r['wig']:.2f} % | {100*r['cov']:.2f} % | {r['nv']} / {r['ne']} / {r['nsbt']} | {ints} | {gap} | {sp} |")
    json.dump(dict(det=a.det, arm=a.arm, events=nev, check=dict(chk), levels=rows), open(a.out + ".json", "w"), indent=1)
    print(open(a.out + ".txt").read())


if __name__ == "__main__":
    main()
