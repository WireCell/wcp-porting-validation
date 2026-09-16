#!/usr/bin/env python3
"""doc pdvd/111 round 2, A5 -- the offline counterfactual ladder on the dumped Steiner graphs.

The rule, the candidates and the metrics are in figs/111s_ladder_rule.txt (sha256 in 111s_ladder_rule.sha256), frozen
before this script ran on the full arms.  In short: every round-1 rough walk (STMRP "rough") of the dump arm is replayed
with scipy Dijkstra on its cluster's dumped steiner_graph, once with the dumped weights (the base, which reproduces the
C++ seed; A3 replay check) and once per candidate re-weighting; each replayed seed is scored against the image ridge.

Usage: d111s_replay.py --det pdhd --arm d111hst [--events a,b] [--jobs 12] --out figs/111s_replay_pdhd
       -> <out>.txt (per-candidate table + the qualification columns) and <out>.json (the numbers the rule reads)
"""
import argparse, collections, glob, json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np
import uproot
from scipy.spatial import cKDTree
import d111s_common as C
A = C.A

CANDIDATES = ([("base", 0)] + [(f, k) for f in ("G_any", "G_3pl", "G_2pl") for k in (2, 5, 10)] + [("E_len", k) for k in (0.5, 1, 2)]
              + [("P_paint", k) for k in (2, 5, 10)])      # P_paint: amendment 1 of the rule (reported, not buildable)
SPOTS = {  # name: (det, event, cluster, click, window cm) -- doc 111 sec 1
    "h1": ("pdhd", "029107_16", 108, (75.3, 438.3, 450.1), 12.0),
    "h2": ("pdhd", "029107_16", 106, (182.0, 537.4, 401.8), 24.0),
    "v1": ("pdvd", "039349_20", 80, (313.7, 95.9, 124.9), 12.0),
    "v2": ("pdvd", "039349_20", 80, (94.5, 30.7, 165.0), 12.0),
    "v3": ("pdvd", "039349_20", 25, (-270.2, -243.7, 36.0), 12.0),
}


def painted_vertices(sb):
    """a Steiner vertex on a PAINTED cell: some plane with zero retiled charge and no dead channel at r0.2/ch0."""
    q0 = sb.R_q[sb.V_old] < 1.0
    dead = np.c_[[(sb.V_m02 >= 0) & (((sb.V_m02 >> (3 + p)) & 1) > 0) for p in range(3)]].T
    return (q0 & ~dead).any(axis=1)


def weights(sb, fam, k):
    w = sb.E_w.copy()
    if fam == "base":
        return w
    if fam == "E_len":
        m = sb.E_len >= 3.0
        w[m] *= (1.0 + k)
        return w
    if fam == "P_paint":
        painted = painted_vertices(sb)
        p = painted[sb.E_s].astype(float) + painted[sb.E_t].astype(float)
        return w * (1.0 + k * p / 2.0)
    S = sb.S02
    n = np.maximum(S[:, 0], 1)
    if fam == "G_any":
        bad = ((S[:, 0] - S[:, 3]) + 0.25 * (S[:, 3] - S[:, 4])) / n
    elif fam == "G_3pl":
        bad = 1.0 - S[:, 1] / n
    else:
        bad = 1.0 - S[:, 2] / n
    m = (sb.E_len >= 0.5) & (S[:, 0] > 0)
    w[m] *= (1.0 + k * bad[m])
    return w


def score(P, R, itree, qimg, qtot):
    X, wl = C.polyline_samples(P, 0.3)
    if len(X) == 0:
        return dict(L=0.0, off=0.0, far=0.0, wig=0, nint=0, cov=0.0, qtot=qtot, maxoff=np.nan)
    o = R.offset(X)
    nw, ni = C.wiggle_share(P)
    near = set()
    for lst in itree.query_ball_point(X, 1.5):
        near.update(lst)
    cov = float(qimg[list(near)].sum()) if near else 0.0
    return dict(L=float(wl.sum()), off=float(wl[o > 1.0].sum()), far=float(wl[o > 5.0].sum()), wig=nw, nint=ni,
                cov=cov, qtot=qtot, maxoff=float(np.nanmax(o)))


def one(args):
    det, arm, logd, pre = args
    out = collections.defaultdict(collections.Counter)
    spots = {}
    d = f"{C.IMG}/{det}/work/{pre}_{arm}"
    lg = f"{logd}/evt_{pre}.log.gz"
    if not (os.path.exists(lg) and os.path.exists(f"{d}/tracking-stm.root")):
        return out, spots, 1
    tr = C.parse_trace(lg)
    t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(["cluster_id"], library="np")
    have = set(t["cluster_id"].tolist())
    img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    seen = set()
    cache = {}
    for rp in tr["rp"]:
        order, cl, kind, src, dst, npath = rp
        if kind != "rough" or cl not in have or (cl, src, dst) in seen:
            continue
        seen.add((cl, src, dst))
        sb = C.graph_for(tr, cl, order)
        mi = img[1] == cl
        if sb is None or mi.sum() < 5 or max(src, dst) >= sb.nv:
            out["base:0"]["walks_skipped"] += 1
            continue
        if cl not in cache:
            R = A.Ridge(img[0][mi], img[2][mi])
            q = np.clip(img[2][mi], 0, None)
            cache[cl] = (R, cKDTree(img[0][mi]), q, float(q.sum()), {})
        R, itree, qimg, qtot, wc = cache[cl]
        if not R.ok:
            out["base:0"]["walks_skipped"] += 1
            continue
        base_path = None
        for fam, k in CANDIDATES:
            key = f"{fam}:{k}"
            if key not in wc:
                wc[key] = weights(sb, fam, k)
            p, c = sb.walk(src, dst, wc[key])
            if key == "base:0":
                base_path = p
            acc = out[key]
            acc["walks"] += 1
            if p is None:
                acc["unreach"] += 1
                continue
            s = score(sb.V[p], R, itree, qimg, qtot)
            for kk in ("L", "off", "far", "wig", "nint", "cov", "qtot"):
                acc[kk] += s[kk]
            acc["changed"] += int(key != "base:0" and p != base_path)
            for name, (sdet, sev, scl, click, win) in SPOTS.items():
                if sdet == det and sev == pre and scl == cl:
                    P = sb.V[p]
                    X, _ = C.polyline_samples(P, 0.3)
                    if len(X):
                        m = np.linalg.norm(X - np.array(click), axis=1) <= win
                        if m.any():
                            v = float(np.max(R.offset(X[m])))
                            spots.setdefault(name, {}).setdefault(key, []).append(v)
    return out, spots, 0


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
    evs = sorted(os.path.basename(p)[:-len(a.arm) - 1] for p in glob.glob(f"{C.IMG}/{a.det}/work/*_{a.arm}"))
    if a.events:
        evs = [e for e in evs if e in a.events.split(",")]
    tot = collections.defaultdict(collections.Counter)
    spots = collections.defaultdict(lambda: collections.defaultdict(list))
    missing = 0
    with Pool(a.jobs) as pool:
        for out, sp, miss in pool.imap_unordered(one, [(a.det, a.arm, logd, e) for e in evs]):
            missing += miss
            for k, v in out.items():
                tot[k].update(v)
            for name, d in sp.items():
                for key, vals in d.items():
                    spots[name][key].extend(vals)
    base = tot["base:0"]
    def share(acc, num, den):
        return acc[num] / acc[den] if acc[den] else float("nan")
    b_off = share(base, "off", "L"); b_wig = share(base, "wig", "nint"); b_cov = share(base, "cov", "qtot")
    L = [f"# doc pdvd/111 round 2 counterfactual ladder: det={a.det} arm={a.arm} events={len(evs)} (missing {missing}); "
         f"rule figs/111s_ladder_rule.txt",
         f"# walks replayed {base['walks']}, skipped {base['walks_skipped']}", "",
         "| candidate | off (> 1 cm) | rel. change | far (> 5 cm) | wiggle | cov | seed length (m) | walks changed | unreachable | off-rule | wig-rule | cov-rule |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for fam, k in CANDIDATES:
        key = f"{fam}:{k}"
        acc = tot[key]
        off = share(acc, "off", "L"); far = share(acc, "far", "L"); wig = share(acc, "wig", "nint"); cov = share(acc, "cov", "qtot")
        rel = (off - b_off) / b_off if b_off else float("nan")
        r_off = rel <= -0.25; r_wig = wig <= 1.10 * b_wig; r_cov = cov >= b_cov - 0.01
        res[key] = dict(off=off, rel=rel, far=far, wig=wig, cov=cov, L=acc["L"], unreach=acc["unreach"], changed=acc["changed"],
                        rule_off=bool(r_off), rule_wig=bool(r_wig), rule_cov=bool(r_cov), rule_unr=acc["unreach"] == base["unreach"])
        L.append(f"| {key} | {100*off:.2f} % | {100*rel:+.1f} % | {100*far:.2f} % | {100*wig:.2f} % | {100*cov:.2f} % | "
                 f"{acc['L']/100:.1f} | {acc['changed']} | {acc['unreach']} | {'pass' if r_off else 'FAIL'} | "
                 f"{'pass' if r_wig else 'FAIL'} | {'pass' if r_cov else 'FAIL'} |")
    if spots:
        L += ["", "## spots: seed max ridge offset in the window (cm), per walk through it",
              "| spot | " + " | ".join(f"{f}:{k}" for f, k in CANDIDATES) + " |", "|---|" + "---|" * len(CANDIDATES)]
        for name in sorted(spots):
            L.append(f"| {name} | " + " | ".join(
                "/".join(f"{v:.2f}" for v in spots[name].get(f"{f}:{k}", [])) or "-" for f, k in CANDIDATES) + " |")
    txt = "\n".join(L) + "\n"
    open(a.out + ".txt", "w").write(txt)
    json.dump(dict(det=a.det, arm=a.arm, events=len(evs), candidates=res,
                   spots={n: dict(v) for n, v in spots.items()}), open(a.out + ".json", "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
