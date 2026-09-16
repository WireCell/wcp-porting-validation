#!/usr/bin/env python3
"""doc pdvd/113 -- what an arm loses or gains against its base, on the common events: the gap-jumping clauses of
figs/113_pred.txt.  Read-only.

  (a) truncation -- persisted STM fits (tracking-stm.root T_rec_charge, one record per (cluster_id, pass)) present in
      both arms: fit length (sum of consecutive row distances, rows in stored order) arm / base; the share of records
      below 0.9 (truncated) and above 1.1 (extended), and the median ratio.
  (c) candidates -- T_stm_michel rows (tracking-pr.root) keyed (event, cluster_id): only in base, only in arm, and
      is_stm / michel_found 1 -> 0 and 0 -> 1 on the common ones.  With --logd-base (a dump arm), the base-only and
      is_stm-lost clusters are split by what the BASE Steiner graph of that cluster had: a vertex on a dead channel
      (some plane at zero charge with its dead bit set) -- where the retile's dead fill acts -- and a bridge edge
      (base provenance ctpc or mst) on its rough seed.
  records only in one arm are counted at the cluster level too (a cluster that lost every STM fit).

Usage: d113_compare.py --det pdhd --base d113hbase --arm d113hnone [--logd-base DIR] [--out figs/113_compare_...txt]
"""
import argparse, collections, glob, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import uproot
import d111s_common as C
from d112_terminal_planes import vertex_class


def events(det, arm):
    return {os.path.basename(p)[:-len(arm) - 1] for p in glob.glob(f"{C.IMG}/{det}/work/*_{arm}")
            if os.path.exists(f"{p}/tracking-stm.root") and os.path.exists(f"{p}/tracking-pr.root")}


def fits(d):
    f = uproot.open(f"{d}/tracking-stm.root")
    if "T_rec_charge" not in f:                # an event with no STM fit writes no tree
        return {}
    t = f["T_rec_charge"].arrays(["cluster_id", "pass", "x", "y", "z"], library="np")
    out = {}
    for cl, ps in sorted(set(zip(t["cluster_id"].tolist(), t["pass"].tolist()))):
        m = (t["cluster_id"] == cl) & (t["pass"] == ps)
        P = np.c_[t["x"][m], t["y"][m], t["z"][m]]
        out[(cl, ps)] = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
    return out


def cands(d):
    f = uproot.open(f"{d}/tracking-pr.root")
    if "T_stm_michel" not in f:                # an event with no STM candidate writes no tree
        return {}
    t = f["T_stm_michel"].arrays(["cluster_id", "is_stm", "michel_found"], library="np")
    return {int(c): (int(s), int(m)) for c, s, m in zip(t["cluster_id"], t["is_stm"], t["michel_found"])}


def base_graph_flags(logd, pre, clusters):
    """{cluster: (touches_dead, seed_bridged)} from the base dump: the cluster's first rough walk's graph."""
    lg = f"{logd}/evt_{pre}.log.gz"
    if not os.path.exists(lg) or not clusters:
        return {}
    tr = C.parse_trace(lg)
    out = {}
    for order, cl, kind, src, dst, npath in tr["rp"]:
        if kind != "rough" or cl not in clusters or cl in out:
            continue
        sb = C.graph_for(tr, cl, order)
        if sb is None or max(src, dst) >= sb.nv:
            continue
        cls, zero, dead = vertex_class(sb)
        touches = bool((zero & dead).any())
        p, _ = sb.walk(src, dst)
        bridged = False
        if p is not None:
            for u, v in zip(p[:-1], p[1:]):
                k = sb.edge_index.get((min(u, v), max(u, v)))
                if k is not None and sb.E_base[k] in (C.BASE_CODE["ctpc"], C.BASE_CODE["mst"]):
                    bridged = True
                    break
        out[cl] = (touches, bridged)
    for cl in clusters:
        out.setdefault(cl, (None, None))       # no rough walk dumped for it
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--logd-base")
    ap.add_argument("--out")
    a = ap.parse_args()
    evs = sorted(events(a.det, a.base) & events(a.det, a.arm))
    ratios, only_b, only_a = [], 0, 0
    cl_only_b = cl_only_a = 0
    cand = collections.Counter()
    split = collections.Counter()
    lost_examples = []
    for pre in evs:
        db, da = f"{C.IMG}/{a.det}/work/{pre}_{a.base}", f"{C.IMG}/{a.det}/work/{pre}_{a.arm}"
        fb, fa = fits(db), fits(da)
        for k in fb:
            if k in fa and fb[k] > 0:
                ratios.append(fa[k] / fb[k])
        only_b += len(set(fb) - set(fa)); only_a += len(set(fa) - set(fb))
        cb, ca = {k[0] for k in fb}, {k[0] for k in fa}
        cl_only_b += len(cb - ca); cl_only_a += len(ca - cb)
        xb, xa = cands(db), cands(da)
        cand["base"] += len(xb); cand["arm"] += len(xa)
        ob = set(xb) - set(xa); oa = set(xa) - set(xb)
        cand["only_base"] += len(ob); cand["only_arm"] += len(oa)
        cand["only_base_is_stm"] += sum(xb[c][0] for c in ob); cand["only_arm_is_stm"] += sum(xa[c][0] for c in oa)
        common = set(xb) & set(xa)
        stm_lost = {c for c in common if xb[c][0] == 1 and xa[c][0] == 0}
        cand["is_stm_1to0"] += len(stm_lost)
        cand["is_stm_0to1"] += sum(1 for c in common if xb[c][0] == 0 and xa[c][0] == 1)
        cand["michel_1to0"] += sum(1 for c in common if xb[c][1] == 1 and xa[c][1] == 0)
        cand["michel_0to1"] += sum(1 for c in common if xb[c][1] == 0 and xa[c][1] == 1)
        if a.logd_base:
            lost = {c for c in ob if xb[c][0] == 1} | stm_lost          # is_stm tags the arm no longer has
            tagged = {c for c in xb if xb[c][0] == 1}                    # every base is_stm tag: the denominators
            flags = base_graph_flags(a.logd_base, pre, tagged)
            for c in sorted(tagged):
                t, b = flags.get(c, (None, None))
                dcls = "dead" if t else "nodead" if t is not None else "nowalk"
                split[f"tags_{dcls}"] += 1
                if c in lost:
                    split[f"lost_{dcls}"] += 1
                    key = dcls + "/" + ("bridged" if b else "unbridged" if b is not None else "nowalk")
                    split[key] += 1
                    if t:
                        lost_examples.append(f"{pre}/{c}")
    r = np.array(ratios)
    L = [f"# doc pdvd/113 compare ({a.det}): base {a.base} vs arm {a.arm}; common events {len(evs)}",
         "## (a) persisted STM fits (event, cluster, pass)",
         f"common records {len(r)}; only in base {only_b}, only in arm {only_a}; clusters with fits only in base "
         f"{cl_only_b}, only in arm {cl_only_a}",
         f"length ratio arm/base: median {np.median(r):.3f}; truncated (< 0.9) {np.mean(r < 0.9)*100:.2f} % "
         f"({int(np.sum(r < 0.9))}); extended (> 1.1) {np.mean(r > 1.1)*100:.2f} % ({int(np.sum(r > 1.1))})"
         if len(r) else "no common records",
         "## (c) STM candidates (T_stm_michel)",
         f"candidates base {cand['base']}, arm {cand['arm']}; only in base {cand['only_base']} (is_stm 1: "
         f"{cand['only_base_is_stm']}), only in arm {cand['only_arm']} (is_stm 1: {cand['only_arm_is_stm']})",
         f"common: is_stm 1->0 {cand['is_stm_1to0']}, 0->1 {cand['is_stm_0to1']}; michel_found 1->0 "
         f"{cand['michel_1to0']}, 0->1 {cand['michel_0to1']}",
         f"net is_stm tags arm - base: {cand['only_arm_is_stm'] + cand['is_stm_0to1'] - cand['only_base_is_stm'] - cand['is_stm_1to0']:+d}"]
    if a.logd_base:
        L.append("## (c') gap jumping: base is_stm tags by whether the BASE Steiner graph of the cluster has a vertex on a "
                 "dead channel, and the share the arm loses (Gc)")
        for dcls in ("dead", "nodead", "nowalk"):
            n, l = split[f"tags_{dcls}"], split[f"lost_{dcls}"]
            L.append(f"  base tags {dcls:7s} {n:4d}; lost in the arm {l:3d} ({100*l/max(1,n):.1f} %)")
        L.append("  lost, by dead-channel vertex / ctpc-mst bridge on the base rough seed: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(split.items()) if "/" in k))
        L.append("  lost tags whose base graph touches a dead channel: " + (", ".join(lost_examples) or "none"))
    txt = "\n".join(L) + "\n"
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
