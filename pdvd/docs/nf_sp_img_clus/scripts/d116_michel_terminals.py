#!/usr/bin/env python3
"""doc pdvd/116 sec 2 -- does the new trajectory configuration drop the Steiner terminals / vertices of the Michel
electron?  For every Michel tag lost between arm A and arm B on a cluster that is an accepted stopper in both
(the MICHEL class of d116_mechanism.py), take A's Michel member points (T_stm_michel_pts role 3) and count, in each
arm's Bee set, the steiner_terminals-global and steiner_graph-global points within R of them.  A control: the same
count for clusters whose Michel tag held in both arms.  Read-only.

Usage: d116_michel_terminals.py --det pdvd --a d115voff --b d115vp3bwp05 --mechanism figs/116_mechanism_pdvd.tsv --out figs/116_michel_terminals_pdvd.txt
"""
import argparse, csv, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import uproot
from scipy.spatial import cKDTree
import d111s_common as C
A_ = C.A
R = 1.5


def michel_pts(det, arm, ev, cid):
    d = f"{C.IMG}/{det}/work/{ev}_{arm}"
    t = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel_pts"].arrays(["cluster_id", "role", "x", "y", "z"], library="np")
    m = (t["cluster_id"] == cid) & (t["role"] == 3)
    return np.c_[t["x"][m], t["y"][m], t["z"][m]]


def count(det, arm, ev, P):
    d = f"{C.IMG}/{det}/work/{ev}_{arm}"
    out = {}
    for layer in ("steiner_terminals-global", "steiner_graph-global", "clustering-global"):
        L = A_.bee_layer(f"{d}/mabc-pr.zip", layer)
        if L is None or len(L[0]) == 0 or len(P) == 0:
            out[layer] = 0; continue
        out[layer] = int(np.sum(cKDTree(L[0]).query(P)[0] <= R)) if True else 0
        # points of the layer within R of any Michel point
        out[layer] = int(np.sum(cKDTree(P).query(L[0])[0] <= R))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--mechanism", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--control", type=int, default=25)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.mechanism), delimiter="\t") if r["metric"] == "michel_found" and r["how"].startswith("MICHEL")]
    L = [f"# doc pdvd/116 Michel terminals {a.det}: A={a.a} B={a.b}; layer points within {R} cm of A's Michel members (role 3)"]
    L.append(f"{'class':8s} {'key':16s} {'how':34s} {'nMichel':>7s} | terminals A  B | graph A  B | image A  B")
    agg = {}
    for r in rows:
        ev, cid = r["key"].split("/"); cid = int(cid)
        src_arm = a.a if r["class"] in ("tp_lost", "fp_gone") else a.b
        P = michel_pts(a.det, src_arm, ev, cid)
        ca, cb = count(a.det, a.a, ev, P), count(a.det, a.b, ev, P)
        L.append(f"{r['class']:8s} {r['key']:16s} {r['how'][:34]:34s} {len(P):7d} | {ca['steiner_terminals-global']:5d} {cb['steiner_terminals-global']:3d} | "
                 f"{ca['steiner_graph-global']:5d} {cb['steiner_graph-global']:3d} | {ca['clustering-global']:5d} {cb['clustering-global']:3d}")
        agg.setdefault(r["class"], []).append((ca, cb, len(P)))
    # control: clusters with michel_found in both arms
    import glob
    ctl = []
    for d in sorted(glob.glob(f"{C.IMG}/{a.det}/work/*_{a.a}")):
        ev = os.path.basename(d)[:-len(a.a) - 1]
        try:
            ta = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel"].arrays(["cluster_id", "is_stm", "michel_found"], library="np")
            tb = uproot.open(f"{C.IMG}/{a.det}/work/{ev}_{a.b}/tracking-pr.root")["T_stm_michel"].arrays(["cluster_id", "is_stm", "michel_found"], library="np")
        except Exception:
            continue
        mb = {int(c): (int(s), int(m)) for c, s, m in zip(tb["cluster_id"], tb["is_stm"], tb["michel_found"])}
        for c, s, m in zip(ta["cluster_id"], ta["is_stm"], ta["michel_found"]):
            if s and m and mb.get(int(c)) == (1, 1):
                ctl.append((ev, int(c)))
        if len(ctl) >= a.control:
            break
    L.append(f"\n== control: {len(ctl)} clusters with an accepted stopper + Michel in BOTH arms (first events)")
    cs = []
    for ev, cid in ctl[:a.control]:
        P = michel_pts(a.det, a.a, ev, cid)
        ca, cb = count(a.det, a.a, ev, P), count(a.det, a.b, ev, P)
        cs.append((ca, cb, len(P)))
    def summ(lst, nm):
        if not lst:
            return f"  {nm}: none"
        ta = np.array([x[0]['steiner_terminals-global'] for x in lst]); tb = np.array([x[1]['steiner_terminals-global'] for x in lst])
        ga = np.array([x[0]['steiner_graph-global'] for x in lst]); gb = np.array([x[1]['steiner_graph-global'] for x in lst])
        ia = np.array([x[0]['clustering-global'] for x in lst]); ib = np.array([x[1]['clustering-global'] for x in lst])
        return (f"  {nm}: n {len(lst)}; terminals sum A {ta.sum()} B {tb.sum()} (B/A {tb.sum() / max(ta.sum(), 1):.2f}; clusters with B < A: {int(np.sum(tb < ta))}, B == 0 & A > 0: {int(np.sum((tb == 0) & (ta > 0)))}); "
                f"graph vertices sum A {ga.sum()} B {gb.sum()} (B/A {gb.sum() / max(ga.sum(), 1):.2f}); image points A {ia.sum()} B {ib.sum()}")
    L.append("\n== summary")
    for cls, lst in agg.items():
        L.append(summ(lst, cls))
    L.append(summ(cs, "control (Michel in both)"))
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
