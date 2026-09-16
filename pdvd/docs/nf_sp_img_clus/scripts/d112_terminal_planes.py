#!/usr/bin/env python3
"""doc pdvd/112 -- are the off-image Steiner terminals the ones that see charge on only two planes?

Read-only, on the doc-111 round-2 dump arms (d111hst / d111vst, identical to production).  Population as
d111s_painted_split.py: the clusters with a persisted STM record, one Steiner dump per cluster (the one before its first
path block).  Per Steiner vertex, from the dump:
  * the three per-plane charges of its retiled point (STGR q at STGV old) and the dead bits of its test_good_point mask
    (STGV m02, bit 3+p);
  * class   3live    no plane at zero charge
            1blank   exactly one plane at zero charge, and that plane is not dead
            1dead    exactly one plane at zero charge, and that plane is dead
            2blank+  two or three planes at zero charge
  * role    terminal (non-extreme) / extreme (Phase 4, no charge test) / interior;
  * ridge offset (d111_stage_attrib.Ridge on the cluster's image).

Why the class matters: create_steiner_tree's Phase 1 grades a point with Cluster::calc_charge_wcp.  A plane at charge 0
(or whose uncertainty is above 1e10 -- the retiler's forced/dead sentinel is (0, 1e12)) passes the quality test and
leaves the charge RMS, so a 1blank point is graded on its other two planes.  With two planes blank ncharge <= 1 and the
charge is 0, so 2blank+ can only enter as an extreme.

Printed:
  checks   the dumped vertex charge equals the RMS over the nonzero planes (0 when fewer than two), all vertices;
           the control: non-extreme terminals in class 2blank+ (expected 0)
  table A  per role, ridge-offset distribution of each class, and each class's share of the role's vertices on / off;
           for non-extreme terminals of class 3live / 1blank on / off the ridge, the second-strongest plane charge
  table B  persisted round-2 seeds (matched to T_rec_charge as d111s_graph_census.py): the seed's distinct vertices on
           (<= 1 cm) and off (> 1 cm) the ridge, by class x role

Usage: d112_terminal_planes.py --det pdhd --arm d111hst [--jobs 10] > figs/112_terminal_planes_pdhd.txt
"""
import argparse, collections, glob, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np
import uproot
import d111s_common as C
A = C.A

CLASSES = ("3live", "1blank", "1dead", "2blank+")
ROLES = ("terminal", "extreme", "interior")
Q2BINS = (("<=1000", -1.0, 1000.0), ("1000-2000", 1000.0, 2000.0), ("2000-5000", 2000.0, 5000.0), (">5000", 5000.0, np.inf))
BINS = (("<=1", 0.0, 1.0), ("1-2", 1.0, 2.0), ("2-5", 2.0, 5.0), (">5", 5.0, np.inf))


def vertex_class(sb):
    """-> (class index per vertex, zero-plane mask (n,3), dead mask (n,3))."""
    q = sb.R_q[sb.V_old]
    zero = q < 0.5                      # STGR prints %.0f; a real charge never rounds to 0
    m = sb.V_m02
    dead = np.c_[[(m >= 0) & (((m >> (3 + p)) & 1) > 0) for p in range(3)]].T
    nz = zero.sum(axis=1)
    cls = np.full(sb.nv, 0)
    one = nz == 1
    cls[one & ~(zero & dead).any(axis=1)] = 1
    cls[one & (zero & dead).any(axis=1)] = 2
    cls[nz >= 2] = 3
    return cls, zero, dead


def vertex_role(sb):
    role = np.full(sb.nv, 2)
    role[sb.V_term & ~sb.V_ext] = 0
    role[sb.V_ext] = 1
    return role


def rms_check(sb, zero):
    q = sb.R_q[sb.V_old]
    n = (~zero).sum(axis=1)
    rms = np.sqrt(np.where(zero, 0.0, q ** 2).sum(axis=1) / np.maximum(n, 1))
    rms = np.where(n > 1, rms, 0.0)
    return np.abs(rms - sb.V_q) <= 2.0 + 1e-4 * np.abs(sb.V_q)


def one(args):
    det, arm, pre = args
    acc = collections.Counter()
    d = f"{C.IMG}/{det}/work/{pre}_{arm}"
    lg = f"/home/xqian/tmp/d111/arm_{arm}/evt_{pre}.log.gz"
    if not (os.path.exists(lg) and os.path.exists(f"{d}/tracking-stm.root")):
        acc["events_missing"] += 1
        return acc
    tr = C.parse_trace(lg)
    acc["events"] += 1
    t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(["x", "y", "z", "cluster_id", "pass"], library="np")
    have = set(t["cluster_id"].tolist())
    img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    ridges = {}

    def ridge(cl):
        if cl not in ridges:
            mi = img[1] == cl
            ridges[cl] = A.Ridge(img[0][mi], img[2][mi]) if mi.sum() >= 5 else None
        R = ridges[cl]
        return R if (R is not None and R.ok) else None

    # ---- table A + checks: one dump per cluster
    done = set()
    for b in tr["blocks"]:
        cl = b["cl"]
        if cl not in have or cl in done:
            continue
        sb = C.graph_for(tr, cl, b["order"])
        R = ridge(cl)
        if sb is None or R is None or sb.nv == 0:
            continue
        done.add(cl)
        acc["clusters"] += 1
        cls, zero, dead = vertex_class(sb)
        role = vertex_role(sb)
        ok = rms_check(sb, zero)
        acc["chk_vertices"] += sb.nv
        acc["chk_rms_ok"] += int(ok.sum())
        for c in range(4):
            acc[("chk_rms_ok_cls", c)] += int((ok & (cls == c)).sum())
            acc[("chk_n_cls", c)] += int((cls == c).sum())
        acc["ctl_term_2blank"] += int(((role == 0) & (cls == 3)).sum())
        acc["ctl_term_q_le_thr"] += int(((role == 0) & (sb.V_q <= 500)).sum())
        off = R.offset(sb.V)
        # second-strongest plane charge of each terminal: with one plane blank it is the weaker of the two graded planes
        q2 = np.sort(sb.R_q[sb.V_old], axis=1)[:, 1]
        for c in (0, 1):
            for side, sm in (("on", off <= 1.0), ("off", off > 1.0)):
                mm = (role == 0) & (cls == c) & sm
                for name, lo, hi in Q2BINS:
                    acc[("Q2", c, side, name)] += int((mm & (q2 > lo) & (q2 <= hi)).sum())
        binmask = {name: ((off > lo) if lo > 0 else np.ones(sb.nv, bool)) & (off <= hi) for name, lo, hi in BINS}
        for r in range(3):
            for c in range(4):
                mm = (role == r) & (cls == c)
                for name, _, _ in BINS:
                    acc[("A", r, c, name)] += int((mm & binmask[name]).sum())

    # ---- table B: persisted round-2 seeds
    blocks = [b for b in tr["blocks"] if b["cl"] >= 0 and "seed" in b["st"]]
    r2 = collections.defaultdict(list)
    for b in blocks:
        if b["rnd"] == "r2" and "final" in b["st"]:
            r2[b["cl"]].append(b)
    for (cl, ps) in sorted(set(zip(t["cluster_id"].tolist(), t["pass"].tolist()))):
        m = (t["cluster_id"] == cl) & (t["pass"] == ps)
        F = np.c_[t["x"][m], t["y"][m], t["z"][m]]
        match = None
        for b in r2.get(cl, []):
            if len(b["st"]["final"]) == len(F) and np.max(np.abs(b["st"]["final"] - F)) < 0.01:
                match = b
        acc["B_records"] += 1
        if match is None:
            acc["B_records_unmatched"] += 1
            continue
        sb = C.graph_for(tr, cl, match["order"])
        R = ridge(cl)
        if sb is None or R is None:
            acc["B_records_nograph_or_noridge"] += 1
            continue
        j = C.join_vertices(sb, match["st"]["seed"])
        acc["B_seed_points"] += len(j); acc["B_seed_unjoined"] += int((j < 0).sum())
        vid = np.unique(j[j >= 0])
        cls, zero, dead = vertex_class(sb)
        role = vertex_role(sb)
        off = R.offset(sb.V[vid])
        for k, v in enumerate(vid):
            side = "off" if off[k] > 1.0 else "on"
            acc[("B", side, role[v], cls[v])] += 1
    return acc


def pct(a, b):
    return f"{100.0 * a / b:.1f} %" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    evs = sorted(os.path.basename(p)[:-len(a.arm) - 1] for p in glob.glob(f"{C.IMG}/{a.det}/work/*_{a.arm}"))
    tot = collections.Counter()
    with Pool(a.jobs) as pool:
        for acc in pool.imap_unordered(one, [(a.det, a.arm, e) for e in evs]):
            tot.update(acc)
    P = print
    P(f"# doc pdvd/112 terminal planes: det={a.det} arm={a.arm} events={tot['events']} (missing {tot['events_missing']}); "
      f"clusters with a persisted STM record and a dump: {tot['clusters']}")
    P("")
    P("## checks")
    P(f"- dumped vertex charge == RMS over the nonzero planes (0 when < 2): {tot['chk_rms_ok']} of {tot['chk_vertices']} "
      f"({pct(tot['chk_rms_ok'], tot['chk_vertices'])})")
    for c, name in enumerate(CLASSES):
        P(f"  - {name}: {tot[('chk_rms_ok_cls', c)]} of {tot[('chk_n_cls', c)]} ({pct(tot[('chk_rms_ok_cls', c)], tot[('chk_n_cls', c)])})")
    P(f"- control: non-extreme terminals in class 2blank+: {tot['ctl_term_2blank']} (expected 0); "
      f"non-extreme terminals with charge <= 500: {tot['ctl_term_q_le_thr']}")
    P("")
    P("## table A -- Steiner vertices by role x class x ridge offset")
    P("| role | class | vertices | <= 1 cm | 1-2 cm | 2-5 cm | > 5 cm | share > 1 cm |")
    P("|---|---|---|---|---|---|---|---|")
    for r, rn in enumerate(ROLES):
        for c, cn in enumerate(CLASSES):
            n = [tot[("A", r, c, b[0])] for b in BINS]
            N = sum(n)
            P(f"| {rn} | {cn} | {N} | " + " | ".join(str(x) for x in n) + f" | {pct(N - n[0], N)} |")
    P("")
    P("Class mix of each role on (<= 1 cm) and off (> 1 cm) the ridge:")
    P("| role | side | vertices | " + " | ".join(CLASSES) + " |")
    P("|---|---|---|" + "---|" * len(CLASSES))
    for r, rn in enumerate(ROLES):
        for side, sel in (("<= 1 cm", ("<=1",)), ("> 1 cm", ("1-2", "2-5", ">5")), ("1-2 cm", ("1-2",)), ("2-5 cm", ("2-5",))):
            n = [sum(tot[("A", r, c, b)] for b in sel) for c in range(4)]
            N = sum(n)
            P(f"| {rn} | {side} | {N} | " + " | ".join(pct(x, N) for x in n) + " |")
    P("")
    P("Non-extreme terminals: the second-strongest plane charge (for 1blank: the weaker of the two planes the charge test")
    P("graded; the per-plane cut is terminal_charge_threshold = 500):")
    P("| class | side | terminals | " + " | ".join(b[0] for b in Q2BINS) + " |")
    P("|---|---|---|" + "---|" * len(Q2BINS))
    for c in (0, 1):
        for side, sn in (("on", "<= 1 cm"), ("off", "> 1 cm")):
            n = [tot[("Q2", c, side, b[0])] for b in Q2BINS]
            N = sum(n)
            P(f"| {CLASSES[c]} | {sn} | {N} | " + " | ".join(pct(x, N) for x in n) + " |")
    P("")
    P(f"## table B -- persisted round-2 seeds: records {tot['B_records']} (unmatched {tot['B_records_unmatched']}, "
      f"no graph/ridge {tot['B_records_nograph_or_noridge']}); seed points {tot['B_seed_points']} "
      f"(not joined {tot['B_seed_unjoined']}); distinct seed vertices per record")
    P("| side | role | vertices | " + " | ".join(CLASSES) + " |")
    P("|---|---|---|" + "---|" * len(CLASSES))
    for side in ("on", "off"):
        for r, rn in list(enumerate(ROLES)) + [(None, "all")]:
            rr = range(3) if r is None else (r,)
            n = [sum(tot[("B", side, x, c)] for x in rr) for c in range(4)]
            N = sum(n)
            P(f"| {'<= 1 cm' if side == 'on' else '> 1 cm'} | {rn} | {N} | " + " | ".join(pct(x, N) for x in n) + " |")
    P("")
    off_all = sum(tot[("B", "off", x, c)] for x in range(3) for c in range(4))
    off_1b_term = tot[("B", "off", 0, 1)]
    off_1b = sum(tot[("B", "off", x, 1)] for x in range(3))
    P(f"- off-ridge seed vertices that are 1blank: {off_1b} of {off_all} ({pct(off_1b, off_all)}); "
      f"1blank terminals: {off_1b_term} ({pct(off_1b_term, off_all)})")


if __name__ == "__main__":
    main()
