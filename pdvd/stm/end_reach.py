#!/usr/bin/env python3
"""doc pdvd/25 sec 13.6 follow-up: does the imaging charge continue past the fit end?

An accepted PDVD STM pass ends a median 30 cm from any active boundary with a FLAT,
collinear, ~0.9 MIP end (stm_eval_anchor.py, residual_tail.py).  Two readings: the muon
really stops there and the Bragg peak is lost, or the CLUSTER/fit ends early while the
muon keeps going.  This separates them with the imaging charge itself: sum the Bee
`clustering-global` charge in a cylinder that starts --gap cm beyond the fit end and
extends --reach cm along the muon direction, radius --radius, from ANY cluster.

Control: the same cylinder rotated 90 deg about the muon direction (same origin, same
volume) measures uncorrelated nearby charge.  A genuine stop has neither; a prematurely
ended reconstruction has the forward cylinder full and the perpendicular one empty.

The forward count is also split by Bee cluster_id (= cluster ident, 93 % path match, doc 25
sec 13.7): charge from the muon's OWN cluster means the cluster is complete and the skeleton
or the rough path stops short; charge from ANOTHER cluster means clustering split the muon.
Those are different fixes, so the split is what sec 13.9 item 1 is keyed on.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 stm/end_reach.py --tag stm2 --out stm/end_reach_stm2.tsv
"""
import argparse, glob, json, os, re, sys, zipfile
import numpy as np, uproot
HERE = os.path.dirname(os.path.abspath(__file__)); PDVD = os.path.dirname(HERE); sys.path.insert(0, HERE)
import collect_stm_sample as C

def bee_points(workdir):
    """clustering-global layer of the PR Bee dump (same reader as michel_stop_charge.py;
    duplicated because that module runs its analysis at import time)."""
    z = zipfile.ZipFile(os.path.join(workdir, "mabc-pr.zip"))
    for n in z.namelist():
        if n.endswith("-clustering-global.json"):
            d = json.loads(z.read(n))
            return (np.column_stack([d["x"], d["y"], d["z"]]).astype(float),
                    np.asarray(d["q"], float), np.asarray(d["cluster_id"], int))
    raise RuntimeError("no clustering-global layer in " + workdir)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="stm2")
ap.add_argument("--out")
ap.add_argument("--gap", type=float, default=3.0, help="cm; skip this much past the fit end (fit/veto slack)")
ap.add_argument("--reach", type=float, default=25.0, help="cm; cylinder length")
ap.add_argument("--radius", type=float, default=6.0, help="cm; cylinder radius")
ap.add_argument("--veto", type=float, default=2.5, help="cm; drop points this close to the fitted path")
ap.add_argument("--max-absx", type=float, default=305.0)
a = ap.parse_args()

def cyl_sum(P, Q, origin, axis, gap, reach, radius, path):
    d = P - origin
    t = d @ axis
    s = (t > gap) & (t <= gap + reach)
    if not s.any():
        return 0.0, 0, np.empty(0, int)
    perp = np.linalg.norm(d[s] - np.outer(t[s], axis), axis=1)
    k = perp <= radius
    if not k.any():
        return 0.0, 0, np.empty(0, int)
    idx = np.where(s)[0][k]
    keep = np.ones(len(idx), bool)
    for pp in path:
        keep &= np.linalg.norm(P[idx] - pp, axis=1) > a.veto
    return float(Q[idx][keep].sum()), int(keep.sum()), idx[keep]

rows = []
for wd in sorted(glob.glob(os.path.join(PDVD, "work", f"*_{a.tag}"))):
    fp = os.path.join(wd, "tracking-stm.root")
    m = re.match(r"^(\d{6})_(\d+)", os.path.basename(wd))
    if not os.path.exists(fp) or not m or not os.path.exists(os.path.join(wd, "mabc-pr.zip")):
        continue
    run, idx = m.groups()
    try:
        f = uproot.open(fp); P, Q, CID = bee_points(wd)
    except Exception:
        continue
    t = f["T_rec_charge"].arrays(["x", "y", "z", "rr", "ndf", "status"], library="np")
    if len(t["ndf"]) == 0:
        continue
    vd = C.read_verdicts(os.path.join(wd, f"wct_pr_{run}_{idx}.log"))
    for b in sorted(set(t["ndf"].tolist())):
        cid = int(b) // 10
        v = vd.get(cid, {}); mk = t["ndf"] == b
        if v.get("stm") != 1 or v.get("tgm") == 1 or int(t["status"][mk][0]) != 0:
            continue
        rr = t["rr"][mk]
        if mk.sum() < 20:
            continue
        pts = np.stack([t["x"][mk], t["y"][mk], t["z"][mk]], 1)
        end = pts[np.argmin(rr)]
        if abs(end[0]) > a.max_absx:
            continue
        up = (rr >= 5) & (rr < 25)
        if up.sum() < 5:
            continue
        ax = end - pts[up][np.argmax(rr[up])]
        n = np.linalg.norm(ax)
        if n == 0:
            continue
        ax = ax / n
        # a perpendicular axis (any vector orthogonal to ax)
        tmp = np.array([0.0, 0.0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        pax = np.cross(ax, tmp); pax /= np.linalg.norm(pax)
        qf, nf, if_ = cyl_sum(P, Q, end, ax, a.gap, a.reach, a.radius, pts)
        qp, npp, _ = cyl_sum(P, Q, end, pax, a.gap, a.reach, a.radius, pts)
        n_same = int(np.sum(CID[if_] == cid)) if len(if_) else 0
        rows.append((f"{run}_{idx}", cid, qf, nf, qp, npp, n_same, nf - n_same))

r = np.array([(x[2], x[3], x[4], x[5], x[6], x[7]) for x in rows], float)
print(f"accepted STM passes with a usable end and a Bee dump (tag {a.tag}): {len(rows)}")
if len(rows):
    qf, nf, qp, npp = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    print(f"\n  cylinder: {a.gap}-{a.gap+a.reach} cm past the fit end, radius {a.radius} cm, "
          f"path veto {a.veto} cm; charge in electrons")
    print(f"  FORWARD (along the muon):    median {np.median(qf):.3g} e over {np.median(nf):.0f} points; "
          f"empty (0 points): {100*np.mean(nf == 0):.0f} %")
    print(f"  PERPENDICULAR (control):     median {np.median(qp):.3g} e over {np.median(npp):.0f} points; "
          f"empty (0 points): {100*np.mean(npp == 0):.0f} %")
    for thr in (10, 30, 100):
        print(f"  >= {thr} points: forward {100*np.mean(nf >= thr):.0f} %   perpendicular {100*np.mean(npp >= thr):.0f} %")
    ns, no = r[:, 4], r[:, 5]
    sel = nf >= 10
    print(f"\n  of the {int(sel.sum())} passes with >= 10 forward points, whose charge is it?")
    print(f"    the muon's OWN cluster dominates (same > other): {100*np.mean(ns[sel] > no[sel]):.0f} %"
          f"   -> the cluster is complete, the skeleton / rough path stops short")
    print(f"    ANOTHER cluster dominates:                       {100*np.mean(no[sel] > ns[sel]):.0f} %"
          f"   -> clustering split the muon")
    print(f"    median same-cluster points {np.median(ns[sel]):.0f}, other-cluster {np.median(no[sel]):.0f};"
          f" purely same-cluster (other == 0): {100*np.mean(no[sel] == 0):.0f} %; purely other: {100*np.mean(ns[sel] == 0):.0f} %")
if a.out:
    with open(a.out, "w") as fo:
        fo.write("event\tcluster\tq_forward\tn_forward\tq_perp\tn_perp\tn_fwd_same_cluster\tn_fwd_other_cluster\n")
        for x in rows:
            fo.write(f"{x[0]}\t{x[1]}\t{x[2]:.0f}\t{x[3]}\t{x[4]:.0f}\t{x[5]}\n")
    print("\nwrote", a.out)
