#!/usr/bin/env python3
"""doc sbnd_xin/117 sec 3: what the 2-step (dual) chain actually did, read off the arms that
already ran -- no new compute.

SBND production runs the whole vertex-decision sequence twice: a second, exclusion-free PR pass
(TaggerCheckNeutrino::run_dual_chain_off_pass, m_fit_exclusion forced false) proposes the neutrino
vertex and the production pass decides, by snapping the proposal to its nearest own candidate and
accepting iff d <= dual_chain_transfer_max (2 cm).  doc pr/112 sec 11.  Every event of every arm
in docs 115/116 carries the record of that in its calib dump's vertex_scoreboard:

    dual_chain = {used, mode, transfer, has_vertex, x,y,z, nearest_id, d, agree, transferred,
                  n_voxels, off_ms}            off_ms = the second pass's own wall time
    rows[]     = every candidate vertex with its composite rerank score `total`, `dl_winner`,
                 `trad_winner`, and its position
    final_*    = what shipped

So the census -- how often the second pass ran, agreed, transferred, and how much it cost -- is
exact.  The truth score is a PROXY: for the events where the transfer moved the pick away from the
composite winner (argmax `total`), it compares the distance to the nearest true in-FV vertex of the
transferred row against that of the composite winner.  It is a proxy because the no-dual-chain
arm's acceptance logic (min_accept floor, distance gate, traditional fallback) is not replayed
here; doc 117's c1e / t1e arms measure the same thing by running it.  The two are compared in the
doc, and the proxy is never quoted alone.

With no arguments it reports the two 2-step corners (c2, t2).  Extra arms are added as
`label:sample:arm_dir:products_dir` (the products dir may be empty for a beam-off arm), which is how
the doc-117 1-step cells get their route census -- there the dual block is absent and the
interesting column is `route`.

Usage: python3 scripts/d117/dual_chain_census.py [label:sample:arm:products ...]
                                                 > docs/117_figs/117_dual_census.txt
"""
import collections
import csv
import glob
import json
import math
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARMS = [("c2 (doc-115 baseline)", "cv", "work-r3cv-d115pr", "products/d115/cv"),
        ("c2 (doc-115 baseline)", "nuecc", "work-r3nue-d115pr", "products/d115/nuecc"),
        ("t2 (doc-116 tfull)", "cv", "work-r3cv-d116tfull", "products/d116/cv-tfull"),
        ("t2 (doc-116 tfull)", "nuecc", "work-r3nue-d116tfull", "products/d116/nuecc-tfull"),
        ("c2 (doc-115 baseline)", "off", "work-r3off-d115pr", ""),
        ("t2 (doc-116 tfull)", "off", "work-r3off-d116tfull", "")]


def in_fv(x, y, z):
    return 5.0 < abs(x) < 190.0 and abs(y) < 190.0 and 10.0 < z < 450.0


def sign_p(k, n):
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def rse_of(d):
    for p in glob.glob(os.path.join(d, "nusel-evt*.tsv")):
        with open(p) as fh:
            fh.readline()
            r = fh.readline().split()
        if len(r) >= 3:
            return (r[0], r[1], r[2])
    return None


def one(d):
    """-> dict for one pr_evt dir, or None when there is no scoreboard"""
    cal = glob.glob(os.path.join(d, "calib-pr-evt*.json"))
    if not cal:
        return None
    try:
        j = json.load(open(cal[0]))
    except Exception:
        return None
    vs = j.get("vertex_scoreboard") or {}
    if not vs.get("filled"):
        return None
    dc = vs.get("dual_chain") or {}
    rows = vs.get("rows") or []
    best = max(rows, key=lambda r: r.get("total", 0.0)) if rows else None
    trad = next((r for r in rows if r.get("trad_winner")), None)
    nid = dc.get("nearest_id")
    snapped = next((r for r in rows if r.get("vertex_id") == nid), None)
    return {
        "rse": rse_of(d), "route": vs.get("route"), "dl_accepted": bool(vs.get("dl_accepted")),
        "used": bool(dc), "agree": bool(dc.get("agree")), "transferred": bool(dc.get("transferred")),
        "has_vertex": bool(dc.get("has_vertex")), "d": dc.get("d"), "off_ms": dc.get("off_ms"),
        "n_rows": len(rows),
        "final": (vs.get("final_x"), vs.get("final_y"), vs.get("final_z")),
        "snapped_xyz": (snapped["x"], snapped["y"], snapped["z"]) if snapped else None,
        "best_xyz": (best["x"], best["y"], best["z"]) if best else None,
        "trad_xyz": (trad["x"], trad["y"], trad["z"]) if trad else None,
        "moved": bool(snapped and best and snapped.get("vertex_id") != best.get("vertex_id")),
    }


def truth_map(prod):
    """(run,subrun,event) -> [ (x,y,z) of every true interaction with the vertex in the FV ]"""
    out = {}
    p = os.path.join(SX, prod, "truth.tsv")
    if not prod or not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p), delimiter="\t"):
        x, y, z = float(r["vx"]), float(r["vy"]), float(r["vz"])
        if in_fv(x, y, z):
            out.setdefault((r["run"], r["subrun"], r["event"]), []).append((x, y, z))
    return out


def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")


def main(extra=()):
    print("# doc 117 sec 3 -- the 2-step (dual) chain census, from the arms that already ran")
    for label, s, arm, prod in list(ARMS) + list(extra):
        dirs = sorted(glob.glob(os.path.join(SX, arm, "f*", "pr_evt*")) +
                      glob.glob(os.path.join(SX, arm, "pr_evt*")))
        if not dirs:
            print(f"\n## {label} / {s}: arm not on disk ({arm})")
            continue
        with ProcessPoolExecutor(12) as ex:
            res = [r for r in ex.map(one, dirs, chunksize=8) if r]
        T = truth_map(prod)
        n = len(res)
        routes = collections.Counter(r["route"] for r in res)
        used = [r for r in res if r["used"]]
        hv = [r for r in used if r["has_vertex"]]
        tr = [r for r in used if r["transferred"]]
        mv = [r for r in tr if r["moved"]]
        offs = [r["off_ms"] for r in used if r["off_ms"]]
        ds = [r["d"] for r in hv if r["d"] is not None]
        print(f"\n## {label} / {s}: {len(dirs)} events, {n} with a filled scoreboard")
        print("  routes                   : " + ", ".join(f"{k}={v}" for k, v in routes.most_common()))
        print(f"  second pass ran          : {len(used)} ({100.0*len(used)/max(n,1):.1f} %)")
        print(f"  ... proposed a vertex    : {len(hv)}")
        print(f"  ... chains agreed        : {sum(1 for r in used if r['agree'])}")
        print(f"  ... transfer accepted    : {len(tr)}")
        print(f"  ... and it MOVED the pick: {len(mv)}  (the rest snapped onto the composite winner)")
        if ds:
            print(f"  transfer distance (cm)   : median {st.median(ds):.3f}  p90 {q(ds,.9):.2f}  max {max(ds):.2f}")
        if offs:
            print(f"  second-pass cost off_ms  : median {st.median(offs):.0f}  p90 {q(offs,.9):.0f}  "
                  f"sum {sum(offs)/3600000.0:.2f} core-h over the arm")
        if not T or not mv:
            continue
        closer = further = same = nokey = 0
        for r in mv:
            k = r["rse"]
            tv = T.get(k) if k else None
            if not tv or not r["snapped_xyz"] or not r["best_xyz"]:
                nokey += 1
                continue
            da = min(math.dist(r["snapped_xyz"], t) for t in tv)
            db = min(math.dist(r["best_xyz"], t) for t in tv)
            if da < db - 0.01:
                closer += 1
            elif da > db + 0.01:
                further += 1
            else:
                same += 1
        print(f"  PROXY truth score on the {len(mv)} events where the transfer moved the pick:")
        print(f"    transferred vertex is CLOSER to a true in-FV vertex than the composite winner: {closer}")
        print(f"    FURTHER: {further}   same: {same}   no truth key: {nokey}")
        print(f"    exact two-sided sign test p = {sign_p(min(closer, further), closer + further):.3g}")


if __name__ == "__main__":
    ex = []
    for a in sys.argv[1:]:
        f = a.split(":")
        ex.append((f[0], f[1], f[2], f[3] if len(f) > 3 else ""))
    main(ex)
