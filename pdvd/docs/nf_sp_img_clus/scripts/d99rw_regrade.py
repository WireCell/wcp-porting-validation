#!/usr/bin/env python3
"""doc pdvd/99 sec 4.4 -- re-grade the blind swap scan (sec 6.4) on an arm pair run with each event's real readout window.

    python3 d99rw_regrade.py --on p99rgon --prod p99rgprod > ../d99/rw_regrade_guard.txt
    python3 d99rw_regrade.py --on p99rwon --prod p99rwprod > ../d99/rw_regrade_full.txt

The scan's items were judged on p98vonq (on_only) and p96vprod (prod_only).  Each scanned key is carried to the new arm of
its side: where the new arm's pctree is the source's own file (a guard arm, d99rw_arms.sh WAVE=guard) the key names the
same object and is taken as is; otherwise (re-clustered) it is carried by geometry with the record carry's own matcher and
pre-registered thresholds (d99_match.match_cluster / status_of on the Bee clustering-global points, as
d99_swap_scan_set.rows_of does), and a non-ok match is reported as "object not matched".

  1  the swap sets before (p98vonq vs p96vprod) and after (ON vs PROD), by run group, same matcher;
  2  every scanned item: still in its group / its own arm no longer tags it / the other side now tags it too / not matched;
  3  the 19 frame-edge items of d99/swap_scan_window_flagged.tsv one by one, with the guard line for their cluster;
  4  purity on the scanned items that stay in their set (Wilson 68 %), the difference with a bootstrap (seed 20260915,
     as sec 6.4) -- NOT pre-registered: the sets changed, and the unscanned members of the new sets are counted, not
     judged.
"""
import argparse, collections, csv, glob, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
import d99_match as M
import d99_population as Pp
import d99_swap_scan_set as W
import d99_swap_scan_score as S

ND = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus"
WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
SW99 = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/scan/pdvd_stm_michel_sw99_verdicts.json"
BASE = {"on": "p98vonq", "prod": "p96vprod"}


def run_group(key):
    return "039349" if key.startswith("039349") else "039252+3"


def pctree_real(evt, arm):
    f = sorted(glob.glob(f"{WORK}/{evt}_{arm}/pctree-evt*.tar.gz"))
    return os.path.realpath(f[0]) if f else None


def guard_lines(job):
    evt, arm = job
    out = {}
    for lg in glob.glob(f"{WORK}/{evt}_{arm}/wct_pr_*.log"):
        for line in open(lg, errors="replace"):
            m = re.search(r"readout_edge_guard: cluster (\d+) rejected: (.*)", line)
            if m:
                out[f"{evt}/{m.group(1)}"] = m.group(2).strip()
    return out


def carry(job):
    """(evt, base arm, new arm, [cluster ids]) -> {old key: (new key or None, how)}"""
    evt, base, new, cids = job
    if pctree_real(evt, base) is not None and pctree_real(evt, base) == pctree_real(evt, new):
        return {f"{evt}/{c}": (f"{evt}/{c}", "same pctree") for c in cids}
    ba, bn = M.load_bee(evt, base), M.load_bee(evt, new)
    if ba is None or bn is None:
        return {f"{evt}/{c}": (None, "no Bee dump") for c in cids}
    tree = cKDTree(bn[0])
    out = {}
    for c in cids:
        pts = ba[0][ba[1] == c]
        if len(pts) == 0:
            out[f"{evt}/{c}"] = (None, "no base points"); continue
        m = M.match_cluster(pts, bn[0], bn[1], tree)
        st = M.status_of(m)
        out[f"{evt}/{c}"] = (f"{evt}/{m['best']}", "geometry ok") if st == "ok" else (None, f"geometry {st}")
    return out


def sets(ev, on, prod, jobs):
    with ProcessPoolExecutor(jobs) as ex:
        ro = [x for xx in ex.map(W.rows_of, [(e, on, prod) for e in ev]) for x in xx]
        rp = [x for xx in ex.map(W.rows_of, [(e, prod, on) for e in ev]) for x in xx]
    return ({r["key"]: r for r in ro if r["status"] != "is_stm"}, {r["key"]: r for r in rp if r["status"] != "is_stm"},
            {r["key"] for r in ro}, {r["key"] for r in rp})


def purity(rows):
    k = sum(r == "stop" for r in rows); n = sum(r in ("stop", "non") for r in rows)
    p, lo, hi = S.wilson(k, n)
    return k, n, p, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True)
    ap.add_argument("--prod", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--nboot", type=int, default=10000)
    a = ap.parse_args()
    ev = Pp.events()
    NEW = {"on": a.on, "prod": a.prod}
    print(f"# doc pdvd/99 sec 4.4 -- the sec 6.4 swap scan on {a.on} vs {a.prod} (each event's real readout window); "
          f"scanned keys carried from {BASE['on']} / {BASE['prod']}; d99rw_regrade.py")

    on0, prod0, ison0, isprod0 = sets(ev, BASE["on"], BASE["prod"], a.jobs)
    on1, prod1, ison1, isprod1 = sets(ev, a.on, a.prod, a.jobs)
    print("\n== 1. swap sets (is_stm on one arm, the matched object not is_stm on the other)")
    for name, s0, s1, t0, t1 in (("on_only", on0, on1, ison0, ison1), ("prod_only", prod0, prod1, isprod0, isprod1)):
        for rg in ("039349", "039252+3", "all"):
            f = (lambda k: True) if rg == "all" else (lambda k, rg=rg: run_group(k) == rg)
            A = {k for k in s0 if f(k)}; B = {k for k in s1 if f(k)}
            TA = {k for k in t0 if f(k)}; TB = {k for k in t1 if f(k)}
            print(f"  {name:9s} {rg:9s} is_stm {len(TA):3d} -> {len(TB):3d}; set {len(A):3d} -> {len(B):3d}"
                  + (f": stay {len(A & B):3d}, leave {len(A - B):3d}, enter {len(B - A):3d} (ids comparable: same pctrees)"
                     if all(pctree_real(e, NEW[g]) == pctree_real(e, BASE[g]) for e in ev for g in ("on", "prod")) else
                     " (re-clustered: set membership by id is not comparable; see section 2 for the scanned objects)"))

    key = {r["key"]: r for r in csv.DictReader((l for l in open(f"{ND}/d99/swap_scan_key.tsv") if not l.startswith("#")), delimiter="\t")}
    ver = {r["key"]: r for r in json.load(open(SW99))}
    assert set(ver) <= set(key), "sw99 key not in the scan key"
    flagged = {l.split("\t")[0] for l in open(f"{ND}/d99/swap_scan_window_flagged.tsv") if l.strip() and not l.startswith("#")}
    jobs = collections.defaultdict(list)
    for k, v in ver.items():
        g = "on" if v["group"] == "on_only" else "prod"
        assert v["display_arm"] == BASE[g], (k, v["display_arm"])
        jobs[(k.split("/")[0], g)].append(int(k.split("/")[1]))
    cmap = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for res in ex.map(carry, [(e, BASE[g], NEW[g], c) for (e, g), c in sorted(jobs.items())]):
            cmap.update(res)
        guard = {"on": {}, "prod": {}}      # per side: cluster ids are arm-specific, never pool the two arms' lines
        for x in ("on", "prod"):
            for g in ex.map(guard_lines, [(e, NEW[x]) for e in ev]):
                guard[x].update(g)
    how = collections.Counter(h for _, h in cmap.values())
    print(f"\n== 2. the scanned items (carry: {dict(how)})")
    rows = []
    for k, v in sorted(ver.items()):
        g = "on" if v["group"] == "on_only" else "prod"
        s1, t1 = (on1, ison1) if g == "on" else (prod1, isprod1)
        nk, h = cmap[k]
        fate = ("not matched" if nk is None else "stay" if nk in s1 else
                "own arm untagged" if nk not in t1 else "other side now tags")
        rows.append(dict(key=k, new_key=nk, how=h, group=v["group"], cls=S.cls(v["verdict"]), verdict=v["verdict"],
                         conf=v["confidence"], fate=fate, flagged=k in flagged, run=run_group(k),
                         guard=guard[g].get(nk, "") if nk else ""))
    FATES = ("stay", "own arm untagged", "other side now tags", "not matched")
    for grp in ("on_only", "prod_only"):
        for rg in ("039349", "039252+3", "all"):
            R = [r for r in rows if r["group"] == grp and (rg == "all" or r["run"] == rg)]
            c = collections.Counter((r["fate"], r["cls"]) for r in R)
            print(f"  {grp:9s} {rg:9s} scanned {len(R):3d}: " + "; ".join(
                f"{f} {sum(v for (ff, _), v in c.items() if ff == f)} (stop {c[(f, 'stop')]}, non {c[(f, 'non')]}, excl {c[(f, 'excl')]})"
                for f in FATES))

    print("\n== 3. the 19 frame-edge items (sec 6.4, post hoc regex)")
    for r in sorted((r for r in rows if r["flagged"]), key=lambda r: (r["group"], r["key"])):
        print(f"  {r['key']:14s} {r['group']:9s} scan {r['verdict']:12s} {r['conf']:6s} -> {r['fate']:20s} "
              f"(new {r['new_key']}) {r['guard'][:100]}")
    fl = [r for r in rows if r["flagged"]]
    print(f"  flagged items leaving their set: {sum(r['fate'] != 'stay' for r in fl)}/{len(fl)}; unflagged items leaving: "
          f"{sum(r['fate'] != 'stay' for r in rows if not r['flagged'])}/{sum(not r['flagged'] for r in rows)}")
    print("  unflagged leavers: " + str([(r["key"], r["group"], r["verdict"], r["fate"], r["guard"][:60])
                                        for r in rows if r["fate"] != "stay" and not r["flagged"]]))

    print("\n== 4. purity on the scanned items that stay (NOT pre-registered; unscanned members of the new sets not judged)")
    P = {}
    for grp, s1 in (("on_only", on1), ("prod_only", prod1)):
        R = [r["cls"] for r in rows if r["group"] == grp and r["fate"] == "stay"]
        k, n, p, lo, hi = purity(R); P[grp] = np.array([1 if c == "stop" else 0 for c in R if c in ("stop", "non")])
        scanned_new = {r["new_key"] for r in rows if r["group"] == grp and r["new_key"]}
        unscanned = sorted(x for x in s1 if x not in scanned_new)
        print(f"  {grp:9s} scanned and staying {len(R):3d}: stoppers {k}/{n} = {p:.3f} [{lo:.3f}, {hi:.3f}]; new set {len(s1)}, "
              f"unscanned members {len(unscanned)} {dict(collections.Counter(run_group(x) for x in unscanned))}")
    rng = np.random.default_rng(a.seed)
    d = [P["on_only"][rng.integers(0, len(P["on_only"]), len(P["on_only"]))].mean()
         - P["prod_only"][rng.integers(0, len(P["prod_only"]), len(P["prod_only"]))].mean() for _ in range(a.nboot)]
    d0 = P["on_only"].mean() - P["prod_only"].mean()
    q68, q95 = np.quantile(d, [0.16, 0.84]), np.quantile(d, [0.025, 0.975])
    print(f"  difference {d0:+.3f}  68 % [{q68[0]:+.3f}, {q68[1]:+.3f}]  95 % [{q95[0]:+.3f}, {q95[1]:+.3f}]")
    print(f"  (sec 6.4 pre-registered, 10000-tick window everywhere: 0.846 vs 0.907, -0.061 [-0.174, +0.060])")
    for grp, s1 in (("on_only", on1), ("prod_only", prod1)):
        scanned_new = {r["new_key"] for r in rows if r["group"] == grp and r["new_key"]}
        print(f"  unscanned {grp}: {sorted(x for x in s1 if x not in scanned_new)}")


if __name__ == "__main__":
    sys.exit(main())
