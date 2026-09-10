#!/usr/bin/env python3
"""doc pdvd/62 (T3) -- doc pdvd/54 sec 2.5's question: of the unfitted lumps
near a stop that emit NO pr54 drop line, was a residual component ever
PROPOSED there (and killed by a named filter), or did step 1 of
find_other_segments tag its points as "already covered" so no component
existed at all?

Lump definition = stm_unfitted_lump_steiner.py's (doc pdvd/54 sec 2.4): in-bundle
image points within 20 cm of the stop, > 3 cm from every fitted-segment
point, greedy 3 cm linkage, best group by charge, kept if >= 20 points and
>= 2e5 e.  Lumps are defined on the BASELINE prep (the same payload
geometry doc 54 used) and matched into the fos arm's log by bbox overlap.

Usage: python3 d62_fos.py <fos_arm> <baseline_prep> <baseline_arm>
"""
import sys, os, re, glob, json, collections
import numpy as np
from scipy.spatial import cKDTree
HERE = "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan"
sys.path.insert(0, HERE)
import census_lib as C

fos_arm, prep, base_arm = sys.argv[1], sys.argv[2], sys.argv[3]
W = C.WORK
RE_DROP = re.compile(r"pr54 isolated-residual drop: cluster (\d+) n_points=(\d+) length=([\d.]+) cm dir_mag=([\d.]+) cm v1=\(([-\d.]+),([-\d.]+),([-\d.]+)\) v2=\(([-\d.]+),([-\d.]+),([-\d.]+)\)")
RE_KEEP = re.compile(r"pr54 keep-isolated(?: near-anchor)?: cluster (\d+) n_points=(\d+) length=([\d.]+) cm.*v1=\(([-\d.]+),([-\d.]+),([-\d.]+)\) v2=\(([-\d.]+),([-\d.]+),([-\d.]+)\)")
RE_FOS = re.compile(r"pr67 fos (\S+): cluster=(\d+) group=(-?\d+) npts=(\d+) len=([\d.]+) cm nnf=(\d+) .*bbox=\[([-\d.]+),([-\d.]+)\]x\[([-\d.]+),([-\d.]+)\]x\[([-\d.]+),([-\d.]+)\] -> (.*)$")
RE_FOS8 = re.compile(r"pr67 fos step8: cluster=(\d+) group=(\d+) npts=(\d+) front=\(([-\d.]+),([-\d.]+),([-\d.]+)\) -> (.*)$")
RE_HDR = re.compile(r"pr67 fos: cluster=(\d+) existing_segments=(\d+) steiner_N=(\d+) tagged=(\d+) terminals=(\d+)")

def parse(ev, arm):
    logs = glob.glob(os.path.join(W, ev + "_" + arm, "wct_pr_*.log"))
    drops, keeps, comps, hdrs = [], [], [], {}
    if not logs: return drops, keeps, comps, hdrs
    for line in open(logs[0], errors="replace"):
        m = RE_DROP.search(line)
        if m:
            g = m.groups(); drops.append((int(g[0]), int(g[1]), float(g[2]), np.array(g[4:7], float), np.array(g[7:10], float))); continue
        m = RE_KEEP.search(line)
        if m:
            g = m.groups(); keeps.append((int(g[0]), int(g[1]), float(g[2]), np.array(g[3:6], float), np.array(g[6:9], float), "near-anchor" in line)); continue
        m = RE_FOS.search(line)
        if m:
            g = m.groups()
            lo = np.array([g[6], g[8], g[10]], float); hi = np.array([g[7], g[9], g[11]], float)
            comps.append(dict(stage=g[0], cid=int(g[1]), grp=int(g[2]), npts=int(g[3]), len=float(g[4]), lo=lo, hi=hi, verdict=g[12].strip())); continue
        m = RE_FOS8.search(line)
        if m:
            g = m.groups(); f = np.array(g[3:6], float)
            comps.append(dict(stage="step8", cid=int(g[0]), grp=int(g[1]), npts=int(g[2]), lo=f, hi=f, verdict=g[6].strip())); continue
        m = RE_HDR.search(line)
        if m:
            g = list(map(int, m.groups())); hdrs.setdefault(g[0], []).append(dict(existing=g[1], N=g[2], tagged=g[3], terminals=g[4]))
    return drops, keeps, comps, hdrs

rec = C.load_record()
P, missing = C.load_payloads(prep, rec)
rows = []
for k in sorted(P):
    p = P[k]
    mu = p.get("muon") or {}; segs = (p.get("pf") or {}).get("seg") or []; im = p.get("image_near") or {}
    if not mu.get("x") or not segs or not im.get("x"): continue
    MX = np.array([mu["x"], mu["y"], mu["z"]]).T
    stop = MX[int(np.argmin(np.array(mu["rr"])))]
    SP = np.array([[x, y, z] for s in segs for x, y, z in zip(s["x"], s["y"], s["z"])])
    IX = np.array([im["x"], im["y"], im["z"]]).T; Q = np.array(im["q"]); B = np.array(im["b"])
    dd = np.linalg.norm(IX - stop, axis=1); m = (dd <= 20) & (B == 1)
    if m.sum() == 0: continue
    dfit, _ = cKDTree(SP).query(IX[m], k=1); un = dfit > 3.0
    if un.sum() == 0: continue
    Pp = IX[m][un]; q = Q[m][un]
    t = cKDTree(Pp); seen = np.zeros(len(Pp), bool); best = None
    for i in range(len(Pp)):
        if seen[i]: continue
        st = [i]; seen[i] = True; grp = [i]
        while st:
            j = st.pop()
            for kk in t.query_ball_point(Pp[j], 3.0):
                if not seen[kk]: seen[kk] = True; st.append(kk); grp.append(kk)
        if best is None or q[grp].sum() > best[1]: best = (len(grp), q[grp].sum(), Pp[grp])
    if best[0] < 20 or best[1] < 2e5: continue
    ev = k.split("/")[0]; cid = p["cluster_id"]; lump = best[2]
    lo, hi = lump.min(0) - 5.0, lump.max(0) + 5.0
    dB, kB, _, _ = parse(ev, base_arm)
    dF, kF, cF, hF = parse(ev, fos_arm)
    def near(v1, v2): return min(np.linalg.norm(lump - v1, axis=1).min(), np.linalg.norm(lump - v2, axis=1).min()) <= 5.0
    drop_base = [d for d in dB if near(d[3], d[4])]
    keep_fos = [d for d in kF if near(d[3], d[4])]
    drop_fos = [d for d in dF if near(d[3], d[4])]
    # a proposed component overlaps the lump if its bbox (padded 5 cm) intersects the lump's bbox
    comp = [c for c in cF if c["cid"] == cid and np.all(c["hi"] >= lo) and np.all(c["lo"] <= hi)]
    verdicts = collections.Counter((c["stage"], c["verdict"].split()[0] if c["verdict"] else "?") for c in comp)
    h = hF.get(cid, [])
    rows.append((k, C.bare(rec[k]["verdict"]) if k in rec else "?", best[0], best[1], len(drop_base), len(drop_fos), len(keep_fos), len(comp), dict(verdicts),
                 (h[0]["tagged"], h[0]["N"], h[0]["terminals"]) if h else None))
    agg = globals().setdefault("AGG", {})
    cls = "with_drop" if drop_base else ("component_only" if comp else "nothing")
    for c in comp:
        agg.setdefault(cls, collections.Counter())[(c["stage"], c["verdict"])] += 1
print("lumps (>= 20 pts, >= 2e5 e, in-bundle, unfitted, within 20 cm of the stop) on the baseline payloads: %d" % len(rows))
print("%-16s %-11s %5s %9s %5s %5s %5s %5s  %-34s %s" % ("item", "scan", "blobs", "charge", "dropB", "dropF", "keepF", "comps", "component verdicts (stage,verdict)", "fos hdr (tagged,N,terminals)"))
n_drop = n_comp_only = n_nothing = 0
for r in rows:
    k, sc, nb, qq, db, df, kf, nc, vd, hd = r
    if db: n_drop += 1
    elif nc: n_comp_only += 1
    else: n_nothing += 1
    print("%-16s %-11s %5d %9.2e %5d %5d %5d %5d  %-34s %s" % (k, sc, nb, qq, db, df, kf, nc, ",".join("%s/%s:%d" % (a, b, n) for (a, b), n in vd.items())[:34], hd))
print("\nlumps WITH a pr54 drop in the baseline: %d" % n_drop)
print("lumps with NO drop but a proposed fos component overlapping them (killed by a named filter): %d" % n_comp_only)
print("lumps with NO drop and NO proposed component (step-1 'already covered' tagging, or no terminals there): %d" % n_nothing)
print("(lumps whose drop the fos arm's anchor keep converted to a keep: %d)" % sum(1 for r in rows if r[6] > 0))

print("\nverdicts of the proposed components, by lump class (stage, verdict -> n):")
for cls, cnt in sorted(globals().get("AGG", {}).items()):
    print("  %-15s" % cls, ", ".join("%s/%s:%d" % (a, b, n) for (a, b), n in cnt.most_common()))
