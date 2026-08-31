#!/usr/bin/env python3
"""doc pr/104 -- junction-snap census: would a main-vertex re-point to a nearby
>=N-prong junction land nearer the owner's click?  Forked in spirit from
pr103_busy_vertex_census.py (graph reading) and pr103_click_topology_ab.py
(vtx100 label join); reads calib-pr-evt*.json only.

Usage: pr104_junction_census.py <arm> [<arm> ...] [--tags vtx100] [--r 4.0]
          [--min-arm 3.0] [--collinear 150] [--tsv OUT] [--all]

Per labelled event (click = vtx100 truth), M = reco main vertex:
  J candidates : vertices of M's cluster reachable from M through a chain of
                 segments of total path <= --r (chain members are "the M-J path")
  prong(X)     : incident segment of X not on the M-J path, track-like
                 (flag_shower=0), path length >= --min-arm; a shorter stub whose
                 far vertex has degree 2 and continues into such a track also
                 counts (direction taken from the continuation)
  strength(X)  : number of distinct prong DIRECTION classes at X; two prongs
                 whose outward chords fold to > --collinear deg are one class
                 (a track passing through X)
  tier A       : strength(M) == 0 and strength(J) >= 2          (M is a stub end / kink point)
  tier B       : strength(M) >= 1, strength(J) >= 1, strength(M)+strength(J) >= 3:
                 joint least-squares intersection of ALL prong lines of M and J
                 (direction = PCA axis of the prong's fit points 1-8 cm out from
                 its own vertex); snap iff the fit point is nearer J than M by
                 > --margin and the RMS transverse residual < --rms
  decision     : snap -> J (best J = max strength, tie nearest) or stay
  verdict      : FIX   click nearer J than M by > 0.5 cm and decision snap
                 BREAK click nearer M than J by > 0.5 cm and decision snap
                 MISS  click nearer J by > 0.5 cm and decision stay
                 OK    otherwise (stay and click not nearer J)
Without --all only events with at least one J candidate are printed.
"""
import json, math, sys, os, argparse, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vtx_rules"))
import vtx_io

def P(v): f = v["fit"]; return np.array([f["x"], f["y"], f["z"]])
def seg_pts(s): return np.array([[p["x"], p["y"], p["z"]] for p in s["points"]]) if s["points"] else np.zeros((0, 3))
def load(arm, ev):
    p = f"{arm}/pr_evt{ev}/calib-pr-evt{ev}.json"
    try: return json.load(open(p))
    except Exception: return None

def path_len(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0

def oriented(s, vid):
    """fit points of s ordered so index 0 is at vertex vid"""
    pts = seg_pts(s)
    return pts if s["start_vertex_id"] == vid else pts[::-1]

def chord_dir(pts, rin=1.0, rout=8.0):
    """PCA axis of points in (rin, rout] cm from pts[0], oriented outward; None if <3 pts"""
    if len(pts) < 2: return None, None
    d = np.linalg.norm(pts - pts[0], axis=1)
    sel = pts[(d > rin) & (d <= rout)]
    if len(sel) < 3:
        sel = pts[d > 0.3]
        if len(sel) < 2: return None, None
        v = sel[-1] - pts[0]; n = np.linalg.norm(v)
        return (v / n if n > 0 else None), sel[0]
    c = sel.mean(axis=0)
    u, sv, vt = np.linalg.svd(sel - c)
    ax = vt[0]
    if np.dot(ax, sel[-1] - pts[0]) < 0: ax = -ax
    return ax, sel[0]

class G:
    def __init__(self, d):
        self.V = {v["id"]: v for v in d["vertices"]}
        self.S = {s["id"]: s for s in d["segments"]}
        self.inc = collections.defaultdict(list)
        for s in d["segments"]:
            self.inc[s["start_vertex_id"]].append(s)
            if s["end_vertex_id"] != s["start_vertex_id"]:
                self.inc[s["end_vertex_id"]].append(s)
    def other(self, s, vid): return s["end_vertex_id"] if s["start_vertex_id"] == vid else s["start_vertex_id"]

def prongs(g, vid, exclude_segs, min_arm):
    """list of (dir, anchor_point, length, seg_id) for prongs at vid"""
    out = []
    for s in g.inc[vid]:
        if s["id"] in exclude_segs: continue
        pts = oriented(s, vid)
        L = path_len(pts)
        if s["flag_shower"]: continue
        if L >= min_arm:
            ax, anc = chord_dir(pts)
            if ax is not None: out.append((ax, anc, L, s["id"]))
            continue
        # stub with a degree-2 far vertex continuing into a track
        far = g.other(s, vid)
        cont = [t for t in g.inc[far] if t["id"] != s["id"] and t["id"] not in exclude_segs]
        if len(cont) == 1 and not cont[0]["flag_shower"]:
            cpts = oriented(cont[0], far)
            if path_len(cpts) >= min_arm:
                allp = np.vstack([pts, cpts[1:]])
                ax, anc = chord_dir(allp)
                if ax is not None: out.append((ax, anc, L + path_len(cpts), cont[0]["id"]))
    return out

def strength(pr, collinear):
    """direction classes: merge prongs whose outward dirs fold > collinear deg"""
    n = 0; used = [False] * len(pr)
    for i in range(len(pr)):
        if used[i]: continue
        used[i] = True; n += 1
        for j in range(i + 1, len(pr)):
            if used[j]: continue
            c = float(np.clip(np.dot(pr[i][0], pr[j][0]), -1, 1))
            if math.degrees(math.acos(c)) > collinear: used[j] = True
    return n

def joint_fit(pr_list):
    """LSQ point minimising sum of squared transverse distances to lines (p_i, d_i)"""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for ax, anc, L, sid in pr_list:
        Pm = np.eye(3) - np.outer(ax, ax)
        A += Pm; b += Pm @ anc
    try: x = np.linalg.solve(A + 1e-9 * np.eye(3), b)
    except np.linalg.LinAlgError: return None, None
    res = [float(np.linalg.norm((np.eye(3) - np.outer(ax, ax)) @ (x - anc))) for ax, anc, L, sid in pr_list]
    return x, float(np.sqrt(np.mean(np.square(res))))

def reach(g, mvid, R):
    """vertices reachable from M via segment chains with path <= R; returns {vid: (path, segs_on_path)}"""
    best = {mvid: (0.0, frozenset())}
    frontier = [mvid]
    while frontier:
        nxt = []
        for v in frontier:
            pl, segs = best[v]
            for s in g.inc[v]:
                if s["id"] in segs: continue
                L = path_len(seg_pts(s))
                w = g.other(s, v)
                if pl + L > R: continue
                if w not in best or pl + L < best[w][0]:
                    best[w] = (pl + L, segs | {s["id"]}); nxt.append(w)
        frontier = nxt
    best.pop(mvid, None)
    return best

def analyse(d, a):
    mv = d.get("main_vertex")
    if not mv or not d.get("vertices"): return None
    mvp = np.array([mv["x"], mv["y"], mv["z"]])
    g = G(d)
    mains = [v for v in d["vertices"] if v.get("is_main")]
    mvid = mains[0]["id"] if mains else min(g.V, key=lambda k: np.linalg.norm(P(g.V[k]) - mvp))
    M = g.V[mvid]
    cands = reach(g, mvid, a.r)
    cands = {k: v for k, v in cands.items() if g.V[k]["cluster_id"] == M["cluster_id"]}
    best = None
    for jid, (pl, segs) in cands.items():
        prM = prongs(g, mvid, segs, a.min_arm); prJ = prongs(g, jid, segs, a.min_arm)
        sM = strength(prM, a.collinear); sJ = strength(prJ, a.collinear)
        tier = None; fx = None; rms = None; dfM = dfJ = None
        if sM == 0 and sJ >= 2: tier = "A"
        elif sM >= 1 and sJ >= 1 and sM + sJ >= 3:
            fx, rms = joint_fit(prM + prJ)
            if fx is not None:
                dfM = float(np.linalg.norm(fx - P(M))); dfJ = float(np.linalg.norm(fx - P(g.V[jid])))
                if dfM - dfJ > a.margin and rms < a.rms: tier = "B"
        rec = dict(jid=jid, path=pl, sM=sM, sJ=sJ, tier=tier, rms=rms, dfM=dfM, dfJ=dfJ, degJ=len(g.inc[jid]))
        key = (1 if tier else 0, sJ, -pl)
        if best is None or key > best[0]: best = (key, rec)
    if best is None: return dict(mvid=mvid, nJ=0, rec=None, M=P(M))
    return dict(mvid=mvid, nJ=len(cands), rec=best[1], M=P(M), J=P(g.V[best[1]["jid"]]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+"); ap.add_argument("--tags", default="vtx100")
    ap.add_argument("--r", type=float, default=4.0); ap.add_argument("--min-arm", type=float, default=3.0)
    ap.add_argument("--collinear", type=float, default=150.0); ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--rms", type=float, default=1.0)
    ap.add_argument("--tsv"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--unlabelled", action="store_true", help="iterate every pr_evt dir of the arms instead of the labels (footprint count; click unknown)")
    a = ap.parse_args()
    tags = {"vtx100": vtx_io.TAGS_VTX100, "harv3": vtx_io.TAGS_HARV3, "mcp2k": vtx_io.TAGS_MCP2K}[a.tags]
    rows = []; seen = set()
    import glob
    if a.unlabelled:
        docs = []
        for arm in a.arms:
            for p in sorted(glob.glob(f"{arm}/pr_evt*/calib-pr-evt*.json")):
                docs.append({"eventNo": int(os.path.basename(p)[len("calib-pr-evt"):-5]), "truth": None})
    else:
        docs = [doc for doc in vtx_io.load_labels(tags=tags) if doc["truth"] is not None]
    for doc in docs:
        ev = doc["eventNo"]
        if ev in seen: continue
        click = np.array(doc["truth"][:3], dtype=float) if doc["truth"] is not None else None
        d = None; arm = None
        for arm in a.arms:
            d = load(arm, ev)
            if d: break
        if not d: continue
        seen.add(ev)
        r = analyse(d, a)
        if r is None: continue
        dM = float(np.linalg.norm(click - r["M"])) if click is not None else float("nan")
        if r["rec"] is None:
            rows.append(dict(evt=ev, arm=arm, nJ=0, dM=dM, dJ=None, sM=None, sJ=None, tier=None, snap=0, verdict="-", path=None, rms=None, jid=None))
            continue
        rec = r["rec"]; dJ = float(np.linalg.norm(click - r["J"])) if click is not None else float("nan")
        snap = 1 if rec["tier"] else 0
        nearerJ = (dM - dJ) > 0.5; nearerM = (dJ - dM) > 0.5
        verdict = ("FIX" if nearerJ else "BREAK" if nearerM else "NEUTRAL") if snap else ("MISS" if nearerJ else "OK")
        if click is None: verdict = "SNAP" if snap else "-"
        rows.append(dict(evt=ev, arm=arm, nJ=r["nJ"], dM=dM, dJ=dJ, sM=rec["sM"], sJ=rec["sJ"], tier=rec["tier"], snap=snap,
                         verdict=verdict, path=rec["path"], rms=rec["rms"], jid=rec["jid"], degJ=rec["degJ"]))
    print(f"# labelled events found in arms: {len(rows)}; with a J candidate within {a.r} cm: {sum(1 for r in rows if r['nJ'])}")
    cnt = collections.Counter((r["tier"], r["verdict"]) for r in rows if r["nJ"])
    for k in sorted(cnt, key=str): print(f"#   tier={k[0]} verdict={k[1]}: {cnt[k]}")
    print("evt      arm                      dM     dJ    sM sJ degJ tier path  rms   verdict")
    for r in rows:
        if not r["nJ"] and not a.all: continue
        if not r["nJ"]:
            print(f"{r['evt']:7d} {r['arm']:24s} {r['dM']:6.2f}     -    -  -   -   -    -     -    -"); continue
        print(f"{r['evt']:7d} {r['arm']:24s} {r['dM']:6.2f} {r['dJ']:6.2f}  {r['sM']:2d} {r['sJ']:2d}  {r['degJ']:2d}  {str(r['tier']):4s} {r['path']:4.1f} "
              f"{('%.2f' % r['rms']) if r['rms'] is not None else '  -  ':5s} {r['verdict']}")
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("evt\tarm\tnJ\tdM\tdJ\tsM\tsJ\tdegJ\ttier\tsnap\tpath\trms\tverdict\n")
            for r in rows:
                f.write("\t".join(str(r.get(k)) for k in ["evt", "arm", "nJ", "dM", "dJ", "sM", "sJ", "degJ", "tier", "snap", "path", "rms", "verdict"]) + "\n")

if __name__ == "__main__":
    main()
