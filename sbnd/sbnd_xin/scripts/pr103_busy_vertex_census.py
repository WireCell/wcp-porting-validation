#!/usr/bin/env python3
"""doc pr/103 -- busy-vertex census at the reco main vertex (forked in spirit
from pr86_orphan_census.py; reads calib-pr-evt*.json only, no labels needed).

Usage: pr103_busy_vertex_census.py <arm> [<arm> ...] [--tsv OUT] [--r 4.0] [--tol 1.0] [--top N]

Per event (main vertex M = dump main_vertex, vertex record = nearest vertices[]
entry, is_main preferred):
  n_trk_near / n_sh_near : track-like (flag_shower=0, len>=3 cm) / shower-flagged
                           segments with a fitted point within 2 cm of M
  em_excl  : EM exclusion rule -- the longest segment touching M is shower-flagged
             AND fewer than 2 track-like segments touch M  (owner: "we do not care
             about the EM shower event, vertex found inside the shower")
  deg      : graph degree of M
  stubs    : incident segments shorter than 2.5 cm
  passthru : non-incident segment S whose interior passes within --tol of M, with
             an endpoint vertex J within --r of M that carries another prong
             (the 18255-405707 shape: the prong at J "takes a shortcut")
  shortcut : prong (len>=3) ending on a non-main vertex J within --r of M, J not
             reached through S above, excluding the stub M-J itself
  orphans  : non-incident track-like segments (len>=3) with a point within 2 cm of M
             that are not the passthru S
  kink     : max perpendicular deviation (cm) of the first 3 cm of an incident
             track prong (len>=10) from the straight line fitted to its 5-20 cm
             stretch (the 18255-283713 shape); kink_seg = which prong
Rank key: passthru, shortcut, orphans, stubs, kink.
"""
import json, math, sys, os, glob, argparse
import numpy as np

def load(p):
    try: return json.load(open(p))
    except Exception: return None

def P(v): f = v["fit"]; return np.array([f["x"], f["y"], f["z"]])

def seg_pts(s): return np.array([[p["x"], p["y"], p["z"]] for p in s["points"]]) if s["points"] else np.zeros((0, 3))

def analyse(d, R, TOL):
    mv = d["main_vertex"]; mvp = np.array([mv["x"], mv["y"], mv["z"]])
    V = {v["id"]: v for v in d["vertices"]}
    if not V: return None
    # main vertex record
    mains = [v for v in d["vertices"] if v.get("is_main")]
    mvid = mains[0]["id"] if mains else min(V, key=lambda k: np.linalg.norm(P(V[k]) - mvp))
    segs = d["segments"]
    inc = {}
    for s in segs:
        for vid in (s["start_vertex_id"], s["end_vertex_id"]):
            inc.setdefault(vid, []).append(s)
    incident = inc.get(mvid, [])
    inc_ids = {s["id"] for s in incident}
    # near segments
    near = []
    for s in segs:
        pts = seg_pts(s)
        if len(pts) == 0: continue
        dmin = np.linalg.norm(pts - mvp, axis=1).min()
        if dmin < 2.0: near.append((s, dmin))
    n_trk_near = sum(1 for s, _ in near if not s["flag_shower"] and s["length"] >= 3.0)
    n_sh_near = sum(1 for s, _ in near if s["flag_shower"])
    longest = max(near, key=lambda t: t[0]["length"])[0] if near else None
    em_excl = bool(longest is not None and longest["flag_shower"] and n_trk_near < 2)
    stubs = [s for s in incident if s["length"] < 2.5]
    # passthru
    passthru = []
    pt_segids = set()
    for s in segs:
        if s["id"] in inc_ids: continue
        pts = seg_pts(s)
        if len(pts) < 3 or s["length"] < 3.0: continue
        dd = np.linalg.norm(pts - mvp, axis=1)
        k = int(dd.argmin())
        if dd[k] >= TOL or k == 0 or k == len(pts) - 1: continue
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        for jvid, jend in ((s["start_vertex_id"], 0), (s["end_vertex_id"], len(pts) - 1)):
            if jvid == mvid or jvid not in V: continue
            dj = np.linalg.norm(P(V[jvid]) - mvp)
            if dj >= R: continue
            arc_j = abs(arc[k] - arc[jend]); rem = arc[-1] - arc_j
            if rem < 3.0: continue
            others = [o for o in inc.get(jvid, []) if o["id"] != s["id"]
                      and not (mvid in (o["start_vertex_id"], o["end_vertex_id"]))]
            if not others: continue
            passthru.append(dict(S=s["id"], S_len=round(s["length"], 1), J=jvid, dJ=round(dj, 2),
                                 miss=round(float(dd[k]), 2), arc=round(float(arc_j), 2),
                                 others=[(o["id"], round(o["length"], 1), int(o["flag_shower"])) for o in others]))
            pt_segids.add(s["id"])
    # shortcut: prongs ending on non-main J within R
    shortcut = []
    for jvid, v in V.items():
        if jvid == mvid: continue
        dj = np.linalg.norm(P(v) - mvp)
        if dj >= R or dj < 0.05: continue
        for s in inc.get(jvid, []):
            if s["id"] in pt_segids or s["length"] < 3.0: continue
            if mvid in (s["start_vertex_id"], s["end_vertex_id"]): continue
            link = [o for o in inc.get(jvid, []) if mvid in (o["start_vertex_id"], o["end_vertex_id"])]
            shortcut.append(dict(seg=s["id"], len=round(s["length"], 1), sh=int(s["flag_shower"]), J=jvid, dJ=round(dj, 2),
                                 Jdeg=len(inc.get(jvid, [])), link=[(o["id"], round(o["length"], 2)) for o in link]))
    orphans = [dict(seg=s["id"], len=round(s["length"], 1), dmin=round(dmin, 2)) for s, dmin in near
               if s["id"] not in inc_ids and s["id"] not in pt_segids and not s["flag_shower"] and s["length"] >= 3.0]
    # kink
    kink = 0.0; kink_seg = -1
    for s in incident:
        if s["flag_shower"] or s["length"] < 10: continue
        pts = seg_pts(s)
        if len(pts) < 8: continue
        if np.linalg.norm(pts[-1] - mvp) < np.linalg.norm(pts[0] - mvp): pts = pts[::-1]
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        m = (arc > 5) & (arc < 20)
        if m.sum() < 4: continue
        c = pts[m].mean(0); u = np.linalg.svd(pts[m] - c)[2][0]
        dv = pts[arc <= 3.0] - c
        perp = np.linalg.norm(dv - np.outer(dv @ u, u), axis=1)
        if len(perp) and perp.max() > kink: kink = float(perp.max()); kink_seg = s["id"]
    return dict(mvid=mvid, deg=len(incident), n_trk_near=n_trk_near, n_sh_near=n_sh_near, em_excl=em_excl,
                stubs=[(s["id"], round(s["length"], 2)) for s in stubs], passthru=passthru, shortcut=shortcut,
                orphans=orphans, kink=round(kink, 2), kink_seg=kink_seg,
                numu=d.get("tagger", {}).get("numu_score"), nue=d.get("tagger", {}).get("nue_score"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+"); ap.add_argument("--tsv"); ap.add_argument("--r", type=float, default=4.0)
    ap.add_argument("--tol", type=float, default=1.0); ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--events", nargs="*", type=int)
    a = ap.parse_args()
    rows = []
    for arm in a.arms:
        for p in sorted(glob.glob(f"{arm}/pr_evt*/calib-pr-evt*.json")):
            ev = int(os.path.basename(p)[len("calib-pr-evt"):-5])
            if a.events and ev not in a.events: continue
            d = load(p)
            if not d or not d.get("main_vertex") or not d.get("vertices"): continue
            r = analyse(d, a.r, a.tol)
            if r is None: continue
            r["evt"] = ev; r["arm"] = arm; rows.append(r)
    n = len(rows); ne = sum(r["em_excl"] for r in rows)
    keep = [r for r in rows if not r["em_excl"]]
    def cnt(f): return sum(1 for r in keep if f(r))
    print(f"# events with dump: {n}   EM-excluded: {ne}   evaluated: {len(keep)}")
    print(f"# passthru events: {cnt(lambda r: r['passthru'])}   shortcut events: {cnt(lambda r: r['shortcut'])}"
          f"   orphan events: {cnt(lambda r: r['orphans'])}   stub events: {cnt(lambda r: r['stubs'])}"
          f"   kink>1cm events: {cnt(lambda r: r['kink'] > 1.0)}   busy(n_trk_near>=3): {cnt(lambda r: r['n_trk_near'] >= 3)}")
    def key(r): return (len(r["passthru"]), len(r["shortcut"]), len(r["orphans"]), len(r["stubs"]), r["kink"])
    keep.sort(key=key, reverse=True)
    for r in keep[:a.top]:
        print(f"{r['evt']:7d} deg={r['deg']} trk={r['n_trk_near']} sh={r['n_sh_near']} stubs={r['stubs']} "
              f"passthru={r['passthru']} shortcut={r['shortcut']} orphans={r['orphans']} kink={r['kink']}@{r['kink_seg']}")
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("evt\tarm\tem_excl\tdeg\tn_trk_near\tn_sh_near\tn_stubs\tn_passthru\tn_shortcut\tn_orphans\tkink\tkink_seg\tnumu\tnue\tdetail\n")
            for r in rows:
                f.write(f"{r['evt']}\t{r['arm']}\t{int(r['em_excl'])}\t{r['deg']}\t{r['n_trk_near']}\t{r['n_sh_near']}\t{len(r['stubs'])}\t"
                        f"{len(r['passthru'])}\t{len(r['shortcut'])}\t{len(r['orphans'])}\t{r['kink']}\t{r['kink_seg']}\t{r['numu']}\t{r['nue']}\t"
                        f"{json.dumps(dict(stubs=r['stubs'], passthru=r['passthru'], shortcut=r['shortcut'], orphans=r['orphans']))}\n")
if __name__ == "__main__": main()
