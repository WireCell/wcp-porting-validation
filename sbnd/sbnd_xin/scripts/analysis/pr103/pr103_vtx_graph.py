#!/usr/bin/env python3
"""doc pr/103 -- print the PR graph around the main vertex of a calib dump.

Usage: pr103_vtx_graph.py <arm> <evt> [--r 10] [--vid VID]
For every vertex within --r cm of the main vertex (or of --vid): degree, incident
segments (length, cluster, shower flag, pid, the far vertex and its degree,
direction angle of the first 3 cm toward/away from the main vertex), plus
non-incident segments whose nearest point comes within 3 cm (orphans).
Read-only.  Vertex positions are vertices[].fit.{x,y,z} (the dump schema).
"""
import json, math, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("arm"); ap.add_argument("evt", type=int)
ap.add_argument("--r", type=float, default=10.0)
ap.add_argument("--vid", type=int, default=None)
a = ap.parse_args()
d = json.load(open(f"{a.arm}/pr_evt{a.evt}/calib-pr-evt{a.evt}.json"))
def P(v): f = v["fit"]; return (f["x"], f["y"], f["z"])
def dist(p, q): return math.sqrt(sum((p[i]-q[i])**2 for i in range(3)))
V = {v["id"]: v for v in d["vertices"]}
mv = d["main_vertex"]; mvp = (mv["x"], mv["y"], mv["z"])
if a.vid is not None: mvp = P(V[a.vid])
segs = d["segments"]
inc_of = {}
for s in segs:
    for vid in (s["start_vertex_id"], s["end_vertex_id"]):
        inc_of.setdefault(vid, []).append(s)
print(f"evt {a.evt} main_vertex cl={mv['cluster_id']} pos=({mvp[0]:.2f},{mvp[1]:.2f},{mvp[2]:.2f})  nseg={len(segs)} nvtx={len(V)}")
near = sorted((dist(P(v), mvp), vid) for vid, v in V.items() if dist(P(v), mvp) < a.r)
def seg_dir(s, vid, L=3.0):
    pts = [(p["x"], p["y"], p["z"]) for p in s["points"]]
    if s["end_vertex_id"] == vid and s["start_vertex_id"] != vid: pts = pts[::-1]
    p0 = pts[0]; far = pts[-1]
    for p in pts:
        if dist(p, p0) > L: far = p; break
    return tuple(far[i]-p0[i] for i in range(3))
def ang(u, v):
    nu = math.sqrt(sum(x*x for x in u)); nv = math.sqrt(sum(x*x for x in v))
    if nu == 0 or nv == 0: return float("nan")
    c = max(-1, min(1, sum(u[i]*v[i] for i in range(3))/(nu*nv))); return math.degrees(math.acos(c))
for dd, vid in near:
    v = V[vid]; inc = inc_of.get(vid, [])
    print(f" vtx {vid} d_main={dd:.2f} deg={len(inc)} cl={v['cluster_id']} main={v.get('is_main')} cand={v.get('main_candidate')} fitd={v.get('fit_distance'):.2f}")
    for s in inc:
        far = s["end_vertex_id"] if s["start_vertex_id"] == vid else s["start_vertex_id"]
        fdeg = len(inc_of.get(far, [])); fdm = dist(P(V[far]), mvp) if far in V else -1
        dv = seg_dir(s, vid); to_main = tuple(mvp[i]-P(v)[i] for i in range(3))
        amain = ang(dv, to_main) if dd > 0.05 else float("nan")
        print(f"    seg {s['id']:6d} len={s['length']:6.1f} npts={len(s['points']):4d} sh={int(s['flag_shower'])} pid={s['particle_id']:5d} ps={s['particle_score']:.2f} dirweak={int(s['dir_weak'])} far={far}(deg{fdeg},d_main={fdm:.1f}) ang(dir,->main)={amain:.0f}")
# orphans: non-incident segments near main vertex
print(" non-incident segments with a point within 3 cm of main vertex:")
inc_main = {s["id"] for s in inc_of.get(V and min(near)[1], [])} if near else set()
for s in segs:
    if s["start_vertex_id"] in [vid for _, vid in near] or s["end_vertex_id"] in [vid for _, vid in near]: continue
    pts = [(p["x"], p["y"], p["z"]) for p in s["points"]]
    if not pts: continue
    dm = min(dist(p, mvp) for p in pts)
    if dm < 3.0:
        i = min(range(len(pts)), key=lambda k: dist(pts[k], mvp))
        print(f"    seg {s['id']:6d} len={s['length']:6.1f} cl={s['cluster_id']} sh={int(s['flag_shower'])} mind={dm:.2f} at point {i}/{len(pts)} sv={s['start_vertex_id']} ev={s['end_vertex_id']}")
