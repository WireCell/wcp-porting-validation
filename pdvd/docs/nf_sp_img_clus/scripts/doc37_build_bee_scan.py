#!/usr/bin/env python3
"""doc pdvd/37: a Bee scan set showing what R1 thinning does, on one event per
detector.  Built from existing calib-pr dumps -- no reconstruction is run.

Layers per event:
  steiner        the whole Steiner point cloud
  term-prod      the terminals as production makes them
  term-R<r>      the terminals after greedy thinning at radius r
  term-R<r>exm   the same, with terminals within X cm of a PR vertex exempted
  skeleton       the FITTED PR trajectory points (segments[].points)
  vertices       PR graph vertices, q=15000 for the main vertex

Bee renders any data/<n>/<n>-<name>.json whose name does not contain '-track',
does not start with 'channel' and does not contain 'auto-sel'.
"""
import json, os, sys, zipfile
import numpy as np
from scipy.spatial import cKDTree

R = 1.0
EXEMPT = 2.0

def greedy(P, order, R, prot=None):
    keep = np.zeros(len(P), bool) if prot is None else prot.copy()
    t = cKDTree(P); dead = np.zeros(len(P), bool); out = []
    for i in order:
        if keep[i] or dead[i]: continue
        out.append(i)
        for j in t.query_ball_point(P[i], R):
            if j != i and not keep[j]: dead[j] = True
    out += np.nonzero(keep)[0].tolist()
    return np.asarray(sorted(set(out)), int)

def layer(name, P, cid, meta, q=None):
    n = len(P)
    return {
        "runNo": str(meta["run"]), "subRunNo": str(meta["subrun"]),
        "eventNo": str(meta["event"]), "geom": meta["geom"], "type": name,
        "x": [round(float(v), 3) for v in P[:, 0]],
        "y": [round(float(v), 3) for v in P[:, 1]],
        "z": [round(float(v), 3) for v in P[:, 2]],
        "q": (q if q is not None else [1.0] * n),
        "cluster_id": [int(c) for c in cid],
        "real_cluster_id": [int(c) for c in cid],
    }

def build(dump, meta):
    d = json.load(open(dump))
    m = d.get("meta", {})
    meta = dict(meta)
    for k, mk in (("run", "runNo"), ("subrun", "subRunNo"), ("event", "eventNo")):
        if mk in m:
            meta[k] = m[mk]
    out = {}
    Ps, Cs, Ts = [], [], []
    for e in d.get("steiner", []):
        if "flag_terminal" not in e: continue
        P = np.stack([e["x"], e["y"], e["z"]], 1).astype(float)
        Ps.append(P); Cs.append(np.full(len(P), e["cluster_id"]))
        Ts.append(np.asarray(e["flag_terminal"], bool))
    if not Ps: return None
    P = np.vstack(Ps); C = np.concatenate(Cs); T = np.concatenate(Ts)
    out["steiner"] = layer("steiner", P, C, meta)
    out["term-prod"] = layer("term-prod", P[T], C[T], meta)

    vp = np.array([[v["fit"]["x"], v["fit"]["y"], v["fit"]["z"]]
                   for v in d.get("vertices", []) if "fit" in v], float)
    # thin per cluster, as the real pass would
    keep_g, keep_e = [], []
    for cid in np.unique(C[T]):
        m = (C == cid) & T
        Q = P[m]
        if len(Q) < 2:
            keep_g.append(Q); keep_e.append(Q); continue
        ordr = np.argsort(np.linalg.norm(Q - Q.mean(0), axis=1))
        keep_g.append(Q[greedy(Q, ordr, R)])
        prot = (np.linalg.norm(Q[:, None, :] - vp[None, :, :], axis=-1).min(1) < EXEMPT
                if len(vp) else np.zeros(len(Q), bool))
        keep_e.append(Q[greedy(Q, ordr, R, prot)])
    G = np.vstack(keep_g); E = np.vstack(keep_e)
    out[f"term-R{R}"] = layer(f"term-R{R}", G, np.zeros(len(G)), meta)
    out[f"term-R{R}exm"] = layer(f"term-R{R}exm", E, np.zeros(len(E)), meta)

    S = [], []
    sp, sc = [], []
    for s in d.get("segments", []):
        for q in s["points"]:
            sp.append([q["x"], q["y"], q["z"]]); sc.append(s["cluster_id"])
    if sp:
        out["skeleton"] = layer("skeleton", np.array(sp, float), sc, meta)
    if len(vp):
        qq = [15000.0 if v.get("is_main") else 0.0
              for v in d.get("vertices", []) if "fit" in v]
        out["vertices"] = layer("vertices", vp, np.zeros(len(vp)), meta, q=qq)
    print(f"  {os.path.basename(dump)}: steiner {len(P)}, terminals {T.sum()} -> "
          f"R={R} {len(G)} ({len(G)/max(1,T.sum()):.2f}) -> exempt {len(E)} "
          f"({len(E)/max(1,T.sum()):.2f}); {len(sp)} skeleton pts, {len(vp)} vertices")
    return out

specs = [
    ("PDVD",  sys.argv[1], dict(run=39252, subrun=0, event=298595, geom="protodunevd")),
    ("SBND",  sys.argv[2], dict(run=0, subrun=0, event=0, geom="sbnd")),
    ("uBooNE", sys.argv[3], dict(run=5384, subrun=130, event=6501, geom="uboone")),
]
zpath = "/home/xqian/tmp/doc37/d37_thinning_scan.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
    for i, (lab, dump, meta) in enumerate(specs):
        print(f"[{i}] {lab}")
        o = build(dump, meta)
        if o is None: continue
        for name, obj in o.items():
            zf.writestr(f"data/{i}/{i}-{name}-global.json", json.dumps(obj))
print("wrote", zpath, os.path.getsize(zpath), "bytes")
