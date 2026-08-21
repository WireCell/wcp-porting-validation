#!/usr/bin/env python3
"""doc pr/103 -- topology at the owner's CLICK, arm A vs arm B (vtx100 labels).

Usage: pr103_click_topology_ab.py <armA> <armB> [--tags vtx100] [--r 4.0] [--tsv OUT]

Per labelled event present in both arms:
  dclick   : distance click -> nearest PR-graph vertex (any cluster)      [pr/85 "scorable" = <= 1 cm]
  dmain    : distance click -> main_vertex
  cdeg     : degree of the nearest vertex to the click
  ctrk     : track-like (non-shower, >= 3 cm) segments incident on that vertex
  sc       : "shortcut" prongs at the click = track prongs (>= 3 cm) ending on a
             NON-nearest vertex within --r of the click
Classes (A -> B): LOST-CLICK (dclick <=1 in A, >1 in B) is the adverse class;
GAINED-CLICK the converse; ctrk up/down; sc down/up.
"""
import json, math, sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vtx_rules"))
import vtx_io

def P(v): f = v["fit"]; return np.array([f["x"], f["y"], f["z"]])
def load(arm, ev):
    p = f"{arm}/pr_evt{ev}/calib-pr-evt{ev}.json"
    try: return json.load(open(p))
    except Exception: return None
def score(d, click, R):
    V = {v["id"]: v for v in d.get("vertices") or []}
    if not V or not d.get("main_vertex"): return None
    segs = d["segments"]
    inc = {}
    for s in segs:
        for vid in (s["start_vertex_id"], s["end_vertex_id"]): inc.setdefault(vid, []).append(s)
    dists = {vid: float(np.linalg.norm(P(v) - click)) for vid, v in V.items()}
    nv = min(dists, key=dists.get)
    mv = d["main_vertex"]; dmain = float(np.linalg.norm(np.array([mv["x"], mv["y"], mv["z"]]) - click))
    cinc = inc.get(nv, [])
    ctrk = sum(1 for s in cinc if not s["flag_shower"] and s["length"] >= 3.0)
    sc = 0
    for vid, dd in dists.items():
        if vid == nv or dd >= R: continue
        for s in inc.get(vid, []):
            if s["flag_shower"] or s["length"] < 3.0: continue
            if nv in (s["start_vertex_id"], s["end_vertex_id"]): continue
            sc += 1
    return dict(dclick=dists[nv], dmain=dmain, cdeg=len(cinc), ctrk=ctrk, sc=sc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armA"); ap.add_argument("armB"); ap.add_argument("--tags", default="vtx100")
    ap.add_argument("--r", type=float, default=4.0); ap.add_argument("--tsv")
    a = ap.parse_args()
    tags = {"vtx100": vtx_io.TAGS_VTX100, "harv3": vtx_io.TAGS_HARV3, "mcp2k": vtx_io.TAGS_MCP2K}[a.tags]
    rows = []
    for doc in vtx_io.load_labels(tags=tags):
        if doc["truth"] is None: continue
        ev = doc["eventNo"]; click = np.array(doc["truth"][:3], dtype=float)
        dA = load(a.armA, ev); dB = load(a.armB, ev)
        if not dA or not dB: continue
        sA = score(dA, click, a.r); sB = score(dB, click, a.r)
        if not sA or not sB: continue
        cls = []
        if sA["dclick"] <= 1.0 and sB["dclick"] > 1.0: cls.append("LOST-CLICK")
        if sA["dclick"] > 1.0 and sB["dclick"] <= 1.0: cls.append("GAINED-CLICK")
        if sB["ctrk"] > sA["ctrk"]: cls.append("ctrk+")
        if sB["ctrk"] < sA["ctrk"]: cls.append("ctrk-")
        if sB["sc"] < sA["sc"]: cls.append("sc-")
        if sB["sc"] > sA["sc"]: cls.append("sc+")
        if abs(sB["dmain"] - sA["dmain"]) > 0.1: cls.append("main-moved")
        rows.append((ev, sA, sB, cls))
    from collections import Counter
    c = Counter(x for _, _, _, cls in rows for x in cls)
    print(f"# events scored in both arms: {len(rows)}")
    print("# " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    for ev, sA, sB, cls in rows:
        if cls:
            print(f"{ev:7d} dclick {sA['dclick']:.2f}->{sB['dclick']:.2f} dmain {sA['dmain']:.2f}->{sB['dmain']:.2f} "
                  f"cdeg {sA['cdeg']}->{sB['cdeg']} ctrk {sA['ctrk']}->{sB['ctrk']} sc {sA['sc']}->{sB['sc']}  {' '.join(cls)}")
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("evt\tdclickA\tdclickB\tdmainA\tdmainB\tcdegA\tcdegB\tctrkA\tctrkB\tscA\tscB\tclasses\n")
            for ev, sA, sB, cls in rows:
                f.write(f"{ev}\t{sA['dclick']:.2f}\t{sB['dclick']:.2f}\t{sA['dmain']:.2f}\t{sB['dmain']:.2f}\t{sA['cdeg']}\t{sB['cdeg']}\t{sA['ctrk']}\t{sB['ctrk']}\t{sA['sc']}\t{sB['sc']}\t{' '.join(cls)}\n")
if __name__ == "__main__": main()
