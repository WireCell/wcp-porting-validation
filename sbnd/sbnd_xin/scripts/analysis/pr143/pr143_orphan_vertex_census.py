#!/usr/bin/env python3
"""Census of CLUSTERLESS PR vertices in SBND PrDisplayDump calib dumps.

PrDisplayDump writes  vertices[].cluster_id = vtx->cluster() ? id : 0  and
vertices[].id = cluster_id*1000 + graph_index  (PrDisplayDump.cxx:429-433).
A vertex whose cluster() is null is therefore reported with cluster_id 0.
The only vertex factory in the PR chain that can leave cluster() null is
PR::break_segment (PRSegmentFunctions.cxx:1173) at the two call sites in
NeutrinoShowerClustering.cxx (1919 nv_bridge_track, 2246
shower_clustering_with_nv_from_vertices) that do not stamp it afterwards.

Usage: census.py <dump.json> ...   (prints one TSV row per event)
"""
import json, sys, os

print('\t'.join(['event','n_vtx','n_vtx_cl0','n_seg','n_seg_cl0','mainv_cl','seg_cl_min',
                 'cl0_deg2','cl0_deg3p','cl0_touch_main','cl0_touch_ids','cl0_ids']))
for path in sys.argv[1:]:
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        print('# WARN %s %s' % (path, e), file=sys.stderr); continue
    ev = (d.get('meta') or {}).get('eventNo', os.path.basename(path))
    # A dump with nu_per_bundle candidates repeats the seven per-candidate keys
    # under "candidates" (PrDisplayDump.cxx:207-239), and candidates[0] IS the
    # top-level block.  Reading only the top level therefore misses every
    # vertex of candidates[1..] -- which is how the first census reported 42
    # events while the artefact comparison found 44 movers.  Walk the
    # candidates array when it exists, the top-level block when it does not.
    cands = d.get('candidates')
    if not (isinstance(cands, list) and cands):
        cands = [d]
    V = [v for c in cands for v in (c.get('vertices') or [])]
    S = [sg for c in cands for sg in (c.get('segments') or [])]
    mainv_cl = (d.get('main_vertex') or {}).get('cluster_id', -1)
    seg_cl = [s.get('cluster_id', -1) for s in S]
    cl0 = [v for v in V if v.get('cluster_id', -1) == 0]
    ids0 = set(v['id'] for v in cl0)
    # incident segments of each clusterless vertex
    touch_main = 0; touch_ids = set()
    for v in cl0:
        inc = [s for s in S if s.get('start_vertex_id') == v['id'] or s.get('end_vertex_id') == v['id']]
        if any(s.get('cluster_id') == mainv_cl for s in inc):
            touch_main += 1
        for s in inc:
            touch_ids.add(s.get('cluster_id', -1))
    print('\t'.join(str(x) for x in [
        ev, len(V), len(cl0), len(S), sum(1 for c in seg_cl if c == 0), mainv_cl,
        min(seg_cl) if seg_cl else -1,
        sum(1 for v in cl0 if v.get('degree', -1) == 2),
        sum(1 for v in cl0 if v.get('degree', -1) >= 3),
        touch_main,
        ','.join(str(i) for i in sorted(touch_ids)) or '-',
        ','.join(str(i) for i in sorted(ids0)) or '-']))
