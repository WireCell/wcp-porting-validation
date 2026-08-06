#!/usr/bin/env python3
"""doc pr/39 full-scale verification: for every shower in every calib-pr-evt<ID>.json
under the given roots, check whether data.end coincides with the shower's own
start_vertex position (the exact defect calculate_kinematics{,_long_muon} had
pre-fix: the farthest-vertex search picking m_start_vertex itself).  This reads
the direct PR output (start_vertex_id -> vertices[].fit position), not the Bee
mc.json used for the original 9-event sample -- more direct and sample-agnostic
(works for nueCC48 even though it has no pi0/gamma structure).
"""
import json, glob, math, sys

def dist(a, b):
    return math.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)

EPS = 1e-6  # cm; the bug reproduces the vertex position exactly (same float value)

def check_file(path):
    d = json.load(open(path))
    vtx_by_id = {v['id']: v['fit'] for v in d.get('vertices', [])}
    rows = []
    for s in d.get('showers', []):
        svid = s.get('start_vertex_id')
        vpos = vtx_by_id.get(svid)
        if vpos is None:
            rows.append((s['id'], s['start_connection_type'], None, None, 'no-start-vertex'))
            continue
        d_start = dist(s['start'], vpos)
        d_end = dist(s['end'], vpos)
        reversed_ = d_end < EPS and d_start >= EPS
        rows.append((s['id'], s['start_connection_type'], d_start, d_end, 'reversed' if reversed_ else 'ok'))
    return rows

def main(roots):
    total_ok = total_rev = total_other = 0
    per_event = []
    for root in roots:
        for f in sorted(glob.glob(f"{root}/pr_evt*/calib-pr-evt*.json")):
            evt = f.split('pr_evt')[1].split('/')[0]
            rows = check_file(f)
            ok = sum(1 for r in rows if r[4] == 'ok')
            rev = sum(1 for r in rows if r[4] == 'reversed')
            other = sum(1 for r in rows if r[4] == 'no-start-vertex')
            total_ok += ok; total_rev += rev; total_other += other
            per_event.append((root, evt, len(rows), ok, rev, other))
            if rev:
                for r in rows:
                    if r[4] == 'reversed':
                        print(f"  REVERSED: {root} evt{evt} shower {r[0]} conn_type={r[1]} d_start={r[2]:.3f} d_end={r[3]:.3f}")
    print()
    print(f"{'root':<28}{'evt':>8}{'n_showers':>11}{'ok':>6}{'reversed':>10}{'no_vtx':>8}")
    for root, evt, n, ok, rev, other in per_event:
        print(f"{root:<28}{evt:>8}{n:>11}{ok:>6}{rev:>10}{other:>8}")
    print()
    print(f"TOTAL showers={total_ok+total_rev+total_other} ok={total_ok} reversed={total_rev} no_start_vertex={total_other}")

if __name__ == '__main__':
    main(sys.argv[1:])
