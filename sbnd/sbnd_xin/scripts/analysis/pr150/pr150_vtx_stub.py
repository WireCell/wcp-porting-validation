#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 sec 4 -- stub-branch attribution of the vertex movers (pr/149 sec 7.2 mechanism).

For every vtx105-labelled event present in both arms, read each arm's calib dump and describe the MAIN vertex:
its degree, the number of attached segments that end at a degree-1 vertex and are shorter than 3 / 5 / 10 cm
(stubs), the shortest attached segment, and whether the main vertex is a degree-1 vertex itself (the vertex sits
at a track end).  Then tabulate by mover class (from pr150_vtx_movers.py: away / toward / non-mover):
  share of B's main vertices that carry a stub < 5 cm, vs A's, per class.
If the "away" movers' new vertices carry stubs far more often than the non-movers' vertices, the stub-branch
mechanism of pr/149 sec 7.2 is live and a degree-1 stub floor at the main vertex (mvga_stub / es3sg_stub_max /
steiner_terminal_min_separation) is the lever to try.
Usage: pr150_vtx_stub.py --a pr150s0 --b pr150csp3bw --samples S... --movers <tsv> [--tsv out]
"""
import argparse, csv, glob, json, math, os
SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'


def seglen(seg):
    if seg.get('length') is not None:
        return float(seg['length'])
    P = seg.get('points') or []
    return sum(math.dist((P[i]['x'], P[i]['y'], P[i]['z']), (P[i + 1]['x'], P[i + 1]['y'], P[i + 1]['z'])) for i in range(len(P) - 1))


def describe(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path)); mv = d.get('main_vertex') or {}
    if mv.get('cluster_id') is None:
        return None
    verts = {v['id']: v for v in d.get('vertices', [])}
    # the main vertex id: the vertex nearest main_vertex position in the main cluster
    best, bd = None, 1e9
    for v in verts.values():                       # vertices: {id, cluster_id, degree, is_main, fit:{x,y,z,...}}
        if v.get('cluster_id') != mv['cluster_id'] or not (v.get('fit') or {}).get('x') is not None:
            continue
        f = v['fit']
        dd = 0.0 if v.get('is_main') else math.dist((f['x'], f['y'], f['z']), (mv['x'], mv['y'], mv['z']))
        if dd < bd:
            best, bd = v, dd
    if best is None:
        return None
    att = [s for s in d.get('segments', []) if s.get('start_vertex_id') == best['id'] or s.get('end_vertex_id') == best['id']]
    out = dict(deg=best.get('degree', len(att)), natt=len(att), snap=bd)
    lens = []
    for s in att:
        far = s['end_vertex_id'] if s.get('start_vertex_id') == best['id'] else s['start_vertex_id']
        fdeg = verts.get(far, {}).get('degree', 0)
        lens.append((seglen(s), fdeg, bool(s.get('flag_shower'))))
    for L in (3, 5, 10):
        out[f'stub{L}'] = sum(1 for l, fd, sh in lens if l < L and fd == 1)
    out['shortest'] = min([l for l, _, _ in lens], default=float('nan'))
    out['shower_att'] = sum(1 for _, _, sh in lens if sh)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--movers', required=True); ap.add_argument('--tsv')
    a = ap.parse_args()
    mov = {(r['sample'], int(r['event'])): r['direction'] for r in csv.DictReader(open(a.movers), delimiter='\t')}
    import sys; sys.path.insert(0, f'{SX}/vtx_rules'); import vtx_io
    labs = {L['event'] for L in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105) if L['truth'] is not None}
    rows = []
    for s in a.samples:
        for d in sorted(glob.glob(f'{SX}/work-{s}-{a.a}/pr_evt*')):
            ev = int(os.path.basename(d)[6:])
            if f'evt{ev}' not in labs:
                continue
            A = describe(f'{d}/calib-pr-evt{ev}.json'); B = describe(f'{SX}/work-{s}-{a.b}/pr_evt{ev}/calib-pr-evt{ev}.json')
            if A is None or B is None:
                continue
            rows.append((s, ev, mov.get((s, ev), 'non-mover'), A, B))
    print(f'# pr150 stub attribution A={a.a} B={a.b}: labelled events with a main vertex in both {len(rows)}')
    print('| class | n | A: main vertex with a stub < 3 / 5 / 10 cm | B: the same | A deg-1 main | B deg-1 main | A shortest att. median (cm) | B |')
    print('|---|---|---|---|---|---|---|---|')
    import numpy as np
    for cls in ('away', 'toward', 'non-mover'):
        R = [r for r in rows if r[2] == cls]
        if not R:
            continue
        f = lambda X, k: sum(1 for r in R if r[X][k] > 0)
        g = lambda X: sum(1 for r in R if r[X]['deg'] == 1)
        med = lambda X: np.nanmedian([r[X]['shortest'] for r in R])
        print(f"| {cls} | {len(R)} | {f(3, 'stub3')} / {f(3, 'stub5')} / {f(3, 'stub10')} | {f(4, 'stub3')} / {f(4, 'stub5')} / {f(4, 'stub10')} | {g(3)} | {g(4)} | {med(3):.1f} | {med(4):.1f} |")
    # transitions on the away movers: gained a stub?
    for cls in ('away', 'toward'):
        R = [r for r in rows if r[2] == cls]
        gained = sum(1 for r in R if r[4]['stub5'] > 0 and r[3]['stub5'] == 0); lost = sum(1 for r in R if r[3]['stub5'] > 0 and r[4]['stub5'] == 0)
        print(f'{cls} movers {len(R)}: B main vertex gained a < 5 cm stub (A had none) {gained}; lost one {lost}; both have {sum(1 for r in R if r[4]["stub5"] > 0 and r[3]["stub5"] > 0)}')
    if a.tsv:
        with open(a.tsv, 'w') as fh:
            fh.write('sample\tevent\tclass\tA_deg\tA_stub3\tA_stub5\tA_stub10\tA_shortest\tB_deg\tB_stub3\tB_stub5\tB_stub10\tB_shortest\n')
            for s, ev, cls, A, B in rows:
                fh.write(f"{s}\t{ev}\t{cls}\t{A['deg']}\t{A['stub3']}\t{A['stub5']}\t{A['stub10']}\t{A['shortest']:.1f}\t{B['deg']}\t{B['stub3']}\t{B['stub5']}\t{B['stub10']}\t{B['shortest']:.1f}\n")


if __name__ == '__main__':
    main()
