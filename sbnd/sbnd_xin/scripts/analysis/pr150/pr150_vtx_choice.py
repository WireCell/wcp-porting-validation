#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 sec 4 -- for each > 10 cm vertex mover: did the candidate SET change, or the CHOICE?

For a mover (A = s0, B = the cell), with the vtx105 click:
  A_in_B   A's main vertex position still exists as a PR-graph vertex candidate in B (within --tol cm, any cluster)
  B_in_A   B's main vertex position existed as a candidate in A
  click_in_A / click_in_B   a candidate within --tol of the click exists in that arm
Classes: CHOICE (A's vertex is still a candidate in B, B chose another) vs STRUCTURE (A's vertex is no longer a
candidate in B: the trajectory/graph changed under it) for the 'away' movers, and the mirror for 'toward'.
Also: whether the main CLUSTER (bundle) changed (calib main_vertex.cluster_id differs, checked by position overlap
of the bundles' vertices) and the DL route if the dump carries it.
Usage: pr150_vtx_choice.py --a pr150s0 --b pr150csp3bw --movers <tsv> [--tol 1.0]
"""
import argparse, csv, json, math, os, sys
SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
sys.path.insert(0, f'{SX}/vtx_rules'); import vtx_io  # noqa: E402


def load(arm, s, ev):
    p = f'{SX}/work-{s}-{arm}/pr_evt{ev}/calib-pr-evt{ev}.json'
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); mv = d.get('main_vertex') or {}
    cands = [(v['fit']['x'], v['fit']['y'], v['fit']['z'], v['cluster_id']) for v in d.get('vertices', []) if (v.get('fit') or {}).get('x') is not None]
    return dict(mv=(mv['x'], mv['y'], mv['z']) if mv.get('x') is not None else None, cid=mv.get('cluster_id'), cands=cands,
                route=(d.get('kine') or {}).get('vertex_route') or d.get('vertex_route'))


def near(p, cands, tol):
    return p is not None and any(math.dist(p, c[:3]) <= tol for c in cands)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--movers', required=True); ap.add_argument('--tol', type=float, default=1.0)
    a = ap.parse_args()
    labs = {L['event']: L for L in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105)}
    cnt = {}
    print('sample\tevent\tdirection\tdA\tdB\tA_in_B\tB_in_A\tclick_in_A\tclick_in_B\tsame_cluster\tclass')
    for r in csv.DictReader(open(a.movers), delimiter='\t'):
        s, ev, d = r['sample'], int(r['event']), r['direction']
        A, B = load(a.a, s, ev), load(a.b, s, ev)
        if A is None or B is None or A['mv'] is None or B['mv'] is None:
            c = 'LOST-CANDIDATE' if (B is None or B['mv'] is None) else 'GAINED-CANDIDATE'
            cnt[(d, c)] = cnt.get((d, c), 0) + 1
            print(f'{s}\t{ev}\t{d}\t{r["dA"]}\t{r["dB"]}\t\t\t\t\t\t{c}'); continue
        click = labs[f'evt{ev}']['truth']
        a_in_b, b_in_a = near(A['mv'], B['cands'], a.tol), near(B['mv'], A['cands'], a.tol)
        ca, cb = near(click, A['cands'], a.tol), near(click, B['cands'], a.tol)
        same = near(A['mv'], [c for c in B['cands'] if c[3] == B['cid']], 5.0) or near(B['mv'], [c for c in A['cands'] if c[3] == A['cid']], 5.0)
        if d == 'away':
            c = 'CHOICE' if a_in_b else ('STRUCTURE-click-gone' if not cb else 'STRUCTURE')
        else:
            c = 'CHOICE' if b_in_a else ('STRUCTURE-click-new' if not ca and cb else 'STRUCTURE')
        cnt[(d, c)] = cnt.get((d, c), 0) + 1
        print(f'{s}\t{ev}\t{d}\t{r["dA"]}\t{r["dB"]}\t{int(a_in_b)}\t{int(b_in_a)}\t{int(ca)}\t{int(cb)}\t{int(same)}\t{c}')
    print('# ' + '; '.join(f'{d} {c} {n}' for (d, c), n in sorted(cnt.items())))


if __name__ == '__main__':
    main()
