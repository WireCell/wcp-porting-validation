#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 sec 4 -- the > 10 cm vertex movers between two arms on the vtx105-labelled events.

From the pr150_metrics.py per-event TSVs (vx vy vz has_calib): for every labelled event present in both arms,
dA / dB = distance of each arm's main vertex to the rank-1 vtx105 click (inf when the arm has no vertex),
moved = |vA - vB|.  A mover is moved > --min-move cm (or one side missing); direction 'away' if dB > dA else
'toward'.  Writes a TSV (sample event direction dA dB moved) for pr150_vtx_scan.py prepare.
Usage: pr150_vtx_movers.py --a pr150s0 --b pr150csp3bw --samples S... --out movers.tsv [--min-move 10]
"""
import argparse, csv, math, os, sys
SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
sys.path.insert(0, os.path.join(SX, 'vtx_rules'))
import vtx_io  # noqa: E402
M = os.path.join(SX, 'docs/pr/150_figs/metrics')


def load(arm, samples):
    R = {}
    for s in samples:
        for r in csv.DictReader(open(f'{M}/{arm}-{s}.tsv'), delimiter='\t'):
            v = None
            if r.get('has_calib') == '1' and r['vx'] not in ('', 'nan'):
                v = (float(r['vx']), float(r['vy']), float(r['vz']))
            R[(s, int(r['event']))] = v
    return R


def dist(a, b):
    return math.inf if a is None or b is None else math.dist(a, b)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--out', required=True); ap.add_argument('--min-move', type=float, default=10.0)
    a = ap.parse_args()
    A, B = load(a.a, a.samples), load(a.b, a.samples)
    labs = {L['event']: L for L in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105)}
    n_lab = n_mov = 0; away = toward = 0
    with open(a.out, 'w') as fh:
        fh.write('sample\tevent\tdirection\tdA\tdB\tmoved\n')
        for k in sorted(set(A) & set(B)):
            L = labs.get(f'evt{k[1]}')
            if L is None or L['truth'] is None:
                continue
            n_lab += 1
            dA, dB, mv = dist(L['truth'], A[k]), dist(L['truth'], B[k]), dist(A[k], B[k])
            if mv > a.min_move or (A[k] is None) != (B[k] is None):
                n_mov += 1; d = 'away' if dB > dA else 'toward'; away += d == 'away'; toward += d == 'toward'
                fh.write(f'{k[0]}\t{k[1]}\t{d}\t{dA:.2f}\t{dB:.2f}\t{mv:.2f}\n')
    print(f'{a.a} vs {a.b}: labelled in both {n_lab}; movers > {a.min_move} cm {n_mov} (away {away}, toward {toward}) -> {a.out}')


if __name__ == '__main__':
    main()
