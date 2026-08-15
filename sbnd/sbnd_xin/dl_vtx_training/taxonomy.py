#!/usr/bin/env python3
'''
doc pr/77 round 2 (S8a) -- failure taxonomy of the production DL vertex on
every hand-scanned event.  Decides where investment goes BEFORE any training:

  correct            production main vertex within --tol of the truth pick
  candidate-missing  no scoreboard candidate (rows[]) within --tol of truth:
                     no net training can fix this -- pr/51 graph territory
  selection-wrong    a recorded top-5 net voxel IS within --tol of truth but
                     the chosen vertex is elsewhere -- rerank/snap error
  net-wrong          truth has a nearby candidate but none of the net's top-5
                     voxels is near it -- the net's heat is the problem

Classes are mutually exclusive and sum to the label count (asserted).
Truth = the label's rank-1 pick; the recorded scoreboard voxels[] are the
actual pre-refit net output (no rebuild approximation).

Usage:
  python3 taxonomy.py --tsv runs/taxonomy-20260814.tsv
'''
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio

ALL_TAGS = ['vtxscan-prod0813', 'vtxscan-prod0813-ncpi0', 'vtxscan-prod0813-mcp1k']
CLASSES = ['correct', 'candidate-missing', 'net-wrong', 'selection-wrong']


def numu50_set(here, manifest=None):
    """The frozen numu50 snapshot defines the 50-numu reporting subset.
    round 3: pass a snapshot manifest with a `sample` column to use its
    numu<K>-flagged rows instead (e.g. data/full473 -> numu100)."""
    if manifest:
        with open(manifest) as fh:
            import csv
            return {int(r['evt']) for r in csv.DictReader(fh, delimiter='\t')
                    if r.get('sample', '').startswith('numu')}
    path = os.path.join(here, 'data', 'numu50', 'manifest.tsv')
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        next(fh)
        return {int(l.split('\t')[0]) for l in fh if l.strip()}


def classify(label, sb, tol):
    truth = label['truth_xyz']
    fin = label.get('main_vertex') or {}
    prod = np.array([fin.get('x', np.nan), fin.get('y', np.nan),
                     fin.get('z', np.nan)], dtype=np.float32)
    d_prod = float(np.linalg.norm(prod - truth))

    rows = sb.get('rows') or []
    cand = np.array([[r['x'], r['y'], r['z']] for r in rows], dtype=np.float32) \
        if rows else np.empty((0, 3), np.float32)
    d_cand = float(np.linalg.norm(cand - truth, axis=1).min()) if len(cand) else np.inf

    voxels = sb.get('voxels') or []
    vox = np.array([[v['x'], v['y'], v['z']] for v in voxels], dtype=np.float32) \
        if voxels else np.empty((0, 3), np.float32)
    d_vox = float(np.linalg.norm(vox - truth, axis=1).min()) if len(vox) else np.inf

    if d_prod <= tol:
        cls = 'correct'
    elif d_cand > tol:
        cls = 'candidate-missing'
    elif d_vox <= tol:
        cls = 'selection-wrong'
    else:
        cls = 'net-wrong'
    return cls, d_prod, d_cand, d_vox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=ALL_TAGS)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--tol2', type=float, default=3.0, help='loose tolerance')
    ap.add_argument('--tsv', default=None)
    ap.add_argument('--numu-manifest', default=None,
                    help='snapshot manifest whose numu<K> sample column '
                         'defines the numu reporting subset (round 3)')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    numu = numu50_set(here, args.numu_manifest)

    recs = []
    for label in vio.iter_labels(args.sbnd_root, args.tags):
        calib = vio.load_calib(vio.calib_path_for_label(args.sbnd_root, label))
        sb = calib.get('vertex_scoreboard') or {}
        cls1, d_prod, d_cand, d_vox = classify(label, sb, args.tol)
        cls2, _, _, _ = classify(label, sb, args.tol2)
        recs.append(dict(
            evt=label['eventNo'], sample=vio.sample_of_label(label, numu),
            route=sb.get('route', ''), cls=cls1, cls_loose=cls2,
            d_prod=d_prod, d_cand=d_cand, d_vox=d_vox,
            corrective=int(vio.is_corrective(label, tol=args.tol))))

    def table(tol_key, tol):
        print('\n== taxonomy at tol = %.1f cm (%s) ==' % (tol, tol_key))
        print('%-8s %5s | %8s %12s %10s %10s' %
              ('sample', 'n', 'correct', 'cand-missing', 'net-wrong', 'sel-wrong'))
        samples = ['nuecc', 'ncpi0', 'mcp1k', 'numu50', 'ALL']
        for s in samples:
            if s == 'ALL':
                sel = recs
            elif s == 'numu50':
                sel = [r for r in recs if r['sample'] == 'numu50']
            elif s == 'mcp1k':
                sel = [r for r in recs if r['sample'] in ('mcp1k', 'numu50')]
            else:
                sel = [r for r in recs if r['sample'] == s]
            if not sel:
                continue
            counts = {c: sum(1 for r in sel if r[tol_key] == c) for c in CLASSES}
            assert sum(counts.values()) == len(sel), (s, counts)
            print('%-8s %5d | %8d %12d %10d %10d' %
                  (s, len(sel), counts['correct'], counts['candidate-missing'],
                   counts['net-wrong'], counts['selection-wrong']))

    table('cls', args.tol)
    table('cls_loose', args.tol2)

    miss = [r for r in recs if r['cls'] != 'correct']
    print('\nmisses at tol %.1f: %d' % (args.tol, len(miss)))
    for r in sorted(miss, key=lambda r: -r['d_prod']):
        print('  evt%-8d %-7s %-17s d_prod=%7.2f d_cand=%7.2f d_vox=%7.2f  route=%s'
              % (r['evt'], r['sample'], r['cls'], r['d_prod'],
                 r['d_cand'], r['d_vox'], r['route']))

    if args.tsv:
        cols = ['evt', 'sample', 'route', 'corrective', 'cls', 'cls_loose',
                'd_prod', 'd_cand', 'd_vox']
        with open(args.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in recs:
                fh.write('\t'.join('%.3f' % r[c] if isinstance(r[c], float)
                                   else str(r[c]) for c in cols) + '\n')
        print('\nwrote %s' % args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
