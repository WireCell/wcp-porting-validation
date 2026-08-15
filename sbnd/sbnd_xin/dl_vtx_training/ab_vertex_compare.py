#!/usr/bin/env python3
'''
doc pr/79 -- score a NEW production arm against the hand-scan vertex labels.

taxonomy.py cannot do this: it trusts label['source'] (an absolute path into
the OLD prod0813 arm, which still exists) and reads the production answer
from the frozen label['main_vertex'], never from a calib.  This tool resolves
each event's calib in an explicitly-given arm root, reads the NEW answer from
that calib, and compares against both the truth (rank-1 pick) and the frozen
prod0813 baseline answer.

Per event:  d_new = |new main_vertex - truth|, d_base = |label main_vertex -
truth| (or a second arm via --base-roots), taxonomy class recomputed from the
NEW scoreboard, route flips, and the candidate-set identity check (the step-1
"replay assumption": only the acceptance threshold moved, so rows[] should be
unchanged).

Transitions: fixed (base wrong -> new correct), regressed (correct -> wrong),
still-correct, still-wrong.

Missing calibs are listed and counted, never silently dropped.

Usage (identity check -- must reproduce the pr/78 s3 taxonomy counts, zero
transitions):
  python3 ab_vertex_compare.py \
      --arm-roots vtxscan-prod0813=work-nuecc48-prod0813 \
                  vtxscan-prod0813-ncpi0=work-ncpi0-prod0813 \
                  vtxscan-prod0813-mcp1k=work-mcp1k-prod0813

Usage (step-1 arm):
  python3 ab_vertex_compare.py \
      --arm-roots vtxscan-prod0813=work-nuecc48-ma10 \
                  vtxscan-prod0813-ncpi0=work-ncpi0-ma10 \
                  vtxscan-prod0813-mcp1k=work-mcp1k-ma10 \
      --numu-manifest data/full473/manifest.tsv \
      --lockbox-manifest data/full473/manifest.tsv \
      --tsv runs/ab-ma10-<date>.tsv
'''
import argparse
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from taxonomy import classify, CLASSES, ALL_TAGS


def parse_roots(pairs, sbnd_root):
    """['tag=path', ...] or a single bare path (applies to all tags)."""
    if len(pairs) == 1 and '=' not in pairs[0]:
        return {None: os.path.join(sbnd_root, pairs[0])}
    out = {}
    for p in pairs:
        tag, _, path = p.partition('=')
        if not path:
            raise SystemExit('bad --*-roots entry (want tag=path): %s' % p)
        out[tag] = os.path.join(sbnd_root, path)
    return out


def calib_in_root(roots, label):
    root = roots.get(label['scan_tag']) if roots.get(label['scan_tag']) \
        else roots.get(None)
    if root is None:
        return None
    evt = label['eventNo']
    return os.path.join(root, 'pr_evt%d' % evt, 'calib-pr-evt%d.json' % evt)


def rows_key(sb, ndigits=3):
    """Order-independent identity of the candidate set (xyz rounded)."""
    rows = sb.get('rows') or []
    return tuple(sorted((round(r['x'], ndigits), round(r['y'], ndigits),
                         round(r['z'], ndigits)) for r in rows))


def flags_set(manifest, column):
    if not manifest:
        return set()
    with open(manifest) as fh:
        return {int(r['evt']) for r in csv.DictReader(fh, delimiter='\t')
                if r.get(column) == '1'}


def numu_set(manifest):
    if not manifest:
        return set()
    with open(manifest) as fh:
        return {int(r['evt']) for r in csv.DictReader(fh, delimiter='\t')
                if r.get('sample', '').startswith('numu')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=ALL_TAGS)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--arm-roots', nargs='+', required=True,
                    help='tag=path pairs (or one bare path) for the NEW arm')
    ap.add_argument('--base-roots', nargs='+', default=None,
                    help='optional tag=path pairs for the baseline arm; '
                         'default = the frozen label main_vertex (prod0813)')
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--numu-manifest', default=None)
    ap.add_argument('--lockbox-manifest', default=None,
                    help='manifest whose lockbox=1 rows define the '
                         'never-trained-on reporting subset (step 2)')
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    arm_roots = parse_roots(args.arm_roots, args.sbnd_root)
    base_roots = parse_roots(args.base_roots, args.sbnd_root) \
        if args.base_roots else None
    numu = numu_set(args.numu_manifest)
    lockbox = flags_set(args.lockbox_manifest, 'lockbox')

    recs, missing = [], []
    for label in vio.iter_labels(args.sbnd_root, args.tags):
        evt = label['eventNo']
        truth = label['truth_xyz']

        new_path = calib_in_root(arm_roots, label)
        if not new_path or not os.path.exists(new_path):
            missing.append((evt, label['scan_tag'], new_path))
            continue
        new_calib = vio.load_calib(new_path)
        new_sb = new_calib.get('vertex_scoreboard') or {}
        new_mv = new_calib.get('main_vertex') or {}
        if not new_mv and new_sb.get('filled'):
            new_mv = dict(x=new_sb['final_x'], y=new_sb['final_y'],
                          z=new_sb['final_z'])
        new_label = dict(label, main_vertex=new_mv)
        cls_new, d_new, d_cand_new, d_vox_new = classify(new_label, new_sb,
                                                         args.tol)

        # baseline: frozen label main_vertex + the OLD calib's scoreboard
        old_path = calib_in_root(base_roots, label) if base_roots \
            else vio.calib_path_for_label(args.sbnd_root, label)
        base_sb, cls_base, d_base = {}, '', np.inf
        if os.path.exists(old_path):
            old_calib = vio.load_calib(old_path)
            base_sb = old_calib.get('vertex_scoreboard') or {}
            base_label = label if not base_roots else \
                dict(label, main_vertex=old_calib.get('main_vertex') or {})
            cls_base, d_base, _, _ = classify(base_label, base_sb, args.tol)

        ok_new, ok_base = d_new <= args.tol, d_base <= args.tol
        trans = ('still-correct' if ok_base and ok_new else
                 'regressed' if ok_base else
                 'fixed' if ok_new else 'still-wrong')
        recs.append(dict(
            evt=evt, sample=vio.sample_of_label(label, numu, 'numu'),
            lockbox=int(evt in lockbox),
            d_base=float(d_base), d_new=float(d_new),
            cls_base=cls_base, cls_new=cls_new,
            route_base=base_sb.get('route', ''),
            route_new=new_sb.get('route', ''),
            rows_same=int(bool(base_sb) and
                          rows_key(base_sb) == rows_key(new_sb)),
            min_accept=new_sb.get('dl_min_accept_score'),
            top_k=new_sb.get('dl_top_k'),
            trans=trans))

    if missing:
        print('MISSING calibs: %d' % len(missing))
        for evt, tag, path in missing:
            print('  evt%-8d %-25s %s' % (evt, tag, path))

    def summarize(name, sel):
        if not sel:
            return
        n = len(sel)
        cb = sum(1 for r in sel if r['d_base'] <= args.tol)
        cn = sum(1 for r in sel if r['d_new'] <= args.tol)
        t = {k: sum(1 for r in sel if r['trans'] == k)
             for k in ('fixed', 'regressed', 'still-correct', 'still-wrong')}
        print('%-8s %5d | base %3d  new %3d  (%+d) | fixed %3d regressed %3d'
              % (name, n, cb, cn, cn - cb, t['fixed'], t['regressed']))

    print('\n== correct within %.1f cm (n=%d scored) ==' %
          (args.tol, len(recs)))
    for s in ('nuecc', 'ncpi0', 'mcp1k'):
        summarize(s, [r for r in recs if r['sample'] in
                      ((s, 'numu') if s == 'mcp1k' else (s,))])
    summarize('numu', [r for r in recs if r['sample'] == 'numu'])
    summarize('ALL', recs)
    if lockbox:
        summarize('lockbox', [r for r in recs if r['lockbox']])

    print('\n== taxonomy of the NEW arm ==')
    counts = {c: sum(1 for r in recs if r['cls_new'] == c) for c in CLASSES}
    assert sum(counts.values()) == len(recs)
    print('  ' + '  '.join('%s %d' % (c, counts[c]) for c in CLASSES))

    same = sum(1 for r in recs if r['rows_same'])
    print('\ncandidate-set identical to baseline: %d/%d' % (same, len(recs)))
    flips = [r for r in recs if r['route_new'] != r['route_base']]
    print('route flips: %d' % len(flips))

    prov = {(r['min_accept'], r['top_k']) for r in recs}
    print('provenance (min_accept, top_k) seen: %s' % sorted(
        prov, key=lambda p: (p[0] is None, p)))

    reg = [r for r in recs if r['trans'] == 'regressed']
    if reg:
        print('\nREGRESSED events (%d):' % len(reg))
        for r in sorted(reg, key=lambda r: -r['d_new']):
            print('  evt%-8d %-6s d %6.2f -> %6.2f  route %s -> %s  cls %s'
                  % (r['evt'], r['sample'], r['d_base'], r['d_new'],
                     r['route_base'], r['route_new'], r['cls_new']))
    fx = [r for r in recs if r['trans'] == 'fixed']
    if fx:
        print('\nFIXED events (%d):' % len(fx))
        for r in sorted(fx, key=lambda r: -r['d_base']):
            print('  evt%-8d %-6s d %6.2f -> %6.2f  route %s -> %s  was %s'
                  % (r['evt'], r['sample'], r['d_base'], r['d_new'],
                     r['route_base'], r['route_new'], r['cls_base']))

    if args.tsv:
        cols = ['evt', 'sample', 'lockbox', 'd_base', 'd_new', 'cls_base',
                'cls_new', 'route_base', 'route_new', 'rows_same', 'trans']
        with open(args.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in recs:
                fh.write('\t'.join('%.3f' % r[c] if isinstance(r[c], float)
                                   else str(r[c]) for c in cols) + '\n')
        print('\nwrote %s' % args.tsv)
    return 0 if not missing else 1


if __name__ == '__main__':
    sys.exit(main())
