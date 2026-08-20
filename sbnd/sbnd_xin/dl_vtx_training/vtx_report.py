#!/usr/bin/env python3
'''
doc pr/100 -- the neutrino-vertex report card.

Not a new scorer.  `ab_vertex_compare.py` (per-event d_new vs truth, taxonomy
recomputed from the NEW scoreboard, route flips, candidate-set identity) and
`taxonomy.classify()` (the four-way decomposition) already do the scoring;
this is the thin driver that prints the numbers a re-rank tuning round is
actually gated on, registered here so no future round has to re-derive what
"correct" means:

  - BOTH tolerances, always, side by side: 1.0 cm is the registered gate, 1.5
    cm is always reported beside it.  Non-negotiable -- pr/89 round 5's
    selection-only s_topo was -2/1014 at 1.0 cm and +2/1014 at 1.5 cm.  A tool
    that only prints one number lets the tolerance pick the round's verdict
    without anyone deciding that on purpose.
  - split by label_source: human is primary, ai-scanner is secondary and
    carries a measured ~2.5% error floor (doc pr/88 sec 7, the 39/40 gate).
  - raw (unweighted label-pool count) and IPW-weighted (doc pr/89 sec 1.2)
    side by side, the two estimands always labelled, never averaged together.
  - the four-way class (correct / candidate-missing / net-wrong /
    selection-wrong), plus the pr/82 admission-vs-selection framing: only
    `candidate-missing` is an admission gap (no net training or rerank tuning
    reaches it); `net-wrong` + `selection-wrong` is the target.
  - the sampling-limit caveat prints on every run: every pr/88 fill tier
    selected on `not agrees`, so the REVIEW-agree stratum has near-zero owner
    labels and pool accuracy is not production accuracy.

Arm resolution is keyed by SAMPLE (nuecc/ncpi0/mcp1k/mcp2k), not by scan tag,
unlike ab_vertex_compare.py's --arm-roots.  A tag-keyed map breaks on
vtxscan-harv3-delta, which spans several samples' arms (doc pr/82 sec 4 --
the same fact carry_labels.py's --arms {sample} placeholder and
build_dataset.py's --harvest-roots @arm both exist to handle).
scn_vtx.io.sample_of_label() already resolves delta correctly per label, so
keying on its output sidesteps the problem instead of re-solving it.

Usage:
  python3 vtx_report.py \
      --tags vtxscan-harv3-nuecc48 vtxscan-harv3-ncpi0 vtxscan-harv3-mcp1k \
             vtxscan-harv3-delta vtxscan-mcp2k vtxscan-mcp2k-auto \
             vtxscan-mcp2k-ragree \
      --arm-roots nuecc=work-vtx100-base-nuecc48 ncpi0=work-vtx100-base-ncpi0 \
                  mcp1k=work-vtx100-base-mcp1k mcp2k=work-vtx100-base-mcp2k \
      --ipw-tsv runs/ipw-mcp2k-<date>.tsv \
      --tsv runs/vtx100-report-<date>.tsv
'''
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from taxonomy import classify, CLASSES

TOLS = (1.0, 1.5)
SAMPLES = ('nuecc', 'ncpi0', 'mcp1k', 'mcp2k')

SAMPLING_LIMIT_NOTE = (
    "SAMPLING LIMIT (doc pr/88): every pr/88 mcp2k fill tier was selected on "
    "`not agrees` -- REVIEW-agree events have near-zero owner labels, so the "
    "unweighted mcp2k accuracy below is NOT production accuracy on that arm. "
    "Read it beside the IPW estimate, and treat an UNDEF stratum in the IPW "
    "table as a bound, not a point (see ipw_weights.py).")


def parse_sample_roots(pairs):
    """['sample=path', ...] -> {sample: path}.  Keys must be a subset of
    SAMPLES; unlisted samples have no root and their events are reported
    as missing rather than silently skipped."""
    out = {}
    for p in pairs:
        s, _, path = p.partition('=')
        if not path:
            raise SystemExit('bad --arm-roots entry (want sample=path): %s' % p)
        if s not in SAMPLES:
            raise SystemExit('unknown sample %r (want one of %s)' % (s, SAMPLES))
        out[s] = path
    return out


def load_ipw(path):
    if not path:
        return {}
    with open(path) as fh:
        return {int(r['evt']): float(r['weight'])
                for r in csv.DictReader(fh, delimiter='\t')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True,
                    help='label tags to score -- no default (taxonomy.py\'s '
                         'ALL_TAGS default is stale; passing it here would '
                         'silently drop everything past pr/78)')
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--arm-roots', nargs='+', required=True,
                    help='sample=path pairs, sample in %s' % (SAMPLES,))
    ap.add_argument('--ipw-tsv', default=None,
                    help='ipw_weights.py output; enables the IPW estimand '
                         'for whichever events it covers')
    ap.add_argument('--tsv', default=None, help='per-event record TSV')
    args = ap.parse_args()

    roots = parse_sample_roots(args.arm_roots)
    ipw = load_ipw(args.ipw_tsv)

    recs, missing = [], []
    for label in vio.iter_labels(args.sbnd_root, args.tags):
        evt = label['eventNo']
        sample = vio.sample_of_label(label)
        root = roots.get(sample)
        if root is None:
            missing.append((evt, label['scan_tag'], sample, 'no arm-root for sample'))
            continue
        path = os.path.join(args.sbnd_root, root, 'pr_evt%d' % evt,
                            'calib-pr-evt%d.json' % evt)
        if not os.path.exists(path):
            missing.append((evt, label['scan_tag'], sample, path))
            continue

        calib = vio.load_calib(path)
        sb = calib.get('vertex_scoreboard') or {}
        mv = calib.get('main_vertex') or {}
        if not mv and sb.get('filled'):
            mv = dict(x=sb['final_x'], y=sb['final_y'], z=sb['final_z'])
        new_label = dict(label, main_vertex=mv)

        cls = {}
        for tol in TOLS:
            c, d_prod, d_cand, d_vox = classify(new_label, sb, tol)
            cls[tol] = dict(cls=c, d_prod=d_prod, d_cand=d_cand, d_vox=d_vox)

        recs.append(dict(
            evt=evt, tag=label['scan_tag'], sample=sample,
            label_source=label['label_source'],
            weight=ipw.get(evt), route=sb.get('route', ''),
            **{'tol%.1f' % t: cls[t] for t in TOLS}))

    if missing:
        print('MISSING calibs / no arm-root: %d' % len(missing))
        for evt, tag, sample, why in missing[:20]:
            print('  evt%-8d %-25s sample=%-6s %s' % (evt, tag, sample, why))
        if len(missing) > 20:
            print('  ... (%d more)' % (len(missing) - 20))

    def acc(sel, tol, weighted=False):
        if not sel:
            return None
        key = 'tol%.1f' % tol
        if weighted:
            sel = [r for r in sel if r['weight'] is not None]
            if not sel:
                return None
            wsum = sum(r['weight'] for r in sel)
            ok = sum(r['weight'] for r in sel if r[key]['cls'] == 'correct')
            return ok / wsum, len(sel)
        ok = sum(1 for r in sel if r[key]['cls'] == 'correct')
        return ok / len(sel), len(sel)

    def class_counts(sel, tol):
        key = 'tol%.1f' % tol
        return {c: sum(1 for r in sel if r[key]['cls'] == c) for c in CLASSES}

    print('\n== registered gate: correct within tol, raw (unweighted) ==')
    header = '%-12s %6s ' % ('group', 'n') + ' '.join(
        '%9s' % ('%.1fcm' % t) for t in TOLS)
    print(header)
    groups = [('ALL', recs)]
    groups += [(s, [r for r in recs if r['sample'] == s]) for s in SAMPLES]
    groups += [('human', [r for r in recs if r['label_source'] == 'human']),
               ('ai-scanner', [r for r in recs if r['label_source'] == 'ai-scanner'])]
    for name, sel in groups:
        if not sel:
            continue
        cells = []
        for t in TOLS:
            r = acc(sel, t)
            cells.append('%8.1f%%' % (100 * r[0]) if r else '     n/a')
        print('%-12s %6d ' % (name, len(sel)) + ' '.join(cells))

    if ipw:
        print('\n== IPW mcp2k-arm estimate (doc pr/89 sec 1.2 -- do not mix '
              'with the raw table above) ==')
        mcp2k = [r for r in recs if r['sample'] == 'mcp2k']
        for t in TOLS:
            r = acc(mcp2k, t, weighted=True)
            if r:
                print('  tol %.1fcm: %.1f%%  (n=%d weighted events; %d mcp2k '
                      'labels have no weight and are excluded from this line)'
                      % (t, 100 * r[0], r[1],
                         sum(1 for x in mcp2k if x['weight'] is None)))

    print('\n== four-way decomposition (pr/82: only candidate-missing is an '
          'admission gap; net-wrong + selection-wrong is the tuning target) ==')
    for t in TOLS:
        c = class_counts(recs, t)
        n = len(recs)
        print('  tol %.1fcm: correct %d  candidate-missing %d  net-wrong %d  '
              'selection-wrong %d  (of %d)'
              % (t, c['correct'], c['candidate-missing'], c['net-wrong'],
                 c['selection-wrong'], n))

    print('\n' + SAMPLING_LIMIT_NOTE)

    if args.tsv:
        cols = ['evt', 'tag', 'sample', 'label_source', 'weight', 'route']
        for t in TOLS:
            cols += ['cls%.1f' % t, 'd_prod%.1f' % t]
        with open(args.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in recs:
                row = [r['evt'], r['tag'], r['sample'], r['label_source'],
                       '' if r['weight'] is None else '%.6f' % r['weight'],
                       r['route']]
                for t in TOLS:
                    row += [r['tol%.1f' % t]['cls'],
                            '%.4f' % r['tol%.1f' % t]['d_prod']]
                fh.write('\t'.join(str(x) for x in row) + '\n')
        print('\nwrote %s' % args.tsv)

    return 0 if not missing else 1


if __name__ == '__main__':
    sys.exit(main())
