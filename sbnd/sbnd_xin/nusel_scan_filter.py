#!/usr/bin/env python3
"""Doc 59: pick the hand-scan subset out of a full-sample nusel production.

Classify every event of a work root by what the taggers said about its
IN-BEAM bundles, and emit the subset worth hand-scanning:

  KEEP an event iff
    (a) it has at least one in-beam bundle -- a table row with in_beam=1 whose
        label is not 'no-bundle' (that synthetic row is an in-window physical
        flash that matched NO qualifying bundle, i.e. nothing to scan), and
    (b) none of its in-beam bundles is labeled TGM or LM.

  So a kept event carries only STM-tagged and/or untagged ('nu-candidate')
  in-beam bundles -- the population a neutrino-selection scan is about.

The decision reads the table's `label` column, not the raw tgm/stm/lm columns:
`nusel_extract.label_of()` already applies the beam window and the
TGM > STM > LM priority, and the raw lm column is 0/1/2 (verdict codes), not a
boolean.

Events with BOTH a cosmic-tagged and a keepable in-beam bundle ("mixed") are
DROPPED by rule (b) and counted separately, so the cut can be revisited if that
class is ever large.  `--keep-mixed` keeps them instead (rule (b) becomes "at
least one in-beam bundle is STM or untagged"), which is what the doc-59 scan
runs with since the owner asked for those 9 events on 2026-07-26: a mixed event
is exactly the case where a hand scan decides something the taggers cannot
(evt280281 carries an STM and a TGM bundle in the same beam window).  The
verdict column keeps saying `mixed`, so the census stays comparable either way.

Usage:
  python3 nusel_scan_filter.py -w work-mcp1kall-d59k \\
      [--events-out scan-events.txt] [--tsv-list-out scan-tsvs.txt] \\
      [--census-out scan-census.tsv] [--chunk 100 --chunk-prefix scan-chunk]

  -w/--work-root is repeatable (first root that has the event wins), matching
  merge_mabc_bee.py / make_stmfit_bee.py.

Nothing is written unless the corresponding --*-out is given; the census always
goes to stdout.
"""
import argparse
import glob
import os
import sys

# label values produced by nusel_extract.label_of() / the no-bundle row
COSMIC = ('TGM', 'LM')          # in-beam labels that disqualify an event
KEEPABLE = ('STM', 'nu-candidate')


def read_tsv(path):
    """[(dict header->value)] for a nusel-evt<ID>.tsv (whitespace-aligned)."""
    with open(path) as f:
        rows = [ln.split() for ln in f.read().splitlines() if ln.strip()]
    if not rows:
        return []
    head = rows[0]
    return [dict(zip(head, r)) for r in rows[1:] if len(r) == len(head)]


def classify(rows):
    """(verdict, inbeam_labels) for one event's table rows.

    verdict: 'keep' | 'tgm' | 'lm' | 'mixed' | 'no-inbeam-bundle' | 'empty'
    """
    if not rows:
        return 'empty', []
    inbeam = [r['label'] for r in rows
              if r.get('in_beam') == '1' and r.get('label') != 'no-bundle']
    if not inbeam:
        return 'no-inbeam-bundle', []
    bad = [l for l in inbeam if l in COSMIC]
    if not bad:
        return 'keep', inbeam
    if any(l in KEEPABLE for l in inbeam):
        return 'mixed', inbeam
    # every in-beam bundle is cosmic-tagged: name it by the first cosmic label
    return ('tgm' if 'TGM' in bad else 'lm'), inbeam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--work-root', required=True, action='append',
                    help='work root holding nusel_evt<ID>/nusel-evt<ID>.tsv.  '
                         'Repeatable; the first root that has an event wins.')
    ap.add_argument('--events-out', help='write the kept event ids, one per line')
    ap.add_argument('--tsv-list-out',
                    help='write the kept events\' TSV paths, one per line '
                         '(feed to serve_nusel_scan.sh: the viewer\'s '
                         'discover_events() takes explicit TSV paths and '
                         'derives each work root from the path)')
    ap.add_argument('--census-out', help='write the per-event census TSV')
    ap.add_argument('--keep-mixed', action='store_true',
                    help='also keep events that have a cosmic-tagged AND a '
                         'keepable in-beam bundle (verdict "mixed"); they are '
                         'dropped by default')
    ap.add_argument('--chunk', type=int,
                    help='also write the kept ids in chunks of this size '
                         '(for one Bee upload per chunk)')
    ap.add_argument('--chunk-prefix', default='scan-chunk',
                    help='<prefix>-NN.txt for --chunk (default scan-chunk)')
    args = ap.parse_args()

    # event id -> tsv path, first work root wins
    seen = {}
    for root in args.work_root:
        for t in glob.glob(os.path.join(root, 'nusel_evt*', 'nusel-evt*.tsv')):
            evt = os.path.basename(t)[len('nusel-evt'):-len('.tsv')]
            seen.setdefault(evt, t)
    if not seen:
        sys.exit('ERROR: no nusel_evt*/nusel-evt*.tsv under: '
                 + ', '.join(args.work_root))

    census, counts, kept = [], {}, []
    n_stm = n_nucand = 0
    for evt in sorted(seen, key=int):
        rows = read_tsv(seen[evt])
        verdict, inbeam = classify(rows)
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict == 'keep' or (verdict == 'mixed' and args.keep_mixed):
            kept.append(evt)
            if 'STM' in inbeam:
                n_stm += 1
            else:
                n_nucand += 1
        census.append((evt, verdict, len(rows), len(inbeam), ','.join(inbeam) or '-'))

    tot = len(seen)
    print(f'events with a table: {tot}')
    for k in ('keep', 'tgm', 'lm', 'mixed', 'no-inbeam-bundle', 'empty'):
        if k in counts:
            tail = '  <- KEPT (--keep-mixed)' if (k == 'mixed' and args.keep_mixed) else ''
            print(f'  {k:>17}: {counts[k]:5d}  ({100.0 * counts[k] / tot:.1f}%){tail}')
    print(f'kept: {len(kept)}  (with an STM-tagged in-beam bundle: {n_stm}; '
          f'all-untagged: {n_nucand})')

    if args.census_out:
        with open(args.census_out, 'w') as f:
            f.write('event\tverdict\tn_row\tn_inbeam\tinbeam_labels\n')
            for row in census:
                f.write('\t'.join(str(c) for c in row) + '\n')
        print('census  -> ' + args.census_out)
    if args.events_out:
        with open(args.events_out, 'w') as f:
            f.write('\n'.join(kept) + '\n')
        print('events  -> ' + args.events_out)
    if args.tsv_list_out:
        with open(args.tsv_list_out, 'w') as f:
            f.write('\n'.join(os.path.abspath(seen[e]) for e in kept) + '\n')
        print('tsvs    -> ' + args.tsv_list_out)
    if args.chunk:
        for k in range(0, len(kept), args.chunk):
            name = f'{args.chunk_prefix}-{k // args.chunk:02d}.txt'
            with open(name, 'w') as f:
                f.write('\n'.join(kept[k:k + args.chunk]) + '\n')
            print(f'chunk   -> {name} ({len(kept[k:k + args.chunk])} events)')


main()
