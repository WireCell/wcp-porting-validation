#!/usr/bin/env python3
'''doc pr/105 -- one accuracy table over many arms (vertex-selection strategies).

Not a new scorer: "correct" is |final main vertex - label| <= tol, exactly the
d_prod leg of taxonomy.classify(); this driver only lays several arms side by
side on the SAME label set so strategies are compared on identical events.

Truth rule: the carried label (vtxscan-<carry>-*, re-anchored on the base arm
by position, TOL 1.0) when the event carried, else the original frozen label
(pr/100 sec 3.3: carrying is a filter; on the same events the two truths are
statistically indistinguishable).  Denominator = every label event (default)
or --carried-only.  --exclude-events / --only-events honour a sealed lockbox.

Output: rows = arms, columns = sample x {1.0, 1.5} cm, raw counts and
percentages, plus human / ai-scanner split and the IPW mcp2k estimand when
--ipw-tsv is given.  --events-tsv writes d per event per arm (the mover
ledgers and Bee lists are built from it).

Usage:
  python3 vtx_strategy_table.py \\
      --carried-tags vtxscan-vtx105-nuecc48 ... --orig-tags vtxscan-harv3-nuecc48 ... \\
      --arms base=work-vtx105-base-{sample} dlonly=work-vtx105-dlonly-{sample} ... \\
      --exclude-events runs/vtx100-20260820/lockbox.txt --events-tsv out.tsv
'''
import argparse, collections, csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio

TOLS = (1.0, 1.5)
SAMPLES = ('nuecc', 'ncpi0', 'mcp1k', 'mcp2k')
ARM_SAMPLE = dict(nuecc='nuecc48', ncpi0='ncpi0', mcp1k='mcp1k', mcp2k='mcp2k')


def read_event_file(path):
    if not path:
        return None
    out = set()
    with open(path) as fh:
        for line in fh:
            line = line.split('#', 1)[0].strip()
            if line:
                out.add(int(line))
    return out


def load_truth(root, carried_tags, orig_tags):
    '''evt -> dict(truth, sample, label_source, carried)'''
    truth = {}
    for tags, carried in ((carried_tags, True), (orig_tags, False)):
        if not tags:
            continue
        for lab in vio.iter_labels(root, tags):
            e = lab['eventNo']
            if e in truth:
                continue
            truth[e] = dict(truth=np.array(lab['truth_xyz'], dtype=float),
                            sample=vio.sample_of_label(lab),
                            label_source=lab.get('label_source', 'human'),
                            carried=carried)
    return truth


def final_xyz(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    mv = d.get('main_vertex') or {}
    if mv and mv.get('x') is not None:
        return np.array([mv['x'], mv['y'], mv['z']], dtype=float)
    sb = d.get('vertex_scoreboard') or {}
    if sb.get('filled'):
        return np.array([sb['final_x'], sb['final_y'], sb['final_z']], dtype=float)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--carried-tags', nargs='*', default=[])
    ap.add_argument('--orig-tags', nargs='*', default=[])
    ap.add_argument('--arms', nargs='+', required=True,
                    help='name=dir-template with {sample} (nuecc48/ncpi0/mcp1k/mcp2k)')
    ap.add_argument('--ipw-tsv', default=None)
    ap.add_argument('--exclude-events', default=None)
    ap.add_argument('--only-events', default=None)
    ap.add_argument('--carried-only', action='store_true')
    ap.add_argument('--events-tsv', default=None)
    ap.add_argument('--md', action='store_true', help='markdown table output')
    args = ap.parse_args()

    excl = read_event_file(args.exclude_events)
    only = read_event_file(args.only_events)
    truth = load_truth(args.sbnd_root, args.carried_tags, args.orig_tags)
    ipw = {}
    if args.ipw_tsv:
        with open(args.ipw_tsv) as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                ipw[int(r['evt'])] = float(r['weight'])
    arms = []
    for a in args.arms:
        name, tpl = a.split('=', 1)
        arms.append((name, tpl))

    events = sorted(e for e in truth
                    if not (excl and e in excl)
                    and (only is None or e in only)
                    and (not args.carried_only or truth[e]['carried']))
    # d per event per arm
    D = {}
    missing = collections.Counter()
    for e in events:
        t = truth[e]
        for name, tpl in arms:
            p = os.path.join(args.sbnd_root, tpl.format(sample=ARM_SAMPLE[t['sample']]),
                             'pr_evt%d' % e, 'calib-pr-evt%d.json' % e)
            xyz = final_xyz(p)
            if xyz is None:
                missing[name] += 1
                D[(e, name)] = None
            else:
                D[(e, name)] = float(np.linalg.norm(xyz - t['truth']))

    def table(sel, title):
        groups = [('ALL', sel)] + [(s, [e for e in sel if truth[e]['sample'] == s]) for s in SAMPLES]
        groups += [('human', [e for e in sel if truth[e]['label_source'] == 'human']),
                   ('ai-scanner', [e for e in sel if truth[e]['label_source'] != 'human'])]
        groups = [(g, ev) for g, ev in groups if ev]
        print('\n== %s ==' % title)
        if args.md:
            print('| arm | ' + ' | '.join('%s n=%d 1.0 / 1.5' % (g, len(ev)) for g, ev in groups) + ' |')
            print('|---|' + '---|' * len(groups))
        else:
            print('%-10s ' % 'arm' + ' '.join('%22s' % ('%s(n=%d)' % (g, len(ev))) for g, ev in groups))
        for name, _ in arms:
            cells = []
            for g, ev in groups:
                c = []
                for tol in TOLS:
                    ok = sum(1 for e in ev if D[(e, name)] is not None and D[(e, name)] <= tol)
                    c.append('%d (%.1f%%)' % (ok, 100.0 * ok / len(ev)))
                cells.append(' / '.join(c))
            if args.md:
                print('| %s | %s |' % (name, ' | '.join(cells)))
            else:
                print('%-10s ' % name + ' '.join('%22s' % x for x in cells))
        if ipw:
            m = [e for e in sel if truth[e]['sample'] == 'mcp2k' and e in ipw]
            if m:
                wsum = sum(ipw[e] for e in m)
                print('IPW mcp2k-arm (n=%d weighted; do not mix with raw): ' % len(m) + '  '.join(
                    '%s %.1f%%/%.1f%%' % (name, *[100.0 * sum(ipw[e] for e in m if D[(e, name)] is not None and D[(e, name)] <= tol) / wsum for tol in TOLS])
                    for name, _ in arms))

    ncar = sum(1 for e in events if truth[e]['carried'])
    print('events %d (carried %d, frozen %d); excluded %d; arms %s'
          % (len(events), ncar, len(events) - ncar, len(excl or ()), [a for a, _ in arms]))
    if missing:
        print('MISSING dumps per arm: %s' % dict(missing))
    table(events, 'all label events (carried truth where carried, frozen otherwise)')
    if not args.carried_only and ncar != len(events):
        table([e for e in events if truth[e]['carried']], 'carried subset only')

    if args.events_tsv:
        with open(args.events_tsv, 'w') as fh:
            fh.write('evt\tsample\tlabel_source\tcarried\t' + '\t'.join('d_' + a for a, _ in arms) + '\n')
            for e in events:
                t = truth[e]
                fh.write('%d\t%s\t%s\t%d\t' % (e, t['sample'], t['label_source'], int(t['carried'])) +
                         '\t'.join('' if D[(e, a)] is None else '%.3f' % D[(e, a)] for a, _ in arms) + '\n')
        print('wrote %s' % args.events_tsv)
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
