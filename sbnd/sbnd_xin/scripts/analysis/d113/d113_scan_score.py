#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 3 -- score the blind scans: join each scanner's labels.tsv with the panel sidecars
(class, Q, Qfrac; hidden from the scanners) and print precision per class, the control false-positive rate,
scanner agreement, and the YES list with the census numbers.

Usage: d113_scan_score.py [--tags a b] [--panels docs/113_figs/panels] [--out docs/113_figs/113_scan_score.txt]
"""
import argparse, collections, csv, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=['a', 'b'])
    ap.add_argument('--panels', default=f'{C.SX}/docs/113_figs/panels')
    ap.add_argument('--out', default=f'{C.SX}/docs/113_figs/113_scan_score.txt')
    ap.add_argument('--labels-prefix', dest='prefix', default='mscan-d113-')
    ap.add_argument('--classes', nargs='+', default=['NOBLOB-SLICE', 'NOBLOB-INSLICE', 'THRESH', 'CONTROL'])
    a = ap.parse_args()
    side = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f'{a.panels}/*.json')}
    # join the PR-recovery share (fq_cov4: the component's charge already held by the candidate's T_proj_data)
    fq = {}
    flp = f'{C.SX}/docs/113_figs/113_flags.tsv'
    if os.path.exists(flp):
        for r in csv.DictReader(open(flp), delimiter='\t'):
            if 'fq_cov4' in r:
                fq[(r['sample'], int(r['event']), r['cls'], int(r['apa']), int(r['plane']), int(r['comp']))] = float(r['fq_cov4'])
    for it, x in side.items():
        x['fq_cov4'] = fq.get((x['sample'], x['event'], x['cls'], x.get('apa'), x.get('plane'), x.get('comp')), float('nan')) if 'apa' in x else float('nan')
    lab = {}
    for t in a.tags:
        p = f'{C.SX}/missing_labels/{a.prefix}{t}/labels.tsv'
        rows = list(csv.DictReader(open(p), delimiter='\t'))
        lab[t] = {r['item'].strip().replace('.png', ''): r for r in rows}
    lines = []
    log = lambda s: (print(s), lines.append(s))
    items = sorted(side)
    log(f'items {len(items)}; scanners {a.tags}: ' + ', '.join(f'{t} labelled {len(lab[t])}' for t in a.tags))
    # per class x scanner verdict counts
    for t in a.tags:
        log(f'\nscanner {t}: verdict by class (hidden from the scanner)')
        tab = collections.defaultdict(collections.Counter)
        for it in items:
            v = lab[t].get(it, {}).get('verdict', 'MISSING').strip().upper()
            tab[side[it]['cls']][v] += 1
        for cls in a.classes:
            c = tab.get(cls)
            if not c:
                continue
            n = sum(c.values()); y = c.get('YES', 0); u = c.get('UNSURE', 0)
            log(f'  {cls:15s} n={n:3d}  YES {y:3d} ({100*y/n:.0f} %)  NO {c.get("NO",0):3d}  UNSURE {u:3d}  ' +
                ('<- precision (YES / all)' if cls != 'CONTROL' else '<- false-positive rate on covered charge'))
    # agreement
    if len(a.tags) >= 2:
        t0, t1 = a.tags[:2]
        agree = collections.Counter()
        for it in items:
            v0 = lab[t0].get(it, {}).get('verdict', '?').strip().upper(); v1 = lab[t1].get(it, {}).get('verdict', '?').strip().upper()
            agree[(v0, v1)] += 1
        both_yes = agree[('YES', 'YES')]; any_yes = sum(v for k, v in agree.items() if 'YES' in k)
        same = sum(v for k, v in agree.items() if k[0] == k[1])
        log(f'\nagreement {t0} vs {t1}: identical verdict on {same} / {len(items)}; YES/YES {both_yes}, YES by either {any_yes}')
        log('  pairs: ' + ', '.join(f'{k[0]}/{k[1]}:{v}' for k, v in sorted(agree.items())))
        # consensus per class: YES/YES, YES+UNSURE, split
        log('\nconsensus by class (both YES | one YES one UNSURE | one YES one NO | neither):')
        for cls in a.classes:
            c = collections.Counter()
            for it in items:
                if side[it]['cls'] != cls:
                    continue
                vs = sorted([lab[t0].get(it, {}).get('verdict', '?').strip().upper(), lab[t1].get(it, {}).get('verdict', '?').strip().upper()])
                if vs == ['YES', 'YES']: c['both'] += 1
                elif vs == ['UNSURE', 'YES']: c['yes+unsure'] += 1
                elif vs == ['NO', 'YES']: c['split'] += 1
                else: c['neither'] += 1
            n = sum(c.values())
            if n:
                log(f'  {cls:15s} n={n:3d}  both {c["both"]}  yes+unsure {c["yes+unsure"]}  split {c["split"]}  neither {c["neither"]}  -> strict precision {100*c["both"]/n:.0f} %, lenient {100*(c["both"]+c["yes+unsure"])/n:.0f} %')
    # the YES list
    log('\nitems judged YES by every scanner (item, class, plane, Q, Qfrac, kinds, comments):')
    for it in items:
        vs = [lab[t].get(it, {}).get('verdict', '?').strip().upper() for t in a.tags]
        if all(v == 'YES' for v in vs):
            s = side[it]
            kinds = '/'.join(lab[t].get(it, {}).get('kind', '?').strip() for t in a.tags)
            com = ' | '.join(lab[t].get(it, {}).get('comment', '').strip()[:70] for t in a.tags)
            pl = "UVW"[int(s["plane"])] if "plane" in s else "-"
            log(f'  {it:24s} {s["cls"]:15s} {pl}  Q={float(s["Q"]):.3g} Qfrac={float(s["Qfrac"]):.3f} PRrec={s.get("fq_cov4", float("nan")):.2f}  {kinds}  {com}')
    # PR recovery of the YES items: how much of the un-imaged charge the candidate holds anyway
    ys = [side[it] for it in items if any(lab[t].get(it, {}).get('verdict', '?').strip().upper() == 'YES' for t in a.tags) and side[it]['cls'] != 'CONTROL']
    import math
    v = [x['fq_cov4'] for x in ys if not math.isnan(x['fq_cov4'])]
    if v:
        log(f'\nPR re-tiling recovery on the {len(v)} items judged YES by either scanner: share of the un-imaged charge already in T_proj_data: '
            f'median {sorted(v)[len(v)//2]:.2f}; items with < 0.5 recovered (energy-relevant loss): {sum(1 for q in v if q < 0.5)}; >= 0.9: {sum(1 for q in v if q >= 0.9)}')
    log('\ncontrol items judged YES by any scanner:')
    for it in items:
        if side[it]['cls'] == 'CONTROL':
            vs = [lab[t].get(it, {}).get('verdict', '?').strip().upper() for t in a.tags]
            if 'YES' in vs:
                log(f'  {it}: {vs}  ' + ' | '.join(lab[t].get(it, {}).get('comment', '')[:60] for t in a.tags))
    open(a.out, 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
