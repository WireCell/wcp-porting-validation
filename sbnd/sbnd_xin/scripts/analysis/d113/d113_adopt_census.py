#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 7 -- what nu_adopt_touching did on an ON arm: every "[nu_adopt_touching] ... adopted cluster"
log line, joined to the OFF/ON per-event Enu, nu_evaluated and scores (pr150 metrics tables) and to the PR-stage
census flags of the OFF arm (was the adopted cluster a flagged NEARBY-UNMATCHED item / a scan YES?).

Usage: d113_adopt_census.py --on d113adopt [--off pr150s0] [--out docs/113_figs/113_adopt_census.txt]
"""
import argparse, csv, glob, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C

pat = re.compile(r'\[nu_adopt_touching\] gid (-?\d+): adopted cluster (\d+) \(gid (-?\d+), L ([\d.]+) cm, closest approach ([\d.]+) cm\) as a companion of main (\d+)')


def metrics(arm):
    out = {}
    for f in glob.glob(f'{C.SX}/docs/pr/150_figs/metrics/{arm}-*.tsv'):
        for r in csv.DictReader(open(f), delimiter='\t'):
            out[(r['sample'], int(r['event']))] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--on', default='d113adopt'); ap.add_argument('--off', default='pr150s0')
    ap.add_argument('--out', default=f'{C.SX}/docs/113_figs/113_adopt_census.txt')
    a = ap.parse_args()
    lines = []; log = lambda s: (print(s), lines.append(s))
    adopted = collections.defaultdict(list)
    for s in C.SAMPLES:
        for lg in glob.glob(f'{C.SX}/work-{s}-{a.on}/pr_evt*/wct_pr_evt*.log'):
            ev = int(re.search(r'pr_evt(\d+)', lg).group(1))
            for m in pat.finditer(open(lg, errors='replace').read()):
                adopted[(s, ev)].append(dict(gid=int(m.group(1)), cluster=int(m.group(2)), cgid=int(m.group(3)), L=float(m.group(4)), d=float(m.group(5)), main=int(m.group(6))))
    n_ev = collections.Counter(k[0] for k in adopted)
    log(f'events with >= 1 adopted cluster: {sum(n_ev.values())} {dict(n_ev)}; adopted clusters {sum(len(v) for v in adopted.values())}; '
        f'length median {np.median([x["L"] for v in adopted.values() for x in v]) if adopted else 0:.1f} cm, closest approach median {np.median([x["d"] for v in adopted.values() for x in v]) if adopted else 0:.2f} cm')
    M0, M1 = metrics(a.off), metrics(a.on)
    flags = {}
    fp = f'{C.SX}/docs/113_figs/113_pr_flags.tsv'
    if os.path.exists(fp):
        for r in csv.DictReader(open(fp), delimiter='\t'):
            if r['kind'] == 'NEARBY':
                flags[(r['sample'], int(r['event']), int(float(r['cluster'])))] = r['cls']
    yes = set()
    for t in ('c', 'd'):
        p = f'{C.SX}/missing_labels/prscan-d113-{t}/labels.tsv'
        if os.path.exists(p):
            for r in csv.DictReader(open(p), delimiter='\t'):
                if r['verdict'].strip().upper() == 'YES':
                    yes.add(r['item'].strip())
    side = {}
    import json
    for pj in glob.glob(f'{C.SX}/docs/113_figs/pr_panels/*.json'):
        x = json.load(open(pj)); side[(x['sample'], int(x['event']), int(float(x.get('cluster', -1))))] = os.path.basename(pj)[:-5]
    log('\nadopted clusters (sample event cluster gid L d | OFF-census class | scan item & YES? | Enu OFF -> ON | nu_evaluated OFF/ON | numu OFF/ON):')
    dE = []
    for (s, ev), v in sorted(adopted.items()):
        m0, m1 = M0.get((s, ev), {}), M1.get((s, ev), {})
        e0, e1 = m0.get('Enu', ''), m1.get('Enu', '')
        try:
            dE.append(float(e1) - float(e0))
        except ValueError:
            pass
        for x in v:
            cls = flags.get((s, ev, x['cluster']), '-'); it = side.get((s, ev, x['cluster']), '-')
            log(f"  {s} {ev} cl {x['cluster']} gid {x['cgid']} L {x['L']:.1f} d {x['d']:.2f} | {cls} | {it} {'YES' if it in yes else ('no' if it != '-' else '')} | Enu {e0} -> {e1} | nu_eval {m0.get('nu_evaluated','')}/{m1.get('nu_evaluated','')} | numu {m0.get('numu_score','')}/{m1.get('numu_score','')}")
    if dE:
        dE = np.array(dE); log(f'\nEnu ON - OFF on adopted events: n {len(dE)}, median {np.median(dE):+.1f} MeV, mean {dE.mean():+.1f}, rose in {(dE>0).sum()}, fell in {(dE<0).sum()}')
    # inertness elsewhere
    common = set(M0) & set(M1); un = [k for k in common if k not in adopted]
    d_un = []
    for k in un:
        try: d_un.append(float(M1[k]['Enu']) - float(M0[k]['Enu']))
        except ValueError: pass
    d_un = np.array(d_un)
    if len(d_un):
        log(f'events with NO adoption: {len(un)}; Enu changed in {(np.abs(d_un)>0.5).sum()} of them (|dEnu| median {np.median(np.abs(d_un)):.2f} MeV, max {np.abs(d_un).max():.1f})')
    open(a.out, 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
