#!/usr/bin/env python3
"""doc pdvd/39 -- who actually gets an STM fit, and why the rest do not.

Two censuses that doc 36/38's grading scripts cannot give:

1. The ABSOLUTE unfitted population.  d36_fit_twoaxis_scan.py builds its cluster
   list from the UNION of the arms' fit outputs, so a cluster that never got a
   fit in ANY arm is invisible to it.  That makes it structurally unable to
   answer "did we lose fits entirely?".  This script starts from the clustering
   output instead.

2. Restricted to the population TaggerCheckSTM actually evaluates.  The tagger
   loops over flash-matched MAIN clusters only (TaggerCheckSTM.cxx:562), so a
   census over every cluster in the event answers the wrong question -- most
   clusters are never offered to the fitter at all.  Main clusters are read from
   the tagger's own per-cluster verdict line, and the pre-fit exit reasons from
   its "no STM fit:" lines.

Cluster ids are counted as DISTINCT ids, never as log lines: the verdict line is
printed once per cluster per visitor pass and other components echo a matching
substring (feedback_log_line_count_is_not_object_count).

Usage:
  d39_fit_population.py <logdir>:<tag> [<logdir>:<tag> ...]
  # arms with no surviving logs still get census 1:
  d39_fit_population.py :<tag>
"""
import glob, json, os, re, sys, zipfile
import numpy as np
from multiprocessing import Pool

PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))                      # <repo>/pdvd
W = os.path.join(PDVD, 'work')
MINQ = 50
VERDICT = re.compile(r'visit: TaggerCheckSTM: cluster (\d+) . STM=(\d)')
NOFIT = re.compile(r'check_stm_conditions: cluster (\d+) no STM fit: (.*)')


def sizes_and_fits(zp):
    z = zipfile.ZipFile(zp)
    cl = json.loads(z.read('data/0/0-clustering-global.json'))
    pc = np.asarray(cl['cluster_id'], dtype=int)
    cids, cnts = np.unique(pc, return_counts=True)
    ft = json.loads(z.read('data/0/0-stm_fit-global.json'))
    have = {int(c) for c in np.unique(np.asarray(ft['cluster_id'], dtype=int))}
    return dict(zip(cids.tolist(), cnts.tolist())), have


def _work(args):
    """One event, one arm.  Module level so multiprocessing can pickle it."""
    logdir, tag, base = args
    zp = f'{W}/{base}_{tag}/mabc-pr.zip'
    if not os.path.exists(zp):
        return None
    size, have = sizes_and_fits(zp)
    big = {c for c, n in size.items() if n >= MINQ}
    census_all = (len(big), sum(size[c] for c in big),
                  len(big - have), sum(size[c] for c in big - have))
    ev = evb = nfb = 0
    reasons = []
    lg = f'{logdir}/{tag}_{base}.log'
    if logdir and os.path.exists(lg):
        txt = open(lg, errors='replace').read()
        seen = {int(m.group(1)) for m in VERDICT.finditer(txt)}
        ev = len(seen)
        sb = seen & big
        evb = len(sb)
        nfb = len(sb - have)
        for m in NOFIT.finditer(txt):
            cid = int(m.group(1))
            if cid in big:
                reasons.append((re.sub(r'[0-9][0-9.]*', 'N', m.group(2)).strip(), cid))
    return census_all, (ev, evb, nfb), reasons


def run(spec):
    logdir, tag = spec.rsplit(':', 1)
    bases = sorted({os.path.basename(d)[:-(len(tag) + 1)]
                    for d in glob.glob(f'{W}/*_{tag}')})
    tot = [0, 0, 0, 0]
    ev = evb = nfb = 0
    rc = {}
    nev = 0
    with Pool(16) as pool:
        for r in pool.imap_unordered(_work, [(logdir, tag, b) for b in bases]):
            if r is None:
                continue
            nev += 1
            a, b, reasons = r
            for i in range(4):
                tot[i] += a[i]
            ev += b[0]; evb += b[1]; nfb += b[2]
            for why, cid in reasons:
                rc.setdefault(why, set()).add((nev, cid))

    print('=== %s  (%d events) ===' % (tag, nev))
    print('  every cluster >=%d charge points : %5d clusters, %9d charge points'
          % (MINQ, tot[0], tot[1]))
    print('    of them with NO stm_fit        : %5d (%4.1f%%) holding %4.1f%% of that charge'
          % (tot[2], 100.0 * tot[2] / max(tot[0], 1), 100.0 * tot[3] / max(tot[1], 1)))
    if ev:
        print('  clusters TaggerCheckSTM evaluated: %5d (all sizes)' % ev)
        print('    of them >=%d charge points     : %5d, of which NO fit %4d (%4.1f%%)'
              % (MINQ, evb, nfb, 100.0 * nfb / max(evb, 1)))
        print('    pre-fit exit reasons (distinct clusters, >=%d pts):' % MINQ)
        for w, s in sorted(rc.items(), key=lambda x: -len(x[1])):
            print('      %-62s %5d' % (w[:62], len(s)))


for spec in sys.argv[1:]:
    run(spec if ':' in spec else ':' + spec)
