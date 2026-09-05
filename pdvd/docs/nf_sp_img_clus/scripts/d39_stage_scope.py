#!/usr/bin/env python3
"""doc pdvd/39 -- the fitting config governs TWO stages, so measure both.

pdvd_track_fitting.json (where good_point_pitch_frac and end_trim_gap_len live)
is handed to BOTH taggers in the PDVD PR job:

    cfg/pgrapher/experiment/protodunevd/pr.jsonnet:1341  tagger_check_stm
    cfg/pgrapher/experiment/protodunevd/pr.jsonnet:1562  tagger_check_neutrino

and ctpc_aniso_metric is a Grouping flag, so it changes every ctpc query in the
job.  Docs 36 and 38 graded only the STM tagger's fit.  This script counts both
output layers on the same events so the neutrino side stops being invisible:

    data/0/0-stm_fit-global.json    TaggerCheckSTM::persist_stm_fit  (TaggerCheckSTM.cxx:614)
    data/0/0-track_fit-global.json  TaggerCheckNeutrino, grouping slot "nu<idx>"
                                    (TaggerCheckNeutrino.cxx:3577)

Only events present in EVERY named arm are counted, so the totals are like for
like: an arm that produced one extra event would otherwise look like a physics
effect (feedback_check_numbers_in_both_directions).

Usage:
  d39_stage_scope.py <tag> [<tag> ...]      # first tag is the baseline
"""
import glob, json, os, sys, zipfile
import numpy as np
from multiprocessing import Pool

PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
W = os.path.join(PDVD, 'work')
MINQ = 50
TAGS = sys.argv[1:]


def _work(base):
    out = {}
    for t in TAGS:
        p = f'{W}/{base}_{t}/mabc-pr.zip'
        if not os.path.exists(p):
            return None                      # drop the event from every arm
        z = zipfile.ZipFile(p)
        cl = json.loads(z.read('data/0/0-clustering-global.json'))
        cids, cnts = np.unique(np.asarray(cl['cluster_id'], dtype=int), return_counts=True)
        size = dict(zip(cids.tolist(), cnts.tolist()))
        ft = json.loads(z.read('data/0/0-stm_fit-global.json'))
        have = {int(c) for c in np.unique(np.asarray(ft['cluster_id'], dtype=int))}
        try:
            tf = json.loads(z.read('data/0/0-track_fit-global.json'))
            nu = (len(tf['x']), len(np.unique(np.asarray(tf['cluster_id'], dtype=int))))
        except KeyError:
            nu = (0, 0)
        out[t] = (len({c for c in have if size.get(c, 0) >= MINQ}), len(ft['x']),
                  nu[0], nu[1])
    return out


bases = sorted({os.path.basename(d)[:-(len(TAGS[0]) + 1)]
                for d in glob.glob(f'{W}/*_{TAGS[0]}')})
agg = {t: [0, 0, 0, 0] for t in TAGS}
nev = 0
with Pool(16) as pool:
    for r in pool.imap_unordered(_work, bases):
        if r is None:
            continue
        nev += 1
        for t in TAGS:
            for i in range(4):
                agg[t][i] += r[t][i]

print('%d events present in every arm\n' % nev)
print('%-9s | %-32s | %-25s' % ('', 'STM tagger  (stm_fit)', 'neutrino PR  (track_fit)'))
print('%-9s | %14s %17s | %11s %13s' % (
    'arm', 'clusters>=%dpt' % MINQ, 'fit points', 'clusters', 'fit points'))
b = agg[TAGS[0]]
for t in TAGS:
    a = agg[t]
    d = '' if t == TAGS[0] else '   (%+d / %+.1f%%   %+d / %+.1f%%)' % (
        a[0] - b[0], 100.0 * (a[1] - b[1]) / max(b[1], 1),
        a[3] - b[3], 100.0 * (a[2] - b[2]) / max(b[2], 1))
    print('%-9s | %14d %17d | %11d %13d%s' % (t, a[0], a[1], a[3], a[2], d))
print('\n(deltas are vs the baseline arm %s: STM clusters / STM points, '
      'nu clusters / nu points)' % TAGS[0])

# A delta is only attributable to ONE change when both arms come from the same
# config epoch.  d36* ran before the doc-37 Steiner terminal thinning landed and
# d38* after it, so any d36->d38 delta above bundles the thinning in with the
# knob under test (feedback_check_the_cfg_epoch_between_arms).  Print the
# adjacent pairs and let the caller name which ones are single-change.
print('\nadjacent-pair deltas -- ONLY same-epoch pairs are attributable to one change:')
for i in range(1, len(TAGS)):
    a, c = agg[TAGS[i - 1]], agg[TAGS[i]]
    print('  %-9s -> %-9s  STM %+5d cl / %+6.1f%% pts    nu %+5d cl / %+6.1f%% pts'
          % (TAGS[i - 1], TAGS[i], c[0] - a[0], 100.0 * (c[1] - a[1]) / max(a[1], 1),
             c[3] - a[3], 100.0 * (c[2] - a[2]) / max(a[2], 1)))
