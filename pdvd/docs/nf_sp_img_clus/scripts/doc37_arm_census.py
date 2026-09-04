#!/usr/bin/env python3
"""doc pdvd/37 R1 -- per-event PR-level census of one PDVD arm.

Answers the grading row doc 37 sec.6 deferred ("does not cost physics"), which
could not be measured before the knob existed.  For every event of a tag it
reads the PrDisplayDump (calib-pr-evt*.json) and the job log and reports:

  nterm / nsteiner_pts                what the DUMP carries, i.e. what every
                                      downstream consumer sees
  thin_in -> thin_out                 the knob's own log line (ON arms only)
  segments, segment length, vertices  the PR skeleton the terminals feed
  showers, main-vertex presence
  TGM / STM / FC tag counts           the cosmic verdicts downstream

TWO DENOMINATORS, DO NOT CONFLATE THEM.  thin_out/thin_in is the knob's own
effect, measured at phase 3b -- BEFORE phase 4 adds get_extreme_points_for_
reference.  nterm is what the dump carries, i.e. AFTER those extremes and after
every cluster that never reached the thinning at all.  nterm(ON)/nterm(OFF) is
therefore always the milder number, and it is the one production sees.  Quote
both; they are not the same fraction and they should not reconcile.

READ THE TAG COUNTS AS A CHAIN, NOT AS THREE INDEPENDENT NUMBERS.  The STM
tagger skips a cluster TGM already claimed ("cluster N already TGM; skipping"),
so nstm is CONDITIONED on ntgm.  If ntgm moves, nstm moves for a reason that is
not the thinning.  nclus_tgm_eval is the common denominator.

Usage: doc37_arm_census.py <tag> [<tag> ...]      (writes a TSV to stdout)
"""
import glob, json, os, re, sys

PDVD = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

RE_TGM  = re.compile(r'TaggerCheckTGM: cluster (\d+) . TGM=(true|false)')
RE_STM  = re.compile(r'TaggerCheckSTM: cluster (\d+) . STM=(\d) TGM=(\d)')
RE_FC   = re.compile(r'TaggerCheckFC: cluster (\d+) . FC=(true|false)')
RE_THIN = re.compile(r'steiner_thin: nterm_in=(\d+) nterm_out=(\d+) min_sep_cm=([\d.]+)')


def census_event(d):
    """One work dir -> a dict of counts, or None if the dump is missing."""
    dumps = glob.glob(os.path.join(d, 'calib-pr-evt*.json'))
    if not dumps:
        return None
    with open(dumps[0]) as f:
        j = json.load(f)

    st = j.get('steiner') or []
    npts = sum(len(c.get('x', [])) for c in st)
    nterm = sum(sum(c.get('flag_terminal', [])) for c in st)

    segs = j.get('segments') or []
    verts = j.get('vertices') or []
    shows = j.get('showers') or []
    seglen = sum(s.get('length', 0.0) for s in segs)

    row = dict(nsteiner_clusters=len(st), nsteiner_pts=npts, nterm=nterm,
               nseg=len(segs), seglen=seglen, nvtx=len(verts), nshower=len(shows),
               has_main_vertex=int(bool(j.get('main_vertex'))),
               thin_in=0, thin_out=0, sep_cm=0.0,
               ntgm=0, nstm=0, nfc=0, nclus_tgm_eval=0)

    logs = glob.glob(os.path.join(d, 'wct_pr_*.log'))
    if logs:
        with open(logs[0], errors='replace') as f:
            for line in f:
                m = RE_THIN.search(line)
                if m:
                    row['thin_in'] += int(m.group(1))
                    row['thin_out'] += int(m.group(2))
                    row['sep_cm'] = float(m.group(3))
                    continue
                m = RE_TGM.search(line)
                if m:
                    row['nclus_tgm_eval'] += 1
                    row['ntgm'] += (m.group(2) == 'true')
                    continue
                m = RE_STM.search(line)
                if m:
                    row['nstm'] += (m.group(2) == '1')
                    continue
                m = RE_FC.search(line)
                if m:
                    row['nfc'] += (m.group(2) == 'true')
    return row


COLS = ['nsteiner_clusters', 'nsteiner_pts', 'nterm', 'thin_in', 'thin_out', 'sep_cm',
        'nseg', 'seglen', 'nvtx', 'nshower', 'has_main_vertex',
        'nclus_tgm_eval', 'ntgm', 'nstm', 'nfc']


def main(tags):
    print('\t'.join(['tag', 'run', 'evt'] + COLS))
    for tag in tags:
        for d in sorted(glob.glob(os.path.join(PDVD, 'work', '*_' + tag))):
            base = os.path.basename(d)[:-(len(tag) + 1)]
            run, _, evt = base.rpartition('_')
            row = census_event(d)
            if row is None:
                print('\t'.join([tag, run, evt] + ['NA'] * len(COLS)))
                continue
            print('\t'.join([tag, run, evt] +
                            [('%.3f' % row[c]) if isinstance(row[c], float) else str(row[c])
                             for c in COLS]))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
