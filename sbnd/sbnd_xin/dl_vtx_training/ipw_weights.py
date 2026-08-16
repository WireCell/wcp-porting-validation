#!/usr/bin/env python3
'''
doc pr/89 sec 1.2 -- inverse-propensity weights over the mcp2k scan tiers.

The pr/88 label pool is tier-selected (every fill tier selected on `not
agrees`), so a raw average over labelled events does not estimate production
accuracy on the arm.  The scan run record carries the tier of all 845 scanned
events, so each labelled event can be weighted by
N_arm_stratum / n_labelled_stratum, where the five strata are the
(bucket, agrees) cross-tab of waves/*/review.json.

TWO ESTIMANDS -- DO NOT MIX THEM (doc pr/89 sec 1.2):
  - "on all 999": the raw, unweighted label-pool count, comparable to every
    pr/78-82 number.  No weights.
  - "IPW mcp2k-arm": the unbiased production estimate on the 845-event mcp2k
    arm, over mcp2k-tag labels only, with these weights.  A stratum with
    ZERO labelled events (REVIEW-agree until the pr/89 ragree scan lands)
    makes the estimate a BOUND, not a point -- this script says so loudly.

Usage:
  python3 ipw_weights.py --runs ../vtx_rules/runs/mcp2k-20260816 \
      --tags vtxscan-mcp2k vtxscan-mcp2k-auto vtxscan-mcp2k-ragree \
      --tsv runs/ipw-mcp2k-<date>.tsv
'''
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scn_vtx import io as vio                                    # noqa: E402


def stratum_of(row):
    b = row['bucket']
    if b == 'REVIEW':
        return 'REVIEW-agree' if row['agrees'] else 'REVIEW-disagree'
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True,
                    help='scan run record with waves/*/review.json')
    ap.add_argument('--tags', nargs='+',
                    default=['vtxscan-mcp2k', 'vtxscan-mcp2k-auto'],
                    help='label tags that count as "labelled" (add '
                         'vtxscan-mcp2k-ragree once the sec 6 scan lands)')
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--tsv', required=True)
    args = ap.parse_args()

    strat = {}
    for w in sorted(glob.glob(os.path.join(args.runs, 'waves', 'wave*',
                                           'review.json'))):
        for r in json.load(open(w))['rows']:
            strat[int(r['event'][3:])] = stratum_of(r)
    if not strat:
        raise SystemExit('no waves/*/review.json under %s' % args.runs)

    labelled = {}
    for tag in args.tags:
        d = os.path.join(args.sbnd_root, 'vertex_labels', tag)
        if not os.path.isdir(d):
            print('NOTE: tag %s has no label dir yet -- skipped' % tag)
            continue
        for label in vio.iter_labels(args.sbnd_root, [tag]):
            labelled[label['eventNo']] = tag

    n_arm, n_lab = {}, {}
    for evt, s in strat.items():
        n_arm[s] = n_arm.get(s, 0) + 1
        if evt in labelled:
            n_lab[s] = n_lab.get(s, 0) + 1
    off_arm = sorted(e for e in labelled if e not in strat)
    if off_arm:
        print('WARNING: %d labelled events not in the scan record (weight '
              'undefined, omitted): %s' % (len(off_arm), off_arm[:5]))

    print('%-38s %5s %9s %8s' % ('stratum', 'N_arm', 'labelled', 'weight'))
    unmeasured = []
    for s in sorted(n_arm):
        nl = n_lab.get(s, 0)
        w = (n_arm[s] / nl) if nl else float('nan')
        print('%-38s %5d %9d %8s'
              % (s, n_arm[s], nl, ('%.3f' % w) if nl else 'UNDEF'))
        if not nl:
            unmeasured.append(s)
    if unmeasured:
        print('\nUNMEASURED STRATA (%s): the IPW estimate over this file is '
              'a BOUND, not a point -- %d of %d arm events carry weight '
              'nothing in the pool can claim.'
              % (', '.join(unmeasured),
                 sum(n_arm[s] for s in unmeasured), sum(n_arm.values())))

    with open(args.tsv, 'w') as fh:
        fh.write('evt\tweight\tstratum\ttag\n')
        for evt in sorted(labelled):
            if evt not in strat:
                continue
            s = strat[evt]
            fh.write('%d\t%.6f\t%s\t%s\n'
                     % (evt, n_arm[s] / n_lab[s], s, labelled[evt]))
    print('wrote %s (%d weighted events)' % (args.tsv,
                                             len([e for e in labelled
                                                  if e in strat])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
