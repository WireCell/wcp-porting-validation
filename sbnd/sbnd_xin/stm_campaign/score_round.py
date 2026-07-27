#!/usr/bin/env python3
"""Score one STM-campaign arm against the doc-62 owner baseline.

Reads nusel-evt<ID>.tsv from the round's work root, takes the stm/label column
for each baseline bundle, and prints:
  - the confusion vs owner truth (fixes / regressions vs the reference arm),
  - verdict flips on NON-adjudicated in-beam bundles in the same events
    (collateral: no truth label, listed for hand judgment).

Usage: score_round.py --round work-stmcamp-r1 [--ref work-stmcamp-r0]
Paths are relative to sbnd_xin/ (or absolute).
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SBND = os.path.normpath(os.path.join(HERE, '..'))

AP = argparse.ArgumentParser()
AP.add_argument('--round', required=True)
AP.add_argument('--ref')
AP.add_argument('--baseline', default=os.path.join(SBND, 'scan-d59k', 'stm-baseline.tsv'))
args = AP.parse_args()


def workpath(p):
    return p if os.path.isabs(p) else os.path.join(SBND, p)


def read_arm(root):
    """{(event, main_id): stm_verdict} for every in-beam bundle; stm in {0,1}."""
    out = {}
    root = workpath(root)
    for d in sorted(os.listdir(root)):
        if not d.startswith('nusel_evt'):
            continue
        evt = d[len('nusel_evt'):]
        tsv = os.path.join(root, d, f'nusel-evt{evt}.tsv')
        if not os.path.exists(tsv):
            print(f'  WARNING missing {tsv}')
            continue
        with open(tsv) as f:
            # space-aligned columns (nusel_extract.py), not tab-separated
            hdr = f.readline().split()
            for line in f:
                r = dict(zip(hdr, line.split()))
                if r.get('in_beam') != '1' or r.get('main_id') in (None, '', '-'):
                    continue
                stm = r.get('stm', '-1')
                out[(r['event'], r['main_id'])] = dict(
                    stm=1 if stm == '1' else 0, label=r.get('label', ''))
    return out


base = []
with open(args.baseline) as f:
    hdr = None
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.rstrip('\n').split('\t')
        if hdr is None:
            hdr = parts
            continue
        base.append(dict(zip(hdr, parts)))

arm = read_arm(args.round)
ref = read_arm(args.ref) if args.ref else None

fixes, regressions, unchanged_err, missing = [], [], [], []
for b in base:
    key = (b['event'], b['main_id'])
    if key not in arm:
        missing.append(key)
        continue
    got = arm[key]['stm']
    want = 1 if b['owner_verdict'] == 'STM' else 0
    was = ref[key]['stm'] if ref and key in ref else (1 if b['tagger'] == 'STM' else 0)
    tag = f"{b['event']}:{b['main_id']}"
    if got == want and was != want:
        fixes.append((tag, b['class']))
    elif got != want and was == want:
        regressions.append((tag, b['class']))
    elif got != want:
        unchanged_err.append((tag, b['class']))

n_ok = len(base) - len(missing) - len(unchanged_err) - len(regressions)
print(f'== {args.round} vs owner baseline ({len(base)} bundles) ==')
print(f'correct: {n_ok}  still-wrong: {len(unchanged_err)}  '
      f'FIXED: {len(fixes)}  REGRESSED: {len(regressions)}  missing: {len(missing)}')
for tag, cls in fixes:
    print(f'  FIX  {tag}  ({cls})')
for tag, cls in regressions:
    print(f'  REG  {tag}  ({cls})')
if unchanged_err:
    print('still wrong: ' + ' '.join(t for t, _ in unchanged_err))
for key in missing:
    print(f'  MISSING {key[0]}:{key[1]}')

if ref:
    basekeys = {(b['event'], b['main_id']) for b in base}
    flips = [(k, ref[k]['stm'], v['stm']) for k, v in arm.items()
             if k in ref and v['stm'] != ref[k]['stm'] and k not in basekeys]
    if flips:
        print(f'collateral flips on non-adjudicated bundles ({len(flips)}):')
        for (evt, mid), was, now in flips:
            print(f'  {evt}:{mid}  stm {was} -> {now}')
    else:
        print('no collateral flips on non-adjudicated bundles.')
