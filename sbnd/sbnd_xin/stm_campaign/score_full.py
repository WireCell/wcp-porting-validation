#!/usr/bin/env python3
"""Full-population regression check for one STM-campaign arm (doc 63).

Compares the arm's per-bundle STM verdict against the STORED d59k production
for EVERY in-beam bundle (the 600+ cases the owner filtered the 72-bundle
baseline from), and classifies each flip:

  - baseline bundle  -> expected fix (or regression) per stm-baseline.tsv;
  - non-adjudicated  -> a potential regression: tagger and the doc-61 AI scan
    agreed there and the owner confirmed by silence, so every flip is listed
    with the AI-scan verdict for individual judgment.

Usage: score_full.py --round work-stmcamp-r2full [--ref work-mcp1kall-d59k]
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SBND = os.path.normpath(os.path.join(HERE, '..'))

AP = argparse.ArgumentParser()
AP.add_argument('--round', required=True)
AP.add_argument('--ref', default='work-mcp1kall-d59k')
AP.add_argument('--baseline', default=os.path.join(SBND, 'scan-d59k', 'stm-baseline.tsv'))
args = AP.parse_args()


def workpath(p):
    return p if os.path.isabs(p) else os.path.join(SBND, p)


def read_arm(root):
    out = {}
    root = workpath(root)
    for d in sorted(os.listdir(root)):
        if not d.startswith('nusel_evt'):
            continue
        evt = d[len('nusel_evt'):]
        tsv = os.path.join(root, d, f'nusel-evt{evt}.tsv')
        if not os.path.exists(tsv):
            continue
        with open(tsv) as f:
            hdr = f.readline().split()
            for line in f:
                r = dict(zip(hdr, line.split()))
                if r.get('in_beam') != '1' or r.get('main_id') in (None, '', '-'):
                    continue
                out[(r['event'], r['main_id'])] = dict(
                    stm=1 if r.get('stm') == '1' else 0,
                    label=r.get('label', ''), t=r.get('flash_time_us', ''),
                    length=r.get('len_main_cm', ''))
    return out


def read_ai():
    """AI-scan verdicts keyed (event, main_id) from both hand-scan TSVs."""
    out = {}
    for name in ('handscan-first20.tsv', 'handscan-batch2.tsv'):
        p = os.path.join(SBND, 'scan-d59k', name)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            hdr = None
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.rstrip('\n').split('\t')
                if hdr is None:
                    hdr = parts
                    continue
                r = dict(zip(hdr, parts))
                if 'event' in r and 'main_id' in r:
                    out[(r['event'], r['main_id'])] = r.get('verdict', '?')
    return out


base = {}
with open(args.baseline) as f:
    hdr = None
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.rstrip('\n').split('\t')
        if hdr is None:
            hdr = parts
            continue
        r = dict(zip(hdr, parts))
        base[(r['event'], r['main_id'])] = r

arm = read_arm(args.round)
ref = read_arm(args.ref)
ai = read_ai()

common = sorted(set(arm) & set(ref), key=lambda k: (int(k[0]), int(k[1])))
missing_in_arm = sorted(set(ref) - set(arm), key=lambda k: (int(k[0]), int(k[1])))
flips = [k for k in common if arm[k]['stm'] != ref[k]['stm']]

fix = reg = other = 0
print(f'== {args.round} vs {args.ref}: {len(common)} in-beam bundles compared, '
      f'{len(flips)} verdict flips ==')
for k in flips:
    evt, mid = k
    was, now = ref[k]['stm'], arm[k]['stm']
    b = base.get(k)
    if b:
        want = 1 if b['owner_verdict'] == 'STM' else 0
        status = 'FIX' if now == want else 'REGRESSION'
        if now == want:
            fix += 1
        else:
            reg += 1
        print(f'  {status:<10} {evt}:{mid}  stm {was}->{now}  owner={b["owner_verdict"]}  ({b["class"]})')
    else:
        other += 1
        print(f'  UNLABELED  {evt}:{mid}  stm {was}->{now}  ai-scan={ai.get(k, "none")}  '
              f't={arm[k]["t"]}us len={arm[k]["length"]}cm')
print(f'summary: {fix} baseline fixes, {reg} baseline regressions, '
      f'{other} unlabeled flips (review each), '
      f'{len(missing_in_arm)} ref bundles missing from the arm')
for k in missing_in_arm[:20]:
    print(f'  MISSING-FROM-ARM {k[0]}:{k[1]}')
