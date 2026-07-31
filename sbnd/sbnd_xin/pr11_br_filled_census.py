#!/usr/bin/env python3
"""doc pr/11 section 2: bucket nue_score properly.

nue_score = -15 is NOT a low score -- it is UbooneNueBDTScorer.cxx:1925's
`default_val`, written whenever ti.br_filled != 1, i.e. the nue tagger never
filled its BDT variable block so the BDT never ran.  Lumping it into a
percentile table with real scores makes the median read as "very
background-like" when it actually means "not evaluated".

br_filled IS a T_tagger branch (UbooneTaggerOutputVisitor.cxx:889), so this
counts it directly rather than inferring it from nue_score == -15.

Buckets, per sample:
  no_tagger    : no T_tagger row at all (event never reached the nue tagger)
  not_filled   : br_filled != 1  -> nue_score is the -15 sentinel
  bdt_clamp_neg: br_filled == 1 and nue_score <= -4.30 (true log-odds clamp)
  bdt_clamp_pos: br_filled == 1 and nue_score >=  4.30
  bdt_interior : br_filled == 1 and |nue_score| < 4.30 (unsaturated BDT output)
"""
import glob
import os
import sys
from collections import Counter, defaultdict

import uproot

TAGS = [
    ('mcp1k', 'work-mcp1kall-pr11v3'),
    ('nuecc48', 'work-nuecc48-pr11v3'),
    ('r1qlmc', 'work-r1qlmc-pr11v3'),
    ('r2mc', 'work-r2mc-pr11v3'),
]
BASE = sys.argv[1] if len(sys.argv) > 1 else '.'

buckets = defaultdict(Counter)
interior = defaultdict(list)

for sample, tag in TAGS:
    for d in sorted(glob.glob(os.path.join(BASE, tag, 'pr_evt*'))):
        rf = os.path.join(d, 'tracking-pr.root')
        if not os.path.exists(rf):
            buckets[sample]['no_root'] += 1
            continue
        try:
            with uproot.open(rf) as f:
                if 'T_tagger' not in f:
                    buckets[sample]['no_tagger'] += 1
                    continue
                t = f['T_tagger']
                if t.num_entries == 0:
                    buckets[sample]['no_tagger'] += 1
                    continue
                bf = t['br_filled'].array(library='np')
                ns = t['nue_score'].array(library='np')
        except Exception as exc:  # noqa: BLE001
            buckets[sample][f'read_error'] += 1
            print(f"  ! {d}: {exc}", file=sys.stderr)
            continue
        for b, s in zip(bf, ns):
            if float(b) != 1.0:
                buckets[sample]['not_filled'] += 1
            elif s <= -4.30:
                buckets[sample]['bdt_clamp_neg'] += 1
            elif s >= 4.30:
                buckets[sample]['bdt_clamp_pos'] += 1
            else:
                buckets[sample]['bdt_interior'] += 1
                interior[sample].append(float(s))

KEYS = ['no_root', 'no_tagger', 'not_filled', 'bdt_clamp_neg',
        'bdt_clamp_pos', 'bdt_interior', 'read_error']
print(f"{'sample':<10} " + ' '.join(f"{k:>14}" for k in KEYS) + f" {'total':>7}")
tot = Counter()
for sample, _ in TAGS:
    c = buckets[sample]
    tot.update(c)
    n = sum(c.values())
    print(f"{sample:<10} " + ' '.join(f"{c.get(k,0):>14}" for k in KEYS) +
          f" {n:>7}")
print(f"{'ALL':<10} " + ' '.join(f"{tot.get(k,0):>14}" for k in KEYS) +
      f" {sum(tot.values()):>7}")

print("\nREAL nue BDT evaluations (br_filled==1):")
for sample, _ in TAGS:
    c = buckets[sample]
    real = c.get('bdt_clamp_neg', 0) + c.get('bdt_clamp_pos', 0) + \
        c.get('bdt_interior', 0)
    n = sum(c.values())
    print(f"  {sample:<10} {real:>5}/{n:<5} ({100.0*real/max(1,n):.1f}%)")

print("\nUnsaturated nue_score values (br_filled==1, |s|<4.30):")
for sample, _ in TAGS:
    v = sorted(interior[sample])
    if not v:
        print(f"  {sample:<10} none")
    else:
        print(f"  {sample:<10} n={len(v)} min={min(v):.3f} max={max(v):.3f} "
              f"median={v[len(v)//2]:.3f}")
