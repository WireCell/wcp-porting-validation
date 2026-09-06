#!/usr/bin/env python3
"""doc pdhd/04 sec 11 -- score the mover hand scan against the PRE-REGISTERED bar.

The bar is fixed in doc pdhd/04 sec 11.5 and is NOT to be re-tuned after the
labels are seen:

  Primary, over TGM-gained objects with npts >= 1000:
    PASS if  (THRU + FRAG>THRU) >= 80 % of the JUDGEABLE ones (UNCLEAR excluded)
       and  (STOP + FRAG>STOP) <= 10 % of the same denominator.

Everything else -- the STM gains, the unexplained STM removals, the hard and
unjudgeable bands -- is REPORTED, never gated: only 4 of 8 STM gains and 2 of 5
removals clear 1000 points, so a threshold on them would be unsatisfiable by
construction (doc pdhd/stm-tagger-chain sec 13).

Usage:
  d04_movers_score.py --sheet <...sheet.tsv> --key <...KEY.tsv>
"""
import argparse, csv, collections

THRU = {'THRU', 'FRAG>THRU'}
STOP = {'STOP', 'FRAG>STOP'}
CONT = {'CONT', 'FRAG>CONT'}
VALID = THRU | STOP | CONT | {'MESSY', 'UNCLEAR'}
UNJUDGEABLE = {'UNCLEAR'}

def rows(path):
    with open(path) as fh:
        return list(csv.DictReader((l for l in fh if not l.startswith('#')), delimiter='\t'))

def band(n):
    return '>=1000' if n >= 1000 else '200-1000' if n >= 200 else '<200'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet', required=True)
    ap.add_argument('--key', required=True)
    a = ap.parse_args()

    key = {(int(r['event']), int(r['cluster'])): (int(r['npts']), r['directions'].split('+'))
           for r in rows(a.key)}
    lab = {}
    bad = []
    for r in rows(a.sheet):
        k = (int(r['event']), int(r['cluster']))
        v = (r.get('label') or '').strip().upper().replace('FRAG->', 'FRAG>')
        if not v:
            continue
        if v not in VALID:
            bad.append((k, v))
            continue
        lab[k] = v
    if bad:
        # An unrecognised label must never fall through into a class silently
        # (feedback_fragment_label_carries_object_verdict).
        print("REFUSING: unrecognised labels (fix the sheet or extend VALID):")
        for k, v in bad:
            print(f"   evt {k[0]} cluster {k[1]}: {v!r}")
        raise SystemExit(2)

    print(f"labelled {len(lab)} of {len(key)} objects\n")
    census = collections.defaultdict(collections.Counter)
    for k, v in lab.items():
        n, dirs = key[k]
        for d in dirs:
            census[(d, band(n))][v] += 1
    for kk in sorted(census):
        c = census[kk]
        print(f"{kk[0]:18s} {kk[1]:9s} n={sum(c.values()):3d}  " +
              "  ".join(f"{l}={c[l]}" for l in sorted(c)))

    prim = {k: v for k, v in lab.items()
            if 'TGM_gained' in key[k][1] and key[k][0] >= 1000}
    judge = {k: v for k, v in prim.items() if v not in UNJUDGEABLE}
    n = len(judge)
    print(f"\n--- PRIMARY BAR: TGM-gained, npts >= 1000 ---")
    print(f"  labelled {len(prim)}, judgeable {n} (UNCLEAR {len(prim)-n})")
    if not n:
        print("  UNDECIDED: no judgeable object in the primary stratum.")
        return
    t = sum(1 for v in judge.values() if v in THRU)
    st = sum(1 for v in judge.values() if v in STOP)
    print(f"  through-going  {t}/{n} = {t/n:.3f}   (bar: >= 0.80)")
    print(f"  stopping       {st}/{n} = {st/n:.3f}   (bar: <= 0.10)")
    ok = (t / n >= 0.80) and (st / n <= 0.10)
    print(f"  => {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  the objects to read individually:")
        for k, v in sorted(judge.items()):
            if v not in THRU:
                print(f"     evt {k[0]} cluster {k[1]}  npts {key[k][0]}  label {v}")

if __name__ == '__main__':
    main()
