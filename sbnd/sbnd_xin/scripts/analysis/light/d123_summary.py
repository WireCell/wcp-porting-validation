#!/usr/bin/env python3
"""doc 123 -- per rescue move, hit-level onset test on the TPC(s) that lack a reco1 flash.
prompt = hit PE in [t-0.05, t+0.15] us; pre = hit PE in [t-0.35, t-0.15] us (same TPC, same width);
an onset is prompt >= 100 PE and prompt >= 3*pre (a new pulse, not the tail of an earlier flash)."""
import csv, sys, collections
H = collections.defaultdict(list)
for fn in sys.argv[2:]:
    for line in open(fn):
        r = line.split('\t')
        if r[0] == 'H': H[int(r[1])].append((int(r[3]), float(r[4]), float(r[5])))
def pe(ev, side, lo, hi): return sum(p for ch, t, p in H[ev] if ch % 2 == side and lo <= t <= hi)
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
out = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
out.writerow(['event', 'sample', 'entry', 'run', 'pass', 'rule', 'adopted', 'class', 'dt0_us',
              'A_side', 'A_t_raw_us', 'A_prompt_pe', 'A_pre_pe', 'A_onset',
              'B_side', 'B_t_raw_us', 'B_prompt_pe', 'B_pre_pe', 'B_onset', 'hit_level_split_expected'])
cnt = collections.Counter()
for r in rows:
    ev = int(r['event'])
    if r['near_reco1_t_raw_us'] in ('', 'NOTFOUND') or r['dt0_us'] == '':
        tn = float(r['near_reco1_t_raw_us']) if r['near_reco1_t_raw_us'] not in ('', 'NOTFOUND') else None
        # unmatched pass: only side A (far TPC at the near/beam time) is testable
        if tn is None: continue
        a_s = int(r['far_tpc']); a = pe(ev, a_s, tn - .05, tn + .15); ap = pe(ev, a_s, tn - .35, tn - .15)
        ao = a >= 100 and a >= 3 * ap
        out.writerow([ev, r['sample'], r['entry'], r['run'], r['pass'], r['rule'], '', r['class'], '',
                      a_s, f'{tn:.4f}', f'{a:.0f}', f'{ap:.0f}', int(ao), '', '', '', '', '', 'yes' if ao else 'no'])
        cnt[(r['class'], 'yes' if ao else 'no')] += 1
        continue
    tn = float(r['near_reco1_t_raw_us']); dt = float(r['dt0_us']); tf = tn + dt
    a_s = int(r['far_tpc']); b_s = int(r['near_tpc'])
    a = pe(ev, a_s, tn - .05, tn + .15); ap = pe(ev, a_s, tn - .35, tn - .15)
    b = pe(ev, b_s, tf - .05, tf + .15); bp = pe(ev, b_s, tf - .35, tf - .15)
    ao = a >= 100 and a >= 3 * ap; bo = b >= 100 and b >= 3 * bp
    cls = r['class']
    if cls.startswith('same-time'): exp = 'n/a (both flashes exist)'
    elif cls.startswith('far-flash') or cls.startswith('8-13'):
        exp = 'n/a (outside the 8 us veto)'
    else: exp = 'yes' if (ao or bo) else 'no'
    cnt[(cls, exp)] += 1
    out.writerow([ev, r['sample'], r['entry'], r['run'], r['pass'], r['rule'], r['adopted'], cls, r['dt0_us'],
                  a_s, f'{tn:.4f}', f'{a:.0f}', f'{ap:.0f}', int(ao), b_s, f'{tf:.4f}', f'{b:.0f}', f'{bp:.0f}', int(bo), exp])
for k, v in sorted(cnt.items()): print('#', k, v, file=sys.stderr)
