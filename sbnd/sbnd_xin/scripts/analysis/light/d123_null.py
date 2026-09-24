#!/usr/bin/env python3
"""doc 123 -- null floor for the sec 6 onset test, and the PMT channel -> TPC parity check.
Null: for every within-8us-veto move, re-run the onset test on the SAME TPC at t_miss + delta,
delta in {-1.5,-0.8,+0.8,+1.5} us, skipping any shifted time within 0.3 us of a reco1 flash on
that side (so a real flash cannot count as a null hit).  Sides A and B as in d123_summary.py."""
import csv, sys, json, collections
chmap = json.load(open(sys.argv[1]))
rows = list(csv.DictReader(open(sys.argv[2]), delimiter='\t'))
H = collections.defaultdict(list); F = collections.defaultdict(list); chans = collections.Counter()
for fn in sys.argv[3:]:
    for line in open(fn):
        r = line.split('\t')
        if r[0] == 'H':
            H[int(r[1])].append((int(r[3]), float(r[4]), float(r[5]))); chans[int(r[3])] += 1
        elif r[0] == 'F': F[int(r[1])].append((int(r[3]), float(r[4])))
t0s, t1s = set(chmap['tpc0']), set(chmap['tpc1'])
bad = [c for c in chans if not ((c in t0s and c % 2 == 0) or (c in t1s and c % 2 == 1))]
print(f'# parity: {len(chans)} distinct hit channels, {sum(chans.values())} hits; '
      f'{len([c for c in chans if c in t0s])} in tpc0 list (all even), {len([c for c in chans if c in t1s])} in tpc1 list (all odd); '
      f'violations {len(bad)} {sorted(bad)[:10]}')
def pe(ev, side, lo, hi): return sum(p for ch, t, p in H[ev] if ch % 2 == side and lo <= t <= hi)
def onset(ev, side, t):
    a = pe(ev, side, t - .05, t + .15); b = pe(ev, side, t - .35, t - .15)
    return a >= 100 and a >= 3 * b
out = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
out.writerow(['event', 'side_label', 'tpc', 't_miss_raw_us', 'delta_us', 'null_onset'])
n = k = 0; per_move_real = per_move_null_any = 0; moves = 0
for r in rows:
    if r['class'] != 'within-8us-veto': continue
    ev = int(r['event']); moves += 1
    real = int(r['A_onset']) or int(r['B_onset'])
    per_move_real += real
    anyn = False
    for lab, s, t in (('A', int(r['A_side']), float(r['A_t_raw_us'])), ('B', int(r['B_side']), float(r['B_t_raw_us']))):
        for d in (-1.5, -0.8, 0.8, 1.5):
            tt = t + d
            if any(side == s and abs(tf - tt) < 0.3 for side, tf in F[ev]): continue
            o = onset(ev, s, tt); n += 1; k += o; anyn |= o
            out.writerow([ev, lab, s, f'{t:.4f}', d, int(o)])
    per_move_null_any += anyn
print(f'# within-veto moves {moves}: real onset (A or B) {per_move_real}/{moves}; '
      f'null probes {k}/{n} = {k/max(n,1):.3f} per probe; moves with >=1 null onset over their <=8 probes {per_move_null_any}/{moves}', file=sys.stderr)
