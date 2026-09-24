#!/usr/bin/env python3
"""doc 123 -- rescue-fired events (cbr3 census, all rescue knobs ON, mcp1k+mcp2k 3000 evts)
vs reco1 OpHits/OpFlashes.  For each rescue move: locate the near (beam-side) and far
reco1 flashes by the rescue's dt0 (offset-invariant), then ask whether the far-side TPC
has prompt light at the near flash time that reco1 did not turn into a flash there.
Inputs: probe TSVs from d123_ophit_probe.C, census-firing-detail.txt.
Output: TSV on stdout."""
import re, sys, collections
probe = sys.argv[1:-1]; detail = sys.argv[-1]
N = {}; H = collections.defaultdict(list); F = collections.defaultdict(list)
for fn in probe:
    tag = fn.split('/')[-1].replace('.tsv', '')
    for line in open(fn):
        r = line.rstrip('\n').split('\t')
        ev = int(r[1])
        if r[0] == 'N': N[ev] = (tag, int(r[2]), int(r[3]), int(r[4]), int(r[5]))
        elif r[0] == 'H': H[ev].append((int(r[3]), float(r[4]), float(r[5]), float(r[6])))
        elif r[0] == 'F': F[ev].append((int(r[3]), float(r[4]), float(r[5])))
pat = re.compile(r'(unmatched rescue|rescue) round (\d+): c\d+ \(gid (\d+), t0 ([-\d.]+) us, ([\d.]+) cm\) \+ c\d+ \((?:gid (\d+), t0 ([-\d.]+) us|unmatched), ([\d.]+) cm[^)]*\) -> gid (\d+)(?: \(([\w-]+))?')
def tpc(gid): return 1 if gid >= 1000000 else 0
def prompt_pe(ev, side, t, lo=-0.05, hi=0.15):
    return sum(pe for ch, th, pe, w in H[ev] if ch % 2 == side and t + lo <= th <= t + hi)
def nearest(ev, side, t):
    c = [(abs(tf - t), tf, pe) for s, tf, pe in F[ev] if s == side]
    return min(c) if c else (None, None, None)
print('\t'.join(['event', 'sample', 'entry', 'run', 'subrun', 'pass', 'rule', 'near_tpc', 'near_t0_us', 'far_tpc', 'far_t0_us',
                 'dt0_us', 'class', 'near_reco1_t_raw_us', 'near_reco1_pe', 'farTPC_prompt_pe_at_near_t', 'nearTPC_prompt_pe',
                 'farTPC_reco1_nearest_dt_us', 'farTPC_reco1_nearest_pe', 'adopted', 'nearTPC_prompt_pe_at_far_t', 'nearTPC_reco1_nearest_dt_to_far_t_us', 'nhits']))
for line in open(detail):
    ev = int(line.split(':')[0])
    for m in pat.finditer(line):
        kind, rnd, g1, t1, L1, g2, t2, L2, gout, rule = m.groups()
        g1 = int(g1); t1 = float(t1)
        far_t = float(t2) if t2 else None
        near_side = tpc(g1)
        far_side = tpc(int(g2)) if g2 else 1 - near_side
        dt0 = (far_t - t1) if far_t is not None else None
        # locate near reco1 flash (raw time) : flash on near side whose partner on far side sits at +dt0
        best = None
        for s, tf, pe in F[ev]:
            if s != near_side: continue
            if dt0 is None:
                cand = (0, tf, pe)
            else:
                d = min([abs(tf2 - tf - dt0) for s2, tf2, pe2 in F[ev] if s2 == far_side] or [9e9])
                cand = (d, tf, pe)
            if best is None or cand[0] < best[0]: best = cand
        if dt0 is None:
            # unmatched pass: take the near-side flash inside the raw in-time region (-2..+1 us) with max PE
            c = [(-pe, tf, pe) for s, tf, pe in F[ev] if s == near_side and -2.0 < tf < 1.5]
            best = (0, c and min(c)[1], c and min(c)[2]) if c else (9e9, None, None)
        ok = best is not None and best[0] < 0.03 and best[1] is not None
        tn = best[1] if ok else None
        if dt0 is None: cls = 'far-unmatched'
        elif abs(dt0) < 0.05: cls = 'same-time(no-light-loss)'
        elif abs(dt0) <= 8.0: cls = 'within-8us-veto'
        elif abs(dt0) <= 13.0: cls = '8-13us'
        else: cls = 'far-flash->13us(geom-first)'
        fpe = prompt_pe(ev, far_side, tn) if ok else ''
        npe = prompt_pe(ev, near_side, tn) if ok else ''
        dd, tfn, pen = nearest(ev, far_side, tn) if ok else ('', '', '')
        s = N.get(ev, ('?', -1, -1, -1, -1))
        adopted = 'near' if int(gout) == g1 else 'far'
        if ok and dt0 is not None:
            tfar = tn + dt0
            bpe = f'{prompt_pe(ev, near_side, tfar):.0f}'
            d2, tf2, pe2 = nearest(ev, near_side, tfar)
            bdt = f'{(tf2 - tfar):.3f}' if tf2 is not None else ''
        else:
            bpe = ''; bdt = ''
        print('\t'.join(map(str, [ev, s[0], s[1], s[3], s[4], kind.replace(' ', '_'), rule or '', near_side, t1, far_side,
                                  far_t if far_t is not None else '', f'{dt0:.3f}' if dt0 is not None else '', cls,
                                  f'{tn:.4f}' if ok else 'NOTFOUND', f'{best[2]:.0f}' if ok else '',
                                  f'{fpe:.0f}' if ok else '', f'{npe:.0f}' if ok else '',
                                  f'{(tfn - tn):.3f}' if ok and tfn is not None else '', f'{pen:.0f}' if ok and pen is not None else '',
                                  adopted, bpe, bdt, s[2]])))
