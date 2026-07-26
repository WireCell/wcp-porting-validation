#!/usr/bin/env python3
"""Join the 10 sub-agent verdict slices onto the keyed skeleton -> batch-2 TSV.

The agents only ever wrote (event, main_id, verdict, quality, conf, reason);
flash_gid / t_us / len_cm / auto_label come from the event's own nusel TSV here,
so the --ai-scan join key cannot be corrupted by a transcription slip.

Control events (scanned in all 10 slices, blind) are split out into an
inter-agent agreement table instead of the batch TSV.
"""
import glob
import os
import sys

SBND = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
R = f'{SBND}/work-mcp1kall-d59k'
B = '/home/xqian/tmp/nusel-b2'
# seeded into every slice: 48301/48895/50787 are the events the criteria are
# calibrated on (an agent recognises them), 48367/51865/52723 are BLIND -- also
# scanned in the first-20 batch but cited nowhere in the criteria and cut out of
# the examples file the agents were given.  All six are excluded from the batch
# TSV (their verdicts already live in handscan-first20.tsv) and go into the
# agreement table instead.
CONTROLS = {'48301', '48895', '50787', '48367', '51865', '52723'}
BLIND = {'48367', '51865', '52723'}
VERDICTS = {'STM', 'nu'}
QUALITY = {'clean', 'weak', 'cosmic-like', 'junk', 'unclear'}
CONF = {'high', 'med', 'low'}


def inbeam_rows(evt):
    p = f'{R}/nusel_evt{evt}/nusel-evt{evt}.tsv'
    rows = [l.split() for l in open(p).read().splitlines() if l.strip()]
    head = rows[0]
    return [dict(zip(head, r)) for r in rows[1:]
            if len(r) == len(head)
            and dict(zip(head, r)).get('in_beam') == '1'
            and dict(zip(head, r)).get('label') != 'no-bundle']


# expected work: union of the slice skeletons, minus controls
expect = {}
for i in range(10):
    for ln in open(f'{B}/skel-s{i}.tsv').read().splitlines()[1:]:
        f = ln.split('\t')
        if f[0] in CONTROLS:
            continue
        expect[(f[0], f[1])] = i

# Two bundles were grabbed WITHOUT the display ever focusing them: the row-0
# click did not take and both rows' screenshots show the last row's bundle
# (evt62495 r0/r1 and evt400174 r0/r1 are byte-identical triples).  The two
# agents said so instead of inventing a verdict.  Both were re-grabbed with
# per-row focus verification (regrab_verified.py: click another row first, then
# assert the info div names the wanted main) and judged by the main session from
# those images; the override is recorded here rather than by editing an agent's
# own file.
OVERRIDE = {
    ('62495', '16'): dict(
        slice='1-ovr', verdict='nu', quality='unclear', conf='med',
        reason='re-grabbed with verified focus (909 pts drawn in pr): sharp V, apex '
               'at (75,153,29) mid-track, one 30 cm arm crossing the top wall '
               'y=199.5 and one ending inside at (66,134,52); dQ/dx overshoots the '
               'muon expectation ~2x (300 ke/cm at rr=0 where the table gives 165) '
               'and stays ~1.8x high over 30 cm, and the fit rejects on 33 cm of '
               '2.33 MIP leftover past kink@90 -- reads as a vertex/heavy-ionizing '
               'end rather than a muon Bragg, so nu; unclear because a late-scatter '
               'stopping muon is still live'),
    ('400174', '8'): dict(
        slice='8-ovr', verdict='STM', quality='clean', conf='high',
        reason='re-grabbed with verified focus (721 pts drawn in pr): single 85 cm '
               'track entering the top wall at (166,199.7,236), curving as it slows '
               'and stopping at (153,118,241) some 80 cm clear of every face; dQ/dx '
               'tracks the green muon curve exactly, MIP 56 out at rr=40-85 rising '
               'to ~190 ke/cm at rr=0 (ks1=0.018) -- the tagger STM tag is right'),
}

got, ctrl, bad = {}, {}, []
for p in sorted(glob.glob(f'{B}/verdicts/s*.tsv')):
    sl = os.path.basename(p)[1:-4]
    lines = [l for l in open(p).read().splitlines() if l.strip()]
    for ln in lines:
        f = ln.split('\t')
        if f[0] in ('event', '#event') or ln.startswith('#'):
            continue
        if len(f) < 6:
            bad.append(f'{p}: only {len(f)} fields: {ln[:70]}')
            continue
        ev, mid, v, q, c, why = f[0].replace('evt', ''), f[1], f[2], f[3], f[4], '\t'.join(f[5:])
        why = ' '.join(why.split())
        if v not in VERDICTS:
            bad.append(f'{p}: illegal verdict {v!r} on {ev}:{mid}')
        if q not in QUALITY:
            bad.append(f'{p}: illegal quality {q!r} on {ev}:{mid}')
        if c not in CONF:
            bad.append(f'{p}: illegal conf {c!r} on {ev}:{mid}')
        if not why:
            bad.append(f'{p}: empty reason on {ev}:{mid}')
        rec = dict(slice=sl, verdict=v, quality=q, conf=c, reason=why)
        if ev in CONTROLS:
            ctrl.setdefault((ev, mid), []).append(rec)
        else:
            if (ev, mid) in got:
                bad.append(f'duplicate verdict for {ev}:{mid} '
                           f'(s{got[(ev, mid)]["slice"]} and s{sl})')
            got[(ev, mid)] = rec

import json
# did every judged bundle actually have evidence in its slice's info.json?
drawn = {}
for i in range(10):
    p = f'{B}/shots/s{i}/info.json'
    if not os.path.exists(p):
        continue
    for b in json.load(open(p))['bundles']:
        tr = b['table_row']
        if len(tr) > 5:
            drawn[(b['event'].replace('evt', ''), tr[5])] = b['bbox']['n']
noev = [k for k in expect if k not in drawn]
nopts = [k for k in expect if drawn.get(k) == 0]
print(f'bundles with NO image/info entry: {len(noev)} {noev[:12]}')
print(f'bundles whose drawn point count is 0: {len(nopts)} {nopts[:12]}')

for k, v in OVERRIDE.items():
    assert k in expect, f'override key {k} is not in the work list'
    got[k] = v

missing = sorted(set(expect) - set(got), key=lambda k: (int(k[0]), int(k[1])))
extra = sorted(set(got) - set(expect), key=lambda k: (int(k[0]), int(k[1])))
print(f'expected {len(expect)} bundles, got {len(got)}; '
      f'missing {len(missing)}, extra {len(extra)}, controls {len(ctrl)} keys')
for k in missing[:20]:
    print(f'  MISSING {k[0]}:{k[1]}  (slice {expect[k]})')
for k in extra[:20]:
    print(f'  EXTRA   {k[0]}:{k[1]}')
for m in bad[:30]:
    print('  BAD ' + m)

# ---- write the batch TSV, keys taken from the nusel tables ------------------
out = f'{SBND}/scan-d59k/handscan-batch2.tsv'
n = 0
tally, qtal = {}, {}
with open(out, 'w') as f:
    f.write('# doc 61 sec 5b -- batch 2: events 11-393 of the d59k scan set (393-event\n'
            '# FC-cut set), scanned 2026-07-26 by 10 Claude sub-agents from the same\n'
            '# evidence images the owner sees on :5011.  UNVALIDATED.\n'
            '# verdict is STM|nu ONLY (a non-stopping-muon stays a neutrino candidate);\n'
            '# quality = clean|weak|cosmic-like|junk|unclear qualifies it (doc 61 sec 1).\n'
            '# t_us/len_cm/auto_label/flash_gid are joined from nusel-evt<ID>.tsv, not typed.\n')
    f.write('event\tmain_id\tflash_gid\tt_us\tlen_cm\tauto_label\tverdict\tquality\tconf\treason\n')
    for (ev, mid) in sorted(got, key=lambda k: (int(k[0]), int(k[1]))):
        rows = [r for r in inbeam_rows(ev) if r['main_id'] == mid]
        if len(rows) != 1:
            print(f'  KEY FAIL {ev}:{mid} matches {len(rows)} in-beam rows')
            continue
        d, g = rows[0], got[(ev, mid)]
        f.write('\t'.join([ev, mid, d['flash_gid'], d['flash_time_us'],
                           d['len_main_cm'], d['label'], g['verdict'],
                           g['quality'], g['conf'], g['reason']]) + '\n')
        n += 1
        tally[g['verdict']] = tally.get(g['verdict'], 0) + 1
        qtal[g['quality']] = qtal.get(g['quality'], 0) + 1
print(f'\nwrote {out}: {n} rows')
print('  verdict:', dict(sorted(tally.items())))
print('  quality:', dict(sorted(qtal.items())))

# ---- tagger vs scan, and per-slice rates (a slice that drifted shows here) --
conf = {}
per_slice = {}
for ln in open(out).read().splitlines():
    if ln.startswith('#') or ln.startswith('event'):
        continue
    f = ln.split('\t')
    conf[(f[5], f[6])] = conf.get((f[5], f[6]), 0) + 1
for k in sorted(got):
    g = got[k]
    d = per_slice.setdefault(g['slice'], {'STM': 0, 'nu': 0})
    d[g['verdict']] += 1
print('\ntagger label vs scan verdict:')
for (auto, v), c in sorted(conf.items()):
    print(f'  auto {auto:<13} -> {v:<4} {c:4d}')
print('\nper-slice STM rate (drift check):')
for s in sorted(per_slice, key=lambda s: (len(s), s)):
    d = per_slice[s]
    t = d['STM'] + d['nu']
    print(f'  s{s}: {d["STM"]:3d} STM / {t:3d}  ({100.0 * d["STM"] / t:.0f}%)')

# ---- control-event agreement ----------------------------------------------
print('\ncontrol events (blind, in all 10 slices):')
mine = {}
for ln in open(f'{SBND}/scan-d59k/handscan-first20.tsv'):
    if ln.startswith('#'):
        continue
    f = ln.split('\t')
    if f[0] in CONTROLS:
        mine[f[0]] = (f[6], f[7])          # verdict, quality
for (ev, mid), recs in sorted(ctrl.items(), key=lambda kv: int(kv[0][0])):
    vs = [r['verdict'] for r in recs]
    qs = [r['quality'] for r in recs]
    agree = sum(1 for v in vs if v == mine[ev][0])
    kind = 'BLIND ' if ev in BLIND else 'known '
    print(f'  {kind}evt{ev} main {mid}: n={len(recs):2d}  '
          f'verdicts={ {v: vs.count(v) for v in set(vs)} }  '
          f'qualities={ {q: qs.count(q) for q in set(qs)} }  '
          f'-> matches the first-20 call ({mine[ev][0]}/{mine[ev][1]}): '
          f'{agree}/{len(recs)}')

sys.exit(1 if (bad or missing or extra) else 0)
