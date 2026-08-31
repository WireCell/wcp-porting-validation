#!/usr/bin/env python3
"""doc pr/139 sec 2 -- what the shipped splitter actually did to the four pi0s.

Supersedes the 2026-08-31 reading of doc pr/139 sec 1, which said 281485 and
396222 were "correct cuts whose pair broke because the finder paired on the
reduced gamma energy" and that split-aware pairing was the ONLY fix.  Reading the
arms says otherwise.

THE JOIN HAZARD, and it bit the first pass.  The dump's `showers[].id` is the
START SEGMENT display id, not a shower identity: a peeled daughter can be seeded
on a segment that already ROOTS another shower (evt165157 does exactly that), so
an id-keyed off/on comparison silently merges two distinct objects into one row
and invents charge that "moved".  Everything here is keyed on `shower_id`.

Arms: work-pr138r2-c90{off,on}-* (c90on == the shipped SBND baseline, which
work-pr138r3-flipchk-* reproduces byte-for-byte).
"""
import json, os, glob, csv, collections, sys

ARMA, ARMB = 'work-pr138r2-c90off', 'work-pr138r2-c90on'
SAMPLES = ('mcp1k', 'mcp2k', 'ncpi0', 'nuecc48')
FOUR = {281485: 'partial -> none', 396222: 'partial -> none',
        165157: 'partial -> no-group', 54332: 'exact -> partial'}


def dumps(arm, sample, ev):
    p = f'{arm}-{sample}/pr_evt{ev}/calib-pr-evt{ev}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def walk():
    for s in SAMPLES:
        if not os.path.isdir(f'{ARMA}-{s}'):
            continue
        for d in sorted(glob.glob(f'{ARMA}-{s}/pr_evt*')):
            ev = int(os.path.basename(d)[6:])
            A, B = dumps(ARMA, s, ev), dumps(ARMB, s, ev)
            if A and B:
                yield s, ev, A, B


tot = collections.Counter()
rows, path = [], []
for s, ev, A, B in walk():
    O = {x['shower_id']: x for x in A['showers']}
    N = {x['shower_id']: x for x in B['showers']}
    new = [i for i in N if i not in O]
    if not new:
        continue
    tot['events_fired'] += 1
    # the parents: present in both, fewer segments after
    parents = [i for i in O if i in N and N[i]['num_segments'] < O[i]['num_segments']]
    tot['parents'] += len(parents)
    # a parent holding segments ANOTHER shower also owns: num_segments exceeds
    # the number of dump segments whose (single-valued) shower_id points at it
    cnt = collections.Counter(sg.get('shower_id') for sg in A['segments'])
    for i in parents:
        if cnt.get(O[i]['id'], 0) < O[i]['num_segments']:
            tot['parents_with_shared_members'] += 1
            path.append((ev, O[i]['id'], O[i]['num_segments'], cnt.get(O[i]['id'], 0),
                         O[i]['kine_charge'], N[i]['kine_charge']))
    offid = collections.Counter(x['id'] for x in O.values())
    for i in new:
        x = N[i]
        tot['daughters'] += 1
        tot['pdg_%d' % abs(x['particle_id'])] += 1
        if x['kine_charge'] < 1e-6:
            tot['daughters_kine_zero'] += 1
        if offid.get(x['id']):
            tot['daughters_seed_collides'] += 1
        rows.append(dict(event=ev, sample=s, shower_id=i, start_seg=x['id'],
                         pdg=x['particle_id'], kine=x['kine_charge'],
                         nseg=x['num_segments'], pio=x['pio_id'],
                         kine_zero=int(x['kine_charge'] < 1e-6),
                         seed_collides=int(bool(offid.get(x['id'])))))

print("=== the shipped splitter, off vs on, keyed on shower_id ===")
for k in sorted(tot):
    print("  %-32s %d" % (k, tot[k]))

print("\n=== sec 2 -- the four broken pi0s, shower by shower ===")
for ev, verdict in FOUR.items():
    got = None
    for s in SAMPLES:
        A, B = dumps(ARMA, s, ev), dumps(ARMB, s, ev)
        if A and B:
            got = (s, A, B)
            break
    if not got:
        print("  evt%-8d NOT ON DISK" % ev)
        continue
    s, A, B = got
    O = {x['shower_id']: x for x in A['showers']}
    N = {x['shower_id']: x for x in B['showers']}
    print("\n  evt%-8d (%s)  census %s" % (ev, s, verdict))
    f = lambda x: '-' if x is None else \
        'seg%-7d pdg%-5d k=%9.2f ns=%-4d pio=%-3d m=%7.1f' % (
            x['id'], x['particle_id'], x['kine_charge'], x['num_segments'],
            x['pio_id'], x['pio_mass'])
    for i in sorted(set(O) | set(N)):
        o, n = O.get(i), N.get(i)
        if o and n and abs(o['kine_charge'] - n['kine_charge']) < 0.01 \
           and o['pio_id'] == n['pio_id'] and o['particle_id'] == n['particle_id']:
            continue
        print("    sid=%-4d OFF %-48s ON %s" % (i, f(o), f(n)))

print("\n=== fired parents holding segments another shower also owns ===")
print("  %-9s %-9s %5s %6s %10s %10s" % ('event', 'parent', 'nseg', 'mapped', 'k_off', 'k_on'))
for r in sorted(path):
    print("  %-9d %-9d %5d %6d %10.2f %10.2f" % r)

with open('docs/pr/pr139-broken-pi0.tsv', 'w') as fh:
    w = csv.DictWriter(fh, delimiter='\t', fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("\nwrote docs/pr/pr139-broken-pi0.tsv  (%d daughters)" % len(rows))
