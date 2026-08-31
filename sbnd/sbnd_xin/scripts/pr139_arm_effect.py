#!/usr/bin/env python3
"""doc pr/139 phase 1 -- what ONE knob did, against the shipped baseline.

  ./scripts/pr139_arm_effect.py <armtag> [more armtags ...]

Baseline is work-pr139r1-off-* (the new binary, every pr/139 knob at its shipped
default, gate PASS 478/478 vs work-pr138r3-flipchk).  Comparing against the OFF
arm rather than against work-pr138r2-c90on keeps binary and config identical, so
every difference is the knob.

Keyed on `shower_id` throughout -- `showers[].id` is the START SEGMENT display id
and a daughter can be seeded on a segment that already roots another shower
(doc pr/139 sec 2).
"""
import json, os, glob, csv, sys, collections

SAMPLES = ('mcp1k', 'mcp2k', 'ncpi0', 'nuecc48')
BASE = 'work-pr139r1-off'
FOUR = (281485, 396222, 165157, 54332)
GAINS = (280972, 56243, 314838, 269774)


def load(arm, s, ev):
    p = f'{arm}-{s}/pr_evt{ev}/calib-pr-evt{ev}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def scan(arm):
    tot = collections.Counter()
    rows = []
    for s in SAMPLES:
        if not os.path.isdir(f'{BASE}-{s}'):
            continue
        for d in sorted(glob.glob(f'{BASE}-{s}/pr_evt*')):
            ev = int(os.path.basename(d)[6:])
            A, B = load(BASE, s, ev), load(arm, s, ev)
            if not (A and B):
                tot['missing'] += 1
                continue
            O = {x['shower_id']: x for x in A['showers']}
            N = {x['shower_id']: x for x in B['showers']}
            tot['events'] += 1
            if len(N) != len(O):
                tot['events_shower_count_moved'] += 1
            offid = collections.Counter(x['id'] for x in O.values())
            # a "daughter" here = a shower_id the OFF arm's SPLIT already made;
            # for a knob arm what matters is the set present in THIS arm that the
            # pre-split world did not have, so recompute it the same way: any
            # shower_id in this arm that the OFF arm also lacks is a new object.
            for i in N:
                if i in O:
                    continue
                x = N[i]
                tot['new_objects'] += 1
                tot['pdg_%d' % abs(x['particle_id'])] += 1
                if x['kine_charge'] < 1e-6:
                    tot['kine_zero'] += 1
                if offid.get(x['id']):
                    tot['seed_collides'] += 1
            for i in O:
                if i not in N:
                    tot['objects_gone'] += 1
    return tot, rows


def pi0_rows(arm, ev):
    for s in SAMPLES:
        A, B = load(BASE, s, ev), load(arm, s, ev)
        if A and B:
            O = {x['shower_id']: x for x in A['showers']}
            N = {x['shower_id']: x for x in B['showers']}
            out = []
            for i in sorted(set(O) | set(N)):
                o, n = O.get(i), N.get(i)
                if o and n and abs(o['kine_charge'] - n['kine_charge']) < 0.01 \
                   and o['pio_id'] == n['pio_id'] and o['particle_id'] == n['particle_id']:
                    continue
                f = lambda x: '-' if x is None else \
                    'seg%-7d pdg%-5d k=%9.2f ns=%-4d pio=%-3d' % (
                        x['id'], x['particle_id'], x['kine_charge'],
                        x['num_segments'], x['pio_id'])
                out.append("    sid=%-4d OFF %-44s ON %s" % (i, f(o), f(n)))
            return s, out
    return None, []


for arm_tag in sys.argv[1:]:
    arm = f'work-pr139r1-{arm_tag}'
    print("\n========== %s vs %s ==========" % (arm, BASE))
    tot, _ = scan(arm)
    for k in sorted(tot):
        print("  %-32s %d" % (k, tot[k]))
    print("\n  -- the four broken pi0s (doc pr/139 sec 2) --")
    for ev in FOUR:
        s, out = pi0_rows(arm, ev)
        print("  evt%-8d (%s)  %s" % (ev, s, 'unchanged by this knob' if not out else ''))
        for l in out:
            print(l)
    print("\n  -- the four census GAINS the splitter bought (must not be lost) --")
    for ev in GAINS:
        s, out = pi0_rows(arm, ev)
        print("  evt%-8d (%s)  %s" % (ev, s, 'unchanged by this knob' if not out else ''))
        for l in out:
            print(l)
