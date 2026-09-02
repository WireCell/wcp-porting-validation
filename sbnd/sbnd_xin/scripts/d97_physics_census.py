#!/usr/bin/env python3
"""doc 97 -- physics-level A/B between two PR arms, from the calib dumps.

A flip table counts VERDICTS.  A clustering knob can also change what is
inside a bundle whose verdict never moves -- the reconstructed neutrino
energy, the shower count, the pi0 mass -- and the owner's "no regression"
question is about those too.  This reads calib-pr-evt<ID>.json from both arms
and compares:

    kine_reco_Enu, kine_reco_add_energy, kine_energy_excluded,
    kine_pio_mass / kine_pio_flag, main_vertex (x,y,z),
    len(showers), len(segments), len(vertices), sum of shower kine_best

Deliberately NOT a whole-file diff: vertex_scoreboard.dual_chain.off_ms is a
wall-clock TIMER, so two byte-identical reconstructions "differ" on it every
time (recorded lesson).  Everything compared here is a physics quantity.

  Usage: d97_physics_census.py <on-suffix> [off-suffix] [sample ...]
Read-only.  Exit 0 always -- this is a measurement, not a gate.
"""
import glob, json, math, os, sys
from collections import Counter

ON = sys.argv[1] if len(sys.argv) > 1 else 'd97onpr'
OFF = sys.argv[2] if len(sys.argv) > 2 else 'r3entry'
SAMPLES = sys.argv[3:] or ['ncpi0', 'nuecc48', 'mcp1k', 'mcp2k']
TOL = 1e-6          # relative; anything above this is a real change


def summary(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    k = d.get('kine', {})
    mv = d.get('main_vertex', {}) or {}
    sh = d.get('showers', []) or []
    return {
        'Enu': k.get('kine_reco_Enu'),
        'add_energy': k.get('kine_reco_add_energy'),
        'excluded': k.get('kine_energy_excluded'),
        'pio_mass': k.get('kine_pio_mass'),
        'pio_flag': k.get('kine_pio_flag'),
        'vtx_x': mv.get('x'), 'vtx_y': mv.get('y'), 'vtx_z': mv.get('z'),
        'n_showers': len(sh),
        'n_segments': len(d.get('segments', []) or []),
        'n_vertices': len(d.get('vertices', []) or []),
        'shower_sum': sum(s.get('kine_best') or 0.0 for s in sh),
    }


def differs(a, b):
    if a is None or b is None:
        return a is not b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return False
        scale = max(abs(a), abs(b), 1.0)
        return abs(a - b) / scale > TOL
    return a != b


tot = Counter()
moved = []
for smp in SAMPLES:
    for p in sorted(glob.glob(f'work-{smp}-{ON}/pr_evt*/calib-pr-evt*.json')):
        evt = p.split('pr_evt')[1].split('/')[0]
        a = summary(p)
        b = summary(p.replace(f'-{ON}/', f'-{OFF}/'))
        if a is None or b is None:
            tot['event-missing'] += 1
            continue
        tot['events'] += 1
        fields = [f for f in a if differs(a[f], b[f])]
        if fields:
            tot['events-moved'] += 1
            moved.append((smp, evt, fields, b, a))
            for f in fields:
                tot[f'field:{f}'] += 1

print(f'ON = work-*-{ON}   OFF = work-*-{OFF}   samples: {" ".join(SAMPLES)}')
print(f'events with a calib dump on both sides: {tot["events"]}   '
      f'missing on one side: {tot["event-missing"]}')
print(f'events where ANY compared physics quantity moved: {tot["events-moved"]}')
print()
print(f'  {"quantity":<14}{"events moved":>14}')
for f in sorted(k for k in tot if k.startswith('field:')):
    print(f'  {f[6:]:<14}{tot[f]:>14}')

if moved:
    print(f'\nPER-EVENT DETAIL ({len(moved)}):')
    print(f'  {"sample":<8}{"event":>8}  {"Enu off":>9}{"Enu on":>9}'
          f'{"dEnu":>9}{"nsh off":>8}{"nsh on":>7}{"vtx move cm":>12}  fields')
    for smp, evt, fields, b, a in moved:
        eo, en = b['Enu'], a['Enu']
        d = (en - eo) if (eo is not None and en is not None) else float('nan')
        dv = float('nan')
        if None not in (b['vtx_x'], a['vtx_x']):
            dv = math.dist((b['vtx_x'], b['vtx_y'], b['vtx_z']),
                           (a['vtx_x'], a['vtx_y'], a['vtx_z']))
        print(f'  {smp:<8}{evt:>8}  {eo if eo is None else f"{eo:9.1f}"}'
              f'{en if en is None else f"{en:9.1f}"}{d:9.1f}'
              f'{b["n_showers"]:>8}{a["n_showers"]:>7}{dv:>12.2f}  '
              f'{",".join(fields)}')
