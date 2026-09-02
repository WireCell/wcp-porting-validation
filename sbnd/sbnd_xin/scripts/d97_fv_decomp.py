#!/usr/bin/env python3
"""doc 97 -- which of sep_fv_point's four values does the work?

sep_fv_point sets four things at once on ClusteringSeparate:
    fv_inset_yz 15 cm, far_point_x_cut 14 cm, far_point_mid_dis 60 cm,
    dec1_guard_main_angle 45 deg
doc 96 sec 8.3 measured that neither the inset nor the far-point pair alone
reached 105-23-21 -- but it measured that with the inset applied to the SHARED
DetectorVolumes FV, which also moves clustering_neutrino and the containment
taggers.  The separation-scoped decomposition has never been measured, and the
owner is being asked to flip the knob.

Each arm patches the event's OWN compiled Q/L config from a byte-identical OFF
arm and redirects the output paths, so the only difference from production is
the listed keys.  Arm `off` (paths moved, nothing else) is the control.

Writes <out>/<arm>/evt<E>.json plus <out>/<arm>/ql_evt<E>/ for the products.
"""
import argparse, json, os

ARMS = {
    'off':        {},
    'inset':      {'fv_inset_yz': 150.0},
    'farpoint':   {'far_point_x_cut': 140.0, 'far_point_mid_dis': 600.0},
    'dec1':       {'dec1_guard_main_angle': 45.0},
    'inset+far':  {'fv_inset_yz': 150.0, 'far_point_x_cut': 140.0,
                   'far_point_mid_dis': 600.0},
    'all4':       {'fv_inset_yz': 150.0, 'far_point_x_cut': 140.0,
                   'far_point_mid_dis': 600.0, 'dec1_guard_main_angle': 45.0},
}

AP = argparse.ArgumentParser()
AP.add_argument('--src', default='work-dbg25a-d97off')
AP.add_argument('--out', default='/home/xqian/tmp/d97/decomp')
AP.add_argument('events', nargs='+')
A = AP.parse_args()

for evt in A.events:
    base = json.load(open(os.path.join(A.src, f'ql_evt{evt}', f'.wct-cfg-evt{evt}.json')))
    for arm, keys in ARMS.items():
        d = json.loads(json.dumps(base))
        outdir = os.path.join(A.out, arm, f'ql_evt{evt}')
        os.makedirs(outdir, exist_ok=True)
        nsep = nout = 0
        for n in d:
            data = n.get('data')
            if not isinstance(data, dict):
                continue
            if n.get('type') == 'ClusteringSeparate':
                data.update(keys)
                nsep += 1
            for k in ('bee_zip', 'outname'):
                v = data.get(k)
                if isinstance(v, str) and f'ql_evt{evt}' in v:
                    data[k] = os.path.join(outdir, os.path.basename(v))
                    nout += 1
        p = os.path.join(A.out, arm, f'evt{evt}.json')
        json.dump(d, open(p, 'w'))
        print(f'evt{evt} {arm:<11} {nsep} ClusteringSeparate block(s), '
              f'{nout} output path(s) -> {p}')
