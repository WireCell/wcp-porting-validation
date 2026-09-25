#!/usr/bin/env python3
"""gen_iso_tracks.py -- the isochronous-track gun event list for the FD-HD 1x2x6 workspace
(wcfm/docs/01 sec 6 W2, wcfm/docs/02 sec 2).

Writes one JSON per event, events/<run6>_<evt>.json:
  {run, event, seed, kind: iso|cosmic, step_mm, tracks: [{tail:[cm], head:[cm], charge,
   angle_deg, length_cm}], anodes: [idents the tracks cross]}
and the gate manifest abtest_events.txt (`wcfm <run> <evt>` lines, abtest/events.txt format).

An isochronous track is a straight line at angle theta to the anode plane (theta = 0 is
exactly parallel: constant x, all its charge in a few 2 us slices).  Overlay events put
2-4 iso tracks at the SAME drift distance (|dx| < 3 cm) inside the same APA region so their
projections share slices and wires -- that is where the projective-readout ghosts come
from.  Cosmic controls are ordinary straight tracks with a random 3-D direction.

Geometry (mm, from the wires file; see wcfm_params.jsonnet): APA rows y<0 (even ident) /
y>0 (odd), z columns of 2306.4 mm every 2323.9 mm, drift |x| up to 3629 mm.

    python3 gen_iso_tracks.py            # writes events/*.json + abtest_events.txt (refuses to overwrite)
    python3 gen_iso_tracks.py --force
    python3 gen_iso_tracks.py --extend 100   # wcfm/docs/05: adds events 11..100 (random kinds, seeds 1000+evt)
                                             # and gnn_events.txt; never rewrites an existing event file

Extended events (--extend, doc 05 GNN sample): per event the seeded RNG first draws the kind --
30 % single iso track (theta in THETAS), 40 % iso overlay of 2-4 tracks, 30 % cosmic (1-2 tracks)
-- then the geometry exactly as make_event does for the hand-listed events 1-10.
"""
import argparse
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
Z_PITCH_MM, Z_LEN_MM = 2323.9, 2306.4
X_ABS_CM = (20.0, 340.0)       # keep off the anode cut-off plane and the cathode
Y_LIM_CM = (-595.0, 595.0)
Z_LIM_CM = (2.0, 1390.0)
CHARGE_PER_STEP = -500         # electrons per 0.1 mm step = 5000 e/mm (pdhd_sim MIP convention)
STEP_MM = 0.1
RUN = 1

# (event, kind, [(theta_deg or None, length_cm), ...]); overlays share x and APA region
EVENTS = [
    (1, 'iso', [(0.0, 300)]),
    (2, 'iso', [(0.0, 500)]),
    (3, 'iso', [(2.0, 300)]),
    (4, 'iso', [(5.0, 300)]),
    (5, 'iso', [(10.0, 300)]),
    (6, 'iso', [(0.0, 300), (1.0, 250)]),
    (7, 'iso', [(0.0, 300), (1.0, 250), (2.0, 400)]),
    (8, 'iso', [(0.0, 300), (0.0, 200), (3.0, 350), (5.0, 250)]),
    (9, 'cosmic', [(None, 400)]),
    (10, 'cosmic', [(None, 300), (None, 350)]),
]


def inside(p):
    return (X_ABS_CM[0] <= abs(p[0]) <= X_ABS_CM[1] and Y_LIM_CM[0] <= p[1] <= Y_LIM_CM[1]
            and Z_LIM_CM[0] <= p[2] <= Z_LIM_CM[1])


def anode_of(y_cm, z_cm):
    row = 1 if y_cm > 0 else 0
    col = min(max(int(z_cm * 10.0 // Z_PITCH_MM), 0), 5)
    return 2 * col + row


def anodes_touched(tail, head, step_cm=1.0):
    n = max(2, int(math.dist(tail, head) / step_cm))
    ids = set()
    for i in range(n + 1):
        f = i / n
        y = tail[1] + f * (head[1] - tail[1])
        z = tail[2] + f * (head[2] - tail[2])
        ids.add(anode_of(y, z))
    return sorted(ids)


def endpoints(center, direction, length):
    tail = [center[k] - direction[k] * length / 2 for k in range(3)]
    head = [center[k] + direction[k] * length / 2 for k in range(3)]
    return tail, head


def iso_direction(rng, theta_deg):
    th = math.radians(theta_deg)
    phi = rng.uniform(0, 2 * math.pi)
    sx = rng.choice([-1, 1])
    return (sx * math.sin(th), math.cos(th) * math.sin(phi), math.cos(th) * math.cos(phi))


def cosmic_direction(rng):
    # downward-going, zenith angle with cos in [0.3, 1], random azimuth
    cz = rng.uniform(0.3, 1.0)
    sz = math.sqrt(1 - cz * cz)
    az = rng.uniform(0, 2 * math.pi)
    return (sz * math.cos(az), -cz, sz * math.sin(az))


THETAS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)


def random_spec(rng):
    """(kind, [(theta_deg|None, length_cm), ...]) for an extended event; consumes rng first."""
    u = rng.random()
    if u < 0.3:
        return 'iso', [(rng.choice(THETAS), rng.randrange(200, 501, 50))]
    if u < 0.7:
        n = rng.randrange(2, 5)
        return 'iso', [(rng.choice(THETAS), rng.randrange(200, 501, 50)) for _ in range(n)]
    n = rng.randrange(1, 3)
    return 'cosmic', [(None, rng.randrange(250, 451, 50)) for _ in range(n)]


def make_event(evt, kind, spec, rng):
    tracks = []
    center0 = None
    for theta, length in spec:
        for _ in range(10000):
            if center0 is None:
                center = [rng.choice([-1, 1]) * rng.uniform(60, 300), rng.uniform(-500, 500), rng.uniform(100, 1290)]
            else:   # overlay: same drift distance, same APA region
                center = [center0[0] + rng.uniform(-3, 3), center0[1] + rng.uniform(-100, 100),
                          center0[2] + rng.uniform(-100, 100)]
            d = iso_direction(rng, theta) if kind == 'iso' else cosmic_direction(rng)
            tail, head = endpoints(center, d, length)
            if inside(tail) and inside(head) and (tail[0] > 0) == (head[0] > 0):
                break
        else:
            raise RuntimeError(f'event {evt}: no contained track for {theta} {length}')
        if center0 is None:
            center0 = center
        tracks.append({'tail': [round(v, 3) for v in tail], 'head': [round(v, 3) for v in head],
                       'charge': CHARGE_PER_STEP, 'angle_deg': theta, 'length_cm': length})
    anodes = sorted(set(a for t in tracks for a in anodes_touched(t['tail'], t['head'])))
    return {'run': RUN, 'event': evt, 'seed': 1000 * RUN + evt, 'kind': kind, 'step_mm': STEP_MM,
            'tracks': tracks, 'anodes': anodes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--extend', type=int, metavar='N', help='add events max(EVENTS)+1..N (doc 05); existing files are kept')
    a = ap.parse_args()
    evdir = os.path.join(HERE, 'events')
    os.makedirs(evdir, exist_ok=True)
    if a.extend:
        first = max(e for e, _, _ in EVENTS) + 1
        lines = ['# wcfm GNN sample manifest (gen_iso_tracks.py --extend): det run evt kind']
        for evt in range(1, a.extend + 1):
            rng = random.Random(1000 * RUN + evt)
            path = os.path.join(evdir, f'{RUN:06d}_{evt}.json')
            if evt < first:
                e = json.load(open(path))
            else:
                kind, spec = random_spec(rng)
                e = make_event(evt, kind, spec, rng)
                if not os.path.exists(path):
                    with open(path, 'w') as f:
                        json.dump(e, f, indent=1)
                elif json.load(open(path)) != e:
                    raise SystemExit(f'REFUSING: {path} exists and differs')
            lines.append(f'wcfm {RUN} {evt} {e["kind"]}{len(e["tracks"])}')
            print(f'{os.path.basename(path)}  {e["kind"]:6s}  anodes {e["anodes"]}  '
                  + ' + '.join(f"{t['angle_deg']}deg/{t['length_cm']}cm" if t['angle_deg'] is not None
                               else f"cosmic/{t['length_cm']}cm" for t in e['tracks']))
        manifest = os.path.join(HERE, 'gnn_events.txt')
        with open(manifest, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'-> {manifest}')
        return
    manifest = os.path.join(HERE, 'abtest_events.txt')
    if os.path.exists(manifest) and not a.force:
        raise SystemExit(f'REFUSING: {manifest} exists (use --force)')
    lines = ['# wcfm gate manifest (gen_iso_tracks.py): det run evt', ]
    for evt, kind, spec in EVENTS:
        rng = random.Random(1000 * RUN + evt)
        e = make_event(evt, kind, spec, rng)
        path = os.path.join(evdir, f'{RUN:06d}_{evt}.json')
        with open(path, 'w') as f:
            json.dump(e, f, indent=1)
        lines.append(f'wcfm {RUN} {evt}')
        desc = ' + '.join(f"{t['angle_deg']}deg/{t['length_cm']}cm" if t['angle_deg'] is not None
                          else f"cosmic/{t['length_cm']}cm" for t in e['tracks'])
        print(f'{os.path.basename(path)}  {kind:6s}  anodes {e["anodes"]}  x0 {e["tracks"][0]["tail"][0]:7.1f}  {desc}')
    with open(manifest, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'-> {manifest}')


if __name__ == '__main__':
    main()
