#!/usr/bin/env python3
"""doc pr/57 round 2: machine first-pass labelling of S6 separation candidates.

The owner hand-scanned 117 events on the port-5018 display (doc pr/57 sec 6)
and stated the labelling rules in prose:

  1. edge inside a busy EM shower              -> OK
  2. a long track separated in the middle      -> bad
  3. near the neutrino vertex                  -> good
  4. OK gets filtered out before analysis, so commit to good/bad elsewhere,
     even when unsure.

This script turns those rules into measurable features taken from the
`WCT_OC56_SCAN_DUMP` JSONL (the same records the viewer draws) plus the
reconstructed neutrino vertex, CALIBRATES the thresholds against the owner's
own labels, reports the confusion matrix, and then writes labels for a new arm
in the viewer's exact schema so they can be audited in the display.

It is a first pass, not a substitute for the human scan: `calibrate` prints the
measured agreement and the bad-class recall, and those numbers are what any
downstream use of these labels must be discounted by.

Two structural choices that follow from the owner's round-1 instructions:

* Verdicts are decided at the level of the component PAIR (event, graph_call,
  j, k), not the individual edge -- "what matters is the separation of two
  clusters, not necessarily the individual edge". Every edge of a pair gets the
  pair's verdict, so a pair can never carry contradictory labels.
* Rule 2 is evaluated BEFORE rule 3. Vertex proximity by itself does not mean
  "good": in the owner's labels, edges with d_vtx < 3 cm split bad 11 / good 5,
  because long tracks get broken near the vertex too.

Usage:
    oc56_autoscan.py calibrate --arm work-pr58-scan48 --arm work-pr58-scan19 \
        --arm work-pr58-scan50 --labels overclustering_labels
      # fit thresholds on the owner's labels, print the confusion matrix

    oc56_autoscan.py features --arm <arm> [--events-file f] > feat.tsv
      # per-pair feature table (for inspection / plotting)

    oc56_autoscan.py label --arm work-pr57r2-scan395 --tag claude-scan50 \
        --first 50 [--params L=8,N=50,D=2000,L2=25]
      # write overclustering_labels/<tag>/labels-evt<ID>.json for the first N
      # events (by event id) of the arm that have a non-empty dump

    oc56_autoscan.py verify --arm <arm> --tag <tag>
      # every written key resolves to a real edge; every pair is self-consistent
"""
import argparse
import collections
import glob
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oc56_dump_check import pca_length_axis, angle_between_deg  # noqa: E402

SBND = os.path.normpath(os.path.join(HERE, '..', '..', '..'))

# viewer default filter (overclustering_scan_viewer.py:420-422): the population
# the owner actually labels.
DIS_MAX = 5.0

# Local-density radius (cm) used for the "busy EM shower" test.
DENS_R = 15.0

# Default operating point, fitted by `calibrate` against the owner's 575
# labels (see doc pr/57 sec 10). L/N = the long-track-break floors, D = the
# busy-shower density floor, Wd/Ld/Dd = the dead-W-crossing branch.
#
# Two branches that were measured and REJECTED, recorded so they are not
# re-invented: (a) a weak "one long component" branch (Lmax > 15 with no npts
# floor) buys +2 good/bad hits but calls 29 more owner-OK pairs bad; (b) a
# "both components long and populated => bad even with a W gap" branch
# (Lmin > 20, npmin >= 500) recovers evt137238 but costs 5 of 59 good recall.
#
# Ld=40 rather than 50 is the one threshold the calibration set does NOT fix:
# good/bad agreement is 89/91 for every Ld in [20, 60]. It was set by the
# visual audit -- evt167112 (a 47.6 cm track continuing past a 6-wire dead-W
# band with V never closing) is the owner's bad case (a) and is missed at 50,
# while Ld=30 would additionally flip evt169356, a 2.8 cm stub at a track end
# that should stay good.
DEFAULT_PARAMS = dict(L=6.0, N=50, D=2000.0, Wd=3, Ld=40.0, Dd=3.0)

CAUSE_INDUCTION = 'induction inefficiency'
CAUSE_DEAD = 'dead channel'
CAUSE_SHOWER = 'prolonged shower signal'
CAUSE_GENUINE = 'genuine separation'


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def nu_vertex(arm, evt):
    """Reconstructed neutrino vertex (cm) for one event, or None.

    Same route as scripts/analysis/pr51/vtx_struct.py: the single row with
    q == 15000 in mabc-pr.zip::data/0/0-vertices-global.json
    (MultiAlgBlobClustering.cxx:1054, kNeutrinoVertex).
    """
    zp = os.path.join(arm, 'pr_evt%s' % evt, 'mabc-pr.zip')
    if not os.path.isfile(zp):
        return None
    try:
        with zipfile.ZipFile(zp) as z:
            v = json.loads(z.read('data/0/0-vertices-global.json'))
    except (KeyError, zipfile.BadZipFile, ValueError):
        return None
    k = [i for i, q in enumerate(v['q']) if q == 15000.0]
    if len(k) != 1:
        return None
    i = k[0]
    return np.array([v['x'][i], v['y'][i], v['z'][i]], dtype=float)


def arm_events(arm):
    """[(evt_id_str, jsonl_path)] for every non-empty dump under an arm."""
    out = []
    for f in sorted(glob.glob(os.path.join(arm, 'pr_evt*', 'oc56scan-evt*.jsonl'))):
        if os.path.getsize(f) == 0:
            continue
        out.append((os.path.basename(f)[len('oc56scan-evt'):-len('.jsonl')], f))
    out.sort(key=lambda t: int(t[0]))
    return out


def load_event(path):
    comps, edges = {}, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r['type'] == 'component':
                comps[(r['graph_call'], r['comp'])] = np.asarray(
                    r['points'], dtype=float).reshape(-1, 3)
            elif r['type'] == 'edge':
                edges.append(r)
    return comps, edges


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def transverse_rms(points):
    if len(points) < 3:
        return 0.0
    c = points - points.mean(axis=0)
    w = np.linalg.eigvalsh(np.cov(c.T))
    return float(np.sqrt(max(w[0], 0.0) + max(w[1], 0.0)))


def w_dead_crossing(rec):
    """Number of distinct dead W wire indices spanning the two seed centroids.

    This is the owner's bad case (a): a dead-channel band in the middle of the
    W plane, with U/V distorted around it. W is passable through dead wires
    (s6_cell_passable), so the signature is 'W connected THROUGH dead wires'.
    """
    wp = [p for p in rec['planes'] if p['plane'] == 2]
    if not wp:
        return 0
    wp = wp[0]
    sa = [c[0] for c in wp['seeds_a']]
    sb = [c[0] for c in wp['seeds_b']]
    if not sa or not sb:
        return 0
    lo, hi = sorted([sum(sa) / len(sa), sum(sb) / len(sb)])
    return len({d[0] for d in wp['dead'] if lo - 2 <= d[0] <= hi + 2})


def edge_key(evt, rec):
    """The viewer's geometric key (overclustering_scan_viewer.py:170-175)."""
    p1, p2 = rec['p1'], rec['p2']
    return 'evt%s|%s|%.2f,%.2f,%.2f|%.2f,%.2f,%.2f' % (
        evt, rec['blk'], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])


def shown_in_viewer(rec):
    """The viewer's default filter: removed (or floor-blocked) and dis <= 5."""
    removed = rec['killed'] or rec.get('killed_ignore_floor')
    return bool(removed) and rec['dis'] <= DIS_MAX


def pair_table(arm, evt, path, want_shown_only=True):
    """-> {(call, j, k): pair_dict} with features and the member edges."""
    comps, edges = load_event(path)
    vtx = nu_vertex(arm, evt)
    by_call = collections.defaultdict(list)
    for (call, _), P in comps.items():
        by_call[call].append(P)

    pairs = {}
    for rec in edges:
        if want_shown_only and not shown_in_viewer(rec):
            continue
        call = rec['graph_call']
        j, k = min(rec['j'], rec['k']), max(rec['j'], rec['k'])
        A, B = comps.get((call, j)), comps.get((call, k))
        if A is None or B is None:
            continue
        pk = (call, j, k)
        p = pairs.get(pk)
        if p is None:
            La, axa = pca_length_axis(A)
            Lb, axb = pca_length_axis(B)
            p = pairs[pk] = dict(
                evt=evt, call=call, j=j, k=k, edges=[],
                Lmin=min(La, Lb), Lmax=max(La, Lb),
                Tmin=min(transverse_rms(A), transverse_rms(B)),
                Tmax=max(transverse_rms(A), transverse_rms(B)),
                npmin=min(len(A), len(B)), npmax=max(len(A), len(B)),
                angle=angle_between_deg(axa, axb),
                gw=False, wdeadX=0, dis=1e9, dens=1e9, dvtx=-1.0)
        p['edges'].append(rec)
        p['gw'] = p['gw'] or bool(rec['gap'][2])
        p['wdeadX'] = max(p['wdeadX'], w_dead_crossing(rec))
        p['dis'] = min(p['dis'], rec['dis'])
        mid = (np.asarray(rec['p1']) + np.asarray(rec['p2'])) / 2.0
        dens = sum(int((np.linalg.norm(P - mid, axis=1) < DENS_R).sum())
                   for P in by_call[call])
        p['dens'] = min(p['dens'], float(dens))
        if vtx is not None:
            d = float(np.linalg.norm(mid - vtx))
            p['dvtx'] = d if p['dvtx'] < 0 else min(p['dvtx'], d)
    return pairs


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------

def classify(p, prm):
    """-> (verdict, cause, rule_name, confidence).

    Order matters: rule 2 (long-track break) outranks rule 3 (near vertex),
    because d_vtx alone does not separate good from bad in the owner's labels
    (d_vtx < 3 cm is bad 11 / good 5).
    """
    if p['Lmin'] > prm['L'] and p['npmin'] >= prm['N'] and not p['gw']:
        cause = CAUSE_DEAD if p['wdeadX'] >= prm['Wd'] else CAUSE_INDUCTION
        return 'bad', cause, 'R2 long-track break, no W gap', 'high'
    if (p['wdeadX'] >= prm['Wd'] and not p['gw']
            and p['Lmax'] > prm['Ld'] and p['dis'] < prm['Dd']):
        # owner's bad case (a): dead-channel band in W, U/V distorted around it
        return 'bad', CAUSE_DEAD, 'R2d dead-W band, induction-only gap', 'low'
    if p['gw']:
        return 'good', CAUSE_GENUINE, 'R3 W-plane gap (robust plane sees a hole)', 'high'
    if p['dens'] > prm['D'] and p['Lmin'] <= prm['L']:
        return 'OK', CAUSE_SHOWER, 'R1 busy EM shower', 'high'
    return 'good', CAUSE_GENUINE, 'R4 residual, committed good', 'low'


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

def load_owner_labels(labels_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(labels_dir, 'labels-evt*.json'))):
        for k, v in json.load(open(f))['labels'].items():
            out[k] = v['verdict']
    return out


def cmd_calibrate(args):
    truth = load_owner_labels(args.labels)
    print('owner labels: %d over %d files  %s' % (
        len(truth), len(glob.glob(os.path.join(args.labels, 'labels-evt*.json'))),
        dict(collections.Counter(truth.values()))))

    rows = []
    for arm in args.arm:
        for evt, path in arm_events(arm):
            for pk, p in pair_table(arm, evt, path).items():
                labs = [truth[edge_key(evt, r)] for r in p['edges']
                        if edge_key(evt, r) in truth]
                if not labs:
                    continue
                # a pair the owner labelled: bad dominates, then good
                p['truth'] = ('bad' if 'bad' in labs else
                              'good' if 'good' in labs else 'OK')
                p['nlab'] = len(labs)
                rows.append(p)
    gb = [p for p in rows if p['truth'] in ('good', 'bad')]
    print('labelled pairs joined: %d (good/bad subset: %d)' % (len(rows), len(gb)))

    grid = [dict(DEFAULT_PARAMS, L=float(L), N=N, D=float(D))
            for L in (4, 6, 8, 10, 12)
            for N in (30, 50, 100, 200)
            for D in (1200, 1600, 2000, 3000)]

    def fit(train):
        best = None
        for prm in grid:
            hit = sum(1 for p in train if classify(p, prm)[0] == p['truth'])
            # tie-break toward fewer owner-OK pairs called bad; the owner's
            # round-1 note says OK "can go either way", so this is a weak
            # preference, never allowed to trade away a good/bad hit.
            ok_bad = sum(1 for p in rows
                         if p['truth'] == 'OK' and classify(p, prm)[0] == 'bad')
            if best is None or (hit, -ok_bad) > best[0]:
                best = ((hit, -ok_bad), prm)
        return best[1]

    prm = fit(gb)
    print('fitted params: %s' % prm)

    cm = collections.Counter((p['truth'], classify(p, prm)[0]) for p in gb)
    n = len(gb)
    hit = sum(v for (t, q), v in cm.items() if t == q)
    print('good/bad agreement (in-sample): %d/%d = %.1f%%'
          % (hit, n, 100.0 * hit / max(n, 1)))
    for t in ('good', 'bad'):
        tot = sum(v for (tt, _), v in cm.items() if tt == t)
        print('  %-4s recall %d/%d = %.1f%%   -> %s' % (
            t, cm[(t, t)], tot, 100.0 * cm[(t, t)] / max(tot, 1),
            {q: v for (tt, q), v in cm.items() if tt == t}))
    okrows = [p for p in rows if p['truth'] == 'OK']
    print('  on OK-labelled pairs (owner: "can go either way"): %s'
          % dict(collections.Counter(classify(p, prm)[0] for p in okrows)))

    # Honest number: refit on all events but one, score the held-out event.
    # Grouping by EVENT (not pair) so pairs of the same event cannot leak.
    tot = cvhit = 0
    for e in sorted({p['evt'] for p in gb}):
        prm_e = fit([p for p in gb if p['evt'] != e])
        for p in [p for p in gb if p['evt'] == e]:
            tot += 1
            cvhit += classify(p, prm_e)[0] == p['truth']
    print('leave-one-EVENT-out CV good/bad agreement: %d/%d = %.1f%%'
          % (cvhit, tot, 100.0 * cvhit / max(tot, 1)))
    print('PARAMS=' + ','.join('%s=%g' % (k, v) for k, v in sorted(prm.items())))
    return prm


# ---------------------------------------------------------------------------
# label / features / verify
# ---------------------------------------------------------------------------

def parse_params(s):
    prm = dict(DEFAULT_PARAMS)
    if s:
        for tok in s.split(','):
            k, v = tok.split('=')
            prm[k] = float(v)
    prm['N'] = int(prm['N'])
    return prm


def select_events(arm, first, events_file):
    evs = arm_events(arm)
    if events_file:
        want = {l.split()[0] for l in open(events_file) if l.strip()}
        evs = [e for e in evs if e[0] in want]
    if first:
        evs = evs[:first]
    return evs


def cmd_features(args):
    prm = parse_params(args.params)
    cols = ['evt', 'call', 'j', 'k', 'nedge', 'Lmin', 'Lmax', 'Tmax', 'npmin',
            'angle', 'gw', 'wdeadX', 'dis', 'dens', 'dvtx', 'verdict', 'conf',
            'rule']
    print('\t'.join(cols))
    for arm in args.arm:
        for evt, path in select_events(arm, args.first, args.events_file):
            for pk, p in sorted(pair_table(arm, evt, path).items()):
                v, _, rule, conf = classify(p, prm)
                print('\t'.join(str(x) for x in [
                    p['evt'], p['call'], p['j'], p['k'], len(p['edges']),
                    '%.2f' % p['Lmin'], '%.2f' % p['Lmax'], '%.2f' % p['Tmax'],
                    p['npmin'], '%.1f' % (p['angle'] if p['angle'] is not None else -1),
                    int(p['gw']), p['wdeadX'], '%.2f' % p['dis'],
                    '%.0f' % p['dens'], '%.1f' % p['dvtx'], v, conf, rule]))


def labels_path(tag, evt):
    d = os.path.join(SBND, 'overclustering_labels', tag)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'labels-evt%s.json' % evt)


def cmd_label(args):
    prm = parse_params(args.params)
    overrides = {}
    if args.overrides and os.path.isfile(args.overrides):
        overrides = json.load(open(args.overrides))
    n_ev = n_pair = n_edge = 0
    tally = collections.Counter()
    for arm in args.arm:
        for evt, path in select_events(arm, args.first, args.events_file):
            pairs = pair_table(arm, evt, path)
            if not pairs:
                continue
            entries = {}
            for pk, p in sorted(pairs.items()):
                pid = '%s|%d|%d|%d' % (evt, p['call'], p['j'], p['k'])
                v, cause, rule, conf = classify(p, prm)
                note = ''
                if pid in overrides:
                    ov = overrides[pid]
                    note = ' | visual override: %s (was %s)' % (
                        ov.get('why', ''), v)
                    v = ov['verdict']
                    cause = ov.get('cause', cause)
                severed = all(r['killed'] for r in p['edges'])
                comment = ('auto %s conf=%s [%s] Lmin=%.1f Lmax=%.1f Tmax=%.1f npmin=%d '
                           'ang=%.0f gapW=%d wdeadX=%d dis=%.2f dens=%.0f '
                           'dvtx=%.1f nedge=%d severed=%d%s') % (
                    v, conf, rule, p['Lmin'], p['Lmax'], p['Tmax'], p['npmin'],
                    p['angle'] if p['angle'] is not None else -1, int(p['gw']),
                    p['wdeadX'], p['dis'], p['dens'], p['dvtx'],
                    len(p['edges']), int(severed), note)
                for r in p['edges']:
                    entries[edge_key(evt, r)] = dict(
                        verdict=v, cause=cause, comment=comment,
                        blk=r['blk'], j=r['j'], k=r['k'], dis=r['dis'],
                        killed=r['killed'], p1=r['p1'], p2=r['p2'],
                        near_miss='', score=0.0)
                    n_edge += 1
                tally[v] += 1
                n_pair += 1
            out = labels_path(args.tag, evt)
            data = {}
            if os.path.isfile(out):
                try:
                    data = json.load(open(out))
                except ValueError:
                    data = {}
            data['event'] = 'evt%s' % evt
            data['source'] = os.path.basename(path)
            data.setdefault('labels', {}).update(entries)
            with open(out, 'w') as fh:
                json.dump(data, fh, indent=1, sort_keys=True)
            n_ev += 1
    print('wrote %d events, %d pairs, %d edge labels -> overclustering_labels/%s/'
          % (n_ev, n_pair, n_edge, args.tag))
    print('pair verdicts: %s' % dict(tally))


def cmd_verify(args):
    keys = {}
    for arm in args.arm:
        for evt, path in arm_events(arm):
            _, edges = load_event(path)
            for r in edges:
                keys[edge_key(evt, r)] = (evt, r['graph_call'],
                                          min(r['j'], r['k']), max(r['j'], r['k']))
    orphans = 0
    bypair = collections.defaultdict(set)
    n = 0
    for f in sorted(glob.glob(os.path.join(SBND, 'overclustering_labels',
                                           args.tag, 'labels-evt*.json'))):
        for k, v in json.load(open(f))['labels'].items():
            n += 1
            if k not in keys:
                orphans += 1
                if orphans <= 5:
                    print('  ORPHAN %s' % k)
            else:
                bypair[keys[k]].add(v['verdict'])
    split = {p: s for p, s in bypair.items() if len(s) > 1}
    print('labels: %d   orphans: %d   pairs: %d   pairs with mixed verdicts: %d'
          % (n, orphans, len(bypair), len(split)))
    for p, s in list(split.items())[:5]:
        print('  MIXED %s %s' % (p, s))
    if orphans or split:
        sys.exit(1)
    print('PASS: every label resolves to a real edge and every pair is '
          'self-consistent.')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('calibrate', 'features', 'label', 'verify'):
        s = sub.add_parser(name)
        s.add_argument('--arm', action='append', required=True)
        s.add_argument('--labels', default=os.path.join(SBND, 'overclustering_labels'))
        s.add_argument('--tag', default='claude-scan50')
        s.add_argument('--first', type=int, default=0)
        s.add_argument('--events-file', default='')
        s.add_argument('--params', default='')
        s.add_argument('--overrides', default='')
    args = ap.parse_args()
    {'calibrate': cmd_calibrate, 'features': cmd_features,
     'label': cmd_label, 'verify': cmd_verify}[args.cmd](args)


if __name__ == '__main__':
    main()
