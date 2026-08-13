#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/74 round 1 -- prevalence census of the FOUR PF-tree
shapes the owner's four track/shower cases exhibit, over an arbitrary set of
already-archived arms.  Read-only: everything comes from `0-mc.json` (the
particle-flow tree) and `0-shower_track-global.json` inside each event's
mabc-pr.zip.  No wire-cell run, no knob.

What is measured, and what is NOT
---------------------------------
This counts PF-TREE STRUCTURE, which is a real structural signal.  It does NOT
count classifier-flag firings: the Bee paint takes shower MEMBERSHIP first and
falls back to per-segment flags, so "painted track" carries no information
about whether kShowerTopology/kShowerTrajectory ever fired.  Numbers here are
therefore labelled *PF-structure prevalence*, never *mechanism footprint*.
Mechanism footprint only comes from an A/B arm.

Shapes (one column each):
  A  muon/pion trunk with an EM child  -- a PF node with |pdg| in {13,211}
     hanging directly off the PF root, whose subtree contains an e-/gamma node
     of at least --em-mev MeV.  This is the 90055 / 469665 shape: the EM
     shower is parented by a track trunk instead of starting at the vertex.
  B  long PENCIL electron              -- an `e-` PF node whose |start-end| is
     at least --long-cm AND whose painted points have a transverse RMS below
     --trans-cm.  The 53361 shape (a 119 cm e- that is a pencil, i.e. a muon).
     The length cut alone is NOT discriminating -- a real EM shower routinely
     exceeds 50 cm (61% of events have one); the transverse cut is what
     separates a MIP track from a shower.
  C  root gamma fan-out                -- number of root-level `gamma` nodes;
     reported when it is at least --fan.  The 469665 fragmentation shape.
  D  painted-but-PF-absent object      -- an object painted in shower_track
     with at least --orphan-pts points whose id appears nowhere in the PF
     tree.  The 142421 shape (seg 7013, 2975 points, 266 MeV, invisible).

Usage:
  pr74_census.py <arm> [<arm> ...] [--em-mev 50] [--long-cm 50]
                 [--fan 3] [--orphan-pts 200] [--tsv out.tsv]

Example (the doc's numbers):
  python3 scripts/analysis/pr74/pr74_census.py \
      work-pr51r7-on48 work-pr51r7-on19 work-pr51r7-on50 \
      --tsv docs/pr/74_pf_shape_census.tsv
"""
import argparse
import glob
import json
import os
import re
import sys
import zipfile

import numpy as np

SB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

PDG_RE = re.compile(r'^\s*(\S+)\s+(-?[\d.]+)\s*MeV')


def parse_text(text):
    """'mu-  347 MeV' -> ('mu-', 347.0).  Returns (name, mev) or (text, 0)."""
    m = PDG_RE.match(text or '')
    if not m:
        return (text or '', 0.0)
    return (m.group(1), float(m.group(2)))


def straight_len(node):
    d = node.get('data', {}) or {}
    s, e = d.get('start'), d.get('end')
    if not s or not e:
        return 0.0
    return float(np.linalg.norm(np.asarray(s) - np.asarray(e)))


def subtree(node):
    yield node
    for c in node.get('children', []) or []:
        yield from subtree(c)


def load_json_member(zp, suffix):
    with zipfile.ZipFile(zp) as z:
        hits = [n for n in z.namelist() if n.endswith(suffix)]
        if not hits:
            return None
        with z.open(sorted(hits)[0]) as fh:
            return json.load(fh)


TRACK_NAMES = ('mu-', 'mu+', 'pi-', 'pi+')
EM_NAMES = ('e-', 'e+', 'gamma')


def transverse_rms_by_rcid(st):
    """RMS distance of each painted object's points from its own leading PCA
    axis, in cm, keyed by str(real_cluster_id).  A MIP track is a pencil
    (<~0.6 cm); an EM shower fans out (several cm).  This is the measure that
    separates the 53361 shape from a genuine long shower."""
    out = {}
    if not st or not st.get('real_cluster_id'):
        return out
    p = np.stack([st['x'], st['y'], st['z']], axis=1)
    rcid = np.asarray(st['real_cluster_id'])
    for r in set(rcid.tolist()):
        pts = p[rcid == r]
        if len(pts) < 10:
            continue
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = pts - c
        along = d @ vt[0]
        perp2 = (d * d).sum(axis=1) - along * along
        out[str(int(r))] = float(np.sqrt(max(perp2.mean(), 0.0)))
    return out


def analyse(zp, args):
    pf = load_json_member(zp, '-mc.json')
    if pf is None:
        return None
    row = dict(A=0, B=0, C=0, D=0, a_ids=[], b_ids=[], d_ids=[])

    # A: track trunk off the PF root carrying an EM descendant
    for root in pf:
        name, _ = parse_text(root.get('text'))
        if name in TRACK_NAMES:
            for n in subtree(root):
                if n is root:
                    continue
                cname, cmev = parse_text(n.get('text'))
                if cname in EM_NAMES and cmev >= args.em_mev:
                    row['A'] += 1
                    row['a_ids'].append('%s:%s>%s:%.0fMeV' %
                                        (root.get('id'), name, cname, cmev))
                    break

    st = load_json_member(zp, '-shower_track-global.json')

    # B: long PENCIL electron -- length AND transverse RMS of its painted points
    trans = transverse_rms_by_rcid(st) if st is not None else {}
    for root in pf:
        for n in subtree(root):
            nname, _ = parse_text(n.get('text'))
            if nname not in ('e-', 'e+') or straight_len(n) < args.long_cm:
                continue
            t = trans.get(str(n.get('id')))
            if t is None or t > args.trans_cm:
                continue
            row['B'] += 1
            row['b_ids'].append('%s:%.0fcm/rt%.2fcm' %
                                (n.get('id'), straight_len(n), t))

    # C: root-level gamma fan-out
    nfan = sum(1 for r in pf if parse_text(r.get('text'))[0] == 'gamma')
    row['C'] = nfan if nfan >= args.fan else 0

    # D: painted object with no PF node of that id
    if st is not None and st.get('real_cluster_id'):
        pf_ids = set(str(n.get('id')) for r in pf for n in subtree(r))
        rcid = np.asarray(st['real_cluster_id'])
        for r in sorted(set(rcid.tolist())):
            n = int((rcid == r).sum())
            if n < args.orphan_pts:
                continue
            if str(r) not in pf_ids:
                row['D'] += 1
                row['d_ids'].append('%d:%dpts' % (r, n))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arms', nargs='+')
    ap.add_argument('--em-mev', type=float, default=50.0)
    ap.add_argument('--long-cm', type=float, default=50.0)
    ap.add_argument('--trans-cm', type=float, default=1.0,
                    help='shape B: max transverse RMS (cm) of the painted points')
    ap.add_argument('--fan', type=int, default=3)
    ap.add_argument('--orphan-pts', type=int, default=200)
    ap.add_argument('--tsv')
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        for zp in sorted(glob.glob(os.path.join(SB, arm, 'pr_evt*', 'mabc-pr.zip'))):
            evt = int(os.path.basename(os.path.dirname(zp))[6:])
            try:
                r = analyse(zp, args)
            except Exception as ex:                       # noqa: BLE001
                print('  [skip] %s evt %d: %s' % (arm, evt, ex), file=sys.stderr)
                continue
            if r is None:
                continue
            r['arm'], r['evt'] = arm, evt
            rows.append(r)

    n = len(rows)
    print('events analysed: %d  (arms: %s)' % (n, ', '.join(args.arms)))
    print()
    print('shape                                          events   (of %d)' % n)
    for key, label in (('A', 'A  mu/pi trunk carrying an EM child >=%.0f MeV' % args.em_mev),
                       ('B', 'B  pencil e- >= %.0f cm (trans RMS < %.1f cm)'
                             % (args.long_cm, args.trans_cm)),
                       ('C', 'C  >= %d root-level gamma nodes' % args.fan),
                       ('D', 'D  painted object >=%d pts absent from PF' % args.orphan_pts)):
        hit = sum(1 for r in rows if r[key])
        print('%-46s %4d     %5.1f%%' % (label, hit, 100.0 * hit / n if n else 0))

    print()
    print('per-event detail (only events with at least one shape):')
    print('%-20s %-9s %3s %3s %3s %3s  detail' % ('arm', 'evt', 'A', 'B', 'C', 'D'))
    for r in sorted(rows, key=lambda r: (-(r['A'] + r['B'] + r['D']), r['arm'], r['evt'])):
        if not (r['A'] or r['B'] or r['C'] or r['D']):
            continue
        det = ' '.join(r['a_ids'][:3] + r['b_ids'][:3] + r['d_ids'][:3])
        print('%-20s %-9d %3d %3d %3d %3d  %s' %
              (r['arm'], r['evt'], r['A'], r['B'], r['C'], r['D'], det))

    if args.tsv:
        with open(args.tsv, 'w') as fh:
            fh.write('arm\tevt\tA_track_trunk_em\tB_long_electron\t'
                     'C_root_gamma_fan\tD_pf_absent\tdetail\n')
            for r in sorted(rows, key=lambda r: (r['arm'], r['evt'])):
                det = ' '.join(r['a_ids'] + r['b_ids'] + r['d_ids'])
                fh.write('%s\t%d\t%d\t%d\t%d\t%d\t%s\n' %
                         (r['arm'], r['evt'], r['A'], r['B'], r['C'], r['D'], det))
        print('\nwrote %s' % args.tsv)


if __name__ == '__main__':
    sys.exit(main())
