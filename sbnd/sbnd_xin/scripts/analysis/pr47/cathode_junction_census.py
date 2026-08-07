#!/usr/bin/env python3
"""doc pr/47: offline census of cathode-crossing PR segments.

Owner question: event 18255-52085 has its TRUE vertex inside the cathode
(near x=0), and the reconstruction shows a 137.6 cm single segment sailing
straight through it with a real ~35 deg kink that never gets recognized.
This script finds every cathode-crossing segment in the existing 1000-event
production sample (work-mcp1k-cb0805) and characterizes the junction with
two distortion-tolerant statistics, so the doc pr/47 options section has a
measured cost/benefit instead of a guess.

Reads the ALREADY-PRODUCED calib-pr-evt<ID>.json PR-display dumps in
work-mcp1k-cb0805/pr_evt<ID>/ (445/1000 events have one -- the rest were
produced by an earlier sweep that did not request the -stm-style calib dump;
see doc pr/47 sec 5 for the coverage caveat). No new runs, no rebuild.

For every segment with fit points on both sides of x=0 ("a cathode crosser"):
  - the crossing index (first sign change in x along the point sequence)
  - arm lengths (arclength) on each side of the crossing
  - whether the segment sits on the neutrino-candidate (main) cluster
  - distance from the crossing point to the event's main_vertex
  - the SKIRT-EXCLUDED PCA TURN ANGLE between the two arms, at several
    (skirt, baseline) settings -- this is the distortion-tolerant substitute
    for segment_search_kink's index-windowed refl_angle
  - the dQ/dx ASYMMETRY between the two arms (median over 10 cm adjacent to
    the crossing, skirt-excluded, in MIP units; MIP scale = 43000 e/cm =
    NeutrinoPatternBase.h:120 m_mip_dqdx_median default)
  - a REPLAY of segment_search_kink's four accept criteria
    (PRSegmentFunctions.cxx:348-361) at the crossing index, over the WHOLE
    segment (index-windowed refl/para/sum, exactly as shipped) -- this is
    what the doc calls the "narrow/noise-sensitive" statistic, printed
    alongside the wide/stable one for direct comparison.

CAVEATS (also stated in the doc):
  - calib-pr-evt*.json is the FINAL fitted point cloud (after the whole PR
    chain, including any multi-tracking refit), not necessarily what
    segment_search_kink saw at break time. doc pr/47 Phase 1 cross-checked
    52085 itself against a break-time diagnostic and found the same
    qualitative miss (see doc sec 3) -- but for the other events in this
    census that check was NOT repeated; treat the accept-criteria replay as
    a proxy, per the identical caveat in kink_probe.py (evt 169824, pr/20).
  - flag_check (PRSegmentFunctions.cxx:287, "seen the walk's own test point
    within 0.1 cm") is a STATEFUL walk condition this offline replay cannot
    reconstruct; the criteria replay here evaluates every index past the
    1 cm front/back/start skip, which is a superset of what the real walk
    evaluates. A crosser that "passes" here may not have been reached by the
    real walk at all -- so this census answers "would the shipped test ever
    accept this junction ANYwhere in the segment", not "did it".
  - Sample is MC. Data cathode-region transverse distortion runs
    ~1.2-1.4 cm vs MC's ~0.35-0.48 cm (docs/14, docs/18) -- separation
    measured here is optimistic for data.

Usage:
    python3 cathode_junction_census.py [--root work-mcp1k-cb0805] [--out OUT.tsv]
"""
import argparse
import glob
import json
import math
import os

import numpy as np

MIP_DQDX = 43000.0  # e/cm, NeutrinoPatternBase.h:120 m_mip_dqdx_median default
SKIRT_CM = 3.0       # default skirt for the headline turn-angle number
BASELINE_CM = 15.0   # default baseline for the headline turn-angle number


def seg_points(seg):
    P = np.array([[p['x'], p['y'], p['z']] for p in seg['points']], float)
    dQ = np.array([p['dQ'] for p in seg['points']], float)
    dx = np.array([p['dx'] for p in seg['points']], float)
    dqdx = np.where(dx > 1e-9, dQ / np.maximum(dx, 1e-9), 0.0)
    return P, dqdx


def arclen(P):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def find_crossing(P):
    """First index i such that P[i].x and P[i+1].x have opposite sign
    (or one is exactly 0). Returns i, or None if the segment never crosses."""
    x = P[:, 0]
    for i in range(len(x) - 1):
        if (x[i] <= 0 and x[i + 1] > 0) or (x[i] >= 0 and x[i + 1] < 0):
            return i
    return None


def pca_dir(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    return vt[0]


def skirt_turn_angle(P, cum, i0, skirt, L):
    """PCA direction of each arm over arclength window (skirt, skirt+L] from
    the crossing, angle between the two (oriented away from the crossing).
    Returns None if either arm has < 3 points in the window."""
    x = P[:, 0]
    neg_idx = [k for k in range(0, i0 + 1)
               if x[k] <= 0 and skirt <= (cum[i0] - cum[k]) <= skirt + L]
    pos_idx = [k for k in range(i0 + 1, len(P))
               if x[k] > 0 and skirt <= (cum[k] - cum[i0]) <= skirt + L]
    if len(neg_idx) < 3 or len(pos_idx) < 3:
        return None, len(neg_idx), len(pos_idx)
    va = pca_dir(P[neg_idx])
    vb = pca_dir(P[pos_idx])
    # Both index lists are in increasing-k (increasing arclength / direction
    # of travel) order, so orienting each PCA vector along its OWN arm's
    # increasing-k direction gives two vectors in a common "direction of
    # travel" convention -- the angle between them (no extra negation) is
    # the turn angle: ~0 deg for a straight through-going track.
    if np.dot(P[neg_idx[-1]] - P[neg_idx[0]], va) < 0:
        va = -va
    if np.dot(P[pos_idx[-1]] - P[pos_idx[0]], vb) < 0:
        vb = -vb
    ang = math.degrees(math.acos(max(-1, min(1, float(np.dot(va, vb))))))
    return ang, len(neg_idx), len(pos_idx)


def arm_dqdx_mip(P, dqdx, cum, i0, skirt, L, side):
    x = P[:, 0]
    if side == 'neg':
        idx = [k for k in range(0, i0 + 1)
               if x[k] <= 0 and skirt <= (cum[i0] - cum[k]) <= skirt + L]
    else:
        idx = [k for k in range(i0 + 1, len(P))
               if x[k] > 0 and skirt <= (cum[k] - cum[i0]) <= skirt + L]
    if not idx:
        return None
    return float(np.median(dqdx[idx])) / MIP_DQDX


def replay_kink_criteria(P, dqdx, i0):
    """Exact PRSegmentFunctions.cxx:211-361 arithmetic (index-windowed,
    min-over-scale refl/para, +-2 sum windows), evaluated at every index --
    flag_check and the cathode veto are NOT modeled (see module docstring).
    Returns the best (highest-scoring) index near the crossing and which of
    the four criteria it satisfies."""
    n = len(P)
    drift = np.array([1.0, 0.0, 0.0])
    refl = np.zeros(n)
    para = np.zeros(n)
    for i in range(n):
        a1 = a2 = None
        for j in range(6):
            off = (j + 1) * 2
            v10 = P[i] - (P[i - off] if i >= off else P[0])
            v20 = (P[i + off] if i + off < n else P[-1]) - P[i]
            m1, m2 = np.linalg.norm(v10), np.linalg.norm(v20)
            if j == 0:
                a1 = (math.degrees(math.acos(np.clip(np.dot(v10, v20) / (m1 * m2), -1, 1)))
                      if m1 * m2 > 0 else 0.0)
                d1 = math.degrees(math.acos(np.clip(np.dot(v10, drift) / m1, -1, 1))) if m1 > 0 else 90.
                d2 = math.degrees(math.acos(np.clip(np.dot(v20, drift) / m2, -1, 1))) if m2 > 0 else 90.
                a2 = max(abs(d1 - 90), abs(d2 - 90))
            elif m1 > 0 and m2 > 0:
                c = math.degrees(math.acos(np.clip(np.dot(v10, v20) / (m1 * m2), -1, 1)))
                a1 = min(a1, c)
                d1 = math.degrees(math.acos(np.clip(np.dot(v10, drift) / m1, -1, 1)))
                d2 = math.degrees(math.acos(np.clip(np.dot(v20, drift) / m2, -1, 1)))
                a2 = min(a2, max(abs(d1 - 90), abs(d2 - 90)))
        refl[i] = a1 if a1 is not None else 0.0
        para[i] = a2 if a2 is not None else 90.0

    best = None
    win = range(max(0, i0 - 15), min(n, i0 + 15))
    for i in win:
        sa = sa1 = 0.0
        ns = ns1 = 0
        lo, hi = max(0, i - 2), min(n, i + 3)
        for k in range(lo, hi):
            if para[k] > 10:
                sa += refl[k] ** 2
                ns += 1
            if para[k] > 7.5:
                sa1 += refl[k] ** 2
                ns1 += 1
        sa = math.sqrt(sa / ns) if ns else 0.0
        sa1 = math.sqrt(sa1 / ns1) if ns1 else 0.0
        vals = dqdx[lo:hi]
        ave = float(np.mean(vals)) if len(vals) else 0.0
        mx = float(np.max(vals)) if len(vals) else 0.0
        c1 = para[i] > 10 and refl[i] > 30 and sa > 15
        c2 = para[i] > 7.5 and refl[i] > 45 and sa1 > 25
        c3 = para[i] > 15 and refl[i] > 27 and sa > 12.5
        c4 = (para[i] > 15 and refl[i] > 22 and sa > 19 and
              mx > MIP_DQDX * 1.5 and ave > MIP_DQDX)
        fired = c1 or c2 or c3 or c4
        rec = dict(i=i, refl=refl[i], para=para[i], sum=sa, sum1=sa1,
                   ave_mip=ave / MIP_DQDX, max_mip=mx / MIP_DQDX,
                   c1=c1, c2=c2, c3=c3, c4=c4, fired=fired,
                   # margin to the nearest-missed criterion, for a "how close" stat
                   c4_sum_margin=sa - 19.0)
        if best is None or rec['fired'] and not best['fired']:
            best = rec
        elif rec['fired'] == best['fired'] and rec['refl'] > best['refl']:
            best = rec
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='work-mcp1k-cb0805')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sbnd_xin = os.path.abspath(os.path.join(here, '..', '..', '..'))
    root = os.path.join(sbnd_xin, args.root)

    dumps = sorted(glob.glob(os.path.join(root, 'pr_evt*', 'calib-pr-evt*.json')))
    rows = []
    n_events_with_dump = 0
    n_events_total = len(sorted(glob.glob(os.path.join(root, 'pr_evt*'))))
    for path in dumps:
        n_events_with_dump += 1
        evt = int(os.path.basename(path).replace('calib-pr-evt', '').replace('.json', ''))
        d = json.load(open(path))
        mv = d.get('main_vertex')
        for seg in d['segments']:
            P, dqdx = seg_points(seg)
            if len(P) < 8:
                continue
            i0 = find_crossing(P)
            if i0 is None:
                continue
            cum = arclen(P)
            neg_len = cum[i0]
            pos_len = cum[-1] - cum[i0]
            crossing_pt = P[i0]
            ang_hl, nneg_hl, npos_hl = skirt_turn_angle(P, cum, i0, SKIRT_CM, BASELINE_CM)
            stability = {}
            for sk in (0.0, 1.5, 3.0, 5.0):
                for L in (10.0, 15.0, 20.0, 30.0):
                    a, _, _ = skirt_turn_angle(P, cum, i0, sk, L)
                    stability[(sk, L)] = a
            dqdx_neg = arm_dqdx_mip(P, dqdx, cum, i0, SKIRT_CM, 10.0, 'neg')
            dqdx_pos = arm_dqdx_mip(P, dqdx, cum, i0, SKIRT_CM, 10.0, 'pos')
            crit = replay_kink_criteria(P, dqdx, i0)
            dist_to_mv = None
            if mv is not None:
                dist_to_mv = float(np.linalg.norm(
                    crossing_pt - np.array([mv['x'], mv['y'], mv['z']])))
            rows.append(dict(
                evt=evt, seg_id=seg['id'], cluster_id=seg['cluster_id'],
                is_main=bool(seg.get('is_main_cluster')),
                npts=len(P), neg_len_cm=neg_len, pos_len_cm=pos_len,
                x_cross=float(crossing_pt[0]),
                turn_ang_skirt3_L15=ang_hl, turn_stability=stability,
                dqdx_mip_neg=dqdx_neg, dqdx_mip_pos=dqdx_pos,
                dist_to_main_vertex_cm=dist_to_mv,
                kink_replay=crit,
            ))

    out = args.out or os.path.join(here, 'census_output.tsv')
    with open(out, 'w') as f:
        f.write('evt\tseg_id\tcluster_id\tis_main\tnpts\tneg_len_cm\tpos_len_cm\t'
                'x_cross\tturn_skirt3_L15\tdqdx_mip_neg\tdqdx_mip_pos\t'
                'dist_to_main_vertex_cm\tcrit_fired\tcrit_which\tcrit_best_refl\t'
                'crit_best_sum\n')
        for r in rows:
            crit = r['kink_replay']
            which = ''.join(k for k in 'c1234' if k != '4' and crit.get('c' + k))
            which = ''.join([n for n, ok in
                              (('C1', crit['c1']), ('C2', crit['c2']),
                               ('C3', crit['c3']), ('C4', crit['c4'])) if ok]) or '-'
            f.write('%d\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%.3f\t%s\t%s\t%s\t%s\t%s\t%s\t%.2f\t%.2f\n' % (
                r['evt'], r['seg_id'], r['cluster_id'], int(r['is_main']), r['npts'],
                r['neg_len_cm'], r['pos_len_cm'], r['x_cross'],
                ('%.1f' % r['turn_ang_skirt3_L15']) if r['turn_ang_skirt3_L15'] is not None else 'nan',
                ('%.2f' % r['dqdx_mip_neg']) if r['dqdx_mip_neg'] is not None else 'nan',
                ('%.2f' % r['dqdx_mip_pos']) if r['dqdx_mip_pos'] is not None else 'nan',
                ('%.1f' % r['dist_to_main_vertex_cm']) if r['dist_to_main_vertex_cm'] is not None else 'nan',
                int(crit['fired']), which, crit['refl'], crit['sum'],
            ))

    ang = np.array([r['turn_ang_skirt3_L15'] for r in rows if r['turn_ang_skirt3_L15'] is not None])
    asym = np.array([abs((r['dqdx_mip_neg'] or 0) - (r['dqdx_mip_pos'] or 0)) for r in rows
                      if r['dqdx_mip_neg'] is not None and r['dqdx_mip_pos'] is not None])
    fired = sum(1 for r in rows if r['kink_replay']['fired'])

    print('coverage: %d/%d events have a calib-pr dump' % (n_events_with_dump, n_events_total))
    print('cathode-crossing segments found: %d' % len(rows))
    print('  on the neutrino-candidate (main) cluster: %d' %
          sum(1 for r in rows if r['is_main']))
    print('  segment_search_kink criteria fire SOMEWHERE in the segment (proxy, see caveats): %d/%d'
          % (fired, len(rows)))
    if len(ang):
        print('turn angle (skirt=3cm, L=15cm) distribution:')
        edges = [0, 2, 5, 10, 15, 20, 30, 45, 60, 90, 180]
        hist, _ = np.histogram(ang, bins=edges)
        for lo, hi, c in zip(edges[:-1], edges[1:], hist):
            print('  [%3d,%3d) deg: %d' % (lo, hi, c))
        print('  median %.1f  p90 %.1f  max %.1f' % (np.median(ang), np.percentile(ang, 90), ang.max()))
    print('wrote %s' % out)

    tail = sorted([r for r in rows if (r['turn_ang_skirt3_L15'] or 0) >= 20],
                   key=lambda r: -(r['turn_ang_skirt3_L15'] or 0))
    print('\nturn angle >= 20 deg (tail, for Bee eyeballing):')
    for r in tail:
        print('  evt %-8d seg %-6d turn=%.1f  dqdx_mip(neg,pos)=(%s,%s)  fired=%s  dist_to_mv=%s' % (
            r['evt'], r['seg_id'], r['turn_ang_skirt3_L15'],
            ('%.2f' % r['dqdx_mip_neg']) if r['dqdx_mip_neg'] is not None else 'nan',
            ('%.2f' % r['dqdx_mip_pos']) if r['dqdx_mip_pos'] is not None else 'nan',
            r['kink_replay']['fired'],
            ('%.1f' % r['dist_to_main_vertex_cm']) if r['dist_to_main_vertex_cm'] is not None else 'nan'))


if __name__ == '__main__':
    main()
