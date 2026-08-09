#!/usr/bin/env python3
"""doc pr/53 round 7 Step 2: offline threshold scan for the S5 "3D-image
support" ghost predicate, against real data, before any C++ is written.

For every event with an existing WCT_RELAXED_EDGE_CENSUS rerun (the round-6
movers' work-pr53-cen48/cen19 arms, plus the round-7 owner events), this:

  1. Finds every fit-gap run (fitgap_exam.find_gap_runs) and routes it to an
     emitted OC53CENSUS-S closest-pair edge (fitgap_exam.route_run). Routed
     edges are the MUST-KILL set -- the owner's hand-scanned defect.
  2. Computes the same ghost-step metric (fitgap_exam.ghost_steps_along_edge)
     for EVERY OTHER emitted closest-pair edge in the same events (no fit-gap
     routed to it) -- these are the MUST-NOT-KILL background: edges the
     current, already-validated relaxed_strict flavor keeps, and no owner
     hand-scan flagged.
  3. Reports the distribution of n_ghost_unexcused (count) and its ratio to
     n_interior for both sets, so a threshold can be chosen where the
     must-kill set is separated from the must-not-kill background -- not
     guessed.
  4. 21073 is not scanned here (round-6 control: A/B share one closely-
     component, so protect_bundle emits no inter-component edge for it at
     all -- it is immune to this rule as it was to relaxed_strict, by
     construction, not by threshold).

Usage:
  threshold_scan.py <spec> [<spec> ...]
    <spec> := <arm>:<cen_arm>:<evt>[,<evt>...]
  e.g.
  threshold_scan.py work-pr53-on48a:work-pr53-cen48:388,10550,52672,111412,172230,\\
174637,214469,219295,235435,267597,268067,269774,271851,350186,400474,422851,447477,\\
469665,489330 work-pr53-on19a:work-pr53-cen19:71372,142421,259542,359980,399860,\\
463565,506746,521075
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from oc53_probe import Loader  # noqa: E402
from fitgap_exam import (load_image, load_fit, find_gap_runs, route_run,  # noqa: E402
                          emitted_closest_edges, ghost_steps_along_edge, SB)
from scipy.spatial import cKDTree  # noqa: E402


def scan_event(arm, cen_arm, evt, d0=1.0, min_run=2, route_tol=5.0, img_radius=1.0):
    P, rcid = load_image(arm, evt)
    F, cid, seg, src = load_fit(arm, evt)
    img_tree = cKDTree(P)
    runs = find_gap_runs(F, cid, seg, src, img_tree, d0=d0, min_run=min_run)
    census_edges = emitted_closest_edges(cen_arm, evt)

    routed_jk = set()
    for run in runs:
        best, best_d = route_run(run, census_edges, route_tol=route_tol)
        if best is not None:
            routed_jk.add((best['sig'], best['jk']))

    ld = Loader(os.path.join(SB, arm, 'pr_evt%s' % evt))
    kill_rows, bg_rows = [], []
    try:
        for e in census_edges:
            c = e.get('closest')
            if c is None or c['nsteps'] < 2:
                continue
            g = ghost_steps_along_edge(ld, c, img_radius=img_radius)
            row = dict(evt=evt, jk=e['jk'], sig=e['sig'], dis=c['dis'], nsteps=c['nsteps'],
                       nb1=c['nb1'], n_interior=g['n_interior'], n_ghost=g['n_ghost'],
                       n_ghost_unexcused=g['n_ghost_unexcused'], maxgap=g['maxgap'],
                       ratio=g['n_ghost_unexcused'] / max(g['n_interior'], 1))
            if (e['sig'], e['jk']) in routed_jk:
                kill_rows.append(row)
            else:
                bg_rows.append(row)
    finally:
        ld.cleanup()
    return kill_rows, bg_rows


def summarize(name, rows):
    if not rows:
        print('  %s: (empty)' % name)
        return
    n = np.array([r['n_ghost_unexcused'] for r in rows])
    ratio = np.array([r['ratio'] for r in rows])
    ni = np.array([r['n_interior'] for r in rows])
    print('  %s: n=%d  n_ghost_unexcused: min=%d p10=%.0f median=%.0f p90=%.0f max=%d'
          % (name, len(rows), n.min(), np.percentile(n, 10), np.median(n), np.percentile(n, 90), n.max()))
    print('  %s: ratio(unexcused/interior): min=%.2f p10=%.2f median=%.2f p90=%.2f max=%.2f  (n_interior median=%.0f)'
          % (name, ratio.min(), np.percentile(ratio, 10), np.median(ratio), np.percentile(ratio, 90), ratio.max(),
             np.median(ni)))


if __name__ == '__main__':
    all_kill, all_bg = [], []
    for spec in sys.argv[1:]:
        arm, cen_arm, evts = spec.split(':')
        for evt in evts.split(','):
            k, b = scan_event(arm, cen_arm, evt)
            all_kill += k
            all_bg += b
            print('evt %s: %d must-kill edge(s), %d background edge(s)' % (evt, len(k), len(b)))
            for r in k:
                print('    KILL  j,k=%s dis=%.2fcm nsteps=%d n_ghost_unexcused=%d/%d ratio=%.2f maxgap=%.1fcm'
                      % (r['jk'], r['dis'], r['nsteps'], r['n_ghost_unexcused'], r['n_interior'], r['ratio'], r['maxgap']))

    print('\n=== overall ===')
    summarize('MUST-KILL  (fit-gap routed)', all_kill)
    summarize('BACKGROUND (no fit-gap routed to them)', all_bg)

    # scan a few candidate operating points for count-based OR ratio-based kill
    print('\n=== candidate thresholds (kill if n_ghost_unexcused >= N) ===')
    for N in (1, 2, 3, 4, 5):
        tp = sum(1 for r in all_kill if r['n_ghost_unexcused'] >= N)
        fp = sum(1 for r in all_bg if r['n_ghost_unexcused'] >= N)
        print('  N=%d: kills %d/%d must-kill (%.0f%%), %d/%d background false-positive (%.0f%%)'
              % (N, tp, len(all_kill), 100.0 * tp / max(len(all_kill), 1),
                 fp, len(all_bg), 100.0 * fp / max(len(all_bg), 1)))

    print('\n=== candidate thresholds (kill if ratio >= R) ===')
    for R in (0.15, 0.25, 0.35, 0.5, 0.75):
        tp = sum(1 for r in all_kill if r['ratio'] >= R)
        fp = sum(1 for r in all_bg if r['ratio'] >= R)
        print('  R=%.2f: kills %d/%d must-kill (%.0f%%), %d/%d background false-positive (%.0f%%)'
              % (R, tp, len(all_kill), 100.0 * tp / max(len(all_kill), 1),
                 fp, len(all_bg), 100.0 * fp / max(len(all_bg), 1)))
