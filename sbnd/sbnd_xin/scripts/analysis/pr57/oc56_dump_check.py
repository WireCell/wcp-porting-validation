#!/usr/bin/env python3
"""doc pr/57: verify the overclustering-separation scan dump does not lie.

The dump (clus/src/connect_graph_relaxed_strict.cxx, WCT_OC56_SCAN_DUMP) is
what the hand-scan display (overclustering_display/) shows a human. If the
dumped fired/dead cells don't actually reproduce the algorithm's own
gap_u/gap_v/gap_w verdict, the display is worthless -- an operator could
label a "removal" as wrong while looking at pixels that never determined the
removal. This script replays the SAME bounded Chebyshev-box BFS the C++
`s6_planes_connected` runs, entirely from the dumped fired/dead cells and
seed lists, at every logged (dw, ds) in the round-3 scan matrix (not just the
shipped operating point), and asserts the replayed connectivity bit matches
the logged matrix bit -- for every plane of every S6-evaluated edge.

It also carries `long_track_break_score()`, the Python-side "is this really
one long track that got broken in two" heuristic from doc pr/56 round 3's
plan (both pieces >10 cm by PCA length, axes within 20 degrees, facing
endpoints close) -- imported by the viewer so revising the metric never
costs a rerun, and computed here once so this file is the single definition
both consume.

Usage:
    oc56_dump_check.py <arm_dir> [<arm_dir> ...]
      # replays every oc56scan-evt*.jsonl found under each arm_dir's
      # pr_evt*/ subdirectories and reports pass/fail counts.

    oc56_dump_check.py <arm_dir> --max-mismatches 20
      # print up to N mismatch details (default 10); 0 = summary only.

Exit code is nonzero if any mismatch was found (for use in a gate script).
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oc56_conn  # noqa: E402  (doc pr/57 round 4)

try:
    import numpy as np
except ImportError:
    np = None


# ---------------------------------------------------------------------------
# BFS replay -- mirrors s6_planes_connected() in
# clus/src/connect_graph_relaxed_strict.cxx line for line. Keep the two in
# sync; if you change one, change both and rerun this script.
# ---------------------------------------------------------------------------

CELL_BUDGET = 20000  # must match cell_budget in the C++


def replay_connected(fired, dead_ranges, seeds_a, seeds_b, win, slice_step, dw, ds):
    """fired: set of (wind, slice). dead_ranges: dict wind -> list of
    (slo, shi) inclusive intervals. seeds_a/seeds_b: list of [wind, slice].
    win: [wlo, whi, slo, shi]. Returns (connected, budget_hit)."""
    wlo, whi, slo, shi = win

    def dead(wind, slice_):
        for lo, hi in dead_ranges.get(wind, ()):
            if lo <= slice_ <= hi:
                return True
        return False

    def passable(wind, slice_):
        return (wind, slice_) in fired or dead(wind, slice_)

    if not seeds_a or not seeds_b:
        return False, False

    target = {tuple(s) for s in seeds_b}
    steps = [(a, b * slice_step)
             for a in range(-dw, dw + 1)
             for b in range(-ds, ds + 1)
             if not (a == 0 and b == 0)]

    visited = set()
    frontier = []
    for s in seeds_a:
        t = tuple(s)
        if t in target:
            return True, False
        if t not in visited:
            visited.add(t)
            frontier.append(t)

    while frontier:
        if len(visited) > CELL_BUDGET:
            return False, True  # circuit breaker: fails closed, same as C++
        nxt = []
        for (w, sl) in frontier:
            for (dw_, ds_) in steps:
                nb = (w + dw_, sl + ds_)
                if nb[0] < wlo or nb[0] > whi:
                    continue
                if nb[1] < slo or nb[1] > shi:
                    continue
                if nb in visited:
                    continue
                if nb in target:
                    return True, False
                if not passable(*nb):
                    continue
                visited.add(nb)
                nxt.append(nb)
        frontier = nxt
    return False, False


def check_edge(rec, max_report=3):
    """Returns list of mismatch dicts (empty if this edge's dump is fully
    self-consistent with its own logged matrix)."""
    mismatches = []
    slice_step = rec['slice_step']
    for p in rec['planes']:
        plane = p['plane']
        if p.get('native_step', 1) != 1:
            continue  # non-exhaustive dump for this plane; not checkable, see header
        fired = {tuple(c) for c in p['fired']}
        dead_ranges = {}
        for wind, lo, hi in p['dead']:
            dead_ranges.setdefault(wind, []).append((lo, hi))
        matrix = rec['matrix'][plane]
        if len(matrix) != 16:
            continue  # census/matrix wasn't populated for this run
        for mdw in range(1, 5):
            for mds in range(1, 5):
                expect = matrix[(mdw - 1) * 4 + (mds - 1)] == '1'
                got, budget_hit = replay_connected(
                    fired, dead_ranges, p['seeds_a'], p['seeds_b'],
                    p['win'], slice_step, mdw, mds)
                if got != expect:
                    if len(mismatches) < max_report:
                        mismatches.append(dict(
                            blk=rec['blk'], j=rec['j'], k=rec['k'], plane=plane,
                            dw=mdw, ds=mds, expected=expect, got=got,
                            budget_hit=budget_hit, log=rec.get('_log')))
                    else:
                        mismatches.append(None)  # counted but not detailed
    return mismatches


# ---------------------------------------------------------------------------
# Long-track-break scorer -- shared by the viewer's default edge-list sort.
# ---------------------------------------------------------------------------

def pca_length_axis(points_cm):
    """points_cm: list of [x,y,z] in cm. Returns (length_cm, unit_axis) or
    (0.0, None) if too few points to define an axis."""
    if np is None or len(points_cm) < 2:
        return 0.0, None
    pts = np.asarray(points_cm, dtype=float)
    c = pts.mean(axis=0)
    pts0 = pts - c
    cov = pts0.T @ pts0
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmax(w))]
    proj = pts0 @ axis
    length = float(proj.max() - proj.min())
    return length, axis


def angle_between_deg(a, b):
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    cosang = abs(float(np.dot(a, b)) / (na * nb))
    cosang = min(1.0, max(-1.0, cosang))
    return math.degrees(math.acos(cosang))


def long_track_break_score(points_a_cm, points_b_cm, edge_dis_cm,
                            length_floor_cm=10.0, angle_ceiling_deg=20.0):
    """doc pr/56 round 3 plan definition: both pieces long (PCA length
    > length_floor_cm), their axes nearly collinear (< angle_ceiling_deg),
    facing endpoints close (edge_dis_cm). Higher score = more likely this
    removal broke one real long track in two; this is a SORT KEY for the
    hand-scan, not a verdict -- the human still labels it. Returns
    (score, detail_dict)."""
    len_a, ax_a = pca_length_axis(points_a_cm)
    len_b, ax_b = pca_length_axis(points_b_cm)
    angle = angle_between_deg(ax_a, ax_b)
    both_long = len_a > length_floor_cm and len_b > length_floor_cm
    collinear = angle is not None and angle < angle_ceiling_deg
    score = 0.0
    if both_long and collinear:
        angle_term = max(0.0, (angle_ceiling_deg - angle) / angle_ceiling_deg)
        dis_term = max(0.0, (10.0 - edge_dis_cm) / 10.0) + 0.2
        score = min(len_a, len_b) * angle_term * dis_term
    return score, dict(len_a=len_a, len_b=len_b, angle_deg=angle,
                        both_long=both_long, collinear=collinear)


# ---------------------------------------------------------------------------
# CLI: run the BFS-replay check over one or more arms.
# ---------------------------------------------------------------------------

def iter_edges(arm_dir):
    for logf in sorted(glob.glob(os.path.join(arm_dir, 'pr_evt*', 'oc56scan-evt*.jsonl'))):
        with open(logf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec['type'] != 'edge':
                    continue
                rec['_log'] = logf
                yield rec


def iter_events(arm_dir):
    """(logfile, [records]) per event -- needed for the round-4 checks, which
    are per graph call rather than per edge."""
    for logf in sorted(glob.glob(os.path.join(arm_dir, 'pr_evt*', 'oc56scan-evt*.jsonl'))):
        recs = []
        with open(logf) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        yield logf, recs


def check_connectivity(arm_dirs, max_report=10):
    """doc pr/57 round 4, two independent checks on the connectivity record.

    (a) Self-consistency: `final[]` comes from connected_components() on the
        graph, `edges[]` is accumulated at the three emit sites. They must
        agree about every pair or the record is buggy.
    (b) Blind spot: an edge record is only written for candidates that reach
        two_d_gap_kill (S1-S5 let them through, same APA/face, and either at
        or above the 1cm floor or the W-override). So a pair can end up in
        different final components with NO row in the scan table at all --
        a separation the owner is never shown. Counted, not fixed."""
    n_calls = n_pairs = n_sep = n_sep_nodump = 0
    n_missing = 0
    bad = []
    for arm in arm_dirs:
        for logf, recs in iter_events(arm):
            conn = oc56_conn.index_conn(recs)
            edges = [r for r in recs if r.get('type') == 'edge']
            calls_with_edges = set(e['graph_call'] for e in edges)
            n_missing += len(calls_with_edges - set(conn))
            by_call = {}
            for e in edges:
                by_call.setdefault(e['graph_call'], set()).add(
                    (min(e['j'], e['k']), max(e['j'], e['k'])))
            for gc, rec in conn.items():
                n_calls += 1
                for msg in oc56_conn.check_consistency(rec):
                    if len(bad) < max_report:
                        bad.append('%s %s' % (os.path.basename(logf), msg))
                final = rec.get('final', [])
                dumped = by_call.get(gc, set())
                for j in range(len(final)):
                    for k in range(j + 1, len(final)):
                        n_pairs += 1
                        if final[j] == final[k]:
                            continue
                        n_sep += 1
                        if (j, k) not in dumped:
                            n_sep_nodump += 1
    return dict(calls=n_calls, pairs=n_pairs, separated=n_sep,
                separated_no_dumped_edge=n_sep_nodump,
                calls_missing_record=n_missing, bad=bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm_dirs', nargs='+')
    ap.add_argument('--max-mismatches', type=int, default=10)
    ap.add_argument('--conn-only', action='store_true',
                    help='run only the round-4 connectivity checks')
    args = ap.parse_args()

    if args.conn_only:
        r = check_connectivity(args.arm_dirs, args.max_mismatches)
        print('graph calls with a connectivity record: %d' % r['calls'])
        print('graph calls with edge records but NO connectivity record: %d'
              % r['calls_missing_record'])
        print('component pairs: %d, of which separated: %d' % (r['pairs'], r['separated']))
        print('separated pairs with NO dumped candidate edge (scan blind spot): '
              '%d (%.1f%% of separations)'
              % (r['separated_no_dumped_edge'],
                 100.0 * r['separated_no_dumped_edge'] / max(r['separated'], 1)))
        print('final[] vs edges[] disagreements: %d' % len(r['bad']))
        for b in r['bad']:
            print('  INCONSISTENT', b)
        if r['bad']:
            sys.exit(1)
        print('PASS: every connectivity record agrees with its own emitted edge list.')
        return

    if np is None:
        print('WARNING: numpy not importable -- long_track_break_score() is '
              'unusable, BFS replay check is unaffected.', file=sys.stderr)

    n_edges = 0
    n_plane_checks = 0
    n_native = 0
    n_nonnative = 0
    all_mismatches = []
    for arm in args.arm_dirs:
        for rec in iter_edges(arm):
            n_edges += 1
            for p in rec['planes']:
                if p.get('native_step', 1) == 1:
                    n_native += 1
                else:
                    n_nonnative += 1
            mm = check_edge(rec, max_report=args.max_mismatches - len(all_mismatches))
            n_plane_checks += sum(1 for p in rec['planes'] if p.get('native_step', 1) == 1) * 16
            all_mismatches.extend(m for m in mm if m is not None)
            if len(all_mismatches) > args.max_mismatches:
                all_mismatches = all_mismatches[:args.max_mismatches]

    print(f'edges checked: {n_edges}')
    print(f'plane records: native_step=1 (exhaustive, checked): {n_native}  '
          f'non-native (skipped, window over cap): {n_nonnative}')
    print(f'matrix cells replayed: {n_plane_checks}')
    print(f'mismatches: {len(all_mismatches)}')
    for m in all_mismatches:
        print('  MISMATCH', m)

    if all_mismatches:
        print('FAIL: the dump does not reproduce its own logged verdict -- '
              'see mismatches above. Fix before trusting any label collected '
              'against this dump.', file=sys.stderr)
        sys.exit(1)
    print('PASS: every native-step=1 plane record reproduces its full '
          'dw=1..4 x ds=1..4 matrix from its own dumped fired/dead cells.')


if __name__ == '__main__':
    main()
