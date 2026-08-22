#!/usr/bin/env python3
"""doc pr/109 sec 9 -- how hard fit_exclusion bites, from the pr/108 sec 8 stage dump.

update_association (TrackFitting.cxx:2717-2898) keeps a 2-D cell for a segment iff

    min_dis_track < min-over-OTHER-SEGMENTS   or   min_dis_track < 0.3 cm

where "other segments" is get_segment_edges() restricted by m_cluster_filter, i.e.
the fitted cluster's OWN segments.  The keep test is therefore a "strictly closest
of all siblings" competition, and its severity grows with the number of siblings.

The pr/108 stage dump records, per trajectory point per form_map_graph pass,
the association sizes BEFORE exclusion (n0) and AFTER (n1), so the strip fraction
1 - sum(n1)/sum(n0) is directly measurable, and the segment index per line gives
the size of the arbitration universe per call.

Repro:
    WCT_TRAJ_DUMP=/path/dump.txt SBND_FIT_EXCLUSION=true PR_JOBS=1 \
        ./run_pr_chain_batch.sh <base> <arm> data <evt>
    WCT_TRAJ_DUMP=/path/dump.txt QL_FIT_EXCLUSION=true ./run_one.sh <idx> <label>
    pr109_traj_strip.py --dump /path/dump.txt --root <tracking-pr.root|track_com_*.root>

Lines with "call excl=0" are the deliberate exclusion-free sites (the 5 hard-false
call sites, NeutrinoPatternBase.cxx:2264/2339/2531, NeutrinoVertexFinder.cxx:780/4806)
and are reported separately as the null control: their strip fraction must be 0.0 %.
"""
import argparse, collections, statistics as st
import numpy as np
import uproot

BANDS = [(0, 3), (3, 6), (6, 10), (10, 20), (20, float('inf'))]


def parse(dump):
    cur = None
    excl = {}
    rows = []                       # (excl, stage, x, y, z, n0, n1, kept)
    segs = collections.defaultdict(set)
    for line in open(dump):
        p = line.split()
        if len(p) >= 3 and p[1] == 'call':
            cur = int(p[0])
            excl[cur] = int(p[2].split('=')[1])
            continue
        if len(p) >= 21 and p[1].startswith('map'):
            stage = int(p[1][3:])
            if stage == 1:
                segs[cur].add(int(p[2]))
            rows.append((excl.get(cur, 0), stage,
                         float(p[4]), float(p[5]), float(p[6]),
                         sum(int(v) for v in p[7:10]),
                         sum(int(v) for v in p[10:13]),
                         int(p[16])))
    return rows, excl, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--root', required=True, help='the same run\'s tracking ROOT, for T_kine')
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    g = uproot.open(args.root)
    k = g['T_kine'].arrays(['kine_nu_x_corr', 'kine_nu_y_corr', 'kine_nu_z_corr'], library='np')
    V = np.array([k['kine_nu_x_corr'][0], k['kine_nu_y_corr'][0], k['kine_nu_z_corr'][0]], float)

    rows, excl, segs = parse(args.dump)
    print('# %s   nu vertex %s' % (args.label, np.round(V, 2)))

    print('%-6s %-6s %8s %10s %10s %8s %8s' %
          ('excl', 'stage', 'npts', 'sum_n0', 'sum_n1', 'strip%', 'dropped%'))
    for e in (0, 1):
        for stage in (1, 2, 3):
            s = [r for r in rows if r[0] == e and r[1] == stage]
            if not s:
                continue
            n0 = sum(r[5] for r in s); n1 = sum(r[6] for r in s)
            nd = sum(1 for r in s if r[7] == 0)
            print('%-6d %-6d %8d %10d %10d %7.1f%% %7.1f%%' %
                  (e, stage, len(s), n0, n1,
                   100 * (1 - n1 / n0) if n0 else 0, 100 * nd / len(s)))

    print('\n# exclusion calls only, by 3-D distance to the neutrino vertex')
    print('%-12s %8s %10s %10s %8s' % ('dist(cm)', 'npts', 'sum_n0', 'sum_n1', 'strip%'))
    for lo, hi in BANDS:
        s = [r for r in rows if r[0] == 1
             and lo <= np.linalg.norm(np.array(r[2:5]) - V) < hi]
        if not s:
            continue
        n0 = sum(r[5] for r in s); n1 = sum(r[6] for r in s)
        print('%-12s %8d %10d %10d %7.1f%%' %
              ('%g-%g' % (lo, hi if hi != float('inf') else 999), len(s), n0, n1,
               100 * (1 - n1 / n0) if n0 else 0))

    n = [len(v) for c, v in segs.items() if excl.get(c) == 1]
    if n:
        print('\n# arbitration universe (all_segments per exclusion call)')
        print('  calls %d   median %.1f   mean %.1f   max %d' %
              (len(n), st.median(n), sum(n) / len(n), max(n)))
        print('  distribution:', dict(sorted(collections.Counter(n).items())))


if __name__ == '__main__':
    main()
