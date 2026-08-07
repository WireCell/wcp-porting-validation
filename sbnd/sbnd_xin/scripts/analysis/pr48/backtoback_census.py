#!/usr/bin/env python3
"""doc pr/48: census over the 445/1000 calib-pr dumps in work-mcp1k-cb0805
(same coverage caveat as pr/20's kink_probe.py and pr/47's
cathode_junction_census.py -- this is a read-only PROXY measurement on
final post-fit geometry, not a break-time measurement; see doc pr/48 sec
"Phase 1" for the live-diagnostic cross-check on the two named events).

Three measurements, all per main-cluster segment, run over EVERY interior
fit index (not just cathode crossings -- pr/47's wide-turn helper is
evaluated un-gated here):

  1. Wide-baseline PCA turn angle (pr/47's segment_cathode_wide_kink_accepts
     statistic, re-implemented in Python) at TWO operating points:
     (skirt=3cm, base=15cm) -- the 57485 tier (tier 1, ~50 deg)
     (skirt=3cm, base=35cm) -- the 57903 tier (tier 2, ~25-30 deg)
     Both directions measured (turn_bothdir.py gotcha); interior guard
     >=6cm from both segment ends (51513 end-artifact gotcha, doc pr/47).

  2. Two-end dQ/dx rise statistic: median dQ/dx over the first/last 8cm of
     each segment vs the segment's interior median.  A "two-end rise"
     candidate is a segment where BOTH ends read above a MIP-multiple floor
     relative to the interior -- the primary, angle-independent detector
     that reaches 51513/56211 as well as 57903/57485.

  3. Simple-topology selectivity: main-cluster segment count == 1 AND both
     graph endpoints degree-1 (proxy: this script only sees segments/points,
     not the boost graph degree -- reported as "single main segment" count,
     a necessary but not sufficient proxy for the true gate).

Usage: python3 backtoback_census.py > census_output.tsv
"""
import glob, json, math, os, sys
import numpy as np

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
ROOT = os.path.join(SB, 'work-mcp1k-cb0805')
MIP = 43000.0  # e/cm, m_mip_dqdx_median default; SBND production overrides
               # to 48000 (cfg/pgrapher/experiment/sbnd/clus.jsonnet:901) --
               # this census reports RATIOS to a fixed 43000 for comparability
               # across events; absolute MIP-multiple thresholds should be
               # re-checked against the SBND 48000 value before any knob ships.

SKIRT = 3.0
BASELINES = (15.0, 35.0)
INTERIOR_GUARD = 6.0  # cm from either segment end (pr/47 51513 gotcha)


def pca_dir(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    v = vt[0]
    if np.dot(pts[-1] - pts[0], v) < 0:
        v = -v
    return v


def wide_turn(P, cum, i, skirt, base):
    """Both-direction crossing turn angle -- fires at ANY interior index,
    not just sign-change crossings (un-gated from pr/47's cathode scope)."""
    ia = [k for k in range(0, i + 1) if skirt <= cum[i] - cum[k] <= skirt + base]
    ib = [k for k in range(i, len(P)) if skirt <= cum[k] - cum[i] <= skirt + base]
    if len(ia) < 3 or len(ib) < 3:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, float(
        np.dot(pca_dir(P[ia]), pca_dir(P[ib]))
    )))))


def main():
    dumps = sorted(glob.glob(os.path.join(ROOT, 'pr_evt*', 'calib-pr-evt*.json')))
    n_events_seen = 0
    n_segments = 0

    # Tail population at each operating point: counts of the MAX interior
    # turn angle per segment falling in various angle bins, split by
    # segment length bucket (short <30cm, mid 30-60cm, long >60cm) --
    # this is what decides whether tier 1 (~50deg) or tier 2 (~25-30deg on
    # a 35cm baseline) is safe to ship.
    bins = [0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 999]
    tail = {b: {lo: [0] * (len(bins) - 1) for lo in ('short', 'mid', 'long')}
            for b in BASELINES}

    two_end_rise_events = []  # (evt, seg_id, ratio_lo, ratio_hi) candidates
    n_single_main_segment = 0

    out_rows = []

    for path in dumps:
        evt = int(os.path.basename(path).replace('calib-pr-evt', '').replace('.json', ''))
        try:
            d = json.load(open(path))
        except Exception as e:
            print(f"# SKIP evt {evt}: {e}", file=sys.stderr)
            continue
        n_events_seen += 1

        main_segs = [s for s in d.get('segments', []) if s.get('is_main_cluster')]
        if len(main_segs) == 1:
            n_single_main_segment += 1

        for s in main_segs:
            pts = s.get('points', [])
            if len(pts) < 8:
                continue
            P = np.array([[p['x'], p['y'], p['z']] for p in pts], float)
            dq = np.array([p['dQ'] / (p['dx'] + 1e-9) for p in pts]) / MIP
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
            L = cum[-1]
            if L < 2 * INTERIOR_GUARD:
                continue
            n_segments += 1

            length_bucket = 'short' if L < 30 else ('mid' if L < 60 else 'long')

            for base in BASELINES:
                max_turn = 0.0
                for i in range(len(P)):
                    if cum[i] < INTERIOR_GUARD or L - cum[i] < INTERIOR_GUARD:
                        continue
                    t = wide_turn(P, cum, i, SKIRT, base)
                    if t is not None and t > max_turn:
                        max_turn = t
                bidx = min(np.searchsorted(bins, max_turn, side='right') - 1, len(bins) - 2)
                tail[base][length_bucket][bidx] += 1

            # Two-end rise: median dQ/dx over first/last 8cm vs interior
            # (>12cm from both ends, matching the earlier hand analysis).
            end_lo = dq[cum <= 8]
            end_hi = dq[cum >= L - 8]
            mid = dq[(cum > 12) & (cum < L - 12)]
            if len(end_lo) >= 3 and len(end_hi) >= 3 and len(mid) >= 3:
                m_lo, m_hi, m_mid = np.median(end_lo), np.median(end_hi), np.median(mid)
                if m_mid > 0:
                    r_lo, r_hi = m_lo / m_mid, m_hi / m_mid
                    if r_lo > 1.3 and r_hi > 1.3:
                        two_end_rise_events.append((evt, s.get('id'), round(r_lo, 2), round(r_hi, 2)))

            out_rows.append((evt, s.get('id'), round(L, 1), n_events_seen))

    print(f"# events with calib-pr dump: {n_events_seen} (coverage caveat: 445/1000 nominal)", file=sys.stderr)
    print(f"# main-cluster segments (L>={2*INTERIOR_GUARD}cm, npts>=8): {n_segments}", file=sys.stderr)
    print(f"# events with exactly 1 main-cluster segment: {n_single_main_segment}", file=sys.stderr)
    print(f"# two-end-rise candidates (both ends >1.3x interior median): {len(two_end_rise_events)}", file=sys.stderr)
    for evt, sid, rlo, rhi in two_end_rise_events:
        print(f"#   evt {evt} seg {sid} ratio_lo={rlo} ratio_hi={rhi}", file=sys.stderr)

    print("baseline_cm\tlength_bucket\tbin_lo\tbin_hi\tcount")
    for base in BASELINES:
        for bucket in ('short', 'mid', 'long'):
            for k in range(len(bins) - 1):
                print(f"{base}\t{bucket}\t{bins[k]}\t{bins[k+1]}\t{tail[base][bucket][k]}")


if __name__ == '__main__':
    main()
