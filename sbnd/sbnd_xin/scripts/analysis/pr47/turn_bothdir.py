#!/usr/bin/env python3
"""doc pr/47 sec 8: BOTH-DIRECTION recompute of the wide-baseline turn angle,
mirroring the C++ helper segment_cathode_wide_kink_accepts exactly:
  - EVERY sign-change crossing per segment (census took only the first)
  - side filter matched to each arm's own side (census assumed neg->pos and
    returned nan for the 30 pos->neg crossings)
Reads the same calib-pr dumps as the census (445/1000 coverage caveat).
Usage: python3 turn_bothdir.py [evt ...]   (default: all events with a dump)
"""
import glob, json, math, os, sys
import numpy as np

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
ROOT = os.path.join(SB, 'work-mcp1k-cb0805')
SKIRT, BASE = 3.0, 15.0  # cm, the C++ defaults


def arms_turn(P, ic):
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    xa, xb = P[ic, 0], P[ic + 1, 0]
    ia = [k for k in range(0, ic + 1)
          if SKIRT <= cum[ic] - cum[k] <= SKIRT + BASE
          and ((P[k, 0] <= 0) if xa <= 0 else (P[k, 0] >= 0))]
    ib = [k for k in range(ic + 1, len(P))
          if SKIRT <= cum[k] - cum[ic] <= SKIRT + BASE
          and ((P[k, 0] >= 0) if xb > 0 else (P[k, 0] <= 0))]
    if len(ia) < 3 or len(ib) < 3:
        return None
    def pca(idx):
        pts = P[idx]
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c)
        v = vt[0]
        if np.dot(pts[-1] - pts[0], v) < 0:
            v = -v
        return v
    va, vb = pca(ia), pca(ib)
    return math.degrees(math.acos(max(-1, min(1, float(np.dot(va, vb))))))


def main():
    want = set(int(a) for a in sys.argv[1:])
    dumps = sorted(glob.glob(os.path.join(ROOT, 'pr_evt*', 'calib-pr-evt*.json')))
    fires = []
    for path in dumps:
        evt = int(os.path.basename(path).replace('calib-pr-evt', '').replace('.json', ''))
        if want and evt not in want:
            continue
        d = json.load(open(path))
        for seg in d['segments']:
            P = np.array([[p['x'], p['y'], p['z']] for p in seg['points']], float)
            if len(P) < 8:
                continue
            x = P[:, 0]
            for ic in range(len(x) - 1):
                if not ((x[ic] <= 0 and x[ic + 1] > 0) or (x[ic] >= 0 and x[ic + 1] < 0)):
                    continue
                t = arms_turn(P, ic)
                if t is not None and t >= 20:
                    fires.append((evt, seg.get('id'), ic, round(t, 1)))
                    print('evt %6d seg %6s ic %3d turn %.1f deg  x=(%.2f -> %.2f)'
                          % (evt, seg.get('id'), ic, t, x[ic], x[ic + 1]))
    print('crossings with both-dir turn >= 20 deg:', len(fires),
          'in events:', sorted(set(f[0] for f in fires)))


if __name__ == '__main__':
    main()
