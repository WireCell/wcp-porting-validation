#!/usr/bin/env python3
"""Imaged-charge profiles across the structural gaps, and vs drift (doc pdvd/33).

  usage: fv_gap_profile.py PDVD:<arm-dir> [PDVD:<arm-dir> ...] [SBND:<arm-dir> ...]
         fv_gap_profile.py $(ls -d work/*_d28dlfp | sed 's/^/PDVD:/' | head -40)

Reads only the Bee mabc-pr.zip of arms already on disk.

Two profiles:

  A. point density vs signed distance from each structural boundary.  The
     geometric gap (from the AnodePlane 'sensvol' log lines) is what
     DetectorVolumes::contained() encodes; the density dip is how wide the gap
     actually is to the reconstruction, which is what a fiducial volume should
     be built from.

  B. median point charge vs drift distance.  PDVD's tgm_fv_x_margin = 30 cm was
     chosen against a "dQ/dx rises ~50 % over the last ~30 cm before the CRP"
     observation; this asks whether the rise is a near-CRP feature or a global
     drift gradient.  CAVEAT: Bee q is per-point blob charge, not the fitted
     dQ/dx the observation was made on, and a data arm carries an electron
     lifetime a MC arm does not.  Suggestive only.
"""
import collections, json, sys, zipfile
import numpy as np

GEOM = {   # detector: (drift half width cm, structural boundaries to profile)
    'PDVD': (339.91, [
        ('central y seam (union hole |y|<0.61)',      'y',      0.0,   4.0, 0.5),
        ('CRP y seam |y|=168.5 (union gap 0.02 cm)',  'absy', 168.5,   4.0, 0.5),
        ('CRP z seam z=149.65 (union gap 0.1-0.26)',  'z',    149.65,  4.0, 0.5),
        ('cathode face |x|=3.0 (union hole |x|<3)',   'absx',   3.0,   6.0, 1.0),
    ]),
    'SBND': (201.45, [
        ('CPA face |x|=0.45 (union hole |x|<0.45)',   'absx',   0.45,  6.0, 1.0),
        ('mid-y (no structure -- control)',           'y',      0.0,   4.0, 0.5),
    ]),
}


def load(arms):
    X, Y, Z, Q = [], [], [], []
    for arm in arms:
        try:
            d = json.loads(zipfile.ZipFile(arm + '/mabc-pr.zip').read('data/0/0-clustering-global.json'))
        except Exception:
            continue
        x, y, z, q = (np.array(d[k]) for k in ('x', 'y', 'z', 'q'))
        ok = np.abs(x) < 1e4                    # drop the no-t0 sentinel x
        X.append(x[ok]); Y.append(y[ok]); Z.append(z[ok]); Q.append(q[ok])
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(Z), np.concatenate(Q))


def profile(v, centre, half, step, label):
    d = v - centre
    m = np.abs(d) < half
    edges = np.arange(-half, half + step, step)
    h, _ = np.histogram(d[m], bins=edges)
    ctr = edges[:-1] + step / 2
    far = h[np.abs(ctr) > half * 0.6]
    ref = far[far > 0].mean() if (far > 0).any() else 1.0
    print(f"  {label}")
    print(f"    bin centre cm : " + ' '.join(f"{c:5.1f}" for c in ctr))
    print(f"    density/far   : " + ' '.join(f"{n / ref:5.2f}" for n in h))


def main(args):
    byd = collections.defaultdict(list)
    for a in args:
        det, _, arm = a.partition(':')
        byd[det].append(arm)
    for det, arms in byd.items():
        xw, bounds = GEOM[det]
        X, Y, Z, Q = load(arms)
        print(f"\n=== {det}: {len(arms)} arms, {len(X)} imaged points, drift half-width {xw} cm ===")
        print("\nA. point density across the structural boundaries")
        for label, axis, centre, half, step in bounds:
            v = {'absx': np.abs(X), 'absy': np.abs(Y), 'y': Y, 'z': Z}[axis]
            profile(v, centre, half, step, label)
        print("\nB. median point charge vs drift distance (cm inside the anode wall)")
        drift = xw - np.abs(X)
        good = Q > 0
        rows = []
        for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 50),
                       (50, 100), (100, 150), (150, 200), (200, 300), (300, 400)):
            m = good & (drift >= lo) & (drift < hi)
            if m.sum() < 200:
                continue
            rows.append((lo, hi, float(np.median(Q[m])), int(m.sum())))
        bulk = [r[2] for r in rows if r[0] >= 100]
        ref = float(np.median(bulk)) if bulk else rows[-1][2]
        for lo, hi, mq, n in rows:
            print(f"    {lo:4d}-{hi:4d} cm  median q {mq:8.0f}  ({mq / ref:4.2f} x bulk)  n={n}")
        sel = [r for r in rows if r[0] >= 10]
        if len(sel) > 3:
            v_drift = 0.148073 if det == 'PDVD' else 0.16
            t = np.array([(r[0] + r[1]) / 2 for r in sel]) / v_drift
            p = np.polyfit(t, np.log([r[2] for r in sel]), 1)
            print(f"    exponential fit over drift > 10 cm: tau = {-1 / p[0]:.0f} us"
                  f"  => {np.exp(p[0] * xw / v_drift):.2f} x over the full drift")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
