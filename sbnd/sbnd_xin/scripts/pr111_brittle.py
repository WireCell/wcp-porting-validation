#!/usr/bin/env python3
"""doc pr/111 sec 7 -- how brittle is the net?  Does a PHYSICALLY NEGLIGIBLE
perturbation of the cloud move the global argmax as far as fit_exclusion does?

sec 6 measured the trigger scale: at 122660's rival voxel a +3.4 % charge change
takes the net's response from 0.9998 to 0.0020.  So the perturbations here are
deliberately far below anything physical:

  P1  per-point iid Gaussian position jitter, sigma = 0.05 cm.  That is 1/10 of a
      voxel and ~6x below the SBND slice pitch (3.13 mm) -- well under the
      trajectory fit's own precision.  Unlike a RIGID translation (which is an
      exact no-op, because SCN_Vertex.py normalises x - x.min()), this re-phases
      the lattice and re-bins points.
  P2  coherent dQ rescale by a single factor 1 +/- 0.03.  A 3 % calibration-scale
      change applied identically to every point; it changes no geometry at all.

Reported: the fraction of draws whose global argmax moves > 2 cm from the
unperturbed argmax, the response spread, and the PRE-REGISTERED verdict comparing
the actual exclusion effect |net_ON - net_OFF| against the P1 band.

Pre-registered rule, fixed before the numbers were seen:
  |net_ON - net_OFF| INSIDE the null band on >=5 of 8 exhibits  => H2 (the effect
      is within the net's own input sensitivity; no cloud-side fix is reliable)
  OUTSIDE on >=6 of 8  => H1/H3 (exclusion does something specific to the cloud)

NOTE on an earlier version of this test: the first null family tried was a RIGID
translation of the whole cloud.  That is an exact no-op -- SCN_Vertex.py
normalises x - x.min(axis=0), so translating every point leaves the voxel
coordinates bit-identical -- and it produced a zero-width band on all 8 events.
P1 below uses iid per-point jitter instead, which genuinely re-phases the lattice.

Repro:  python3 scripts/pr111_brittle.py [N]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr111_scn_lib import load_cloud, infer

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ON = os.path.join(H, 'work-vtx106-harv-base-nuecc48')
OFF = os.path.join(H, 'work-vtx106-harv-nofitx-nuecc48')
EXH = ['46363', '235435', '389538', '122660', '268067', '271851', '360535', '111412']
SEED = 20260822
SIG = 0.05     # cm
DQF = 0.03     # 3 %


def peak(cloud):
    v = infer(cloud, top_k=2)[0]
    return np.array(v[:3]), float(v[3])


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rng = np.random.default_rng(SEED)
    print(f"# N={N} draws, seed={SEED};  P1 = iid position jitter sigma={SIG} cm;  "
          f"P2 = coherent dQ rescale +/-{100*DQF:.0f} %")
    print(f"{'evt':>7s} {'|dON-OFF|':>9s} | {'P1 argmax>2cm':>13s} {'P1 med move':>11s} "
          f"{'P1 score band':>21s} {'verdict':>8s} | {'P2 argmax>2cm':>13s} {'P2 score band':>21s}")
    tot1 = tot2 = tot = 0
    inside = outside = 0
    for e in EXH:
        cloud, b = load_cloud(ON, e)
        p0, s0 = peak(cloud)
        s_off = peak(load_cloud(OFF, e)[0])[1]
        dexcl = abs(s0 - s_off)
        x, y, z, q = cloud
        dQ = (q.astype(np.float64) + 1000.0) / 0.1
        res = {}
        for tag in ('P1', 'P2'):
            mv, sc = [], []
            for _ in range(N):
                if tag == 'P1':
                    j = rng.normal(0, SIG, size=(len(x), 3)).astype(np.float32)
                    c = (x + j[:, 0], y + j[:, 1], z + j[:, 2], q)
                else:
                    g = 1.0 + rng.uniform(-DQF, DQF)
                    c = (x, y, z, (dQ * g * 0.1 - 1000.0).astype(np.float32))
                p, s = peak(c)
                mv.append(float(np.linalg.norm(p - p0))); sc.append(s)
            res[tag] = (sum(1 for m in mv if m > 2.0), float(np.median(mv)), min(sc), max(sc))
        a1, m1, lo1, hi1 = res['P1']; a2, m2, lo2, hi2 = res['P2']
        tot1 += a1; tot2 += a2; tot += N
        out = dexcl > (hi1 - lo1)
        outside += out; inside += (not out)
        print(f"{e:>7s} {dexcl:9.4f} | {a1:6d}/{N:<6d} {m1:11.2f} [{lo1:.4f},{hi1:.4f}] w={hi1-lo1:.4f} "
              f"{'OUTSIDE' if out else 'inside':>8s} | {a2:6d}/{N:<6d} [{lo2:.4f},{hi2:.4f}] w={hi2-lo2:.4f}")
    print(f"\nPOOLED: a negligible perturbation moves the global argmax > 2 cm in "
          f"P1 {tot1}/{tot} ({100*tot1/tot:.0f} %),  P2 {tot2}/{tot} ({100*tot2/tot:.0f} %)")
    print(f"PRE-REGISTERED VERDICT: |dON-OFF| inside the P1 band on {inside}/8, outside on {outside}/8"
          f"  =>  {'H2' if inside >= 5 else ('H1/H3' if outside >= 6 else 'indeterminate')}")


if __name__ == '__main__':
    main()
