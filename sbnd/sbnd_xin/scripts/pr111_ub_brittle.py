#!/usr/bin/env python3
"""doc pr/111 sec 8 -- is the SCN net as brittle on uBooNE clouds (its NATIVE,
in-distribution detector) as it is on SBND clouds?

sec 7 showed a 0.05 cm iid position jitter relocates the net's global argmax by
>2 cm in 30 % of draws on SBND clouds, while fit_exclusion itself moves cloud
points by 0.277 cm median.  If uBooNE is stable under the same test, SBND's
DL-vertex sensitivity to exclusion is an out-of-distribution effect (the shipped
CP24 weights are uBooNE-trained -- doc pr/110) rather than a property of the net.

*** SCOPE CAVEAT, load-bearing ***
The uBooNE arms carry no dl_vtx_harvest, so their cloud is rebuilt from the
persisted T_rec_charge tree.  That is the END-OF-JOB fitted trajectory, NOT the
DL-time cloud: validated on SBND 46363, the tree has 887 points against the
harvest's 731, and only 23 % of harvested points have a <0.01 cm match (median
nearest distance 0.18 cm).  So this section does NOT reproduce any uBooNE DL
decision.  It characterises the net's STABILITY on each detector's fitted-
trajectory cloud, and to keep that comparison honest the SBND side is rebuilt the
SAME way, from its own T_rec_charge -- never from the harvest.

Repro:  python3 scripts/pr111_ub_brittle.py [N]
"""
import os, sys, glob
import numpy as np
import uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr111_scn_lib import infer

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport/scripts/sweep/pr109e_wct_on'
SB = os.path.join(H, 'work-vtx106-harv-base-nuecc48')
SEED = 20260822
SIG = 0.05


def cloud_from_tree(path):
    t = uproot.open(path)['T_rec_charge']
    a = t.arrays(['x', 'y', 'z', 'q'], library='np')
    return (a['x'].astype(np.float32), a['y'].astype(np.float32),
            a['z'].astype(np.float32), a['q'].astype(np.float32))


def peak(c):
    v = infer(c, top_k=2)[0]
    return np.array(v[:3]), float(v[3])


def run(tag, items, N, rng):
    nmove = ntot = 0
    print(f"\n--- {tag} ---")
    print(f"{'evt':>8s} {'npts':>6s} {'q range':>20s} {'S0':>7s} | {'argmax>2cm':>10s} {'med move':>8s} {'score band':>21s}")
    for name, path in items:
        try:
            c = cloud_from_tree(path)
        except Exception as ex:
            print(f'{name:>8s}  ERR {ex}'); continue
        if len(c[0]) < 20:
            print(f'{name:>8s}  too few points ({len(c[0])}), skipped'); continue
        p0, s0 = peak(c)
        mv, sc = [], []
        for _ in range(N):
            j = rng.normal(0, SIG, size=(len(c[0]), 3)).astype(np.float32)
            p, s = peak((c[0] + j[:, 0], c[1] + j[:, 1], c[2] + j[:, 2], c[3]))
            mv.append(float(np.linalg.norm(p - p0))); sc.append(s)
        k = sum(1 for m in mv if m > 2.0)
        nmove += k; ntot += N
        print(f"{name:>8s} {len(c[0]):6d} [{c[3].min():8.1f},{c[3].max():8.1f}] {s0:7.4f} | "
              f"{k:4d}/{N:<5d} {np.median(mv):8.2f} [{min(sc):.4f},{max(sc):.4f}] w={max(sc)-min(sc):.4f}")
    print(f"{tag} POOLED: argmax moves > 2 cm in {nmove}/{ntot} ({100*nmove/ntot:.0f} %)")
    return nmove, ntot


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rng = np.random.default_rng(SEED)
    ub = []
    for d in sorted(glob.glob(UB + '/*_*'))[:8]:
        g = glob.glob(d + '/track_com_*.root')
        if g:
            ub.append((os.path.basename(d).split('_')[-1], g[0]))
    sb = [(e, os.path.join(SB, f'pr_evt{e}', 'tracking-pr.root')) for e in
          ['46363', '235435', '389538', '122660', '268067', '271851', '360535', '111412']]
    print(f"# N={N} draws/event, seed={SEED}, iid position jitter sigma={SIG} cm")
    print(f"# clouds rebuilt from T_rec_charge on BOTH detectors (see the caveat in the header)")
    a, at = run('uBooNE (toolkit, exclusion ON)', ub, N, rng)
    b, bt = run('SBND (toolkit, exclusion ON)', sb, N, rng)
    print(f"\nSTABILITY: uBooNE {100*a/at:.0f} %  vs  SBND {100*b/bt:.0f} %  "
          f"of draws relocate the argmax > 2 cm under a {SIG} cm jitter")


if __name__ == '__main__':
    main()
