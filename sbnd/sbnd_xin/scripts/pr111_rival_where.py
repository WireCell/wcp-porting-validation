#!/usr/bin/env python3
"""doc pr/111 sec 6 -- WHERE the rival maximum is, and what exclusion does there.

sec 5 showed the OFF-arm win is won by the rival maximum DISAPPEARING, not by the
vertex getting brighter (389538: response at the true vertex is HIGHER with
exclusion ON, 0.526 vs 0.428 -- it loses only because a 0.982 rival sits 213 cm
away).  This asks what the cloud looks like at the rival.

Repro:  python3 scripts/pr111_rival_where.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr111_scn_lib import load_cloud, field

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM = {'ON': os.path.join(H, 'work-vtx106-harv-base-nuecc48'),
       'OFF': os.path.join(H, 'work-vtx106-harv-nofitx-nuecc48')}
EXH = [('46363', 'M-a', 19001, 19001), ('235435', 'M-a', 2001, 2001),
       ('389538', 'M-b', 11002, 11004), ('122660', 'M-b', 9014, 9013),
       ('268067', 'M-d', 15007, 15004), ('271851', 'BREAK', 23001, 23001)]
R = 3.0


def tgt(b, vid):
    for r in b.get('rows', []):
        if r['vertex_id'] == vid:
            return np.array([r['x'], r['y'], r['z']])
    return None


def stats(cloud, p, r=R):
    x, y, z, q = cloud
    P = np.stack((x, y, z), axis=1).astype(np.float64)
    d = np.linalg.norm(P - p[None, :], axis=1)
    m = d < r
    dQ = (q.astype(np.float64) + 1000.0) / 0.1
    return int(m.sum()), float(dQ[m].sum())


def main():
    print(f"# rival = the ON arm's global argmax voxel; cloud stats in a {R:.0f} cm ball")
    print(f"{'evt':>7s} {'cls':>5s} {'|rival-tgt|':>11s} | {'nON':>5s} {'nOFF':>5s} {'dn%':>7s} | "
          f"{'qON':>10s} {'qOFF':>10s} {'dq%':>7s} | {'S_rival ON':>10s} {'S_rival OFF':>11s} {'drop':>7s}")
    for e, cls, ton, toff in EXH:
        cON, bON = load_cloud(ARM['ON'], e)
        cOFF, bOFF = load_cloud(ARM['OFF'], e)
        fON, fOFF = field(cON), field(cOFF)
        t = tgt(bON, ton)
        i = int(np.argmax(fON[:, 3]))
        p = fON[i, :3]
        s_on = float(fON[i, 3])
        dOFF = np.linalg.norm(fOFF[:, :3] - p[None, :], axis=1)
        s_off = float(fOFF[dOFF < 2.0, 3].max()) if (dOFF < 2.0).any() else 0.0
        nA, qA = stats(cON, p)
        nB, qB = stats(cOFF, p)
        dn = 100 * (nB - nA) / nA if nA else float('nan')
        dq = 100 * (qB - qA) / qA if qA else float('nan')
        print(f"{e:>7s} {cls:>5s} {np.linalg.norm(p - t):11.2f} | {nA:5d} {nB:5d} {dn:+7.1f} | "
              f"{qA:10.3e} {qB:10.3e} {dq:+7.1f} | {s_on:10.4f} {s_off:11.4f} "
              f"{100*(s_off-s_on)/s_on if s_on else float('nan'):+6.0f}%")


if __name__ == '__main__':
    main()
