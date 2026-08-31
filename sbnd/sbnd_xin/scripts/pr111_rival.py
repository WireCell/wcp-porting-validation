#!/usr/bin/env python3
"""doc pr/111 sec 5 -- the net's response AT the target vs its GLOBAL peak.

The "starved vertex voxel" story (H1) predicts that with exclusion ON the net's
response collapses AT the true vertex.  This measures that directly, and
separately measures where the winning maximum actually sits.

Repro:  python3 scripts/pr111_rival.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr111_scn_lib import load_cloud, field

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM = {'ON': os.path.join(H, 'work-vtx106-harv-base-nuecc48'),
       'OFF': os.path.join(H, 'work-vtx106-harv-nofitx-nuecc48')}
EXH = [('46363', 'M-a', 19001, 19001), ('235435', 'M-a', 2001, 2001),
       ('389538', 'M-b', 11002, 11004), ('122660', 'M-b', 9014, 9013),
       ('360535', 'M-b', 7022, 7022), ('268067', 'M-d', 15007, 15004),
       ('111412', 'M-c', 18001, 18001), ('271851', 'BREAK', 23001, 23001)]


def tgt(b, vid):
    for r in b.get('rows', []):
        if r['vertex_id'] == vid:
            return np.array([r['x'], r['y'], r['z']])
    c = b.get('hv_cloud') or {}
    for i, v in enumerate(c.get('vertex_ids', [])):
        if v == vid:
            return np.array([c['x'][i], c['y'][i], c['z'][i]])
    return None


def main():
    print(f"{'evt':>7s} {'cls':>5s} {'arm':>3s} | {'S@tgt':>7s} {'S_peak':>7s} {'peak-tgt':>9s} | "
          f"{'S@tgt/S_peak':>12s} | {'#vox S>0.5*Speak':>16s} {'#>2cm from tgt':>14s}")
    for e, cls, ton, toff in EXH:
        for lab, vid in (('ON', ton), ('OFF', toff)):
            cloud, b = load_cloud(ARM[lab], e)
            f = field(cloud)
            t = tgt(b, vid)
            d = np.linalg.norm(f[:, :3] - t[None, :], axis=1)
            s_t = float(f[d < 2.0, 3].max()) if (d < 2.0).any() else float('nan')
            i = int(np.argmax(f[:, 3])); s_p = float(f[i, 3])
            dpk = float(np.linalg.norm(f[i, :3] - t))
            hi = f[:, 3] > 0.5 * s_p
            far = hi & (d > 2.0)
            print(f"{e:>7s} {cls:>5s} {lab:>3s} | {s_t:7.4f} {s_p:7.4f} {dpk:9.2f} | "
                  f"{s_t/s_p if s_p else float('nan'):12.3f} | {int(hi.sum()):16d} {int(far.sum()):14d}")
        print()


if __name__ == '__main__':
    main()
