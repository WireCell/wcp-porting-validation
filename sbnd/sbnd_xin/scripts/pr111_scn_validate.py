#!/usr/bin/env python3
"""doc pr/111 sec 3 -- GATE: the offline net must reproduce the live voxels[] top-5.

Every later pr/111 number depends on this.  If it does not reproduce, stop.

Repro:  python3 scripts/pr111_scn_validate.py
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr111_scn_lib import load_cloud, infer

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {'ON': os.path.join(H, 'work-vtx106-harv-base-nuecc48'),
        'OFF': os.path.join(H, 'work-vtx106-harv-nofitx-nuecc48')}

def main():
    evts = sorted((os.path.basename(p).replace('pr_evt', '')
                   for p in glob.glob(ARMS['ON'] + '/pr_evt*')), key=int)
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    nok = nbad = 0
    worst_s = worst_d = 0.0
    for lab, arm in ARMS.items():
        for e in evts[:want]:
            try:
                cloud, b = load_cloud(arm, e)
            except Exception as ex:
                print(f'{lab} {e}: {ex}'); continue
            live = b.get('voxels') or []
            if not live:
                continue
            got = infer(cloud, top_k=len(live))
            ds = max(abs(g[3] - l['dl_score']) for g, l in zip(got, live))
            dd = max(float(np.linalg.norm(np.array(g[:3]) - np.array([l['x'], l['y'], l['z']])))
                     for g, l in zip(got, live))
            ok = ds < 1e-6 and dd < 1e-3
            nok += ok; nbad += (not ok)
            worst_s = max(worst_s, ds); worst_d = max(worst_d, dd)
            print(f"{lab:>3s} {e:>7s}  n={len(live)}  max|dscore|={ds:.3e}  max|dpos|={dd:.3e} cm  {'OK' if ok else 'MISMATCH'}")
    print(f"\nGATE: {nok} reproduced, {nbad} mismatched;  worst |dscore|={worst_s:.3e}  worst |dpos|={worst_d:.3e} cm")
    return 0 if nbad == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
