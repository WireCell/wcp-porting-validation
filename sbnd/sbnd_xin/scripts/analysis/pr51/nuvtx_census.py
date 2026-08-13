#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/51 rounds 6-7: nu-candidate vertex + Enu census
between two arms, measured identically to the round-5/6 calibration rows --
T_tagger nu_{x,y,z} row 0 and T_kine kine_reco_Enu row 0 of each event's
tracking-pr.root (exactly one row = the nu candidate; never read the vertex
from the PF-root mode).

Usage: nuvtx_census.py <ARM_BASE> <ARM_NEW> [--vtx-cm 10] [--enu-mev 100]
"""
import sys, os, glob
import numpy as np
import uproot

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'

base, new = sys.argv[1], sys.argv[2]
vtx_cut = 10.0
enu_cut = 100.0
if '--vtx-cm' in sys.argv:
    vtx_cut = float(sys.argv[sys.argv.index('--vtx-cm') + 1])
if '--enu-mev' in sys.argv:
    enu_cut = float(sys.argv[sys.argv.index('--enu-mev') + 1])


def read(arm, evt):
    p = os.path.join(SB, arm, 'pr_evt%d' % evt, 'tracking-pr.root')
    if not os.path.exists(p):
        return None
    try:
        f = uproot.open(p)
        t = f['T_tagger']
        v = np.array([t['nu_x'].array()[0], t['nu_y'].array()[0], t['nu_z'].array()[0]])
        e = float(f['T_kine']['kine_reco_Enu'].array()[0])
        return v, e
    except Exception as ex:
        print('  [skip] %s evt %d: %s' % (arm, evt, ex), file=sys.stderr)
        return None


events = sorted(int(os.path.basename(d)[6:])
                for d in glob.glob(os.path.join(SB, base, 'pr_evt*')))
rows = []
for evt in events:
    a, b = read(base, evt), read(new, evt)
    if a is None or b is None:
        continue
    dv = float(np.linalg.norm(a[0] - b[0]))
    de = b[1] - a[1]
    rows.append((evt, dv, de))

nv = sum(1 for r in rows if r[1] > vtx_cut)
ne = sum(1 for r in rows if abs(r[2]) > enu_cut)
print('%s -> %s : %d events compared' % (base, new, len(rows)))
print('nu-vtx > %.0f cm : %d    |dEnu| > %.0f MeV : %d' % (vtx_cut, nv, enu_cut, ne))
big = [r for r in rows if r[1] > vtx_cut or abs(r[2]) > enu_cut]
for evt, dv, de in sorted(big, key=lambda r: -r[1]):
    print('  evt %-8d dvtx=%8.2f cm   dEnu=%+9.1f MeV' % (evt, dv, de))
sub = [r for r in rows if 0.01 < r[1] <= vtx_cut]
print('sub-threshold movers (0.01 < dvtx <= %.0f cm): %d' % (vtx_cut, len(sub)))
