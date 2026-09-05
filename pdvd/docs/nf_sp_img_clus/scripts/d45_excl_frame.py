#!/usr/bin/env python3
"""doc pdvd/45 -- read TrackFitting's exclusion-decision dump (env WCT_EXCL_DUMP, doc
pr/109 sec 9; one row per candidate 2-D cell per interior fit point per pass:
  call stage plane apa face wire time q flag min_dis_track min_other keep floored nseg px py pz
with '# call N excl=E cluster=C' headers) and test the drift-frame hypothesis:
if update_association compares a cell built in the RAW drift frame against segment
clouds in the t0-CORRECTED frame, the own-segment distance min_dis_track is ~ v_drift*t0
of the cluster (metres on a PDVD cosmic) instead of sub-cm, no cell is ever inside the
0.3 cm floor, and the keep decision is decided by x-reach, not by proximity.

Per cluster (attributed by the fit point (px,py,pz) -> nearest clustering-global point of
the arm's mabc-pr.zip, so the id is the Bee cluster_id; the header's cluster= is the
Facade ident and is reported alongside):
  rows, median/IQR min_dis_track [cm], median min_other, floored frac, kept frac,
  predicted v*t0 from tracking-pr.root T_cluster.flash_time_us, ratio median/pred.
Usage: d45_excl_frame.py <excl_dump.txt> <workdir> [--vdrift 0.148073]
"""
import sys, os, json, zipfile, argparse
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument('dump'); ap.add_argument('workdir')
ap.add_argument('--vdrift', type=float, default=0.148073, help='cm/us')
ap.add_argument('--tsv')
ap.add_argument('--trig-bot', type=float, default=None, help='trigger_offset_bot_us from the pctree .tlas (x<0 side); default: read the sidecar')
ap.add_argument('--trig-top', type=float, default=None, help='trigger_offset_top_us (x>0 side)')
a = ap.parse_args()

rows, hdr = [], None
for line in open(a.dump):
    if line.startswith('#'):
        parts = line.split()
        hdr = int(parts[4].split('=')[1]) if len(parts) > 4 and parts[4].startswith('cluster=') else -1
        continue
    f = line.split()
    if len(f) < 17: continue
    rows.append((hdr, int(f[1]), int(f[2]), float(f[9]), float(f[10]), int(f[11]), int(f[12]), int(f[13]),
                 float(f[14]), float(f[15]), float(f[16])))
R = np.array(rows, dtype=float)
print(f'{len(R)} decision rows, {len(set(R[:,0]))} distinct header idents')

z = zipfile.ZipFile(os.path.join(a.workdir, 'mabc-pr.zip'))
cl = json.loads(z.read('data/0/0-clustering-global.json'))
C = np.c_[cl['x'], cl['y'], cl['z']]; cid = np.asarray(cl['cluster_id'])
d, idx = cKDTree(C).query(R[:, 8:11])
bee = cid[idx]
t0 = {}
try:
    import uproot
    t = uproot.open(os.path.join(a.workdir, 'tracking-pr.root'))['T_cluster'].arrays(['cluster_id', 'flash_time_us'], library='np')
    t0 = {int(c): float(v) for c, v in zip(t['cluster_id'], t['flash_time_us'])}
except Exception as e:
    print('no T_cluster t0:', e)
# The cluster t0 the transform applies is the flash time RELATIVE TO THE TRIGGER:
# flash_time_us + trigger_offset_<side>_us (the sidecar's TLAs; -2493/-2465 us on
# 039252).  Read them unless given.
import glob
trig = {'bot': a.trig_bot, 'top': a.trig_top}
for tl in glob.glob(os.path.join(a.workdir, 'pctree-evt*.tlas')):
    for line in open(tl):
        k, _, v = line.strip().partition('=')
        if k == 'trigger_offset_bot_us' and trig['bot'] is None: trig['bot'] = float(v)
        if k == 'trigger_offset_top_us' and trig['top'] is None: trig['top'] = float(v)
print('trigger offsets (us):', trig)
def pred_cm(c, m):
    side = 'top' if np.median(R[m, 8]) > 0 else 'bot'
    tt = t0.get(int(c), float('nan'))
    off = trig[side] if trig[side] is not None else 0.0
    return a.vdrift * abs(tt + off)

out = open(a.tsv, 'w') if a.tsv else None
if out: out.write('cluster\tident\trows\tmed_own_cm\tq25\tq75\tmed_other_cm\tfloored\tkept\tpred_vt0_cm\tratio\n')
print(' cluster ident   rows  med_own[cm]   q25     q75  med_other floored  kept   v*t0[cm]  med/pred   (t0 = flash + trigger offset)')
for c in sorted(set(bee.tolist())):
    m = bee == c
    if m.sum() < 20: continue
    own, oth = R[m, 3], R[m, 4]
    idents = sorted(set(R[m, 0].astype(int).tolist()))
    pred = pred_cm(c, m)
    med = np.median(own)
    q25, q75 = np.percentile(own, [25, 75])
    fl, kp = R[m, 6].mean(), R[m, 5].mean()
    print(f' {int(c):7d} {",".join(map(str, idents)):>5s} {m.sum():6d} {med:11.2f} {q25:7.2f} {q75:7.2f} {np.median(oth[oth < 1e8]) if (oth < 1e8).any() else float("nan"):10.2f} {fl:7.3f} {kp:5.2f} {pred:10.1f} {med / pred if pred == pred and pred > 0 else float("nan"):9.3f}')
    if out: out.write(f'{int(c)}\t{",".join(map(str, idents))}\t{m.sum()}\t{med:.3f}\t{q25:.3f}\t{q75:.3f}\t{np.median(oth):.3f}\t{fl:.4f}\t{kp:.4f}\t{pred:.2f}\t{med / pred if pred > 0 else float("nan"):.4f}\n')
print(f'ALL: floored frac {R[:,6].mean():.4f}, kept frac {R[:,5].mean():.3f}, min own distance seen {R[:,3].min():.3f} cm, rows with own < 1 cm: {(R[:,3] < 1).sum()}')
