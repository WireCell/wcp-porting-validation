#!/usr/bin/env python3
"""doc pr/51 scoping: dump the PR structure near the main vertex for one
arm/event -- main vertex, nearby PR vertices, and every segment
(real_cluster_id = cluster*1000+segment in shower_track-global) with a point
within RADIUS of the main vertex: point count, closest approach, arc extent,
shower/track flag, and image support (per-point distance to the nearest
clustering-global point of the same PR cluster -- large => ghost).

Usage: python3 vtx_struct.py <ARM_LABEL> <EVT> [RADIUS_CM]
"""
import sys, os, json, zipfile
import numpy as np

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
arm, evt = sys.argv[1], int(sys.argv[2])
RAD = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0

zp = os.path.join(SB, arm, 'pr_evt%d' % evt, 'mabc-pr.zip')
z = zipfile.ZipFile(zp)
def load(name):
    return json.loads(z.read('data/0/0-%s.json' % name))

vtx = load('vertices-global')
st = load('shower_track-global')
img = load('clustering-global')
fit = load('track_fit-global')

vp = np.array([vtx['x'], vtx['y'], vtx['z']]).T
vq = np.array(vtx['q'])
main = vp[vq == 15000]
print('arm=%s evt=%d  radius=%.1f cm' % (arm, evt, RAD))
if len(main) != 1:
    print('main-vertex candidates (q=15000): %d' % len(main))
mv = main[0]
print('main vertex: (%.2f, %.2f, %.2f)' % tuple(mv))

d = np.linalg.norm(vp - mv, axis=1)
near = np.argsort(d)
print('\nPR vertices within %.1f cm:' % RAD)
for i in near:
    if d[i] > RAD: break
    print('  (%.2f, %.2f, %.2f)  d=%.2f cm  q=%d' % (vp[i][0], vp[i][1], vp[i][2], d[i], int(vq[i])))

sp = np.array([st['x'], st['y'], st['z']]).T
sq = np.array(st['q'])
rcid = np.array(st['real_cluster_id'])
cid = np.array(st['cluster_id'])

ip = np.array([img['x'], img['y'], img['z']]).T
icid = np.array(img['cluster_id'])

fp = np.array([fit['x'], fit['y'], fit['z']]).T
fc = np.array(fit['real_cluster_id']) if 'real_cluster_id' in fit else None

print('\nsegments with a point within %.1f cm of the main vertex:' % RAD)
print('%-8s %-6s %5s %8s %8s %7s %7s %-7s %s' % ('rcid', 'clus', 'npts', 'dmin', 'extent', 'imgmax', 'imgavg', 'flag', 'far end'))
for r in sorted(set(rcid)):
    m = rcid == r
    pts = sp[m]
    dd = np.linalg.norm(pts - mv, axis=1)
    if dd.min() > RAD: continue
    c = int(cid[m][0])
    flag = 'shower' if sq[m][0] == 15000 else 'track'
    # image support vs ANY image charge (cluster_id here is the bundle id,
    # not the PR cluster, so a same-cluster lookup is unreliable): a fit
    # point far from every image point is a ghost.
    mx = 0.0; mean = 0.0
    for p in pts:
        dn = np.linalg.norm(ip - p, axis=1).min()
        mean += dn
        if dn > mx: mx = dn
    mean /= max(1, len(pts))
    far = pts[np.argmax(dd)]
    print('%-8d %-6d %5d %8.2f %8.2f %7.2f %7.2f %-7s (%.1f,%.1f,%.1f) d=%.1f' % (
        r, c, len(pts), dd.min(), dd.max() - dd.min(), mx, mean, flag,
        far[0], far[1], far[2], dd.max()))
