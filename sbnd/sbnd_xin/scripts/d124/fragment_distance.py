#!/usr/bin/env python3
"""doc sbnd_xin/124 -- for the 'fragment' flips: how many PR-input points differ between the arms and how far
they sit from the neutrino vertex of the arm that has them (3-D, using that arm's own t0-corrected x).

usage: fragment_distance.py <anatomy.tsv> <pr_root_A> <pr_root_B>
"""
import csv, json, os, sys, zipfile
import numpy as np, uproot
sys.path.insert(0, os.path.dirname(__file__))
def arm(d):
    z = zipfile.ZipFile(os.path.join(d, 'mabc-pr.zip')); n = [x for x in z.namelist() if x.endswith('-clustering-global.json')][0]
    cg = json.loads(z.read(n))
    t = uproot.open(os.path.join(d, 'tracking-pr.root'))['T_tagger'].arrays(['numu_score', 'cluster_id', 'act_cluster_id', 'act_in_pr', 'nu_x', 'nu_y', 'nu_z'], library='np')
    i = int(np.argmax(t['numu_score']))
    inp = set(int(c) for c, p in zip(t['act_cluster_id'][i], t['act_in_pr'][i]) if p == 1) | {int(t['cluster_id'][i])}
    v = np.array([t['nu_x'][i], t['nu_y'][i], t['nu_z'][i]], float)
    pts = {}
    for x, y, zz, q, c in zip(cg['x'], cg['y'], cg['z'], cg['q'], cg['cluster_id']):
        if c in inp:
            pts[(round(y, 1), round(zz, 1), round(q, 0))] = (x, y, zz)
    return pts, v
tsv, RA, RB = sys.argv[1:4]
for r in csv.DictReader(open(tsv), delimiter='\t'):
    if r['class'] != 'fragment':
        continue
    e = r['event']
    pa, va = arm(os.path.join(RA, 'pr_evt' + e)); pb, vb = arm(os.path.join(RB, 'pr_evt' + e))
    oa = [pa[k] for k in set(pa) - set(pb)]; ob = [pb[k] for k in set(pb) - set(pa)]
    da = [np.linalg.norm(np.array(p) - va) for p in oa]; db = [np.linalg.norm(np.array(p) - vb) for p in ob]
    print('%-6s %-6s only-A %3d pts (min/med dist to A vertex %s cm)  only-B %3d pts (%s cm)  dvtx %s  numu %s -> %s' % (
        e, r['kind'], len(oa), '%.0f/%.0f' % (min(da), np.median(da)) if da else '-', len(ob), '%.0f/%.0f' % (min(db), np.median(db)) if db else '-',
        r['dvtx_cm'], r['numu_A'], r['numu_B']))
