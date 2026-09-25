#!/usr/bin/env python3
"""doc pdvd/118 sec 5.3-5.5: is jjo's certified deposit the triggering beam
particle?  Needs out/beam_flash.tsv (beam_flash.py) and out/light_events.tsv.
  A. cert cluster vs its Q/L flash and vs the beam flash (KS, chi2/ndf, pred/meas)
     and the implied depth below the top anode under both t0 hypotheses;
  B. the y-z principal axis and perpendicular offset of the points inside the
     cert tick slab (+-5 cm raw x) -> common direction? common line?
  C. clusters Q/L matches to the beam flash: size, angle to the common axis, offset;
  D. horizontal beam-parallel long tracks per event, 39305 vs 039349 q35flip."""
import csv, glob, json, os
import numpy as np
KDIR = os.path.dirname(os.path.abspath(__file__)); PDVD = os.path.dirname(KDIR)
rows = {int(r['event']): r for r in csv.DictReader(open(f'{KDIR}/out/beam_flash.tsv'), delimiter='\t')}
lt = {int(r['event']): r for r in csv.DictReader(open(f'{KDIR}/out/light_events.tsv'), delimiter='\t') if r['run'] == '39305'}
def dump(e): return json.load(open(f'{PDVD}/work/039305_{e}/calib-evt{e}.json'))
def axis(P):
    a = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
    return a if a[1] >= 0 else -a
print('A. cert cluster: selected flash vs beam flash')
axes = {}; offs = {}
for e in sorted(rows):
    r = rows[e]; d = dump(e); v = d['drift_speed']; A = d['geometry']['4']['anode_x']
    tb = float(lt[e]['trig_us']) + d['trigger_offsets_us'][0]
    fl = {f['id']: f for f in d['flashes']}
    cert = int(r['cert_clusters'].split('(')[0]); c = [c for c in d['clusters'] if c['uid'] == cert][0]
    x = np.array(c['x']); xr = np.median(x)
    sel = [b for b in d['bundles'] if b['main_cluster'] == cert and b['auto_selected']]
    bid = int(r['beam_flash_id'])
    vb = [b for b in d['bundles'] if b['main_cluster'] == cert and b['flash_id'] == bid]
    s = sel[0]; tm = fl[s['flash_id']]['time']
    line = (f'  {e:>6}: selected flash dt {tm - tb:+6.0f} us  KS {s["ks_dis"]:.2f} pred/meas {s["total_pred_light"] / s["total_PE"]:.2f}'
            f'  depth {A - (xr + tm * v):5.1f} cm | at beam flash: ')
    line += (f'KS {vb[0]["ks_dis"]:.2f} chi2/ndf {vb[0]["chi2"] / max(vb[0]["ndf"], 1):5.1f} pred {vb[0]["total_pred_light"]:6.0f} vs meas {vb[0]["total_PE"]:6.0f}'
             if vb else 'no beam flash / not a candidate') + f'  depth {A - (xr + tb * v):5.1f} cm'
    print(line)
    m = np.abs(x - float(r['cert_xraw'])) < 5
    P = np.c_[np.array(c['y'])[m], np.array(c['z'])[m]]
    axes[e] = axis(P); offs[e] = P.mean(0)
M = np.mean(list(axes.values()), axis=0); M /= np.linalg.norm(M); N = np.array([M[1], -M[0]])
print(f'B. cert-slab axis: mean (y,z) = ({M[0]:+.3f},{M[1]:+.3f}), {np.degrees(np.arctan2(M[1], -M[0])):.1f} deg from -y;'
      f' per-event deviation (deg) {[round(float(np.degrees(np.arccos(min(1, abs(a @ M))))), 1) for a in axes.values()]}')
print(f'   perpendicular offsets (cm): {[(e, round(float(N @ o))) for e, o in offs.items()]}')
print('C. clusters Q/L matches to the beam flash')
for e in sorted(rows):
    bid = int(rows[e]['beam_flash_id']); d = dump(e)
    if bid < 0: print(f'  {e:>6}: no beam flash'); continue
    v = d['drift_speed']; A = d['geometry']['4']['anode_x']; tb = {f['id']: f for f in d['flashes']}[bid]['time']
    cl = {c['uid']: c for c in d['clusters']}; out = []
    for b in [b for b in d['bundles'] if b['auto_selected'] and b['flash_id'] == bid]:
        c = cl[b['main_cluster']]; P = np.c_[c['y'], c['z']]; x = np.array(c['x'])
        if len(P) < 30: out.append(f'{c["uid"]} (n={len(P)}, tiny)'); continue
        dep = A - (x + tb * v)
        out.append(f'{c["uid"]} (n={len(P)}, {np.degrees(np.arccos(abs(axis(P) @ M))):.0f} deg, offset {N @ P.mean(0):+.0f} cm,'
                   f' depth [{dep.min():.0f},{dep.max():.0f}], KS {b["ks_dis"]:.2f}, pred/meas {b["total_pred_light"] / b["total_PE"]:.2f})')
    print(f'  {e:>6}: ' + ('; '.join(out) if out else 'none'))
print('D. horizontal (x-span < 0.25 y-z length) long (>= 1 m) top clusters within 10 deg of the axis, per event')
for name, files in (('039305 kaon', sorted(glob.glob(f'{PDVD}/work/039305_*/calib-evt*.json'))),
                    ('039349 q35flip', sorted(glob.glob(f'{PDVD}/work/039349_*_q35flip/calib-evt*.json')))):
    T = H = B = 0
    for p in files:
        for c in json.load(open(p))['clusters']:
            if c['apa'] != 4 or c['npoints'] < 300: continue
            P = np.c_[c['y'], c['z']]; a = axis(P); L = np.ptp(P @ a)
            if L < 100: continue
            x = np.array(c['x']); hor = (np.quantile(x, .98) - np.quantile(x, .02)) < 0.25 * L
            T += 1; H += hor; B += hor and np.degrees(np.arccos(abs(a @ M))) < 10
    n = len(files)
    print(f'  {name} ({n} evts): long {T / n:.1f}/evt, horizontal {H / n:.1f}/evt, horizontal+beam-parallel {B / n:.1f}/evt')
