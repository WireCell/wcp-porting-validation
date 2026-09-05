#!/usr/bin/env python3
"""doc pdvd/45 -- downstream movers between two PR arms over a manifest: per event, the
neutrino candidates (T_tagger rows, matched by cluster_id), their vertex shift, and the
kine_reco_Enu change (T_kine, same row order as T_tagger); plus the pre-dQ/dx drop
totals from the logs and the wall time from pr_resource_*.txt when present.
Usage: d45_downstream.py <base_tag> <arm_tag> [events.txt] [--tsv out.tsv]
"""
import sys, os, re, glob, argparse
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('base'); ap.add_argument('arm'); ap.add_argument('events', nargs='?')
ap.add_argument('--tsv')
a = ap.parse_args()
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
W = os.path.join(PDVD, 'work')
ev_file = a.events or os.path.join(PDVD, 'stm', 'events.txt')
events = ['%06d_%s' % (int(l.split()[0]), l.split()[1]) for l in open(ev_file) if l.strip() and not l.startswith('#')]
import uproot

def cands(wd):
    p = os.path.join(wd, 'tracking-pr.root')
    if not os.path.exists(p): return None
    f = uproot.open(p)
    if 'T_tagger' not in f or 'T_kine' not in f: return {}
    t = f['T_tagger'].arrays(['cluster_id', 'nu_x', 'nu_y', 'nu_z', 'nu_index'], library='np')
    k = f['T_kine'].arrays(['kine_reco_Enu'], library='np')
    out = {}
    for i in range(len(t['cluster_id'])):
        out[int(t['cluster_id'][i])] = (float(t['nu_x'][i]), float(t['nu_y'][i]), float(t['nu_z'][i]),
                                        float(k['kine_reco_Enu'][i]) if i < len(k['kine_reco_Enu']) else np.nan)
    return out

def drops(wd):
    d = n = 0
    for lg in glob.glob(os.path.join(wd, 'wct_pr_*.log')):
        for line in open(lg, errors='replace'):
            m = re.search(r'dropped (\d+) of (\d+) trajectory', line)
            if m: d += int(m.group(1)); n += int(m.group(2))
    return d, n

def wall(wd):
    for f in glob.glob(os.path.join(wd, 'pr_resource_*.txt')):
        for line in open(f):
            m = re.search(r'wall[_ ]?s?[=: ]+([0-9.]+)', line)
            if m: return float(m.group(1))
    return np.nan

rows, D0, N0, D1, N1, w0s, w1s = [], 0, 0, 0, 0, [], []
same = gained = lost = 0; dv = []; dE = []
for e in events:
    b, r = os.path.join(W, f'{e}_{a.base}'), os.path.join(W, f'{e}_{a.arm}')
    cb, cr = cands(b), cands(r)
    if cb is None or cr is None:
        rows.append((e, 'MISSING')); continue
    d0, n0 = drops(b); d1, n1 = drops(r); D0 += d0; N0 += n0; D1 += d1; N1 += n1
    w0, w1 = wall(b), wall(r); w0s.append(w0); w1s.append(w1)
    for c in set(cb) | set(cr):
        if c in cb and c in cr:
            same += 1
            dv.append(np.linalg.norm(np.array(cb[c][:3]) - np.array(cr[c][:3])))
            dE.append(cr[c][3] - cb[c][3])
            rows.append((e, c, 'both', f'{dv[-1]:.2f}', f'{cb[c][3]:.1f}', f'{cr[c][3]:.1f}'))
        elif c in cb:
            lost += 1; rows.append((e, c, 'lost', '', f'{cb[c][3]:.1f}', ''))
        else:
            gained += 1; rows.append((e, c, 'gained', '', '', f'{cr[c][3]:.1f}'))
dv, dE = np.array(dv), np.array(dE)
print(f'{a.base} -> {a.arm}: {len(events)} events; candidates in both {same}, lost {lost}, gained {gained}')
print(f'  vertex shift [cm]: median {np.median(dv):.2f}, p90 {np.percentile(dv, 90):.2f}, max {dv.max():.2f}; moved > 1 cm: {(dv > 1).sum()}, > 5 cm: {(dv > 5).sum()}')
fin = dE[np.isfinite(dE)]
print(f'  Enu change [MeV] ({len(fin)} finite of {len(dE)}): median {np.median(fin):+.1f}, p10/p90 {np.percentile(fin, 10):+.1f}/{np.percentile(fin, 90):+.1f}; |dE| > 50 MeV: {(np.abs(fin) > 50).sum()}, > 200 MeV: {(np.abs(fin) > 200).sum()}')
print(f'  pre-dQ/dx drop: {D0}/{N0} = {D0 / N0 if N0 else float("nan"):.3f}  ->  {D1}/{N1} = {D1 / N1 if N1 else float("nan"):.3f}')
w0s, w1s = np.array(w0s), np.array(w1s)
if np.isfinite(w0s).any():
    print(f'  wall [s]: sum {np.nansum(w0s):.0f} -> {np.nansum(w1s):.0f}, median per event {np.nanmedian(w0s):.1f} -> {np.nanmedian(w1s):.1f}')
if a.tsv:
    with open(a.tsv, 'w') as o:
        o.write('event\tcluster\tstatus\tvertex_shift_cm\tEnu_base\tEnu_arm\n')
        for r in rows: o.write('\t'.join(str(x) for x in r) + '\n')
