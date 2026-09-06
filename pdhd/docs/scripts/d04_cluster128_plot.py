#!/usr/bin/env python3
"""doc pdhd/04 sec 8 -- why the TGM chord-charge guard vetoes 029107/1079 cluster 128.

Repro:
  python3 docs/scripts/d04_cluster128_plot.py \
      docs/figs/d04_tgm_path_cluster128.csv.gz docs/figs/d04_exclusion_by_apa.tsv \
      -o docs/figs/d04_cluster128_path_veto.png

Inputs are the env-gated TGMPROBE dumps (WCT_TGM_PATH_DUMP), one row per 3-D point of
the cluster: x,y,z (cm, t0-corrected -- the same frame Bee draws), the excluded flag
that `TaggerCheckTGM::path_components` honours, the path component id, and the
per-plane charges the exclusion is computed from.
"""
import argparse, csv, gzip, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('csv'); ap.add_argument('apatsv')
ap.add_argument('-o', '--out', default='d04_cluster128_path_veto.png')
a = ap.parse_args()

op = gzip.open if a.csv.endswith('.gz') else open
rows = list(csv.DictReader(op(a.csv, 'rt')))
P = np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in rows])
ex = np.array([int(r['excluded']) for r in rows]).astype(bool)
cp = np.array([int(r['comp']) for r in rows])
q = np.array([[float(r['qu']), float(r['qv']), float(r['qw'])] for r in rows])
ng = (q > 10).sum(1)

fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.4))
fig.suptitle('PDHD run 029107 event 1079 (evt 12), cluster 128 — why TaggerCheckTGM vetoed a 790 cm through-going muon',
             fontsize=13, y=0.985)

# --- A: what Bee shows vs what the guard walks -------------------------------
ax = axes[0, 0]
ax.scatter(P[:, 2], P[:, 1], s=0.4, c='#c9ccd4', lw=0, label=f'excluded from the walk  ({ex.sum()}, {ex.mean():.0%})')
ax.scatter(P[~ex, 2], P[~ex, 1], s=0.4, c='#1f77b4', lw=0, label=f'kept  ({(~ex).sum()})')
ax.set_xlabel('z [cm]'); ax.set_ylabel('y [cm]')
ax.set_title('A.  Every point Bee draws (grey+blue) vs the points\n'
             '$\\it{path\\_components}$ actually walks (blue)', fontsize=10.5)
ax.legend(markerscale=14, fontsize=8.5, loc='upper left')
ax.axvspan(80, 200, color='#d62728', alpha=0.07)
ax.text(140, P[:, 1].min() + 12, '130 cm: both induction planes read 0', ha='center',
        fontsize=8.5, color='#a01c1c')

# --- B: the surviving components ---------------------------------------------
ax = axes[0, 1]
sizes = collections.Counter(cp[~ex].tolist())
cols = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']
for i, (c, n) in enumerate(sizes.most_common()):
    m = (~ex) & (cp == c)
    ax.scatter(P[m, 2], P[m, 1], s=0.6, lw=0, c=cols[i % len(cols)],
               label=f'component {c}  ({n} pts)')
ax.scatter(P[ex, 2], P[ex, 1], s=0.3, c='#e8e9ec', lw=0, zorder=0)
ax.set_xlabel('z [cm]'); ax.set_ylabel('y [cm]')
ax.set_title('B.  The 30 cm-step walk on the kept points splits the track:\n'
             'the two ends land in DIFFERENT components → every pair vetoed', fontsize=10.5)
ax.legend(markerscale=10, fontsize=8.5, loc='upper left')
ends = [(-306.1, 120.3, 0.7), (60.3, 561.8, 460.7)]
for (ex_, ey, ez), lab in zip(ends, ['end A (comp 2)', 'end B (comp 0)']):
    ax.plot(ez, ey, marker='*', ms=17, mfc='k', mec='w', mew=1.0, zorder=5)
    ax.annotate(lab, (ez, ey), textcoords='offset points', xytext=(6, -14), fontsize=8.5)

# --- C: the profile along z ---------------------------------------------------
ax = axes[1, 0]
edges = np.arange(0, 481, 10)
mid = 0.5 * (edges[:-1] + edges[1:])
frac, mult = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (P[:, 2] >= lo) & (P[:, 2] < hi)
    frac.append(ex[m].mean() if m.sum() else np.nan)
    mult.append(ng[m].mean() if m.sum() else np.nan)
ax.plot(mid, np.array(frac) * 100, color='#d62728', lw=1.8, label='excluded fraction [%]')
ax.set_xlabel('z [cm]'); ax.set_ylabel('excluded fraction [%]', color='#d62728')
ax.tick_params(axis='y', labelcolor='#d62728'); ax.set_ylim(-3, 103)
ax2 = ax.twinx()
ax2.plot(mid, mult, color='#1f77b4', lw=1.8, label='mean planes with charge > 10')
ax2.axhline(2, color='#1f77b4', ls=':', lw=1.2)
ax2.set_ylabel('mean # planes with charge > 10', color='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#1f77b4'); ax2.set_ylim(0, 3.1)
ax2.text(300, 2.06, 'is_point_good threshold (2 of 3)', color='#1f77b4', fontsize=8)
ax.axvspan(80, 200, color='#d62728', alpha=0.07)
ax.set_title('C.  Along the track the exclusion is not scattered — it is one\n'
             'contiguous 130 cm run where only the collection plane has charge', fontsize=10.5)

# --- D: population, per APA ---------------------------------------------------
ax = axes[1, 1]
tsv = list(csv.DictReader(open(a.apatsv), delimiter='\t'))
apa = [int(r['apa']) for r in tsv]
excl = [100 * float(r['excl_frac']) for r in tsv]
p1 = [100 * float(r['p1']) for r in tsv]
w = 0.38
ax.bar([x - w/2 for x in apa], excl, w, color='#d62728', label='points excluded [%]')
ax.bar([x + w/2 for x in apa], p1, w, color='#7f7f7f', label='points with only 1 plane [%]')
for x, v in zip(apa, excl):
    ax.text(x - w/2, v + 0.6, f'{v:.1f}', ha='center', fontsize=9)
ax.set_xticks(apa)
ax.set_xticklabels(['APA0\nnp04hd-garfield\n(different FR)', 'APA1\ndune-1d565',
                    'APA2\ndune-1d565', 'APA3\ndune-1d565'], fontsize=8.5)
ax.set_ylabel('[%] of 3-D points'); ax.set_ylim(0, 34)
ax.legend(fontsize=8.5, loc='upper right')
ax.set_title('D.  Population (6 events, 776k points): the deficit is detector-wide,\n'
             'NOT an APA0 field-response effect — 26.2 % vs 19.8–25.1 %', fontsize=10.5)

fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(a.out, dpi=135)
print('wrote', a.out)
