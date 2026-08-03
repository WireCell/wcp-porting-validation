#!/usr/bin/env python3
"""doc pr/12 sec 7: the two figures for "effect on the other PR steps".

Reads the at-HEAD census TSV (cathode_nu_census.py --merge output) and, for the
per-point dQ/dx profile, the same arms' tracking-pr.root T_rec_charge.

  fig 1  pr12_vertex_and_pid.png
     (a) neutrino-vertex |x| for the 57 crossing candidates, against the null
         built from each candidate's OWN fitted points -- "is the vertex pulled
         to the cathode?" is only answerable relative to where the vertex could
         have landed at all.
     (b) nearest PR graph vertex |x| for the 44 spanned events: the spurious
         junction in the dead gap.
     (c) particle_id on the two sides of the cathode for the 44 spanned events;
         every point on the diagonal == the hypothesis never flips.

  fig 2  pr12_dqdx_gap.png
     (a) fitted dQ/dx vs |x|, each event normalised to its own 3-15 cm median,
         pooled over the 44 spanned events: the notch.
     (b) the per-event ratio the doc quotes (median of |x|<3 over 3-15 cm).

dQ/dx = (q - dQdx_offset)/dQdx_scale / nq  (SbndPrMagnifyTrackingVisitor.cxx
:363-364,:497-498; Trun carries the scale/offset, 0.1 / -1000 on SBND).

Usage:
  python3 cathode_plots.py --census docs/pr/12_cathode-census.tsv \
      --root mcp1k=work-mcp1kall-cath01 --root nuecc48=work-nuecc48-cath01 \
      --outdir docs/pics
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import uproot

CATH = 'tab:red'
BAND = '#4c72b0'


def fnum(r, k):
    v = r.get(k, '')
    return float(v) if v not in ('', 'nan', None) else None


def load_points(root, evt):
    """Fitted segment points of the candidate: |x|, dQ/dx, dropping vertex rows."""
    p = os.path.join(root, f'pr_evt{evt}', 'tracking-pr.root')
    f = uproot.open(p)
    trun = f['Trun'].arrays(library='np')
    scale, off = float(trun['dQdx_scale'][0]), float(trun['dQdx_offset'][0])
    t = f['T_rec_charge'].arrays(library='np')
    m = (t['sub_cluster_id'] != -1) & (t['nq'] > 0)
    return np.abs(t['x'][m]), (t['q'][m] - off) / scale / t['nq'][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--census', required=True)
    ap.add_argument('--root', action='append', required=True,
                    help='sample=dir, repeatable')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    roots = dict(s.split('=', 1) for s in args.root)

    rows = list(csv.DictReader(open(args.census), delimiter='\t'))
    spanned = [r for r in rows if r['cls'] == 'spanned']

    # ---------------------------------------------------------------- figure 1
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) neutrino vertex |x| vs the null from the candidates' own fit points.
    nux = [fnum(r, 'nu_absx') for r in rows if r['nu_sentinel'] == '0']
    nux = np.array([v for v in nux if v is not None])
    # Null: each candidate's own fitted-point |x| distribution, i.e. where its
    # vertex could have landed.  Weight each event equally so a long track does
    # not dominate, then scale to the number of candidates.
    bins = np.arange(0, 205, 10)
    null = np.zeros(len(bins) - 1)
    null_frac = []
    for r in rows:
        ax_, _ = load_points(roots[r['sample']], r['event'])
        null_frac.append(float((ax_ < 5).mean()))
        h, _ = np.histogram(ax_, bins=bins)
        if h.sum():
            null += h / h.sum()
    exp5 = float(np.mean(null_frac)) * len(rows)
    obs5 = int((nux < 5).sum())
    ax[0].hist(nux, bins=bins, color=BAND, alpha=.85, edgecolor='white',
               label='reconstructed neutrino vertex')
    ax[0].step(bins[:-1], null, where='post', color='0.35', lw=1.8,
               label='null: the candidates’ own charge')
    ax[0].axvline(5, color=CATH, ls='--', lw=1.5)
    ax[0].legend(loc='center right', fontsize=8)
    ax[0].set_xlabel('neutrino vertex |x|  (cm from the cathode)')
    ax[0].set_ylabel('candidates')
    ax[0].set_title('(a) the vertex is not pulled to the cathode', fontsize=10)
    ax[0].text(.97, .95, f'within 5 cm of the cathode:\n'
                         f'observed {obs5}/{len(rows)}, expected {exp5:.1f}',
               transform=ax[0].transAxes, ha='right', va='top', fontsize=8.5)

    # (b) nearest PR graph vertex |x|, spanned events.
    vx = np.array([fnum(r, 'vtx_absx') for r in spanned
                   if fnum(r, 'vtx_absx') is not None])
    n3 = int((vx < 3).sum())
    ax[1].hist(vx, bins=np.arange(0, 41, 1.5), color=BAND, alpha=.85,
               edgecolor='white')
    ax[1].axvspan(0, 3, color=CATH, alpha=.15)
    ax[1].axvline(3, color=CATH, ls='--', lw=1.5)
    ax[1].set_xlabel('nearest PR graph vertex |x|  (cm)')
    ax[1].set_ylabel('spanned events')
    ax[1].set_title('(b) a spurious junction inside the dead gap', fontsize=10)
    ax[1].text(.97, .95, f'{n3}/{len(vx)} within 3 cm\nmedian {np.median(vx):.1f} cm',
               transform=ax[1].transAxes, ha='right', va='top', fontsize=9)

    # (c) particle_id either side of the cathode.
    lab = {11: 'e', 13: 'μ', 211: 'π', 2212: 'p', -1: 'none', 1: 'shwr',
           4: 'trk'}
    pn = np.array([int(r['pid_neg']) for r in spanned])
    pp = np.array([int(r['pid_pos']) for r in spanned])
    codes = sorted(set(pn.tolist()) | set(pp.tolist()))
    idx = {c: i for i, c in enumerate(codes)}
    rng = np.random.default_rng(0)          # jitter only, fixed seed
    jx = np.array([idx[c] for c in pn]) + rng.uniform(-.16, .16, len(pn))
    jy = np.array([idx[c] for c in pp]) + rng.uniform(-.16, .16, len(pp))
    same_sh = np.array([r['shower_neg'] == r['shower_pos'] for r in spanned])
    ax[2].plot([-.5, len(codes) - .5], [-.5, len(codes) - .5], color='0.7',
               lw=1, zorder=0)
    ax[2].scatter(jx[same_sh], jy[same_sh], s=34, color=BAND, alpha=.8,
                  label='track/shower flag also agrees')
    if (~same_sh).any():
        ax[2].scatter(jx[~same_sh], jy[~same_sh], s=40, color=CATH, marker='x',
                      label='track/shower flag differs')
    ax[2].set_xticks(range(len(codes)))
    ax[2].set_yticks(range(len(codes)))
    ax[2].set_xticklabels([lab.get(c, c) for c in codes])
    ax[2].set_yticklabels([lab.get(c, c) for c in codes])
    ax[2].set_xlabel('particle_id at x < 0')
    ax[2].set_ylabel('particle_id at x > 0')
    ax[2].set_title('(c) the hypothesis never flips across the gap', fontsize=10)
    ax[2].legend(loc='lower right', fontsize=8)
    nflip = int((pn != pp).sum())
    ax[2].text(.03, .95, f'particle_id differs: {nflip}/{len(spanned)}\n'
                         f'shower flag differs: {int((~same_sh).sum())}/{len(spanned)}',
               transform=ax[2].transAxes, va='top', fontsize=9)

    fig.suptitle('doc pr/12 sec 7 — vertex and PID across the SBND cathode '
                 f'({len(rows)} crossing candidates, {len(spanned)} spanned)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    p1 = os.path.join(args.outdir, 'pr12_vertex_and_pid.png')
    fig.savefig(p1, dpi=140)
    print('wrote', p1)

    # ---------------------------------------------------------------- figure 2
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    edges = np.arange(0, 20.5, 1.0)
    pooled = [[] for _ in range(len(edges) - 1)]
    for r in spanned:
        axs, dqdx = load_points(roots[r['sample']], r['event'])
        band = (axs >= 3) & (axs < 15)
        if not band.any():
            continue
        ref = np.median(dqdx[band])
        if ref <= 0:
            continue
        norm = dqdx / ref
        which = np.digitize(axs, edges) - 1
        for b, v in zip(which, norm):
            if 0 <= b < len(pooled):
                pooled[b].append(v)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([np.median(v) if v else np.nan for v in pooled])
    q1 = np.array([np.percentile(v, 25) if v else np.nan for v in pooled])
    q3 = np.array([np.percentile(v, 75) if v else np.nan for v in pooled])
    ax[0].fill_between(ctr, q1, q3, color=BAND, alpha=.25, label='25–75%')
    ax[0].plot(ctr, med, color=BAND, lw=2, label='median')
    ax[0].axhline(1.0, color='0.5', ls=':', lw=1)
    ax[0].axvspan(0, 0.45, color=CATH, alpha=.18)
    ax[0].axvline(3, color=CATH, ls='--', lw=1.2)
    ax[0].set_xlabel('|x|  (cm from the cathode)')
    ax[0].set_ylabel('fitted dQ/dx  /  the event’s own 3–15 cm median')
    ax[0].set_title('(a) the dQ/dx notch in the dead gap', fontsize=10)
    ax[0].text(.04, .95, 'shaded: inactive |x| < 0.45 cm',
               transform=ax[0].transAxes, fontsize=8.5, color=CATH, va='top')
    ax[0].legend(loc='lower right', fontsize=8.5)

    ratio = np.array([fnum(r, 'dqdx_ratio') for r in spanned
                      if fnum(r, 'dqdx_ratio') is not None])
    ax[1].hist(ratio, bins=np.arange(0.3, 1.45, .05), color=BAND, alpha=.85,
               edgecolor='white')
    ax[1].axvline(np.median(ratio), color=CATH, lw=2,
                  label=f'median {np.median(ratio):.2f}')
    ax[1].axvline(1.0, color='0.5', ls=':', lw=1.2, label='no suppression')
    ax[1].set_xlabel('median dQ/dx in |x| < 3 cm  /  in 3–15 cm')
    ax[1].set_ylabel('spanned events')
    ax[1].set_title('(b) per-event suppression', fontsize=10)
    ax[1].text(.03, .95, f'{int((ratio < 0.8).sum())}/{len(ratio)} below 0.8\n'
                         f'{int((ratio < 0.6).sum())}/{len(ratio)} below 0.6',
               transform=ax[1].transAxes, va='top', fontsize=9)
    ax[1].legend(loc='upper right', fontsize=8.5)

    fig.suptitle('doc pr/12 sec 7 — fitted dQ/dx through the cathode dead gap '
                 f'({len(spanned)} spanned candidates)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))
    p2 = os.path.join(args.outdir, 'pr12_dqdx_gap.png')
    fig.savefig(p2, dpi=140)
    print('wrote', p2)

    print(f'\nnu vertex within 5 cm: observed {obs5}/{len(rows)}, '
          f'expected {exp5:.1f} from the candidates’ own charge')
    print(f'nearest PR vertex within 3 cm: {n3}/{len(vx)} spanned')
    print(f'dQ/dx ratio: median {np.median(ratio):.3f}, '
          f'{int((ratio < 0.8).sum())} below 0.8, {int((ratio < 0.6).sum())} below 0.6')
    print(f'profile minimum: {np.nanmin(med):.3f} at |x| = {ctr[np.nanargmin(med)]:.1f} cm')


if __name__ == '__main__':
    main()
