#!/usr/bin/env python3
"""doc 85: reconstructed-energy and BDT-score distributions of the four SBND
production samples, from the prod0825 full-sample PR arms.

Input is what pr_scores_table.py writes -- one row per event, the T_tagger /
T_kine scalars of the PR job:

  products/prod0825/<sample>-scores-prod0825.tsv

Population ("the ones with PR"): nu_evaluated == 1, i.e. the PR log carries
"TaggerCheckNeutrino: selected main cluster ...".  Only then are TaggerInfo /
KineInfo filled for real; otherwise the ROOT row holds struct DEFAULTS and
pr_scores_table.py blanks it (see its module docstring).

Two figures, both written to docs/85_dists/:

  d85_enu.png    kine_reco_Enu_MeV, four samples overlaid.  Left panel is
                 area-normalised (the shapes -- 18 vs 879 events otherwise
                 makes two samples invisible); right panel is raw counts on a
                 log y.  Last bin is an explicit OVERFLOW bin, not a clip.

  d85_bdt.png    nue_score vs numu_score.  Split vertically because
                 nue_score = -15 is NOT missing data: UbooneNueBDTScorer.cxx
                 sets it when br_filled != 1 ("background-like default,
                 matches prototype default_val = -15"), which is ~95 % of the
                 evaluated numu-sample events.  Top panel = the filled
                 subset, where the real 2-D structure is; bottom strip = the
                 -15 population at its true numu_score.  Both BDTs are
                 log-odds of a clamped xgboost output, so they saturate at
                 +-4.301 -- the pile-ups at the frame edges are the clamp,
                 not a defect.

Usage:
  python3 scripts/analysis/d85_dists.py [--prod prod0825] [--outdir docs/85_dists]
"""
import argparse
import csv
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SAMPLES = [
    # key,      legend label,                     colour,    marker
    ('ncpi0',   'NC$\\pi^0$ (19)',                '#d62728', 'o'),
    ('nuecc48', r'$\nu_e$CC (48)',                '#2ca02c', 's'),
    ('mcp1k',   r'$\nu_\mu$ 1000',                '#1f77b4', '^'),
    ('mcp2k',   r'$\nu_\mu$ 2000',                '#ff7f0e', 'v'),
]

# log-odds clamp of both BDTs: val1 is clipped to +-0.9999 before
# log10((1+v)/(1-v)), so |score| <= 4.30103.
CLAMP = 4.30103
NUE_UNFILLED = -15.0

E_LO, E_HI, E_BW = 0.0, 3000.0, 200.0   # MeV; >E_HI goes into the overflow bin


def load(prod, sample):
    """Evaluated rows only, as floats.  Blank kine_reco_Enu_MeV is kept out of
    the energy list but the event still counts as evaluated -- both numbers are
    reported."""
    path = os.path.join('products', prod, f'{sample}-scores-{prod}.tsv')
    with open(path) as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t')
                if r['nu_evaluated'] == '1']
    out = {'n_eval': len(rows), 'enu': [], 'numu': [], 'nue': [],
           'n_cosmic_tagged': sum(1 for r in rows if r['event_label'] == 'cosmic-tagged')}
    for r in rows:
        if r['kine_reco_Enu_MeV']:
            out['enu'].append(float(r['kine_reco_Enu_MeV']))
        if r['numu_score'] and r['nue_score']:
            out['numu'].append(float(r['numu_score']))
            out['nue'].append(float(r['nue_score']))
    out['n_enu_blank'] = out['n_eval'] - len(out['enu'])
    return out


def overflow_hist(vals):
    """Counts in [E_LO, E_HI) plus an explicit overflow bin for >= E_HI."""
    nb = int(round((E_HI - E_LO) / E_BW))
    edges = [E_LO + i * E_BW for i in range(nb + 2)]   # + one overflow bin
    counts = [0] * (nb + 1)
    for v in vals:
        i = int((v - E_LO) / E_BW)
        counts[min(max(i, 0), nb)] += 1
    return edges, counts


def step_xy(edges, counts):
    x, y = [], []
    for i, c in enumerate(counts):
        x += [edges[i], edges[i + 1]]
        y += [c, c]
    return x, y


def fig_enu(data, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, mode in zip(axes, ('norm', 'count')):
        for key, lab, col, _ in SAMPLES:
            d = data[key]
            edges, counts = overflow_hist(d['enu'])
            n = sum(counts)
            y = [c / n / (E_BW / 1000.0) for c in counts] if mode == 'norm' else counts
            xs, ys = step_xy(edges, y)
            med = st.median(d['enu']) if d['enu'] else float('nan')
            ax.plot(xs, ys, color=col, lw=1.8,
                    label=f'{lab}  N={n}, med {med:.0f} MeV')
        ax.axvline(E_HI, color='0.5', ls=':', lw=1)
        ax.set_xlabel('reconstructed $E_\\nu$  [MeV]   (kine_reco_Enu)')
        ax.set_xlim(E_LO, E_HI + E_BW)
        ax.grid(alpha=0.25)
        if mode == 'norm':
            ax.set_ylabel('events / GeV, area-normalised')
            ax.set_title('shape (area-normalised)')
            ax.legend(fontsize=9, loc='upper right')
        else:
            ax.set_yscale('log')
            ax.set_ylabel('events / 200 MeV')
            ax.set_title('raw counts (log $y$)')
    from matplotlib.transforms import blended_transform_factory as blend
    for ax in axes:
        ax.text(E_HI + E_BW / 2, 0.45, 'overflow', rotation=90,
                ha='center', va='center', fontsize=8, color='0.45',
                transform=blend(ax.transData, ax.transAxes))
    fig.suptitle('SBND prod0825 -- reconstructed neutrino energy of the PR-evaluated events '
                 '(nu_evaluated = 1)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(outdir, 'd85_enu.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_bdt(data, outdir):
    fig = plt.figure(figsize=(9.2, 9.0))
    gs = GridSpec(2, 1, height_ratios=[3.4, 1.35], hspace=0.24, figure=fig)
    top, bot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    hb = [-4.85 + i * 0.35 for i in range(29)]   # numu_score bins for the strip
    for key, lab, col, mk in SAMPLES:
        d = data[key]
        fx = [x for x, y in zip(d['numu'], d['nue']) if y > NUE_UNFILLED + 0.1]
        fy = [y for y in d['nue'] if y > NUE_UNFILLED + 0.1]
        ux = [x for x, y in zip(d['numu'], d['nue']) if y <= NUE_UNFILLED + 0.1]
        top.scatter(fx, fy, s=36, marker=mk, facecolor='none', edgecolor=col,
                    linewidths=1.3, alpha=0.9,
                    label=f'{lab}   nue BDT filled {len(fx)} / {len(d["nue"])}')
        if ux:
            bot.hist(ux, bins=hb, histtype='step', color=col, lw=1.7,
                     label=f'{lab}   N={len(ux)}')

    for ax in (top, bot):
        ax.axvline(0.0, color='0.75', lw=0.9)
        for s in (-CLAMP, CLAMP):
            ax.axvline(s, color='0.6', ls='--', lw=0.9)
        ax.set_xlim(-4.85, 4.85)
        ax.grid(alpha=0.25)
    top.axhline(0.0, color='0.75', lw=0.9)
    top.axhline(CLAMP, color='0.6', ls='--', lw=0.9)
    top.axhline(-CLAMP, color='0.6', ls='--', lw=0.9)
    top.set_ylim(-4.85, 4.85)
    top.set_ylabel('nue_score   (log-odds)')
    top.set_xticklabels([])
    top.set_title('SBND prod0825 -- $\\nu_e$ vs $\\nu_\\mu$ BDT score, PR-evaluated events\n'
                  'dashed frame = the $\\pm$4.301 log-odds clamp (rows of points ON it '
                  'are saturated, not degenerate)', fontsize=10.5)
    top.legend(fontsize=9, loc='lower left', framealpha=0.9)

    bot.set_yscale('log')
    bot.set_ylabel('events / 0.35')
    bot.set_xlabel('numu_score   (log-odds)')
    bot.set_title('the nue_score $=-15$ population: br_filled $\\neq$ 1, so the nue BDT was '
                  'never evaluated\n(background-like default -- NOT missing data); '
                  'their numu_score, projected', fontsize=9.5)
    bot.legend(fontsize=8.5, loc='upper left', ncol=2, framealpha=0.9)

    fig.tight_layout()
    p = os.path.join(outdir, 'd85_bdt.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def summary(data, prod, outdir):
    cols = ['sample', 'n_eval', 'n_cosmic_tagged', 'n_enu', 'enu_blank',
            'enu_min', 'enu_med', 'enu_max', 'n_enu_overflow',
            'nue_filled', 'nue_unfilled', 'numu_med', 'numu_at_clamp']
    path = os.path.join(outdir, f'd85-summary-{prod}.tsv')
    with open(path, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for key, _, _, _ in SAMPLES:
            d = data[key]
            e, m = d['enu'], d['numu']
            f.write('\t'.join(str(x) for x in [
                key, d['n_eval'], d['n_cosmic_tagged'], len(e), d['n_enu_blank'],
                f'{min(e):.1f}', f'{st.median(e):.1f}', f'{max(e):.1f}',
                sum(1 for v in e if v >= E_HI),
                sum(1 for v in d['nue'] if v > NUE_UNFILLED + 0.1),
                sum(1 for v in d['nue'] if v <= NUE_UNFILLED + 0.1),
                f'{st.median(m):.3f}',
                sum(1 for v in m if abs(abs(v) - CLAMP) < 1e-3),
            ]) + '\n')
    return path


def bee_picks(prod, outdir, n=10):
    """The three hand-scan categories the owner asked for, drawn from the numu
    samples (mcp1k + mcp2k) only, PR-evaluated rows only.  NO documented
    numu_score / nue_score operating point exists in these docs, so every cut
    below is RANK order on the score, not a threshold -- stated as such in the
    doc.  Event ids do not collide between mcp1k and mcp2k (checked), so the
    three lists index cleanly into one Bee set each.

    Writes <cat>.txt (bee-index order = the order printed) and a .tsv carrying
    the scores, so a link can be read back to a row later."""
    rows = []
    for s in ('mcp1k', 'mcp2k'):
        path = os.path.join('products', prod, f'{s}-scores-{prod}.tsv')
        for r in csv.DictReader(open(path), delimiter='\t'):
            if r['nu_evaluated'] == '1' and r['numu_score'] and r['nue_score']:
                r['_s'], r['_numu'], r['_nue'] = s, float(r['numu_score']), float(r['nue_score'])
                rows.append(r)
    cand = [r for r in rows if r['event_label'] == 'nu-candidate']
    cats = [
        # name,          pool,                                          key, pool label
        ('cosmiclike',   [r for r in rows if r['event_label'] == 'cosmic-tagged'],
         lambda r: r['_numu'], 'cosmic-tagged, lowest numu_score'),
        ('nuelike',      cand,
         lambda r: -r['_nue'], 'nu-candidate, highest nue_score'),
        ('neither',      [r for r in cand if r['_nue'] < NUE_UNFILLED + 0.1],
         lambda r: r['_numu'], 'nu-candidate + nue BDT unfilled, lowest numu_score'),
    ]
    out = []
    for name, pool, key, desc in cats:
        pick = sorted(pool, key=key)[:n]
        tp = os.path.join(outdir, f'd85-{name}.txt')
        with open(tp, 'w') as f:
            f.write('\n'.join(r['event'] for r in pick) + '\n')
        sp = os.path.join(outdir, f'd85-{name}.tsv')
        with open(sp, 'w') as f:
            f.write(f'# {desc}; pool = {len(pool)} events of {len(rows)} '
                    f'PR-evaluated mcp1k+mcp2k\n')
            f.write('bee_idx\tsample\trun\tsubrun\tevent\tnumu_score\tnue_score\t'
                    'kine_reco_Enu_MeV\tevent_label\n')
            for i, r in enumerate(pick):
                f.write(f"{i}\t{r['_s']}\t{r['run']}\t{r['subrun']}\t{r['event']}\t"
                        f"{r['_numu']:.3f}\t{r['_nue']:.3f}\t{r['kine_reco_Enu_MeV']}\t"
                        f"{r['event_label']}\n")
        out += [tp, sp]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prod', default='prod0825')
    ap.add_argument('--outdir', default='docs/85_dists')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    data = {k: load(a.prod, k) for k, _, _, _ in SAMPLES}
    for p in ([fig_enu(data, a.outdir), fig_bdt(data, a.outdir),
               summary(data, a.prod, a.outdir)] + bee_picks(a.prod, a.outdir)):
        print('wrote', p)


if __name__ == '__main__':
    main()
