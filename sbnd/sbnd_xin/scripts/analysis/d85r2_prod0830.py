#!/usr/bin/env python3
"""doc 85 round 2 -- the prod0830 read-out.

prod0830 is the same 3067 events as prod0825, re-run through stage B only,
from the SAME work-<s>-grp0825 Q/L input, at toolkit 546f52a8.  Two production
changes land in it:

  1. the BDT log-odds clamp is GONE (toolkit 59f75bb8).  Scores were capped at
     +-4.30103; they now span +-16.25562, the prototype's range.  MicroBooNE's
     nue_score > 7.0 working point is expressible for the first time.
  2. T_kine carries the excluded-energy census (toolkit 546f52a8):
     kine_energy_excluded{,_main,_other}, kine_n_excluded, kine_energy_flagged.

WHAT ISOLATES WHAT.  prod0830 also inherits every SBND config flip shipped
between 08-25 and 08-30 (the 0.84 shower fudge, the pi0 round-3 trio), so a
prod0825-vs-prod0830 diff is NOT a clamp measurement.  The clamp is isolated
exactly and without a second arm instead: clamping v to +-0.9999 before
log10((1+v)/(1-v)) is the same map as clipping the SCORE at +-4.30103, so the
clamped counterfactual is clip(score_new, +-CLAMP) computed from prod0830's
own rows.  Every clamp claim in the doc uses that, never the cross-arm diff.

Usage:
  python3 scripts/analysis/d85r2_prod0830.py [--prod prod0830] [--outdir docs/85r2_dists]
"""
import argparse
import csv
import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d85_dists import (degenerate, overflow_hist, step_xy,  # noqa: E402
                       SAMPLES, E_LO, E_HI, E_BW)

CLAMP = 4.30103        # the removed clamp's ceiling, now an interior value
MAXSCORE = 16.25562     # log10(2/nextafter-gap): the unclamped ceiling
NUE_UNFILLED = -15.0   # br_filled != 1 sentinel -- NOT a floor any more
NUMU_SEL = 0.9         # MicroBooNE working points, tagger_validation_plan.md:229
NUE_SEL = 7.0

# excluded-energy binning: MeV, with an overflow bin
X_LO, X_HI, X_BW = 0.0, 1000.0, 50.0


def rows(prod, sample, evaluated_only=True):
    path = os.path.join('products', prod, f'{sample}-scores-{prod}.tsv')
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            r['_sample'] = sample
            if evaluated_only and (r['nu_evaluated'] != '1' or not r['numu_score']
                                   or not r['nue_score'] or degenerate(r)):
                continue
            out.append(r)
    return out


def f(r, k):
    return float(r[k]) if r.get(k) not in (None, '') else None


def unfilled(r):
    """The nue BDT's "not filled" sentinel, tested EXACTLY.

    Round 1 tested nue_score < -14.9 for this.  That test is now wrong: with
    the clamp gone a genuinely background-like event can score below -15, so
    the sentinel and the live distribution overlap.  The count of rows in
    (-MAXSCORE, -14.9) that are NOT exactly -15 is reported by census() -- the
    ambiguity is measured, not assumed away."""
    return r['nue_score'] and abs(float(r['nue_score']) - NUE_UNFILLED) < 1e-6


def load(prod):
    data = {}
    for key, _lab, _c, _m in SAMPLES:
        rs = rows(prod, key)
        allrs = rows(prod, key, evaluated_only=False)
        d = {'rows': rs,
             'n_eval': sum(1 for r in allrs if r['nu_evaluated'] == '1'),
             'n_degenerate': sum(1 for r in allrs
                                 if r['nu_evaluated'] == '1' and degenerate(r)),
             'enu': [], 'numu': [], 'nue': [], 'exc': [], 'excfrac': []}
        for r in rs:
            e = f(r, 'kine_reco_Enu_MeV')
            if e is not None:
                d['enu'].append(e)
            d['numu'].append(f(r, 'numu_score'))
            d['nue'].append(f(r, 'nue_score'))
            x = f(r, 'kine_energy_excluded_MeV')
            if x is not None:
                d['exc'].append(x)
                if e is not None and e + x > 0:
                    d['excfrac'].append(x / (e + x))
        data[key] = d
    return data


# ---------------------------------------------------------------------------
# figure 1 -- reconstructed neutrino energy (round 1's sec 2, on the new arm)
# ---------------------------------------------------------------------------
def fig_enu(data, prod, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, mode in zip(axes, ('norm', 'count')):
        for key, lab, col, _ in SAMPLES:
            d = data[key]
            edges, counts = overflow_hist(d['enu'])
            n = sum(counts)
            y = [c / n / (E_BW / 1000.0) for c in counts] if mode == 'norm' else counts
            xs, ys = step_xy(edges, y)
            med = st.median(d['enu']) if d['enu'] else float('nan')
            ax.plot(xs, ys, color=col, lw=1.8, label=f'{lab}  N={n}, med {med:.0f} MeV')
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
        ax.text(E_HI + E_BW / 2, 0.45, 'overflow', rotation=90, ha='center',
                va='center', fontsize=8, color='0.45',
                transform=blend(ax.transData, ax.transAxes))
    fig.suptitle(f'SBND {prod} -- reconstructed neutrino energy of the PR-evaluated '
                 'events (nu_evaluated = 1)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(outdir, 'd85r2_enu.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# figure 2 -- the score plane, unclamped, with the clamped counterfactual
# ---------------------------------------------------------------------------
def fig_bdt(data, prod, outdir):
    fig = plt.figure(figsize=(13.6, 7.0))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.18,
                  left=0.065, right=0.985, bottom=0.085, top=0.845, figure=fig)
    ax = fig.add_subplot(gs[0])
    axc = fig.add_subplot(gs[1])

    lim = MAXSCORE + 0.8
    for a, clip in ((ax, False), (axc, True)):
        for key, lab, col, mk in SAMPLES:
            d = data[key]
            xs = [min(max(v, -CLAMP), CLAMP) if clip else v for v in d['numu']]
            ys = [min(max(v, -CLAMP), CLAMP) if clip else v for v in d['nue']]
            a.scatter(xs, ys, s=13, marker=mk, alpha=0.55, linewidths=0,
                      color=col, label=f'{lab}  N={len(xs)}')
        # the removed clamp's box, drawn on both so the eye can size it
        a.plot([-CLAMP, CLAMP, CLAMP, -CLAMP, -CLAMP],
               [-CLAMP, -CLAMP, CLAMP, CLAMP, -CLAMP],
               color='0.35', ls='--', lw=1.1, zorder=5)
        a.axhline(NUE_SEL, color='#2ca02c', ls='-', lw=1.2, zorder=4)
        a.axvline(NUMU_SEL, color='#1f77b4', ls='-', lw=1.2, zorder=4)
        a.axhline(NUE_UNFILLED, color='0.5', ls=':', lw=1.0, zorder=3)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_xlabel('$\\nu_\\mu$ BDT score  (numu_score)')
        a.grid(alpha=0.22)
    ax.set_ylabel('$\\nu_e$ BDT score  (nue_score)')
    ax.set_title(f'{prod} as produced — clamp REMOVED', fontsize=11)
    axc.set_title('the same rows, clipped at $\\pm$4.30103\n'
                  '= exactly what the clamp used to report', fontsize=11)
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.92)

    # annotate the two working points and the sentinel once, on the left panel.
    # The sentinel label goes on the RIGHT: the lower-left corner is the legend.
    ax.text(-lim + 0.4, NUE_SEL + 0.35, 'uB $\\nu_e$CC: nue_score > 7.0',
            fontsize=8.5, color='#2ca02c')
    ax.text(NUMU_SEL + 0.25, -lim + 0.6, 'uB $\\nu_\\mu$CC: > 0.9',
            fontsize=8.5, color='#1f77b4', rotation=90)
    ax.text(lim - 0.4, NUE_UNFILLED + 0.45, 'nue_score = $-15$ (br_filled $\\neq$ 1)',
            fontsize=8, color='0.45', ha='right')
    ax.text(CLAMP + 0.15, CLAMP + 0.15, 'the removed clamp', fontsize=8, color='0.35')

    n_over = sum(1 for k, _l, _c, _m in SAMPLES
                 for v in data[k]['nue'] if v > CLAMP)
    n_ub = sum(1 for k, _l, _c, _m in SAMPLES
               for v in data[k]['nue'] if v > NUE_SEL)
    fig.suptitle(f'SBND {prod} — $\\nu_e$ vs $\\nu_\\mu$ BDT score.  '
                 f'{n_over} events now sit above the old +4.30103 ceiling; '
                 f'{n_ub} pass the MicroBooNE $\\nu_e$ point\n'
                 '(under the clamp both counts were 0 by construction — right panel)',
                 fontsize=11.5, y=0.965)
    p = os.path.join(outdir, 'd85r2_bdt.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# figure 3 -- the new variable
# ---------------------------------------------------------------------------
def fig_excluded(data, prod, outdir):
    # An arm produced before toolkit 546f52a8 has no such branches, and so no
    # figure.  Say so and return rather than dying inside a log-scale tick
    # locator, which is how this first showed up.
    if not any(data[k]['exc'] for k, _l, _c, _m in SAMPLES):
        print(f'note: {prod} carries no kine_energy_excluded column '
              '(arm predates toolkit 546f52a8) -- skipping the excluded figure')
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    nb = int(round((X_HI - X_LO) / X_BW))
    edges = [X_LO + i * X_BW for i in range(nb + 2)]

    for key, lab, col, _ in SAMPLES:
        d = data[key]
        counts = [0] * (nb + 1)
        for v in d['exc']:
            counts[min(max(int((v - X_LO) / X_BW), 0), nb)] += 1
        xs, ys = step_xy(edges, counts)
        n = sum(counts)
        med = st.median(d['exc']) if d['exc'] else float('nan')
        axes[0].plot(xs, ys, color=col, lw=1.8, label=f'{lab}  N={n}, med {med:.0f} MeV')

        # fraction of the candidate's reconstructed energy left out of Enu
        fb = [0.0 + i * 0.02 for i in range(52)]
        fc = [0] * 51
        for v in d['excfrac']:
            fc[min(max(int(v / 0.02), 0), 50)] += 1
        fxs, fys = step_xy(fb, fc)
        tot = sum(fc)
        fmed = st.median(d['excfrac']) if d['excfrac'] else float('nan')
        axes[1].plot(fxs, [c / tot if tot else 0 for c in fys], color=col, lw=1.8,
                     label=f'{lab}  med {100 * fmed:.1f}%')

        axes[2].scatter([f(r, 'kine_reco_Enu_MeV') for r in d['rows']
                         if f(r, 'kine_energy_excluded_MeV') is not None],
                        [f(r, 'kine_energy_excluded_MeV') for r in d['rows']
                         if f(r, 'kine_energy_excluded_MeV') is not None],
                        s=11, alpha=0.5, linewidths=0, color=col, label=lab)

    axes[0].axvline(X_HI, color='0.5', ls=':', lw=1)
    axes[0].set_yscale('log')
    axes[0].set_xlim(X_LO, X_HI + X_BW)
    axes[0].set_xlabel('kine_energy_excluded  [MeV]')
    axes[0].set_ylabel('events / 50 MeV')
    axes[0].set_title('energy in the PR graph that $E_\\nu$ does NOT carry')
    axes[0].legend(fontsize=8.5)

    axes[1].set_xlim(0, 1.0)
    # log y: the first bin holds 47-75 % of every sample, so on a linear axis
    # the tail -- the events this variable exists to find -- is invisible.
    axes[1].set_yscale('log')
    axes[1].set_ylim(2e-4, 1.5)
    axes[1].set_xlabel('excluded / (excluded + $E_\\nu$)')
    axes[1].set_ylabel('fraction of events / 0.02  (log $y$)')
    axes[1].set_title('as a fraction of the candidate\'s reconstructed energy')
    axes[1].legend(fontsize=8.5)

    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('kine_reco_Enu  [MeV]')
    axes[2].set_ylabel('kine_energy_excluded  [MeV]')
    axes[2].set_title('excluded vs included, per event')
    axes[2].legend(fontsize=8.5, loc='upper left')
    for a in axes:
        a.grid(alpha=0.22)

    fig.suptitle(f'SBND {prod} — the excluded-energy census (new T_kine branches).  '
                 'Excluded = PR-graph segments whose energy entered neither a track '
                 'row nor a shower row of the kine tree.', fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(outdir, 'd85r2_excluded.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# figure 4 -- the MicroBooNE working points, now actually applicable
# ---------------------------------------------------------------------------
def fig_cuts(prod, outdir):
    pool = rows(prod, 'mcp1k') + rows(prod, 'mcp2k')
    noncos = [r for r in pool if f(r, 'cosmict_flag') == 0]
    sets = {
        'all, cosmics excluded': noncos,
        f'numu_score > {NUMU_SEL}': [r for r in noncos if f(r, 'numu_score') > NUMU_SEL],
        f'nue_score > {NUE_SEL} (uB point)': [r for r in noncos if f(r, 'nue_score') > NUE_SEL],
        f'nue_score > {CLAMP} (old ceiling)': [r for r in noncos if f(r, 'nue_score') > CLAMP],
    }
    styles = {
        'all, cosmics excluded': ('0.55', '-', 1.4),
        f'numu_score > {NUMU_SEL}': ('#1f77b4', '-', 2.0),
        f'nue_score > {NUE_SEL} (uB point)': ('#2ca02c', '-', 2.0),
        f'nue_score > {CLAMP} (old ceiling)': ('#d62728', '--', 1.6),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, mode in zip(axes, ('count', 'norm')):
        for name, sel in sets.items():
            e = [f(r, 'kine_reco_Enu_MeV') for r in sel if r['kine_reco_Enu_MeV']]
            col, ls, lw = styles[name]
            if not e:
                ax.plot([], [], color=col, ls=ls, lw=lw, label=f'{name}   N=0')
                continue
            edges, counts = overflow_hist(e)
            n = sum(counts)
            y = [c / n / (E_BW / 1000.0) for c in counts] if mode == 'norm' else counts
            xs, ys = step_xy(edges, y)
            ax.plot(xs, ys, color=col, ls=ls, lw=lw,
                    label=f'{name}   N={n}, med {st.median(e):.0f} MeV')
        ax.set_xlabel('reconstructed $E_\\nu$  [MeV]   (kine_reco_Enu)')
        ax.set_xlim(E_LO, E_HI + E_BW)
        ax.axvline(E_HI, color='0.5', ls=':', lw=1)
        ax.grid(alpha=0.25)
        if mode == 'count':
            ax.set_yscale('log')
            ax.set_ylabel('events / 200 MeV')
            ax.set_title('selected counts (log $y$)')
            ax.legend(fontsize=8.5, loc='upper right')
        else:
            ax.set_ylabel('events / GeV, area-normalised')
            ax.set_title('shape of each selection (area-normalised)')
    fig.suptitle(f'SBND {prod}, the 3000-event $\\nu_\\mu$ data sample — MicroBooNE BDT '
                 'working points, cosmics excluded (cosmict_flag = 0)', fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(outdir, 'd85r2_numusample_cuts.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)

    lines = ['selection\tn_mcp1k\tn_mcp2k\tn_total\tmedian_Enu_MeV']
    for name, sel in sets.items():
        e = [f(r, 'kine_reco_Enu_MeV') for r in sel if r['kine_reco_Enu_MeV']]
        lines.append('\t'.join([name,
                                str(sum(1 for r in sel if r['_sample'] == 'mcp1k')),
                                str(sum(1 for r in sel if r['_sample'] == 'mcp2k')),
                                str(len(sel)),
                                f'{st.median(e):.1f}' if e else '-']))
    cp = os.path.join(outdir, 'd85r2-numusample-census.tsv')
    with open(cp, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return [p, cp]


# ---------------------------------------------------------------------------
# the census the doc quotes
# ---------------------------------------------------------------------------
def census(data, prod, outdir):
    lines = ['sample\tn_eval\tn_degenerate\tn_scored\tmed_Enu_MeV\t'
             'n_nue_gt_clamp\tn_nue_gt_ub\tn_numu_gt_clamp\tn_nue_unfilled\t'
             'n_nue_below_-14.9_not_sentinel\tmed_excluded_MeV\tmed_excluded_frac']
    tot = dict(gtc=0, gtub=0, ngtc=0, unf=0, amb=0, n=0)
    for key, _lab, _c, _m in SAMPLES:
        d = data[key]
        gtc = sum(1 for v in d['nue'] if v > CLAMP)
        gtub = sum(1 for v in d['nue'] if v > NUE_SEL)
        ngtc = sum(1 for v in d['numu'] if v > CLAMP)
        unf = sum(1 for r in d['rows'] if unfilled(r))
        amb = sum(1 for r in d['rows']
                  if not unfilled(r) and f(r, 'nue_score') < -14.9)
        tot['gtc'] += gtc; tot['gtub'] += gtub; tot['ngtc'] += ngtc
        tot['unf'] += unf; tot['amb'] += amb; tot['n'] += len(d['rows'])
        lines.append('\t'.join([
            key, str(d['n_eval']), str(d['n_degenerate']), str(len(d['rows'])),
            f"{st.median(d['enu']):.1f}" if d['enu'] else '-',
            str(gtc), str(gtub), str(ngtc), str(unf), str(amb),
            f"{st.median(d['exc']):.1f}" if d['exc'] else '-',
            f"{st.median(d['excfrac']):.4f}" if d['excfrac'] else '-']))
    lines.append('\t'.join(['ALL', '-', '-', str(tot['n']), '-',
                            str(tot['gtc']), str(tot['gtub']), str(tot['ngtc']),
                            str(tot['unf']), str(tot['amb']), '-', '-']))
    p = os.path.join(outdir, f'd85r2-summary-{prod}.tsv')
    with open(p, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return p


# ---------------------------------------------------------------------------
# Bee pick lists -- round 1's six categories, recomputed on the new arm
# ---------------------------------------------------------------------------
def picks(prod, outdir):
    written = []

    def emit(name, sel, desc, sample_of=lambda r: r['_sample']):
        tp = os.path.join(outdir, f'd85r2-{name}.txt')
        with open(tp, 'w') as fh:
            fh.write('\n'.join(r['event'] for r in sel) + '\n')
        sp = os.path.join(outdir, f'd85r2-{name}.tsv')
        with open(sp, 'w') as fh:
            fh.write(f'# {desc}\n')
            fh.write('bee_idx\tsample\trun\tsubrun\tevent\tnu_evaluated\tnumu_score\t'
                     'nue_score\tcosmict_flag\tkine_reco_Enu_MeV\t'
                     'kine_energy_excluded_MeV\tevent_label\n')
            for i, r in enumerate(sel):
                fh.write('\t'.join([
                    str(i), sample_of(r), r['run'], r['subrun'], r['event'],
                    r['nu_evaluated'], r['numu_score'] or '-', r['nue_score'] or '-',
                    r['cosmict_flag'] or '-', r['kine_reco_Enu_MeV'] or '-',
                    r.get('kine_energy_excluded_MeV') or '-', r['event_label']]) + '\n')
        written.extend([tp, sp])

    # --- the three numu-sample rank-order sets (round 1 sec 5) --------------
    numu = rows(prod, 'mcp1k') + rows(prod, 'mcp2k')
    cand = [r for r in numu if r['event_label'] == 'nu-candidate']
    emit('cosmiclike',
         sorted([r for r in numu if r['event_label'] == 'cosmic-tagged'],
                key=lambda r: f(r, 'numu_score'))[:10],
         f'cosmic-tagged mcp1k+mcp2k, LOWEST numu_score; rank order, no threshold; '
         f'pool {sum(1 for r in numu if r["event_label"] == "cosmic-tagged")} '
         f'of {len(numu)} scored')
    emit('nuelike',
         sorted(cand, key=lambda r: -f(r, 'nue_score'))[:10],
         f'nu-candidate mcp1k+mcp2k, HIGHEST nue_score; rank order; '
         f'pool {len(cand)}.  Unlike round 1 these are now ORDERED: the clamp '
         f'used to flatten every strong candidate onto 4.300936')
    # "neither": the nue BDT never filled.  Tested EXACTLY on the sentinel --
    # round 1's `< -14.9` no longer identifies it (see unfilled()).
    emit('neither',
         sorted([r for r in cand if unfilled(r)],
                key=lambda r: f(r, 'numu_score'))[:10],
         f'nu-candidate with nue_score exactly -15 (br_filled != 1), LOWEST '
         f'numu_score; pool {sum(1 for r in cand if unfilled(r))}')

    # --- the three working-point sets (round 1 sec 7) ------------------------
    allnue = rows(prod, 'nuecc48', evaluated_only=False)
    fail = [r for r in allnue
            if not (r['nue_score'] and f(r, 'nue_score') > NUE_SEL
                    and r['nu_evaluated'] == '1' and not degenerate(r))]
    emit('nuecc-failnue', fail,
         f'nuecc48 events NOT passing the MicroBooNE nue_score > {NUE_SEL}; '
         f'{len(fail)} of {len(allnue)}.  Round 1 could only ask this of the '
         f'clamp ceiling (>= 4.30), a strict superset',
         sample_of=lambda r: 'nuecc48')

    ncp = rows(prod, 'ncpi0')
    numupass = [r for r in ncp if f(r, 'numu_score') > NUMU_SEL]
    emit('ncpi0-numupass', numupass,
         f'ncpi0 events passing numu_score > {NUMU_SEL}; {len(numupass)} of '
         f'{len(ncp)} evaluated', sample_of=lambda r: 'ncpi0')

    leak = [r for r in ncp if f(r, 'cosmict_flag') == 1 or f(r, 'nue_score') > NUE_SEL]
    if leak:
        emit('ncpi0-leak', leak,
             f'ncpi0 events passing cosmict_flag == 1 OR nue_score > {NUE_SEL}; '
             f'{len(leak)} of {len(ncp)}', sample_of=lambda r: 'ncpi0')
    else:
        near = sorted([r for r in ncp if not unfilled(r)],
                      key=lambda r: -f(r, 'nue_score'))
        emit('ncpi0-nearmiss', near,
             f'NO ncpi0 event passes cosmict_flag == 1 or nue_score > {NUE_SEL} '
             f'(0 of {len(ncp)}); this is instead the {len(near)} whose nue BDT '
             f'filled at all -- the nearest misses', sample_of=lambda r: 'ncpi0')
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prod', default='prod0830')
    ap.add_argument('--outdir', default='docs/85r2_dists')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    data = load(a.prod)
    made = [fig_enu(data, a.prod, a.outdir),
            fig_bdt(data, a.prod, a.outdir),
            fig_excluded(data, a.prod, a.outdir),
            census(data, a.prod, a.outdir)]
    made += fig_cuts(a.prod, a.outdir)
    made += picks(a.prod, a.outdir)
    for p in made:
        if p:
            print('wrote', p)


if __name__ == '__main__':
    main()
