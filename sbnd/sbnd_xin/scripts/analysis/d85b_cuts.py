#!/usr/bin/env python3
"""doc 85 sec 7: the MicroBooNE BDT working points applied to the prod0825
SBND samples, and the Bee pick lists that go with them.

The MicroBooNE (WCP) working points, as recorded in this repo at
clus/docs/tagger/tagger_validation_plan.md:229 ("representative working
points, e.g. nue_score > 7.0, numu_score > 0.9"), plus the cosmic verdict:

  numu CC        numu_score > 0.9
  nue  CC        nue_score  > 7.0        <-- UNREACHABLE HERE, see below
  not cosmic     cosmict_flag == 0       (the OR of the ten cosmic tests,
                                          NeutrinoTaggerCosmic.cxx:1384;
                                          cosmic_flag is NOT this -- it is a
                                          BDT input equal to !cosmict_flag_9)

THE nue WORKING POINT DOES NOT TRANSFER.  Both toolkit scorers clamp the
xgboost output to +-0.9999 BEFORE the log-odds transform
(UbooneNueBDTScorer.cxx:1988, UbooneNumuBDTScorer.cxx:588), so no score can
exceed log10(1.9999/0.0001) = 4.30103 and `nue_score > 7.0` selects exactly
zero events in every sample.  MicroBooNE's 7.0 corresponds to a raw BDT
output of 0.9999998; the clamp erases everything above 0.9999.  The
toolkit-side stand-in used here is the ceiling itself,

  NUE_SEL   nue_score >= 4.30   -- a STRICT SUPERSET of the uB selection

and the looser `nue_score > 0.7` (the other number in that same doc, its
event-fraction table) is reported beside it as a sanity bracket.  The numu
working point transfers unchanged: 0.9 is far below the clamp.

Population is doc 85 sec 1 -- nu_evaluated == 1, minus the sec 1.1 degenerate
class (imported from d85_dists.py so the two sections cannot drift apart).

Usage:
  python3 scripts/analysis/d85b_cuts.py [--prod prod0825] [--outdir docs/85_dists]
"""
import argparse
import csv
import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d85_dists import degenerate, overflow_hist, step_xy, E_LO, E_HI, E_BW  # noqa: E402

NUMU_SEL = 0.9      # MicroBooNE working point, transfers unchanged
NUE_UB = 7.0        # MicroBooNE working point -- unreachable after the clamp
NUE_SEL = 4.30      # toolkit stand-in: the clamp ceiling
NUE_LOOSE = 0.7     # the other number in tagger_validation_plan.md


def rows(prod, sample, evaluated_only=True):
    path = os.path.join('products', prod, f'{sample}-scores-{prod}.tsv')
    out = []
    for r in csv.DictReader(open(path), delimiter='\t'):
        r['_sample'] = sample
        if evaluated_only and (r['nu_evaluated'] != '1' or not r['numu_score']
                               or not r['nue_score'] or degenerate(r)):
            continue
        out.append(r)
    return out


def f(r, k):
    return float(r[k]) if r[k] else None


def fig_cuts(prod, outdir):
    """The numu sample (mcp1k + mcp2k = 3000 events): reconstructed energy of
    what each BDT working point selects, cosmics excluded."""
    pool = rows(prod, 'mcp1k') + rows(prod, 'mcp2k')
    noncos = [r for r in pool if f(r, 'cosmict_flag') == 0]
    sets = {
        'all, cosmics excluded': noncos,
        f'numu_score > {NUMU_SEL}': [r for r in noncos if f(r, 'numu_score') > NUMU_SEL],
        f'nue_score >= {NUE_SEL} (clamp ceiling)': [r for r in noncos if f(r, 'nue_score') >= NUE_SEL],
        f'nue_score > {NUE_LOOSE}': [r for r in noncos if f(r, 'nue_score') > NUE_LOOSE],
        f'nue_score > {NUE_UB} (uB point)': [r for r in noncos if f(r, 'nue_score') > NUE_UB],
    }
    styles = {
        'all, cosmics excluded': ('0.55', '-', 1.4),
        f'numu_score > {NUMU_SEL}': ('#1f77b4', '-', 2.0),
        f'nue_score >= {NUE_SEL} (clamp ceiling)': ('#d62728', '-', 2.0),
        f'nue_score > {NUE_LOOSE}': ('#ff7f0e', '--', 1.6),
        f'nue_score > {NUE_UB} (uB point)': ('#2ca02c', ':', 1.6),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, mode in zip(axes, ('count', 'norm')):
        for name, sel in sets.items():
            e = [f(r, 'kine_reco_Enu_MeV') for r in sel if r['kine_reco_Enu_MeV']]
            if not e:
                ax.plot([], [], color=styles[name][0], ls=styles[name][1],
                        lw=styles[name][2], label=f'{name}   N=0')
                continue
            edges, counts = overflow_hist(e)
            n = sum(counts)
            y = [c / n / (E_BW / 1000.0) for c in counts] if mode == 'norm' else counts
            xs, ys = step_xy(edges, y)
            col, ls, lw = styles[name]
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
    from matplotlib.transforms import blended_transform_factory as blend
    for ax in axes:
        ax.text(E_HI + E_BW / 2, 0.45, 'overflow', rotation=90,
                ha='center', va='center', fontsize=8, color='0.45',
                transform=blend(ax.transData, ax.transAxes))
    fig.suptitle('SBND prod0825, the 3000-event $\\nu_\\mu$ data sample -- MicroBooNE BDT working '
                 'points, cosmics excluded (cosmict_flag = 0)', fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(outdir, 'd85b_numusample_cuts.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)

    # the census that goes in the doc
    lines = ['selection\tn_mcp1k\tn_mcp2k\tn_total\tmedian_Enu_MeV']
    for name, sel in sets.items():
        e = [f(r, 'kine_reco_Enu_MeV') for r in sel if r['kine_reco_Enu_MeV']]
        lines.append('\t'.join([
            name,
            str(sum(1 for r in sel if r['_sample'] == 'mcp1k')),
            str(sum(1 for r in sel if r['_sample'] == 'mcp2k')),
            str(len(sel)),
            f'{st.median(e):.1f}' if e else '-']))
    cp = os.path.join(outdir, 'd85b-numusample-census.tsv')
    open(cp, 'w').write('\n'.join(lines) + '\n')
    return [p, cp]


def picks(prod, outdir):
    """The three Bee categories of sec 7.2-7.4.  Unlike sec 5 these are CUTS,
    not rank order, so the list length is whatever the cut returns."""
    out = []

    # 7.2 -- nueCC-sample events the nue selection does NOT keep.  Evaluated
    # rows failing the cut, PLUS any row with no PR evaluation at all: those
    # fail the selection too, and saying so is the point of the set.
    allnue = rows(prod, 'nuecc48', evaluated_only=False)
    keep = [r for r in allnue
            if not (r['nue_score'] and f(r, 'nue_score') >= NUE_SEL
                    and r['nu_evaluated'] == '1' and not degenerate(r))]
    out.append(('nuecc-failnue', 'nuecc48', keep,
                f'nuecc48 events NOT passing nue_score >= {NUE_SEL}; '
                f'{len(keep)} of {len(allnue)}'))

    # 7.3 -- NC pi0 events the numu selection keeps.
    ncp = rows(prod, 'ncpi0')
    numupass = [r for r in ncp if f(r, 'numu_score') > NUMU_SEL]
    out.append(('ncpi0-numupass', 'ncpi0', numupass,
                f'ncpi0 events passing numu_score > {NUMU_SEL}; '
                f'{len(numupass)} of {len(ncp)} evaluated'))

    # 7.4 -- NC pi0 events that leak into the cosmic OR nue selection.  This
    # comes back EMPTY (see the doc); the fallback set is the only ncpi0
    # events whose nue BDT filled at all, i.e. the nearest misses.
    leak = [r for r in ncp if f(r, 'cosmict_flag') == 1 or f(r, 'nue_score') >= NUE_SEL]
    if leak:
        out.append(('ncpi0-leak', 'ncpi0', leak,
                    f'ncpi0 events passing cosmict_flag == 1 OR nue_score >= {NUE_SEL}; '
                    f'{len(leak)} of {len(ncp)}'))
    else:
        near = sorted([r for r in ncp if f(r, 'nue_score') > -14.9],
                      key=lambda r: -f(r, 'nue_score'))
        out.append(('ncpi0-nearmiss', 'ncpi0', near,
                    f'NO ncpi0 event passes cosmict_flag == 1 or nue_score >= {NUE_SEL} '
                    f'(0 of {len(ncp)}); this is instead the {len(near)} whose nue BDT '
                    f'filled at all -- the nearest misses'))

    written = []
    for name, sample, sel, desc in out:
        tp = os.path.join(outdir, f'd85b-{name}.txt')
        open(tp, 'w').write('\n'.join(r['event'] for r in sel) + '\n')
        sp = os.path.join(outdir, f'd85b-{name}.tsv')
        with open(sp, 'w') as fh:
            fh.write(f'# {desc}\n')
            fh.write('bee_idx\tsample\trun\tsubrun\tevent\tnu_evaluated\tnumu_score\t'
                     'nue_score\tcosmict_flag\tkine_reco_Enu_MeV\tevent_label\n')
            for i, r in enumerate(sel):
                fh.write('\t'.join([str(i), sample, r['run'], r['subrun'], r['event'],
                                    r['nu_evaluated'], r['numu_score'] or '-',
                                    r['nue_score'] or '-', r['cosmict_flag'] or '-',
                                    r['kine_reco_Enu_MeV'] or '-', r['event_label']]) + '\n')
        written += [tp, sp]
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prod', default='prod0825')
    ap.add_argument('--outdir', default='docs/85_dists')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for p in fig_cuts(a.prod, a.outdir) + picks(a.prod, a.outdir):
        print('wrote', p)


if __name__ == '__main__':
    main()
