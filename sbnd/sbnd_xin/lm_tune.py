#!/usr/bin/env python3
"""LM (light-mismatch) tagger cut tuning on the MCP2025C 20-event sample.

Reads the per-event Q/L calib dumps written with `run_ql_evt.sh -lm -calib`
(doc 34): every candidate bundle carries the per-drift-side LM metrics the C++
computed (lm_ks[2], lm_pred[2], lm_meas[2], lm_length_cm) plus the stamped
verdict `lm` (0 pass / 1 low-energy / 2 light mismatch), and the dump's
quality_params.lm block records the active cuts.

Two jobs:
  1. Population study: scatter per-side KS vs log10(pred_side/meas_side) for
     the AUTO-SELECTED (matched) bundles, split by the relax flag
     (close_to_PMT / at_x_boundary), with the current cut box drawn and every
     verdict!=0 bundle annotated.  The hand-scan ground truth (evt286021
     main 8 / flash gid 1000007, labeled LM) must separate.
  2. Cut re-scan: --scan re-evaluates the verdicts OFFLINE under alternative
     cut values (same logic as QLMatching::check_light_mismatch), so choosing
     the operating point needs no C++ rerun; only the final defaults do.

Repro (doc 34):
  python3 lm_tune.py work-mcp10-lm work-mcp1000-lm --out /home/xqian/tmp/lm_tune
"""
import argparse
import glob
import gzip
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_bundles(roots):
    """One record per candidate bundle across every calib dump under roots."""
    recs = []
    for root in roots:
        pat = os.path.join(root, 'ql_evt*', 'calib-evt*.json')
        # .json.gz too: the 2026-07-24 consolidation gzipped archived dumps.
        for fn in sorted(glob.glob(pat) + glob.glob(pat + '.gz')):
            with (gzip.open(fn, 'rt') if fn.endswith('.gz') else open(fn)) as fh:
                d = json.load(fh)
            evt = os.path.basename(fn).split('-evt')[1].split('.')[0]
            cuts = d.get('quality_params', {}).get('lm')
            for b in d.get('bundles', []):
                if 'lm' not in b:
                    continue
                recs.append(dict(
                    evt=int(evt), root=os.path.basename(root),
                    main=b['main_cluster'] % 1000000, uid=b['main_cluster'],
                    gid=b['flash_gid'],
                    apa=b['apa'], sel=bool(b['auto_selected']),
                    ks=b['lm_ks'], pred=b['lm_pred'], meas=b['lm_meas'],
                    len_cm=b['lm_length_cm'], lm=b['lm'],
                    ks_all=b['ks_dis'], chi2=b['chi2'], ndf=b['ndf'],
                    relax=bool(b['close_to_PMT'] or b['at_x_boundary']),
                    close_pmt=bool(b['close_to_PMT']),
                    x_bnd=bool(b['at_x_boundary']),
                    cuts=cuts))
    return recs


def verdict(r, cuts):
    """Offline replica of QLMatching::check_light_mismatch (same branches).

    NB: this is the PER-BUNDLE verdict; the stamped cluster scalar is the
    flash-resolved one (the verdict of the flash's largest-total-pred bundle).
    """
    pred0, pred1 = r['pred']
    meas0, meas1 = r['meas']
    total_pred, total_meas = pred0 + pred1, meas0 + meas1
    small = (total_pred < cuts['pred_pe_min']) or (r['len_cm'] < cuts['length_min_cm'])
    if small and total_meas < cuts['flash_pe_bright']:
        return 1
    if small:
        ks_max, lograt_min = cuts['small_ks_max'], cuts['small_lograt_min']
    else:
        ks_max = cuts['ks_max_relax'] if r['relax'] else cuts['ks_max']
        lograt_min = cuts['lograt_min_relax'] if r['relax'] else cuts['lograt_min']
    # Good-shape guard defaults (round 2): pre-guard dumps carry no shape_*
    # keys, so absent keys reproduce the round-1 verdicts (guard disabled).
    shape_ks_max = cuts.get('shape_ks_max', -1.0)
    shape_lograt_min = cuts.get('shape_lograt_min', 0.0)
    any_judged, fail_shape, fail_norm, guardable = False, False, False, True
    for s in range(2):
        if r['pred'][s] < cuts['side_pred_min']:
            continue
        any_judged = True
        lograt = math.log10(max(r['pred'][s], 1e-6) / max(r['meas'][s], 1e-6))
        if r['ks'][s] >= 0 and r['ks'][s] > ks_max:
            fail_shape = True
        if lograt < lograt_min or lograt > cuts['lograt_max']:
            fail_norm = True
            if not (lograt < lograt_min and lograt <= cuts['lograt_max']
                    and r['ks'][s] >= 0 and r['ks'][s] < shape_ks_max
                    and lograt >= shape_lograt_min):
                guardable = False
    fail = fail_shape or (fail_norm and not guardable)
    if not any_judged:
        lograt = math.log10(max(total_pred, 1e-6) / max(total_meas, 1e-6))
        if lograt < lograt_min or lograt > cuts['lograt_max']:
            fail = True
    return 2 if fail else 0


def lograt(p, m):
    return math.log10(max(p, 1e-6) / max(m, 1e-6))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('roots', nargs='+', help='work roots (ql_evt*/calib-evt*.json)')
    ap.add_argument('--out', default='.', help='output dir for plots/tables')
    ap.add_argument('--all-bundles', action='store_true',
                    help='include non-auto-selected candidate bundles')
    ap.add_argument('--scan', action='append', default=[],
                    help='cut override key=val (repeatable), re-evaluated offline; '
                         'keys: pred_pe_min length_min_cm flash_pe_bright '
                         'side_pred_min ks_max ks_max_relax lograt_min '
                         'lograt_min_relax lograt_max small_ks_max small_lograt_min '
                         'shape_ks_max shape_lograt_min')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    recs = load_bundles(args.roots)
    if not recs:
        sys.exit('no calib dumps with lm keys found (run with -lm -calib?)')
    sel = [r for r in recs if r['sel'] or args.all_bundles]
    cuts = dict(recs[0]['cuts'])
    for kv in args.scan:
        k, v = kv.split('=')
        cuts[k] = float(v)
    print(f'{len(recs)} candidate bundles, {sum(1 for r in recs if r["sel"])} '
          f'auto-selected, across {len(set((r["root"], r["evt"]) for r in recs))} events')
    print('cuts:', json.dumps(cuts, sort_keys=True))

    # Per-side scatter, sides with judgeable prediction only.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for irx, rx in enumerate([False, True]):
        ax = axes[irx]
        pts = []
        for r in sel:
            if r['relax'] != rx:
                continue
            for s in range(2):
                if r['pred'][s] < cuts['side_pred_min']:
                    continue
                pts.append((lograt(r['pred'][s], r['meas'][s]), r['ks'][s], r, s))
        v = [verdict(p[2], cuts) for p in pts]
        for code, col, mk, lab in [(0, '#1f77b4', 'o', 'pass'),
                                   (1, '#7f7f7f', 's', 'low-E'),
                                   (2, '#d62728', '^', 'LM')]:
            xs = [p[0] for p, vv in zip(pts, v) if vv == code]
            ys = [p[1] for p, vv in zip(pts, v) if vv == code]
            ax.scatter(xs, ys, s=28, c=col, marker=mk, alpha=0.75,
                       label=f'{lab} ({len(xs)})')
        ks_max = cuts['ks_max_relax'] if rx else cuts['ks_max']
        lr_min = cuts['lograt_min_relax'] if rx else cuts['lograt_min']
        ax.axhline(ks_max, color='k', ls='--', lw=1)
        ax.axvline(lr_min, color='k', ls='--', lw=1)
        ax.axvline(cuts['lograt_max'], color='k', ls=':', lw=1)
        for p, vv in zip(pts, v):
            if vv == 2:
                r = p[2]
                ax.annotate(f'{r["evt"]}/{r["main"]}', (p[0], p[1]),
                            fontsize=7, xytext=(3, 3), textcoords='offset points')
        ax.set_title(('relaxed (close_to_PMT / at_x_boundary)' if rx
                      else 'clean (no boundary flags)') + ' — judged sides')
        ax.set_xlabel('log10(pred_side / meas_side)')
        ax.set_ylabel('per-side KS distance')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle('LM tagger tuning: per-side KS vs normalization '
                 f'({"all candidate" if args.all_bundles else "auto-selected"} bundles)')
    fig.tight_layout()
    png = os.path.join(args.out, 'lm_tune_scatter.png')
    fig.savefig(png, dpi=130)
    print('wrote', png)

    # Verdict table (offline cuts) for every non-pass auto-selected bundle.
    print(f'\n{"root":>14} {"evt":>7} {"main":>5} {"gid":>8} {"lm":>2}{"->":>3} '
          f'{"len":>6} {"ks0":>6} {"ks1":>6} {"p0":>8} {"p1":>8} {"m0":>8} {"m1":>8} flags')
    for r in sorted(sel, key=lambda r: (r['root'], r['evt'])):
        v = verdict(r, cuts)
        if r['lm'] == 0 and v == 0:
            continue
        fl = ('P' if r['close_pmt'] else '') + ('X' if r['x_bnd'] else '')
        print(f'{r["root"]:>14} {r["evt"]:>7} {r["main"]:>5} {r["gid"]:>8} '
              f'{r["lm"]:>2} {v:>2} {r["len_cm"]:>6.1f} '
              f'{r["ks"][0]:>6.3f} {r["ks"][1]:>6.3f} '
              f'{r["pred"][0]:>8.1f} {r["pred"][1]:>8.1f} '
              f'{r["meas"][0]:>8.1f} {r["meas"][1]:>8.1f} {fl}')


if __name__ == '__main__':
    main()
