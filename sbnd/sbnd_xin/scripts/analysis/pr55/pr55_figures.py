#!/usr/bin/env python3
"""doc pr/55: figures for the case catalogue built by pr55_metrics.py.

Two kinds of output, both under <outdir>:
  55_<evt>_overview.png     whole-event (z,y)+(z,x) views: IMG grey, ASC by
                            track/shower, FIT polylines, every headline case
                            marked+labelled with its case_id, owner points
                            starred.
  55_<case_id>.png          per-case zoom, layout depends on family:
    Family A (ghost run):   2D view + arclength profile (d_img, d_asc_any, q)
                             + per-plane 2D verdict markers from walk_and_score
    Family B (uncovered):   2D view + coverage histogram (ASC->cluster FIT dist)
    Family C (phantom):     2D view + coverage-by-any-ASC profile along the fit

"Headline" selection for the overview labels and for which cases get their own
figure: every owner-tagged case, every Family A/B case (small in number), and
every "material" Family C case (fit_len>=10cm or n_pts>=20) -- the full
non-material Family C census stays in the TSV only (per doc pr/55 Step 5b).

Usage:
  pr55_figures.py <outdir> [--evt EVT ...]

Repro (doc pr/55):
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 scripts/analysis/pr55/pr55_figures.py pics
"""
import sys, os, argparse
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import pr55_metrics as M  # noqa: E402
from oc53_probe import walk_and_score  # noqa: E402

VIEWS = [(2, 1, 'z (cm)', 'y (cm)', '(y,z) view'), (2, 0, 'z (cm)', 'x (cm) -- drift', '(x,z) view')]


def _scalebar(ax, length_cm=10.0, loc_frac=(0.05, 0.05)):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[0] + loc_frac[0] * (xlim[1] - xlim[0])
    y0 = ylim[0] + loc_frac[1] * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + length_cm], [y0, y0], 'k-', lw=2.5)
    ax.text(x0 + length_cm / 2, y0 + 0.02 * (ylim[1] - ylim[0]), '%.0f cm' % length_cm,
            ha='center', fontsize=7)


def case_center_radius(c):
    cen = c.get('centroid') or c.get('p_first')
    cen = np.array(cen, float)
    L = c.get('L_cm') or c.get('fit_len_cm') or c.get('bbox_diag_cm') or 10.0
    rad = max(15.0, float(L))
    return cen, rad


def plot_2d(ax, i0, i1, img, asc, fit, center, rad, highlight_mask=None, F_case=None):
    near = np.linalg.norm(img['P'] - center, axis=1) < rad
    ax.scatter(img['P'][near][:, i0], img['P'][near][:, i1], s=6, c='lightgray', alpha=0.6, label='IMG (clustering)')
    neara = np.linalg.norm(asc['P'] - center, axis=1) < rad
    if neara.any():
        qa = asc['q'][neara]
        A = asc['P'][neara]
        ax.scatter(A[qa == 0][:, i0], A[qa == 0][:, i1], s=10, c='tab:blue', alpha=0.5, label='ASC track (q=0)')
        ax.scatter(A[qa > 0][:, i0], A[qa > 0][:, i1], s=10, c='tab:orange', alpha=0.5, label='ASC shower (q>0)')
    nearf = np.linalg.norm(fit['P'] - center, axis=1) < rad
    if nearf.any():
        Ff = fit['P'][nearf]
        ax.scatter(Ff[:, i0], Ff[:, i1], s=14, marker='+', c='black', alpha=0.7, label='FIT (all nearby segs)')
    if F_case is not None and len(F_case) >= 2:
        ax.plot(F_case[:, i0], F_case[:, i1], '-', color='red', lw=1.6, label='this case\'s fit')
        if highlight_mask is not None and highlight_mask.any():
            ax.scatter(F_case[highlight_mask][:, i0], F_case[highlight_mask][:, i1], s=55, marker='x',
                       c='red', label='case pts (bad)')
    ax.set_xlim(center[i0] - rad, center[i0] + rad)
    ax.set_ylim(center[i1] - rad, center[i1] + rad)


def make_overview(evt, ev, cases, outdir):
    img, asc, fit = ev['img'], ev['asc'], ev['fit']
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, (i0, i1, l0, l1, title) in zip(axes, VIEWS):
        for rid in np.unique(img['cid']):
            sel = img['cid'] == rid
            ax.scatter(img['P'][sel][:, i0], img['P'][sel][:, i1], s=3, alpha=0.35)
        for c in cases:
            cen = np.array(c.get('centroid') or c.get('p_first'), float)
            ax.scatter([cen[i0]], [cen[i1]], s=60, marker='o', facecolors='none',
                      edgecolors='red', linewidths=1.4)
            ax.annotate(c['case_id'], (cen[i0], cen[i1]), fontsize=6.5, color='red',
                       xytext=(3, 3), textcoords='offset points')
        for (x, y, z, name) in M.OWNER_POINTS.get(evt, []):
            p = (x, y, z)
            ax.scatter([p[i0]], [p[i1]], s=140, marker='*', c='gold', edgecolors='black', linewidths=0.8, zorder=5)
            ax.annotate('owner:' + name, (p[i0], p[i1]), fontsize=6.5, color='black',
                       xytext=(4, -8), textcoords='offset points')
        ax.set_xlabel(l0); ax.set_ylabel(l1); ax.set_title(title)
        _scalebar(ax, 20.0)
    plt.suptitle('doc pr/55 evt %s -- headline case overview (image colored by final cluster; '
                 'red circles = case centroids; gold stars = owner-reported points)' % evt, fontsize=10)
    plt.tight_layout()
    out = os.path.join(outdir, '55_%s_overview.png' % evt)
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print('wrote', out)


def make_case_figure(evt, ev, ld, c, outdir):
    img, asc, fit = ev['img'], ev['asc'], ev['fit']
    cen, rad = case_center_radius(c)
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    seg = c['seg']; mseg = fit['seg'] == seg
    F_case = fit['P'][mseg] if mseg.any() else None

    for ax, (i0, i1, l0, l1, title) in zip((ax1, ax2), VIEWS):
        plot_2d(ax, i0, i1, img, asc, fit, cen, rad, F_case=F_case)
        ax.set_xlabel(l0); ax.set_ylabel(l1); ax.set_title(title)
        _scalebar(ax, max(5.0, round(rad / 3)))
    ax1.legend(fontsize=6, loc='best')

    if c['family'] == 'A':
        p1 = np.array(c['p_first'], float); p2 = np.array(c['p_last'], float)
        r = walk_and_score(ld, p1, p2)
        nst = r['nst']
        ds, qs = [], []
        own_img = img['P'][img['cid'] == c['cid']]
        tree = cKDTree(own_img) if len(own_img) else None
        Fseg = fit['P'][mseg]; qseg = fit['q'][mseg]
        d_own = tree.query(Fseg)[0] if tree is not None else np.full(len(Fseg), 1e9)
        s = np.arange(len(Fseg))
        ax3b = ax3.twinx()
        ax3.plot(s, d_own, '-', color='tab:red', label='d(fit, own-cluster image) [cm]')
        ax3.axhline(1.0, color='tab:red', ls=':', lw=0.8)
        ax3b.plot(s, qseg, '-', color='tab:blue', alpha=0.7, label='q (dQ/dx proxy)')
        ax3.set_xlabel('point index along segment %s (trajectory order)' % seg)
        ax3.set_ylabel('distance to own-cluster image (cm)', color='tab:red')
        ax3b.set_ylabel('q', color='tab:blue')
        ax3.set_title('profile: d_img rising with q -- dQ/dx inflation where the fit has no charge to constrain it')
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')
        verdict = ('kill_base(S1-3)=%s  kill_s5(round7)=%s  max_ghost_run=%d  strict_dis=%.1fcm  branch=%s'
                  % (c['kill_base'], c['kill_s5'], int(c['max_ghost_run']), float(c['strict_dis_cm']), c['branch']))
        title = ('%s  family=A(ghost run)  label=%s  L=%.1fcm d_max=%.1fcm dead_frac=%.2f q_ratio=%s\n'
                'strict-rule replay on this run\'s endpoints: %s\nowner point: %s'
                % (c['case_id'], c['label'], float(c['L_cm']), float(c['d_max_cm']), float(c['dead_frac']),
                   c['q_ratio'], verdict, c['owner_pt'] or '(none within 15cm)'))
    elif c['family'] == 'B':
        A = asc['P'][asc['seg'] == seg]
        F_cid = fit['P'][fit['cid'] == c['cid']]
        d = cKDTree(F_cid).query(A)[0] if len(F_cid) else np.full(len(A), 999.0)
        ax3.hist(d, bins=40, color='tab:orange')
        ax3.axvline(M.UNCOV_D, color='k', ls=':')
        ax3.set_xlabel('distance from ASC point to nearest FIT point in the SAME final cluster (cm)')
        ax3.set_ylabel('count')
        ax3.set_title('coverage histogram (dashed line = %.0fcm flag threshold)' % M.UNCOV_D)
        title = ('%s  family=B(uncovered shower)  n_assoc=%d  cov_med=%.1fcm  uncov_frac3=%.2f  '
                'fit_pts(cluster)=%d  fit_pts(own seg)=%d  spread=%s\nowner point: %s'
                % (c['case_id'], int(c['n_pts']), float(c['cov_med_cm']), float(c['uncov_frac3']),
                   int(c['fit_pts']), int(c['fit_pts_own_seg']), c['spread'], c['owner_pt'] or '(none within 15cm)'))
    else:  # Family C
        Fseg = fit['P'][mseg]
        own_img = img['P'][img['cid'] == c['cid']]
        d_own = cKDTree(own_img).query(Fseg)[0] if len(own_img) else np.full(len(Fseg), 1e9)
        asc_tree = cKDTree(asc['P']) if len(asc['P']) else None
        d_any = asc_tree.query(Fseg)[0] if asc_tree is not None else np.full(len(Fseg), 1e9)
        s = np.arange(len(Fseg))
        ax3.plot(s, d_own, '-', color='tab:red', label='d(fit, own-cluster IMG)')
        ax3.plot(s, d_any, '--', color='tab:green', label='d(fit, nearest ASC point, ANY seg)')
        ax3.axhline(M.PAINT_ONLY_D, color='k', ls=':', lw=0.8)
        ax3.set_xlabel('point index along segment %s (trajectory order)' % seg)
        ax3.set_ylabel('distance (cm)')
        ax3.legend(fontsize=7)
        ax3.set_title('phantom-segment coverage: is the void real, or is it filed under a different seg id?')
        title = ('%s  family=C(phantom segment)  label=%s  fit_len=%.1fcm n_pts=%d  attribution=%s\n'
                'd_own_med=%.2fcm  frac_covered_by_any_asc=%.2f  nearest_dup_seg=%s(%.1fcm)\nowner point: %s'
                % (c['case_id'], c['label'], float(c['fit_len_cm']), int(c['n_pts']), c['attribution'],
                   float(c['d_own_med_cm']), float(c['frac_covered_any_asc']), c['nearest_dup_seg'],
                   float(c['nearest_dup_d_cm']) if c['nearest_dup_d_cm'] == c['nearest_dup_d_cm'] else float('nan'),
                   c['owner_pt'] or '(none within 15cm)'))

    plt.suptitle('doc pr/55 evt %s cid=%s seg=%s\n%s' % (evt, c['cid'], c['seg'], title), fontsize=9)
    plt.tight_layout()
    out = os.path.join(outdir, '55_%s.png' % c['case_id'])
    plt.savefig(out, dpi=135)
    plt.close(fig)
    print('wrote', out)


def headline_cases(cases):
    """Every A/B case, every owner-tagged non-misattributed case, every material
    Family C case -- capped at 2 illustrative MISATTRIBUTED-SHOWER-ID examples per
    event (that class is a single, explained, display-layer mechanism; one figure
    proves it, dozens of near-duplicates would just be noise -- 269774 alone has
    ~35 owner-tagged misattributed segments from two showers, see pr55-cases.tsv
    for the complete census)."""
    out, misattrib_budget = [], 2
    for c in cases:
        if c['family'] in ('A', 'B'):
            out.append(c)
        elif c['label'] == 'MISATTRIBUTED-SHOWER-ID':
            if c['owner_pt'] and misattrib_budget > 0:
                out.append(c); misattrib_budget -= 1
        elif c['owner_pt']:
            out.append(c)
        elif c['family'] == 'C' and c.get('material'):
            out.append(c)
    # dedup by case_id
    seen, dedup = set(), []
    for c in out:
        if c['case_id'] in seen:
            continue
        seen.add(c['case_id']); dedup.append(c)
    return dedup


def run_event(evt, outdir):
    from oc53_probe import Loader
    arm = M.EVENT_ARMS[evt][0]
    ev = M.load_event(arm, evt)
    ld = Loader(os.path.join(M.SB, arm, 'pr_evt%s' % evt))
    try:
        cases = M.family_a(ev, ld) + M.family_b(ev) + M.family_c(ev, ld)
        cases = M.assign_ids(evt, cases)
        cases = M.attribute_owner_points(evt, cases)
        hl = headline_cases(cases)
        print('=== %s: %d total cases, %d headline figures ===' % (evt, len(cases), len(hl)))
        make_overview(evt, ev, hl, outdir)
        for c in hl:
            make_case_figure(evt, ev, ld, c, outdir)
    finally:
        ld.cleanup()
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--evt', nargs='*', default=list(M.EVENT_ARMS.keys()))
    a = ap.parse_args()
    outdir = a.outdir if os.path.isabs(a.outdir) else os.path.join(M.SB, a.outdir)
    os.makedirs(outdir, exist_ok=True)
    for evt in a.evt:
        run_event(evt, outdir)


if __name__ == '__main__':
    main()
