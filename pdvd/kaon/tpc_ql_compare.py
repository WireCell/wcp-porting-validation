#!/usr/bin/env python3
"""doc pdvd/118 sec 4.2-4.3: TPC-side conditions and Q/L quality, run 39305
(kaon candidates, top drift only) vs 039349 production-flip arm (q35flip,
84 events; TOP-volume clusters only, apa==4 in the calib dumps).
  * drift-speed / E-field proxy: anode<->cathode crossers pile up at a raw-x
    span of L*v_prod/v_true (L = anode_x - cathode_x = 336.91 cm), so the
    crosser edge measures v_true/v_prod.  Edge = median span of clusters with
    span in [300, 400] cm (robust span: 0.2-99.8 % x quantiles).
  * Q/L: matched fraction of dumped clusters, KS / chi2/ndf / pred-over-meas
    of the selected (auto_selected) bundles.
Writes kaon/out/tpc_ql_compare.txt (printed) and docs/pics/d118_crosser_span.png."""
import glob, json, os
import numpy as np
KDIR = os.path.dirname(os.path.abspath(__file__)); PDVD = os.path.dirname(KDIR)
SETS = {'039305 kaon (top-only, this doc)': sorted(glob.glob(f'{PDVD}/work/039305_*/calib-evt*.json')),
        '039349 q35flip (top volume)': sorted(glob.glob(f'{PDVD}/work/039349_*_q35flip/calib-evt*.json'))}
out = []; spans_by = {}
for name, files in SETS.items():
    spans, nclu, nsel, ks, c2n, ratio, nmc = [], 0, 0, [], [], [], 0
    L = None
    for p in files:
        d = json.load(open(p))
        g = d['geometry']['4']; L = g['anode_x'] - g['cathode_x']
        top = {c['uid']: c for c in d['clusters'] if c['apa'] == 4}
        for c in top.values():
            x = np.asarray(c['x'])
            if len(x) < 200: continue
            spans.append(np.quantile(x, 0.998) - np.quantile(x, 0.002))
        nclu += len(top)
        nmc += len({b['main_cluster'] for b in d['bundles'] if b['auto_selected'] and b['main_cluster'] in top})
        for b in d['bundles']:
            if b['auto_selected'] and b['main_cluster'] in top:
                nsel += 1; ks.append(b['ks_dis']); c2n.append(b['chi2'] / max(b['ndf'], 1))
                ratio.append(b['total_pred_light'] / max(b['total_PE'], 1))
    spans = np.array(spans); spans_by[name] = spans
    cr = spans[(spans > 300) & (spans < 400)]
    edge = np.median(cr) if len(cr) else np.nan
    err = 1.2533 * cr.std() / np.sqrt(len(cr)) if len(cr) > 1 else np.nan
    out.append(f'{name}: {len(files)} events')
    out.append(f'  crossers (span 300-400 cm): n={len(cr)}  edge={edge:.1f} +- {err:.1f} cm  (L={L:.1f})'
               f'  => v_true/v_prod = {L / edge:.3f} +- {L * err / edge**2:.3f}')
    out.append(f'  Q/L: dumped top clusters {nclu}, with a selected flash {nmc} ({100 * nmc / max(nclu, 1):.0f} %), selected bundles {nsel};'
               f' selected KS median {np.median(ks):.3f}, chi2/ndf median {np.median(c2n):.1f},'
               f' pred/meas median {np.median(ratio):.2f} (p16-p84 {np.percentile(ratio, 16):.2f}-{np.percentile(ratio, 84):.2f})')
print('\n'.join(out))
os.makedirs(f'{KDIR}/out', exist_ok=True)
open(f'{KDIR}/out/tpc_ql_compare.txt', 'w').write('\n'.join(out) + '\n')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
for (name, s), c in zip(spans_by.items(), ('C3', 'C0')):
    ax.hist(s, bins=np.arange(0, 480, 8), histtype='step', density=True, color=c, label=name)
ax.axvline(L, color='k', ls='--', lw=0.8, label=f'anode-cathode {L:.1f} cm')
ax.set(xlabel='cluster raw-x span (cm, at v_prod = 1.48073 mm/us)', ylabel='density', yscale='log')
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f'{PDVD}/docs/pics/d118_crosser_span.png', dpi=110)
print('wrote docs/pics/d118_crosser_span.png')
