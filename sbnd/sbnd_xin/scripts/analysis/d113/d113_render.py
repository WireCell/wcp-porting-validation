#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 3 -- 2-D panels for the missing-charge exhibits (the scan evidence).

One PNG per exhibit row of 113_exhibits.tsv: three planes of the exhibit's APA, a window of slices around the
flagged component (the same slice window in every plane), the dnnsp charge per (wire, slice) in grey, the
stage-A blob coverage as a blue outline, the dead-fork blobs hatched, the PR candidate's cells in green, and the
flagged component's bounding box in red (its partner components in the other planes in orange).  The class and
the census numbers are written to a sidecar json, NOT onto the image, so the panel can be scanned blind.

Usage: d113_render.py --exhibits docs/113_figs/113_exhibits.tsv --flags docs/113_figs/113_flags.tsv
                      [--outdir docs/113_figs/panels] [--ranks 0-59] [--blind]
"""
import argparse, csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C
from d113_missing2d import NSL, blob_grid

PLANES = 'UVW'


def find_group(sample, ev):
    for g in C.group_dirs(sample):
        if ev in C.group_events(g):
            return g
    raise KeyError((sample, ev))


def render(sample, ev, exhibits, flags, outdir, blind, fr=None):
    if fr is None:
        gdir = find_group(sample, ev)
        fr = next(f for e, f in C.iter_frames(gdir, wanted={ev}))
    bad = C.mask_frame(fr)
    F, ch = fr['frame'], fr['channels']
    nch, nq = F.shape
    pad = NSL * C.TICK_SPAN - nq
    S0 = np.pad(F, ((0, 0), (0, pad))).reshape(nch, NSL, C.TICK_SPAN).sum(axis=2)
    apa_c, plane_c, wip_c = C.plane_of_channel(ch)
    pt = C.PCTree(f'{C.SX}/work-{sample}-d102m/ql_evt{ev}/pctree-evt{ev}.tar.gz', ev)
    B = C.load_bundle(f'{C.SX}/work-{sample}-pr150s0/pr_evt{ev}', ev)
    outs = []
    for x in exhibits:
        apa = x['apa']; s_lo = max(0, x['smin'] - 60); s_hi = min(NSL, x['smax'] + 60)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
        for p in range(3):
            ax = axes[p]
            m = (apa_c == apa) & (plane_c == p)
            idx = np.nonzero(m)[0]
            G0 = np.zeros((C.NWIRE[p], NSL)); G0[wip_c[idx]] = S0[idx]
            cov = blob_grid(pt, apa, p, np.ones(len(pt.b_smin), bool))
            covd = blob_grid(pt, apa, p, None, live=False)
            # wire window: the exhibit's own plane centred on its component; other planes on their partner
            # component if any, else on the bundle cells inside the slice window, else on the charge
            part = [f for f in flags if f['sample'] == sample and f['event'] == ev and f['apa'] == apa and f['plane'] == p
                    and f['cls'] == x['cls'] and min(f['smax'], x['smax']) >= max(f['smin'], x['smin'])]
            if p == x['plane']:
                wc = 0.5 * (x['wmin'] + x['wmax'])
            elif part:
                wc = 0.5 * (part[0]['wmin'] + part[0]['wmax'])
            elif B and (apa, p) in B['cells']:
                bw, bs, _ = B['cells'][(apa, p)]
                sel = (bs >= s_lo) & (bs < s_hi)
                wc = float(np.median(bw[sel])) if sel.any() else float(np.median(bw))
            else:
                prof = G0[:, s_lo:s_hi].sum(axis=1)
                wc = float(np.argmax(prof))
            w_lo = int(max(0, wc - 120)); w_hi = int(min(C.NWIRE[p], wc + 120))
            sub = G0[w_lo:w_hi, s_lo:s_hi]
            vmax = np.percentile(sub[sub > 0], 98) if (sub > 0).any() else 1.0
            ax.imshow(sub.T, origin='lower', aspect='auto', cmap='Greys', vmin=0, vmax=vmax,
                      extent=(w_lo, w_hi, s_lo, s_hi), interpolation='nearest')
            ax.contour(np.arange(w_lo, w_hi) + 0.5, np.arange(s_lo, s_hi) + 0.5, cov[w_lo:w_hi, s_lo:s_hi].T.astype(float),
                       levels=[0.5], colors='tab:blue', linewidths=0.8)
            if covd[w_lo:w_hi, s_lo:s_hi].any():
                ax.contourf(np.arange(w_lo, w_hi) + 0.5, np.arange(s_lo, s_hi) + 0.5, covd[w_lo:w_hi, s_lo:s_hi].T.astype(float),
                            levels=[0.5, 1.5], colors='none', hatches=['//'])
            if B and (apa, p) in B['cells']:
                bw, bs, _ = B['cells'][(apa, p)]
                sel = (bs >= s_lo) & (bs < s_hi) & (bw >= w_lo) & (bw < w_hi)
                ax.scatter(bw[sel] + 0.5, bs[sel] + 0.5, s=3, c='tab:green', marker='s', alpha=0.5, linewidths=0)
            for f in part:
                col = 'red' if (p == x['plane'] and f['comp'] == x['comp']) else 'orange'
                ax.add_patch(plt.Rectangle((f['wmin'] - 3, f['smin'] - 3), f['wext'] + 6, f['sext'] + 6, fill=False, ec=col, lw=1.0))
            if p == x['plane']:   # boxes carry a 3-cell margin so a 1-2 wire feature is not hidden under the border
                ax.add_patch(plt.Rectangle((x['wmin'] - 3, x['smin'] - 3), x['wext'] + 6, x['sext'] + 6, fill=False, ec='red', lw=1.2))
            ax.set_xlabel(f'{PLANES[p]} wire (APA {apa})'); ax.set_ylabel('slice (4 ticks = 2 us)')
            ax.set_title(f'plane {PLANES[p]}' + ('' if blind else f'  [{x["cls"]}]' if p == x['plane'] else ''))
        tag = f'{sample}-{ev}-r{x["rank"]:03d}'
        fig.suptitle(f'item {tag}' if blind else f'{sample} {ev} {x["cls"]} apa{apa} plane {PLANES[x["plane"]]} Q={x["Q"]:.3g} Qfrac={x["Qfrac"]:.3f}')
        fig.tight_layout()
        out = f'{outdir}/{tag}.png'
        fig.savefig(out, dpi=110); plt.close(fig)
        json.dump({k: x[k] for k in x}, open(f'{outdir}/{tag}.json', 'w'))
        outs.append(out)
    return outs


def controls(n, seed, outdir, flags, blind):
    """Scan controls: candidate events with NO imaging-class flag; the box is placed on a random blob-covered
    region of the candidate itself (the right answer is NO: the charge is in the 3-D image)."""
    import glob, random
    rng = random.Random(seed)
    evs = []
    for f in sorted(glob.glob(f'{C.SX}/docs/113_figs/census/*.events.json')):
        evs += [e for e in json.load(open(f)) if 'error' not in e and e.get('has_bundle')]
    flagged = {(f['sample'], f['event']) for f in flags if f['cls'] != 'BUNDLE'}
    pool = [e for e in evs if (e['sample'], e['event']) not in flagged]
    rng.shuffle(pool)
    made = 0; outs = []
    for e in pool:
        if made >= n:
            break
        s, ev = e['sample'], e['event']
        B = C.load_bundle(f'{C.SX}/work-{s}-pr150s0/pr_evt{ev}', ev)
        if not B or not B['cells']:
            continue
        (apa, p), (bw, bs, bq) = max(B['cells'].items(), key=lambda kv: len(kv[1][0]))
        if len(bw) < 50:
            continue
        i = rng.randrange(len(bw))
        x = dict(rank=900 + made, sample=s, event=ev, cls='CONTROL', apa=apa, plane=p, comp=-1, ncell=0, Q=0.0, Qfrac=0.0,
                 wmin=int(bw[i]) - 3, wmax=int(bw[i]) + 3, smin=int(bs[i]) - 10, smax=int(bs[i]) + 10, wext=7, sext=21)
        outs += render(s, ev, [x], flags, outdir, blind)
        made += 1
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--controls', type=int, default=0, help='render N control items instead of exhibits')
    ap.add_argument('--seed', type=int, default=113)
    ap.add_argument('--exhibits', default=f'{C.SX}/docs/113_figs/113_exhibits.tsv')
    ap.add_argument('--flags', default=f'{C.SX}/docs/113_figs/113_flags.tsv')
    ap.add_argument('--outdir', default=f'{C.SX}/docs/113_figs/panels')
    ap.add_argument('--ranks', default=None, help='a-b')
    ap.add_argument('--blind', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    conv = lambda r: {k: (int(v) if k in ('rank', 'event', 'apa', 'plane', 'comp', 'ncell', 'wmin', 'wmax', 'smin', 'smax', 'wext', 'sext', 'npartner_planes') else
                          (float(v) if k in ('Q', 'Qfrac', 'cw', 'cs', 'f_act', 'f_noblobslice', 'f_zsum', 'nplane_act', 'd_bundle', 'd_scope', 'Qbundle_plane', 'pca_rms', 'qmax', 'f_topwire', 'f_topslice', 'floor') else v))
                      for k, v in r.items()}
    ex = [conv(r) for r in csv.DictReader(open(a.exhibits), delimiter='\t')]
    fl = [conv(r) for r in csv.DictReader(open(a.flags), delimiter='\t')]
    if a.controls:
        outs = controls(a.controls, a.seed, a.outdir, fl, a.blind)
        print(len(outs), 'control panels'); return
    if a.ranks:
        lo, hi = [int(v) for v in a.ranks.split('-')]
        ex = [x for x in ex if lo <= x['rank'] <= hi]
    by = {}
    for x in ex:
        by.setdefault((x['sample'], x['event']), []).append(x)
    for (s, ev), xs in by.items():
        outs = render(s, ev, xs, fl, a.outdir, a.blind)
        print(s, ev, len(outs), 'panels', flush=True)


if __name__ == '__main__':
    main()
