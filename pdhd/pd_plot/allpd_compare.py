#!/usr/bin/env python3
"""All-PD single-processing flash reco vs the per-stream reco.

Answers the question that motivated the all-PD chain: in the LArSoft / snippet
view a -x flash tops out at ~half the +x ceiling because only the snippet half
(opch 80-119, 40 PDs) is reconstructed; the full-stream half (120-159) lives in
a separate flash collection.  The all-PD chain
(wct-light-allpd-reco.jsonnet: snippet branch + full-stream branch -> OpHitMerge
-> one OpFlashFinder) builds -x flashes over the WHOLE 80-159 wall in a single
processing.

Plots, pooling the per-event opflash products of run 27980:
  1. PDs per flash, by wall: +x, -x (snippet-only), -x (all-PD)
  2. total PE per flash, same split
  3. per-PD PE within flashes, same split

Usage: allpd_compare.py [run] [evt evt ...]   (default 27980 8 16 24 104 120 152)
Writes pdhd/pics/light_allpd_pds_per_flash.png .
"""
import sys, io, tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = '/nfs/data/1/xqian/toolkit-dev/toolkit/pdhd/work'
PICS = '/nfs/data/1/xqian/toolkit-dev/toolkit/pdhd/pics'


def opmat(path):
    tf = tarfile.open(path, 'r:gz')
    for n in tf.getnames():
        if n.endswith('_0_array.npy'):
            return np.load(io.BytesIO(tf.extractfile(n).read()))
    raise RuntimeError(f'no opflash matrix in {path}')


def collect(run, evts):
    """Return per-flash arrays pooled over events:
       +x (all-PD), -x snippet-only, -x all-PD, and per-PD PE arrays."""
    P = dict(px_npd=[], px_pe=[], mxs_npd=[], mxs_pe=[], mxa_npd=[], mxa_pe=[],
             px_ppe=[], mxs_ppe=[], mxa_ppe=[])
    rp = '%06d' % run
    for e in evts:
        allm = opmat(f'{WORK}/{rp}_allpd{e}/opflash_pdhd-allpd-wct.tar.gz')
        snipm = opmat(f'{WORK}/{rp}_snip{e}/opflash_pdhd-wct.tar.gz')
        for m, dst, lo, hi in [(allm, 'px', 0, 80), (allm, 'mxa', 80, 160),
                               (snipm, 'mxs', 80, 160)]:
            pe = m[:, 1:161]
            on_wall = pe[:, lo:hi].sum(1) > 0
            off_wall = (pe[:, :lo].sum(1) + pe[:, hi:].sum(1)) > 0
            sel = on_wall & (~off_wall)            # flashes pure to this wall
            sub = pe[sel, lo:hi]
            P[f'{dst}_npd'].extend((sub > 0).sum(1).tolist())
            P[f'{dst}_pe'].extend(sub.sum(1).tolist())
            P[f'{dst}_ppe'].extend(sub[sub > 0].tolist())
    return {k: np.array(v) for k, v in P.items()}


def main():
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 27980
    evts = [int(x) for x in sys.argv[2:]] or [8, 16, 24, 104, 120, 152]
    P = collect(run, evts)

    def stat(a):
        return f'n={len(a)} med={np.median(a):.0f} p90={np.percentile(a,90):.0f} max={a.max():.0f}'
    print(f'run {run} evts {evts}')
    print('  +x        ', stat(P['px_npd']))
    print('  -x snippet ', stat(P['mxs_npd']))
    print('  -x all-PD  ', stat(P['mxa_npd']))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # 1. PDs per flash
    ax = axes[0]
    bins = np.arange(0.5, 82, 1)
    ax.hist(P['px_npd'], bins=bins, histtype='step', lw=2, color='#4878d0',
            label=f'+x (max {P["px_npd"].max():.0f})')
    ax.hist(P['mxs_npd'], bins=bins, histtype='step', lw=2, color='#ee854a', ls='--',
            label=f'-x snippet only (max {P["mxs_npd"].max():.0f})')
    ax.hist(P['mxa_npd'], bins=bins, histtype='step', lw=2.5, color='#c44e52',
            label=f'-x all-PD (max {P["mxa_npd"].max():.0f})')
    ax.set_yscale('log'); ax.set_xlabel('PDs per flash'); ax.set_ylabel('flashes (log)')
    ax.set_title('PDs per flash: -x reaches the full wall')
    ax.legend(fontsize=8)

    # 2. total PE per flash
    ax = axes[1]
    pe_bins = np.logspace(-1, 5, 50)
    ax.hist(P['px_pe'], bins=pe_bins, histtype='step', lw=2, color='#4878d0', label='+x')
    ax.hist(P['mxs_pe'], bins=pe_bins, histtype='step', lw=2, color='#ee854a', ls='--', label='-x snippet only')
    ax.hist(P['mxa_pe'], bins=pe_bins, histtype='step', lw=2.5, color='#c44e52', label='-x all-PD')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('total PE per flash')
    ax.set_title('total PE per flash'); ax.legend(fontsize=8)

    # 3. per-PD PE
    ax = axes[2]
    ppe_bins = np.logspace(-1, 5, 50)
    ax.hist(P['px_ppe'], bins=ppe_bins, histtype='step', lw=2, color='#4878d0', label='+x')
    ax.hist(P['mxs_ppe'], bins=ppe_bins, histtype='step', lw=2, color='#ee854a', ls='--', label='-x snippet only')
    ax.hist(P['mxa_ppe'], bins=ppe_bins, histtype='step', lw=2.5, color='#c44e52', label='-x all-PD')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('PE per opdet (in a flash)')
    ax.set_title('per-PD PE'); ax.legend(fontsize=8)

    fig.suptitle(f'PDHD all-PD single-processing flash reco vs per-stream — run {run}, '
                 f'{len(evts)} events (pooled)\n'
                 f'+x = opch 0-79 (unchanged); -x snippet = 80-119 only; '
                 f'-x all-PD = full 80-159 wall (OpHitMerge -> one OpFlashFinder)')
    fig.tight_layout()
    out = f'{PICS}/light_allpd_pds_per_flash.png'
    fig.savefig(out, dpi=110)
    print('wrote', out)


if __name__ == '__main__':
    main()
