#!/usr/bin/env python3
"""Per-stage SP ROI diagnosis for one channel.

Reads the frame archive produced by `run_nf_sp_dnnroi_evt.sh --roi-debug`
(work/<RUN>_<EVT><suffix>/protodunehd-sp-dnnroi-frames-anode<N>.tar.bz2) and,
for each requested channel, reports the ROI/waveform coverage at every saved
SP stage:

    tight_lf      -> tight-filter deconvolved waveform (pre-ROI, full)
    cleanup_roi   -> tight ROIs after CleanUpROIs (pre-BreakROI)
    break_roi_1st -> after 1st BreakROIs+CheckROIs+CleanUpROIs loop
    break_roi_2nd -> after 2nd loop
    shrink_roi    -> after ShrinkROIs+CheckROIs+CleanUpROIs (no-op on W)
    extend_roi    -> after CleanUpCollectionROIs + ExtendROIs (final mask)
    gauss         -> final charge output

Optionally overlays the post-NF raw ADC from a baseline archive (-b).

Usage:
  ./sp_w_roi_diag.py work/027409_6_roidbg 3 9543 [8532 ...] \
      [-b work/027409_6] [-o out.png]
"""
import argparse
import io
import sys
import tarfile

import numpy as np

STAGES = ['tight_lf', 'cleanup_roi', 'break_roi_1st', 'break_roi_2nd',
          'shrink_roi', 'extend_roi', 'gauss']


def load_tag(tf, names, tag, evt):
    fname = f'frame_{tag}_{evt}.npy'
    cname = f'channels_{tag}_{evt}.npy'
    if fname not in names:
        return None, None
    frame = np.load(io.BytesIO(tf.extractfile(fname).read()))
    chans = np.load(io.BytesIO(tf.extractfile(cname).read()))
    return frame, chans


def segments(mask_idx):
    """Contiguous [start, end] runs from a sorted index array."""
    if len(mask_idx) == 0:
        return []
    segs = []
    s = p = mask_idx[0]
    for x in mask_idx[1:]:
        if x > p + 1:
            segs.append((s, p))
            s = x
        p = x
    segs.append((s, p))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('workdir', help='ROI-debug work dir (e.g. work/027409_6_roidbg)')
    ap.add_argument('apa', type=int)
    ap.add_argument('channels', type=int, nargs='+', help='global channel idents')
    ap.add_argument('-b', '--baseline', default=None,
                    help='baseline work dir with raw tag (e.g. work/027409_6)')
    ap.add_argument('-o', '--plot', default=None, help='output PNG')
    ap.add_argument('-w', '--window', default=None,
                    help='tick window lo:hi (default: auto from raw/tight_lf)')
    args = ap.parse_args()

    arc = f'{args.workdir}/protodunehd-sp-dnnroi-frames-anode{args.apa}.tar.bz2'
    tf = tarfile.open(arc)
    names = tf.getnames()
    # event number is embedded in the member names: frame_<tag>_<evt>.npy
    evt = sorted({n.split('_')[-1].split('.')[0] for n in names if n.startswith('frame_')})[0]
    print(f'archive: {arc} (event {evt})')

    raw = raw_ch = None
    if args.baseline:
        barc = f'{args.baseline}/protodunehd-sp-dnnroi-frames-anode{args.apa}.tar.bz2'
        btf = tarfile.open(barc)
        raw, raw_ch = load_tag(btf, btf.getnames(), f'raw{args.apa}', evt)

    stage_data = {}
    for st in STAGES:
        tag = f'{st}{args.apa}'
        frame, chans = load_tag(tf, names, tag, evt)
        if frame is None:
            print(f'  [missing tag {tag}]')
            continue
        stage_data[st] = (frame, chans)

    figrows = []
    for ch in args.channels:
        print(f'\n===== channel {ch} =====')
        # window
        nticks = next(iter(stage_data.values()))[0].shape[1]
        if args.window:
            lo, hi = (int(x) for x in args.window.split(':'))
            lo, hi = max(0, lo), min(nticks, hi)
        else:
            ref, refch = (raw, raw_ch) if raw is not None else stage_data['tight_lf']
            row = ref[np.where(refch == ch)[0][0]]
            big = np.where(np.abs(row) > 5 * max(1.0, np.abs(row).std()))[0]
            if len(big) == 0:
                big = np.array([0, ref.shape[1] - 1])
            lo, hi = max(0, big.min() - 200), min(ref.shape[1], big.max() + 200)
        print(f'window: [{lo}, {hi})')

        rows = {}
        if raw is not None:
            rows['raw'] = raw[np.where(raw_ch == ch)[0][0]]
        for st, (frame, chans) in stage_data.items():
            idx = np.where(chans == ch)[0]
            if len(idx) == 0:
                print(f'  {st:14s} channel not in tag')
                continue
            v = frame[idx[0]]
            rows[st] = v
            nz = np.where(v[lo:hi] != 0)[0] + lo
            segs = segments(nz)
            cov = 100.0 * len(nz) / (hi - lo)
            ss = ' '.join(f'[{a},{b}]' for a, b in segs[:12])
            if len(segs) > 12:
                ss += f' ... (+{len(segs)-12})'
            print(f'  {st:14s} cov {cov:5.1f}%  nseg {len(segs):3d}  '
                  f'sum {v[lo:hi].sum():12.0f}  {ss}')
        figrows.append((ch, lo, hi, rows))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        nch = len(figrows)
        order = (['raw'] if raw is not None else []) + STAGES
        fig, axes = plt.subplots(len(order), nch, squeeze=False,
                                 figsize=(7 * nch, 1.6 * len(order)), sharex='col')
        for ic, (ch, lo, hi, rows) in enumerate(figrows):
            for ir, st in enumerate(order):
                ax = axes[ir][ic]
                if st in rows:
                    ax.plot(range(lo, hi), rows[st][lo:hi],
                            lw=0.6, color='k' if st == 'raw' else 'r')
                ax.set_ylabel(st, fontsize=7, rotation=0, ha='right', va='center')
                ax.tick_params(labelsize=6)
                if ir == 0:
                    ax.set_title(f'ch {ch}', fontsize=9)
        fig.suptitle(f'{arc} evt {evt}', fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f'\nplot: {args.plot}')


if __name__ == '__main__':
    sys.exit(main())
