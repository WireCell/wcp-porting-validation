#!/usr/bin/env python3
"""
Per-channel electronics-noise RMS for ProtoDUNE Vertical Drift (PDVD).

Extracts noise RMS vs channel for the U / V / W wire planes of each of the 8
anodes, from raw pre-NF ADC frames.  Works on both real data and the
noise-only simulation -- the `protodune-orig-frames` (data) and
`pdvd-noise-sim` (sim) tar.bz2 archives share the same layout.

Noise RMS method -- the WireCell sigproc Derivations::CalcRMS 4.5-sigma clip
(sigproc/src/Derivations.cxx:7-20), iterated to convergence to fully exclude
signal samples, then the population RMS of the remaining (noise) samples.

Usage:
  ./noise_rms.py --source data    # PDVD run039324 event 0 (raw orig frames)
  ./noise_rms.py --source sim     # noise-only simulation  (pdvd_sim)

Outputs (in noise_rms/ next to this script):
  noise_rms_<src>_anode<N>.png   per-anode U/V/W RMS-vs-channel
  noise_rms_<src>_summary.png    all 8 anodes overlaid, per plane
  noise_rms_<src>.npz            per-anode/plane channel ids + RMS arrays
"""
import os
import io
import argparse
import tarfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(SCRIPTDIR, 'noise_rms')

PLANES = ['U', 'V', 'W']
NANODE = 8
SIGMA = 4.5   # Derivations::CalcRMS signal-clip threshold

SOURCES = {
    'data': {
        'label': 'data (run039324 evt0)',
        'path': os.path.join(SCRIPTDIR, '..', 'input_data', 'run039324',
                             'evt_0', 'protodune-orig-frames-anode%d.tar.bz2'),
    },
    'sim': {
        'label': 'noise-only sim',
        'path': os.path.join(SCRIPTDIR, '..', '..', 'pdvd_sim', 'work',
                             'noise', 'all', 'pdvd-noise-sim-anode%d.tar.bz2'),
    },
}


def load_frame(archive):
    """Return (frame [nch,ntick], channels [nch]) from a WCT frame tar.bz2."""
    arrs = {}
    with tarfile.open(archive, 'r:bz2') as tf:
        for m in tf.getmembers():
            if m.name.endswith('.npy'):
                arrs[m.name] = np.load(io.BytesIO(tf.extractfile(m).read()))
    frame = next(v for k, v in arrs.items()
                 if os.path.basename(k).startswith('frame'))
    chans = next(v for k, v in arrs.items()
                 if os.path.basename(k).startswith('channels'))
    return np.asarray(frame), np.asarray(chans)


def split_planes(frame, channels):
    """Split (nch,ntick) into [(frame_U,ch_U),(frame_V,ch_V),(frame_W,ch_W)]
    by gaps in the channel numbering (same logic as pdvd/plot_frames.py)."""
    gap = np.where(np.diff(channels) > 1)[0]
    starts = [0] + list(gap + 1)
    ends = list(gap + 1) + [len(channels)]
    return [(frame[s:e], channels[s:e]) for s, e in zip(starts, ends)]


def calc_rms(wave):
    """Per-channel noise RMS -- the WireCell sigproc Derivations::CalcRMS
    4.5-sigma clip (sigproc/src/Derivations.cxx:7-20), iterated until the
    surviving-sample set converges so signal samples are fully excluded.
    A single pass leaves a residual signal bias on busy channels; iterating
    to convergence removes it. np.std (ddof=0) matches WCT Waveform::mean_rms."""
    w = wave.astype(np.float64)
    for _ in range(10):
        mean, rms = w.mean(), w.std()
        if rms == 0:
            break
        sub = w[np.abs(w - mean) < SIGMA * rms]
        if sub.size < 2 or sub.size == w.size:
            break
        w = sub
    return float(w.std())


def extract(src):
    """Compute per-channel RMS for all 8 anodes / 3 planes of one source.
    Returns {anode: {plane: (channels, rms)}}."""
    path_tmpl = SOURCES[src]['path']
    result = {}
    for n in range(NANODE):
        archive = path_tmpl % n
        frame, channels = load_frame(archive)
        loc = 'bottom' if n < 4 else 'top'
        planes = split_planes(frame, channels)
        result[n] = {}
        line = []
        for pl, (pframe, pchan) in zip(PLANES, planes):
            rms = np.array([calc_rms(pframe[i]) for i in range(pframe.shape[0])])
            result[n][pl] = (pchan, rms)
            line.append('%s=%.2f' % (pl, np.median(rms)))
        print('  anode %d (%-6s) nticks=%d  median RMS [ADC]: %s'
              % (n, loc, frame.shape[1], '  '.join(line)))
    return result


def plot_anode(src, n, planes_rms):
    """Per-anode figure: 3 stacked panels, RMS vs within-plane channel index."""
    loc = 'bottom' if n < 4 else 'top'
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    fig.suptitle('PDVD noise RMS vs channel  --  anode %d (%s drift)  --  %s'
                 % (n, loc, SOURCES[src]['label']), fontsize=12)
    for ax, pl in zip(axes, PLANES):
        chan, rms = planes_rms[pl]
        x = np.arange(len(rms))
        med = np.median(rms)
        ax.plot(x, rms, lw=0.8, color='C0')
        ax.axhline(med, color='C3', ls='--', lw=1,
                   label='median = %.2f ADC' % med)
        ax.set_ylabel('noise RMS [ADC]')
        ax.set_title('%s plane  (channels %d-%d)'
                     % (pl, int(chan[0]), int(chan[-1])), fontsize=10)
        ax.set_xlim(0, len(rms) - 1)
        ax.set_ylim(0, max(1.0, np.percentile(rms, 99) * 1.3))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
    axes[-1].set_xlabel('channel index within plane')
    plt.tight_layout()
    out = os.path.join(OUTDIR, 'noise_rms_%s_anode%d.png' % (src, n))
    plt.savefig(out, dpi=130)
    plt.close()
    print('  wrote %s' % out)


def plot_summary(src, data):
    """Summary figure: all 8 anodes overlaid per plane (bottom solid, top dashed)."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle('PDVD noise RMS vs channel  --  all anodes  --  %s\n'
                 '(anodes 0-3 bottom = solid, anodes 4-7 top = dashed)'
                 % SOURCES[src]['label'], fontsize=12)
    for ax, pl in zip(axes, PLANES):
        allrms = []
        for n in range(NANODE):
            chan, rms = data[n][pl]
            allrms.append(rms)
            ax.plot(np.arange(len(rms)), rms, lw=0.7,
                    color='C%d' % (n % 4),
                    ls='-' if n < 4 else '--',
                    label='anode %d' % n)
        # clip y-axis to the noise band; signal/bad channels spike far above
        cap = np.percentile(np.concatenate(allrms), 98) * 1.3
        ax.set_ylim(0, max(1.0, cap))
        ax.set_ylabel('noise RMS [ADC]')
        ax.set_title('%s plane' % pl, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=4, loc='upper right')
    axes[-1].set_xlabel('channel index within plane')
    plt.tight_layout()
    out = os.path.join(OUTDIR, 'noise_rms_%s_summary.png' % src)
    plt.savefig(out, dpi=130)
    plt.close()
    print('  wrote %s' % out)


def save_npz(src, data):
    arrays = {}
    for n in range(NANODE):
        for pl in PLANES:
            chan, rms = data[n][pl]
            arrays['anode%d_%s_chan' % (n, pl)] = chan
            arrays['anode%d_%s_rms' % (n, pl)] = rms
    out = os.path.join(OUTDIR, 'noise_rms_%s.npz' % src)
    np.savez(out, **arrays)
    print('  wrote %s' % out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True, choices=sorted(SOURCES),
                    help='which input to analyze: data or sim')
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    print('=== noise RMS extraction: %s ===' % SOURCES[args.source]['label'])
    data = extract(args.source)
    for n in range(NANODE):
        plot_anode(args.source, n, data[n])
    plot_summary(args.source, data)
    save_npz(args.source, data)
    print('done.')


if __name__ == '__main__':
    main()
