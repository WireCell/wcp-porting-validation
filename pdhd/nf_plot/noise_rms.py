#!/usr/bin/env python3
"""
Per-channel electronics-noise RMS for ProtoDUNE Horizontal Drift (PDHD).

Extracts noise RMS vs channel for the U / V / W wire planes of each of the 4
APAs.  For data the input is the post-NF waveform (the NF-output `raw` frame);
for the noise-only simulation it is the raw digitized frame.  The
`protodunehd-sp-frames-raw` (data) and `pdhd-noise-sim` (sim) tar.bz2 archives
share the same frame layout.

Noise RMS method -- the WireCell sigproc Derivations::CalcRMS 4.5-sigma clip
(sigproc/src/Derivations.cxx:7-20), iterated to convergence to fully exclude
signal samples, then the population RMS of the remaining (noise) samples.

Usage:
  ./noise_rms.py --source data    # PDHD run027409 event 0 (post-NF raw frames)
  ./noise_rms.py --source sim     # noise-only simulation  (pdhd_sim)

Outputs (in noise_rms/ next to this script):
  noise_rms_<src>_anode<N>.png   per-APA U/V/W RMS-vs-channel
  noise_rms_<src>_summary.png    all 4 APAs overlaid, per plane
  noise_rms_<src>.npz            per-APA/plane channel ids + RMS arrays
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
NANODE = 4
SIGMA = 4.5   # Derivations::CalcRMS signal-clip threshold

# PDHD APA: 2560 channels, split into planes at fixed channel-count offsets
# (no gaps in the channel numbering, unlike PDVD).  U=0:800, V=800:1600, W=1600:.
HD_BOUNDARIES = [800, 1600]

SOURCES = {
    'data': {
        'label': 'data (run027409 evt0, post-NF)',
        'path': os.path.join(SCRIPTDIR, '..', 'input_data', 'run027409',
                             'evt_0', 'protodunehd-sp-frames-raw-anode%d.tar.bz2'),
    },
    'sim': {
        'label': 'noise-only sim',
        'path': os.path.join(SCRIPTDIR, '..', '..', 'pdhd_sim', 'work',
                             'noise', 'all', 'pdhd-noise-sim-anode%d.tar.bz2'),
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
    """Split (nch,ntick) into [(frame_U,ch_U),(frame_V,ch_V),(frame_W,ch_W)].
    PDHD channels are contiguous, so split at fixed HD plane boundaries."""
    starts = [0] + HD_BOUNDARIES
    ends = HD_BOUNDARIES + [len(channels)]
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
    """Compute per-channel RMS for all 4 APAs / 3 planes of one source.
    Returns {anode: {plane: (channels, rms)}}."""
    path_tmpl = SOURCES[src]['path']
    result = {}
    for n in range(NANODE):
        archive = path_tmpl % n
        frame, channels = load_frame(archive)
        planes = split_planes(frame, channels)
        result[n] = {}
        line = []
        for pl, (pframe, pchan) in zip(PLANES, planes):
            rms = np.array([calc_rms(pframe[i]) for i in range(pframe.shape[0])])
            result[n][pl] = (pchan, rms)
            line.append('%s=%.2f' % (pl, np.median(rms)))
        print('  APA %d  nticks=%d  median RMS [ADC]: %s'
              % (n, frame.shape[1], '  '.join(line)))
    return result


def plot_anode(src, n, planes_rms):
    """Per-APA figure: 3 stacked panels, RMS vs within-plane channel index."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    fig.suptitle('PDHD noise RMS vs channel  --  APA %d  --  %s'
                 % (n, SOURCES[src]['label']), fontsize=12)
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
    """Summary figure: all 4 APAs overlaid per plane."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle('PDHD noise RMS vs channel  --  all APAs  --  %s'
                 % SOURCES[src]['label'], fontsize=12)
    for ax, pl in zip(axes, PLANES):
        allrms = []
        for n in range(NANODE):
            chan, rms = data[n][pl]
            allrms.append(rms)
            ax.plot(np.arange(len(rms)), rms, lw=0.7,
                    color='C%d' % n, label='APA %d' % n)
        # clip y-axis to the noise band; signal/bad channels spike far above
        cap = np.percentile(np.concatenate(allrms), 98) * 1.3
        ax.set_ylim(0, max(1.0, cap))
        ax.set_ylabel('noise RMS [ADC]')
        ax.set_title('%s plane' % pl, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, ncol=4, loc='upper right')
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
