#!/usr/bin/env python3
"""
Compare PDVD electronics-noise frequency spectra: data run039324 vs. simulation.

Reads the noise_spectrum_{data,sim}.npz produced by noise_spectrum.py and

  * overlays data vs. sim amplitude spectra, per drift region / plane /
    strip-length bin   -> noise_spectrum_compare_<region>.png
  * shows the strip-length dependence (band-mean amplitude vs. length),
    data vs. sim        -> noise_spectrum_striplen.png
  * compares the two drift regions in data  -> noise_spectrum_topbottom.png
  * prints a data/sim band-mean ratio table.

The simulation uses the data-retuned noise spectra
(pdvd-bottom-noise-spectra-7d8mVfC-v1 / pdvd-top-noise-spectra-v3), so this is
the data/simulation consistency check; see noise_spectrum_comparison.md.

Normalization -- a waveform of N samples at period T has E|FFT(f)| proportional
to sqrt(N/T)*sqrt(S(f)), so every spectrum is renormalized to a common
(N=10000, T=500 ns) footing before overlay:
  amp_ref = amp * sqrt( (10000/500) / (N/T) ).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(SCRIPTDIR, 'noise_spectrum')
PLANES = ['U', 'V', 'W']
REGIONS = ['bottom', 'top']

REF_N, REF_T = 10000.0, 500.0          # common renormalization footing
BAND = (5.0e-5, 9.5e-4)                # 1/ns; ratio/integral window (~0.05-0.95 MHz)


def load(src):
    """Return {(region,plane,ibin): dict(freq, amp_ref, n, meanlen)}."""
    npz = np.load(os.path.join(OUTDIR, 'noise_spectrum_%s.npz' % src))
    out = {}
    for key in npz.files:
        if not key.endswith('_amp'):
            continue
        pre = key[:-4]
        region, pl, ib = pre.split('_')
        n, meanlen, ntick, period = npz[pre + '_meta']
        ren = np.sqrt((REF_N / REF_T) / (ntick / period))
        out[(region, PLANES.index(pl), int(ib))] = dict(
            freq=npz[pre + '_freq'], amp=npz[pre + '_amp'] * ren,
            n=int(n), meanlen=meanlen)
    return out


def bands(d):
    """Band-mean amplitude of one spectrum dict."""
    m = (d['freq'] > BAND[0]) & (d['freq'] < BAND[1])
    return d['amp'][m].mean()


def plot_compare(data, sim):
    """Per region: U/V/W panels, data (solid) vs sim (dashed), bins by colour."""
    for region in REGIONS:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        fig.suptitle('PDVD noise spectrum: data run039324 vs. simulation  --  '
                     '%s drift  (renormalized to N=10000, T=500 ns)' % region,
                     fontsize=12)
        for ip, ax in enumerate(axes):
            ibs = sorted(ib for (r, p, ib) in data if r == region and p == ip)
            cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(ibs), 1)))
            for j, ib in enumerate(ibs):
                dd = data[(region, ip, ib)]
                ax.plot(dd['freq'] * 1e3, dd['amp'], lw=1.1, color=cmap[j],
                        label='%.0f mm' % dd['meanlen'])
                if (region, ip, ib) in sim:
                    ss = sim[(region, ip, ib)]
                    ax.plot(ss['freq'] * 1e3, ss['amp'], lw=1.0, ls='--',
                            color=cmap[j])
            ax.set_title('%s plane  (solid=data, dashed=sim)' % PLANES[ip],
                         fontsize=10)
            ax.set_xlabel('frequency [MHz]')
            ax.set_ylabel('mean |FFT| amplitude [internal V]')
            ax.set_xlim(0, None)
            ax.set_ylim(0, None)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, title='strip length', ncol=2)
        plt.tight_layout()
        out = os.path.join(OUTDIR, 'noise_spectrum_compare_%s.png' % region)
        plt.savefig(out, dpi=130)
        plt.close()
        print('  wrote %s' % out)


def plot_striplen(data, sim):
    """Band-mean amplitude vs strip length, data vs sim, per region/plane."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, region in zip(axes, REGIONS):
        for ip in range(3):
            ibs = sorted(ib for (r, p, ib) in data if r == region and p == ip)
            L = [data[(region, ip, ib)]['meanlen'] for ib in ibs]
            c = 'C%d' % ip
            ax.plot(L, [bands(data[(region, ip, ib)]) for ib in ibs],
                    'o-', color=c, label='%s data' % PLANES[ip])
            ax.plot(L, [bands(sim[(region, ip, ib)]) for ib in ibs],
                    's--', color=c, alpha=0.6, label='%s sim' % PLANES[ip])
        ax.set_title('%s drift' % region)
        ax.set_xlabel('strip length [mm]')
        ax.set_ylabel('band-mean |FFT| amplitude [internal V]')
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
    fig.suptitle('PDVD noise vs. strip length: data run039324 vs. simulation',
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUTDIR, 'noise_spectrum_striplen.png')
    plt.savefig(out, dpi=130)
    plt.close()
    print('  wrote %s' % out)


def plot_topbottom(data):
    """Top vs bottom data spectra, longest-strip bin per plane."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for region, ls in [('bottom', '-'), ('top', '--')]:
        for ip in range(3):
            ibs = sorted(ib for (r, p, ib) in data if r == region and p == ip)
            d = data[(region, ip, ibs[-1])]
            ax.plot(d['freq'] * 1e3, d['amp'], ls, color='C%d' % ip,
                    label='%s %s (%.0f mm)' % (region, PLANES[ip], d['meanlen']))
    ax.set_title('PDVD data noise spectrum: top vs. bottom drift\n'
                 '(longest-strip bin; renormalized to N=10000, T=500 ns)',
                 fontsize=11)
    ax.set_xlabel('frequency [MHz]')
    ax.set_ylabel('mean |FFT| amplitude [internal V]')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUTDIR, 'noise_spectrum_topbottom.png')
    plt.savefig(out, dpi=130)
    plt.close()
    print('  wrote %s' % out)


def print_table(data, sim):
    """data/sim band-mean amplitude ratio -- the consistency check."""
    print('\n=== data / sim band-mean amplitude ratio ===')
    print('%-7s %-5s %-6s %8s %12s %12s %9s'
          % ('region', 'plane', 'len', 'n', 'data', 'sim', 'data/sim'))
    worst = 0.0
    for region in REGIONS:
        for ip in range(3):
            for ib in sorted(b for (r, p, b) in data if r == region and p == ip):
                dd, ss = data[(region, ip, ib)], sim[(region, ip, ib)]
                ad, as_ = bands(dd), bands(ss)
                worst = max(worst, abs(ad / as_ - 1.0))
                print('%-7s %-5s %6.0f %8d %12.3e %12.3e %9.3f'
                      % (region, PLANES[ip], dd['meanlen'], dd['n'],
                         ad, as_, ad / as_))
    print('  worst |data/sim - 1| = %.1f%%' % (100 * worst))


def main():
    data = load('data')
    sim = load('sim')
    plot_compare(data, sim)
    plot_striplen(data, sim)
    plot_topbottom(data)
    print_table(data, sim)
    print('done.')


if __name__ == '__main__':
    main()
