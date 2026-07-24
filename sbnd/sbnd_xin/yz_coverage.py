#!/usr/bin/env python3
"""Y-Z coverage maps of SBND imaging output, per TPC drift side.

Scans per-event icluster-apa{0,1}-{active,masked}.npz cluster graphs (from
run_img_evt.sh) and accumulates 2D (z, y) histograms of blob activity for
each anode/side separately:

  - charge  : sum of blob signal ('val', bnodes col 2) at the blob's
              corner-mean (y, z)
  - nblob   : blob count at the corner-mean (y, z)
  - ncorner : every valid blob corner filled individually (finer coverage
              proxy near edges/holes than the center alone)
  - dead    : same corner fill for the *masked* (dead-channel) graph, i.e.
              where imaging placed dead-region blobs

Blob-node array layout (aux/docs/ClusterArrays.org "Blob"):
  col 2 = val (signal), col 14 = ncorners, cols 15.. = up to 12 (y, z)
  corner pairs in WCT system-of-units (mm).

Sides: apa0 = east (x < 0), apa1 = west (x > 0) — matches
run_bee_img_evt.sh bee_anode_args (x0 = -/+201.45 cm).

Usage:
  python3 yz_coverage.py accumulate --work-root work-mcp1000 \
      --out pics/yz_coverage/yz-hist-mcp1000.npz
  python3 yz_coverage.py plot --hist pics/yz_coverage/yz-hist-mcp1000.npz \
      --out-dir pics/yz_coverage
"""

import argparse
import glob
import os
import sys

import numpy as np

# Binning: 1 cm x 1 cm over the full active Y-Z face (cm).
Z_EDGES = np.arange(0.0, 510.0 + 1e-9, 1.0)
Y_EDGES = np.arange(-205.0, 205.0 + 1e-9, 1.0)

SIDES = {0: 'east (x<0)', 1: 'west (x>0)'}


def blob_yz(bn):
    """Return (yc, zc, val, corners_y, corners_z) from a bnodes array, cm."""
    val = bn[:, 2]
    nc = bn[:, 14].astype(int)
    nc = np.clip(nc, 0, 12)
    corners = bn[:, 15:39].reshape(len(bn), 12, 2)  # (y, z) pairs, mm
    mask = np.arange(12)[None, :] < nc[:, None]
    # corner-mean center
    ysum = np.where(mask, corners[:, :, 0], 0.0).sum(axis=1)
    zsum = np.where(mask, corners[:, :, 1], 0.0).sum(axis=1)
    good = nc > 0
    yc = np.zeros(len(bn))
    zc = np.zeros(len(bn))
    yc[good] = ysum[good] / nc[good]
    zc[good] = zsum[good] / nc[good]
    cy = corners[:, :, 0][mask]
    cz = corners[:, :, 1][mask]
    return (yc[good] / 10.0, zc[good] / 10.0, val[good],
            cy / 10.0, cz / 10.0)


def load_bnodes(path):
    """bnodes array from an icluster npz, or None if absent/empty."""
    try:
        with np.load(path) as f:
            for name in f.files:
                if name.endswith('_bnodes'):
                    arr = f[name]
                    return arr if len(arr) else None
    except Exception as e:  # zero-cluster events store no arrays
        print(f'  [warn] {path}: {e}', file=sys.stderr)
    return None


def accumulate(args):
    evt_dirs = sorted(glob.glob(os.path.join(args.work_root, 'evt*')))
    evt_dirs = [d for d in evt_dirs if os.path.isdir(d)]
    print(f'{len(evt_dirs)} event dirs under {args.work_root}')

    nz, ny = len(Z_EDGES) - 1, len(Y_EDGES) - 1
    H = {}
    for apa in (0, 1):
        for key in ('charge', 'nblob', 'ncorner', 'dead'):
            H[f'{key}_apa{apa}'] = np.zeros((nz, ny))
    nevt_used = 0
    missing = []

    for i, d in enumerate(evt_dirs):
        used = False
        for apa in (0, 1):
            act = os.path.join(d, f'icluster-apa{apa}-active.npz')
            if not os.path.exists(act):
                missing.append(act)
                continue
            bn = load_bnodes(act)
            if bn is not None:
                yc, zc, val, cy, cz = blob_yz(bn)
                H[f'charge_apa{apa}'] += np.histogram2d(
                    zc, yc, bins=(Z_EDGES, Y_EDGES), weights=val)[0]
                H[f'nblob_apa{apa}'] += np.histogram2d(
                    zc, yc, bins=(Z_EDGES, Y_EDGES))[0]
                H[f'ncorner_apa{apa}'] += np.histogram2d(
                    cz, cy, bins=(Z_EDGES, Y_EDGES))[0]
                used = True
            msk = os.path.join(d, f'icluster-apa{apa}-masked.npz')
            bn = load_bnodes(msk) if os.path.exists(msk) else None
            if bn is not None:
                _, _, _, cy, cz = blob_yz(bn)
                H[f'dead_apa{apa}'] += np.histogram2d(
                    cz, cy, bins=(Z_EDGES, Y_EDGES))[0]
        nevt_used += used
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(evt_dirs)} events scanned')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, z_edges=Z_EDGES, y_edges=Y_EDGES,
                        nevt=nevt_used, **H)
    print(f'events with blobs: {nevt_used}; missing inputs: {len(missing)}')
    print(f'wrote {args.out}')


def _smooth(h, sigma=2.0):
    """Gaussian smoothing via separable convolution (no scipy dependency)."""
    n = int(3 * sigma)
    x = np.arange(-n, n + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 0, h)
    out = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1, out)
    return out


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    f = np.load(args.hist)
    ze, ye = f['z_edges'], f['y_edges']
    zc = 0.5 * (ze[:-1] + ze[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    nevt = int(f['nevt'])
    os.makedirs(args.out_dir, exist_ok=True)

    def heat(ax, h, title, log=True, cmap='viridis'):
        hp = np.ma.masked_where(h <= 0, h)
        norm = LogNorm(vmin=max(hp.min(), 1e-1), vmax=hp.max()) if log else None
        im = ax.pcolormesh(ze, ye, hp.T, norm=norm, cmap=cmap, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel('z [cm]')
        ax.set_ylabel('y [cm]')
        return im

    # 1) full-face blob-count + charge maps, both sides
    for key, label in (('ncorner', 'blob-corner occupancy'),
                       ('charge', 'blob charge sum')):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        for apa, ax in zip((0, 1), axes):
            h = f[f'{key}_apa{apa}']
            im = heat(ax, h, f'APA{apa} {SIDES[apa]} — {label}, '
                              f'{nevt} events')
            fig.colorbar(im, ax=ax, pad=0.01)
        fig.tight_layout()
        p = os.path.join(args.out_dir, f'yz-{key}.png')
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print('wrote', p)

    # 2) contour view of smoothed occupancy
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for apa, ax in zip((0, 1), axes):
        h = _smooth(f[f'ncorner_apa{apa}'], sigma=args.smooth)
        levels = np.quantile(h[h > 0], [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95])
        im = ax.pcolormesh(ze, ye, h.T, cmap='Greys', rasterized=True)
        cs = ax.contour(zc, yc, h.T, levels=levels, cmap='plasma',
                        linewidths=0.9)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.0f')
        ax.set_title(f'APA{apa} {SIDES[apa]} — smoothed occupancy contours '
                     f'(sigma={args.smooth} cm)')
        ax.set_xlabel('z [cm]')
        ax.set_ylabel('y [cm]')
        fig.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    p = os.path.join(args.out_dir, 'yz-contour.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print('wrote', p)

    # 3) dead-region (masked-graph) maps
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for apa, ax in zip((0, 1), axes):
        im = heat(ax, f[f'dead_apa{apa}'],
                  f'APA{apa} {SIDES[apa]} — dead-region (masked) blob '
                  f'occupancy, {nevt} events', cmap='magma')
        fig.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    p = os.path.join(args.out_dir, 'yz-dead.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print('wrote', p)

    # 4) 1D profiles: occupancy vs y and vs z
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for apa in (0, 1):
        h = f[f'ncorner_apa{apa}']
        axes[0].plot(yc, h.sum(axis=0), label=f'APA{apa} {SIDES[apa]}',
                     lw=0.8)
        axes[1].plot(zc, h.sum(axis=1), label=f'APA{apa} {SIDES[apa]}',
                     lw=0.8)
    axes[0].set_xlabel('y [cm]')
    axes[1].set_xlabel('z [cm]')
    for ax in axes:
        ax.set_ylabel('corner occupancy')
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(args.out_dir, 'yz-profiles.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print('wrote', p)

    # 5) middle-region zoom (chosen from the full map; override via args)
    z0, z1, y0, y1 = args.zoom
    iz = (zc >= z0) & (zc <= z1)
    iy = (yc >= y0) & (yc <= y1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for apa, ax in zip((0, 1), axes):
        h = f[f'ncorner_apa{apa}'][np.ix_(iz, iy)]
        hs = _smooth(f[f'ncorner_apa{apa}'], sigma=args.smooth)[np.ix_(iz, iy)]
        hp = np.ma.masked_where(h <= 0, h)
        im = ax.pcolormesh(ze[np.append(iz, False) | np.append(False, iz)],
                           ye[np.append(iy, False) | np.append(False, iy)],
                           hp.T, cmap='viridis', rasterized=True)
        levels = np.quantile(hs[hs > 0], [0.1, 0.3, 0.5, 0.7, 0.9])
        ax.contour(zc[iz], yc[iy], hs.T, levels=levels, colors='w',
                   linewidths=0.7)
        ax.set_title(f'APA{apa} {SIDES[apa]} — middle-region zoom')
        ax.set_xlabel('z [cm]')
        ax.set_ylabel('y [cm]')
        fig.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    p = os.path.join(args.out_dir, 'yz-middle-zoom.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print('wrote', p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('accumulate')
    a.add_argument('--work-root', default='work-mcp1000')
    a.add_argument('--out', default='pics/yz_coverage/yz-hist-mcp1000.npz')
    a.set_defaults(func=accumulate)
    p = sub.add_parser('plot')
    p.add_argument('--hist', default='pics/yz_coverage/yz-hist-mcp1000.npz')
    p.add_argument('--out-dir', default='pics/yz_coverage')
    p.add_argument('--smooth', type=float, default=2.0)
    p.add_argument('--zoom', type=float, nargs=4,
                   default=[180.0, 320.0, -60.0, 60.0],
                   metavar=('Z0', 'Z1', 'Y0', 'Y1'))
    p.set_defaults(func=plot)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
