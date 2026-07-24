#!/usr/bin/env python3
"""Y-Z coverage maps of SBND imaging output, per TPC drift side.

Scans per-event icluster-apa{0,1}-{active,masked}.npz cluster graphs (from
run_img_evt.sh) and accumulates 2D (z, y) histograms of blob activity for
each anode/side separately:

  - charge  : sum of blob signal ('val', bnodes col 2) at the blob's
              corner-mean (y, z)
  - nblob   : blob count at the corner-mean (y, z)
  - ncorner : every valid blob corner filled individually (biased toward
              blob BOUNDARIES -- kept for comparison)
  - dead    : same corner fill for the *masked* (dead-channel) graph, i.e.
              where imaging placed dead-region blobs
  - cover   : blob-interior fill -- every 1 cm bin whose center lies inside
              the blob's convex (y, z) polygon gets +1 (uniform sampling of
              the blob area; the unbiased coverage-density measure)
  - qspread : blob 'val' spread uniformly over its covered bins
  - deadcover : same interior fill for the masked graph = footprint of the
              declared (>=2-plane) dead volume

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

# SBND fiducial geometry in Y-Z, from
# cfg/pgrapher/experiment/sbnd/clus.jsonnet (read at doc-37 round 4):
#   wire bbox            -> the physical active edge
#   sbnd_pr_fv 'bounds'  -> BoxFiducial, bbox inset by 1 cm
#   + sbnd_pr_fv_margins -> fv_tolerance, the effective FV used by
#     tagger_check_tgm / tagger_check_fc.
WIRE_BBOX = dict(ymin=-200.312, ymax=200.312, zmin=-0.15, zmax=501.15)
FV_BOX = dict(ymin=-199.312, ymax=199.312, zmin=0.85, zmax=500.15)

# Production margins (cm).  The jsonnet knob DEFAULTS are the legacy values
# tgm_fv_x_margin=2, tgm_fv_y_margin=2.5, tgm_fv_zmax_margin=3 (byte-identical);
# production passes -fvx 2.5 -fvy 3 -fvz 5 -fvzi 3.  z_min is fixed at 3 cm.
MARGIN_Y = 3.0        # -fvy  (legacy 2.5)
MARGIN_ZMIN = 3.0     # not knob-parametrized
MARGIN_ZMAX = 5.0     # -fvz  (legacy 3)
MARGIN_ZMAX_INTERIOR = 3.0   # -fvzi, doc 35 (endpoint-only widening)
MARGIN_X = 2.5        # -fvx  (legacy 2); not visible in a Y-Z projection,
                      # effective |x| face = 201.05 - MARGIN_X = 198.55 cm
LEGACY_MARGIN_Y = 2.5

FV_EFF = dict(ymin=FV_BOX['ymin'] + MARGIN_Y, ymax=FV_BOX['ymax'] - MARGIN_Y,
              zmin=FV_BOX['zmin'] + MARGIN_ZMIN,
              zmax=FV_BOX['zmax'] - MARGIN_ZMAX)
FV_EFF_ZMAX_INTERIOR = FV_BOX['zmax'] - MARGIN_ZMAX_INTERIOR   # 497.15
FV_EFF_Y_LEGACY = FV_BOX['ymax'] - LEGACY_MARGIN_Y             # 196.812


def draw_fv(ax, which='all'):
    """Overlay the three nested Y-Z boundaries on a (z, y) axes."""
    kw_eff = dict(color='red', lw=1.4, ls='--')
    kw_box = dict(color='white', lw=1.0, ls=':')
    kw_bb = dict(color='0.7', lw=0.8, ls='-')
    for d, kw in ((WIRE_BBOX, kw_bb), (FV_BOX, kw_box), (FV_EFF, kw_eff)):
        ax.plot([d['zmin'], d['zmax'], d['zmax'], d['zmin'], d['zmin']],
                [d['ymin'], d['ymin'], d['ymax'], d['ymax'], d['ymin']], **kw)
    ax.plot([], [], **kw_eff,
            label='effective FV (-fvx 2.5 -fvy 3 -fvz 5)')
    ax.plot([], [], **kw_box, label='BoxFiducial bounds')
    ax.plot([], [], **kw_bb, label='wire bbox (active edge)')


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


def blob_polys(bn):
    """Yield (poly_yz_cm, val) per blob: corners angle-ordered around the
    centroid (blob polygons are convex intersections of wire half-planes)."""
    val = bn[:, 2]
    nc = np.clip(bn[:, 14].astype(int), 0, 12)
    corners = bn[:, 15:39].reshape(len(bn), 12, 2) / 10.0  # (y, z) cm
    for i in range(len(bn)):
        k = nc[i]
        if k < 3:
            continue
        p = corners[i, :k]
        c = p.mean(axis=0)
        order = np.argsort(np.arctan2(p[:, 0] - c[0], p[:, 1] - c[1]))
        yield p[order], val[i]


def rasterize(bn, hist_cover, hist_q=None):
    """Fill hist[z, y] bins whose centers lie inside each blob polygon.

    Convex point-in-polygon by half-plane cross products.  A blob covering
    no bin center falls back to its centroid bin.
    """
    z0, y0 = Z_EDGES[0], Y_EDGES[0]
    nz, ny = hist_cover.shape
    for poly, val in blob_polys(bn):
        py, pz = poly[:, 0], poly[:, 1]
        iz0 = max(int(np.floor(pz.min() - z0)), 0)
        iz1 = min(int(np.ceil(pz.max() - z0)), nz - 1)
        iy0 = max(int(np.floor(py.min() - y0)), 0)
        iy1 = min(int(np.ceil(py.max() - y0)), ny - 1)
        if iz1 < iz0 or iy1 < iy0:
            continue
        zg = z0 + np.arange(iz0, iz1 + 1) + 0.5
        yg = y0 + np.arange(iy0, iy1 + 1) + 0.5
        ZZ, YY = np.meshgrid(zg, yg, indexing='ij')
        inside = np.ones(ZZ.shape, dtype=bool)
        for a in range(len(poly)):
            b = (a + 1) % len(poly)
            # cross((zb-za, yb-ya), (Z-za, Y-ya)) >= 0 for all edges (CCW)
            cr = ((pz[b] - pz[a]) * (YY - py[a])
                  - (py[b] - py[a]) * (ZZ - pz[a]))
            inside &= cr >= 0
        if not inside.any():
            cy = int(py.mean() - y0)
            cz = int(pz.mean() - z0)
            if 0 <= cz < nz and 0 <= cy < ny:
                hist_cover[cz, cy] += 1
                if hist_q is not None:
                    hist_q[cz, cy] += val
            continue
        sub = hist_cover[iz0:iz1 + 1, iy0:iy1 + 1]
        sub[inside] += 1
        if hist_q is not None:
            hist_q[iz0:iz1 + 1, iy0:iy1 + 1][inside] += val / inside.sum()


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
        for key in ('charge', 'nblob', 'ncorner', 'dead',
                    'cover', 'qspread', 'deadcover'):
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
                rasterize(bn, H[f'cover_apa{apa}'], H[f'qspread_apa{apa}'])
                used = True
            msk = os.path.join(d, f'icluster-apa{apa}-masked.npz')
            bn = load_bnodes(msk) if os.path.exists(msk) else None
            if bn is not None:
                _, _, _, cy, cz = blob_yz(bn)
                H[f'dead_apa{apa}'] += np.histogram2d(
                    cz, cy, bins=(Z_EDGES, Y_EDGES))[0]
                rasterize(bn, H[f'deadcover_apa{apa}'])
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

    # 4b) blob-interior coverage density (unbiased area sampling)
    if 'cover_apa0' in f:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        for apa, ax in zip((0, 1), axes):
            im = heat(ax, f[f'cover_apa{apa}'],
                      f'APA{apa} {SIDES[apa]} — blob-interior coverage '
                      f'density, {nevt} events')
            draw_fv(ax)
            ax.legend(loc='lower left', fontsize=7, framealpha=0.75)
            fig.colorbar(im, ax=ax, pad=0.01)
        fig.tight_layout()
        p = os.path.join(args.out_dir, 'yz-cover.png')
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print('wrote', p)

        # FV edge check: 1D roll-off at each of the four Y-Z faces, each
        # normalized to its OWN interior reference 10-30 cm inside (the
        # cosmic illumination gradient makes a global plateau meaningless).
        EDGES = [
            ('z_min', 'z', +1, FV_EFF['zmin'], FV_BOX['zmin'],
             WIRE_BBOX['zmin'], (-6, 22)),
            ('z_max', 'z', -1, FV_EFF['zmax'], FV_BOX['zmax'],
             WIRE_BBOX['zmax'], (478, 506)),
            ('y_min', 'y', +1, FV_EFF['ymin'], FV_BOX['ymin'],
             WIRE_BBOX['ymin'], (-206, -178)),
            ('y_max', 'y', -1, FV_EFF['ymax'], FV_BOX['ymax'],
             WIRE_BBOX['ymax'], (178, 206)),
        ]
        fig, axes = plt.subplots(2, 4, figsize=(19, 8))
        for apa in (0, 1):
            cov = f[f'cover_apa{apa}']
            zin = (zc >= 20) & (zc <= 480)
            yin = (yc >= -190) & (yc <= 190)
            for ie, (nm, ax_, sgn, eff, box, bb, win) in enumerate(EDGES):
                ax = axes[apa][ie]
                if ax_ == 'z':
                    coord, prof = zc, cov[:, yin].sum(axis=1) / yin.sum()
                else:
                    coord, prof = yc, cov[zin].sum(axis=0) / zin.sum()
                # interior reference: 10-30 cm inside this face
                ref_lo, ref_hi = sorted([bb + sgn * 10, bb + sgn * 30])
                ref = np.median(prof[(coord >= ref_lo) & (coord <= ref_hi)])
                r = prof / ref if ref > 0 else prof * 0
                m = (coord >= win[0]) & (coord <= win[1])
                ax.step(coord[m], r[m], where='mid', color='k', lw=1.1)
                ax.axhline(1.0, color='k', lw=0.4, alpha=0.4)
                ax.axhline(0.5, color='0.6', lw=0.6, ls=':')
                ax.axvline(eff, color='red', ls='--', lw=1.3,
                           label='effective FV')
                ax.axvline(box, color='tab:blue', ls=':', lw=1.2,
                           label='BoxFiducial')
                ax.axvline(bb, color='0.4', lw=1.0, label='wire bbox')
                if nm == 'z_max':
                    ax.axvline(FV_EFF_ZMAX_INTERIOR, color='orange', ls='-.',
                               lw=1.0, label='interior FV (-fvzi 3)')
                if nm in ('y_min', 'y_max'):
                    leg = (FV_EFF_Y_LEGACY if nm == 'y_max'
                           else -FV_EFF_Y_LEGACY)
                    ax.axvline(leg, color='green', ls='-.', lw=1.0,
                               label='legacy FV (-fvy 2.5)')
                    ax.legend(fontsize=6.5, loc='lower right')
                # density fraction exactly at the FV face
                fr = np.interp(eff, coord, r)
                ax.plot([eff], [fr], 'ro', ms=4)
                ax.annotate('%.0f%% at FV' % (100 * fr), (eff, fr),
                            textcoords='offset points', xytext=(6, 8),
                            fontsize=8, color='red')
                ax.set_title(f'APA{apa} — {nm}', fontsize=10)
                ax.set_xlabel(f'{ax_} [cm]')
                ax.set_ylabel('density / interior')
                ax.set_ylim(-0.05, 1.35)
                ax.grid(alpha=0.3)
                if apa == 0 and ie == 0:
                    ax.legend(fontsize=7, loc='lower right')
        fig.suptitle('FV faces vs where blob coverage actually ends '
                     '(normalized to the interior 10-30 cm inside each face)')
        fig.tight_layout()
        p = os.path.join(args.out_dir, 'yz-fv-edges.png')
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print('wrote', p)

        # 2D zoom strips along each face -- is the edge uniform, or does it
        # have structure the single FV plane cannot follow?
        fig, axes = plt.subplots(4, 2, figsize=(15, 14))
        for ie, (nm, ax_, sgn, eff, box, bb, win) in enumerate(EDGES):
            for apa in (0, 1):
                ax = axes[ie][apa]
                cov = f[f'cover_apa{apa}']
                if ax_ == 'z':
                    m = (zc >= win[0]) & (zc <= win[1])
                    sub = cov[m]
                    hp = np.ma.masked_where(sub <= 0, sub)
                    im = ax.pcolormesh(
                        ze[np.append(m, False) | np.append(False, m)], ye,
                        hp.T, cmap='viridis', rasterized=True)
                    ax.axvline(eff, color='red', ls='--', lw=1.2)
                    ax.axvline(box, color='w', ls=':', lw=1.0)
                    ax.axvline(bb, color='0.8', lw=0.9)
                    ax.set_xlabel('z [cm]')
                    ax.set_ylabel('y [cm]')
                else:
                    m = (yc >= win[0]) & (yc <= win[1])
                    sub = cov[:, m]
                    hp = np.ma.masked_where(sub <= 0, sub)
                    im = ax.pcolormesh(
                        ze, ye[np.append(m, False) | np.append(False, m)],
                        hp.T, cmap='viridis', rasterized=True)
                    ax.axhline(eff, color='red', ls='--', lw=1.2)
                    ax.axhline(box, color='w', ls=':', lw=1.0)
                    ax.axhline(bb, color='0.8', lw=0.9)
                    ax.set_xlabel('z [cm]')
                    ax.set_ylabel('y [cm]')
                ax.set_title(f'APA{apa} {SIDES[apa]} — {nm} edge '
                             f'(red dashed = effective FV)', fontsize=9)
                fig.colorbar(im, ax=ax, pad=0.01)
        fig.tight_layout()
        p = os.path.join(args.out_dir, 'yz-fv-edge-maps.png')
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print('wrote', p)

        # seam-density answer figure: is there low density BEYOND the
        # declared (masked, >=2-plane) dead footprint?
        seam = (zc >= 248) & (zc <= 253)
        side = ((zc >= 230) & (zc <= 245)) | ((zc >= 256) & (zc <= 271))
        fig, axes = plt.subplots(3, 2, figsize=(14, 13))
        for col, apa in enumerate((0, 1)):
            cov = f[f'cover_apa{apa}']
            dcov = f[f'deadcover_apa{apa}']
            zz = (zc >= 230) & (zc <= 271)
            ax = axes[0][col]
            prof = cov.sum(axis=1)
            ax.step(zc[zz], prof[zz] / np.median(prof[side]), where='mid',
                    label='interior coverage density')
            qs = f[f'qspread_apa{apa}'].sum(axis=1)
            ax.step(zc[zz], qs[zz] / np.median(qs[side]), where='mid',
                    label='charge density (spread)')
            dp = dcov.sum(axis=1)
            ax.step(zc[zz], dp[zz] / max(dp[zz].max(), 1), where='mid',
                    label='declared-dead footprint (a.u.)', alpha=0.6)
            ax.axvspan(248, 253, color='r', alpha=0.08)
            ax.axhline(1, color='k', lw=0.5)
            ax.set_title(f'APA{apa} {SIDES[apa]} — density at the seam')
            ax.set_xlabel('z [cm]')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            # y-resolved seam ratio + dead fraction
            ax = axes[1][col]
            s = cov[seam].sum(axis=0) / seam.sum()
            b = cov[side].sum(axis=0) / side.sum()
            r = np.where(b > 0, s / b, np.nan)
            ax.step(yc, r, where='mid', lw=0.8, label='seam/sideband density')
            dfrac = (dcov[seam] > 0).mean(axis=0)
            ax.step(yc, dfrac, where='mid', lw=0.8, color='r', alpha=0.6,
                    label='fraction of seam bins declared dead')
            ax.axhline(1, color='k', lw=0.5)
            ax.set_title(f'APA{apa} — seam(248-253) density ratio vs y')
            ax.set_xlabel('y [cm]')
            ax.set_ylim(0, 2.5)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            # 2D density zoom with declared-dead contour
            ax = axes[2][col]
            zz2 = (zc >= 240) & (zc <= 261)
            hp = np.ma.masked_where(cov[zz2] <= 0, cov[zz2])
            im = ax.pcolormesh(ze[np.append(zz2, False) | np.append(False, zz2)],
                               ye, hp.T, cmap='viridis', rasterized=True)
            dd = (dcov[zz2] > 0).astype(float)
            ax.contour(zc[zz2], yc, dd.T, levels=[0.5], colors='r',
                       linewidths=0.6)
            ax.set_title(f'APA{apa} — density zoom, declared dead in red')
            ax.set_xlabel('z [cm]')
            ax.set_ylabel('y [cm]')
            fig.colorbar(im, ax=ax, pad=0.01)
        fig.tight_layout()
        p = os.path.join(args.out_dir, 'yz-seam-density.png')
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print('wrote', p)

    # 4c) LITERAL OVERLAY: coverage density + declared 2-plane-dead footprint,
    #     middle region, FULL y range -- "what is left low after the dead
    #     blobs are accounted for?"
    if 'deadcover_apa0' in f:
        pers = args.dead_persist * 7 * nevt  # dead blob in >=frac of events
        z0, z1 = args.mid_z
        zz = (zc >= z0) & (zc <= z1)
        fig, axes = plt.subplots(1, 2, figsize=(15, 9))
        for apa, ax in zip((0, 1), axes):
            cov = f[f'cover_apa{apa}']
            dc = f[f'deadcover_apa{apa}']
            hp = np.ma.masked_where(cov[zz] <= 0, cov[zz])
            im = ax.pcolormesh(ze[np.append(zz, False) | np.append(False, zz)],
                               ye, hp.T, cmap='viridis', rasterized=True)
            # declared dead: persistent (red) and transient (thin white)
            yy, xx = np.meshgrid(yc, zc[zz], indexing='ij')
            pm = (dc[zz] >= pers).T
            tm = ((dc[zz] > 0) & (dc[zz] < pers)).T
            ax.scatter(xx[tm], yy[tm], s=1.2, c='w', alpha=0.35, marker='s',
                       linewidths=0)
            ax.scatter(xx[pm], yy[pm], s=6, c='red', marker='s',
                       linewidths=0,
                       label=f'persistent 2-plane dead (>{args.dead_persist:.0%} evts)')
            ax.scatter([], [], s=6, c='w', marker='s',
                       label='transient dead blob')
            ax.set_title(f'APA{apa} {SIDES[apa]} — coverage density + dead '
                         f'overlay')
            ax.set_xlabel('z [cm]')
            ax.set_ylabel('y [cm]')
            ax.set_ylim(ye[0], ye[-1])
            ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
            fig.colorbar(im, ax=ax, pad=0.01)
        fig.suptitle('Middle region, full Y range: is any low-density area '
                     'NOT covered by the declared dead footprint?')
        fig.tight_layout()
        p = os.path.join(args.out_dir, 'yz-overlay-middle.png')
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print('wrote', p)

        # 4d) coherence map: per-y-band local z-ratio.  Detector structure is
        # COHERENT across y (a column); cosmic-track texture is not.
        BAND = 20
        bands = [(y, y + BAND) for y in range(int(ye[0]) + 5,
                                              int(ye[-1]) - 5, BAND)]
        fig, axes = plt.subplots(4, 1, figsize=(15, 13),
                                 gridspec_kw={'height_ratios': [3, 1, 3, 1]})
        for apa in (0, 1):
            axm, axc = axes[2 * apa], axes[2 * apa + 1]
            cov = f[f'cover_apa{apa}']
            dc = f[f'deadcover_apa{apa}']
            R = np.full((len(zc), len(bands)), np.nan)
            for ib, (y0, y1) in enumerate(bands):
                sel = (yc >= y0) & (yc < y1)
                prof = cov[:, sel].sum(axis=1)
                for iz in range(25, len(zc) - 25):
                    loc = np.concatenate([prof[iz - 22:iz - 3],
                                          prof[iz + 4:iz + 23]])
                    e = np.median(loc)
                    if e > 0:
                        R[iz, ib] = prof[iz] / e
            im = axm.pcolormesh(ze, [b[0] for b in bands] + [bands[-1][1]],
                                R.T, cmap='RdBu_r', vmin=0.7, vmax=1.3,
                                rasterized=True)
            zd = zc[(dc >= pers).sum(axis=1) > 0]
            for z in zd:
                axm.axvline(z, color='k', lw=0.8, alpha=0.5)
            axm.set_title(f'APA{apa} {SIDES[apa]} — local z-ratio per {BAND} cm'
                          f' y-band (black lines = persistent 2-plane dead)')
            axm.set_ylabel('y [cm]')
            fig.colorbar(im, ax=axm, pad=0.01,
                         label='density / local sideband')
            # coherence readout: how many y-bands agree at each z
            nd = np.nansum(R < 0.90, axis=1)
            ne = np.nansum(R > 1.15, axis=1)
            inner = (zc >= 2) & (zc <= 499)
            axc.fill_between(zc, 0, ne, step='mid', color='firebrick',
                             alpha=0.8, label='# y-bands in EXCESS (>1.15)')
            axc.fill_between(zc, 0, -nd, step='mid', color='steelblue',
                             alpha=0.8, label='# y-bands DEFICIENT (<0.90)')
            thr = np.percentile(nd[inner], 99)
            axc.axhline(-thr, color='k', ls=':', lw=0.8)
            axc.axhline(thr, color='k', ls=':', lw=0.8)
            for z in zd:
                axc.axvline(z, color='k', lw=0.8, alpha=0.5)
            axc.set_ylabel('coherence')
            axc.set_xlabel('z [cm]')
            axc.set_ylim(-len(bands), len(bands))
            axc.legend(fontsize=7, loc='upper right', ncol=2)
            axc.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(args.out_dir, 'yz-coherence.png')
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


def census(args):
    """Quantify structure vs cosmic texture and list deficits that the
    declared 2-plane dead footprint does NOT explain."""
    f = np.load(args.hist)
    ze, ye = f['z_edges'], f['y_edges']
    zc = 0.5 * (ze[:-1] + ze[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    nevt = int(f['nevt'])
    pers = args.dead_persist * 7 * nevt
    BAND = 20
    bands = [(y, y + BAND) for y in range(int(ye[0]) + 5,
                                          int(ye[-1]) - 5, BAND)]

    def band_ratio(h, z0, z1, y0, y1):
        zs = (zc >= z0) & (zc <= z1)
        ys = (yc >= y0) & (yc < y1)
        sb = (((zc >= z0 - 22) & (zc <= z0 - 4))
              | ((zc >= z1 + 4) & (zc <= z1 + 22)))
        a = h[np.ix_(zs, ys)].sum() / zs.sum()
        b = h[np.ix_(sb, ys)].sum() / sb.sum()
        return a / b if b > 0 else float('nan')

    for apa in (0, 1):
        cov = f[f'cover_apa{apa}']
        dc = f[f'deadcover_apa{apa}']
        print(f'===== APA{apa} {SIDES[apa]} =====')
        # cosmic-texture floor: per-bin scatter vs Poisson
        n = int(3 * 12)
        x = np.arange(-n, n + 1)
        k = np.exp(-0.5 * (x / 12.0) ** 2)
        k /= k.sum()
        g = lambda a: np.apply_along_axis(
            lambda r: np.convolve(r, k, 'same'), 1,
            np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 0, a))
        exp = g(cov) / g(np.ones_like(cov))
        R = cov / np.maximum(exp, 1e-9)
        m = (zc >= 2) & (zc <= 499)
        q = (yc >= -198) & (yc <= 198)
        print('  per-bin texture: sigma(R)=%.3f vs Poisson %.3f '
              '(over-dispersion x%.2f)'
              % (R[np.ix_(m, q)].std(),
                 1 / np.sqrt(np.median(cov[cov > 0])),
                 R[np.ix_(m, q)].std() * np.sqrt(np.median(cov[cov > 0]))))
        # coherence across y bands
        Rm = np.full((len(zc), len(bands)), np.nan)
        for ib, (y0, y1) in enumerate(bands):
            sel = (yc >= y0) & (yc < y1)
            prof = cov[:, sel].sum(axis=1)
            for iz in range(25, len(zc) - 25):
                loc = np.concatenate([prof[iz - 22:iz - 3],
                                      prof[iz + 4:iz + 23]])
                e = np.median(loc)
                if e > 0:
                    Rm[iz, ib] = prof[iz] / e
        nd = np.nansum(Rm < 0.90, axis=1)
        ne = np.nansum(Rm > 1.15, axis=1)
        deadz = (dc >= pers).sum(axis=1)
        inner = (zc >= 2) & (zc <= 499)
        print('  y-band coherence floor: median %d/%d bands deficient, '
              '99th pct %d' % (np.median(nd[inner]), len(bands),
                               np.percentile(nd[inner], 99)))
        print('  -- coherent EXCESS columns (dead-plane 2-view inflation) --')
        for iz in np.argsort(-ne)[:8]:
            if ne[iz] < 10 or not inner[iz]:
                continue
            print('     z=%6.1f  %2d/%d bands  meanR=%.2f  '
                  'persistent-dead bins: %d'
                  % (zc[iz], ne[iz], len(bands), np.nanmean(Rm[iz]),
                     deadz[iz]))
        print('  -- coherent DEFICIT columns (candidate unexplained) --')
        for iz in np.argsort(-nd)[:8]:
            if nd[iz] < 10 or not inner[iz]:
                continue
            print('     z=%6.1f  %2d/%d bands  meanR=%.2f  '
                  'persistent-dead bins: %d'
                  % (zc[iz], nd[iz], len(bands), np.nanmean(Rm[iz]),
                     deadz[iz]))
        print('  -- middle region (seam z 248-253), FULL y --')
        print('     all-y %.3f   TOP(y>0) %.3f   BOTTOM(y<0) %.3f'
              % (band_ratio(cov, 248, 253, -200, 200),
                 band_ratio(cov, 248, 253, 0, 200),
                 band_ratio(cov, 248, 253, -200, 0)))
        for y0 in range(-200, 200, 40):
            print('       y[%5d,%5d] %.3f'
                  % (y0, y0 + 40, band_ratio(cov, 248, 253, y0, y0 + 40)))
        # zero-coverage census in the seam, full y
        seam = (zc >= 248) & (zc <= 253)
        inb = np.abs(yc) <= 199
        zero = [(zc[iz], yc[iy], dc[iz, iy] >= pers)
                for iz in np.where(seam)[0] for iy in np.where(inb)[0]
                if cov[iz, iy] == 0]
        print('     zero-coverage seam bins: %d, of which persistent-dead: %d'
              % (len(zero), sum(1 for *_x, d in zero if d)))
        for z, y, d in zero:
            if not d:
                print('       UNEXPLAINED zero bin z=%.1f y=%.1f' % (z, y))


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
    p.add_argument('--mid-z', type=float, nargs=2, default=[225.0, 276.0],
                   metavar=('Z0', 'Z1'),
                   help='middle-region z window for the dead overlay '
                        '(full y range is always shown)')
    p.add_argument('--dead-persist', type=float, default=0.5,
                   help='a bin counts as declared dead when a dead blob sits '
                        'there in >= this fraction of events (guards against '
                        'a single-event dead blob masking a real deficit)')
    p.set_defaults(func=plot)
    c = sub.add_parser('census')
    c.add_argument('--hist', default='pics/yz_coverage/yz-hist-mcp1000-v2.npz')
    c.add_argument('--dead-persist', type=float, default=0.5)
    c.set_defaults(func=census)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
