#!/usr/bin/env python3
'''
doc pr/77 -- pr/52 Tier B: histogram the actual charge feature
q = dQ*dQdx_scale + dQdx_offset fed to the net on SBND events.

The uBooNE-era transform (0.1 / -1000, a ported "hack for now") was chosen
for the uBooNE training distribution; if SBND's mapped values sit elsewhere,
the net's confidence is distorted before any threshold matters.  Reference
line: a MIP-like point contributes dQ ~ mip_dqdx_median * median(dx), both
taken from the calib meta / points, i.e. q_mip ~ that*scale + offset.

Usage:
  python3 qfeature_check.py --tags vtxscan-prod0813 vtxscan-prod0813-ncpi0 \
      --png runs/qfeature.png
'''
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+',
                    default=['vtxscan-prod0813', 'vtxscan-prod0813-ncpi0'])
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--png', default=None)
    args = ap.parse_args()

    allq, dxs, mips = [], [], []
    for label in vio.iter_labels(args.sbnd_root, args.tags):
        calib = vio.load_calib(vio.calib_path_for_label(args.sbnd_root, label))
        xyz, q, info = vio.rebuild_cloud(calib)
        allq.append(q)
        mip = calib.get('meta', {}).get('mip_dqdx_median')
        if mip:
            mips.append(float(mip))
        for s in calib.get('segments', []):
            for p in s.get('points', [])[1:-1]:
                dxs.append(float(p['dx']))
    q = np.concatenate(allq)
    dx = np.array(dxs)
    scale, offset = 0.1, -1000.0

    print('q = dQ*%.1f%+.0f over %d points (%d events):' % (scale, offset, len(q), len(allq)))
    for p in (1, 5, 25, 50, 75, 95, 99):
        print('  p%02d = %10.1f' % (p, np.percentile(q, p)))
    print('  mean=%.1f  frac<0 = %.3f  frac<-990 (dQ~0) = %.4f'
          % (q.mean(), (q < 0).mean(), (q < -990).mean()))
    q_mip = None
    if mips and len(dx):
        q_mip = np.median(mips) * np.median(dx) * scale + offset
        print('MIP reference: mip_dqdx_median=%.0f x median dx=%.2f cm -> q_mip ~ %.0f'
              % (np.median(mips), np.median(dx), q_mip))

    if args.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(np.clip(q, -2000, 20000), bins=220, histtype='stepfilled', alpha=0.7)
        ax.axvline(0, color='k', lw=0.8, ls=':')
        if q_mip is not None:
            ax.axvline(q_mip, color='r', lw=1.2, ls='--', label='MIP ref ~ %.0f' % q_mip)
            ax.legend()
        ax.set_xlabel('q = dQ*0.1 - 1000 (net input feature, clipped at 2e4)')
        ax.set_ylabel('points')
        ax.set_title('SBND DL-vertex net charge feature, %d events' % len(allq))
        ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(args.png, dpi=120)
        print('wrote %s' % args.png)
    return 0


if __name__ == '__main__':
    sys.exit(main())
