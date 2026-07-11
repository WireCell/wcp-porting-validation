#!/usr/bin/env python3
"""A/B the v5 photon libraries (Ar 128 nm vs Xe 175 nm) on beam-flash GOLD pairs.

Which optical model fits the data better?  (pdvd-questions-dune.md sec 3 --
asked externally, answered here from the data side.)

Uses the same external ground truth as fit_qtol_gold.py: in the CTB
beam-triggered events the beam flash's folded time is KNOWN
(t = tc_us - charge_bde_us) and its charge partner is the largest-predicted
bundle on that flash.  For each gold pair the per-channel prediction is
recomputed IN PYTHON from the calib dump's cluster points (per-point charge q,
drift-shifted x, PE-inclusion gate, trilinear library lookup, official
VUVEfficiency, dump QtoL) -- validated against the dump's own pred_pe
(128 nm production library) -- and then swapped to the 175 nm grid.

Comparison metrics per library, over the SAME live channel set (static
ch_mask + per-event auto-mask, i.e. the Ar-live set -- conservative for the
Xe hypothesis, which would additionally light ch 13/29/32/39):
  - per-pair KS distance between normalized measured and predicted patterns;
  - per-pair total closure ratio sum(meas)/sum(pred) -> scatter;
  - per-channel median meas/pred -> cross-channel spread (the "x13 within
    cathode XA" number) per PD group;
  - paired KS comparison (which library wins per gold pair).

Usage:
    python3 ablib_gold.py 'work/0392*_*/calib-evt*.json' 'work/0393*_*/calib-evt*.json' \
        [--table ../data/jjo_triglight_offsets.txt] \
        [--lib128 ../photlib/pdvd-photlib-vis-v5-128nm.json] \
        [--lib175 ../photlib/pdvd-photlib-vis-v5-175nm.json] \
        [--win 5] [--pe-min 500]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

# Official per-OpDet VUV efficiencies (PDVD_PDS_Mapping_v04152025), verbatim
# from cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet.
VUV_EFF = np.array([
    0.03, 0.03, 0.03, 0.03,
    0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
    0.03, 0.0,
    0.12, 0.036, 0.12, 0.12,
    0.03, 0.03,
    0.12, 0.036, 0.12, 0.12,
    0.036, 0.036, 0.036, 0.036, 0.036, 0.0,
    0.036, 0.036, 0.0, 0.036, 0.036, 0.036,
    0.036, 0.036, 0.036, 0.0])

CH_GROUP = {**{i: 'memXA-top' for i in (0, 1, 2, 3)},
            **{i: 'cathXA' for i in range(4, 12)},
            12: 'memXA-bot', 13: 'memXA-bot', 18: 'memXA-bot', 19: 'memXA-bot',
            **{i: 'zPMT' for i in (14, 15, 16, 17, 20, 21, 22, 23)},
            **{i: 'botPMT' for i in range(24, 40)}}

ANODE_EXT1 = -2.0    # cm, QLMatching m_anode_ext1
CATHODE_EXT1 = 1.2   # cm, QLMatching m_cathode_ext1

# Ar-blind channels (static ch_mask members present in the readout) with the
# type-nominal efficiency they would carry if live at 175 nm: 13 = membrane XA
# (XA 0.03), 29/39 = PEN+Q PMTs and 32 = uncoated PMT (PMT-glass VUV cutoff
# makes 32's true 175 nm efficiency questionable -- treat as an upper bound).
ARBLIND_EFF = {13: 0.03, 29: 0.036, 32: 0.036, 39: 0.036}


class GridLib:
    """Trilinear, boundary-clamped lookup -- PhotonLibraryModel semantics."""

    def __init__(self, meta_path):
        meta = json.load(open(meta_path))
        self.origin = np.array(meta['origin_cm'])
        self.step = np.array(meta['step_cm'])
        self.n = np.array(meta['n'])
        npy = meta['vis_npy']
        if not os.path.isabs(npy):
            npy = os.path.join(os.path.dirname(meta_path), npy)
        self.vis = np.load(npy)          # [nx, ny, nz, nch] float32

    def __call__(self, pts_cm):
        """pts_cm [N,3] -> visibilities [N, nch]."""
        g = (pts_cm - self.origin) / self.step
        g = np.clip(g, 0.0, self.n - 1.000001)
        i0 = g.astype(np.int64)
        f = g - i0
        out = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (np.where(dx, f[:, 0], 1 - f[:, 0])
                         * np.where(dy, f[:, 1], 1 - f[:, 1])
                         * np.where(dz, f[:, 2], 1 - f[:, 2]))
                    v = self.vis[np.minimum(i0[:, 0] + dx, self.n[0] - 1),
                                 np.minimum(i0[:, 1] + dy, self.n[1] - 1),
                                 np.minimum(i0[:, 2] + dz, self.n[2] - 1)]
                    out = out + w[:, None] * v
        return out


def predict(d, bundle, lib, live):
    """Per-channel prediction for a bundle, QLMatching vis-loop semantics."""
    qp = d['quality_params']
    geo = d['geometry'][str(bundle['apa'])]
    toffs = d['trigger_offsets_us']
    v = d['drift_speed']
    fl = next(f for f in d['flashes'] if f['gid'] == bundle['flash_gid'])
    # dump flash times fold the input-0 (BDE) offset; the top-volume (apa 4)
    # side adds the TDE-BDE crate skew
    t_side = fl['time'] + ((toffs[1] - toffs[0]) if bundle['apa'] == 4 else 0.0)
    xoff = geo['sign_offset'] * t_side * v

    cl = {c['uid']: c for c in d['clusters']}
    pred = np.zeros(len(VUV_EFF))
    for uid in [bundle['main_cluster']] + bundle['other_clusters']:
        c = cl[uid]
        x = np.array(c['x']) + xoff
        y, z, q = np.array(c['y']), np.array(c['z']), np.array(c['q'])
        u = geo['s'] * (x - geo['anode_x'])
        keep = ((u >= ANODE_EXT1) & (u <= geo['u_cathode'] + CATHODE_EXT1)
                & (y >= geo['y_lo']) & (y <= geo['y_hi'])
                & (z >= geo['z_lo']) & (z <= geo['z_hi']))
        if not keep.any():
            continue
        pts = np.stack([x[keep], y[keep], z[keep]], axis=1)
        qk = q[keep]
        for lo in range(0, len(pts), 8192):
            vis = lib(pts[lo:lo + 8192])
            pred += (qk[lo:lo + 8192, None] * vis).sum(axis=0)
    pred *= qp['QtoL'] * VUV_EFF
    pred[~live] = 0.0
    return pred


def ks_dis(meas, pred, live):
    m, p = meas[live], pred[live]
    if m.sum() <= 0 or p.sum() <= 0:
        return 1.0
    return float(np.abs(np.cumsum(m) / m.sum() - np.cumsum(p) / p.sum()).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('dumps', nargs='+')
    ap.add_argument('--table', default=os.path.join(os.path.dirname(__file__),
                                                    '..', 'data', 'jjo_triglight_offsets.txt'))
    here = os.path.dirname(__file__)
    ap.add_argument('--lib128', default=os.path.join(here, '..', 'photlib',
                                                     'pdvd-photlib-vis-v5-128nm.json'))
    ap.add_argument('--lib175', default=os.path.join(here, '..', 'photlib',
                                                     'pdvd-photlib-vis-v5-175nm.json'))
    ap.add_argument('--win', type=float, default=5.0)
    ap.add_argument('--pe-min', type=float, default=500.0)
    args = ap.parse_args()

    tbl = {}
    for line in open(args.table):
        if line.startswith('#'):
            continue
        f = line.split()
        tbl[(int(f[0]), int(f[2]))] = float(f[6]) - float(f[8])   # tc - charge_bde

    libs = {'128nm': GridLib(args.lib128), '175nm': GridLib(args.lib175)}

    paths = []
    for pat in args.dumps:
        paths += glob.glob(pat)
    pairs = []           # (label, meas, {lib: pred}, dump_pred, live)
    val_tot, val_ch = [], []
    for p in sorted(paths):
        m = re.search(r'(\d{5,6})_\d+/calib-evt(\d+)\.json', p.replace('\\', '/'))
        if not m:
            continue
        t_exp = tbl.get((int(m.group(1)), int(m.group(2))))
        if t_exp is None:
            continue
        d = json.load(open(p))
        fl = d['flashes']
        ts = np.array([f['time'] for f in fl])
        i = int(np.abs(ts - t_exp).argmin())
        if abs(ts[i] - t_exp) > args.win or fl[i]['total_PE'] < args.pe_min:
            continue
        bs = [b for b in d['bundles'] if b['flash_gid'] == fl[i]['gid']
              and b['total_pred_light'] > 0]
        if not bs:
            continue
        b = max(bs, key=lambda x: x['total_pred_light'])
        live = np.array([od['active'] and not od['auto_masked'] for od in d['opdets']])
        meas = np.array(fl[i]['pe'])
        preds = {k: predict(d, b, lib, live) for k, lib in libs.items()}
        # Xe hypothesis test: 175 nm prediction ALSO on the Ar-blind channels,
        # with their type-nominal efficiencies
        live_ext = live.copy()
        live_ext[list(ARBLIND_EFF)] = True
        eff_sav = VUV_EFF[list(ARBLIND_EFF)].copy()
        VUV_EFF[list(ARBLIND_EFF)] = list(ARBLIND_EFF.values())
        preds['175ext'] = predict(d, b, libs['175nm'], live_ext)
        VUV_EFF[list(ARBLIND_EFF)] = eff_sav
        dump_pred = np.array(b['pred_pe'])
        # validation of the recompute against the dump's own 128 nm result
        if dump_pred.sum() > 0:
            val_tot.append(preds['128nm'].sum() / dump_pred.sum())
            sel = dump_pred > 0.1
            if sel.any():
                val_ch.append(np.abs(preds['128nm'][sel] / dump_pred[sel] - 1).max())
        pairs.append((f'{m.group(1)}/{m.group(2)}', meas, preds, live))

    print(f'gold pairs: {len(pairs)}')
    print(f'recompute validation vs dump pred_pe (128nm): total ratio '
          f'median {np.median(val_tot):.4f} [{np.min(val_tot):.4f},{np.max(val_tot):.4f}], '
          f'worst per-channel rel dev {np.max(val_ch):.3f}')

    qtol = {}
    for name in libs:
        ks = np.array([ks_dis(meas, pr[name], live) for _, meas, pr, live in pairs])
        r = np.array([meas[live].sum() / pr[name][live].sum()
                      for _, meas, pr, live in pairs])
        qtol[name] = np.median(r)
        print(f'\n--- {name} ---')
        print(f'  KS: median {np.median(ks):.3f}  [16,84]% = '
              f'[{np.percentile(ks, 16):.3f}, {np.percentile(ks, 84):.3f}]')
        print(f'  closure sum(meas)/sum(pred): median {np.median(r):.3f}  '
              f'[16,84]% = [{np.percentile(r, 16):.3f}, {np.percentile(r, 84):.3f}]  '
              f'log10 width {(np.log10(np.percentile(r, 84) / np.percentile(r, 16))) / 2:.3f}')
        # per-channel medians -> group spreads
        per_ch = [[] for _ in range(len(VUV_EFF))]
        for _, meas, pr, live in pairs:
            pred = pr[name]
            for j in np.where(live)[0]:
                if pred[j] > 1e-3 and meas[j] > 0.5:
                    per_ch[j].append(meas[j] / pred[j])
        groups = {}
        for j, v in enumerate(per_ch):
            if len(v) >= 5:
                groups.setdefault(CH_GROUP[j], []).append(np.median(v))
        for g_, ms in sorted(groups.items()):
            ms = np.array(ms)
            print(f'  {g_:>10}: group median {np.median(ms):7.3f}   '
                  f'in-group spread max/min {ms.max() / ms.min():6.1f}, '
                  f'sigma(log10) {np.std(np.log10(ms)):.3f}  (n_ch={len(ms)})')

    # --- Xe hypothesis: Ar-blind channel closure under the 175 nm library ---
    # If the light were Xe 175 nm, meas/pred175 on the Ar-blind channels should
    # sit at the same level as on their live same-type peers; if pure Ar, the
    # observed signal there is reflected/late light and lands far BELOW.
    print('\nAr-blind channel closure under 175nm (meas/pred175, type-nominal eff):')
    peer_of = {13: (12, 18, 19), 29: (25, 26, 30, 31), 32: (30, 31, 33),
               39: (35, 36, 37, 38)}
    for ch, peers in peer_of.items():
        rb, rp = [], []
        for _, meas, pr, live in pairs:
            pred = pr['175ext']
            if pred[ch] > 1e-3 and meas[ch] > 0.5:
                rb.append(meas[ch] / pred[ch])
            for pj in peers:
                if live[pj] and pred[pj] > 1e-3 and meas[pj] > 0.5:
                    rp.append(meas[pj] / pred[pj])
        if rb and rp:
            print(f'  ch{ch:2d}: median {np.median(rb):7.3f} (n={len(rb)})   '
                  f'live peers {peers}: median {np.median(rp):7.3f} (n={len(rp)})   '
                  f'ratio blind/peers {np.median(rb) / np.median(rp):.3f}')
        else:
            print(f'  ch{ch:2d}: insufficient samples (blind n={len(rb)}, peers n={len(rp)})')

    ks128 = np.array([ks_dis(meas, pr['128nm'], live) for _, meas, pr, live in pairs])
    ks175 = np.array([ks_dis(meas, pr['175nm'], live) for _, meas, pr, live in pairs])
    d_ = ks175 - ks128
    print(f'\npaired KS (175 - 128): median {np.median(d_):+.3f}; '
          f'175 better in {(d_ < 0).sum()}/{len(d_)} pairs, '
          f'128 better in {(d_ > 0).sum()}/{len(d_)}')

    # crude mixture indicator (partial doping): shape-only KS of
    # alpha*pred128 + (1-alpha)*pred175 -- NOT a physical fraction (the two
    # libraries carry the wrong-for-each-other efficiencies), just "does a
    # mix beat both endpoints?"
    print('mixture scan alpha*128 + (1-alpha)*175, median KS:')
    for a in np.linspace(0, 1, 11):
        ksm = np.median([ks_dis(meas, a * pr['128nm'] + (1 - a) * pr['175nm'], live)
                         for _, meas, pr, live in pairs])
        print(f'  alpha={a:.1f}: {ksm:.3f}')
    print(f'implied QtoL if adopted: 128nm {qtol["128nm"]:.3f}, 175nm {qtol["175nm"]:.3f}')


if __name__ == '__main__':
    main()
