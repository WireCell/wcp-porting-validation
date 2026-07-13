#!/usr/bin/env python3
"""Estimate the Argon (128 nm) QtoL WITHOUT the (removed) jjo beam-flash table.

Context (2026-07-13): reverting the Xe/175 nm verdict (0adb15fa) back to
Ar/128 nm needs the Ar QtoL under the CURRENT geometry.  The canonical route
(fit_qtol_gold.py / ablib_gold.py on the 80 beam-trigger gold pairs) is
blocked: data/jjo_triglight_offsets.txt has been removed and the raw files
changed schema (triglight tree, no charge_bde_us), so t_expect for the beam
flash can no longer be rebuilt without a timing-reconstruction subproject.

Two independent handles instead:

(B) ANCHORED scaling.  The pre-Y-truncation gold-pair fit gave 175->0.082 and
    128->0.11 on the SAME 80 pairs, so 0.11/0.082 already bakes in the correct
    beam/cathode population AND the meas channel-set difference.  Only the
    565ccd62 Y-truncation changed since, moving Xe 0.082->0.070.  If truncation
    scales both libraries ~equally,
        QtoL_128_now ~= 0.11 * (0.070/0.082) = 0.0939.

(A) CORROBORATION, library pred-ratio.  For a given bundle pred_175/pred_128 is
    match- and T0-independent (same points, same flash x-offset -- only the
    library + Ar/Xe eff + static mask differ).  Per bundle
        R = S_175(live_Xe, eff_Xe) / S_128(live_Ar, eff_Ar)   [QtoL=1]
    and QtoL_128 = QtoL_175 * median(R) * (meas_Ar/meas_Xe).  The ratio is
    strongly POSITION-dependent, so it is only representative on the beam/
    cathode population: it climbs toward 0.094 as the bundle set is tightened
    (all auto_selected 0.94 -> flashes>500PE 1.20 -> brightest-flash-per-event
    1.29 -> 0.070*1.29 = 0.090).  meas_Ar/meas_Xe ~= 1 (PE on the 3 Xe-extra
    channels 13/29/39 is <0.4% -- verified here).

Verdict: QtoL_128 ~= 0.094 (estimate).  A full 128 nm gold-pair refit is a
flagged follow-up once the jjo trigger table is restored.

Usage (dumps must be any current-geometry calib dumps; library is recomputed):
    python3 estimate_ar_qtol.py 'work/0392*_*/calib-evt*.json' ...

The regex in fit_qtol_gold.py/ablib_gold.py needs <run>_<idx>/calib-evt; for
tagged dirs (e.g. _ctoff) symlink <run>_<idx> -> <run>_<idx>_ctoff first.
"""
import glob
import json
import os
import sys

import numpy as np

# ablib_gold provides GridLib (trilinear lookup) and predict (vis loop).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablib_gold as ab

PHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'photlib')
lib128 = ab.GridLib(os.path.join(PHOT, 'pdvd-photlib-vis-v5-128nm.json'))
lib175 = ab.GridLib(os.path.join(PHOT, 'pdvd-photlib-vis-v5-175nm.json'))

M_XE = {24, 27, 28, 32, 34}                       # Xe static mask
M_AR = {13, 24, 27, 28, 29, 32, 34, 39}           # Ar static mask
EFF_AR = np.array([0.03]*12 + [0.03, 0.0] + [0.12, 0.036, 0.12, 0.12] +
                  [0.03, 0.03] + [0.12, 0.036, 0.12, 0.12] + [0.036]*5 + [0.0] +
                  [0.036, 0.036, 0.0, 0.036, 0.036, 0.036] +
                  [0.036, 0.036, 0.036, 0.0])
EFF_XE = EFF_AR.copy()
EFF_XE[13] = 0.03; EFF_XE[29] = 0.036; EFF_XE[39] = 0.036

QTOL_XE = 0.070   # current-geometry Xe value (the number we scale from)


def S(d, b, lib, eff, mask, live):
    ab.VUV_EFF[:] = eff
    pr = ab.predict(d, b, lib, live)
    qp = d['quality_params'].get('QtoL', 1.0) or 1.0
    pr = (pr / qp).copy()
    for m in mask:
        pr[m] = 0.0
    return pr


def main():
    paths = []
    for pat in sys.argv[1:]:
        paths += glob.glob(pat)
    def ratio(d, b, live):
        s175 = S(d, b, lib175, EFF_XE, M_XE, live)
        s128 = S(d, b, lib128, EFF_AR, M_AR, live)
        if s175.sum() <= 0 or s128.sum() <= 0:
            return None
        return s175.sum() / s128.sum()

    R_all, R_bright, R_top, mfrac = [], [], [], []
    for p in sorted(paths):
        d = json.load(open(p))
        live = np.array([o['active'] and not o['auto_masked'] for o in d['opdets']])
        fl = {f['gid']: f for f in d['flashes']}
        if not fl:
            continue
        # gold-cluster proxy: the beam flash = brightest flash in the event; the
        # gold cluster = its max-total_pred_light bundle (same rule the gold-pair
        # machinery uses).  This is the population the calibration is anchored on.
        gtop = max(fl.values(), key=lambda f: f['total_PE'])['gid']
        for b in d['bundles']:
            if b.get('total_pred_light', 0) <= 0:
                continue
            r = ratio(d, b, live)
            if r is None:
                continue
            R_all.append(r)
            f = fl.get(b['flash_gid'])
            if f and f['total_PE'] > 500:
                R_bright.append(r)
            if f:
                pe = np.array(f['pe'])
                tot = pe[[i for i in range(40) if i not in M_AR and live[i]]].sum()
                extra = pe[[i for i in (13, 29, 39) if live[i]]].sum()
                if tot > 0:
                    mfrac.append(extra / (tot + extra))
        tops = [b for b in d['bundles']
                if b['flash_gid'] == gtop and b.get('total_pred_light', 0) > 0]
        if tops:
            b = max(tops, key=lambda x: x['total_pred_light'])
            r = ratio(d, b, live)
            if r is not None:
                R_top.append(r)
    R_all, R_bright, R_top = map(np.array, (R_all, R_bright, R_top))
    mfrac = np.array(mfrac)
    print(f'bundles: all={len(R_all)} bright(>500PE)={len(R_bright)} '
          f'brightest-per-evt={len(R_top)}')
    print(f'median S175/S128:  all {np.median(R_all):.3f}  '
          f'bright {np.median(R_bright):.3f}  top {np.median(R_top):.3f}')
    print(f'PE fraction on Xe-extra ch 13/29/39: median {np.median(mfrac):.4f}')
    print(f'\n(A) corroboration: QtoL_Ar = {QTOL_XE} * median(S175/S128)')
    print(f'    all-bundles {QTOL_XE*np.median(R_all):.4f}  (unanchored, cosmic-dominated)')
    print(f'    bright      {QTOL_XE*np.median(R_bright):.4f}')
    print(f'    top-per-evt {QTOL_XE*np.median(R_top):.4f}  (most beam/cathode-like)')
    print(f'(B) anchored: 0.11 * (0.070/0.082) = {0.11*0.070/0.082:.4f}  <-- ADOPTED ~0.094')


if __name__ == '__main__':
    main()
