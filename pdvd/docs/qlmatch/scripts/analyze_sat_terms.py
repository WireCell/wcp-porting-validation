#!/usr/bin/env python3
"""Railed-channel chi2 terms split by whether the bundle is the SELECTED match.

The pooled distribution is dominated by wrong (flash,cluster) candidate pairs,
where a large chi2 on a bright railed channel is the CORRECT outcome -- that is
the discrimination masking used to throw away.  What sizes chi2_sat_inflate is
the selected-match population.
"""
import json, sys
import numpy as np

d = json.load(open(sys.argv[1]))
qp = d['quality_params']
LOWPE_FRAC = qp.get('pe_err_lowpe_frac', 2.0)
LOWPE_KNEE = qp.get('pe_err_lowpe_knee', 10.0)
fl = {f['id']: f for f in d['flashes']}


def perr_on_pred(pred):
    rel = qp['pe_err_frac'] + (LOWPE_FRAC - qp['pe_err_frac']) * np.exp(-pred / LOWPE_KNEE)
    return np.sqrt((rel * pred) ** 2 + qp['pe_err_floor'] ** 2)


for label, want in (("SELECTED matches", True), ("non-selected candidates", False)):
    rows = []
    nb = 0
    for b in d['bundles']:
        if bool(b.get('auto_selected')) != want:
            continue
        f = fl.get(b['flash_id'])
        if not f or not f.get('sat') or np.sum(f['sat']) == 0:
            continue
        nb += 1
        sat = np.asarray(f['sat'])
        pe = np.asarray(f['pe'], float)
        pred = np.asarray(b['pred_pe'], float)
        for j in np.nonzero(sat)[0]:
            if pred[j] <= 0 and pe[j] <= 0:
                continue
            perr = perr_on_pred(pred[j])
            rows.append((pe[j], pred[j], (pred[j] - pe[j]) ** 2 / (pe[j] + perr ** 2)))
    if not rows:
        print(f"\n=== {label}: no railed terms ===")
        continue
    a = np.array(rows)
    meas, pred, chi2 = a[:, 0], a[:, 1], a[:, 2]
    print(f"\n=== {label}: {nb} bundles touch a railed flash, {len(a)} railed terms ===")
    print(f"  measured PE  median {np.median(meas):9.1f}")
    print(f"  predicted PE median {np.median(pred):9.1f}")
    print(f"  meas > pred (clipped direction is pred<meas): {100*np.mean(meas>pred):5.1f}%")
    print(f"  ratio meas/pred median {np.median(meas/np.maximum(pred,1e-9)):9.2f}")
    print(f"  chi2 term  median {np.median(chi2):9.2f}  p90 {np.percentile(chi2,90):9.2f} "
          f"max {chi2.max():9.2f}")
    # What would chi2_sat_inflate=X buy on the selected population?
    if want:
        print("  chi2 term with denom += (meas*inflate)^2:")
        for infl in (0.0, 0.5, 1.0, 2.0):
            c = (pred - meas) ** 2 / (meas + perr_on_pred(pred) ** 2 + (meas * infl) ** 2)
            print(f"    inflate {infl:3.1f} -> median {np.median(c):8.2f}  "
                  f"p90 {np.percentile(c,90):8.2f}  max {c.max():8.2f}")
