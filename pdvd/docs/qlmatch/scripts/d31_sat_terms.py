#!/usr/bin/env python3
"""doc qlmatch/31 -- doc 11 sec 6.2-6.3 (analyze_sat_terms.py) pooled over many calib dumps, plus a cathode-only split
and inflate 0.25 / 0.35 for the pre-registered chi2_sat_inflate rule.  Same term, same per-opdet error replica.

    python3 d31_sat_terms.py <label> calib-evt*.json ...
"""
import json
import sys

import numpy as np

CATH = set(range(4, 12))


def perr_on_pred(qp, pred):
    lf, lk = qp.get('pe_err_lowpe_frac', 2.0), qp.get('pe_err_lowpe_knee', 10.0)
    rel = qp['pe_err_frac'] + (lf - qp['pe_err_frac']) * np.exp(-pred / lk)
    return np.sqrt((rel * pred) ** 2 + qp['pe_err_floor'] ** 2)


def main():
    label, paths = sys.argv[1], sys.argv[2:]
    rows = {True: [], False: []}      # (meas, pred, perr, cathode)
    nb = {True: 0, False: 0}
    nflash = nsat = 0
    for p in paths:
        d = json.load(open(p))
        qp = d['quality_params']
        fl = {f['id']: f for f in d['flashes']}
        nflash += len(fl)
        nsat += sum(1 for f in fl.values() if np.sum(f.get('sat', [])) > 0)
        for b in d['bundles']:
            f = fl.get(b['flash_id'])
            if not f or not f.get('sat') or np.sum(f['sat']) == 0:
                continue
            want = bool(b.get('auto_selected'))
            nb[want] += 1
            pe, pred = np.asarray(f['pe'], float), np.asarray(b['pred_pe'], float)
            for j in np.nonzero(np.asarray(f['sat']))[0]:
                if pred[j] <= 0 and pe[j] <= 0:
                    continue
                rows[want].append((pe[j], pred[j], perr_on_pred(qp, pred[j]), j in CATH))
    print(f"# {label}: {len(paths)} dumps, {nflash} flashes, {nsat} with a rail flag")
    for want, name in ((True, "SELECTED matches"), (False, "non-selected candidates")):
        a = np.array(rows[want])
        if not len(a):
            continue
        for sub, m in (("all railed", np.ones(len(a), bool)), ("cathode railed", a[:, 3] > 0)):
            meas, pred, perr = a[m, 0], a[m, 1], a[m, 2]
            chi2 = (pred - meas) ** 2 / (meas + perr ** 2)
            print(f"== {name} / {sub}: {nb[want]} bundles, {m.sum()} terms | meas med {np.median(meas):8.1f} pred med "
                  f"{np.median(pred):8.1f} | meas>pred {100 * np.mean(meas > pred):5.1f}% | meas/pred med "
                  f"{np.median(meas / np.maximum(pred, 1e-9)):8.2f} | chi2 med {np.median(chi2):8.2f} p90 "
                  f"{np.percentile(chi2, 90):9.2f} max {chi2.max():9.2f}")
            if want:
                for infl in (0.0, 0.25, 0.35, 0.5, 1.0, 2.0):
                    c = (pred - meas) ** 2 / (meas + perr ** 2 + (meas * infl) ** 2)
                    print(f"     inflate {infl:4.2f} -> median {np.median(c):8.2f}  p90 {np.percentile(c, 90):8.2f}  "
                          f"max {c.max():9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
