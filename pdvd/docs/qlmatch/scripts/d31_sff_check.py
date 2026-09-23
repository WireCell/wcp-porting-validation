#!/usr/bin/env python3
"""doc qlmatch/31 sec 2: per-pulse prompt-fraction spread on the SINGLE-start v2 channel parameters, default vs tight
tolerances -- is the single-start sigma_ff 0.034 an early stop, or a property of those parameters?"""
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '.')
import saturation_recovery_study as S
import saturation_tot_study as T
V2 = S._find_templates().replace("pdvd-spe-templates.json", "pdvd-spe-templates-v2.json"); S._find_templates = lambda: V2
T.CACHE = "/home/xqian/tmp/sat_tot_v2"
chd = S.load_channels(); z = T.load_pulses()
seg, pk, b, C, F = z["seg"], z["pk"], z["base"], z["chan"], z["fidx"]
okw = (pk + T.OFF.max() < seg.shape[1]) & (pk + T.OFF.min() >= 0)
sp = np.load(sys.argv[1] if len(sys.argv) > 1 else "/home/xqian/tmp/p31/shape_v2_singlestart.npz")
par = {int(c): p for c, p in zip(sp["chan"], sp["par"])}
for name, kw in (("default tolerances", {}), ("tight tolerances", dict(xtol=1e-12, ftol=1e-12, x_scale=[0.1]))):
    dff = []
    for c in sorted(par):
        ff0, bi, ti, ts, sg = par[c]
        for i in np.nonzero((C == c) & (F == 0) & okw)[0]:
            w = (seg[i, pk[i] + T.OFF] - b[i]) / (seg[i, pk[i]] - b[i])
            r = least_squares(lambda p: T.shape_at(T.model(chd[c], p[0], bi, ti, ts, sg)) - w, [ff0], bounds=([0], [1]), **kw)
            dff.append(r.x[0] - ff0)
    q = np.percentile(dff, [16, 84])
    print(f"single-start v2 parameters, {name}: n {len(dff)}, [16,84] [{q[0]:+.3f}, {q[1]:+.3f}] -> sigma_ff {0.5 * (q[1] - q[0]):.3f}")
