#!/usr/bin/env python3
"""doc qlmatch/31: multi-start refit of the ToT shape for channels whose --shape fit stayed at the start point
[0.3, 0.3, 8, 80, 1] (v2 templates; same data, window, weights and bounds as saturation_tot_study.shape_fit)."""
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
starts = [[0.3, 0.3, 8, 80, 1.0], [0.2, 0.4, 5, 100, 0.8], [0.35, 0.5, 6, 70, 0.7], [0.25, 0.2, 10, 120, 1.2],
          [0.3, 0.45, 7, 90, 0.75], [0.15, 0.3, 4, 110, 0.9]]
for c in [int(x) for x in sys.argv[1:]] or [1051, 1071]:
    sel = np.nonzero((C == c) & (F == 0) & okw)[0]
    W = np.stack([(seg[i, pk[i] + T.OFF] - b[i]) / (seg[i, pk[i]] - b[i]) for i in sel]); med = np.median(W, 0)
    wt = np.where(med > 0.3, 3.0, 1.0)
    for p0 in starts:
        r = least_squares(lambda p: wt * (T.shape_at(T.model(chd[c], *p)) - med), p0,
                          bounds=([0, 0, 2, 20, 0], [1, 1, 40, 400, 20]))
        rms = np.sqrt(np.mean((T.shape_at(T.model(chd[c], *r.x)) - med) ** 2))
        print(c, "start", p0, "->", np.round(r.x, 3), "rms %.4f cost %.5f nfev %d status %d" % (rms, r.cost, r.nfev, r.status))
