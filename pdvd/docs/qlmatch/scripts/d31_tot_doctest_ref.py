#!/usr/bin/env python3
"""doc qlmatch/31: reference numbers pinned by toolkit flash/test/doctest_opdecon_tot.cxx (run from scripts/)."""
import sys, numpy as np
sys.path.insert(0, '.')
import saturation_recovery_study as S
import saturation_tot_study as T
np.set_printoptions(precision=17)
t = np.arange(120, dtype=float)
wave = np.where(t < 10, 100*np.exp((t-10)/1.5), 100*np.exp(-(t-10)/20.0)).astype(np.float32)
ch = S.Channel(wave.astype(np.float64))
print("tau_fall %.17g" % ch.tau_fall)
par = [0.25, 0.4, 6.0, 90.0, 0.8]
y = T.model(ch, *par)
shp = T.Shape(ch, par)
for k in (0, 5, 10, 13, 40, 200, 700, 1599): print("y[%d] %.17g" % (k, y[k]))
print("ip", shp.ip)
for q in (0, 1000, 2000, 2998): print("lam[%d] %.17g u %d w %.17g" % (q, shp.lam[q], shp.u[q], shp.w[q]))
for tot in (5.5, 50.3, 300.7): print("level_for(%g) %.17g" % (tot, shp.level_for(tot)))
# fill reference: pedestal 1000, rail 16383, run [100, 100+tot)
ped, rail = 1000.0, 16383
for d in (2.0, 6.7, 20.0):
    R = rail + 0.5 - ped
    A = d*(rail - ped)
    n = 2000; w = np.full(n, ped); pos = 100
    w[pos:pos+1600] += A*shp.s[:1600]
    true_area = (w - ped).sum()
    clip = np.minimum(w, rail)
    idx = np.flatnonzero(clip >= rail); i, j = idx[0], idx[-1]+1
    tot = j - i
    lam = shp.level_for(tot); Af = R/lam
    k0 = shp.u[min(len(shp.u)-1, int(np.searchsorted(shp.lam, lam)))]
    tt = np.arange(i, j); kk = k0 + (tt - i)
    fill = Af*np.where(kk < len(shp.s), shp.s[np.clip(kk, 0, len(shp.s)-1)], 0.0)
    rep = clip.copy(); rep[i:j] = np.maximum(rep[i:j], ped + fill)
    print("d %g i %d j %d lam %.17g k0 %d filled_sum %.17g area_ratio %.6f" % (d, i, j, lam, k0, (rep[i:j]-ped).sum(), (rep-ped).sum()/true_area))
