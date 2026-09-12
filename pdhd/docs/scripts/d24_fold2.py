#!/usr/bin/env python3
"""Disentangle the time-slice fold from the collection-wire fold.

A track that runs along the collection-wire direction (y on PDHD) is slow in
BOTH z (=> few W wires crossed) and x (=> few time slices crossed), so the
spc>=6 strata of pw and pt largely overlap and either fold can be the other
leaking through.  Three things separate them:

  1. the overlap itself, counted;
  2. a CONDITIONAL test -- subtract the best-fit first harmonic in one phase,
     then re-measure the other;
  3. the folded profiles, drawn.

Null throughout: per-item circular shift of the residual (keeps each item's
autocorrelation, breaks only its alignment with the phase).
"""
import json, sys
import os
import numpy as np

FIGS = os.environ.get("D24_FIGS", "../figs")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from d24_fold import collect

NPERM = 1000
rng = np.random.default_rng(20260912)

def pooled_h1(items, key, shifts=None, sub=None):
    """First harmonic of r folded on frac(item[key]); `sub` names a phase whose
    best-fit first harmonic is removed from r first."""
    num = 0.0 + 0.0j; n = 0
    for i, it in enumerate(items):
        r = it["r"]
        if shifts is not None:
            r = np.roll(r, shifts[i])
        if sub is not None:
            ps = np.mod(it[sub], 1.0)
            c = 2.0*np.mean(r*np.exp(2j*np.pi*ps))          # per-item amplitude
            r = r - (c*np.exp(-2j*np.pi*ps)).real            # remove it
        ph = np.mod(it[key], 1.0)
        num += np.sum(r*np.exp(2j*np.pi*ph)); n += r.size
    return 2.0*np.abs(num)/n

def test(items, key, sub=None, label=""):
    a = pooled_h1(items, key, sub=sub)
    null = np.array([pooled_h1(items, key,
                               [rng.integers(1, it["r"].size) for it in items], sub=sub)
                     for _ in range(NPERM)])
    p = ((null >= a).sum()+1)/(NPERM+1)
    z = (a - null.mean())/(null.std()+1e-12)
    return "%-34s items=%3d  A1=%.4f  null_mean=%.4f null95=%.4f  z=%+.2f  p=%.4f" % (
        label or key, len(items), a, null.mean(), np.percentile(null,95), z, p), a, p

def profile(items, key, nb=12):
    e = np.linspace(0,1,nb+1); m = np.zeros(nb); s = np.zeros(nb); c = np.zeros(nb)
    ph = np.concatenate([np.mod(it[key],1.0) for it in items])
    r  = np.concatenate([it["r"] for it in items])
    idx = np.clip((ph*nb).astype(int), 0, nb-1)
    for b in range(nb):
        g = idx==b; c[b]=g.sum()
        if g.sum(): m[b]=r[g].mean(); s[b]=r[g].std()/np.sqrt(g.sum())
    return 0.5*(e[:-1]+e[1:]), m, s, c

PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
items = collect(PREP)
out = []
W6 = [it for it in items if it["pw_spc"] >= 6]
T6 = [it for it in items if it["pt_spc"] >= 6]
kw = set(it["key"] for it in W6); kt = set(it["key"] for it in T6)
out.append("pw spc>=6: %d items   pt spc>=6: %d items   overlap: %d"
           % (len(W6), len(T6), len(kw & kt)))
out.append("pw spc>=6 AND pt spc<6: %d    pt spc>=6 AND pw spc<6: %d"
           % (len(kw-kt), len(kt-kw)))

WonlyW = [it for it in items if it["pw_spc"] >= 6 and it["pt_spc"] < 6]
TonlyT = [it for it in items if it["pt_spc"] >= 6 and it["pw_spc"] < 6]
BOTH   = [it for it in items if it["pt_spc"] >= 6 and it["pw_spc"] >= 6]

out.append("")
out.append("--- the two folds, on the whole population ---")
for k, lab in (("pt","pt  (time slice)"), ("pw","pw  (W wire)")):
    out.append(test(items, k, label=lab)[0])
out.append("")
out.append("--- disjoint strata: each clock slow, the other NOT ---")
if len(WonlyW) >= 5: out.append(test(WonlyW, "pw", label="pw | W slow, t fast")[0])
if len(TonlyT) >= 5: out.append(test(TonlyT, "pt", label="pt | t slow, W fast")[0])
if len(TonlyT) >= 5: out.append(test(TonlyT, "pw", label="pw | t slow, W fast (control)")[0])
if len(WonlyW) >= 5: out.append(test(WonlyW, "pt", label="pt | W slow, t fast (control)")[0])
out.append("")
out.append("--- conditional: remove the other harmonic first (both-slow stratum, n=%d) ---" % len(BOTH))
if len(BOTH) >= 5:
    out.append(test(BOTH, "pt", label="pt raw")[0])
    out.append(test(BOTH, "pw", label="pw raw")[0])
    out.append(test(BOTH, "pt", sub="pw", label="pt | pw harmonic removed")[0])
    out.append(test(BOTH, "pw", sub="pt", label="pw | pt harmonic removed")[0])
txt = "\n".join(out)
print(txt); open(os.path.join(FIGS, "24_fold2.txt"), "w").write(txt+"\n")

fig, ax = plt.subplots(1, 3, figsize=(14,4), sharey=True)
for a, (sel, k, ttl) in zip(ax, [(items,"pt","pt  all %d items"%len(items)),
                                 (T6,"pt","pt  slow-in-x stratum (%d)"%len(T6)),
                                 (W6,"pw","pw  slow-in-z stratum (%d)"%len(W6))]):
    x,m,s,c = profile(sel,k)
    a.errorbar(x,m,yerr=s,fmt="o-",ms=4)
    a.axhline(0,color="grey",lw=0.6); a.set_title(ttl,fontsize=10)
    a.set_xlabel("phase within one cell"); a.grid(alpha=0.3)
ax[0].set_ylabel("mean detrended dQ/dx residual [sigma]")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "24_fold_profiles.png"),dpi=110)
print("wrote", os.path.join(FIGS, "24_fold_profiles.png"))
