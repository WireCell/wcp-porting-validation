#!/usr/bin/env python3
"""Error-function edge fits for the evt 298609 track-A anode<->cathode<->anode
crosser (Bee ccprod clusters 37 = bottom half, 79 = top half; flash #117).
Same edge model as fit_endpoints_298651.py (see that header): erf midpoint =
arrival tick; sigma = software Gaus_wide filter for the anode (start) edges,
software (+) longitudinal diffusion for the cathode (end) edges.

Channels picked from the aca_crossers_298609.py band tips (W gauss > 400):
  bot start  anode0 ch2202  onset ~2016  (bottom-anode exit tip)
  bot end    anode2 ch5050  end   ~6618  (cathode arrival, bottom side)
  top start  anode6 ch11177 onset ~2018  (top-anode entry tip; imaging corridor
                                          only starts ~2350 -- induction-gap
                                          truncation, W charge is there)
  top end    anode6 ch11198 end   ~6583  (cathode arrival, top side)

Repro:
  cd pdvd/docs/qlmatch
  python3 scripts/fit_endpoints_298609.py   # prints table; writes pics/endpoint_fits_298609.png
"""
import numpy as np, uproot
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os

MAG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039252_3_magnify"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pics")
TICK_US = 0.5
V = 0.148073              # cm/us, only for the diffusion cm->tick conversion
D_DIFF = 6.5782           # cm^2/s longitudinal diffusion coefficient
DFULL = 338.55            # cm, cathode surface -> W collection plane

def sw_sigma_ticks(N=10000, maxfreq=1.0, sigma=0.12, power=2):
    i = np.arange(N); freq = i/N*2*maxfreq; freq = np.where(freq > maxfreq, freq-2*maxfreq, freq)
    filt = np.exp(-0.5*(np.abs(freq)/sigma)**power); filt[0] = 0.0
    kern = np.fft.fftshift(np.real(np.fft.ifft(filt)))
    t = np.arange(N)-N//2; core = np.abs(t) < 20
    k = np.clip(kern[core], 0, None); tk = t[core]
    return np.sqrt(np.sum(tk**2*k)/np.sum(k))
SW = sw_sigma_ticks()

def sigma_diff_ticks(drift_us):
    return np.sqrt(D_DIFF * drift_us*1e-6) / V / TICK_US

def load_row(anode, chan):
    f = uproot.open("%s/magnify-run039252-evt3-anode%d-dnnroi_magnify.root" % (MAG, anode))
    h = f["hw_gauss%d" % anode]; cen = h.axis(0).centers(); vals = h.values()
    j = int(np.argmin(np.abs(cen-chan)))
    return vals[j], int(round(cen[j]))

def rising(t, base, A, t0, s):  return base + 0.5*A*(1 + erf((t-t0)/(np.sqrt(2)*s)))
def falling(t, base, A, t0, s): return base + 0.5*A*(1 - erf((t-t0)/(np.sqrt(2)*s)))

SE = np.hypot(SW, sigma_diff_ticks(2290))
CASES = [
    ("bot start", 0,  2202, "rising",  (2006, 2030), SW),
    ("bot end",   2,  5050, "falling", (6605, 6632), SE),
    ("top start", 6, 11177, "rising",  (2008, 2032), SW),
    ("top end",   6, 11198, "falling", (6572, 6596), SE),
]
print("sigma_sw = %.2f ticks; sigma_end (sw+diff, full drift) = %.2f ticks" % (SW, SE))

res = {}
fig, axs = plt.subplots(2, 2, figsize=(13, 8)); axf = axs.ravel()
for k, (lab, anode, chan, kind, (lo, hi), sig_exp) in enumerate(CASES):
    row, cch = load_row(anode, chan)
    t = np.arange(lo, hi).astype(float); y = row[lo:hi].astype(float)
    model = rising if kind == "rising" else falling
    plateau = y.max()
    base0 = np.median(y[:4]) if kind == "rising" else np.median(y[-4:])
    p0 = [base0, max(plateau-base0, 500), 0.5*(lo+hi), 3.0]
    pf, _ = curve_fit(model, t, y, p0=p0, maxfev=40000)
    m_fix = (lambda t, b, A, t0: model(t, b, A, t0, sig_exp))
    pfx, _ = curve_fit(m_fix, t, y, p0=p0[:3], maxfev=40000)
    res[lab] = dict(t0_free=pf[2], sig_free=abs(pf[3]), t0_fix=pfx[2], sig_exp=sig_exp)
    print("%-9s a%d ch%-5d %-7s [%d,%d]: t0(freeσ)=%.1f σ=%.2f | t0(σ=%.2f fixed)=%.1f"
          % (lab, anode, cch, kind, lo, hi, pf[2], abs(pf[3]), sig_exp, pfx[2]))
    tt = np.linspace(lo, hi, 400)
    axf[k].plot(t, y, "ko", ms=3, label="gauss")
    axf[k].plot(tt, model(tt, *pf), "b-", lw=1.5, label="erf fit (σ=%.1f)" % abs(pf[3]))
    axf[k].plot(tt, m_fix(tt, *pfx), "r--", lw=1.2, label="erf fit (σ=%.1f fixed)" % sig_exp)
    axf[k].axvline(pf[2], color="b", ls=":", lw=1)
    axf[k].set_title("%s  anode%d ch%d  t0=%.1f" % (lab, anode, cch, pf[2]))
    axf[k].set_xlabel("tick (0.5 us)"); axf[k].legend(fontsize=7)
fig.suptitle("evt 298609 track A (clusters 37+79) A<->C<->A crosser: erf edge fits (W gauss)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "endpoint_fits_298609.png"), dpi=110); plt.close(fig)

for tag in ("fix", "free"):
    print("==== velocity, %s-sigma fits, DFULL=%.2f cm ====" % (tag, DFULL))
    vs = []
    for half, s, e in (("BOTTOM", "bot start", "bot end"), ("TOP", "top start", "top end")):
        ts, te = res[s]["t0_"+tag], res[e]["t0_"+tag]
        dt = (te-ts)*TICK_US; v = DFULL/dt; vs.append(v)
        print("  %-6s start=%.1f end=%.1f  dt=%.1f us  v=%.5f cm/us" % (half, ts, te, dt, v))
    print("  mean v = %.5f cm/us" % np.mean(vs))
print("wrote endpoint_fits_298609.png")

# ---- track B (clusters 83+50, flash #163): anode (start) edges only --------
# The cathode ends are truncated at the readout edge (tick 10000), so track B
# yields no velocity -- only the anode-touch T0 used by the light-side check.
print("\n==== track B anode edges (cathode ends readout-truncated) ====")
for lab, anode, chan, (lo, hi) in [("B top start", 4, 8371, (5770, 5800)),
                                   ("B bot start", 3, 5605, (5772, 5802))]:
    row, cch = load_row(anode, chan)
    t = np.arange(lo, hi).astype(float); y = row[lo:hi].astype(float)
    p0 = [np.median(y[:4]), max(y.max()-np.median(y[:4]), 500), 0.5*(lo+hi), 3.0]
    pf, _ = curve_fit(rising, t, y, p0=p0, maxfev=40000)
    m_fix = (lambda t, b, A, t0: rising(t, b, A, t0, SW))
    pfx, _ = curve_fit(m_fix, t, y, p0=p0[:3], maxfev=40000)
    print("%-12s a%d ch%-5d [%d,%d]: t0(freeσ)=%.1f σ=%.2f | t0(σ=%.2f fixed)=%.1f"
          % (lab, anode, cch, lo, hi, pf[2], abs(pf[3]), SW, pfx[2]))
