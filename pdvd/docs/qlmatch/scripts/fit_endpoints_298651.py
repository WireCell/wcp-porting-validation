#!/usr/bin/env python3
"""Error-function edge fits to pin the start/end drift-time ticks of the
evt 298651 A<->C crosser, as a quantitative cross-check of the by-eye Magnify
reads.  Companion to crosser_drifttime_298651.py / 06_pdvd-drift-crosser-298651.md.

Physics of the edge model:
  A track end deposits a step of collection charge; what we observe on the W
  (gauss) trace is that step convolved with the smearing kernels, i.e. an
  error function  S(t) = base + A/2 * (1 +/- erf((t-t0)/(sqrt(2) sigma))).
  The erf midpoint t0 is the (unbiased) arrival tick of the end charge.
    * ANODE end  (start): drift ~ 0, only the SP software (Gaus_wide) filter
      smears it -> sigma = sigma_sw.
    * CATHODE end (end):  full-drift charge, smeared by the software filter AND
      longitudinal diffusion -> sigma = sqrt(sigma_sw^2 + sigma_diff^2).

  sigma_sw: the gauss output uses HfFilter 'Gaus_wide' = exp(-0.5 (f/sigma_f)^2),
  sigma_f = 0.12 MHz (sp-filters.jsonnet).  Its time-domain kernel has
  sigma_sw = 1/(2 pi sigma_f) ~ 2.65 ticks (0.5 us).
  sigma_diff: longitudinal-diffusion width sigma_x = sqrt(D * t_drift),
  D = 6.5782 cm^2/s (user), converted to time with v = 0.147774 cm/us.

Repro:
  cd pdvd/docs/qlmatch
  python3 scripts/fit_endpoints_298651.py   # prints the table; writes endpoint_fits_298651.png
"""
import numpy as np, uproot
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os

WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039252_6"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pics")  # scripts/ -> qlmatch/pics
TICK_US = 0.5
V = 0.147774              # cm/us, for diffusion cm->tick conversion
D_DIFF = 6.5782           # cm^2/s longitudinal diffusion coefficient (user)
DFULL = 338.55            # cm, cathode -> W collection plane

# --- software-filter smearing sigma from Gaus_wide (numeric IFFT, as C++ builds it) ---
def sw_sigma_ticks(N=10000, maxfreq=1.0, sigma=0.12, power=2):
    i = np.arange(N); freq = i/N*2*maxfreq; freq = np.where(freq > maxfreq, freq-2*maxfreq, freq)
    filt = np.exp(-0.5*(np.abs(freq)/sigma)**power); filt[0] = 0.0
    kern = np.fft.fftshift(np.real(np.fft.ifft(filt)))
    t = np.arange(N)-N//2; core = np.abs(t) < 20
    k = np.clip(kern[core], 0, None); tk = t[core]
    return np.sqrt(np.sum(tk**2*k)/np.sum(k))
SW = sw_sigma_ticks()

def sigma_diff_ticks(drift_us):
    sd_cm = np.sqrt(D_DIFF * drift_us*1e-6)
    return sd_cm / V / TICK_US

def load_row(anode, chan):
    f = uproot.open("%s/magnify-run039252-evt6-anode%d-dnnroi.root" % (WORK, anode))
    h = f["hw_gauss%d" % anode]; cen = h.axis(0).centers(); vals = h.values()
    j = int(np.argmin(np.abs(cen-chan)))
    return vals[j], int(round(cen[j]))

def rising(t, base, A, t0, s):  return base + 0.5*A*(1 + erf((t-t0)/(np.sqrt(2)*s)))
def falling(t, base, A, t0, s): return base + 0.5*A*(1 - erf((t-t0)/(np.sqrt(2)*s)))

# (label, half, anode, chan, kind, window, sigma_expected)
# windows chosen to bracket the physical edge (rise / final decline to baseline).
CASES = [
    ("bot start", "bot", 1, 2674, "rising",  (566, 588), SW),
    ("bot end",   "bot", 1, 2667, "falling", (5148, 5169), None),   # sigma_exp filled below
    ("top start", "top", 5, 8874, "rising",  (580, 597), SW),
    ("top end",   "top", 5, 8820, "falling", (5146, 5167), None),
]
# fill end sigmas with software (+) diffusion at the eyeball full-drift time
for c in CASES:
    if c[0] == "bot end": SE_B = np.hypot(SW, sigma_diff_ticks((5166-576)*TICK_US))
    if c[0] == "top end": SE_T = np.hypot(SW, sigma_diff_ticks((5164-590)*TICK_US))
SIG_EXP = {"bot start": SW, "bot end": SE_B, "top start": SW, "top end": SE_T}

print("sigma_sw (Gaus_wide) = %.2f ticks   sigma_diff(full drift) ~ %.2f ticks   "
      "sigma_end = sqrt(sw^2+diff^2) ~ %.2f ticks" % (SW, sigma_diff_ticks(2295), SE_B))

res = {}
fig, axs = plt.subplots(2, 2, figsize=(13, 8)); axf = axs.ravel()
for k, (lab, half, anode, chan, kind, (lo, hi), _s) in enumerate(CASES):
    sig_exp = SIG_EXP[lab]
    row, cch = load_row(anode, chan)
    t = np.arange(lo, hi).astype(float); y = row[lo:hi].astype(float)
    model = rising if kind == "rising" else falling
    plateau = y.max()
    base0 = np.median(y[:4]) if kind == "rising" else np.median(y[-4:])
    p0 = [base0, max(plateau-base0, 500), 0.5*(lo+hi), 3.0]
    pf, _ = curve_fit(model, t, y, p0=p0, maxfev=40000)
    # fixed-sigma (software for start, software+diffusion for end)
    m_fix = (lambda t, b, A, t0: model(t, b, A, t0, sig_exp))
    pfx, _ = curve_fit(m_fix, t, y, p0=p0[:3], maxfev=40000)
    res[lab] = dict(t0_free=pf[2], sig_free=abs(pf[3]), t0_fix=pfx[2], sig_exp=sig_exp, cch=cch, anode=anode)
    print("%-9s a%d ch%-4d %-7s [%d,%d]: t0(freeσ)=%.1f σ=%.2f | t0(σ=%.2f fixed)=%.1f"
          % (lab, anode, cch, kind, lo, hi, pf[2], abs(pf[3]), sig_exp, pfx[2]))
    # plot
    tt = np.linspace(lo, hi, 400)
    axf[k].plot(t, y, "ko", ms=3, label="gauss")
    axf[k].plot(tt, model(tt, *pf), "b-", lw=1.5, label="erf fit (σ=%.1f)" % abs(pf[3]))
    axf[k].plot(tt, m_fix(tt, *pfx), "r--", lw=1.2, label="erf fit (σ=%.1f fixed)" % sig_exp)
    axf[k].axvline(pf[2], color="b", ls=":", lw=1)
    axf[k].set_title("%s  anode%d ch%d  t0=%.1f" % (lab, anode, cch, pf[2]))
    axf[k].set_xlabel("tick (0.5 us)"); axf[k].legend(fontsize=7)
fig.suptitle("evt 298651 A<->C crosser: erf edge fits (W gauss decon)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "endpoint_fits_298651.png"), dpi=110); plt.close(fig)

print("\n==== velocity from erf-midpoint (free-sigma) fits, DFULL=%.2f cm ====" % DFULL)
for half, s, e in (("BOTTOM", "bot start", "bot end"), ("TOP", "top start", "top end")):
    ts, te = res[s]["t0_free"], res[e]["t0_free"]; dt = (te-ts)*TICK_US
    print("  %-6s start=%.1f end=%.1f  dt=%.1f us  v=%.5f cm/us" % (half, ts, te, dt, DFULL/dt))
print("==== velocity from fixed-sigma fits ====")
vs = []
for half, s, e in (("BOTTOM", "bot start", "bot end"), ("TOP", "top start", "top end")):
    ts, te = res[s]["t0_fix"], res[e]["t0_fix"]; dt = (te-ts)*TICK_US; v = DFULL/dt; vs.append(v)
    print("  %-6s start=%.1f end=%.1f  dt=%.1f us  v=%.5f cm/us" % (half, ts, te, dt, v))
print("  mean (fixed-sigma) v = %.5f cm/us" % np.mean(vs))
print("\nwrote endpoint_fits_298651.png")
