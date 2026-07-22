#!/usr/bin/env python3
"""One-figure reconciliation of the track-A (evt 298609, clusters 37+79,
flash #117) timing conventions, prompted by the side-by-side of the
woodpecker charge-clock 2D hist (charge anode lines on the raw-light peak;
calib flash line ~12 us later) and our aca_light_check figure (erf charge
lines +3.6 us after the raw 20%-rise).  Everything below is the SAME
physical configuration expressed in different estimator conventions:

  raw 20%-rise           3519.65   light onset
  reco flash (raw)       3519.75   accumulator peak time == the pulse peak
  threshold-onset charge 3521.1/3521.8   (woodpecker by-eye ticks 2017/2016;
                         first W sample above threshold, sits ~2sigma BEFORE
                         the erf midpoint)
  erf-midpoint charge    3523.26   true tip-charge arrival on the SP axis;
                         rise+3.6 us ~= the configured SP ctoffset (+4 us,
                         sp.jsonnet ctoffset_b/_t)
  calib/Bee flash time   3533.26   = raw flash + 13.507 us
                         (PDVD_QL_EXTRA_OFFSET_US charge-placement pull,
                         folded into the DISPLAYED time; the "~12 us late
                         flash" of the woodpecker figure is THIS, not a
                         PE-weighted-mean tail effect)

Repro:
    cd pdvd/docs/qlmatch
    python3 scripts/aca_A_conventions.py
    # writes ../../docs/pics/pdvd_light_timing_aca_A_conventions.png
"""
import os
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDVD = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd"
RAW = os.path.join(PDVD, "input_data_light",
    "np02vd_raw_run039252_1176_df-s03-d3_dw_0_20250830T054542_rawwf.root")
OUT = os.path.join(PDVD, "docs", "pics", "pdvd_light_timing_aca_A_conventions.png")
TICK_NS = 16.0
EVENT = 298609
OFF_BOT, OFF_TOP = -2513.808001, -2512.608          # metadata offsets, us
PULL = 13.507

d = uproot.open(RAW)["rawdump/raw_waveform"].arrays(
    ["event", "opchannel", "nsamp", "timestamp", "adc"], library="np")
sel = np.nonzero(d["event"] == EVENT)[0]
starts = {i: int(round(d["timestamp"][i] * 62.5))
          - (64 if d["nsamp"][i] <= 1024 else 0) for i in sel}
t0 = min(starts.values())
cidx = [i for i in sel if d["opchannel"][i] // 1000 == 1]
S = np.zeros(int(d["nsamp"][cidx[0]]))
for i in cidx:
    a = np.asarray(d["adc"][i], dtype=np.float64)
    S[:a.size] += a - np.median(a[:200])
off = starts[cidx[0]] - t0
us = (np.arange(len(S)) + off) * TICK_NS / 1000.0

MARKS = [
    ("raw 20%-rise (light onset)",          3519.65,             "0.4",      ":"),
    ("reco flash #117 (raw) = pulse peak",  3519.753,            "tab:green", "-"),
    ("threshold-onset charge (top, tick 2017)", 1008.5 - OFF_TOP, "tab:orange", "-."),
    ("threshold-onset charge (bot, tick 2016)", 1008.0 - OFF_BOT, "tab:orange", ":"),
    ("erf-mid charge bot/top (rise +3.6 us = ctoffset +4)", 3523.26, "tab:blue", "--"),
    ("calib/Bee time 1019.45 on raw axis (= raw flash +13.507 pull)", 3519.753 + PULL, "tab:red", "-"),
]

fig, ax = plt.subplots(figsize=(13, 5))
m = (us >= 3505) & (us <= 3545)
ax.plot(us[m], S[m], "k-", lw=0.9, label="summed cathode raw (ped-sub)")
for lab, x, c, ls in MARKS:
    ax.axvline(x, color=c, ls=ls, lw=1.6, label="%s  [%.2f]" % (lab, x))
ax.annotate("", xy=(3523.26, 0.55*S[m].max()), xytext=(3519.75, 0.55*S[m].max()),
            arrowprops=dict(arrowstyle="<->", color="tab:blue"))
ax.text(3521.5, 0.58*S[m].max(), "+3.5 us\n(SP ctoffset)", ha="center", fontsize=8, color="tab:blue")
ax.annotate("", xy=(3533.26, 0.35*S[m].max()), xytext=(3519.75, 0.35*S[m].max()),
            arrowprops=dict(arrowstyle="<->", color="tab:red"))
ax.text(3526.5, 0.38*S[m].max(), "+13.507 us (matcher pull, display only)",
        ha="center", fontsize=8, color="tab:red")
ax.set_xlabel("raw light axis (us, PDVDOpWaveformSource t0)")
ax.set_ylabel("summed cathode ADC")
ax.set_title("evt 298609 track A (flash #117): every timing convention on one axis")
ax.legend(fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("wrote", OUT)
