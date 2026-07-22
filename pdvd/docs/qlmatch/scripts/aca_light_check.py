#!/usr/bin/env python3
"""Light-side closure for the three anode-crossing cosmics of run 039252
(evts 298609 & 298651): compare the CHARGE-derived track time (W-gauss
anode-touch erf ticks, see aca_crossers_298609.py / fit_endpoints_298609.py /
06_pdvd-drift-crosser-298651.md) against the RAW cathode light waveform and
the reconstructed flashes, on the raw light axis (PDVDOpWaveformSource t0).

Time bases:
  raw light axis   : 16 ns ticks from t0 = min over ALL records of
                     round(timestamp_us*62.5) - (64 if snippet else 0)
  charge frame     : per-crate 0.5 us ticks; frame_us = raw_us + offset_crate
                     (metadata offset_bot_us/offset_top_us, negative)
  folded (Bee) time: raw_us + trigger_offset (= metadata + 13.507 us
                     PDVD_QL_EXTRA_OFFSET_US charge-placement pull)

So a track whose anode-touch W edge is at frame time T appears on the raw
light axis at  T - offset_crate;  the true flash should sit a few us EARLIER
(SP absolute-timing lag: the gauss edge midpoint lands ~+2..+8 us after the
light).

Repro:
    cd pdvd/docs/qlmatch
    python3 scripts/aca_light_check.py
    # writes ../../docs/pics/pdvd_light_timing_aca_{A,B,C}.png and prints the table
"""
import os, sys, io, tarfile
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDVD = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd"
RAW = os.path.join(PDVD, "input_data_light",
    "np02vd_raw_run039252_1176_df-s03-d3_dw_0_20250830T054542_rawwf.root")
OUTPICS = os.path.join(PDVD, "docs", "pics")
TICK_NS = 16.0

# metadata (light<->charge window) offsets, us -- from opflash tensorset metadata
OFF = {298609: {"bot": -2513.808001, "top": -2512.608},
       298651: {"bot": -2515.344,    "top": -2507.743999}}
EXTRA = 13.507                        # PDVD_QL_EXTRA_OFFSET_US (folded times only)

# charge-derived anode-touch W erf ticks (0.5 us) per half:
#   A, B: fit_endpoints_298609.py (fixed-sigma); C: 06_pdvd-drift-crosser-298651.md
CASES = [
    dict(tag="A", event=298609, halves={"bot": 2018.9, "top": 2021.3},
         window=(3450, 3600),
         note="evt 298609 clusters 37+79 (A-C-A, full drift both halves)",
         marks={"matched #117 (22117 PE)": 3519.753}),
    dict(tag="B", event=298609, halves={"bot": 5789.3, "top": 5780.1},
         window=(5140, 5470),
         note="evt 298609 clusters 83+50 (A-C-A, cathode ends readout-truncated)",
         marks={"matched #160 (18967 PE)": 5193.301, "true #163 (24384 PE)": 5399.991}),
    dict(tag="C", event=298651, halves={"bot": 579.8, "top": 592.2},
         window=(2750, 2940),
         note="evt 298651 clusters 35+95 (= doc-06 crosser; 95 matched #84, 35 unmatched)",
         marks={"matched #84 (7382 PE)": 2801.949, "v153 pick gid88 (1982 PE)": 2880.005}),
]

# ---------------- raw light: summed cathode stream per event ----------------
d = uproot.open(RAW)["rawdump/raw_waveform"].arrays(
    ["event", "opchannel", "nsamp", "timestamp", "adc"], library="np")

def cathode_sum(event):
    sel = np.nonzero(d["event"] == event)[0]
    starts = {i: int(round(d["timestamp"][i] * 62.5))
              - (64 if d["nsamp"][i] <= 1024 else 0) for i in sel}
    t0 = min(starts.values())
    cidx = [i for i in sel if d["opchannel"][i] // 1000 == 1]
    S = np.zeros(int(d["nsamp"][cidx[0]]))
    for i in cidx:
        a = np.asarray(d["adc"][i], dtype=np.float64)
        S[:a.size] += a - np.median(a[:200])
    off = starts[cidx[0]] - t0            # cathode stream start on the raw axis
    return S, off

def flashes(event):
    tag = {298609: "039252_light298609_keep", 298651: "039252_light298651_keep"}[event]
    with tarfile.open(os.path.join(PDVD, "work", tag, "opflash_pdvd-wct.tar.gz")) as tf:
        op = np.load(io.BytesIO(tf.extractfile(
            "opflash_tensor_%d_0_array.npy" % event).read()))
    return op[:, 0] / 1000.0, op[:, 1:41].sum(axis=1)

def rise20(S, off, us):
    """20%-of-max rise (raw-axis us) of the summed-cathode pulse nearest `us`."""
    k0 = int((us - off * 0.0) * 1000 / TICK_NS) - off   # us -> stream sample
    lo, hi = max(0, k0 - 1500), min(len(S), k0 + 1500)  # +-24 us
    seg = S[lo:hi]
    if not len(seg) or seg.max() < 100:
        return None
    k = lo + int(np.argmax(seg))
    thr = 0.2 * S[k]
    j = k
    while j > 0 and S[j] > thr:
        j -= 1
    return (j + off) * TICK_NS / 1000.0

_sums = {}
print("%-3s %-5s %10s %10s %12s %12s %10s" %
      ("trk", "half", "erf tick", "frame us", "charge->raw", "rise20 raw", "chg-rise"))
for c in CASES:
    ev = c["event"]
    if ev not in _sums:
        _sums[ev] = cathode_sum(ev)
    S, off = _sums[ev]
    t_fl, pe_fl = flashes(ev)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    lo_us, hi_us = c["window"]
    us = (np.arange(len(S)) + off) * TICK_NS / 1000.0
    m = (us >= lo_us) & (us <= hi_us)
    ax.plot(us[m], S[m], "k-", lw=0.7, label="summed cathode raw (ped-sub)")
    fm = (t_fl >= lo_us) & (t_fl <= hi_us)
    ax.stem(t_fl[fm], pe_fl[fm] * (S[m].max() / max(pe_fl[fm].max(), 1)),
            linefmt="g-", markerfmt="g^", basefmt=" ",
            label="reconstructed flashes (PE, scaled)")
    cols = {"bot": "tab:blue", "top": "tab:red"}
    for half, tick in c["halves"].items():
        frame_us = tick * 0.5
        raw_us = frame_us - OFF[ev][half]
        ax.axvline(raw_us, color=cols[half], ls="--", lw=1.4,
                   label="charge anode-touch, %s (raw %.2f us)" % (half, raw_us))
        r = rise20(S, off, raw_us)
        print("%-3s %-5s %10.1f %10.2f %12.2f %12s %10s" %
              (c["tag"], half, tick, frame_us, raw_us,
               "%.2f" % r if r else "-",
               "%+.2f" % (raw_us - r) if r else "-"))
    for lab, t in c["marks"].items():
        ax.annotate(lab, xy=(t, S[m].max() * 0.92), xytext=(t, S[m].max() * 1.02),
                    ha="center", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_xlabel("raw light axis (us, PDVDOpWaveformSource t0)")
    ax.set_ylabel("summed cathode ADC")
    ax.set_title("track %s: %s" % (c["tag"], c["note"]))
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p = os.path.join(OUTPICS, "pdvd_light_timing_aca_%s.png" % c["tag"])
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("  -> %s" % p)
