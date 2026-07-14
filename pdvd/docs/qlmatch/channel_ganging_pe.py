#!/usr/bin/env python3
"""Per-DAPHNE-channel PE asymmetry for the two SiPM channels of one X-Arapuca.

Each PDVD X-Arapuca (cathode 10xx, membrane 20xx) is read out by TWO DAPHNE
OpChannels that OpFlashFinder gangs into ONE OpDet PE column (see
cfg/.../protodunevd/pdvd-opch-map.json).  The photon library predicts one
visibility per OpDet (nchan=40), so the two channels of a pair get an
IDENTICAL predicted light by construction.  This script measures, from the
raw full-stream data, whether the two channels' MEASURED charge is the same
(it is not).

Read-only.  Integrates each channel's pulse charge above a per-channel median
baseline directly from the rawwf ROOT file (positive input polarity), for every
event, and reports the within-pair ratio.

Repro:
    cd wcp-porting-img/pdvd
    python3 docs/qlmatch/channel_ganging_pe.py \
        input_data_light/np02vd_raw_run039252_1176_df-s03-d3_dw_0_20250830T054542_rawwf.root
"""
import sys
import numpy as np
import uproot

# opch pairs sharing one OpDet, from pdvd-opch-map.json (cathode + membrane XAs).
PAIRS = [
    (1010, 1011, 6), (1020, 1021, 4), (1030, 1031, 8), (1040, 1041, 10),
    (1050, 1051, 7), (1060, 1061, 5), (1070, 1071, 9), (1080, 1081, 11),
    (2010, 2011, 0), (2020, 2021, 2), (2040, 2041, 3), (2050, 2051, 12),
    (2060, 2061, 18), (2070, 2071, 13), (2080, 2081, 19),
]


def pulse_area(w):
    """Sum of samples above baseline+5*sigma (noise-thresholded charge)."""
    w = w.astype(np.float64)
    base = np.median(w)
    d = w - base
    neg = d[d < 0]
    noise = np.median(np.abs(neg)) * 1.4826 if neg.size else 1.0
    thr = 5.0 * max(noise, 1e-6)
    nsat = int((w >= 16383).sum())        # 14-bit ADC rail
    return d[d > thr].sum(), base, nsat


def main(fn):
    rw = uproot.open(fn)["rawdump/raw_waveform"]
    a = rw.arrays(["event", "opchannel", "adc"], library="np")
    ev, oc, adc = a["event"], a["opchannel"], a["adc"]
    events = np.unique(ev)
    print("file:", fn)
    print("events:", len(events), events.min(), "..", events.max())
    print("\n%-14s %4s  %-22s %-22s  %s"
          % ("pair->OpDet", "n", "ratio A/B mean+-std", "range", "sat frac A,B"))
    for chA, chB, od in PAIRS:
        rats, satA, satB = [], 0, 0
        for e in events:
            iA = np.flatnonzero((ev == e) & (oc == chA))
            iB = np.flatnonzero((ev == e) & (oc == chB))
            if not (len(iA) and len(iB)):
                continue
            aA, _, sA = pulse_area(adc[iA[0]])
            aB, _, sB = pulse_area(adc[iB[0]])
            if aB:
                rats.append(aA / aB)
            satA += sA > 0
            satB += sB > 0
        if not rats:
            print("%-14s  (channels absent)" % ("%d/%d->%d" % (chA, chB, od)))
            continue
        r = np.array(rats)
        print("%-14s %4d  %6.3f +- %-11.3f [%.3f, %.3f]      %d/%d"
              % ("%d/%d->%d" % (chA, chB, od), len(r), r.mean(), r.std(),
                 r.min(), r.max(), satA, satB))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: channel_ganging_pe.py <rawwf.root>")
    main(sys.argv[1])
