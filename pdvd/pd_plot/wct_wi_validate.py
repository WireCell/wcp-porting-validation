#!/usr/bin/env python3
"""Validate the C++ OpDecon wiener-inspired mode against an independent
python replication (standalone-chain cross-check).

For one event and branch, re-deconvolve every raw_waveform record with a
float64 mirror of the C++ path -- head pedestal over nped = pre_trigger -
pedestal_buffer = 20 samples (NOT the python study's 40), zero-pad to the
branch `samples`, G = conj(H) F / (|H|^2 + (1e-3 max|H|)^2) with
F = exp(-0.5 (f/sigma)^2) on the df = 62.5/N MHz grid, post-BL
correction over nped, no post-filter, scale 1 -- using the SAME template
values as the C++ (the cfg pdvd-spe-templates.json), and compare against
the dense "decon" frame dumped by wct-light-frames.jsonnet.

Records whose padded [start, start+samples) span overlaps another record
on the same channel are skipped (dense-frame collision, not a decon
issue).

Usage: wct_wi_validate.py <workdir> <branch> [run] [event]
       e.g. wct_wi_validate.py work/039252_light298567 pmt 39252 298567
"""
import io
import json
import os
import sys
import tarfile

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdvd_light as pl

CFG = os.environ.get(
    "PDVD_CFG_DIR",
    "/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/protodunevd")
BRANCH = {
    "cathode": dict(lo=1000, hi=1999, samples=468864, sigma=1.25),
    "membrane": dict(lo=2000, hi=2999, samples=1024, sigma=1.0),
    "pmt": dict(lo=3000, hi=3999, samples=1024, sigma=3.5),
}
NPED = 20          # OpDecon pre_trigger(50) - pedestal_buffer(30)
TRIG_SAMPLE = 64
EPS_REL = 1e-3


def wi_decon(adc, tmpl, N, sigma, power=2.0):
    ped = adc[:NPED].mean()
    xv = np.zeros(N)
    n = min(N, len(adc))
    xv[:n] = adc[:n] - ped                       # input_polarity +1
    wave = np.zeros(N)
    wave[:min(N, len(tmpl))] = tmpl[:N]
    H = np.fft.fft(wave)
    eps = (EPS_REL * np.abs(H).max()) ** 2
    k = np.arange(N)
    f = (62.5 / N) * np.minimum(k, N - k)
    F = np.exp(-0.5 * (f / sigma) ** power)
    G = np.conj(H) * F / (np.abs(H) ** 2 + eps)
    dec = np.fft.ifft(G * np.fft.fft(xv)).real
    return dec - dec[:NPED].mean()               # post-blcorr, scale=1


def main():
    workdir, branch = sys.argv[1], sys.argv[2]
    run = int(sys.argv[3]) if len(sys.argv) > 3 else 39252
    event = int(sys.argv[4]) if len(sys.argv) > 4 else 298567
    b = BRANCH[branch]

    jspe = json.load(open(os.path.join(CFG, "pdvd-spe-templates.json")))
    ch2tmpl = {ch: np.asarray(jspe["templates"][ti]["values"], dtype=np.float32)
               for ch, ti in zip(jspe["channels"], jspe["template_index"])}

    arrs = {}
    with tarfile.open(os.path.join(workdir, f"light-frames-{branch}.tar.bz2")) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".npy"):
                arrs[m.name] = np.load(io.BytesIO(tf.extractfile(m).read()))
    frame = arrs[f"frame_decon_{event}.npy"]
    chans = arrs[f"channels_decon_{event}.npy"]
    tbin0 = int(arrs[f"tickinfo_decon_{event}.npy"][2])

    src = [p for p in pl.data_files() if f"run{run:06d}" in p][0]
    t = uproot.open(src)["raw_waveform"]
    d = t.arrays(["event", "opchannel", "nsamp", "timestamp", "adc"], library="np")
    sel = d["event"] == event
    starts = (np.round(d["timestamp"] * 62.5).astype(np.int64)
              - np.where(d["nsamp"] <= 1024, TRIG_SAMPLE, 0))
    t0 = starts[sel].min()

    # branch records + padded-span overlap veto per channel
    recs = np.nonzero(sel & (d["opchannel"] >= b["lo"]) & (d["opchannel"] <= b["hi"]))[0]
    spans = {}
    for i in recs:
        spans.setdefault(d["opchannel"][i], []).append(
            (int(starts[i] - t0), int(starts[i] - t0) + b["samples"], i))
    todo = []
    nskip = 0
    for ch, lst in spans.items():
        lst.sort()
        for j, (s, e, i) in enumerate(lst):
            if (j > 0 and lst[j - 1][1] > s) or (j + 1 < len(lst) and lst[j + 1][0] < e):
                nskip += 1
            else:
                todo.append((ch, s, i))

    worst = 0.0
    rels = []
    for ch, s, i in todo:
        row = np.nonzero(chans == ch)[0]
        if not len(row):
            continue
        cpp = frame[row[0], s - tbin0:s - tbin0 + b["samples"]].astype(np.float64)
        ref = wi_decon(np.asarray(d["adc"][i], dtype=np.float64),
                       ch2tmpl[ch], b["samples"], b["sigma"])
        scale = max(np.abs(ref).max(), 1e-12)
        rel = np.abs(cpp - ref).max() / scale
        rels.append(rel)
        if rel > worst:
            worst = rel
            print(f"  worst so far: opch {ch} tbin {s}  max|d|/max|ref| = {rel:.3g}")
    rels = np.array(rels)
    print(f"\n{branch}: {len(rels)} records compared ({nskip} skipped for overlap)")
    print(f"  max rel diff    : {rels.max():.3g}")
    print(f"  median rel diff : {np.median(rels):.3g}")
    ok = rels.max() < 1e-4
    print("PASS (float32-template agreement)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
