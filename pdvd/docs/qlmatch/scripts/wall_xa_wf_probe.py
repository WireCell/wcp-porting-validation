#!/usr/bin/env python3
"""Membrane-frame waveform probe for the wall-XA study (doc 25, 7 events).

For every wall-XA (flash, channel) case with cathode-normalized expectation
exp >= 50 PE in the 7 events that have membrane light frames, classify:

  LIGHT-AT-FLASH-TIME  decon integral in [t-3, t+12] us >= max(12, 0.25*exp)
                       -> the light IS in the waveform; the flash PE lost it
  SNIPPET-BUT-FLAT     raw self-trigger snippet within +-30 us but decon flat
                       -> DAPHNE triggered on something, this flash's light
                          was small/diffuse or absent
  NO-SNIPPET-30US      no raw readout at all within +-30 us
                       -> self-trigger silent (readout loss)

and for LIGHT-AT-FLASH-TIME cases list the neighboring flashes (+-12 us) that
hold >10 PE on the channel -- the booking destination.

Time base (verified): light-frame us = dump.time - trigger_offsets_us[0];
16 ns ticks; the membrane-only standalone frames share the all-PD origin
(DELTA scan came out 0).  Decon amplitude integrates to PE (closure on fired
cases ~1.0 for moderate PE; bright cases read low from saturation recovery
in the flash PE).

Frames for events other than 298567 are regenerated to WALLXA_DIR with the
validation job (see the doc 25 Repro block):
  wcsonnet -A input_file=<rawwf.root> -A output_file=$WALLXA_DIR/membrane-frames-<run>-<evt>.tar.bz2 \
           -A run=<run> -A event=<evt> -A branch=membrane -o cfg.json \
           pdvd/wct-light-frames.jsonnet   # then wire-cell -c cfg.json
"""
import csv
import glob
import io
import json
import os
import tarfile
import numpy as np

SP = os.environ.get("WALLXA_DIR", os.path.dirname(os.path.abspath(__file__)))
BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
OPCH = {0: [2010, 2011], 1: [2030], 3: [2040, 2041],
        12: [2050, 2051], 18: [2060, 2061], 19: [2080, 2081]}
TICK = 0.016
EXP_MIN = 50.0
EVENTS = {
    ("039252", "298567"): f"{BASE}/039252_light298567_keep/light-frames-membrane.tar.bz2",
    ("039252", "298777"): f"{SP}/membrane-frames-39252-298777.tar.bz2",
    ("039253", "49746"):  f"{SP}/membrane-frames-39253-49746.tar.bz2",
    ("039253", "49806"):  f"{SP}/membrane-frames-39253-49806.tar.bz2",
    ("039253", "49966"):  f"{SP}/membrane-frames-39253-49966.tar.bz2",
    ("039349", "19709"):  f"{SP}/membrane-frames-39349-19709.tar.bz2",
    ("039349", "55826"):  f"{SP}/membrane-frames-39349-55826.tar.bz2",
}

rows = list(csv.DictReader(open(f"{SP}/wall_xa_flash_channel.tsv"), delimiter="\t"))
f = lambda r, k: float(r[k])
for r in rows:
    r["exp"] = f(r, "pred") * f(r, "r_cath")

counts, neigh = {}, []
for (run, evt), path in EVENTS.items():
    ev = [r for r in rows if r["run"] == run and r["evt"] == evt
          and int(r["ch"]) in OPCH]
    if not ev or not os.path.exists(path):
        print(f"# skipping {run} evt{evt} (no rows or frames)")
        continue
    idx = ev[0]["idx"]
    d = json.load(open(glob.glob(f"{BASE}/{run}_{idx}_keep/calib-evt*.json")[0]))
    off0 = d["trigger_offsets_us"][0]
    flashes = d["flashes"]
    with tarfile.open(path) as tf:
        F = {nm.rsplit("_", 1)[0]: np.load(io.BytesIO(tf.extractfile(nm).read()))
             for nm in tf.getnames()}
    Fd, Cd = F["frame_decon"], list(F["channels_decon"])
    Fr, Cr = F["frame_raw"], list(F["channels_raw"])
    nt = Fd.shape[1]
    for r in ev:
        if r["exp"] < EXP_MIN:
            continue
        meas, cov = f(r, "meas"), int(r["cov"])
        cls = ("DEAD" if cov == 1 and meas < 0.25 * r["exp"] else
               "NOCOV" if cov == 0 else
               "RESP" if cov == 1 and 0.5 * r["exp"] <= meas <= 3 * r["exp"] else None)
        if cls is None:
            continue
        ch = int(r["ch"])
        t = f(r, "t") - off0
        chans = [c for c in OPCH[ch] if c in Cd]
        w = np.sum([Fd[Cd.index(c)] for c in chans], axis=0)
        rw = np.sum([np.abs(Fr[Cr.index(c)].astype(float)) for c in chans], axis=0)
        i0, i1 = max(0, int((t - 3) / TICK)), min(nt, int((t + 12) / TICK))
        inw = float(w[i0:i1].sum())
        raw_any = bool(np.any(rw[max(0, int((t - 30) / TICK)):
                                 min(nt, int((t + 30) / TICK))] > 0))
        thr = max(12.0, 0.25 * r["exp"])
        v = ("LIGHT-AT-FLASH-TIME" if inw >= thr else
             "SNIPPET-BUT-FLAT" if raw_any else "NO-SNIPPET-30US")
        counts[(cls, v)] = counts.get((cls, v), 0) + 1
        if v == "LIGHT-AT-FLASH-TIME" and cls in ("DEAD", "NOCOV"):
            near = [(g["gid"], round(g["time"] - f(r, "t"), 2), round(g["pe"][ch], 1))
                    for g in flashes
                    if abs(g["time"] - f(r, "t")) < 12 and g["gid"] != int(r["gid"])
                    and g["pe"][ch] > 10]
            neigh.append((run, evt, r["gid"], ch, round(r["exp"]), round(meas),
                          round(inw), near))

print("class  verdict                n")
for (cls, v), n in sorted(counts.items()):
    print(f"{cls:6s} {v:20s} {n}")
print("\nDEAD/NOCOV with light at flash time -- neighbors holding the PE "
      "(gid, dt_us, pe[ch]):")
for x in neigh:
    print(" ", x)
