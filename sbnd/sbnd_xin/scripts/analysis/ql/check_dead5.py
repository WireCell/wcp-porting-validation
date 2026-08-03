#!/usr/bin/env python3
"""Check whether channels {67,92,170,218,248} are dead in given opflash datasets.

A channel is "dead" if it reads pe==0 in ~100% of flashes in ITS OWN TPC archive
(apa0 if model x<0, apa1 if x>=0) -- where it would see light if alive.
Reuses the geometry / archive conventions from saturation_pe.py.
"""
import io, json, os, re, tarfile, glob
import numpy as np

SEMIMODEL = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet/semi-analytical-sbnd.json"
INPUT = "/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin/input_files"
CHANS = [67, 92, 170, 218, 248]
EV_RE = re.compile(r"opflash_tensor_(\d+)_0_array\.npy$")

# channel -> TPC/apa from the semi-analytical model
od = json.load(open(SEMIMODEL))["OpDets"]
ch_apa = {c: (0 if od[c]["x"] < 0 else 1) for c in CHANS}

def load_archive(path):
    out = {}
    with tarfile.open(path, "r:gz") as t:
        for m in t.getmembers():
            mo = EV_RE.search(m.name)
            if mo:
                out[int(mo.group(1))] = np.load(io.BytesIO(t.extractfile(m).read()))
    return out

def all_flash_pe(paths):
    """Stack per-channel PE (cols 1..312) across all flashes in the given archives."""
    mats = []
    for p in paths:
        for arr in load_archive(p).values():
            mats.append(arr[:, 1:])   # [nflash, 312]
    return np.vstack(mats) if mats else np.zeros((0, 312))

# (label, {apa: [archive paths]})
DATASETS = {
    "data v10_14_02_02": {a: [f"{INPUT}/input-1file-data-v10_14_02_02/opflash_apa{a}.tar.gz"] for a in (0,1)},
    "data apr29":        {a: [f"{INPUT}/input-apr29/opflash_apa{a}.tar.gz"] for a in (0,1)},
    "mc 10files":        {a: sorted(glob.glob(f"{INPUT}/input-10files-mc/opflash_apa{a}-*.tar.gz")) for a in (0,1)},
    # originals, for reference
    "data 10evt (orig)": {a: [f"{INPUT}/input-10evt-data/opflash_apa{a}.tar.gz"] for a in (0,1)},
    "mc 10evt (orig)":   {a: [f"{INPUT}/input-10evt-mc/opflash_apa{a}.tar.gz"] for a in (0,1)},
}

print("channel -> own TPC:", ch_apa)
for label, apamap in DATASETS.items():
    # skip if archives missing
    missing = [p for ps in apamap.values() for p in ps if not os.path.exists(p)]
    if missing or not any(apamap.values()):
        print(f"\n=== {label}: SKIP (archives not found) ===")
        continue
    pe = {a: all_flash_pe(apamap[a]) for a in (0,1)}
    nfl = {a: pe[a].shape[0] for a in (0,1)}
    print(f"\n=== {label}  (apa0 {nfl[0]} flashes, apa1 {nfl[1]} flashes) ===")
    print(f"  {'chan':>5} {'apa':>3} {'nflash':>7} {'n_zero':>7} {'zero_frac':>9} {'max_PE':>9}  dead?")
    for c in CHANS:
        a = ch_apa[c]
        col = pe[a][:, c]
        n = col.size
        nz = int((col == 0).sum())
        frac = nz / n if n else float("nan")
        dead = "DEAD" if frac == 1.0 else ("~dead" if frac > 0.99 else "LIVE")
        print(f"  {c:>5} {a:>3} {n:>7} {nz:>7} {frac:>9.4f} {col.max():>9.2f}  {dead}")
