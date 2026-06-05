#!/usr/bin/env python3
"""Per-event flash time closest to T0=0 for the lan-reco2 data sample.

Each bundle dir (1/2/3) under input_files/input-3files-lan-reco2/ has one
opflash tar per APA. Per event the opflash tensor is [nflash, nchan+1]; column 0
is the flash time in WCT ns (units::us = 1000). The tar holds ALL flashes of the
event (cosmics span the full ~±1.2 ms readout), so the in-time/beam flash is the
one with min |time|. We take min |time| across BOTH APAs per event.

Run from sbnd_xin/:  python3 flash_t0_lan_reco2.py
Writes pics/flash_t0_lan_reco2.png and prints the per-event table + summary.
"""
import io, os, re, tarfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "input_files/input-3files-lan-reco2")
BUNDLES = ["1", "2", "3"]
US = 1000.0  # WCT ns per us
PAT = re.compile(r"opflash_tensor_(\d+)_0_array\.npy$")


def load_bundle_apa(path):
    """event_id -> array of flash times (us)."""
    out = {}
    with tarfile.open(path, "r:gz") as tf:
        for m in tf.getmembers():
            mm = PAT.search(m.name)
            if not mm:
                continue
            a = np.load(io.BytesIO(tf.extractfile(m).read()))
            out[int(mm.group(1))] = a[:, 0] / US
    return out


rows = []
for b in BUNDLES:
    a0 = load_bundle_apa(f"{BASE}/{b}/opflash_apa0.tar.gz")
    a1 = load_bundle_apa(f"{BASE}/{b}/opflash_apa1.tar.gz")
    for e in sorted(set(a0) | set(a1)):
        times = [a0[e] for _ in (1,) if e in a0 and len(a0[e])]
        times += [a1[e] for _ in (1,) if e in a1 and len(a1[e])]
        if not times:
            continue
        times = np.concatenate(times)
        rows.append((b, e, times[np.argmin(np.abs(times))], len(times)))

print(f"{'bundle':>6} {'event':>8} {'t_closest_us':>14} {'nflash':>7}")
for b, e, t0, n in rows:
    print(f"{b:>6} {e:>8} {t0:>14.4f} {n:>7}")

t0s = np.array([r[2] for r in rows])
peak = t0s[(t0s > -1.2) & (t0s < 0.0)]
print(f"\nN events = {len(t0s)}")
print(f"mean   = {t0s.mean():+.4f} us   median = {np.median(t0s):+.4f} us   std = {t0s.std():.4f}")
print(f"in (-1.2,0): N={len(peak)}  mean={peak.mean():+.4f}  std={peak.std():.4f}")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.6))
ax0.hist(t0s, bins=np.arange(-3.5, 3.6, 0.25), color="#4477AA", edgecolor="k", lw=0.4)
ax0.axvline(0, color="grey", ls=":", lw=1)
ax0.set_xlabel("flash time closest to T0=0  [us]")
ax0.set_ylabel("events / 0.25 us")
ax0.set_title(f"All {len(t0s)} events (3-file lan-reco2 data)")
ax0.text(0.97, 0.95, f"mean   {t0s.mean():+.3f}\nmedian {np.median(t0s):+.3f}\nstd     {t0s.std():.3f}",
         transform=ax0.transAxes, ha="right", va="top", family="monospace", fontsize=9,
         bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax1.hist(peak, bins=np.arange(-1.2, 0.01, 0.02), color="#CC6677", edgecolor="k", lw=0.4)
ax1.axvline(peak.mean(), color="k", ls="--", lw=1, label=f"mean {peak.mean():+.3f} us")
ax1.set_xlabel("flash time closest to T0=0  [us]")
ax1.set_ylabel("events / 0.02 us")
ax1.set_title(f"In-time peak: {len(peak)}/{len(t0s)} events in (-1.2, 0)")
ax1.text(0.03, 0.95, f"mean {peak.mean():+.3f}\nstd   {peak.std():.3f}",
         transform=ax1.transAxes, ha="left", va="top", family="monospace", fontsize=9,
         bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax1.legend(loc="upper right", fontsize=9)
fig.suptitle("SBND data — per-event opflash time nearest T0=0 (both TPCs), input-3files-lan-reco2", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, "pics/flash_t0_lan_reco2.png")
fig.savefig(out, dpi=130)
print("wrote", out)
