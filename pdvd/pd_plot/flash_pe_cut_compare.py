#!/usr/bin/env python3
"""PDHD vs PDVD: effect of the QLMatching flash_minPE floor on flash rate.

Question (user): compare "# of PE per time" WITHOUT and WITH the flash_minPE cut,
for both detectors -- to judge whether the floor value needs adjustment.

What the floor is: QLMatching drops any flash whose TOTAL PE (sum over all opdets)
is below flash_minPE before matching (match/src/QLMatching.cxx:782,
`flash->get_total_PE() < m_flash_minPE`).  PDHD uses flash_minPE=50; PDVD has NO
floor yet (finder-only 2/10), so its "with cut" is a HYPOTHETICAL sweep.

Data provenance (both are POST-finder, PRE-flash_minPE, so the archive itself is the
"without cut" set and masking total_PE>=floor gives "with cut"):
  PDHD  cfg/.../pdhd/flash.jsonnet         finder 5/20 on 160 PDs, QL flash_minPE=50
  PDVD  cfg/.../protodunevd/flash.jsonnet  finder 2/10 on  40 PDs (36 live), no floor

opflash tensor col0 = flash time in NANOSECONDS rel. to event t0 (earliest light
record); cols 1.. = PE per opdet.  A light "event" is the ~6 ms full-stream readout.
The QL-relevant window is ONE charge readout = nticks x tick:
  PDHD 6000 x 0.512 us = 3072 us ;  PDVD 6000 x 0.5 us = 3000 us  (~equal -> fair).

CAVEAT stated in the doc: absolute PE is NOT comparable across detectors (PDVD
cathode PE scale is provisional/uncalibrated).  Each floor is judged against its OWN
distribution's shape; the only legitimate cross-detector number is flashes/window.

Three views per detector:
  (1) per-flash total-PE histogram (log-x) with the floor line -- DECIDES placement
      (valley between junk pile and signal = well-placed; mid-slope = pure trade-off).
  (2) flashes per drift-window, WITHOUT vs WITH the floor -- the count QL must handle.
  (3) summed-PE-per-time profile of one busy event, without vs with floor -- the
      literal "PE per time" reading.
  + a swept curve: mean flashes/window vs floor value (sensitivity).

Usage: flash_pe_cut_compare.py [--out FIG.png]
"""
import glob
import io
import json
import os
import sys
import tarfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDVD_WORK = "/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/pdvd/work"
PDHD_WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/work"

DETS = {
    "PDHD": dict(glob=os.path.join(PDHD_WORK, "*allpd*", "opflash_pdhd-allpd-wct.tar.gz"),
                 nlive=155, floor=50.0, win_us=3072.0,   # 6000 x 0.512 us
                 sweep=[20, 35, 50, 75, 100, 150]),
    "PDVD": dict(glob=os.path.join(PDVD_WORK, "*", "opflash_pdvd-wct.tar.gz"),
                 nlive=36,  floor=None, win_us=3000.0,    # 6000 x 0.5 us; no floor yet
                 sweep=[10, 12.5, 15, 20, 30, 50]),
}
PROP_PDVD_FLOOR = 15.0     # candidate PDVD floor drawn for reference


def load_opflash(archive):
    """-> times_ns[nflash], total_pe[nflash] from one archive (tensor 0 = opflash)."""
    with tarfile.open(archive, "r:gz") as tf:
        names = tf.getnames()
        an = next(n for n in names if n.endswith("_0_array.npy"))
        a = np.load(io.BytesIO(tf.extractfile(an).read()))
    return a[:, 0], a[:, 1:].sum(axis=1)


def peak_window(t_us, win):
    """max flashes inside any win-wide time window (t_us pre-sorted)."""
    if t_us.size == 0:
        return 0
    return max(np.searchsorted(t_us, x + win) - i for i, x in enumerate(t_us))


def gather(cfg):
    """Read every archive ONCE; cache per-event (t_us, pe) sorted by time."""
    archives = sorted(glob.glob(cfg["glob"]))
    evts = []                         # list of (t_us, pe) per event
    pe_all, win_no, per_evt = [], [], []
    busy = None                       # (t_us, pe, n) of the event with most flashes
    for arc in archives:
        try:
            t_ns, pe = load_opflash(arc)
        except Exception as e:
            print(f"[warn] {arc}: {e}", file=sys.stderr)
            continue
        order = np.argsort(t_ns)
        t_us = t_ns[order] / 1000.0
        pe = pe[order]
        evts.append((t_us, pe))
        pe_all.append(pe)
        win_no.append(peak_window(t_us, cfg["win_us"]))
        per_evt.append(len(pe))
        if busy is None or len(pe) > busy[2]:
            busy = (t_us, pe, len(pe))
    return dict(pe=np.concatenate(pe_all), win_no=np.array(win_no),
                per_evt=np.array(per_evt), busy=busy, n=len(evts), evts=evts)


def win_with_floor(evts, floor, win_us):
    """mean peak flashes/window when only flashes with total_PE>=floor are kept."""
    vals = [peak_window(t_us[pe >= floor], win_us) for t_us, pe in evts]
    return np.array(vals)


def main():
    out = ("--out" in sys.argv and sys.argv[sys.argv.index("--out") + 1]
           or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "docs", "pds", "flash_pe_cut_compare.png"))

    data = {}
    for name, cfg in DETS.items():
        d = gather(cfg)
        data[name] = d
        pe, win_no = d["pe"], d["win_no"]
        floor = cfg["floor"] if cfg["floor"] is not None else PROP_PDVD_FLOOR
        print(f"\n=== {name}: {d['n']} events, {len(pe)} flashes (post-finder, pre-floor) ===")
        print(f"  per-flash total PE: min {pe.min():.1f}  median {np.median(pe):.1f}  "
              f"p90 {np.percentile(pe,90):.0f}  max {pe.max():.0f}")
        print(f"  flashes / {cfg['win_us']:.0f}us drift window (NO floor): "
              f"mean {win_no.mean():.0f}  median {np.median(win_no):.0f}  max {win_no.max()}")
        for f in cfg["sweep"]:
            frac = (pe >= f).mean()
            wm = win_with_floor(d["evts"], f, cfg["win_us"]).mean()
            print(f"    floor {f:6.1f} PE:  keep {frac*100:5.1f}% of flashes  "
                  f"-> {wm:4.0f} flashes/window (mean)")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(2, 4, figsize=(19, 9))
    for r, (name, cfg) in enumerate(DETS.items()):
        d = data[name]
        pe, win_no, busy = d["pe"], d["win_no"], d["busy"]
        floor = cfg["floor"] if cfg["floor"] is not None else PROP_PDVD_FLOOR
        win_yes = win_with_floor(d["evts"], floor, cfg["win_us"])

        # (1) per-flash PE hist, log-x, floor line
        a = ax[r, 0]
        lo = max(pe.min(), 1.0)
        bins = np.logspace(np.log10(lo), np.log10(pe.max()), 60)
        a.hist(pe, bins=bins, color="C0")
        a.axvline(floor, color="r", lw=2,
                  label=(f"flash_minPE={floor:g}" if cfg["floor"]
                         else f"proposed floor {floor:g}"))
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("per-flash total PE"); a.set_ylabel("flashes")
        a.set_title(f"{name}: per-flash PE (median {np.median(pe):.0f})")
        a.legend(fontsize=8)

        # (2) flashes / drift window, without vs with floor
        a = ax[r, 1]
        hi = max(win_no.max(), win_yes.max() if win_yes.size else 0, 1)
        b = np.linspace(0, hi, 30)
        a.hist(win_no, bins=b, color="C3", alpha=0.6,
               label=f"no floor (mean {win_no.mean():.0f})")
        a.hist(win_yes, bins=b, color="C2", alpha=0.6,
               label=f">= {floor:g} PE (mean {win_yes.mean():.0f})")
        a.set_xlabel(f"flashes / {cfg['win_us']:.0f}us window")
        a.set_ylabel("events")
        a.set_title(f"{name}: flashes/window before vs after floor")
        a.legend(fontsize=8)

        # (3) summed-PE-per-time profile of the busiest event, without vs with floor
        a = ax[r, 2]
        t_us, pe_ev, _ = busy
        tb = np.arange(t_us.min(), t_us.max() + 100, 100.0)   # 100us bins
        prof_no, _ = np.histogram(t_us, bins=tb, weights=pe_ev)
        m = pe_ev >= floor
        prof_yes, _ = np.histogram(t_us[m], bins=tb, weights=pe_ev[m])
        ctr = 0.5 * (tb[:-1] + tb[1:]) / 1000.0               # -> ms
        a.step(ctr, np.clip(prof_no, 1, None), color="C3", where="mid",
               label="no floor")
        a.step(ctr, np.clip(prof_yes, 1, None), color="C2", where="mid",
               label=f">= {floor:g} PE")
        a.set_yscale("log")
        a.set_xlabel("time (ms)"); a.set_ylabel("summed PE / 100us")
        a.set_title(f"{name}: busiest event PE(t) ({len(pe_ev)} flashes)")
        a.legend(fontsize=8)

        # (4) sensitivity: mean flashes/window vs floor
        a = ax[r, 3]
        xs = cfg["sweep"]
        ys = [win_with_floor(d["evts"], f, cfg["win_us"]).mean() for f in xs]
        a.plot(xs, ys, "o-", color="C4")
        a.axvline(floor, color="r", lw=1.5, ls="--")
        for x, y in zip(xs, ys):
            a.text(x, y, f"{y:.0f}", fontsize=8, ha="left", va="bottom")
        a.set_xlabel("flash_minPE floor (PE)")
        a.set_ylabel(f"mean flashes / {cfg['win_us']:.0f}us window")
        a.set_title(f"{name}: count-limiting sensitivity")

    fig.suptitle("Flash_minPE floor: PE-per-time before vs after "
                 "(PDHD floor=50 real; PDVD floor hypothetical) -- "
                 "absolute PE NOT comparable across detectors", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=105)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
