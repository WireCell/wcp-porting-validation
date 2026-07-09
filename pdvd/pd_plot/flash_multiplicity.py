#!/usr/bin/env python3
"""PDVD flash-multiplicity study for QLMatching prep.

Question: PDVD OpFlashFinder already applies a quality cut
(min_fired_pds=2, min_total_pe=10, min_fired_pe=1.0 -- the 1/4-scaled analog of
PDHD's 5/20 on 160 PDs), yet many flashes survive per event.  QLMatching needs a
small, clean flash set.  This script diagnoses WHICH lever reduces the count:

  (a) a tighter PE / nPD threshold, or
  (b) temporal refinement/grouping of scintillation fragments
      (PDVD runs OpFlashFinder with flash_refine=false, group_by_side=false).

It reads the emitted opflash tensor-sets (already post-2/10, so this can only
evaluate TIGHTENING, never loosening) and reports, per event and pooled:
  - flashes/event,
  - inter-flash time-gap distribution (fragmentation signal),
  - per-flash total-PE and nPD (>=1 PE) and their 2-D density,
  - flashes/event under a swept tighter (nPD, PE) cut.

opflash tensor schema (flash/src/OpFlashFinder.cxx:663-666):
  col 0   = fs.time  (PE-weighted flash time, NANOSECONDS relative to the event
                      t0 = earliest light record; the light event spans the
                      ~6 ms cathode full-stream readout)
  col 1.. = fs.pes[OpDet]  (PE per OpDet; nchan columns)

The QL-relevant number is NOT flashes/event (a ~6 ms readout holding hundreds of
well-separated cosmic flashes) but the peak flash occupancy in one charge drift
window (PDVD 6000 ticks x 0.5 us = 3000 us), since QLMatching matches clusters
from one TPC readout.

Usage: flash_multiplicity.py [WORK_DIR] [--out FIG.png]
  WORK_DIR defaults to the pdvd/work tree of opflash_*.tar.gz archives.
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

DEF_WORK = "/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/pdvd/work"
FIRED_PE = 1.0                       # per-PD "fired" threshold (min_fired_pe)
DRIFT_US = 3000.0                    # PDVD QL drift window: 6000 ticks x 0.5 us
SWEEP = [(2, 10.0), (3, 20.0), (4, 30.0), (5, 40.0), (6, 60.0)]  # (min_pds, min_pe)
PE_FLOORS = [10, 12.5, 15, 20, 50]   # flash_minPE-style match-side floors


def load_opflash(archive):
    """-> (times[nflash], pes[nflash, nchan], event_id) from one archive."""
    with tarfile.open(archive, "r:gz") as tf:
        names = tf.getnames()
        arr_name = next(n for n in names
                        if n.endswith("_0_array.npy"))       # tensor 0 = opflash
        a = np.load(io.BytesIO(tf.extractfile(arr_name).read()))
        ev = None
        for n in names:
            if "tensorset" in n and n.endswith("metadata.json"):
                ev = json.load(tf.extractfile(n)).get("event")
                break
    return a[:, 0], a[:, 1:], ev


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    work = args[0] if args else DEF_WORK
    out = "--out" in sys.argv and sys.argv[sys.argv.index("--out") + 1] \
        or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "docs", "pds", "flash_multiplicity.png")

    archives = sorted(glob.glob(os.path.join(work, "*", "opflash_*.tar.gz")))
    if not archives:
        sys.exit(f"no opflash archives under {work}")

    per_evt_n = []          # flashes/event (post-2/10, as produced)
    per_evt_win = []        # peak flashes in any DRIFT_US window (the QL number)
    all_gaps = []           # inter-flash time gaps (us), within-event
    all_pe = []             # per-flash total PE
    all_npd = []            # per-flash nPD (>=1 PE)
    floor_win = {f: [] for f in PE_FLOORS}   # peak/drift-window with PE>=floor

    for arc in archives:
        try:
            t_ns, pes, ev = load_opflash(arc)
        except Exception as e:
            print(f"[warn] {arc}: {e}", file=sys.stderr)
            continue
        t = np.sort(t_ns) / 1000.0            # ns -> us
        pes = pes[np.argsort(t_ns)]
        n = t.shape[0]
        per_evt_n.append(n)
        tot = pes.sum(axis=1)
        npd = (pes >= FIRED_PE).sum(axis=1)
        all_pe.append(tot)
        all_npd.append(npd)
        if n >= 2:
            all_gaps.append(np.diff(t))

        def peak_window(mask):
            """max flashes (with mask) in any DRIFT_US-wide window."""
            tt = t[mask]
            if tt.size == 0:
                return 0
            return max(np.searchsorted(tt, x + DRIFT_US) - i
                       for i, x in enumerate(tt))
        per_evt_win.append(peak_window(np.ones(n, bool)))
        for f in PE_FLOORS:
            floor_win[f].append(peak_window(tot >= f))

    per_evt_n = np.array(per_evt_n)
    per_evt_win = np.array(per_evt_win)
    all_pe = np.concatenate(all_pe)
    all_npd = np.concatenate(all_npd)
    all_gaps = np.concatenate(all_gaps) if all_gaps else np.array([])

    # ---- report ----
    print(f"PDVD flash multiplicity: {len(per_evt_n)} events, "
          f"{int(per_evt_n.sum())} flashes total (post-2/10)")
    print(f"  flashes / ~6ms light-readout event: mean {per_evt_n.mean():.0f}  "
          f"median {np.median(per_evt_n):.0f}  ({per_evt_n.min()}-{per_evt_n.max()})")
    print(f"  peak flashes in any {DRIFT_US:.0f}us QL drift window (THE QL number): "
          f"mean {per_evt_win.mean():.0f}  max {per_evt_win.max()}")
    if all_gaps.size:
        for thr in (1.0, 2.0, 5.0, 11.0):
            print(f"  inter-flash gaps < {thr:5.1f} us: "
                  f"{(all_gaps < thr).mean()*100:5.1f}%  "
                  f"(2us ~ one scintillation)")
        print(f"  gap median {np.median(all_gaps):.1f} us  "
              f"(mostly genuine, well-separated flashes)")
    print(f"  peak/drift-window under a flash_minPE-style floor:")
    for f in PE_FLOORS:
        c = np.array(floor_win[f])
        print(f"    total_PE >= {f:5.1f}:  mean {c.mean():5.0f}/window  "
              f"max {c.max()}")

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    ax[0, 0].hist(per_evt_n, bins=40, color="C0", label="flashes / ~6ms event")
    ax[0, 0].hist(per_evt_win, bins=40, color="C3",
                  label=f"peak / {DRIFT_US:.0f}us QL window")
    ax[0, 0].set_xlabel("flashes")
    ax[0, 0].set_ylabel("events")
    ax[0, 0].set_title(f"flashes/event {per_evt_n.mean():.0f} vs "
                       f"peak/QL-window {per_evt_win.mean():.0f}")
    ax[0, 0].legend(fontsize=8)

    if all_gaps.size:
        pos = all_gaps[all_gaps > 0]
        ax[0, 1].hist(np.log10(pos), bins=60, color="C1")
        ax[0, 1].axvline(np.log10(2.0), color="k", ls="--", lw=1,
                         label="2 us ~ 1 scintillation")
        ax[0, 1].set_xlabel("log10(inter-flash time gap / us)")
        ax[0, 1].set_ylabel("flash pairs")
        ax[0, 1].set_title(f"time gaps: median {np.median(all_gaps):.0f}us "
                           f"(genuine, well-separated)")
        ax[0, 1].legend(fontsize=8)

    ax[1, 0].hist2d(all_npd, np.log10(np.clip(all_pe, 1e-1, None)),
                    bins=[np.arange(0.5, 40.5, 1), 40], cmap="viridis")
    ax[1, 0].set_xlabel("nPD fired (>=1 PE)")
    ax[1, 0].set_ylabel("log10(total PE)")
    ax[1, 0].set_title("per-flash PE vs nPD density (dim pile at the 2/10 floor)")

    labels = [f">={f:g}" for f in PE_FLOORS]
    means = [np.mean(floor_win[f]) for f in PE_FLOORS]
    ax[1, 1].bar(labels, means, color="C2")
    ax[1, 1].set_xlabel("flash total-PE floor (flash_minPE analog)")
    ax[1, 1].set_ylabel(f"mean peak flashes / {DRIFT_US:.0f}us window")
    ax[1, 1].set_title("count-limiting lever = a match-side PE floor (QL work)")
    for i, m in enumerate(means):
        ax[1, 1].text(i, m, f"{m:.0f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"PDVD flash multiplicity for QLMatching "
                 f"({len(per_evt_n)} events, runs 039252/039253/039349)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
