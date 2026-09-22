#!/usr/bin/env python3
"""plot_flash_waveforms.py -- the matched flash of one STM candidate: the per-PD PE pattern and
the raw and deconvolved waveforms of the brightest photon detectors in the exported window.

    python3 scripts/plot_flash_waveforms.py [release_dir] --event 029107_1135 [--cluster 30] [--nchan 6] [--out ...]

Top: PE per OpDet of the matched flash (T_flash.pe), the drawn channels marked.  Below, one row
per chosen readout channel: raw ADC (pedestal not subtracted) and the deconvolved trace
(units: PE per sample) from T_opwf, with the OpHits of that flash and channel (T_ophit) marked
at their peak times.  The window is [flash - wf_pre_us, flash + wf_post_us] on the light time
axis; the Michel electron's light arrives within a few us after the muon's flash.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stm_release as sr
import relplot as rp
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release", nargs="?", default=None)
    ap.add_argument("--event", required=True, help="<run6>_<evtid>")
    ap.add_argument("--cluster", type=int, default=None, help="candidate cluster id (default: first is_stm candidate)")
    ap.add_argument("--nchan", type=int, default=6, help="how many readout channels to draw (brightest first)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rel = sr.release_dir(a.release)
    info = sr.event_info(sr.open_event(sr.event_files(rel)[0]))
    ev = sr.open_event(f"{rel}/events/{a.event}/stm_michel_{info['det']}_{a.event}.root")
    info = sr.event_info(ev); det = info["det"].upper()
    c = sr.candidates(ev, ["cluster_id", "is_stm", "michel_found", "flash_id", "michel_ke_q2d_region", "muon_ke_range"])
    if a.cluster is None:
        a.cluster = int(c["cluster_id"][c["is_stm"] == 1][0])
    i = int(np.flatnonzero(c["cluster_id"] == a.cluster)[0]); fid = int(c["flash_id"][i])
    fl = sr.flash(ev, fid)
    wfs = sr.waveforms(ev, flash_id=fid)
    order = np.argsort(-np.asarray(wfs["pe"]), kind="stable")[: a.nchan]
    fig = plt.figure(figsize=(12, 2.2 * (len(order) + 1) + 1))
    gs = fig.add_gridspec(len(order) + 1, 2, height_ratios=[1.6] + [1] * len(order))
    ax0 = fig.add_subplot(gs[0, :])
    ax0.bar(np.arange(len(fl["pe"])), fl["pe"], color=rp.BLUE, width=0.8)
    for j in order:
        od = int(wfs["opdet"][j]); ax0.plot(od, fl["pe"][od], "v", color=rp.ORANGE, ms=7)
    ax0.set_xlabel("OpDet"); ax0.set_ylabel("PE")
    ax0.set_title(f"{det} {a.event} cluster {a.cluster} (is_stm {c['is_stm'][i]}, Michel {c['michel_found'][i]}, KE range {c['muon_ke_range'][i]:.0f} MeV, "
                  f"Michel {c['michel_ke_q2d_region'][i]:.1f} MeV)\nmatched flash {fid}: t = {fl['time_us']:.2f} us on the light axis, {fl['total_pe']:.0f} PE; drawn channels marked", fontsize=10)
    hits = sr.ophits(ev, flash_id=fid)
    for r, j in enumerate(order):
        w = sr.waveform(ev, fid, int(wfs["channel"][j]))
        t = w["t_us"] - fl["time_us"]
        axr = fig.add_subplot(gs[r + 1, 0]); axd = fig.add_subplot(gs[r + 1, 1])
        axr.plot(t, w["raw"], color=rp.BLUE, lw=1); axd.plot(t, w["decon"], color=rp.ORANGE, lw=1)
        if len(w["decon_roi"]):
            axd.plot(t, w["decon_roi"], color=rp.AQUA, lw=1, alpha=0.8)
        hh = hits["peak_time_us"][hits["channel"] == w["channel"]] - fl["time_us"]
        for hx in hh:
            axd.axvline(hx, color=rp.INK2, lw=0.8, ls=":")
        axr.set_ylabel(f"ch {w['channel']} (OpDet {w['opdet']})\nraw ADC", fontsize=8); axd.set_ylabel("decon [PE/sample]", fontsize=8)
        axr.text(0.02, 0.85, f"{w['branch']}  PE {w['pe']:.0f}  raw non-zero {w['n_raw_nonzero']}/{w['n']}", transform=axr.transAxes, fontsize=8, color=rp.INK2)
        if r == len(order) - 1:
            axr.set_xlabel("t - flash time [us]"); axd.set_xlabel("t - flash time [us]")
        axr.set_xlim(t[0], t[-1]); axd.set_xlim(t[0], t[-1])
    rp.finish(fig, a.out or f"{rel}/figs/flash_waveforms_{a.event}_c{a.cluster}.png",
              "left: raw ADC (0 = no snippet there)   right: deconvolved [PE/sample] (+ decon_roi in aqua where the branch has it); dotted = OpHit peak times of this flash and channel")


if __name__ == "__main__":
    main()
