#!/usr/bin/env python3
"""plot_dqdx_vs_rr.py -- dQ/dx versus residual range of the stopping muons, against the
expectation table shipped with the release.

    python3 scripts/plot_dqdx_vs_rr.py [release_dir] [--out figs/dqdx_vs_rr.png]
    python3 scripts/plot_dqdx_vs_rr.py [release_dir] --event 029107_1135 --cluster 30 --out figs/dqdx_track.png

Population: T_stm rows with is_stm == 1; points: T_stm_pts rows with role == 1 (the muon chain),
q = dQ/dx in electrons per cm, rr = residual range in cm from the reconstructed stop.  Rows with
rr < 0 or q <= 0 (no profile / dead region) are dropped.  Expectation: dqdx_ref.json ("muon"),
the reconstruction's own table.  Left: the pooled 2-D density with the binned median; right:
the ratio to the expectation with a bootstrap-over-tracks band.  With --event/--cluster a single
track's profile is drawn instead.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stm_release as sr
import relplot as rp
import matplotlib.pyplot as plt

EDGES = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200])


def expectation(rel):
    rr, tab = sr.dqdx_ref(rel)
    return lambda x: np.interp(x, rr, tab["muon"], right=tab["muon"][-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release", nargs="?", default=None)
    ap.add_argument("--event", default=None, help="<run6>_<evtid> for a single track")
    ap.add_argument("--cluster", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rel = sr.release_dir(a.release)
    ref = expectation(rel)
    info = sr.event_info(sr.open_event(sr.event_files(rel)[0]))
    det = info["det"].upper()

    if a.event:
        path = f"{rel}/events/{a.event}/stm_michel_{info['det']}_{a.event}.root"
        ev = sr.open_event(path)
        c = sr.candidates(ev, ["cluster_id", "is_stm", "muon_len", "muon_ke_range", "plateau_med"])
        cid = a.cluster if a.cluster is not None else int(c["cluster_id"][c["is_stm"] == 1][0])
        p = sr.points(ev, cluster_id=cid, role=1)
        m = (p["rr"] >= 0) & (p["q"] > 0)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(p["rr"][m], p["q"][m] / 1e3, ".", color=rp.BLUE, ms=6, label="chain points (role 1)")
        xx = np.linspace(0, max(5, p["rr"][m].max()), 400)
        ax.plot(xx, ref(xx) / 1e3, color=rp.ORANGE, label="expectation (dqdx_ref muon)")
        i = int(np.flatnonzero(c["cluster_id"] == cid)[0])
        ax.set_title(f"{det} {a.event} cluster {cid}: dQ/dx vs residual range  (len {c['muon_len'][i]:.0f} cm, KE range {c['muon_ke_range'][i]:.0f} MeV, is_stm {c['is_stm'][i]})")
        ax.set_xlabel("residual range from the stop [cm]"); ax.set_ylabel("dQ/dx [10$^3$ e/cm]")
        ax.set_ylim(0, min(ax.get_ylim()[1], 400)); ax.legend()
        rp.finish(fig, a.out or f"{rel}/figs/dqdx_track_{a.event}_c{cid}.png", "T_stm_pts.q vs T_stm_pts.rr, role == 1")
        return

    rr_all, q_all, tid_all = [], [], []
    ntrack = 0
    for path in sr.event_files(rel):
        ev = sr.open_event(path)
        if ev["T_stm"].num_entries == 0:
            continue
        c = sr.candidates(ev, ["cluster_id", "is_stm"])
        for cid in c["cluster_id"][c["is_stm"] == 1]:
            p = sr.points(ev, cluster_id=int(cid), role=1)
            m = (p["rr"] >= 0) & (p["q"] > 0)
            if m.sum() < 5:
                continue
            rr_all.append(p["rr"][m]); q_all.append(p["q"][m]); tid_all.append(np.full(int(m.sum()), ntrack)); ntrack += 1
    rr = np.concatenate(rr_all); q = np.concatenate(q_all); tid = np.concatenate(tid_all)
    ratio = q / ref(rr)
    # binned median + bootstrap over tracks
    cen = 0.5 * (EDGES[1:] + EDGES[:-1]); med = np.full(len(cen), np.nan); lo = med.copy(); hi = med.copy(); medq = med.copy()
    rng = np.random.default_rng(30)
    for b in range(len(cen)):
        s = (rr >= EDGES[b]) & (rr < EDGES[b + 1])
        if s.sum() < 10:
            continue
        med[b] = np.median(ratio[s]); medq[b] = np.median(q[s])
        tr = np.unique(tid[s]); boots = []
        for _ in range(300):
            pick = rng.choice(tr, size=len(tr), replace=True)
            sel = np.isin(tid, pick) & s
            boots.append(np.median(ratio[sel]))
        lo[b], hi[b] = np.percentile(boots, [16, 84])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    h = ax1.hist2d(rr, q / 1e3, bins=[np.linspace(0, 100, 101), np.linspace(0, 250, 126)], cmap=rp.SEQ, cmin=1)
    fig.colorbar(h[3], ax=ax1, label="points")
    xx = np.linspace(0.1, 100, 400)
    ax1.plot(xx, ref(xx) / 1e3, color=rp.ORANGE, label="expectation")
    ok = np.isfinite(medq)
    ax1.plot(cen[ok], medq[ok] / 1e3, "o-", color=rp.BLUE, ms=5, lw=1.5, label="binned median")
    ax1.set_xlim(0, 100); ax1.set_ylim(0, 250)
    ax1.set_xlabel("residual range from the stop [cm]"); ax1.set_ylabel("dQ/dx [10$^3$ e/cm]")
    ax1.set_title(f"{det}: {ntrack} stopping muons (is_stm), {len(rr)} muon points"); ax1.legend(loc="upper right")
    ax2.fill_between(cen[ok], lo[ok], hi[ok], color=rp.BLUE, alpha=0.25, lw=0, label="16-84 % bootstrap over tracks")
    ax2.plot(cen[ok], med[ok], "o-", color=rp.BLUE, ms=5, lw=1.5, label="median dQ/dx / expectation")
    ax2.axhline(1, color=rp.ORANGE, lw=1.2)
    ax2.set_xscale("log"); ax2.set_xlim(0.8, 200); ax2.set_ylim(0.5, 1.5)
    ax2.set_xlabel("residual range [cm] (log)"); ax2.set_ylabel("measured / expected"); ax2.legend(loc="upper right")
    plateau = ratio[(rr >= 40) & (rr <= 60)]
    ax2.set_title(f"plateau 40-60 cm: median ratio {np.median(plateau):.3f}" if len(plateau) else "ratio")
    rp.finish(fig, a.out or f"{rel}/figs/dqdx_vs_rr.png",
              "T_stm_pts (role 1, rr >= 0, q > 0) of every T_stm row with is_stm == 1; expectation = dqdx_ref.json muon")


if __name__ == "__main__":
    main()
