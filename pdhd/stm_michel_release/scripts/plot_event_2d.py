#!/usr/bin/env python3
"""plot_event_2d.py -- the 2-D charge picture of one STM candidate in the three wire planes.

    python3 scripts/plot_event_2d.py [release_dir] --event 029107_1135 [--cluster 30] [--zoom 60] [--out ...]

Per plane (U, V, W): the fitted 2-D charge cells of the candidate cluster (T_stm_2d: colour =
measured charge, wire-rank vs tick), the PR-fit trajectory projected into the plane
(T_stm_fit pu/pv/pw vs pt), and the Michel / gamma cells of T_michel_2d (role 3 / 4) outlined.
With --zoom N only the region within N wires and N*nticks_per_slice ticks of the Michel cells
(or of the last muon point when there is no Michel) is drawn.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stm_release as sr
import relplot as rp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release", nargs="?", default=None)
    ap.add_argument("--event", required=True)
    ap.add_argument("--cluster", type=int, default=None)
    ap.add_argument("--zoom", type=float, default=0, help="wires (and slices) around the Michel to draw; 0 = whole cluster")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rel = sr.release_dir(a.release)
    info = sr.event_info(sr.open_event(sr.event_files(rel)[0]))
    ev = sr.open_event(f"{rel}/events/{a.event}/stm_michel_{info['det']}_{a.event}.root")
    info = sr.event_info(ev); det = info["det"].upper(); nts = int(info["nticks_per_slice"])
    c = sr.candidates(ev, ["cluster_id", "is_stm", "michel_found", "michel_ke_q2d_region", "muon_len"])
    if a.cluster is None:
        a.cluster = int(c["cluster_id"][c["is_stm"] == 1][0])
    i = int(np.flatnonzero(c["cluster_id"] == a.cluster)[0])
    cells = sr.cells_2d(ev, cluster_id=a.cluster)
    fit = sr.fit_points(ev, cluster_id=a.cluster)
    mich = sr.michel_cells(ev, cluster_id=a.cluster)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    base = [0, 0, 0]
    for p, ax, lab, pk in zip(range(3), axes, "UVW", ["pu", "pv", "pw"]):
        s = cells["plane"] == p
        w = cells["chan_rank"][s]; t = cells["tick"][s]; q = cells["charge"][s]
        gb = cells["channel_rank_global"][s] - cells["chan_rank"][s]
        base[p] = int(gb[0]) if len(gb) else 0
        if len(w):
            sc = ax.scatter(w, t, c=np.clip(q, 1, None), s=9, marker="s", cmap=rp.SEQ, norm=LogNorm(vmin=max(1, np.percentile(q[q > 0], 5)) if (q > 0).any() else 1, vmax=max(2, q.max())), lw=0)
            fig.colorbar(sc, ax=ax, label="cell charge [e]")
        fw = fit[pk] - base[p]; ft = fit["pt"] * nts
        ax.plot(fw, ft, "-", color=rp.ORANGE, lw=1.2, label="PR fit trajectory")
        if mich:
            for role, col, name in ((3, rp.RED, "Michel cells (role 3)"), (4, rp.MAGENTA, "gamma cells (role 4)")):
                m = (mich["plane"] == p) & (mich["role"] == role)
                if m.any():
                    # T_michel_2d.wire is the wire index in its (apa, face); T_stm_2d uses the global plane rank.
                    # Match through the physical channel id so both live on the same axis.
                    ch2rank = {int(ch): int(r) for ch, r in zip(cells["channel"][s], w)}
                    ranks = np.array([ch2rank.get(int(ch), -1) for ch in mich["channel"][m]])
                    ok = ranks >= 0
                    ax.plot(ranks[ok], mich["time"][m][ok], "s", mfc="none", mec=col, ms=5, mew=1.0, label=name)
        if a.zoom > 0:
            m3 = (mich["plane"] == p) & np.isin(mich["role"], [3, 4]) if mich else np.zeros(0, bool)
            if mich and m3.any():
                ch2rank = {int(ch): int(r) for ch, r in zip(cells["channel"][s], w)}
                rr = np.array([ch2rank.get(int(ch), -1) for ch in mich["channel"][m3]]); rr = rr[rr >= 0]
                cw, ct = (np.median(rr) if len(rr) else fw[-1]), np.median(mich["time"][m3])
            else:
                cw, ct = fw[-1], ft[-1]
            ax.set_xlim(cw - a.zoom, cw + a.zoom); ax.set_ylim(ct - a.zoom * nts, ct + a.zoom * nts)
        ax.set_xlabel(f"plane {lab} wire rank (T_stm_2d.chan_rank = T_stm_fit.{pk} - {base[p]})"); ax.set_ylabel("tick")
        ax.set_title(f"plane {lab}")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{det} {a.event} cluster {a.cluster}: is_stm {c['is_stm'][i]}, Michel {c['michel_found'][i]} ({c['michel_ke_q2d_region'][i]:.1f} MeV), muon length {c['muon_len'][i]:.0f} cm")
    rp.finish(fig, a.out or f"{rel}/figs/event_2d_{a.event}_c{a.cluster}{'_zoom' if a.zoom else ''}.png", "T_stm_2d cells (colour), T_stm_fit projection (orange), T_michel_2d role 3/4 (outlined)")


if __name__ == "__main__":
    main()
