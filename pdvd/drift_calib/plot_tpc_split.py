#!/usr/bin/env python
"""Per-TPC (top vs bottom) span distributions at a clean cathode window."""
import glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/drift_calib")
import calib_drift_velocity as C

WORK = "/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/pdvd/work"
V_RECO, D = 1.57, C.D
OUT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/drift_calib/drift_velocity_tpc_split.png"

rows = []
for zp in sorted(glob.glob(os.path.join(WORK, "*", "mabc-all-apa.zip"))):
    try: rows.extend(C.event_clusters(zp, 50))
    except Exception: pass

edges = np.arange(320, 372, 3.0)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, cath_hi, tag in ((axes[0], 25.0, "loose window [-5,25] (overshoot-contaminated)"),
                         (axes[1], 8.0, "clean window [-5,8] (stops at cathode)")):
    for grp, name, col in ((0, "bottom (anodes 0-3)", "C0"), (1, "top (anodes 4-7)", "C1")):
        full = [r for r in rows if r["grp"] == grp and C.is_full_crosser(r, 330.0, -5.0, cath_hi)]
        sp = np.array([r["span"] for r in full])
        pu = C.span_pileup(sp)
        ax.hist(sp, bins=edges, histtype="step", lw=1.8, color=col,
                label=f"{name}: N={len(sp)}, v={V_RECO*D/pu:.3f}")
    ax.axvline(D, ls="--", color="g", lw=1, label=f"D={D:.1f} (v=1.570)")
    ax.axvline(339.0, ls=":", color="k", lw=1, label="span 339 (v=1.568)")
    ax.set_xlabel("full-crosser drift x-span [cm]"); ax.set_title(tag, fontsize=9)
    ax.legend(fontsize=7.5)
axes[0].set_ylabel("clusters")
fig.suptitle(f"PDVD drift velocity, top vs bottom TPC (data @ v_reco={V_RECO}, D={D:.1f} cm)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("wrote", OUT)
