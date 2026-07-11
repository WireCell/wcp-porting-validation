#!/usr/bin/env python3
"""Evidence figures + numbers for docs/pdvd-qlmatching-debug-evt298567.md
(run 039252 evt 298567, ql_scan flash gid 61 at -127.25 us).

Figure 1  ophit timeline around the flash: cathode vs wall-XA vs PMT hits,
          flash membership, and the +0.76 us burst culled by remove_late_light.
Figure 2  cathode measured vs predicted (c134 + c62 summed, Xe and Ar
          libraries), with the opdet-10 saturation veto marked.

Run from pdvd/:  python3 ql_light_calib/debug_qlmatch_evt298567.py
Inputs: work/039252_0/{calib,128nm-calib}-evt298567.json
        work/039252_light298567/opflash_pdvd-wct.tar.gz
Output: docs/pds/qlmatch-evt298567-flash61-{timeline,cathode}.png
"""
import json
import os
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PDVD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL = os.path.join(PDVD, "work", "039252_0", "calib-evt298567.json")
CAL_AR = os.path.join(PDVD, "work", "039252_0", "128nm-calib-evt298567.json")
TAR = os.path.join(PDVD, "work", "039252_light298567", "opflash_pdvd-wct.tar.gz")
OUT = os.path.join(PDVD, "docs", "pds")

GID = 61          # ql_scan display flash number == calib gid
T_LIGHT = 2390.079  # us, light-chain axis (= calib -127.25 us - offset_bot -2517.328)


def load_ophits():
    with tarfile.open(TAR) as tf:
        for m in tf.getmembers():
            if m.name.endswith("_2_array.npy"):
                import io
                return np.load(io.BytesIO(tf.extractfile(m).read()))
    raise RuntimeError("ophits tensor not found in " + TAR)


def fig_timeline(hits):
    pk = hits[:, 1] / 1e3
    pe = hits[:, 5]
    ch = hits[:, 0].astype(int)
    fid = hits[:, 7].astype(int)
    sel = (pk > 2384) & (pk < 2396)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    pops = [("cathode XA (stream)", (ch >= 1000) & (ch < 2000), 2),
            ("membrane XA (snippet)", (ch >= 2000) & (ch < 3000), 1),
            ("PMT (snippet)", ch >= 3000, 0)]
    for name, m, row in pops:
        mm = m & sel
        infl = mm & (fid >= 0)
        orph = mm & (fid < 0)
        ax.scatter(pk[infl], np.full(infl.sum(), row) + 0.12,
                   s=np.sqrt(pe[infl]) * 14, marker="o", alpha=.75,
                   label=None, color="C0")
        ax.scatter(pk[orph], np.full(orph.sum(), row) - 0.12,
                   s=np.sqrt(pe[orph]) * 14, marker="x", alpha=.85,
                   label=None, color="C3")
        ax.text(2384.1, row + 0.3, name, fontsize=9)
    ax.axvline(2386.939, color="gray", ls=":", lw=1)
    ax.text(2386.94, 2.75, "flash 127\n(479 PE)", fontsize=8, ha="center")
    ax.axvline(T_LIGHT, color="k", ls="--", lw=1)
    ax.text(T_LIGHT, 2.75, "flash 128 = gid 61\n(3949 PE, -127.25 us)",
            fontsize=8, ha="center")
    ax.annotate("burst +0.76 us: 454 PE / 15 PDs,\nculled by remove_late_light",
                xy=(2390.85, -0.15), xytext=(2392.3, -0.42),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8))
    ax.set_xlabel("light-chain time [us]  (charge axis = t - 2517.328)")
    ax.set_yticks([])
    ax.set_ylim(-0.7, 3.1)
    ax.scatter([], [], marker="o", color="C0", label="hit in a kept flash")
    ax.scatter([], [], marker="x", color="C3", label="hit in no flash (flash = -1)")
    ax.legend(loc="center right", fontsize=8)
    ax.set_title("run 039252 evt 298567: ophits around ql_scan flash 61 "
                 "(marker area ~ sqrt PE)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qlmatch-evt298567-flash61-timeline.png"), dpi=130)


def fig_cathode():
    dx = json.load(open(CAL))
    da = json.load(open(CAL_AR))
    f = [f for f in dx["flashes"] if f["gid"] == GID][0]

    def summed_pred(d):
        p = np.zeros(40)
        for b in d["bundles"]:
            if b["flash_gid"] == GID and str(b["main_cluster"]) in ("134", "4000062"):
                p += np.array(b["pred_pe"])
        return p

    px, pa = summed_pred(dx), summed_pred(da)
    m = np.array(f["pe"])
    idx = np.arange(4, 12)
    w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(idx - w, m[idx], w, label="measured", color="C0")
    ax.bar(idx, px[idx], w, label="pred c134+c62, Xe 175nm", color="C1")
    ax.bar(idx + w, pa[idx], w, label="pred c134+c62, Ar 128nm", color="C2")
    ax.annotate("opdet 10: DAPHNE 1040/1041 rail\nat 16383 ADC -> saturation veto",
                xy=(10 - w, 30), xytext=(8.0, 900), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=.8))
    labels = {4: "(124,259)", 5: "(-213,259)", 6: "(290,187)", 7: "(-47,187)",
              8: "(43,112)", 9: "(-213,112)", 10: "(209,41)", 11: "(-128,41)"}
    ax.set_xticks(idx)
    ax.set_xticklabels(["ch%d\n%s" % (i, labels[i]) for i in idx], fontsize=8)
    ax.set_xlabel("cathode X-ARAPUCA opdet (y,z) [cm];  pair crosses cathode at (116,195)")
    ax.set_ylabel("PE")
    ax.legend(fontsize=8)
    ax.set_title("evt 298567 flash gid 61: cathode pattern, measured vs library prediction")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qlmatch-evt298567-flash61-cathode.png"), dpi=130)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_timeline(load_ophits())
    fig_cathode()
    print("wrote", os.path.join(OUT, "qlmatch-evt298567-flash61-{timeline,cathode}.png"))
