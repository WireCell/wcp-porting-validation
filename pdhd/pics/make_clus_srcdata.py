#!/usr/bin/env python3
"""Extract the committed slim npz inputs for the PD-HD clustering+Q/L insets.

Counterpart of the (ad-hoc, uncommitted) extraction that produced the PD-VD
clus_chain_src npz — committed here so the PD-HD inset data provenance is
reproducible.  Reads the canonical PD-HD hand-scan reference event
(run 029107 evt 983, work/029107_0/) and writes:

  clus_chain_src/event3d_clusters_983.npz
      x,y,z,q,cid (+ run,event) — the all-TPC `clustering-global` Bee cloud
      (T0-corrected x_t0cor coordinates) from mabc-all-apa.zip.

  clus_chain_src/qlmatch_flash78.npz
      ch,meas,pred,perr,masked (+ ks,chi2,ndf,total_PE,gid,run,event) — the
      measured vs predicted light pattern over the 160 photon detectors for
      the brightest cleanly-matched (auto_selected) flash, gid 1000078
      (displayed as flash 78; the >=1e6 gid is the shared-flash drift-side
      copy), from calib-evt983.json.

Inputs are pipeline products of our own runs (work/029107_0), read-only.
"""
import io
import json
import os
import zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDHD = os.path.dirname(HERE)
WORK = os.path.join(PDHD, "work", "029107_0")
SRC = os.path.join(HERE, "clus_chain_src")

FLASH_GID = 1000078
# static dead-PD mask (cfg/pgrapher/experiment/pdhd/qlmatching.jsonnet ch_mask)
CH_MASK = [3, 86, 87, 97, 107, 116, 117]


def extract_event3d(out="event3d_clusters_983.npz"):
    z = zipfile.ZipFile(os.path.join(WORK, "mabc-all-apa.zip"))
    name = [n for n in z.namelist() if n.endswith("-clustering-global.json")][0]
    cg = json.loads(z.read(name))
    x = np.asarray(cg["x"], dtype=np.float32)
    y = np.asarray(cg["y"], dtype=np.float32)
    zz = np.asarray(cg["z"], dtype=np.float32)
    q = np.asarray(cg["q"], dtype=np.float32)
    cid = np.asarray(cg["cluster_id"], dtype=np.int32)
    keep = np.abs(x) < 400.0   # drop any unmatched-cluster sentinels
    np.savez_compressed(
        os.path.join(SRC, out), x=x[keep], y=y[keep], z=zz[keep], q=q[keep],
        cid=cid[keep], run=int(cg["runNo"]), event=int(cg["eventNo"]))
    print("wrote", out, "npts", int(keep.sum()),
          "nclus", len(np.unique(cid[keep])))


def extract_qlmatch(out="qlmatch_flash78.npz"):
    d = json.load(open(os.path.join(WORK, "calib-evt983.json")))
    fl = {f["gid"]: f for f in d["flashes"]}
    b = max((b for b in d["bundles"]
             if b.get("auto_selected") and b["flash_gid"] == FLASH_GID),
            key=lambda b: b["strength"])
    f = fl[FLASH_GID]
    n = len(f["pe"])
    ch = np.arange(n)
    masked = np.isin(ch, CH_MASK)
    np.savez_compressed(
        os.path.join(SRC, out), ch=ch,
        meas=np.asarray(f["pe"], dtype=np.float32),
        pred=np.asarray(b["pred_pe"], dtype=np.float32),
        perr=np.asarray(f["pe_err"], dtype=np.float32),
        masked=masked, ks=float(b["ks_dis"]), chi2=float(b["chi2"]),
        ndf=int(b["ndf"]), total_PE=float(f["total_PE"]),
        gid=int(FLASH_GID % 1000000), run=29107, event=983)
    print("wrote", out, "flash gid", FLASH_GID, "totPE %.0f" % f["total_PE"],
          "ks %.3f chi2/ndf %.1f" % (b["ks_dis"],
                                     b["chi2"] / max(1, b["ndf"])))


def main():
    os.makedirs(SRC, exist_ok=True)
    extract_event3d()
    extract_qlmatch()


if __name__ == "__main__":
    main()
