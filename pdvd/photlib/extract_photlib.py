#!/usr/bin/env python3
"""Extract the ProtoDUNE-VD v4 voxel photon libraries (the PDFastSimPVS fallback
of the dunesw v10_05_00d00 refactored chain) into npy arrays and pin down the
library-channel <-> detector mapping.

Library files (StashCache cvmfs), TTree PhotonLibraryData(Voxel,OpChannel,Visibility):
  libext_protodunevd_v4_{Ar,Xe}_Baseline_v09_69_00d00_5e7_25x25x25_landau_20231216.root

Voxel grid (PhotonVisibilityService UseCryoBoundary = cryostat bounding box in
world coordinates; larsim PhotonVoxelDef, x-fastest voxel id):
  protodunevd_v4_refactored.gdml: box "Cryostat" 790 x 854.8 x 854.8 cm,
  volCryostat centered in volDetEnclosure, enclosure at (20, 0, 149.65) cm in
  volWorld -> cryostat center (200, 0, 1496.5) mm, 25x25x25 voxels.

Channel mapping is established through three independent sources and cross-gated:
  1. GDML sensitive-volume positions (pdvd-v4-gdml-opdets.json, from gdml_opdets.py)
     assigned to library channels by nearest hot-voxel centroid (one-to-one).
  2. The official duneprototypes PDVD_PDS_Mapping_v04152025.json: geometry channel
     (= library OpChannel) -> pd_type/wls/eff + offline DAPHNE channels.
  3. Our cfg pdvd-opch-map.json: offline DAPHNE channel -> WCT flash-chain OpDet.
Gates: (a) centroid<->GDML assignment is bijective within one voxel diagonal;
(b) the GDML volume type of the assigned position agrees with the official
pd_type for every channel; (c) each official channel's offline channels all land
on a single WCT OpDet.

Known finding (reported, not gated): the raw-data TTree opdet positions
(pdvd-opdet-geom.json) match the v5 as-built GDML (protodunevd_v5_ggd) EXACTLY;
relative to v4 the Arapuca layout is y-MIRRORED (plus ~3.5-7.5 cm as-built
shifts) and several PMTs moved by up to ~37 cm.  So the v4 voxel library and
the v4 ANN describe a y-mirrored Arapuca layout; a model for real data should
come from the v5 ANN (protodune_vd_v5_128nm_tf2.6) -- see sample_ann.py.

Outputs:
  work/photlib_vis_{Ar,Xe}.npy   float32 [15625, 40]   (not committed)
  work/photlib_grid.json         grid origin/step/shape [mm] + provenance
  pdvd-photlib-chanmap.json      per-channel mapping table (committed)
"""
import argparse
import json
import os

import numpy as np

CVMFS_LIB = "/cvmfs/dune.osgstorage.org/pnfs/fnal.gov/usr/dune/persistent/stash/PhotonPropagation/LibraryData"
LIBS = {
    "Ar": "libext_protodunevd_v4_Ar_Baseline_v09_69_00d00_5e7_25x25x25_landau_20231216.root",
    "Xe": "libext_protodunevd_v4_Xe_Baseline_v09_69_00d00_5e7_25x25x25_landau_20231216.root",
}
PDS_MAP = "/cvmfs/dune.opensciencegrid.org/products/dune/duneprototypes/v10_09_00d00/config_data/PDVD_PDS_Mapping_v04152025.json"

TOOLKIT_CFG = os.environ.get(
    "PDVD_CFG_DIR",
    "/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/protodunevd")

# Grid in mm, world frame (see module docstring for GDML provenance).
N = (25, 25, 25)
CRYO_CENTER = np.array([200.0, 0.0, 1496.5])       # mm
CRYO_DIMS = np.array([7900.0, 8548.0, 8548.0])     # mm
STEP = CRYO_DIMS / np.array(N)
ORIGIN = CRYO_CENTER - CRYO_DIMS / 2.0             # lower corner
NVOX = N[0] * N[1] * N[2]
NCH = 40

HERE = os.path.dirname(os.path.abspath(__file__))


def gdml_type(volname):
    if "ArapucaDouble" in volname:
        return "Cathode"
    if "ArapucaLat" in volname:
        return "Membrane"
    if "PMT" in volname:
        return "PMT"
    raise ValueError(volname)


def voxel_centers():
    """[NVOX, 3] voxel centers in mm, larsim x-fastest id convention."""
    ids = np.arange(NVOX)
    ix = ids % N[0]
    iy = (ids // N[0]) % N[1]
    iz = ids // (N[0] * N[1])
    return ORIGIN + (np.stack([ix, iy, iz], axis=1) + 0.5) * STEP


def load_library(path):
    import uproot
    t = uproot.open(path)["PhotonLibraryData"]
    a = t.arrays(["Voxel", "OpChannel", "Visibility"], library="np")
    vis = np.zeros((NVOX, NCH), dtype=np.float32)
    vis[a["Voxel"], a["OpChannel"]] = a["Visibility"]
    return vis


def hot_centroid(vis_ch, centers, ntop=8):
    top = np.argsort(vis_ch)[-ntop:]
    w = vis_ch[top]
    return (centers[top] * w[:, None]).sum(axis=0) / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.join(HERE, "work"))
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    centers = voxel_centers()
    vox_diag = float(np.linalg.norm(STEP))

    with open(os.path.join(HERE, "pdvd-v4-gdml-opdets.json")) as f:
        gdml = json.load(f)["opdets"]
    gpos = np.array([[o["x"], o["y"], o["z"]] for o in gdml])
    with open(os.path.join(HERE, "pdvd-v5-gdml-opdets.json")) as f:
        v5pos = np.array([[o["x"], o["y"], o["z"]] for o in json.load(f)["opdets"]])

    with open(PDS_MAP) as f:
        pds = {r["channel"]: r for r in json.load(f)}

    with open(os.path.join(TOOLKIT_CFG, "pdvd-opch-map.json")) as f:
        opch2det = {c["opch"]: c["opdet"] for c in json.load(f)["channels"]}
    with open(os.path.join(TOOLKIT_CFG, "pdvd-opdet-geom.json")) as f:
        jjo_pos = {o["opdet"]: np.array([o["x"], o["y"], o["z"]])
                   for o in json.load(f)["opdets"]}

    grid_meta = {
        "comment": "PDVD v4 photon-library voxel grid, mm, world frame; x-fastest voxel id",
        "gdml": "protodunevd_v4_refactored.gdml (dunesw v10_05_00d00)",
        "n": list(N),
        "origin_mm": ORIGIN.tolist(),
        "step_mm": STEP.tolist(),
        "libraries": {k: os.path.join(CVMFS_LIB, v) for k, v in LIBS.items()},
    }
    with open(os.path.join(args.workdir, "photlib_grid.json"), "w") as f:
        json.dump(grid_meta, f, indent=1)

    vis = {}
    for flavor, fname in LIBS.items():
        vis[flavor] = load_library(os.path.join(CVMFS_LIB, fname))
        np.save(os.path.join(args.workdir, f"photlib_vis_{flavor}.npy"), vis[flavor])
        v = vis[flavor]
        print(f"{flavor}: filled voxels {np.count_nonzero(v.any(axis=1))}/{NVOX}, "
              f"vis range [{v[v>0].min():.2e}, {v.max():.2e}]")

    # --- gate (a): assign library channels to GDML opdets by centroid (Ar) ---
    cents = np.array([hot_centroid(vis["Ar"][:, ch], centers) for ch in range(NCH)])
    dist = np.linalg.norm(cents[:, None, :] - gpos[None, :, :], axis=2)
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(dist)
    assert (rows == np.arange(NCH)).all()
    dmax = dist[rows, cols].max()
    print(f"\ncentroid->GDML assignment: bijective, max distance {dmax:.0f} mm "
          f"(gate: < voxel diagonal {vox_diag:.0f} mm)")
    if dmax >= vox_diag:
        raise SystemExit("gate (a) FAILED")

    # --- gates (b) + (c), assemble the mapping table ---
    table = []
    nyflip = 0
    for ch in range(NCH):
        g = gdml[cols[ch]]
        gt, ot = gdml_type(g["node"]), pds[ch]["pd_type"]
        if gt != ot:
            raise SystemExit(f"gate (b) FAILED: ch {ch} GDML type {gt} != official {ot}")
        offline = [h["OfflineChannel"] for h in pds[ch]["HardwareChannel"]]
        dets = {opch2det[o] for o in offline if o in opch2det}
        if len(dets) > 1:
            raise SystemExit(f"gate (c) FAILED: ch {ch} offline {offline} -> opdets {dets}")
        our = dets.pop() if dets else None
        row = {
            "channel": ch,
            "pd_type": ot,
            "wls": pds[ch]["wls"],
            "eff_Ar": pds[ch]["eff_Ar"],
            "eff_Xe": pds[ch]["eff_Xe"],
            "offline": offline,
            "wct_opdet": our,
            "gdml_node": g["node"],
            "x": g["x"], "y": g["y"], "z": g["z"],
        }
        if our is not None and our in jjo_pos:
            d0 = float(np.linalg.norm(jjo_pos[our] - gpos[cols[ch]]))
            dmir = float(np.linalg.norm(jjo_pos[our] * [1, -1, 1] - gpos[cols[ch]]))
            dv5 = float(np.linalg.norm(v5pos - jjo_pos[our], axis=1).min())
            row["v4_vs_data_mm"] = round(d0, 1)
            row["v4_vs_data_ymirror_mm"] = round(dmir, 1)
            row["v5_vs_data_mm"] = round(dv5, 1)
            nyflip += (dmir < d0 - 1.0)
        table.append(row)
        print(f"ch {ch:2d} {ot:8s} {g['node']:38s} ({g['x']:8.1f},{g['y']:8.1f},{g['z']:8.1f})"
              f" offline={offline} wct_opdet={row['wct_opdet']}"
              + (f" d(v4,data)={row.get('v4_vs_data_mm','-')}"
                 f" ymirror={row.get('v4_vs_data_ymirror_mm','-')}"
                 f" d(v5,data)={row.get('v5_vs_data_mm','-')}"
                 if "v4_vs_data_mm" in row else " (dead)"))

    with open(os.path.join(HERE, "pdvd-photlib-chanmap.json"), "w") as f:
        json.dump({
            "comment": "PDVD v4 photon-library/G4 channel -> detector mapping. channel = library OpChannel = official PDS-map geometry channel; x/y/z = v4 GDML world position [mm] of the sensitive volume; wct_opdet = OpDet index of our flash chain (pdvd-opch-map.json), null for dead channels; eff_* = official sim efficiencies (PDVD_PDS_Mapping_v04152025). NOTE: the raw-data positions (pdvd-opdet-geom.json) match the v5 as-built GDML exactly (v5_vs_data_mm ~ 0); relative to v4 the Arapuca layout is y-mirrored (v4_vs_data_ymirror_mm << v4_vs_data_mm) and several PMTs moved up to ~370 mm.",
            "sources": {"gdml": gdml[0] and "pdvd-v4-gdml-opdets.json",
                        "pds_mapping": PDS_MAP,
                        "opch_map": os.path.join(TOOLKIT_CFG, "pdvd-opch-map.json")},
            "channels": table,
        }, f, indent=1)
    v5d = [r["v5_vs_data_mm"] for r in table if "v5_vs_data_mm" in r]
    print(f"\nwrote pdvd-photlib-chanmap.json;  v4-Arapuca y-mirror evidence: {nyflip} "
          f"channels prefer mirrored data y;  data vs v5 GDML: max {max(v5d):.1f} mm "
          f"over {len(v5d)} live channels")

    ar, xe = vis["Ar"], vis["Xe"]
    both = (ar > 0) & (xe > 0)
    r = np.corrcoef(np.log(ar[both]), np.log(xe[both]))[0, 1]
    print(f"Ar-Xe log-visibility correlation over {both.sum()} filled entries: {r:.3f}")


if __name__ == "__main__":
    main()
