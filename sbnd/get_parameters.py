"""
Print SBND PDS optical detector efficiencies and channel mask.

Sources:
  - PD type/TPC map: pds_map_withcoords.json (from WireCell light simulation)
  - Efficiencies:    sbndcode/OpDetSim/digi_pmt_sbnd.fcl
                     sbndcode/OpDetSim/digi_arapuca_sbnd.fcl
  - Channel mask:    sbnd_data/CalibrationDatabase/pds_calibration.db
"""

import json
import sqlite3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
## can also be found generically in sbndcode/OpDetSim/sbnd_pds_mapping.json, 
## but without coords

PDS_MAP = "/exp/sbnd/data/users/lynnt/wirecell/light/pds_map_withcoords.json"
CALIB_DB = (
    "/cvmfs/sbnd.opensciencegrid.org/products/sbnd/sbnd_data"
    "/v01_42_00/CalibrationDatabase/pds_calibration.db"
)

# ---------------------------------------------------------------------------
# Efficiencies (from sbndcode fcl files)
# ---------------------------------------------------------------------------
# PMT coated — digi_pmt_sbnd.fcl
PMT_COATED_VUV = {0: 0.0392, 1: 0.0392}
PMT_COATED_VIS = {0: 0.0260, 1: 0.0260}

# PMT uncoated — digi_pmt_sbnd.fcl
PMT_UNCOATED    = {0: 0.0357, 1: 0.0357}

# X-ARAPUCA — digi_arapuca_sbnd.fcl
XARAPUCA_VUV_VUV = 0.01752   # VUV X-ARAPUCA efficiency to VUV light
XARAPUCA_VUV_VIS = 0.00271   # VUV X-ARAPUCA efficiency to visible light
XARAPUCA_VIS_EFF = 0.01264   # VIS X-ARAPUCA efficiency


def build_efficiency_arrays(pds_map):
    vuv, vis = [], []
    for ch in pds_map:
        pd_type = ch["pd_type"]
        tpc     = ch["tpc"]
        if pd_type == "xarapuca_vis":
            vuv.append(0.0)
            vis.append(XARAPUCA_VIS_EFF)
        elif pd_type == "xarapuca_vuv":
            vuv.append(XARAPUCA_VUV_VUV)
            vis.append(XARAPUCA_VUV_VIS)
        elif pd_type == "pmt_coated":
            vuv.append(PMT_COATED_VUV[tpc])
            vis.append(PMT_COATED_VIS[tpc])
        elif pd_type == "pmt_uncoated":
            vuv.append(0.0)
            vis.append(PMT_UNCOATED[tpc])
        else:
            raise ValueError(f"Unknown pd_type: {pd_type!r} for channel {ch}")
    return vuv, vis


def get_channel_mask(db_path):
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT channel, on_pmt, reconstruct_channel "
        "FROM pds_calibration_data "
        "WHERE __iov_id = (SELECT iov_id FROM pds_calibration_iovs WHERE active = 1) "
        "  AND (on_pmt = 0 OR reconstruct_channel = 0);"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def format_array(values, fmt=".5f"):
    return "[" + ", ".join(f"{v:{fmt}}" for v in values) + "]"


def main():
    # --- efficiencies -------------------------------------------------------
    with open(PDS_MAP) as f:
        pds_map = json.load(f)

    vuv_eff, vis_eff = build_efficiency_arrays(pds_map)

    sbnd_data_version = CALIB_DB.split("/sbnd_data/")[1].split("/")[0]
    print(f"sbnd_data version: {sbnd_data_version}")
    print(f"Number of optical detectors: {len(vuv_eff)}\n")

    print("Efficiency constants (from sbndcode fcl files):")
    print(f"  PMT coated   VUV: tpc0={PMT_COATED_VUV[0]}, tpc1={PMT_COATED_VUV[1]}")
    print(f"  PMT coated   VIS: tpc0={PMT_COATED_VIS[0]}, tpc1={PMT_COATED_VIS[1]}")
    print(f"  PMT uncoated    : tpc0={PMT_UNCOATED[0]}, tpc1={PMT_UNCOATED[1]}")
    print(f"  X-ARAPUCA VUV->VUV: {XARAPUCA_VUV_VUV}")
    print(f"  X-ARAPUCA VUV->VIS: {XARAPUCA_VUV_VIS}")
    print(f"  X-ARAPUCA VIS     : {XARAPUCA_VIS_EFF}")
    print()

    print("OpDetVUVEfficiencies =")
    print(format_array(vuv_eff))
    print()

    print("OpDetVISEfficiencies =")
    print(format_array(vis_eff))
    print()

    # --- channel mask -------------------------------------------------------
    rows = get_channel_mask(CALIB_DB)
    masked_channels = [r[0] for r in rows]

    print("Channel mask (on_pmt == 0 OR reconstruct_channel == 0):")
    print(f"  {len(masked_channels)} channels masked")
    print(f"  ch_mask: {masked_channels}")


if __name__ == "__main__":
    main()
