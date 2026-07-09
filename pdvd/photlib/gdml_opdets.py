#!/usr/bin/env python3
"""Dump the 40 ProtoDUNE-VD optical-detector sensitive-volume world positions
from the simulation GDMLs:

  v4 (dunesw v10_05_00d00) -- the geometry both the PDFastSimANN v4 computable
     graph and the PDFastSimPVS voxel library were built on;
  v5 (dunesw v10_21_02d00, protodunevd_v5_ggd) -- the as-built update.  Its
     positions match the raw-data TTree opdet positions (pdvd-opdet-geom.json)
     EXACTLY; relative to v4 the Arapuca layout is y-mirrored (with ~3.5-7.5 cm
     as-built shifts) and several PMTs moved by up to ~37 cm.

Walks the TGeo tree for nodes/volumes named volOpDetSensitive_* in traversal
order and writes pdvd-{v4,v5}-gdml-opdets.json (mm, world frame).  Note the
world frame: volDetEnclosure sits at (20, 0, 149.65) cm in volWorld, so the
cathode plane (GDML x = -20 cm in the enclosure) lands at world x = 0.

Needs PyROOT: PYTHONPATH must include the ROOT lib dir, e.g.
  export PYTHONPATH=$(root-config --libdir):$PYTHONPATH
"""
import json
import os

import numpy as np

GDMLS = {
    "v4": "/cvmfs/dune.opensciencegrid.org/products/dune/dunecore/v10_05_00d00/gdml/protodunevd_v4_refactored.gdml",
    "v5": "/cvmfs/dune.opensciencegrid.org/products/dune/dunecore/v10_21_02d00/gdml/protodunevd_v5_ggd.gdml",
}
HERE = os.path.dirname(os.path.abspath(__file__))


def dump(version, gdml_path):
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kError + 1
    geo = ROOT.TGeoManager.Import(gdml_path)

    found = []

    def walk(node, mat):
        vol = node.GetVolume()
        if (vol.GetName().startswith("volOpDetSensitive")
                or node.GetName().startswith("volOpDetSensitive")):
            wor = np.zeros(3)
            ROOT.TGeoHMatrix(mat).LocalToMaster(np.zeros(3), wor)
            found.append({"node": node.GetName(), "volume": vol.GetName(),
                          "x": round(wor[0] * 10, 2), "y": round(wor[1] * 10, 2),
                          "z": round(wor[2] * 10, 2)})
            return
        for i in range(node.GetNdaughters()):
            d = node.GetDaughter(i)
            m = ROOT.TGeoHMatrix(mat)
            m.Multiply(d.GetMatrix())
            walk(d, m)

    top = geo.GetTopNode()
    walk(top, ROOT.TGeoHMatrix(top.GetMatrix()))
    assert len(found) == 40, f"expected 40 opdet sensitive volumes, got {len(found)}"

    out = os.path.join(HERE, f"pdvd-{version}-gdml-opdets.json")
    with open(out, "w") as f:
        json.dump({
            "comment": f"PDVD {version} GDML volOpDetSensitive_* world positions [mm], TGeo traversal order (NOT the LArSoft OpDet channel order -- see extract_photlib.py for the channel assignment)",
            "gdml": gdml_path,
            "opdets": found,
        }, f, indent=1)
    print(f"wrote {out} with {len(found)} opdets")


def main():
    for version, path in GDMLS.items():
        dump(version, path)


if __name__ == "__main__":
    main()
