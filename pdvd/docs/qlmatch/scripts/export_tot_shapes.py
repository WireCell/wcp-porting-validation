#!/usr/bin/env python3
"""doc qlmatch/31: write the per-channel ToT shape parameters (saturation_tot_study.py --spe v2 --shape cache)
as the OpDecon tot_shape_file.  Degenerate fits (prompt fraction >= 0.99, a template-only shape) are left out so
those channels keep the twoside fill.

    python3 scripts/export_tot_shapes.py [/home/xqian/tmp/sat_tot_v2/shape.npz] [out.json]
"""
import json
import sys

import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "/home/xqian/tmp/sat_tot_v2/shape.npz"
dst = sys.argv[2] if len(sys.argv) > 2 else \
    "/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/protodunevd/pdvd-tot-shapes-v2.json"
z = np.load(src)
chans, vals, dropped = [], [], []
for c, p in zip(z["chan"], z["par"]):
    if p[0] >= 0.99:
        dropped.append(int(c))
        continue
    chans.append(int(c))
    vals.append([round(float(x), 6) for x in p])
doc = {
    "comment": "OpDecon saturation_repair_mode 'tot' bright-pulse shape per cathode channel: "
               "SPE template (pdvd-spe-templates-v2.json) (x) [ff delta + (1-ff)(bi exp(-t/tau_i)/tau_i + "
               "(1-bi) exp(-t/tau_s)/tau_s)] (x) Gauss(sigma); ticks of 16 ns.",
    "source": "fit to the median shape of bright unrailed pulses of PDVD run 039252 (file 0), "
              "wcp-porting-img pdvd/docs/qlmatch/scripts/saturation_tot_study.py --spe v2 --shape (multi-start fit); docs qlmatch/30, 31",
    "params": ["ff", "bi", "tau_i", "tau_s", "sigma"],
    "channels": chans,
    "values": vals,
}
with open(dst, "w") as fh:
    json.dump(doc, fh, indent=1)
    fh.write("\n")
print(f"wrote {dst}: {len(chans)} channels; dropped (degenerate) {dropped}")
