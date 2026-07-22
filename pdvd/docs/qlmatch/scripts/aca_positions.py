#!/usr/bin/env python3
"""Position ladder at the anode and cathode for the three A-C-A crossers
(doc 23 section 7c): physical planes, QLMatching containment gates, erf-based
track-end positions under the two T0 conventions (flash raw+metadata vs
+13.507 us production pull), and the Bee-displayed imaged endpoints.

Run from pdvd/docs/qlmatch:  python3 scripts/aca_positions.py
Reads the ccprod mabc-all-apa.zip dumps of work/039252_{3,6}_ccprod.
Erf ticks: fit_endpoints_298609.py (track A/B) and doc 06 (track C).
"""
import json, zipfile, numpy as np, os

V = 0.148073          # cm/us (ccprod production drift speed)
W = 341.55            # collection-wire plane |x| [cm]
SHIELD = 339.91       # FV anode = shield plane |x| (DetectorVolumes inner bound)
UCATH = 336.91        # u_cathode (FV cathode |x| = 3.0)
CSURF = 3.0           # cathode drift-facing surface |x| (slab is |x|<3, center x=0)
PULL_US = 13.507      # PDVD_QL_EXTRA_OFFSET_US production pull
PULL_CM = PULL_US * V

# production gates: anode_ext1=-2 cm (C++ default), anode_ext1_margin=2.0 cm
# (runner PDVD_QL_ANODE_MARGIN_CM), cathode_ext1=2.0 cm (runner
# PDVD_QL_CATHODE_EXT1_CM). u = s*(x - anode_x), u=0 at the shield.
ANODE_FLOOR_U = -4.0          # first_u >  -4    -> |x| < 343.91
CATH_CEIL_U = UCATH + 2.0     # last_u  < 338.91 -> |x| >   1.00
ANODE_FLAG_OUT_U = 4.0        # anode flag window outer edge  -> |x| > 335.91
CATH_FLAG_IN_U = UCATH - 2.0  # cathode flag window inner edge -> |x| <   5.00

OFF = {298609: {"bot": -2513.808001, "top": -2512.608},
       298651: {"bot": -2515.344,    "top": -2507.743999}}

# track: (evt, work tag, flash_raw_us, {half: (bee cluster, erf_anode_tick, erf_cath_tick|None)})
TRACKS = {
    "A": (298609, "039252_3_ccprod", 3519.753,
          {"bot": (37, 2018.9, 6615.5), "top": (79, 2021.3, 6578.3)}),
    "B": (298609, "039252_3_ccprod", 5399.991,          # cathode side window-truncated
          {"bot": (50, 5789.3, None),   "top": (83, 5780.1, None)}),
    "C": (298651, "039252_6_ccprod", 2801.95,
          {"bot": (35, 579.8, 5160.0),  "top": (95, 592.2, 5157.5)}),
}

WORK = os.path.join(os.path.dirname(__file__), "..", "..", "..", "work")


def xpos(half, tick, evt, raw, pull):
    """physical x of a charge tick, T0-corrected with the flash (raw+metadata[+pull])."""
    t_flash_frame = raw + OFF[evt][half] + (PULL_US if pull else 0.0)
    d = (tick * 0.5 - t_flash_frame) * V          # drift depth from the W plane
    return (-W + d) if half == "bot" else (W - d)


def u_of(half, x):
    return (x + SHIELD) if half == "bot" else (SHIELD - x)


print(f"pull = {PULL_CM:.3f} cm;  planes |x|: shield {SHIELD}  W {W}  cathode surface {CSURF}")
print(f"gates |x|: anode floor {SHIELD - ANODE_FLOOR_U:.2f} (u=-4), "
      f"cathode ceiling {SHIELD - CATH_CEIL_U:.2f} (u={CATH_CEIL_U:.2f})")
print(f"flag windows |x|: anode > {SHIELD - ANODE_FLAG_OUT_U:.2f}, "
      f"cathode < {SHIELD - CATH_FLAG_IN_U:.2f}\n")

for trk, (evt, tag, raw, halves) in TRACKS.items():
    for half, (cid, ta, tc) in halves.items():
        for pull in (False, True):
            lab = "pull" if pull else "meta"
            xa = xpos(half, ta, evt, raw, pull)
            line = (f"{trk} {half} [{lab:4s}] anode-end x={xa:8.2f} u={u_of(half, xa):7.2f}"
                    f" (vs W {abs(xa) - W:+.2f}, floor margin {u_of(half, xa) - ANODE_FLOOR_U:+.2f})")
            if tc is not None:
                xc = xpos(half, tc, evt, raw, pull)
                uc = u_of(half, xc)
                line += (f" | cath-end x={xc:7.2f} u={uc:7.2f}"
                         f" (past surface {CSURF - abs(xc):+.2f}, ceiling margin {CATH_CEIL_U - uc:+.2f})")
            print(line)
    print()

print("Bee img-global + flash-shift (true flash, WITH pull) -- displayed endpoints:")
for trk, (evt, tag, raw, halves) in TRACKS.items():
    zf = zipfile.ZipFile(os.path.join(WORK, tag, "mabc-all-apa.zip"))
    d = json.loads(zf.read("data/0/0-img-global.json"))
    x_all, cid_all = np.asarray(d["x"]), np.asarray(d["cluster_id"])
    for half, (cid, ta, tc) in halves.items():
        xs = x_all[cid_all == cid]
        fold = raw + OFF[evt][half] + PULL_US
        xc = xs - fold * V if half == "bot" else xs + fold * V
        if half == "bot":
            tip, far = xc.min(), xc.max()
            n_out_anode, n_past_cath = (xc < -W).sum(), (xc > -CSURF).sum()
            deep = np.sort(xc)[-10:]
        else:
            tip, far = xc.max(), xc.min()
            n_out_anode, n_past_cath = (xc > W).sum(), (xc < CSURF).sum()
            deep = np.sort(xc)[:10]
        print(f"{trk} {half} c{cid}: n={len(xc):5d} anode-tip {tip:8.2f} cath-end {far:8.2f}"
              f" | pts beyond W face {n_out_anode:4d} | pts past cathode surface {n_past_cath:4d}")
        print(f"      deepest-10 cathode-side: {np.array2string(deep, precision=1)}")
