#!/usr/bin/env python3
"""Doc 26 step 5: A-C-A crosser closure vs drift velocity at ZERO extra
offset (PDVD_QL_EXTRA_OFFSET_US=0).

Same anchors as aca_positions.py (doc 23 §7c): W-plane erf tick endpoints of
the three crossers under the flash raw+metadata T0 (NO 13.507 us pull), swept
over candidate drift speeds.  For each v prints the anode-end u (vs the
shield; the containment floor is anode_ext1(-2) - margin) and the cathode-end
overshoot past the cathode surface (|x|=3; the ceiling is
u_cathode(336.91) + cathode_ext1).

Run from pdvd/docs/qlmatch:  python3 scripts/retune_positions.py
"""

W = 341.55
SHIELD = 339.91
UCATH = 336.91
CSURF = 3.0

OFF = {298609: {"bot": -2513.808001, "top": -2512.608},
       298651: {"bot": -2515.344,    "top": -2507.743999}}

# track: (evt, flash_raw_us, {half: (erf_anode_tick, erf_cath_tick|None)})
TRACKS = {
    "A": (298609, 3519.753, {"bot": (2018.9, 6615.5), "top": (2021.3, 6578.3)}),
    "B": (298609, 5399.991, {"bot": (5789.3, None),   "top": (5780.1, None)}),
    "C": (298651, 2801.95,  {"bot": (579.8, 5160.0),  "top": (592.2, 5157.5)}),
}

SWEEP_V = [0.148073, 0.14794, 0.14764, 0.147]   # cm/us


def xpos(half, tick, evt, raw, v):
    t_flash = raw + OFF[evt][half]          # NO pull
    d = (tick * 0.5 - t_flash) * v
    return (-W + d) if half == "bot" else (W - d)


def u_of(half, x):
    return (x + SHIELD) if half == "bot" else (SHIELD - x)


for v in SWEEP_V:
    print(f"== v = {v:.5f} cm/us (offset 0) ==")
    worst_anode, worst_cath = None, None
    for trk, (evt, raw, halves) in TRACKS.items():
        for half, (ta, tc) in halves.items():
            xa = xpos(half, ta, evt, raw, v)
            ua = u_of(half, xa)
            line = f"  {trk} {half}: anode u={ua:6.2f} (vs W {abs(xa)-W:+.2f})"
            if worst_anode is None or ua < worst_anode:
                worst_anode = ua
            if tc is not None:
                xc = xpos(half, tc, evt, raw, v)
                uc = u_of(half, xc)
                over = CSURF - abs(xc)   # >0 = past the cathode surface
                line += (f" | cath u={uc:7.2f} past-surface {over:+.2f}"
                         f" (needs ext1 >= {uc - UCATH:+.2f})")
                if worst_cath is None or uc > worst_cath:
                    worst_cath = uc
            print(line)
    print(f"  -> floor needed: anode_ext1 - margin <= {worst_anode:.2f} "
          f"(margin >= {-2.0 - worst_anode:.2f} at anode_ext1 = -2)")
    print(f"  -> ceiling needed: cathode_ext1 >= {worst_cath - UCATH:.2f} cm\n")
