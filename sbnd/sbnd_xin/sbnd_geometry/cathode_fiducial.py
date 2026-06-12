#!/usr/bin/env python3
"""
SBND cathode-plane (CPA) structure-exclusion fiducial volume.

A *staging* description of the region occupied by the CPA mechanical structure,
approximated by simple, overlapping, axis-aligned boxes, per TPC, in the
wire-cell-toolkit coordinate system. Nothing here touches the toolkit; the boxes
are emitted as a paste-ready jsonnet snippet -- see cathode_fiducial.md.

Model (v2): the deep exclusion follows the **tube lattice between the CPA pads**,
NOT a uniform slab over the whole plane. The 16 mesh/foil pads are thin (reach
~0.6 cm); only the steel pipe lattice in the inter-pad gaps reaches ~2.7 cm, and
the knuckle joints reach ~4.1 cm. So:
  * a thin full-plane "pad" slab (default 0.6 cm; set pad=False to drop), plus
  * deep, narrow "tube" bars along the lattice lines, plus
  * deeper "knuckle" boxes at the four joints on the central vertical line.

Coordinate system (wire-cell-toolkit / sbnd_xin), units = cm:
  X = drift, cathode at X = 0.  TPC0 = East (X<0); TPC1 = West (X>0).
  Y = vertical, centered at 0.  Z = beam.
  GDML volTPCActive local z=0 -> toolkit z = 250.5 cm (detector center).

Run:  python3 cathode_fiducial.py
      -> writes cathode_fiducial.png, prints jsonnet, runs self-checks.
"""

from dataclasses import dataclass

MM = 0.1                 # mm -> cm
Z_CENTER_CM = 250.5      # volTPCActive local z=0 -> toolkit z (z_J = z_C + 291.75)

# --------------------------------------------------------------------------
# Structure constants (mm), cathode-centered. From sampling the v02_06 GDML CPA
# solids + the GDML pipe-frame union definitions. See cathode_fiducial.md.
# --------------------------------------------------------------------------
PAD_REACH_MM = 6.0       # thin pad slab depth (mesh 0.5 mm + foil at 6 mm)
PAD_YH_MM    = 2018.5    # pad-grid (foil) envelope, half-extent in y
PAD_ZH_MM    = 2528.0    # pad-grid (foil) envelope, half-extent in z

TUBE_REACH_MM = 27.0     # Oe54 pipe reach into drift (rmax = 27 mm, axis on cathode)
TUBE_HW_MM    = 27.0     # transverse half-width of a tube bar (pipe radius).
                         # NB inter-pad GAP half-width is ~61 mm if you want bars
                         # to fill the whole gap instead of just the pipe.

KNK_REACH_MM = 41.0      # knuckle joint reach into drift
KNK_HY_MM    = 50.0      # knuckle half-size in y (100 mm tall)
KNK_HZ_MM    = 65.0      # knuckle half-size in z (130 mm)

# Lattice line positions (mm, local frame). Pads sit at y=+-517.5,+-1552.5 and
# z=+-667,+-1942; the pipe lattice runs in the gaps between/around them.
HORIZ_Y_MM = [0.0, 1035.0, -1035.0, 2070.0, -2070.0]   # tubes running along z
VERT_Z_MM  = [0.0, 1275.0, -1275.0]                    # tubes running along y
KNK_Y_MM   = [517.5, -517.5, 1552.5, -1552.5]          # knuckles, all on z=0

# Bar long-extent (half), mm -- bars span the whole cathode plane in their long
# direction so every inter-pad gap is covered across all columns/rows.
PLANE_YH_MM = 2070.0
PLANE_ZH_MM = 2528.0


@dataclass
class Box:
    """Axis-aligned box in toolkit coords (cm)."""
    name: str
    xmin: float; xmax: float
    ymin: float; ymax: float
    zmin: float; zmax: float

    def contains(self, p):
        x, y, z = p
        return (self.xmin <= x <= self.xmax and
                self.ymin <= y <= self.ymax and
                self.zmin <= z <= self.zmax)

    def ray(self):
        return [[self.xmin, self.ymin, self.zmin],
                [self.xmax, self.ymax, self.zmax]]

    def kind(self):
        return self.name.split("_")[0].rstrip("0123456789+-")


# --- Per-TPC asymmetry hook -------------------------------------------------
# v02_06 realizes the CPA mirror-symmetric about X=0 (only volCPA_East placed;
# volTPCActive reused under volTPC_West with a pure 180-Y rotation), so by default
# both TPCs use the same reach. If a REAL cathode asymmetry is confirmed, give a
# feature a different reach here, e.g.  PER_TPC_REACH = {"knuckle": {0:41, 1:38}}.
PER_TPC_REACH = {}  # type: dict


def _xspan(tpc, reach_mm, cx_cm, feature=None):
    """X span (cm) for a feature reaching `reach_mm` into TPC drift, + cushion cx.
    TPC0 -> [-(reach+cx), 0]; TPC1 -> [0, +(reach+cx)]. Honors PER_TPC_REACH."""
    if feature in PER_TPC_REACH and tpc in PER_TPC_REACH[feature]:
        reach_mm = PER_TPC_REACH[feature][tpc]
    d = reach_mm * MM + cx_cm
    return (-d, 0.0) if tpc == 0 else (0.0, d)


def _zc(local_mm):
    """local z (mm) -> toolkit z (cm)."""
    return Z_CENTER_CM + local_mm * MM


def cathode_boxes(tpc, cx=0.5, cy=0.5, cz=0.5, pad=True, tube_hw_cm=None,
                  include_boss=False):
    """Per-TPC CPA structure-exclusion boxes (cm, toolkit frame).

    The whole exclusion region is dilated by an independent per-side cushion in
    each dimension (default 0.5 cm), applied to the pad slab, tube bars and
    knuckles alike:

    tpc        : 0 (East, X<0) or 1 (West, X>0)
    cx         : drift (X) cushion, cm, added to every reach (one-sided, deeper).
    cy         : vertical (Y) cushion, cm, added to every Y half-extent (per side).
    cz         : beam (Z) cushion, cm, added to every Z half-extent (per side).
    pad        : include the thin full-plane pad slab (default True).
    tube_hw_cm : tube bar transverse half-width (default = pipe radius 2.7 cm;
                 use ~6.1 cm to fill the whole inter-pad gap).
    include_boss : add the small center-boss box (default off; tubes cover it).
    """
    assert tpc in (0, 1)
    hw = (TUBE_HW_MM * MM) if tube_hw_cm is None else tube_hw_cm
    yplane = PLANE_YH_MM * MM + cy
    zlo, zhi = _zc(-PLANE_ZH_MM) - cz, _zc(PLANE_ZH_MM) + cz
    deep = []

    # deep horizontal tube bars (run along z) at the y gap-lines
    for y in HORIZ_Y_MM:
        x0, x1 = _xspan(tpc, TUBE_REACH_MM, cx, "tube")
        yc = y * MM
        deep.append(Box("htube_y%+05.0f" % y, x0, x1,
                        yc - (hw + cy), yc + (hw + cy), zlo, zhi))

    # deep vertical tube bars (run along y) at the z gap-lines
    for z in VERT_Z_MM:
        x0, x1 = _xspan(tpc, TUBE_REACH_MM, cx, "tube")
        zc = _zc(z)
        deep.append(Box("vtube_z%+05.0f" % z, x0, x1,
                        -yplane, yplane, zc - (hw + cz), zc + (hw + cz)))

    # deeper knuckle boxes on the central vertical line (z=0)
    for y in KNK_Y_MM:
        x0, x1 = _xspan(tpc, KNK_REACH_MM, cx, "knuckle")
        yc = y * MM
        deep.append(Box("knuckle_y%+05.0f" % y, x0, x1,
                        yc - (KNK_HY_MM * MM + cy), yc + (KNK_HY_MM * MM + cy),
                        _zc(-KNK_HZ_MM) - cz, _zc(KNK_HZ_MM) + cz))

    # optional center boss (mostly covered by the central tube crossing)
    if include_boss:
        x0, x1 = _xspan(tpc, 13.5, cx, "boss")
        deep.append(Box("boss", x0, x1, -(13.5 * MM + cy), (13.5 * MM + cy),
                        _zc(-29.5) - cz, _zc(29.5) + cz))

    boxes = []
    # thin full-plane pad slab (mesh + foil), thin in X. Its Y/Z extent is the
    # bounding box of the whole tube/knuckle lattice (cushions included), so it
    # covers the cathode plane out to and including the edge tube bars -- not
    # just the foil-grid envelope (PAD_YH_MM/PAD_ZH_MM, kept for reference).
    if pad:
        x0, x1 = _xspan(tpc, PAD_REACH_MM, cx, "pad")
        ymin = min(b.ymin for b in deep); ymax = max(b.ymax for b in deep)
        zmin = min(b.zmin for b in deep); zmax = max(b.zmax for b in deep)
        boxes.append(Box("pad", x0, x1, ymin, ymax, zmin, zmax))
    boxes += deep
    return boxes


def all_boxes(cx=0.5, cy=0.5, cz=0.5, **kw):
    return {0: cathode_boxes(0, cx, cy, cz, **kw),
            1: cathode_boxes(1, cx, cy, cz, **kw)}


def inside(point_cm, cx=0.5, cy=0.5, cz=0.5, **kw):
    """True if point (x,y,z in cm) is inside ANY CPA exclusion box (either TPC).
    Mirrors a toolkit CompositeFiducial{logic:'or'} over the per-TPC BoxFiducials."""
    for tpc in (0, 1):
        for b in cathode_boxes(tpc, cx, cy, cz, **kw):
            if b.contains(point_cm):
                return True
    return False


# --------------------------------------------------------------------------
# Toolkit jsonnet emitter (BoxFiducial + CompositeFiducial), paste-ready.
# --------------------------------------------------------------------------
def to_jsonnet(cx=0.5, cy=0.5, cz=0.5, **kw):
    f = lambda v: "%.3f" % v
    L = ["// SBND CPA structure-exclusion fiducial (generated by cathode_fiducial.py)",
         "// per-side cushions: cx=%.3f (X), cy=%.3f (Y), cz=%.3f (Z) cm" % (cx, cy, cz),
         "local wc = import 'wirecell.jsonnet';",
         "local cpa_boxes = ["]
    names = []
    for tpc in (0, 1):
        for b in cathode_boxes(tpc, cx, cy, cz, **kw):
            tn = "cpa-tpc%d-%s" % (tpc, b.name)
            names.append(tn)
            (x0, y0, z0), (x1, y1, z1) = b.ray()
            L += ["  { type: 'BoxFiducial', name: '%s'," % tn,
                  "    data: { bounds: {",
                  "      tail: { x: %s*wc.cm, y: %s*wc.cm, z: %s*wc.cm }," % (f(x0), f(y0), f(z0)),
                  "      head: { x: %s*wc.cm, y: %s*wc.cm, z: %s*wc.cm }," % (f(x1), f(y1), f(z1)),
                  "    } } },"]
    L += ["];",
          "local cpa_exclusion = {",
          "  type: 'CompositeFiducial', name: 'cpa-exclusion',",
          "  data: { logic: 'or', fiducials: [",
          "    " + ",\n    ".join("'BoxFiducial:%s'" % n for n in names),
          "  ] },",
          "};",
          "// contained() == true  ->  point is in the CPA structure region (exclude it)."]
    return "\n".join(L)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
KIND_STYLE = {  # edgecolor, fill alpha
    "pad":     ("#5b9bd5", 0.18),
    "htube":   ("#1f77b4", 0.30),
    "vtube":   ("#1f77b4", 0.30),
    "knuckle": ("#d62728", 0.45),
    "boss":    ("#2ca02c", 0.40),
}


def draw(png="cathode_fiducial.png", cx=0.5, cy=0.5, cz=0.5, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    from matplotlib.colors import to_rgba

    boxes = all_boxes(cx, cy, cz, **kw)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)

    def rect(ax, a, b, c, d, ec, alpha, lw=1.2):
        # translucent fill but a SOLID edge, so full-plane regions (the pad slab)
        # are visible by their boundary instead of washing out at low alpha.
        ax.add_patch(Rectangle((a, c), b - a, d - c, fill=True,
                               fc=to_rgba(ec, alpha), ec=ec, lw=lw, zorder=3))

    # ---- Panel 1: YZ cathode face (TPC0) -- the exclusion boxes only ----
    # (the physical CPA pads are NOT part of the FV definition, so not drawn)
    ax = axes[0]
    for b in boxes[0]:
        st = KIND_STYLE[b.kind()]
        rect(ax, b.zmin, b.zmax, b.ymin, b.ymax, st[0], st[1])
    ax.set_xlim(-20, 520); ax.set_ylim(-220, 220); ax.set_aspect("equal")
    ax.set_xlabel("Z [cm] (beam)"); ax.set_ylabel("Y [cm] (vertical)")
    ax.set_title("YZ cathode face (TPC0): pad slab + tube lattice")

    # ---- Panel 2: XZ slice at a pad-row (y = 51.75 cm) ----
    yslice = 51.75
    ax = axes[1]
    for b in boxes[0] + boxes[1]:
        if b.ymin <= yslice <= b.ymax:
            st = KIND_STYLE[b.kind()]
            rect(ax, b.zmin, b.zmax, b.xmin, b.xmax, st[0], st[1])
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlim(-20, 520); ax.set_ylim(-7, 7)
    ax.set_xlabel("Z [cm] (beam)"); ax.set_ylabel("X [cm] (drift)")
    ax.set_title("XZ slice at y=%.1f cm (pad row): thin pad, deep tubes/knuckle" % yslice)
    ax.text(10, 5.2, "TPC1 (West)", fontsize=8); ax.text(10, -5.6, "TPC0 (East)", fontsize=8)
    ax.grid(alpha=0.3)

    # ---- Panel 3: XY slice at a pad-column (z = 183.8 cm) ----
    zslice = _zc(-667.0)  # pad column center
    ax = axes[2]
    for b in boxes[0] + boxes[1]:
        if b.zmin <= zslice <= b.zmax:
            st = KIND_STYLE[b.kind()]
            rect(ax, b.ymin, b.ymax, b.xmin, b.xmax, st[0], st[1])
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlim(-220, 220); ax.set_ylim(-7, 7)
    ax.set_xlabel("Y [cm] (vertical)"); ax.set_ylabel("X [cm] (drift)")
    ax.set_title("XY slice at z=%.1f cm (pad column): thin pad, deep h-tubes" % zslice)
    ax.grid(alpha=0.3)

    leg = [Line2D([0], [0], color=KIND_STYLE[k][0], lw=6, alpha=0.5, label=lbl)
           for k, lbl in [("pad", "pad slab (~0.6 cm)"),
                          ("htube", "tube bars (~2.7 cm)"),
                          ("knuckle", "knuckles (~4.1 cm)")]]
    fig.legend(handles=leg, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("SBND CPA structure-exclusion: tube lattice "
                 "(cx=%.1f, cy=%.1f, cz=%.1f cm cushions)" % (cx, cy, cz), fontsize=13)
    fig.savefig(png, dpi=130)
    print("wrote", png)


def _selfcheck():
    Z0 = dict(cx=0.0, cy=0.0, cz=0.0)        # bare geometry, no cushion
    # --- bare-geometry reaches (zero cushion) ---
    pad_pt = (51.75, 183.8)  # (y, z) cm: pad-row center, pad-column center
    assert inside((-0.3,) + pad_pt, **Z0),       "shallow point on a pad should be inside (pad slab)"
    assert not inside((-2.0,) + pad_pt, **Z0),   "deep point on a pad should be OUTSIDE (no slab cut)"
    # horizontal tube (y=0 line), over a pad column: deep cut present
    assert inside((-2.0, 0.0, 183.8), **Z0),     "deep point on a horizontal tube should be inside"
    assert not inside((-4.0, 0.0, 183.8), **Z0), "tube reaches only ~2.7 cm; -4 outside"
    # vertical tube (z=0 line), off the knuckles: deep cut present
    assert inside((-2.0, 80.0, 250.5), **Z0),    "deep point on the central vertical tube should be inside"
    assert not inside((-4.0, 80.0, 250.5), **Z0),"tube reaches only ~2.7 cm; -4 outside"
    # knuckle: deepest
    assert inside((-3.5, 51.75, 250.5), **Z0),   "knuckle point should be inside"
    assert not inside((-4.5, 51.75, 250.5), **Z0),"knuckle reaches ~4.1 cm; -4.5 outside"
    # mirror: TPC1 is +X
    assert inside((2.0, 0.0, 183.8), **Z0),      "TPC1 horizontal tube at +X should be inside"
    # --- pad slab spans the lattice bbox: covers out to the edge tube bars ---
    # y=203 is beyond the foil envelope (201.85) and not on any tube line, yet
    # within the edge-tube-bar reach -> shallow cut present, deep cut absent.
    assert inside((-0.3, 203.0, 183.8), **Z0),   "pad slab should reach the edge tube bars (y~203)"
    assert not inside((-2.0, 203.0, 183.8), **Z0),"no deep cut out at the plane edge"
    # --- 0.5 cm default cushion dilates every region by 0.5 cm per side ---
    assert inside((-4.5, 51.75, 250.5)),          "knuckle 4.1+0.5 -> -4.5 inside with default cushion"
    assert not inside((-0.3, 210.0, 183.8), **Z0),"y=210 just past the bare plane edge (209.7)"
    assert inside((-0.3, 210.0, 183.8)),          "...but inside once the 0.5 cm Y cushion is added"
    assert inside((-2.0,) + pad_pt, cx=2.0),      "cx=2 widens pad slab past 2 cm"
    print("self-checks passed")


if __name__ == "__main__":
    _selfcheck()
    draw()
    nb = len(cathode_boxes(0))
    print("boxes per TPC: %d (%d total)" % (nb, 2 * nb))
    print("\n--- toolkit jsonnet (paste-ready, NOT applied) ---\n")
    print(to_jsonnet())
