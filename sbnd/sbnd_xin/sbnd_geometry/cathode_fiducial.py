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


def cathode_boxes(tpc, cx=0.0, ct=0.0, pad=True, tube_hw_cm=None, include_boss=False):
    """Per-TPC CPA structure-exclusion boxes (cm, toolkit frame).

    tpc        : 0 (East, X<0) or 1 (West, X>0)
    cx         : drift (X) cushion, cm, added to every reach.
    ct         : transverse (Y,Z) cushion, cm, added to every Y/Z half-extent.
    pad        : include the thin full-plane pad slab (default True).
    tube_hw_cm : tube bar transverse half-width (default = pipe radius 2.7 cm;
                 use ~6.1 cm to fill the whole inter-pad gap).
    include_boss : add the small center-boss box (default off; tubes cover it).
    """
    assert tpc in (0, 1)
    hw = (TUBE_HW_MM * MM) if tube_hw_cm is None else tube_hw_cm
    yplane = PLANE_YH_MM * MM + ct
    zlo, zhi = _zc(-PLANE_ZH_MM) - ct, _zc(PLANE_ZH_MM) + ct
    boxes = []

    # thin full-plane pad slab (mesh + foil)
    if pad:
        x0, x1 = _xspan(tpc, PAD_REACH_MM, cx, "pad")
        boxes.append(Box("pad", x0, x1,
                         -(PAD_YH_MM * MM + ct), (PAD_YH_MM * MM + ct),
                         _zc(-PAD_ZH_MM) - ct, _zc(PAD_ZH_MM) + ct))

    # deep horizontal tube bars (run along z) at the y gap-lines
    for y in HORIZ_Y_MM:
        x0, x1 = _xspan(tpc, TUBE_REACH_MM, cx, "tube")
        yc = y * MM
        boxes.append(Box("htube_y%+05.0f" % y, x0, x1,
                         yc - (hw + ct), yc + (hw + ct), zlo, zhi))

    # deep vertical tube bars (run along y) at the z gap-lines
    for z in VERT_Z_MM:
        x0, x1 = _xspan(tpc, TUBE_REACH_MM, cx, "tube")
        zc = _zc(z)
        boxes.append(Box("vtube_z%+05.0f" % z, x0, x1,
                         -yplane, yplane, zc - (hw + ct), zc + (hw + ct)))

    # deeper knuckle boxes on the central vertical line (z=0)
    for y in KNK_Y_MM:
        x0, x1 = _xspan(tpc, KNK_REACH_MM, cx, "knuckle")
        yc = y * MM
        boxes.append(Box("knuckle_y%+05.0f" % y, x0, x1,
                         yc - (KNK_HY_MM * MM + ct), yc + (KNK_HY_MM * MM + ct),
                         _zc(-KNK_HZ_MM) - ct, _zc(KNK_HZ_MM) + ct))

    # optional center boss (mostly covered by the central tube crossing)
    if include_boss:
        x0, x1 = _xspan(tpc, 13.5, cx, "boss")
        boxes.append(Box("boss", x0, x1, -(13.5 * MM + ct), (13.5 * MM + ct),
                         _zc(-29.5) - ct, _zc(29.5) + ct))
    return boxes


def all_boxes(cx=0.0, ct=0.0, **kw):
    return {0: cathode_boxes(0, cx, ct, **kw), 1: cathode_boxes(1, cx, ct, **kw)}


def inside(point_cm, cx=0.0, ct=0.0, **kw):
    """True if point (x,y,z in cm) is inside ANY CPA exclusion box (either TPC).
    Mirrors a toolkit CompositeFiducial{logic:'or'} over the per-TPC BoxFiducials."""
    for tpc in (0, 1):
        for b in cathode_boxes(tpc, cx, ct, **kw):
            if b.contains(point_cm):
                return True
    return False


# --------------------------------------------------------------------------
# Toolkit jsonnet emitter (BoxFiducial + CompositeFiducial), paste-ready.
# --------------------------------------------------------------------------
def to_jsonnet(cx=0.0, ct=0.0, **kw):
    f = lambda v: "%.3f" % v
    L = ["// SBND CPA structure-exclusion fiducial (generated by cathode_fiducial.py)",
         "// cx=%.3f cm (drift cushion), ct=%.3f cm (transverse cushion)" % (cx, ct),
         "local wc = import 'wirecell.jsonnet';",
         "local cpa_boxes = ["]
    names = []
    for tpc in (0, 1):
        for b in cathode_boxes(tpc, cx, ct, **kw):
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
    "pad":     ("#7fb3d5", 0.12),
    "htube":   ("#1f77b4", 0.30),
    "vtube":   ("#1f77b4", 0.30),
    "knuckle": ("#d62728", 0.45),
    "boss":    ("#2ca02c", 0.40),
}


def _pad_outlines():
    """16 mesh-pad footprints in toolkit coords (z,y) cm, for the face view."""
    out = []
    for y in (517.5, -517.5, 1552.5, -1552.5):
        for z in (667.0, -667.0, 1942.0, -1942.0):
            out.append((_zc(z) - 1153.0 * MM / 2, _zc(z) + 1153.0 * MM / 2,
                        y * MM - 913.0 * MM / 2, y * MM + 913.0 * MM / 2))
    return out


def draw(png="cathode_fiducial.png", cx=0.0, ct=0.0, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    boxes = all_boxes(cx, ct, **kw)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)

    def rect(ax, a, b, c, d, ec, alpha, lw=1.2):
        ax.add_patch(Rectangle((a, c), b - a, d - c, fill=True, fc=ec,
                               alpha=alpha, ec=ec, lw=lw, zorder=3))

    # ---- Panel 1: YZ cathode face (TPC0) -- the lattice grid over the pads ----
    ax = axes[0]
    for (z0, z1, y0, y1) in _pad_outlines():
        ax.add_patch(Rectangle((z0, y0), z1 - z0, y1 - y0, fill=True,
                               fc="0.92", ec="0.6", lw=0.6, zorder=1))
    for b in boxes[0]:
        st = KIND_STYLE[b.kind()]
        rect(ax, b.zmin, b.zmax, b.ymin, b.ymax, st[0], st[1])
    ax.set_xlim(-20, 520); ax.set_ylim(-220, 220); ax.set_aspect("equal")
    ax.set_xlabel("Z [cm] (beam)"); ax.set_ylabel("Y [cm] (vertical)")
    ax.set_title("YZ cathode face (TPC0): pads (gray) + tube lattice")

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
    leg.append(Line2D([0], [0], color="0.6", lw=6, alpha=0.4, label="CPA pads"))
    fig.legend(handles=leg, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("SBND CPA structure-exclusion: tube lattice (cx=%.1f cm, ct=%.1f cm)"
                 % (cx, ct), fontsize=13)
    fig.savefig(png, dpi=130)
    print("wrote", png)


def _selfcheck():
    # pad point away from any tube line: thin cut only
    pad_pt = (51.75, 183.8)  # (y, z) cm: pad-row center, pad-column center
    assert inside((-0.3,) + pad_pt),       "shallow point on a pad should be inside (pad slab)"
    assert not inside((-2.0,) + pad_pt),   "deep point on a pad should be OUTSIDE (no slab cut)"
    # horizontal tube (y=0 line), over a pad column: deep cut present
    assert inside((-2.0, 0.0, 183.8)),     "deep point on a horizontal tube should be inside"
    assert not inside((-4.0, 0.0, 183.8)), "tube reaches only ~2.7 cm; -4 outside"
    # vertical tube (z=0 line), off the knuckles: deep cut present
    assert inside((-2.0, 80.0, 250.5)),    "deep point on the central vertical tube should be inside"
    assert not inside((-4.0, 80.0, 250.5)),"tube reaches only ~2.7 cm; -4 outside"
    # knuckle: deepest
    assert inside((-3.5, 51.75, 250.5)),   "knuckle point should be inside"
    assert not inside((-4.5, 51.75, 250.5)),"knuckle reaches ~4.1 cm; -4.5 outside"
    # cushion widens the pad slab
    assert inside((-2.0,) + pad_pt, cx=2.0),"cx=2 widens pad slab past 2 cm"
    # mirror: TPC1 is +X
    assert inside((2.0, 0.0, 183.8)),       "TPC1 horizontal tube at +X should be inside"
    print("self-checks passed")


if __name__ == "__main__":
    _selfcheck()
    draw()
    nb = len(cathode_boxes(0))
    print("boxes per TPC: %d (%d total)" % (nb, 2 * nb))
    print("\n--- toolkit jsonnet (paste-ready, NOT applied) ---\n")
    print(to_jsonnet())
