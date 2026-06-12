"""Wire-geometry helpers for the PDHD imaging event display.

All public functions return coordinates in the **Bee frame** (cm), i.e. the
wire-geometry frame (mm) divided by 10, matching what ``wirecell-img bee-blobs``
writes into the Bee JSON (``arr/units.cm`` with units.cm == 10).  This lets the
2D blob view overlay the existing Bee sampling points without a re-sample.

Frame facts (same WCT wire-store schema as the PDVD viewer; PDHD store is
``protodunehd-wires-larsoft-v1.json.bz2``):
  * store.anodes[i].ident == i ; each anode has 2 faces ; each face has
    3 planes ordered U(ident0)/V(ident1)/W(ident2).
  * plane.wires[k] is the WIP-sorted index into store.wires ; blob `bounds`
    index this list (half-open [beg,end) per plane, ordered U,V,W).
  * Wire = (ident, channel, segment, tail, head) ; tail/head index
    store.points (x,y,z in mm).  Pitch lies purely in Y-Z:
    U/V = 4.926 mm, W = 4.792 mm.
  * PDHD U/V wires WRAP around the APA: one channel maps to several wire
    segments (possibly on both faces), so channel<->WIP is non-monotonic and
    non-unique (e.g. 1148 U wires share 800 channels per face).  Everything
    here uses per-(anode,face,plane) WIP lists, which stay well-defined; only
    channel labels repeat.
  * blob['faceid'] decodes via schema.plane_face_apa -> (plane, face, apa);
    plane is meaningless for a blob, face & apa are real, apa == faceid>>4.
"""
from __future__ import annotations

import numpy as np
from wirecell.util.wires import persist, schema

MM2CM = 0.1  # units.cm == 10 internal-mm; Bee divides by units.cm.


def load_store(path):
    return persist.load(path)


def faceid_to_anode_face(faceid):
    """Blob faceid bitmask -> (anode, face)."""
    _plane, face, apa = schema.plane_face_apa(faceid)
    return apa, face


def wplane_x_cm(store, anode, face):
    """Collection-plane (W, plane ident 2) wire-center x in cm for (anode,face).

    This is ``xorig`` in the toolkit BlobSampler ``time2drift`` convention
    (BlobSampler.cxx: ``plane_x(2) = anodeface->planes()[2]->wires().front()
    ->center().x()``), which the MABC Bee points are written in.
    """
    a = store.anodes[anode]
    f = store.faces[a.faces[face]]
    pl = next(store.planes[pi] for pi in f.planes
              if store.planes[pi].ident == 2)
    w = store.wires[pl.wires[0]]
    return 0.5 * (store.points[w.tail].x + store.points[w.head].x) * MM2CM


class PlaneGeom:
    """Per-(anode,face,plane) wire lookup + pitch geometry, all in Bee cm."""

    def __init__(self, store, anode, face, plane_ident):
        self.anode = anode
        self.face = face
        self.plane = plane_ident
        a = store.anodes[anode]
        f = store.faces[a.faces[face]]
        # locate the plane object with this ident on this face
        pl = next(store.planes[pi] for pi in f.planes
                  if store.planes[pi].ident == plane_ident)
        self._wire_idx = list(pl.wires)            # WIP -> store.wires index
        self._store = store
        # endpoints (cm) for every WIP: tail/head as (y,z)
        tails = np.empty((len(self._wire_idx), 2))
        heads = np.empty((len(self._wire_idx), 2))
        chans = np.empty(len(self._wire_idx), dtype=np.int64)
        for k, wi in enumerate(self._wire_idx):
            w = store.wires[wi]
            t = store.points[w.tail]
            h = store.points[w.head]
            tails[k] = (t.y * MM2CM, t.z * MM2CM)
            heads[k] = (h.y * MM2CM, h.z * MM2CM)
            chans[k] = w.channel
        self.tails = tails
        self.heads = heads
        self.chans = chans
        self.pitch_cm, self.phat = self._pitch_geom()

    def _pitch_geom(self):
        """Pitch magnitude (cm) and unit pitch direction (y,z) from wire centers."""
        centers = 0.5 * (self.tails + self.heads)
        # median consecutive delta over the first stretch is robust to wrap jumps
        n = min(40, len(centers) - 1)
        deltas = centers[1:n + 1] - centers[:n]
        mags = np.linalg.norm(deltas, axis=1)
        good = deltas[np.abs(mags - np.median(mags)) < 1e-3]
        d = np.median(good, axis=0) if len(good) else deltas[0]
        mag = float(np.linalg.norm(d))
        return mag, (d / mag)

    def band_quad(self, wip):
        """4 (y,z) cm corners of the ±half-pitch cell for wire `wip`."""
        h = 0.5 * self.pitch_cm
        off = h * self.phat
        t = self.tails[wip]
        hd = self.heads[wip]
        return np.array([t - off, hd - off, hd + off, t + off])

    def channel(self, wip):
        return int(self.chans[wip])


def order_polygon(yz):
    """Order a small convex point set (y,z) CCW around its centroid for drawing."""
    yz = np.asarray(yz, dtype=float)
    c = yz.mean(axis=0)
    ang = np.arctan2(yz[:, 0] - c[0], yz[:, 1] - c[1])
    return yz[np.argsort(ang)]


def point_in_poly(y, z, poly):
    """Ray-cast point-in-polygon. poly: ordered (N,2) array of (y,z)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        yi, zi = poly[i]
        yj, zj = poly[j]
        if ((zi > z) != (zj > z)) and \
           (y < (yj - yi) * (z - zi) / (zj - zi + 1e-30) + yi):
            inside = not inside
        j = i
    return inside
