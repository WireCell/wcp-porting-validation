"""Per-segment geometry and dQ/dx primitives for the hand-scan rules (pr/80).

The one design decision worth stating: **nothing here reads `dirsign` or `rr`.**

`rr` (residual range) is accumulated per `dirsign` (PrDisplayDump.cxx:447-454)
and the dump carries no explicit direction vector, so "dQ/dx rises as rr -> 0"
is a statement in rr-space whose mapping back to points[0] vs points[-1] runs
through a flag that itself carries a `dir_weak` unreliability marker.  Since
rules 1, 3 and 5 all consume the answer, a sign error there would invert the
whole track branch with nothing to catch it.

So the Bragg end is found instead from arc length measured from each physical
endpoint of `points[]`, which is polarity-free by construction: there is no
convention to get backwards.  `dirsign` is then used only as a cross-check,
never as an input (see polarity_control.py).
"""
import math

MIP = 43000.0            # e/cm, meta.mip_dqdx_median
BRAGG_WINDOW = 5.0       # cm from each end, over which end dQ/dx is averaged
MIN_POINTS_END = 3       # fewer than this in a window and the end is unmeasured


def seg_points(seg):
    return seg.get("points") or []


def valid_dqdx(p):
    """PR::Fit defaults are dQ = -1, dx = 0 (PRCommon.h), so a point with no
    measurement must be dropped, never read as low dQ/dx.  Same guard the
    display's dQ/dx layer and 1-D panel use."""
    return p.get("dx", 0) > 0 and p.get("dQ", -1) >= 0


def dqdx(p):
    return p["dQ"] / p["dx"]


def arclen(pts):
    """Cumulative arc length (cm) along points[], starting at 0."""
    out = [0.0]
    for a, b in zip(pts, pts[1:]):
        out.append(out[-1] + math.dist((a["x"], a["y"], a["z"]),
                                       (b["x"], b["y"], b["z"])))
    return out


def end_dqdx(seg, window=BRAGG_WINDOW):
    """Mean dQ/dx (e/cm) within `window` of each end of the segment.

    Returns (d0, d1, n0, n1) for the points[0] end and the points[-1] end.
    A mean is None when fewer than MIN_POINTS_END measured points fall in that
    window -- "not measured" is kept distinct from "low", because collapsing
    them is exactly how a Bragg test silently inverts on short segments.
    """
    pts = seg_points(seg)
    if len(pts) < 2:
        return (None, None, 0, 0)
    s = arclen(pts)
    total = s[-1]
    lo, hi = [], []
    for si, p in zip(s, pts):
        if not valid_dqdx(p):
            continue
        if si <= window:
            lo.append(dqdx(p))
        if total - si <= window:
            hi.append(dqdx(p))
    d0 = sum(lo) / len(lo) if len(lo) >= MIN_POINTS_END else None
    d1 = sum(hi) / len(hi) if len(hi) >= MIN_POINTS_END else None
    return (d0, d1, len(lo), len(hi))


def bragg_end(seg, window=BRAGG_WINDOW, ratio=1.3):
    """Which physical end of the segment looks like the stopping (Bragg) end.

    Returns one of "start" (the points[0] end), "end" (the points[-1] end), or
    None when the two ends are not separated by at least `ratio`, or when
    either end is unmeasured.  None means "no opinion" and must not be
    silently turned into a choice.
    """
    d0, d1, _, _ = end_dqdx(seg, window)
    if d0 is None or d1 is None or d0 <= 0 or d1 <= 0:
        return None
    if d1 / d0 >= ratio:
        return "end"
    if d0 / d1 >= ratio:
        return "start"
    return None


def endpoint_vertices(seg):
    """(vertex id at the points[0] end, vertex id at the points[-1] end).

    NOTE this is the *assumed* mapping start_vertex_id <-> points[0]; it is
    verified empirically in polarity_control.py against the vertices' own fitted
    positions before any rule is allowed to depend on it.
    """
    return (seg.get("start_vertex_id"), seg.get("end_vertex_id"))


def far_vertex(seg, vid):
    """The vertex at the other end of `seg` from `vid`, or None."""
    a, b = endpoint_vertices(seg)
    if vid == a:
        return b
    if vid == b:
        return a
    return None


def end_name_of_vertex(seg, vid):
    """"start" / "end" / None -- which physical end of `seg` carries `vid`."""
    a, b = endpoint_vertices(seg)
    if vid == a:
        return "start"
    if vid == b:
        return "end"
    return None


def is_track(seg):
    """Track-like: not flagged as a shower trajectory/topology."""
    return not seg.get("flag_shower")


def principal_axes(pts):
    """(centroid, [a1, a2, a3]) for a list of {x,y,z} dicts, a1 the longest.

    Used to render an event in its own frame.  A track running diagonally is
    foreshortened in all three of X-Y / Z-Y / X-Z at once, which is exactly the
    case where a scanner cannot tell two vertices apart; rotating into the
    cluster's own axes removes that failure mode rather than working around it.
    """
    import numpy as np
    if len(pts) < 3:
        return (None, None)
    a = np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=float)
    c = a.mean(axis=0)
    # eigh on the covariance: symmetric, ascending eigenvalues, deterministic.
    _, vecs = np.linalg.eigh(np.cov((a - c).T))
    axes = [vecs[:, 2], vecs[:, 1], vecs[:, 0]]
    # Sign is arbitrary out of eigh; fix it so a rerun of the same event draws
    # the same picture (a flipped axis is a different picture to a scanner).
    axes = [v if v[abs(v).argmax()] >= 0 else -v for v in axes]
    return (tuple(c), [tuple(v) for v in axes])


def cone_profile(pts, apex, axis=None, nbins=8):
    """Transverse spread vs distance along the axis, from `apex`.

    The blind-safe stand-in for `showers[]`.  Shower membership, `showers[].start`
    and `stem_dqdx` are all produced by `shower_clustering_with_nv`, i.e. after
    the reconstruction has already chosen its neutrino vertex, so none of them
    may be shown to a blind scanner.  What the owner actually reads off the
    screen -- "does this open out as it goes?" -- is recoverable from the point
    cloud alone: a shower's transverse RMS grows with distance from its apex, a
    track's does not.

    Returns (axis, [(longitudinal_cm, transverse_rms_cm, n), ...]); the axis is
    apex -> centroid of the points ahead of the apex unless one is given.
    """
    import numpy as np
    if len(pts) < 4 or apex is None:
        return (None, [])
    a = np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=float)
    o = np.array(apex, dtype=float)
    d = a - o
    if axis is None:
        axis = d.mean(axis=0)
    axis = np.array(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n <= 0:
        return (None, [])
    axis = axis / n
    lon = d @ axis
    trans = np.linalg.norm(d - np.outer(lon, axis), axis=1)
    fwd = lon > 0
    if fwd.sum() < 4:
        return (tuple(axis), [])
    lo, hi = 0.0, float(lon[fwd].max())
    if hi <= 0:
        return (tuple(axis), [])
    edges = np.linspace(lo, hi, nbins + 1)
    out = []
    for i in range(nbins):
        m = fwd & (lon >= edges[i]) & (lon < edges[i + 1] if i < nbins - 1
                                       else lon <= edges[i + 1])
        k = int(m.sum())
        if k < 2:
            out.append((float(0.5 * (edges[i] + edges[i + 1])), None, k))
        else:
            out.append((float(0.5 * (edges[i] + edges[i + 1])),
                        float(np.sqrt((trans[m] ** 2).mean())), k))
    return (tuple(axis), out)


def opening_trend(prof):
    """One number out of `cone_profile`: transverse RMS in the far half minus
    the near half, in cm.  Positive = opens out (shower-like).  None when too
    few bins are measured to say -- which must not be read as "does not open"."""
    ok = [(l, t) for l, t, k in prof if t is not None]
    if len(ok) < 4:
        return None
    h = len(ok) // 2
    near = sum(t for _, t in ok[:h]) / h
    far = sum(t for _, t in ok[len(ok) - h:]) / h
    return far - near


def seg_direction(seg, at_end):
    """Unit vector pointing AWAY from the given end ("start" or "end").

    Uses up to the first/last 10 cm so a local wiggle does not set the
    direction; falls back to the whole segment when it is shorter than that.
    """
    pts = seg_points(seg)
    if len(pts) < 2:
        return None
    s = arclen(pts)
    span = min(10.0, s[-1])
    if span <= 0:
        return None
    if at_end == "start":
        a = pts[0]
        b = next((p for si, p in zip(s, pts) if si >= span), pts[-1])
    else:
        a = pts[-1]
        b = next((p for si, p in zip(reversed(s), reversed(pts))
                  if s[-1] - si >= span), pts[0])
    v = (b["x"] - a["x"], b["y"] - a["y"], b["z"] - a["z"])
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v) if n > 0 else None


# ---------------------------------------------------------------- round 3
# Added for doc pr/80 round 3 (B1/B3/B4).  All of it is tier-1 safe: detector
# geometry is a static property of SBND, and everything else is recomputed from
# fitted points[].  Nothing here reads the reconstruction's verdict.

# Active volume, matching scankit.DET_BOX (kept here as a plain literal rather
# than imported, so this module stays free of a scankit dependency).
DET_FACES = (("x-", "x", -201.05), ("x+", "x", 201.05),
             ("y-", "y", -199.312), ("y+", "y", 199.312),
             ("z-", "z", 0.85), ("z+", "z", 500.15))


def face_distance(pt):
    """(distance in cm to the nearest active-volume face, face name).

    `pt` is a dict with x/y/z or a 3-tuple.  A point outside the box gives a
    negative distance, which is not an error: fitted trajectories overshoot the
    nominal boundary by a few mm routinely.
    """
    if isinstance(pt, dict):
        p = {k: pt[k] for k in "xyz"}
    else:
        p = dict(zip("xyz", pt))
    best, name = None, None
    for nm, ax, val in DET_FACES:
        d = abs(p[ax] - val)
        if p[ax] < val and nm.endswith("-"):
            d = p[ax] - val          # outside the low face -> negative
        elif p[ax] > val and nm.endswith("+"):
            d = val - p[ax]
        if best is None or d < best:
            best, name = d, nm
    return best, name


def seg_end_xyz(seg):
    """((x,y,z) at the points[0] end, (x,y,z) at the points[-1] end)."""
    pts = seg_points(seg)
    if not pts:
        return None, None
    return ((pts[0]["x"], pts[0]["y"], pts[0]["z"]),
            (pts[-1]["x"], pts[-1]["y"], pts[-1]["z"]))


def angle_deg(u, v):
    """Angle between two unit-ish vectors, in degrees, or None."""
    if u is None or v is None:
        return None
    d = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu <= 0 or nv <= 0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, d / (nu * nv)))))


def prong_angle(seg_a, seg_b, vid):
    """Angle between two segments leaving the same vertex, in degrees.

    180 deg means they leave back-to-back, i.e. the "vertex" lies on a straight
    line through both -- the collinearity caution of doc pr/80 sec 10.8.
    """
    ea = end_name_of_vertex(seg_a, vid)
    eb = end_name_of_vertex(seg_b, vid)
    if ea is None or eb is None:
        return None
    return angle_deg(seg_direction(seg_a, ea), seg_direction(seg_b, eb))
