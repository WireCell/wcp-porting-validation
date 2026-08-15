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
