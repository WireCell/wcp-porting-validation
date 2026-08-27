"""doc pr/114 -- the geometry the EM/pi0 hand-scan display needs, mirrored from
the C++ that actually runs, not re-derived.

Every function here has a named counterpart in the toolkit.  The line citations
are checked at toolkit 8d93260d; when a citation and the code disagree, the code
wins and the citation is the bug.  Nothing in this module reads or writes a
file -- it takes plain dicts straight out of `calib-pr-evt<ID>.json`.

Unit convention, stated once because it is the trap this file exists to avoid:
**the calib dump is already in cm** (`PrDisplayDump.cxx` divides by units::cm on
the way out, and `meta.length_unit == "cm"`).  So every distance here is cm and
every angle is DEGREES unless the name says `_rad`.  The C++ works in
`units::cm` internally and returns radians from `std::acos`; the dump's
`kine.kine_pio_angle` is degrees because `NeutrinoKinematics.cxx:597-598`
converts on the way into the tree.  Mixing the two silently produces a pi0 mass
that is wrong by a factor you will not notice.
"""

import math

# ---------------------------------------------------------------------------
# vectors -- plain 3-tuples, no numpy, so this module can be imported by the
# viewer (stdlib + bokeh + numpy diet) and by the prep script alike.
# ---------------------------------------------------------------------------


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vscale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def vmag(a):
    return math.sqrt(vdot(a, a))


def vnorm(a):
    m = vmag(a)
    if m <= 0:
        return (0.0, 0.0, 0.0)
    return (a[0] / m, a[1] / m, a[2] / m)


def angle_deg(a, b):
    """Opening angle in degrees, clamped exactly as every C++ call site does
    (`std::acos(std::clamp(dot/(|a||b|), -1.0, 1.0))`).  Returns None when
    either vector is zero -- the C++ would divide by zero there, and a display
    that prints 90 deg for "no direction" is lying."""
    ma, mb = vmag(a), vmag(b)
    if ma <= 0 or mb <= 0:
        return None
    c = max(-1.0, min(1.0, vdot(a, b) / (ma * mb)))
    return math.degrees(math.acos(c))


def pt(d):
    """{'x':..,'y':..,'z':..} -> tuple.  The dump uses this shape for
    `showers[].start`, `showers[].end`, `main_vertex` and `vertices[].fit`."""
    if d is None:
        return None
    return (d.get("x"), d.get("y"), d.get("z"))


# ---------------------------------------------------------------------------
# membership: the join, and how lossy it is
# ---------------------------------------------------------------------------


def shower_members(shower, segments):
    """Segments of one shower, via the dump's own join.

    `showers[].id` IS the start segment's id (`cluster_id*1000 + seg index`,
    `PrDisplayDump.cxx:577` via `pf_node_id`) and `segments[].shower_id` points
    back at it (`:490`).  `showers[].shower_id` is a different thing -- the
    internal sequential `Shower::get_shower_id()` -- and does NOT join.
    (`scripts/pr93_shower_composition.py:85` has these two swapped in its
    output column names; `pr113_topology_census.py:120-124` has it right.)
    """
    sid = shower.get("id")
    return [s for s in segments if s.get("shower_id") == sid]


def join_completeness(shower, segments):
    """(joined, declared) -- and they are not always equal.

    `segments[].shower_id` stores ONE shower per segment, so when two showers
    overlap the loser's members are invisible to the join and it comes back
    looking empty rather than nested (`NeutrinoShowerClustering.cxx:116-126`,
    which names SBND 347129 and 394532 as the cases that motivated the
    WCT_SHOWER_CONTENT_DEBUG probe).

    Measured over the 67 curated prod0825 events: 11 of 1081 showers (1.0 %),
    every one of them EM, spread over 9 of the 67 events.  Worst two are ncpi0
    evt84229 shower 69134 (43 of 50, a 958 MeV shower) and ncpi0 evt463565
    shower 109073 (0 of 5 -- renders as empty).

    The point of returning both numbers is that the display can SAY SO.  The
    dump is lossy but it is not silent: `num_segments` is the shower's own
    count and disagreement is computable without any re-run.
    """
    return len(shower_members(shower, segments)), int(shower.get("num_segments", 0) or 0)


def seg_points(seg):
    """Ordered fitted trajectory points of a segment as tuples."""
    return [(p["x"], p["y"], p["z"]) for p in (seg.get("points") or [])]


# ---------------------------------------------------------------------------
# direction
# ---------------------------------------------------------------------------


def segment_cal_dir_3vector(seg):
    """Mirror of `segment_cal_dir_3vector(SegmentPtr)`,
    `clus/src/PRSegmentFunctions.cxx:2323-2361`.

    Sums the first (or last) few steps of the fitted trajectory in the
    direction `dirsign` says the segment runs, then normalises.  `dirsign == 0`
    means the direction was never determined and the C++ returns the zero
    vector -- so do we; a display must not invent a direction the
    reconstruction declined to assign.
    """
    fits = seg_points(seg)
    flag_dir = seg.get("dirsign", 0)
    if len(fits) < 2:
        return (0.0, 0.0, 0.0)

    p = (0.0, 0.0, 0.0)
    n = len(fits)
    if flag_dir == 1:
        for i in range(1, min(5, n)):
            p = vadd(p, vsub(fits[i], fits[0]))
    elif flag_dir == -1:
        # C++ loop is `i=1; i<5 && (n-i-1)<n; i++` on size_t, so the guard bites
        # by underflow once i >= n: the effective range is i in [1, min(4, n-1)].
        for i in range(1, min(5, n)):
            p = vadd(p, vsub(fits[n - i - 1], fits[n - 1]))
    else:
        return (0.0, 0.0, 0.0)

    return vnorm(p) if vmag(p) > 0 else (0.0, 0.0, 0.0)


def shower_cal_dir_3vector(members, p, dis_cut=15.0):
    """Mirror of `shower_cal_dir_3vector(Shower&, const Point&, double)`,
    `clus/src/PRShowerFunctions.cxx:132-186`.

    Averages every member fitted point within `dis_cut` of `p` and returns the
    normalised vector from `p` to that average.

    NOT bit-exact, and the doc says so: the C++ walks
    `shower_ordered_edges(shower, view)` -- the shower's view graph -- whereas
    `members` here comes from the dump's `fill_sets` membership
    (`PrDisplayDump.cxx:395-400`).  The two agree in the ordinary case and can
    differ for orphan nodes.  Since this is a sum over an unordered set the
    ORDER does not matter (only FP association does, at the 1e-15 level); what
    matters is whether the SET matches, which is the same 1 % question
    `join_completeness` reports.
    """
    cut_sq = dis_cut * dis_cut
    sx = sy = sz = 0.0
    ncount = 0
    for seg in members:
        for q in seg_points(seg):
            dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
            if dx * dx + dy * dy + dz * dz < cut_sq:
                sx += q[0]
                sy += q[1]
                sz += q[2]
                ncount += 1
    if ncount == 0:
        return (0.0, 0.0, 0.0)
    avg = (sx / ncount, sy / ncount, sz / ncount)
    d = vsub(avg, p)
    return vnorm(d) if vmag(d) > 0 else (0.0, 0.0, 0.0)


def shower_init_dir(shower, segments, vertices_by_id):
    """Mirror of the `data.init_dir` derivation inside
    `Shower::calculate_kinematics` -- single-segment branch
    `clus/src/PRShower.cxx:1552-1562`, multi-segment branch `:1618-1640`.

    Returns `(dir, branch)` where `branch` names which of the four paths was
    taken, so the display can show the caveat where it applies rather than as a
    blanket disclaimer:

      "conn1_seg"    conn_type 1                -> segment_cal_dir_3vector(start seg)
                     (single-segment always; multi-segment when the start
                      segment is longer than 8 cm)
      "conn23_chord" conn_type 2 or 3           -> (start_point - start_vertex).norm()
      "conn1_short"  conn_type 1, multi-segment, start segment <= 8 cm
                                                -> shower_cal_dir_3vector(vtx, 12 cm)
      "fallback"     whatever was computed came out zero
                                                -> shower_cal_dir_3vector(vtx, 12 cm)

    The first two are EXACT from the dump.  Only the last two go through
    `shower_cal_dir_3vector`, i.e. only they inherit that function's membership
    caveat -- and conn 2/3 dominates the population (ncpi0 conn-type counts are
    {1: 20, 2: 69, 3: 235, 4: 44}), so the exact paths carry most events.
    """
    conn = shower.get("start_connection_type")
    start = pt(shower.get("start"))
    svid = shower.get("start_vertex_id", -1)
    svtx = vertices_by_id.get(svid)
    svtx_pt = pt(svtx.get("fit")) if svtx else None
    members = shower_members(shower, segments)

    start_seg = None
    sid = shower.get("id")
    for s in segments:
        if s.get("id") == sid:
            start_seg = s
            break

    nseg = int(shower.get("num_segments", 0) or 0)
    d = (0.0, 0.0, 0.0)
    branch = "none"

    if conn == 1:
        # The 8 cm test exists ONLY in the multi-segment branch (:1619).  The
        # single-segment branch (:1553) goes straight to the segment direction.
        if nseg <= 1 or (start_seg is not None
                         and (start_seg.get("length") or 0.0) > 8.0):
            if start_seg is not None:
                d = segment_cal_dir_3vector(start_seg)
                branch = "conn1_seg"
        elif svtx_pt is not None:
            d = shower_cal_dir_3vector(members, svtx_pt, 12.0)
            branch = "conn1_short"
    elif conn in (2, 3):
        if svtx_pt is not None and start is not None:
            d = vnorm(vsub(start, svtx_pt))
            branch = "conn23_chord"

    if vmag(d) == 0 and svtx_pt is not None:
        d = shower_cal_dir_3vector(members, svtx_pt, 12.0)
        branch = "fallback"

    return d, branch


# ---------------------------------------------------------------------------
# the pass-1 acceptance gate
# ---------------------------------------------------------------------------

# `clus/src/NeutrinoShowerClustering.cxx:1310-1312`.  THREE NESTED TIERS, not
# one cone: a wide-and-near, a medium, and a narrow-and-far.
CONE_TIERS = ((25.0, 80.0), (12.5, 130.0), (5.0, 200.0))

# The ranking metric that decides WHICH shower claims a segment when several
# accept it (`:1314-1315`): an ellipsoid 40 cm along the axis, 5 cm across.
CONE_METRIC_LONG_CM = 40.0
CONE_METRIC_TRANS_CM = 5.0

# SBND drifts along +/-x, so the drift direction is the x axis.
DRIFT_DIR = (1.0, 0.0, 0.0)


def cone_angle_offset(shower_dir):
    """`NeutrinoShowerClustering.cxx:1179-1184`: a shower whose axis sits within
    5 deg of PERPENDICULAR to the drift direction is isochronous -- its 2-D
    projections are degenerate and its direction is poorly measured -- so the
    gate is loosened by 5 deg (and the direction itself is recomputed over a
    50 cm window instead of 15).  Returns 0 or 5."""
    a = angle_deg(shower_dir, DRIFT_DIR)
    if a is None:
        return 0.0
    # the C++ takes |cos| before the acos, folding the angle into [0, 90]
    a = min(a, 180.0 - a)
    return 5.0 if abs(a - 90.0) < 5.0 else 0.0


def cone_tier(angle, dist, angle_offset=0.0):
    """Which pass-1 tier accepts (angle deg, dist cm), or None.

    Mirrors `:1310-1312` including the two different offset multipliers -- the
    middle tier scales the offset by 8/5 and the narrow tier by 2, which is why
    this is not simply "add 5 to every threshold".
    """
    if angle is None:
        return None
    if angle < 25.0 + angle_offset and dist < 80.0:
        return 1
    if angle < 12.5 + angle_offset * 8.0 / 5.0 and dist < 130.0:
        return 2
    if angle < 5.0 + angle_offset * 2.0 and dist < 200.0:
        return 3
    return None


def cone_metric(angle, dist):
    """The `:1314-1315` ellipsoidal closeness used to rank competing showers.
    Smaller is closer.  Not a gate -- a tie-break."""
    if angle is None:
        return None
    a = math.radians(angle)
    return ((dist * math.cos(a)) ** 2 / CONE_METRIC_LONG_CM ** 2
            + (dist * math.sin(a)) ** 2 / CONE_METRIC_TRANS_CM ** 2)


def segment_closest_point(seg, p):
    """(distance, point) of the segment's closest fitted point to p.  The C++
    equivalents walk a point cloud; over a fitted polyline of ~20 points the
    brute force here is the same answer."""
    best = None
    bestd = None
    for q in seg_points(seg):
        d = vmag(vsub(q, p))
        if bestd is None or d < bestd:
            bestd, best = d, q
    return bestd, best


# ---------------------------------------------------------------------------
# pi0
# ---------------------------------------------------------------------------


def ray_closest_points(r1, r2):
    """Mirror of `WireCell::ray_closest_points`, `util/src/Point.cxx:125-171`.

    `r1`/`r2` are (origin, far_point) pairs; the direction is far - origin.
    Returns `(p1, p2, status)` where status is one of "ok", "parallel",
    "degenerate".

    The C++ returns only the pair, and signals the parallel case by returning
    the two ray ORIGINS (`:138-141`) -- indistinguishable from a real answer
    unless you check.  This wrapper reports it instead, because a pi0 vertex
    quietly placed at a shower's own start point is exactly the kind of wrong
    number a hand scan must not be handed.

    "degenerate" covers the case the C++ does not guard: `scale1` divides by
    `dir1_unit.dot(rej.norm())` (`:153`), which is 0/0 or a division by zero
    when the rejection vector vanishes -- i.e. when the two rays already
    intersect, or when one direction lies in the plane spanned by the other and
    the common normal.  In C++ that yields inf/nan and propagates silently.
    """
    o1, f1 = r1
    o2, f2 = r2
    d = vsub(o1, o2)
    dir1 = vsub(f1, o1)
    dir2 = vsub(f2, o2)

    c = vcross(dir1, dir2)
    c_mag = vmag(c)
    if c_mag < 1e-6:
        return o1, o2, "parallel"

    def _one(dd, cc, along, other):
        """closest point on the `along` ray, following :146-154 / :157-168."""
        cc_unit = vnorm(cc)
        m_other = vmag(other)
        if m_other <= 0:
            return None
        proj_mag = vdot(dd, other) / m_other
        proj = vscale(vnorm(other), proj_mag)
        rej = vsub(vsub(dd, proj), vscale(cc_unit, vdot(dd, cc_unit)))
        rej_mag = vmag(rej)
        if rej_mag <= 0:
            return None
        along_unit = vnorm(along)
        den = vdot(along_unit, vnorm(rej))
        if abs(den) < 1e-12:
            return None
        return rej_mag / den, along_unit

    a = _one(d, c, dir1, dir2)
    b = _one(vscale(d, -1.0), vscale(c, -1.0), dir2, dir1)
    if a is None or b is None:
        return o1, o2, "degenerate"

    scale1, u1 = a
    scale2, u2 = b
    p1 = vsub(o1, vscale(u1, scale1))
    p2 = vsub(o2, vscale(u2, scale2))
    return p1, p2, "ok"


def pi0_mass(e1, e2, theta_deg):
    """`mass = sqrt(4 * E1 * E2 * sin^2(theta/2))`, the formula the code itself
    uses at `NeutrinoShowerClustering.cxx:3771`, `:4199` and `:4250`.

    Energies are the showers' `kine_charge` in MeV (that is what
    `get_kine_charge()` returns at those call sites -- NOT `kine_best`), so the
    result is MeV.  `theta_deg` in degrees; the C++ passes radians because its
    angle came straight from `acos`.
    """
    if e1 is None or e2 is None or theta_deg is None or e1 <= 0 or e2 <= 0:
        return None
    return math.sqrt(4.0 * e1 * e2 * math.sin(math.radians(theta_deg) / 2.0) ** 2)


# `:4321` -- the acceptance window the winner loop applies.  Note it is not
# symmetric about 135: the code tests |mass - 135 + 10| < 60, i.e. the window is
# (65, 185) MeV centred on 125.
PI0_MASS_TARGET = 135.0
PI0_MASS_SHIFT = 10.0
PI0_MASS_HALFWIDTH = 60.0


def pi0_mass_accepted(mass):
    if mass is None:
        return False
    return abs(mass - PI0_MASS_TARGET + PI0_MASS_SHIFT) < PI0_MASS_HALFWIDTH


def gamma_ray(shower, segments, anchor):
    """The ray `id_pi0_without_vertex` builds per gamma,
    `NeutrinoShowerClustering.cxx:4139-4142`:

        test_p = closest fitted point of the shower to `anchor`
        dir    = shower_cal_dir_3vector(shower, test_p, 15 cm)
        ray    = (test_p, test_p + dir)

    `anchor` is the main vertex in the code.  Returns (ray, test_p, dir) or
    (None, None, None) when the shower has no fitted points.
    """
    members = shower_members(shower, segments)
    best = None
    bestd = None
    for seg in members:
        d, q = segment_closest_point(seg, anchor)
        if d is not None and (bestd is None or d < bestd):
            bestd, best = d, q
    if best is None:
        return None, None, None
    dirv = shower_cal_dir_3vector(members, best, 15.0)
    if vmag(dirv) == 0:
        return None, best, dirv
    return (best, vadd(best, dirv)), best, dirv


def pi0_backproject(sh1, sh2, segments, anchor):
    """Mirror of the two-gamma vertex reconstruction in
    `id_pi0_without_vertex`, `NeutrinoShowerClustering.cxx:4158-4256`.

    Returns a dict the display can render directly.  `verdict` is the honest
    summary: "ok", "parallel", "degenerate", "both_short" (the C++ `break` at
    `:4255`), "angle_gate" (one of the 25 deg checks failed), or "no_direction".

    The 25 deg gates at `:4196` / `:4225` / `:4246` are the code's own sanity
    test that the back-projected point actually sits BEHIND each shower.  They
    are reported rather than enforced, so the owner can see a vertex the code
    would have thrown away and judge it themselves -- but `verdict` records
    that the code would have refused it.
    """
    out = {"verdict": "no_direction", "vertex": None, "gap": None,
           "theta": None, "mass": None, "angle1": None, "angle2": None,
           "dis1": None, "dis2": None, "len1": None, "len2": None}

    r1, p1s, d1 = gamma_ray(sh1, segments, anchor)
    r2, p2s, d2 = gamma_ray(sh2, segments, anchor)
    if r1 is None or r2 is None:
        return out

    l1 = float(sh1.get("total_length") or 0.0)
    l2 = float(sh2.get("total_length") or 0.0)
    out["len1"], out["len2"] = l1, l2

    if l1 <= 15.0 and l2 <= 15.0:
        out["verdict"] = "both_short"
        return out

    a, b, status = ray_closest_points(r1, r2)
    if status != "ok":
        out["verdict"] = status
        out["vertex"] = vscale(vadd(a, b), 0.5)
        return out

    center = vscale(vadd(a, b), 0.5)
    out["gap"] = vmag(vsub(a, b))

    # `:4183-4191` -- direction from the candidate vertex out to each shower,
    # falling back to the shower's own axis when the vertex lands within 3 cm of
    # the start (where that chord is numerically meaningless).
    v1 = vsub(r1[0], center)
    v2 = vsub(r2[0], center)
    if vmag(v1) < 3.0:
        v1 = d1
    if vmag(v2) < 3.0:
        v2 = d2

    out["vertex"] = center
    out["dis1"] = vmag(vsub(r1[0], center))
    out["dis2"] = vmag(vsub(r2[0], center))
    out["angle1"] = angle_deg(v1, d1)
    out["angle2"] = angle_deg(v2, d2)
    out["theta"] = angle_deg(v1, v2)
    out["mass"] = pi0_mass(sh1.get("kine_charge"), sh2.get("kine_charge"),
                           out["theta"])

    gate1 = out["angle1"] is not None and out["angle1"] > 25.0
    gate2 = out["angle2"] is not None and out["angle2"] > 25.0
    out["verdict"] = "angle_gate" if (gate1 or gate2) else "ok"
    return out


def pi0_groups(showers):
    """The reconstruction's ACTUAL pairing: showers sharing a `pio_id >= 0`.

    This is `TrackFitting::get_map_shower_pio_id()` as dumped at
    `PrDisplayDump.cxx:611-623` -- the pairs the winner loop accepted.

    Do NOT confuse it with the `kine.kine_pio_*` block.  That is a BDT feature
    filled by a SEPARATE highest-energy scan over every candidate pair whether
    accepted or not (`NeutrinoShowerClustering.cxx:3777-3832`, `:4260-4297`), so
    it can name a pair no reconstruction ever accepted.  Measured on ncpi0
    evt21073: the accepted groups are (60081+31023, 127.2 MeV) and
    (11008+63100, 111.2 MeV), while kine_pio_* reports energy_1 = 680.2 (from
    60081) and energy_2 = 104.7 (from 63100) -- a third pairing, mass 207.25.

    That is also why `kine_pio_flag` is not a pi0 selector: doc pr/113 sec 6.4
    measured it firing on 37 of the 48 curated nueCC events.
    """
    groups = {}
    for sh in showers:
        pid = sh.get("pio_id", -1)
        if pid is None or pid < 0:
            continue
        groups.setdefault(pid, []).append(sh)
    return groups
