#!/usr/bin/env python3
# doc pr/138 Phase A -- data model for the shower SPLIT scan tool.  READ-ONLY.
"""Load one scan object and propose an initial grouping.

THE THREE-LEVEL TREE the owner asked for:

    GROUP        a drop target.  "this is one object."  Groups are what the
                 verdict is read off: 1 non-empty group = KEEP, 2 = SPLIT2,
                 3 = SPLIT3, and the JUNK group = TRIM.
      BUNDLE     a spatially connected set of segments -- the thing you drag
                 wholesale, the 'directory'.  This is the middle level, and it
                 exists because dragging 40 segments one at a time is not a scan
                 tool.  Bundles are connected components at `gap` cm (default 4),
                 the same single-linkage idiom production uses
                 (NeutrinoShowerClustering.cxx:8607-8656).
        SEGMENT  one PR segment, the 'file'.  The finest thing the owner can
                 move, and -- this is the load-bearing fact -- also the finest
                 thing the SPLITTER can move: Shower::detach_member_set takes a
                 set of segments (PRShower.cxx:640-700).  So the label space and
                 the action space match exactly (doc pr/137 sec 10.1), and a
                 scan can never ask for a cut the code cannot make.

The proposal that pre-fills the groups is the round-2 trigger + kernel: seeded
angular maxima decide HOW MANY parts (doc pr/137 sec 15.2's valley_best), and the
segment-level ray 2-means decides WHICH (sec 13a, the better kernel).  The owner
corrects it by dragging; the proposal only has to be close enough to be worth
correcting.
"""
import os, sys, json, math, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
import numpy as np
import pr137_lib as L
import pr137_features as F

JUNK = -1                                # the TRIM group
GROUP_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
JUNK_COLOR   = "#999999"


# Category20-ish, for the BUNDLE colour mode.  Deliberately a different and much
# longer palette than GROUP_COLORS: a bundle colour must never be mistaken for a
# group colour, and an object can carry 40+ bundles (evt396222 has 42).
BUNDLE_COLORS = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
                 "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
                 "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
                 "#17becf", "#9edae5"]
# viridis, 9 stops -- the CHARGE mode.  The split criterion IS a charge dip
# between two maxima (doc pr/137 sec 10, ATLAS's local-maxima-with-a-valley), so
# the scanner should be able to see the quantity the trigger reads.
CHARGE_RAMP = ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c",
               "#28ae80", "#5ec962", "#addc30", "#fde725"]


def charge_colors(q):
    """per-point colour on a LOG charge ramp, robust to the dump's negative dQ.

    dQ can be negative (noise-subtracted); those are clamped to the floor rather
    than dropped, so every point keeps a colour and the array stays aligned with
    the cloud."""
    a = np.clip(np.asarray(q, float), 0.0, None)
    pos = a[a > 0]
    if not len(pos):
        return [CHARGE_RAMP[0]] * len(a)
    lo, hi = np.percentile(pos, 5.0), np.percentile(pos, 95.0)
    if not (hi > lo):
        lo, hi = pos.min(), max(pos.max(), pos.min() + 1.0)
    t = (np.log10(np.maximum(a, lo)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    idx = np.clip((t * (len(CHARGE_RAMP) - 1)).round().astype(int),
                  0, len(CHARGE_RAMP) - 1)
    return [CHARGE_RAMP[i] for i in idx]


def connected_bundles(P, segs, ms, gap=4.0):
    """single-linkage over segments at `gap` cm -> list of segment lists,
    ordered by descending charge so bundle 0 is always the main body."""
    from scipy.spatial import cKDTree
    segs = [s for s in segs if s in P and len(P[s])]
    if not segs:
        return []
    tr = {s: cKDTree(P[s][:, :3]) for s in segs}
    par = {s: s for s in segs}
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a, b = segs[i], segs[j]
            if find(a) == find(b):
                continue
            d, _ = tr[a].query(P[b][:, :3], k=1)
            if d.min() < gap:
                par[find(a)] = find(b)
    comp = collections.defaultdict(list)
    for s in segs:
        comp[find(s)].append(s)
    out = [sorted(v) for v in comp.values()]
    out.sort(key=lambda v: -sum(max(ms.get(s, 0.0), 0.0) for s in v))
    return out


def propose(row, sig=None, gap=4.0, flag_junk=False):
    """the pre-filled grouping: {seg_id: group}, plus the bundles and a reason.

    Seeded angular maxima give k (doc pr/137 sec 15.2), the segment-level ray
    2-means gives the assignment (sec 13a).  A bundle is never split by the
    proposal -- it is assigned as a whole by its charge-majority vote, because a
    proposal that cuts through a connected bundle is harder to correct by hand
    than one that is too coarse."""
    sig = sig or L.profile_sigma_fn()
    P, v, segs, ms = row['P'], row['v'], row['segs'], row['ms']
    bundles = connected_bundles(P, segs, ms, gap=gap)
    pts, q, dx = L.pack(P, segs)
    grp = {s: 0 for s in segs}
    reason = "single group (no second angular maximum)"
    if pts is not None and len(pts) >= 8:
        M = L.angular_maxima(pts, q, v, sig, sep_scale=1.6, max_seeds=4)
        k = len(M['dirs'])
        vb = 1.0
        if k >= 2:
            V, fr = M['valley'], M['frac']
            best = None
            for i in range(k):
                for j in range(i + 1, k):
                    if min(fr[i], fr[j]) < 0.03:
                        continue
                    if best is None or V[i, j] < best[0]:
                        best = (float(V[i, j]), i, j)
            if best is not None:
                vb, i, j = best
                if vb <= 0.95:
                    # Assign each BUNDLE to the nearer of the two winning seed
                    # directions, by its own charge-weighted ray from the
                    # reference vertex.  (The first build voted bundles by the
                    # ray 2-means' per-segment labels and collapsed 314838 to
                    # 36 + 1: the 2-means optimises a global partition and its
                    # minority arm need not survive a per-bundle majority.)
                    d0, d1 = M['dirs'][i], M['dirs'][j]
                    for b in bundles:
                        bp, bq, _ = L.pack(P, b)
                        if bp is None:
                            continue
                        c = L.qw_centroid(bp, bq) - v
                        n = np.linalg.norm(c)
                        if n <= 0:
                            continue
                        u = c / n
                        g = 0 if float(u @ d0) >= float(u @ d1) else 1
                        for s in b:
                            grp[s] = g
                    # HONESTY CHECK.  The seeds can be well separated and the
                    # per-bundle vote still send EVERY bundle to the same seed
                    # (evt389538 node19021: valley_best=0.091 yet ng=1), because
                    # a bundle is assigned by its own charge-weighted ray and a
                    # minority lobe that is not its own connected component has
                    # no bundle to carry it.  Saying "2 groups proposed" there is
                    # a lie the scanner cannot see, so the reason reports what
                    # actually happened.
                    if len(set(grp.values())) >= 2:
                        reason = ("2 groups proposed: valley_best=%.3f <= 0.95 "
                                  "(seeded maxima, bundles assigned by ray)" % vb)
                    else:
                        reason = ("single group: valley_best=%.3f <= 0.95 but all "
                                  "%d bundles fell to one seed (no bundle carries "
                                  "the second lobe)" % (vb, len(bundles)))
        if reason.startswith("single") and k >= 2:
            reason = ("single group: %d angular maxima but valley_best=%.3f > 0.95 "
                      "(no charge dip between them)" % (k, vb))
    # JUNK IS NOT PRE-FLAGGED BY DEFAULT, and that is a measurement, not caution.
    # doc pr/137 sec 15.3: HALF of all healthy showers are already spatially
    # fragmented at 2-4 cm, so "small and disconnected" does not imply junk -- a
    # pre-flag on disconnection alone marked 53 of 256587's 128 segments as junk,
    # and 256587 is a textbook single shower.  The owner decides TRIM by eye; the
    # tool only proposes the group split.  flag_junk=True keeps the old rule for
    # experiments, off in the viewer.
    if not flag_junk:
        return grp, bundles, reason
    # A tiny bundle far from ITS OWN group's main body is pre-flagged JUNK.
    # Per-group, not per-object: a global rule measured against bundle 0 swallows
    # the whole second object whenever the second object is itself fragmented
    # (314838 lost all 13 of its group-1 bundles to it on the first build).
    bygrp = collections.defaultdict(list)
    for bi, b in enumerate(bundles):
        g = collections.Counter()
        for s in b:
            g[grp.get(s, 0)] += max(ms.get(s, 0.), 0.)
        bygrp[g.most_common(1)[0][0] if g else 0].append(b)
    for g, bl in bygrp.items():
        if g == JUNK or len(bl) < 2:
            continue
        bl = sorted(bl, key=lambda b: -sum(max(ms.get(s, 0.), 0.) for s in b))
        main = bl[0]
        Qg = sum(max(ms.get(s, 0.), 0.) for b in bl for s in b) or 1.0
        for b in bl[1:]:
            qb = sum(max(ms.get(s, 0.), 0.) for s in b)
            if qb / Qg > 0.05:
                continue
            d = L.min_gap(P, main, b)
            if d == d and d > 8.0:
                for s in b:
                    grp[s] = JUNK
    return grp, bundles, reason


def load_object(event, node, arm='onV1c90', on_tag='emprep-136onV1c90',
                off_tag='emprep-136off2'):
    """one scan object, or None"""
    for r in L.build_population(event, off_tag=off_tag, on_tag=on_tag, arm=arm):
        if r['node'] == node:
            return r
    return None


_DUMP_MEMO = {}


def event_showers(event, arm='onV1c90'):
    """{node: shower record} for the WHOLE event.

    The viewer holds one object; this is how it can still say what that object
    is (particle_id, length, kine) and who its pi0 partner is.  Memoised for one
    event at a time -- the viewer is strictly single-object, and a calib dump is
    ~1-4 MB so a re-read per navigation is free."""
    k = (int(event), arm)
    if k not in _DUMP_MEMO:
        _DUMP_MEMO.clear()
        d = L.dump(int(event), arm)
        _DUMP_MEMO[k] = L.shower_recs(d) if d else {}
    return _DUMP_MEMO[k]


def pio_partner(event, node, arm='onV1c90'):
    """(partner record, pio_mass) for the other shower sharing this pio_id.

    doc pr/138 sec A1.4: when a pi0 is accepted, NeutrinoShowerClustering re-seats
    the MAIN VERTEX at the reconstructed two-photon decay point
    (`main_vertex->fit().point = vtx_point; ... dQ = 0`, :7886 and :6241).  That
    is why evt396222's vertex star sits 14.5 cm off every piece of charge -- it
    is the pi0 decay point, not a failed fit.  The viewer says so instead of
    calling it an extrapolation."""
    SR = event_showers(event, arm)
    me = SR.get(int(node))
    if not me:
        return None, None
    pid = me.get('pio_id', -1)
    if pid is None or pid < 0:
        return None, None
    for n, r in sorted(SR.items()):
        if n != int(node) and r.get('pio_id', -1) == pid:
            return r, me.get('pio_mass')
    return None, me.get('pio_mass')


def theta_phi(pts, q, v):
    """(tx, ty) gnomonic-ish angular map about the object's own axis, in degrees.

    Owner factor 1 made visible: place each member point at radius theta from the
    object's charge-weighted axis and azimuth phi about it.  Two objects are two
    blobs; one object is one blob.  This is the view the one-or-two call is
    actually easiest on, and the 3-D view is slow for it because the separation
    is ANGULAR, not spatial."""
    import numpy as _np
    if len(pts) == 0:
        return _np.zeros(0), _np.zeros(0)
    U, r = L.rays(pts, v)
    c = L.qw_centroid(pts, q)
    ax = c - v
    n = _np.linalg.norm(ax)
    if n <= 0:
        return _np.zeros(len(pts)), _np.zeros(len(pts))
    ax = ax / n
    e1 = _np.cross(ax, [0.0, 0.0, 1.0])
    if _np.linalg.norm(e1) < 1e-6:
        e1 = _np.cross(ax, [0.0, 1.0, 0.0])
    e1 = e1 / _np.linalg.norm(e1)
    e2 = _np.cross(ax, e1)
    th = _np.degrees(_np.arccos(_np.clip(U @ ax, -1.0, 1.0)))
    ph = _np.arctan2(U @ e2, U @ e1)
    return th * _np.cos(ph), th * _np.sin(ph)


def w_single(r):
    """the in-situ single-shower width null, doc pr/137 sec 12.

    w_single(r) = 3.575 + 0.0283 r cm, fitted on 346 SINGLE showers.  Owner
    factor 2 made quantitative -- and NOT a PDG number: the LAr Moliere radius is
    quoted in the doc for scale only and is a threshold nowhere."""
    import numpy as _np
    return 3.575 + 0.0283 * _np.asarray(r, float)


def group_width_profiles(row, grp):
    """[(group, r[], w[])] -- one transverse-RMS-vs-depth curve per non-empty group,
    plus the depth range, so the viewer can draw the null across it."""
    import numpy as _np
    out = []
    lo, hi = None, None
    for g in sorted(set(grp.values())):
        segs = [s for s, gg in grp.items() if gg == g and s in row['P']]
        if not segs:
            continue
        pts, q, _ = L.pack(row['P'], segs)
        if pts is None or len(pts) < 8:
            continue
        r, w = L.width_profile(pts, q, row['v'])
        if not len(r):
            continue
        out.append((g, r, w))
        lo = r.min() if lo is None else min(lo, r.min())
        hi = r.max() if hi is None else max(hi, r.max())
    return out, (lo, hi)


def object_payload(row, gap=4.0):
    """everything the viewer needs for one object, as plain python."""
    grp, bundles, reason = propose(row, gap=gap)
    P, ms, v = row['P'], row['ms'], row['v']
    seg2bundle = {}
    for bi, b in enumerate(bundles):
        for s in b:
            seg2bundle[s] = bi
    segs = []
    for s in sorted(row['segs']):
        A = P.get(s)
        if A is None or not len(A):
            continue
        c = L.qw_centroid(A[:, :3], A[:, 3])
        segs.append(dict(seg=int(s), bundle=int(seg2bundle.get(s, 0)),
                         group=int(grp.get(s, 0)),
                         q=float(max(ms.get(s, 0.0), 0.0)),
                         npts=int(len(A)),
                         length=float(row['M'].get(s, {}).get('length', 0.0)),
                         flag_shower=bool(row['M'].get(s, {}).get('flag_shower', False)),
                         dvtx=float(np.linalg.norm(c - v))))
    return dict(event=int(row['event']), node=int(row['node']),
                Q=float(row['Q']), nseg=len(segs),
                vertex=[float(x) for x in v],
                segs=segs, nbundle=len(bundles), reason=reason,
                proxy=row['cls'])
