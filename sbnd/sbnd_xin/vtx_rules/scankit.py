"""The blind scan kit: everything an AI session needs to hand-scan one event.

Doc pr/80 sec 9.  This replaces `render_event.py`'s single flat PNG, which was
measured (sec 7) to be insufficient: of five misses on the first blind scan, four
were information-presentation failures whose evidence was already in the dump.
Each panel below exists because a specific miss demanded it, and the panel says
which:

  P1 overview         segment identity was invisible -> evt282737, evt283991
  P2 3-D / PCA        a 60 cm shower blob at full-event scale -> evt174637
  P3 zoom (on demand) "zoom into the vertex region", the owner's own workflow
  P4 dQ/dx profiles   hot end blob vs a real Bragg rise -> evt399856
  P5 cone profiles    shower growth, without the contaminated showers[] block
  P6 evidence sheet   the connectivity table that fixes three misses outright

BLINDNESS IS STRUCTURAL, NOT CAREFUL.  `sanitize()` returns a copy of the dump
with every reconstruction-answer field removed, and every renderer here works
only off that copy.  Nothing in this file can read `main_vertex` even by mistake.

That matters more than it first looks.  TaggerCheckNeutrino.cxx:1237-1469 runs
determine_main_vertex -> determine_overall_main_vertex_DL -> improve_vertex ->
examine_direction(main_vertex) -> shower_clustering_with_nv, so `dirsign`,
`dir_weak`, `points[].rr` (which PrDisplayDump.cxx:447-454 reverses according to
`dirsign`) and the whole of `showers[]` are the reconstruction's own answer to
"which end is the vertex end".  Showing any of them to a blind scanner is
handing over the answer in a costume.

  cd sbnd_xin
  python3 vtx_rules/scankit.py prepare --dump <calib.json> --out <dir>
  python3 vtx_rules/scankit.py zoom --dump <calib.json> --vertex 15003 \
      --half-width 8 --out <png>
  python3 vtx_rules/scankit.py selftest          # blindness + determinism
"""
import argparse
import copy
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401,E402  (proj3d)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtx_geom as G                                             # noqa: E402

DQDX_LO, DQDX_HI = 0.0, 150000.0     # e/cm, the display's fixed ramp
MIP = 43000.0                        # meta.mip_dqdx_median
DET_BOX = dict(x=(-201.05, 201.05), y=(-199.312, 199.312), z=(0.85, 500.15))

# Round 3, B2.  Vertices closer than this are one candidate.  See candidates()
# for the measurement that picked 0.8 rather than 1.0 or 1.5.
MERGE_R = 0.8

# Orientation matched to pr_display_viewer.py:205-214 so a picture here and a
# picture on port 5017 can be talked about in the same words.  The old
# render_event.py transposed the third panel, which is a good way to make two
# people describe the same event incompatibly.
PROJ = [("x", "y", "X-Y"), ("z", "y", "Z-Y"), ("x", "z", "X-Z")]

PDG_NAME = {11: "e", -11: "e+", 13: "mu", -13: "mu+", 22: "gamma",
            211: "pi+", -211: "pi-", 321: "K+", -321: "K-",
            2212: "p", 2112: "n", 1: "shwr?", 4: "trk?"}

# ------------------------------------------------------------------ blindness

# Tier 3 of doc pr/80 sec 9: the reconstruction's own answer.  Stripped, not
# merely "not rendered".  `flag_shower`/`shower_id` are here because shower
# membership comes out of shower_clustering_with_nv; `is_main_cluster` because
# swap_main_cluster is called from determine_overall_main_vertex_DL, so it is
# not a clustering property at all (NeutrinoVertexFinder.cxx:4151,4194,4745).
BANNED_TOP = ("main_vertex", "vertex_scoreboard", "showers", "kine", "tagger")
BANNED_VERTEX = ("is_main", "main_candidate", "fit_distance")
BANNED_SEGMENT = ("dirsign", "dir_weak", "flag_shower", "shower_id",
                  "is_main_cluster", "particle_score")
BANNED_POINT = ("rr",)
BANNED = BANNED_TOP + BANNED_VERTEX + BANNED_SEGMENT + BANNED_POINT

KEEP_VERTEX = ("id", "cluster_id", "degree")
KEEP_SEGMENT = ("id", "cluster_id", "particle_id", "length",
                "start_vertex_id", "end_vertex_id")
KEEP_POINT = ("x", "y", "z", "dQ", "dx")


def sanitize(dump):
    """A copy of the dump carrying only what a blind scanner may see.

    Tier 2 (`points[]` geometry, `segments[].length`, the start/end vertex
    connectivity, `particle_id`) is kept: without it the scan is not executable
    at all, and it is exactly what the owner looks at on port 5017.  It is
    nonetheless *vertex-conditioned* -- improve_vertex refits the trajectories
    and re-runs PID after the vertex is chosen -- so the doc states that rather
    than pretending these are raw measurements.
    """
    out = {"meta": copy.deepcopy(dump.get("meta", {})),
           "dqdx_ref": copy.deepcopy(dump.get("dqdx_ref")),
           "vertices": [], "segments": [], "steiner": [], "track_shower": {}}
    for v in dump.get("vertices", []):
        w = {k: v.get(k) for k in KEEP_VERTEX}
        f = v.get("fit") or {}
        w["fit"] = {k: f.get(k) for k in ("x", "y", "z")}
        out["vertices"].append(w)
    for s in dump.get("segments", []):
        w = {k: s.get(k) for k in KEEP_SEGMENT}
        w["points"] = [{k: p.get(k) for k in KEEP_POINT}
                       for p in (s.get("points") or [])]
        out["segments"].append(w)
    for st in dump.get("steiner", []):
        out["steiner"].append({k: st.get(k) for k in
                               ("cluster_id", "x", "y", "z", "flag_terminal")})
    ts = dump.get("track_shower") or {}
    out["track_shower"] = {k: ts.get(k, []) for k in ("x", "y", "z",
                                                      "cluster_id")}
    return out


def assert_blind(obj, where=""):
    """Raise if any tier-3 key name survives into what the scanner is handed.

    A grep, deliberately: `sanitize` is the real guarantee, and this is the
    check that `sanitize` was actually the thing that ran.
    """
    blob = obj if isinstance(obj, str) else json.dumps(obj)
    bad = [w for w in BANNED if ('"%s"' % w) in blob or ("%s=" % w) in blob]
    if bad:
        raise AssertionError("tier-3 field leaked into %s: %s" % (where, bad))


# -------------------------------------------------------------- dump reading


def pdg_name(seg):
    return PDG_NAME.get(seg.get("particle_id"), str(seg.get("particle_id")))


def vertex_xyz(v):
    f = v.get("fit") or {}
    if f.get("x") is None:
        return None
    return (f["x"], f["y"], f["z"])


def clusters_by_size(dump):
    """Cluster ids ordered by total fitted length, longest first.

    The blind stand-in for `vtx_io.main_cluster_id`, which reads `main_vertex`
    and therefore cannot be used here at all.
    """
    tot = {}
    for s in dump.get("segments", []):
        tot[s["cluster_id"]] = tot.get(s["cluster_id"], 0.0) + s.get("length", 0.0)
    return sorted(tot, key=lambda c: (-tot[c], c)), tot


def attached(dump):
    """vertex id -> [segments], and vertex id -> vertex."""
    seg_of = {}
    for s in dump.get("segments", []):
        for k in ("start_vertex_id", "end_vertex_id"):
            vid = s.get(k)
            if vid is not None and vid >= 0:
                seg_of.setdefault(vid, []).append(s)
    for v in seg_of.values():
        v.sort(key=lambda s: s["id"])
    return seg_of, {v["id"]: v for v in dump.get("vertices", [])}


def candidates(dump, merge_r=None):
    """Every PR-graph vertex, all clusters, ordered cluster-by-size then id.

    The main-cluster-only filter this replaces capped the scanner at a 92.0%
    ceiling at 1 cm against 97.7% for the full pool (doc pr/80 sec 9, F1) -- the
    largest single lever in the tooling round, and it was one `if`.

    Vertices within `merge_r` of each other are collapsed to one candidate
    carrying the others as `aliases` (round 3, B2).  The scanner cannot see a
    0.4 cm difference in any panel, so offering both as separate answers asks a
    question the pictures do not contain -- and four of the sixty round-3 events
    were scored wrong by vertex id for picking the twin of the labelled vertex
    while being 0.30-0.76 cm from the click, i.e. right by every physical
    standard.  Radius chosen by measurement, not taste: over all 473 labels,
    0.8 cm collapses 3901 groups and breaks ZERO labels (no representative ends
    up more than 1 cm from its click), while 1.0 cm breaks one and 1.5 cm
    breaks eleven.
    """
    r = MERGE_R if merge_r is None else merge_r
    order, _ = clusters_by_size(dump)
    rank = {c: i for i, c in enumerate(order)}
    out = [v for v in dump.get("vertices", []) if vertex_xyz(v)]
    out.sort(key=lambda v: (rank.get(v["cluster_id"], 1e9), v["id"]))
    if r <= 0:
        return out

    # Single-link grouping in the sorted order, so the result depends only on
    # the data.  Groups are small (a twin or two), so the quadratic scan is
    # cheaper than building an index.
    pos = {v["id"]: vertex_xyz(v) for v in out}
    taken, merged = set(), []
    for v in out:
        if v["id"] in taken:
            continue
        grp = [v]
        taken.add(v["id"])
        for w in out:
            if w["id"] in taken:
                continue
            if any(math.dist(pos[u["id"]], pos[w["id"]]) <= r for u in grp):
                grp.append(w)
                taken.add(w["id"])
        # The most-connected member represents the group: it is the one whose
        # attached segments the scanner is actually looking at.  Ties by lowest
        # id so two runs agree.
        rep = max(grp, key=lambda u: (u.get("degree", 0) or 0, -u["id"]))
        rep = dict(rep)
        rep["aliases"] = sorted(u["id"] for u in grp if u["id"] != rep["id"])
        merged.append(rep)
    merged.sort(key=lambda v: (rank.get(v["cluster_id"], 1e9), v["id"]))
    return merged


def group_ids(v):
    """The vertex's own id plus any it absorbed."""
    return [v["id"]] + list(v.get("aliases") or [])


def attached_merged(dump, cands):
    """rep vertex id -> (outgoing segments, segments internal to the group).

    Merging vertices without merging their connectivity would HIDE segments --
    the opposite of the intent.  A segment whose two ends are both inside one
    group is a sub-centimetre stub between twins, not a prong, so it is reported
    separately rather than counted as evidence for anything.
    """
    seg_of, _ = attached(dump)
    out = {}
    for v in cands:
        ids = set(group_ids(v))
        outgoing, internal, seen = [], [], set()
        for vid in sorted(ids):
            for s in seg_of.get(vid, []):
                if s["id"] in seen:
                    continue
                seen.add(s["id"])
                a, b = s.get("start_vertex_id"), s.get("end_vertex_id")
                if a in ids and b in ids:
                    internal.append(s)
                else:
                    outgoing.append(s)
        outgoing.sort(key=lambda s: s["id"])
        internal.sort(key=lambda s: s["id"])
        out[v["id"]] = (outgoing, internal)
    return out


def all_points(dump, cid=None):
    return [p for s in dump.get("segments", [])
            if cid is None or s["cluster_id"] == cid
            for p in (s.get("points") or [])]


# ------------------------------------------------------------------ plotting


def _seg_color(i):
    return plt.get_cmap("tab20")(i % 20)


def _draw_common(ax, ha, hb, dump, seg_idx, dqdx=True, box=True, alpha=1.0):
    if box:
        (xl, xh), (yl, yh), (zl, zh) = DET_BOX["x"], DET_BOX["y"], DET_BOX["z"]
        lim = dict(x=(xl, xh), y=(yl, yh), z=(zl, zh))
        a0, a1 = lim[ha]
        b0, b1 = lim[hb]
        ax.plot([a0, a1, a1, a0, a0], [b0, b0, b1, b1, b0],
                color="#cc4444", lw=0.8, zorder=1)
        if ha == "x":
            ax.plot([0, 0], [b0, b1], color="#cc4444", lw=0.8, zorder=1)
        if hb == "x":
            ax.plot([a0, a1], [0, 0], color="#cc4444", lw=0.8, zorder=1)

    # Per-segment polylines carry the identity a scatter cannot: the same
    # colour is the same particle in all three panels.  Dimmed under the dQ/dx
    # layer, exactly as the viewer does (pr_display_viewer.py:1242-1244).
    for s in dump.get("segments", []):
        pts = s.get("points") or []
        if len(pts) < 2:
            continue
        ax.plot([p[ha] for p in pts], [p[hb] for p in pts],
                color=_seg_color(seg_idx[s["id"]]), lw=2.0,
                alpha=(0.30 if dqdx else 0.95) * alpha, zorder=2,
                solid_capstyle="round")

    sc = None
    if dqdx:
        mx, my, mc, gx, gy = [], [], [], [], []
        for s in dump.get("segments", []):
            for p in (s.get("points") or []):
                if G.valid_dqdx(p):
                    mx.append(p[ha]); my.append(p[hb]); mc.append(G.dqdx(p))
                else:
                    gx.append(p[ha]); gy.append(p[hb])
        if gx:
            ax.scatter(gx, gy, s=5, c="#9e9e9e", alpha=0.6 * alpha,
                       linewidths=0, zorder=3)
        if mx:
            sc = ax.scatter(mx, my, s=9, c=mc, cmap="turbo", vmin=DQDX_LO,
                            vmax=DQDX_HI, alpha=alpha, linewidths=0, zorder=4)
    return sc


def _draw_vertices(ax, ha, hb, cands, main_cids, fontsize=6):
    for v in cands:
        p = dict(zip("xyz", vertex_xyz(v)))
        big = v["cluster_id"] in main_cids
        ax.plot(p[ha], p[hb], marker="o", ms=6 if big else 3.5, mfc="none",
                mec="#111111" if big else "#8a8a8a", mew=1.1, zorder=6)
        ax.annotate(str(v["id"]), (p[ha], p[hb]), fontsize=fontsize,
                    xytext=(3, 3), textcoords="offset points",
                    color="#111111" if big else "#8a8a8a", zorder=7)


def _data_extent(dump, pad=0.08):
    """Bounding box of the fitted points, with a margin, per coordinate."""
    lo = dict(x=1e9, y=1e9, z=1e9)
    hi = dict(x=-1e9, y=-1e9, z=-1e9)
    for s in dump.get("segments", []):
        for p in (s.get("points") or []):
            for k in "xyz":
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])
    if lo["x"] > hi["x"]:
        return None
    out = {}
    for k in "xyz":
        span = max(hi[k] - lo[k], 20.0)
        out[k] = (lo[k] - pad * span, hi[k] + pad * span)
    return out


def panel_overview(dump, out, title):
    """P1 -- the three standard projections, twice.

    Top row is the whole detector, because containment is real evidence: a
    track that leaves the active volume is a cosmic candidate and its far end
    is not a stopping point.  Bottom row auto-frames on the data, because a
    50 cm event drawn across a 500 cm box puts every candidate vertex inside a
    few pixels -- which is how the first blind scan came to pick a vertex 1.3 cm
    from the right one and be unable to see the difference.
    """
    seg_idx = {s["id"]: i for i, s in enumerate(dump.get("segments", []))}
    cands = candidates(dump)
    order, _ = clusters_by_size(dump)
    main_cids = set(order[:3])
    ext = _data_extent(dump)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0))
    sc = None
    for row in (0, 1):
        for ax, (ha, hb, name) in zip(axes[row], PROJ):
            sc = _draw_common(ax, ha, hb, dump, seg_idx, box=(row == 0)) or sc
            if row == 1:
                _draw_vertices(ax, ha, hb, cands, main_cids, fontsize=5.5)
                if ext:
                    ax.set_xlim(*ext[ha])
                    ax.set_ylim(*ext[hb])
            else:
                (xl, xh), (yl, yh), (zl, zh) = (DET_BOX["x"], DET_BOX["y"],
                                                DET_BOX["z"])
                lim = dict(x=(xl, xh), y=(yl, yh), z=(zl, zh))
                ax.set_xlim(*lim[ha])
                ax.set_ylim(*lim[hb])
            ax.set_xlabel(ha + " [cm]", fontsize=8)
            ax.set_ylabel(hb + " [cm]", fontsize=8)
            ax.set_title("%s -- %s" % (name, "detector" if row == 0
                                       else "zoomed to the event"), fontsize=9)
            ax.set_aspect("equal", adjustable="box" if row == 1 else "datalim")
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.15)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, orientation="horizontal",
                          fraction=0.035, pad=0.07)
        cb.set_label("track-fit dQ/dx [e/cm]  (MIP 43000, 2x MIP 86000; "
                     "fixed range, not per-event)", fontsize=8)
    fig.suptitle("%s   |   P1 overview   |   one colour = one segment; circles "
                 "= candidate vertices, ALL clusters (the truth is often not in "
                 "the biggest one)" % title, fontsize=10)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def panel_3d(dump, out, title):
    """P2 -- the owner's "the real thing is in 3-D" made executable.

    Four rotations plus the cluster's own principal frame.  A track running
    diagonally is foreshortened in X-Y, Z-Y and X-Z simultaneously; no amount of
    zooming on those three fixes it, and rotating does.
    """
    seg_idx = {s["id"]: i for i, s in enumerate(dump.get("segments", []))}
    order, tot = clusters_by_size(dump)
    # Every substantial cluster, not just the biggest: on evt408534 the owner's
    # vertex sits in the THIRD-largest (55 cm, behind 105 and 87), so a panel
    # that drew only the leader would hide the answer.  Single-point orphan
    # clusters ARE dropped -- one stray hit 200 cm away stretches the 3-D axes
    # and squashes the region being scanned into nothing.
    nseg = {}
    for s in dump.get("segments", []):
        nseg[s["cluster_id"]] = nseg.get(s["cluster_id"], 0) + 1
    keep = {c for c in tot if nseg.get(c, 0) >= 2 or tot[c] >= 5.0}
    if not keep:
        keep = set(order[:1])
    segs = [s for s in dump.get("segments", []) if s["cluster_id"] in keep]
    pts = [p for s in segs for p in (s.get("points") or [])]
    cands = [v for v in candidates(dump) if v["cluster_id"] in keep]
    if len(segs) < 4:
        return False

    mx = [p["x"] for p in pts if G.valid_dqdx(p)]
    my = [p["y"] for p in pts if G.valid_dqdx(p)]
    mz = [p["z"] for p in pts if G.valid_dqdx(p)]
    mc = [G.dqdx(p) for p in pts if G.valid_dqdx(p)]

    fig = plt.figure(figsize=(16.5, 9.0))
    views = [(20, -60), (20, 0), (20, 60), (72, -60)]
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        for s in segs:
            p = s.get("points") or []
            if len(p) < 2:
                continue
            ax.plot([q["x"] for q in p], [q["y"] for q in p],
                    [q["z"] for q in p], color=_seg_color(seg_idx[s["id"]]),
                    lw=1.2, alpha=0.55)
        if mx:
            ax.scatter(mx, my, mz, s=5, c=mc, cmap="turbo", vmin=DQDX_LO,
                       vmax=DQDX_HI, linewidths=0, depthshade=False)
        for v in cands:
            q = vertex_xyz(v)
            ax.scatter([q[0]], [q[1]], [q[2]], s=20, facecolors="none",
                       edgecolors="#111111", linewidths=1.0, depthshade=False)
            ax.text(q[0], q[1], q[2], str(v["id"]), fontsize=5.5)
        ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7)
        ax.set_zlabel("z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("3-D  elev %d  azim %d" % (elev, azim), fontsize=9)

    c, axes3 = G.principal_axes(pts)
    for j, (i1, i2, lab) in enumerate([(0, 1, "a1-a2"), (0, 2, "a1-a3")]):
        ax = fig.add_subplot(2, 3, 5 + j)
        if c is None:
            ax.text(0.5, 0.5, "too few points for a principal frame",
                    ha="center", fontsize=8)
            ax.set_axis_off()
            continue

        def proj(q):
            d = (q[0] - c[0], q[1] - c[1], q[2] - c[2])
            return (sum(d[k] * axes3[i1][k] for k in range(3)),
                    sum(d[k] * axes3[i2][k] for k in range(3)))

        for s in segs:
            p = s.get("points") or []
            if len(p) < 2:
                continue
            uv = [proj((q["x"], q["y"], q["z"])) for q in p]
            ax.plot([u for u, _ in uv], [v for _, v in uv],
                    color=_seg_color(seg_idx[s["id"]]), lw=1.8)
        for v in cands:
            u, w = proj(vertex_xyz(v))
            ax.plot(u, w, marker="o", ms=5, mfc="none", mec="#111111", mew=1.0)
            ax.annotate(str(v["id"]), (u, w), fontsize=5.5, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_title("principal frame %s (all clusters)" % lab, fontsize=9)
        ax.set_xlabel("%s [cm]" % lab.split("-")[0], fontsize=7)
        ax.set_ylabel("%s [cm]" % lab.split("-")[1], fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.15)

    fig.suptitle("%s   |   P2 rotations + principal frame   |   clusters with "
                 ">=2 segments or >=5 cm; points coloured by dQ/dx as in P1"
                 % title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return True


def _template(dump, name):
    ref = dump.get("dqdx_ref") or {}
    grid = ref.get("grid") or {}
    y = ref.get(name)
    if not y or not grid:
        return (None, None)
    x = [grid.get("start", 0.0) + i * grid.get("step", 0.25)
         for i in range(len(y))]
    return (x, y)


def panel_profiles(dump, out, title, max_seg=12, min_len=4.0):
    """P4 -- dQ/dx along each segment, measured from BOTH physical ends.

    Never `rr`: PrDisplayDump.cxx:447-454 reverses `rr` according to `dirsign`,
    which examine_direction sets relative to the reconstruction's main vertex.
    A Bragg plot in rr-space silently tells the scanner which end the
    reconstruction believes stops.  Arc length recomputed from points[] has no
    convention to get backwards.

    The stopping-particle templates are drawn anchored at *each* end, dashed, so
    the panel poses the question ("if this end were the stop, a muon would look
    like this") instead of answering it.
    """
    segs = sorted([s for s in dump.get("segments", [])
                   if s.get("length", 0) >= min_len and len(s.get("points") or []) > 3],
                  key=lambda s: -s.get("length", 0))[:max_seg]
    if not segs:
        return False
    n = len(segs)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.1 * nrow),
                             squeeze=False)
    tmu = _template(dump, "muon")
    tpr = _template(dump, "proton")
    seg_idx = {s["id"]: i for i, s in enumerate(dump.get("segments", []))}
    for k, s in enumerate(segs):
        ax = axes[k // ncol][k % ncol]
        pts = s.get("points") or []
        sarc = G.arclen(pts)
        L = sarc[-1]
        xs = [a for a, p in zip(sarc, pts) if G.valid_dqdx(p)]
        ys = [G.dqdx(p) for p in pts if G.valid_dqdx(p)]
        ax.plot(xs, ys, "-o", ms=2.6, lw=1.0,
                color=_seg_color(seg_idx[s["id"]]))
        ax.axhline(MIP, color="#666666", lw=0.7, ls=":")
        ax.axhline(2 * MIP, color="#666666", lw=0.7, ls=":")
        for (tx, ty), col in ((tmu, "#000000"), (tpr, "#8b4513")):
            if not tx:
                continue
            # anchored at the points[0] end (range measured backwards) ...
            ax.plot([L - u for u in tx if u <= L],
                    [v for u, v in zip(tx, ty) if u <= L],
                    color=col, lw=0.8, ls="--", alpha=0.55)
            # ... and at the points[-1] end.
            ax.plot([u for u in tx if u <= L],
                    [v for u, v in zip(tx, ty) if u <= L],
                    color=col, lw=0.8, ls="--", alpha=0.55)
        d0, d1, n0, n1 = G.end_dqdx(s)
        f = lambda d: ("%.0f" % d) if d else "n/a"
        ax.set_title("seg %s  %s  L=%.1f cm\nend@%s: %s   end@%s: %s"
                     % (s["id"], pdg_name(s), L, s.get("start_vertex_id"),
                        f(d0), s.get("end_vertex_id"), f(d1)), fontsize=8)
        ax.set_xlabel("arc length from the %s end [cm]  ->  %s end"
                      % (s.get("start_vertex_id"), s.get("end_vertex_id")),
                      fontsize=7)
        ax.set_ylabel("dQ/dx [e/cm]", fontsize=7)
        ax.set_ylim(0, max(DQDX_HI * 0.9, (max(ys) * 1.15) if ys else 1))
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.15)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].set_axis_off()
    fig.suptitle("%s   |   P4 dQ/dx vs arc length, both ends   |   dashed = "
                 "stopping muon (black) / proton (brown) template anchored at "
                 "each end   |   dotted = 1x, 2x MIP" % title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return True


def panel_cone(dump, out, title, top=6):
    """P5 -- does the cloud open out from this candidate?  showers[] is tier 3.

    Only drawn where it can say anything: a two-segment event has no cone to
    measure and the panel would be noise dressed as evidence.
    """
    nseg = {}
    for s in dump.get("segments", []):
        nseg[s["cluster_id"]] = nseg.get(s["cluster_id"], 0) + 1
    # Only clusters with enough structure to have a cone at all.  On a two-
    # segment track event this panel would be noise wearing the costume of
    # evidence, which is worse than an absent panel.
    busy = {c for c, n in nseg.items() if n >= 5}
    if not busy:
        return False
    cands = [v for v in candidates(dump) if v["cluster_id"] in busy]
    cands.sort(key=lambda v: (-(v.get("degree") or 0), v["id"]))
    cands = cands[:top]
    if not cands:
        return False
    pts_of = {c: all_points(dump, c) for c in busy}
    ncol = 3
    nrow = (len(cands) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.0 * nrow),
                             squeeze=False)
    for k, v in enumerate(cands):
        ax = axes[k // ncol][k % ncol]
        _, prof = G.cone_profile(pts_of[v["cluster_id"]], vertex_xyz(v))
        xs = [l for l, t, _ in prof if t is not None]
        ys = [t for _, t, _ in prof if t is not None]
        trend = G.opening_trend(prof)
        ax.plot(xs, ys, "-o", ms=3, lw=1.2, color="#1f77b4")
        ax.set_title("apex = vertex %s (clus %s, deg %s)   far-near = %s"
                     % (v["id"], v["cluster_id"], v.get("degree"),
                        ("%+.1f cm" % trend) if trend is not None else "n/a"),
                     fontsize=8)
        ax.set_xlabel("distance from apex [cm]", fontsize=7)
        ax.set_ylabel("transverse RMS [cm]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.15)
    for k in range(len(cands), nrow * ncol):
        axes[k // ncol][k % ncol].set_axis_off()
    fig.suptitle("%s   |   P5 cone opening from each candidate apex   |   "
                 "rising = shower-like, flat = track-like" % title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return True


def panel_zoom(dump, out, vid, half=8.0, title=""):
    """P3 -- the vertex region, at a scale where 1 cm is visible.

    The executable form of the owner's "zoom into the vertex region".  Segments
    touching this vertex keep their colour, everything else goes grey, and each
    attached segment gets an arrow pointing toward its own increasing dQ/dx --
    which is the rule-1 question ("does this come out of here?") drawn rather
    than asserted.  Box growth x1,2,4,8 mirrors pr_display_viewer.py:2229-2246.
    """
    seg_idx = {s["id"]: i for i, s in enumerate(dump.get("segments", []))}
    seg_of, vmap = attached(dump)
    v = vmap.get(vid)
    if v is None or vertex_xyz(v) is None:
        raise SystemExit("no vertex %s in this event" % vid)
    c = dict(zip("xyz", vertex_xyz(v)))
    here = {s["id"] for s in seg_of.get(vid, [])}

    for grow in (1, 2, 4, 8):
        h = half * grow
        n = sum(1 for s in dump.get("segments", [])
                for p in (s.get("points") or [])
                if all(abs(p[k] - c[k]) <= h for k in "xyz"))
        if n >= 2:
            break

    cands = candidates(dump)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
    for ax, (ha, hb, name) in zip(axes, PROJ):
        mx, my, mc, gx, gy = [], [], [], [], []
        for s in dump.get("segments", []):
            pts = s.get("points") or []
            if len(pts) < 2:
                continue
            mine = s["id"] in here
            ax.plot([p[ha] for p in pts], [p[hb] for p in pts],
                    color=_seg_color(seg_idx[s["id"]]) if mine else "#cfcfcf",
                    lw=2.4 if mine else 1.0, alpha=0.9 if mine else 0.5,
                    zorder=3 if mine else 2)
            for p in pts:
                if G.valid_dqdx(p):
                    mx.append(p[ha]); my.append(p[hb]); mc.append(G.dqdx(p))
                else:
                    gx.append(p[ha]); gy.append(p[hb])
        if gx:
            ax.scatter(gx, gy, s=10, c="#9e9e9e", linewidths=0, zorder=4)
        if mx:
            ax.scatter(mx, my, s=16, c=mc, cmap="turbo", vmin=DQDX_LO,
                       vmax=DQDX_HI, linewidths=0, zorder=5)
        # arrows: from this vertex, toward the hotter end of each attached seg
        for s in seg_of.get(vid, []):
            end = G.end_name_of_vertex(s, vid)
            d = G.seg_direction(s, end)
            d0, d1, _, _ = G.end_dqdx(s)
            if d is None or d0 is None or d1 is None:
                continue
            far_hotter = (d1 > d0) if end == "start" else (d0 > d1)
            dd = dict(zip("xyz", d))
            ax.annotate("", xy=(c[ha] + dd[ha] * h * 0.55,
                                c[hb] + dd[hb] * h * 0.55),
                        xytext=(c[ha], c[hb]), zorder=8,
                        arrowprops=dict(arrowstyle="-|>", lw=1.4,
                                        color="#2ca02c" if far_hotter
                                        else "#d62728"))
        for w in cands:
            q = dict(zip("xyz", vertex_xyz(w)))
            if any(abs(q[k] - c[k]) > h for k in "xyz"):
                continue
            ax.plot(q[ha], q[hb], marker="o", ms=7 if w["id"] == vid else 5,
                    mfc="none", mec="#111111", mew=1.4 if w["id"] == vid else 1.0,
                    zorder=9)
            ax.annotate(str(w["id"]), (q[ha], q[hb]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points", zorder=9)
        ax.set_xlim(c[ha] - h, c[ha] + h)
        ax.set_ylim(c[hb] - h, c[hb] + h)
        ax.set_xlabel(ha + " [cm]"); ax.set_ylabel(hb + " [cm]")
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.suptitle("%s   |   P3 zoom on vertex %s, half-width %.0f cm   |   "
                 "green arrow = that segment gets HOTTER away from this vertex "
                 "(stops elsewhere);  red = hotter AT this vertex"
                 % (title, vid, h), fontsize=10)
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------- evidence sheet


def sheet(dump, title):
    """P6 -- the connectivity table.  Evidence, never a verdict.

    Three of the five first-round misses (evt282737, evt283991, evt174637) were
    events where the answer is immediate from this table and invisible in a
    scatter plot.  It reports, per attached segment, whether dQ/dx is higher at
    the far end than at this one -- the owner's rule 1 as a measurement.  It does
    NOT rank the vertices: doc pr/80 sec 9 F3 measured the engine's own ranking
    at 8/20 against the unaided eye's 11/20, so shipping its conclusion would
    make the scanner worse.
    """
    order, tot = clusters_by_size(dump)
    seg_of, vmap = attached(dump)
    cands = candidates(dump)
    amerged = attached_merged(dump, cands)
    zs = sorted(vertex_xyz(v)[2] for v in cands)
    L = []
    L.append("%s   --   %d vertices in %d clusters" % (title, len(cands),
                                                       len(order)))
    L.append("clusters by fitted length: " + ", ".join(
        "%s (%.0f cm, %d seg)" % (c, tot[c],
                                  sum(1 for s in dump["segments"]
                                      if s["cluster_id"] == c))
        for c in order[:8]))
    if zs:
        L.append("candidate z range %.1f .. %.1f cm   (low z = upstream)"
                 % (zs[0], zs[-1]))
    L.append("")
    L.append("dQ/dx quoted as the mean over the 5 cm nearest each end, in e/cm."
             "  MIP = 43000.")
    L.append("'unmeasured' is NOT 'low' -- a short or badly fitted end has no"
             " opinion and must not be read as one.")
    L.append("")

    for c in order[:8]:
        vs = [v for v in cands if v["cluster_id"] == c]
        if not vs:
            continue
        L.append("=" * 78)
        L.append("CLUSTER %s   %.1f cm total   %d vertices" % (c, tot[c], len(vs)))
        for v in sorted(vs, key=lambda v: vertex_xyz(v)[2]):
            p = vertex_xyz(v)
            alias = v.get("aliases") or []
            L.append("  vertex %-8s deg %-2s  (%8.1f, %8.1f, %8.1f)%s"
                     % (v["id"], v.get("degree"), p[0], p[1], p[2],
                        ("   [also called %s -- same point within %.1f cm, "
                         "one candidate]" % (", ".join(str(a) for a in alias),
                                             MERGE_R)) if alias else ""))
            outgoing, internal = amerged.get(v["id"], (seg_of.get(v["id"], []),
                                                       []))
            if internal:
                L.append("      (%d sub-cm stub%s between the co-located "
                         "vertices not counted as prongs: %s)"
                         % (len(internal), "" if len(internal) == 1 else "s",
                            ", ".join(str(s["id"]) for s in internal)))
            away = toward = flat = unk = 0
            gids = set(group_ids(v))
            for s in outgoing:
                # The segment may hang off an absorbed twin rather than off the
                # representative, so ask the segment which id it actually holds.
                own = next((i for i in (s.get("start_vertex_id"),
                                        s.get("end_vertex_id")) if i in gids),
                           v["id"])
                end = G.end_name_of_vertex(s, own)
                d0, d1, n0, n1 = G.end_dqdx(s)
                near, far = (d0, d1) if end == "start" else (d1, d0)
                nn, nf = (n0, n1) if end == "start" else (n1, n0)
                far_vid = G.far_vertex(s, own)
                fv = vmap.get(far_vid)
                if near is None or far is None:
                    verdict, unk = "unmeasured", unk + 1
                elif far / near >= 1.3:
                    verdict, away = "RISES AWAY  (x%.2f)" % (far / near), away + 1
                elif near / far >= 1.3:
                    verdict, toward = "rises TOWARD (x%.2f)" % (near / far), toward + 1
                else:
                    verdict, flat = "flat", flat + 1
                L.append("      seg %-8s len %6.1f  %-6s  here %9s (n=%d)"
                         "  far %9s (n=%d)  -> vtx %-8s deg %-2s  %s"
                         % (s["id"], s.get("length", 0), pdg_name(s),
                            ("%.0f" % near) if near else "  -", nn,
                            ("%.0f" % far) if far else "  -", nf,
                            far_vid, (fv or {}).get("degree"), verdict))
            L.append("      => %d of %d attached segments rise away from this "
                     "vertex; %d rise toward it, %d flat, %d unmeasured"
                     % (away, len(outgoing), toward, flat, unk))
    if len(order) > 8:
        L.append("")
        L.append("(%d further clusters omitted, all under %.0f cm)"
                 % (len(order) - 8, tot[order[8]]))
    # The per-cluster tables above stop at the eighth cluster, so without this
    # a scanner reading the sheet would never learn that a candidate it can see
    # in the pictures absorbed a twin.  The merge must be visible wherever it
    # happened, not only where the table happens to reach.
    top = set(order[:8])
    away = [v for v in cands if v.get("aliases") and v["cluster_id"] not in top]
    if away:
        L.append("")
        L.append("merged candidates in the omitted clusters (same point within "
                 "%.1f cm, one candidate each):" % MERGE_R)
        for v in sorted(away, key=lambda v: (v["cluster_id"], v["id"])):
            L.append("  cluster %-5s vertex %-8s also called %s"
                     % (v["cluster_id"], v["id"],
                        ", ".join(str(a) for a in v["aliases"])))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ commands


def prepare(dump_path, out, title=None):
    with open(dump_path) as fh:
        raw = json.load(fh)
    d = sanitize(raw)
    assert_blind(d, "sanitized dump")
    title = title or os.path.basename(dump_path)
    os.makedirs(out, exist_ok=True)
    made = []
    panel_overview(d, os.path.join(out, "p1-overview.png"), title)
    made.append("p1-overview.png")
    if panel_3d(d, os.path.join(out, "p2-3d.png"), title):
        made.append("p2-3d.png")
    if panel_profiles(d, os.path.join(out, "p4-dqdx.png"), title):
        made.append("p4-dqdx.png")
    if panel_cone(d, os.path.join(out, "p5-cone.png"), title):
        made.append("p5-cone.png")
    txt = sheet(d, title)
    assert_blind(txt, "evidence sheet")
    with open(os.path.join(out, "p6-evidence.txt"), "w") as fh:
        fh.write(txt)
    made.append("p6-evidence.txt")
    return made


def selftest():
    """Blindness + determinism, over whatever dumps are reachable."""
    import glob
    import hashlib
    paths = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "work-mcp1k-ma10", "pr_evt*", "calib-pr-evt*.json")))[:25]
    if not paths:
        print("no dumps found"); return 1
    bad = 0
    for p in paths:
        with open(p) as fh:
            raw = json.load(fh)
        # The raw dump must actually contain what we claim to be stripping;
        # a "pass" on a dump with no main_vertex proves nothing.
        if not any(w in raw for w in BANNED_TOP):
            print("SKIP (nothing to strip): %s" % p)
            continue
        d = sanitize(raw)
        try:
            assert_blind(d, p)
            t1 = sheet(d, "t")
            assert_blind(t1, p)
            t2 = sheet(sanitize(raw), "t")
            if hashlib.md5(t1.encode()).hexdigest() != \
               hashlib.md5(t2.encode()).hexdigest():
                print("NON-DETERMINISTIC sheet: %s" % p); bad += 1
        except AssertionError as e:
            print("FAIL %s: %s" % (p, e)); bad += 1
    print("selftest: %d dumps, %d failures" % (len(paths), bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("--dump", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--title")
    p2 = sub.add_parser("zoom")
    p2.add_argument("--dump", required=True)
    p2.add_argument("--vertex", type=int, required=True)
    p2.add_argument("--half-width", type=float, default=8.0)
    p2.add_argument("--out", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "prepare":
        for m in prepare(a.dump, a.out, a.title):
            print(os.path.join(a.out, m))
        return 0
    if a.cmd == "zoom":
        with open(a.dump) as fh:
            raw = json.load(fh)
        d = sanitize(raw)
        panel_zoom(d, a.out, a.vertex, a.half_width,
                   os.path.basename(a.dump))
        print(a.out)
        return 0
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
