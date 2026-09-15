#!/usr/bin/env python3
"""doc pdvd/101 -- toy emulation of the PR-stage chain that turns imaged charge into a fitted
track trajectory and dQ/dx, on the REAL wire lattices of SBND, PDHD and PDVD.

Chain emulated (toolkit file:line of what each step ports):
  truth muon -> per-(plane, wire, 4-tick slice) charge with diffusion + effective width + noise
  -> per-slice blob tiling (strip intersection, refined bounds)
  -> BlobSampler 'stepped'                         clus/src/BlobSampler.cxx:782-977
  -> base graph (same blob + adjacent slices)      stands in for ctpc_ref_pid / basic_pid
  -> Steiner terminals: calc_charge_wcp            clus/src/Facade_Cluster.cxx:1031-1112
                        per-blob peak finding      clus/src/SteinerGrapher.cxx:577-760
                        min-separation thinning    clus/src/SteinerThinning.cxx (doc pdvd/37)
                        extreme points (P4)
  -> Voronoi-union Steiner graph + charge weights  clus/src/SteinerGrapher.cxx:1224-1406
  -> Dijkstra end-to-end = segment wcpts
  -> do_single_tracking                            clus/src/TrackFitting.cxx:10077-10563
       organize_orig_path 1.2 cm                   :2503-2594
       form_map / form_point_association           :4658-4765 / :2940-3340
       examine_point_association                   :3862-4300
       trajectory_fit method 1, then 2             :5397-6060 (fit rows :5594-5840)
       skip_trajectory_point + area smoothing      :6260-6591 / :5921-6015
       organize_ps_path 0.6 cm                     :2827-2937
       dQ_dx_fit                                   :8501-9286 (cal_gaus_integral :6594,
                                                    calculate_compact_matrix :6965)

Deliberate simplifications (stated so nobody reads the toy as the C++):
  * single anode face, no dead channels, no wrapped wires, t0 = 0;
  * the PR retile (ImproveCluster_2) is not re-run: for one clean track the retiled blobs are the
    imaging blobs; its sentinel paint (hack_activity_improved) is not emulated;
  * the base graph is a proxy (all pairs inside a blob, adjacent-slice pairs within a radius,
    components bridged by their closest pair), not connect_graph_ctpc;
  * Steiner phases P2 (reference filter) and P3 (path constraints) are skipped (they remove
    ~6.6 % and 0 % of terminals on PDVD, doc pdvd/31 round 2);
  * examine_end_ps_vec's is_good_point end trim is skipped (interior metrics only);
  * SP is represented by its effective transverse width c (doc pdvd/47) and a flat noise.
Every lever the study scans is an option in Opts below; the production path is Opts().

Units: mm, microseconds, electrons.
"""
import bz2, json, math, os, sys
from dataclasses import dataclass, field, replace
import numpy as np
from scipy.special import erf
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d47_make_xtracks import load_geom, plane_geometry  # noqa: E402

DATA = "/home/xqian/toolkit-dev/wire-cell-data"
CM = 10.0
SQ2 = math.sqrt(2.0)


def cround(x):
    """std::round: half away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


# --------------------------------------------------------------------------------------------
# detector
# --------------------------------------------------------------------------------------------
DETS = {
    # wires file, (anode ident, face index), drift speed mm/us (the value the sim/PR chain uses),
    # sim transport DL/DT cm^2/s, effective transverse non-diffusion width c per plane (mm, the sim
    # S1 floor of doc pdvd/47 sec 12), SP longitudinal add_sigma_L (mm), Steiner terminal
    # threshold (e), terminal_min_separation (mm)
    "sbnd": dict(wires="sbnd-wires-geometry-v0206.json.bz2", anode=0, face=0, v=1.563,
                 DL=4.0, DT=8.8, c=(1.35, 1.33, 0.54), addL=2.4876, term_thr=4000.0, min_sep=0.0),
    "pdhd": dict(wires="protodunehd-wires-larsoft-v1.json.bz2", anode=1, face=0, v=1.576,
                 DL=4.1207, DT=8.2017, c=(2.69, 2.79, 0.39), addL=2.0902, term_thr=500.0, min_sep=5.0),
    "pdvd": dict(wires="protodunevd-wires-larsoft-v7-uvwfit.json.bz2", anode=1, face=0, v=1.568,
                 DL=4.1307, DT=7.9135, c=(1.29, 1.27, 0.54), addL=1.9639, term_thr=500.0, min_sep=5.0),
}


@dataclass
class Detector:
    name: str
    pitch: np.ndarray       # (3,)
    pdir: np.ndarray        # (3,2) unit pitch direction in (y,z)
    c0: np.ndarray          # (3,2) centre of wire 0 in (y,z)
    angle: np.ndarray       # (3,) WCT wire angle: pdir = (-sin a, cos a)
    v: float
    tick: float = 500.0e-3 * 1000.0 / 1000.0  # 0.5 us
    nt: int = 4
    DL: float = 0.0          # mm^2/us
    DT: float = 0.0
    c: np.ndarray = None
    addL: float = 0.0
    term_thr: float = 4000.0
    min_sep: float = 0.0

    @property
    def ttw(self):          # mm of drift per tick (TrackFitting's time_tick_width)
        return self.v * self.tick


_GEOM_CACHE = {}


def load_detector(name):
    if name in _GEOM_CACHE:
        return _GEOM_CACHE[name]
    d = DETS[name]
    anodes, faces, planes, wires, P = load_geom(os.path.join(DATA, d["wires"]))
    an = next(a for a in anodes if a["ident"] == d["anode"])
    face = faces[an["faces"][d["face"]]]
    pg = [plane_geometry(planes[pi], wires, P) for pi in face["planes"][:3]]
    pitch = np.array([g["pitch"] for g in pg])
    pdir = np.array([g["pdir"][1:] / np.linalg.norm(g["pdir"][1:]) for g in pg])
    c0 = np.array([g["c0"][1:] for g in pg])
    angle = np.array([math.atan2(-p[0], p[1]) for p in pdir])
    det = Detector(name=name, pitch=pitch, pdir=pdir, c0=c0, angle=angle, v=d["v"], tick=0.5,
                   DL=d["DL"] * 1e-4, DT=d["DT"] * 1e-4, c=np.array(d["c"]), addL=d["addL"],
                   term_thr=d["term_thr"], min_sep=d["min_sep"])
    _GEOM_CACHE[name] = det
    return det


def wire_cont(det, l, y, z):
    """continuous wire coordinate: integer = wire centre (TrackFitting.cxx:829-857 convention)."""
    return ((y - det.c0[l, 0]) * det.pdir[l, 0] + (z - det.c0[l, 1]) * det.pdir[l, 1]) / det.pitch[l]


def yz_from_wires(det, l1, w1, l2, w2):
    """(y,z) where plane l1 wire coordinate = w1 and plane l2 = w2."""
    A = np.array([det.pdir[l1], det.pdir[l2]])
    b = np.array([w1 * det.pitch[l1] + det.pdir[l1] @ det.c0[l1],
                  w2 * det.pitch[l2] + det.pdir[l2] @ det.c0[l2]])
    return np.linalg.solve(A, b)


# --------------------------------------------------------------------------------------------
# options = the levers
# --------------------------------------------------------------------------------------------
@dataclass
class Opts:
    # sampling (BlobSampler::Stepped)
    min_step: float = 3.0
    max_step_fraction: float = 1.0 / 12.0
    half_pitch: bool = False        # also sample at half-wire positions in the min/max views
    x_mid: bool = False             # place sampled points at the slice centre, not its start
    # 'both' = the sampling levers change every sampled point (imaging + retile, the round-1/2 toy);
    # 'steiner' = they change only the cloud the Steiner stage builds from (the PR job's retile
    # samplers), while form_point_association reads the production-sampled cluster points -- what a
    # PR-job-only sampler knob can actually do in the C++ chain
    sampler_scope: str = "both"
    # terminals
    term_centroid: bool = False     # move terminals to the per-slice 3-view charge centroid
    min_sep: float = None           # None = detector production value (mm)
    min_sep_pitch: float = 0.0      # if > 0: min_sep = this * max pitch
    # fit
    assoc_floor_pitch: float = 0.0  # form_point_association: per-plane dis_cut floor = k * pitch
    assoc_cont_center: bool = False # centre the association wire window on the continuous coordinate
    fit_weight_pow: float = 2.0     # position LSQ weight = (q/err * fac)^pow ; production 2
    div_sigma: float = 6.0          # mm ; charge sharing width of method 2
    div_sigma_pitch: float = 0.0    # if > 0: div_sigma = max(div_sigma, k * pitch of the plane)
    # oracles
    seed: str = "steiner"           # 'steiner' | 'truth'
    final_fill_charge_test: bool = True   # traj_final_fill_charge_test (PDHD/PDVD production = 1)
    # fit model (the sim-matched TrackFitting constants; None = detector truth values)
    fit_DT: float = None
    fit_c: tuple = None
    lam: float = 5e-4


# --------------------------------------------------------------------------------------------
# truth + charge
# --------------------------------------------------------------------------------------------
def make_track(det, length_cm, theta_deg, phi_deg, xmid_cm=150.0, ymid=None, zmid=None, rng=None):
    """straight track; theta = angle to the drift (x) axis, phi = azimuth in (y,z) measured from the
    collection-wire direction.  Returns (p0, p1) in mm."""
    th = math.radians(theta_deg); ph = math.radians(phi_deg)
    pw = det.pdir[2]; ww = np.array([-pw[1], pw[0]])          # W wire direction in (y,z)
    dyz = math.sin(th) * (math.cos(ph) * ww + math.sin(ph) * pw)
    d = np.array([math.cos(th), dyz[0], dyz[1]])
    if ymid is None:
        ymid = det.c0[2, 0] + 400.0
    if zmid is None:
        zmid = det.c0[2, 1] + 800.0
    if rng is not None:                                        # random sub-pitch phase of the track
        ymid += rng.uniform(0, 10); zmid += rng.uniform(0, 10); xmid_cm += rng.uniform(0, 1)
    mid = np.array([xmid_cm * CM, ymid, zmid])
    h = 0.5 * length_cm * CM
    return mid - h * d, mid + h * d


def deposit(det, p0, p1, dqdx=5000.0, step=0.5, noise=300.0, thr=900.0, rng=None, fit_c=None, fluct=0.0):
    """Charge per (plane, slice, wire).  Anode at x = 0, drift distance = x.  Returns
    cells[l] = dict((slice_tick, wire) -> [q_meas, q_true]) of cells with q_true > 1 e, and the
    active-cell sets (q_meas > thr).  fluct > 0: every 3 mm of track carries a Moyal-distributed
    charge factor with that scale, truncated at 5 and normalised to unit mean over the track (a
    Landau-like physical fluctuation; 0 = the flat truth, so any dQ/dx structure is the chain's)."""
    rng = rng or np.random.default_rng(0)
    L = np.linalg.norm(p1 - p0); n = int(L / step)
    s = (np.arange(n) + 0.5) / n
    P = p0[None, :] + s[:, None] * (p1 - p0)[None, :]
    dq = dqdx * L / n
    fac = np.ones(n)
    if fluct > 0:
        nseg = max(1, int(L / 3.0))
        from scipy.stats import moyal
        f = np.minimum(moyal.rvs(loc=1.0, scale=fluct, size=nseg, random_state=rng), 5.0)
        f = np.maximum(f, 0.05); f /= f.mean()
        fac = f[np.minimum((s * nseg).astype(int), nseg - 1)]
    t = P[:, 0] / det.v                                        # drift time (us)
    T = t / det.tick                                           # arrival tick
    sig_t = np.sqrt(2 * det.DL * t + det.addL ** 2) / det.ttw  # ticks
    cells = [dict(), dict(), dict()]
    for l in range(3):
        c = det.c[l]
        sig_w = np.sqrt(2 * det.DT * t + c ** 2) / det.pitch[l]
        w = wire_cont(det, l, P[:, 1], P[:, 2])
        acc = {}
        for i in range(n):
            w0 = math.floor(w[i] - 4 * sig_w[i]); w1 = math.ceil(w[i] + 4 * sig_w[i])
            ws = np.arange(w0, w1 + 1)
            fw = 0.5 * (erf((ws + 0.5 - w[i]) / SQ2 / sig_w[i]) - erf((ws - 0.5 - w[i]) / SQ2 / sig_w[i]))
            s0 = math.floor((T[i] - 4 * sig_t[i]) / det.nt); s1 = math.floor((T[i] + 4 * sig_t[i]) / det.nt)
            ss = np.arange(s0, s1 + 1)
            ft = 0.5 * (erf((ss * det.nt + det.nt - T[i]) / SQ2 / sig_t[i]) - erf((ss * det.nt - T[i]) / SQ2 / sig_t[i]))
            for a, sv in enumerate(ss):
                if ft[a] < 1e-6:
                    continue
                for b, wv in enumerate(ws):
                    v = dq * fac[i] * ft[a] * fw[b]
                    if v < 1e-3:
                        continue
                    k = (int(sv * det.nt), int(wv))
                    acc[k] = acc.get(k, 0.0) + v
        for k, v in acc.items():
            if v > 1.0:
                cells[l][k] = [v + rng.normal(0, noise), v]
    active = [{k: v for k, v in cells[l].items() if v[0] > thr} for l in range(3)]
    return P, active


def charge_err(q):
    """measured PDHD/PDVD/SBND T_proj_data err vs q (doc pdvd/101 sec 1): ~flat 1.1-1.8 ke."""
    return math.sqrt(1100.0 ** 2 + (0.04 * q) ** 2)


# --------------------------------------------------------------------------------------------
# imaging: per-slice tiling
# --------------------------------------------------------------------------------------------
def _runs(ws):
    ws = sorted(ws); out = []; s = ws[0]; p = ws[0]
    for w in ws[1:]:
        if w != p + 1:
            out.append((s, p + 1)); s = w
        p = w
    out.append((s, p + 1))
    return out


def _clip(poly, g, lo, hi):
    """clip polygon (list of yz) by lo <= g.yz <= hi."""
    def clip1(pts, f):
        out = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            fa, fb = f(a), f(b)
            if fa >= 0:
                out.append(a)
            if (fa >= 0) != (fb >= 0):
                tt = fa / (fa - fb); out.append(a + tt * (b - a))
        return out
    poly = clip1(poly, lambda p: g @ p - lo)
    if not poly:
        return poly
    return clip1(poly, lambda p: hi - g @ p)


def tile(det, active):
    """blobs: list of dict(slice, strips[(lo,hi) ray bounds per plane], poly)."""
    slices = sorted(set(k[0] for l in range(3) for k in active[l]))
    blobs = []
    for ts in slices:
        runs = []
        for l in range(3):
            ws = [k[1] for k in active[l] if k[0] == ts]
            if not ws:
                break
            runs.append(_runs(ws))
        if len(runs) < 3:
            continue
        for ru in runs[0]:
            for rv in runs[1]:
                for rw in runs[2]:
                    poly = [np.array(p) for p in ([-1e5, -1e5], [1e5, -1e5], [1e5, 1e5], [-1e5, 1e5])]
                    for l, r in enumerate((ru, rv, rw)):
                        g = det.pdir[l] / det.pitch[l]; off = det.pdir[l] @ det.c0[l] / det.pitch[l]
                        poly = _clip(poly, g, r[0] - 0.5 + off, r[1] - 0.5 + off)
                        if len(poly) < 3:
                            break
                    if len(poly) < 3:
                        continue
                    Pp = np.array(poly)
                    area = 0.5 * abs(np.dot(Pp[:, 0], np.roll(Pp[:, 1], 1)) - np.dot(Pp[:, 1], np.roll(Pp[:, 0], 1)))
                    if area < 1e-3:
                        continue
                    strips = []
                    for l, r in enumerate((ru, rv, rw)):
                        wc = [wire_cont(det, l, p[0], p[1]) for p in poly]
                        lo = max(r[0], math.floor(min(wc) + 0.5 + 1e-6)); hi = min(r[1], math.ceil(max(wc) + 0.5 - 1e-6))
                        strips.append((lo, max(hi, lo + 1)))
                    blobs.append(dict(slice=ts, strips=strips, poly=Pp))
    return blobs


# --------------------------------------------------------------------------------------------
# sampling: BlobSampler::Stepped
# --------------------------------------------------------------------------------------------
def sample(det, blobs, active, o: Opts):
    pts = []; blob_of = []; charges = []; wires = []
    for bi, b in enumerate(blobs):
        st = b["strips"]
        width = [s[1] - s[0] for s in st]
        mx = 0
        if width[1] > width[mx]: mx = 1
        if width[2] > width[mx]: mx = 2
        mn = 1
        if width[0] < width[mn]: mn = 0
        if width[2] < width[mn]: mn = 2
        md = [i for i in range(3) if i != mx and i != mn]
        md = md[0] if md else 2
        nmin = int(max(o.min_step, o.max_step_fraction * width[mn]))
        nmax = int(max(o.min_step, o.max_step_fraction * width[mx]))
        smin = sorted(set(list(range(st[mn][0], st[mn][1], nmin)) + [st[mn][1] - 1]))
        smax = sorted(set(list(range(st[mx][0], st[mx][1], nmax)) + [st[mx][1] - 1]))
        if o.half_pitch:                                       # lever A2: half-wire sub-steps
            smin = sorted(set(smin + [w + 0.5 for w in smin if w + 0.5 < st[mn][1] - 0.5]))
            smax = sorted(set(smax + [w + 0.5 for w in smax if w + 0.5 < st[mx][1] - 0.5]))
        x = (b["slice"] + (det.nt / 2.0 if o.x_mid else 0.0)) * det.ttw
        for a in smin:
            for c in smax:
                y, z = yz_from_wires(det, mn, a, mx, c)       # the wire-centre crossing (offset 0.5)
                pr = wire_cont(det, md, y, z) + 0.5            # mid-view ray-relative location
                if not (st[md][0] - 0.03 < pr < st[md][1] + 0.03):
                    continue
                p = np.array([x, y, z])
                qs = []; wl = []
                for l in range(3):
                    wi = int(math.floor(wire_cont(det, l, y, z) + 0.1 / det.pitch[l] + 0.5))
                    cell = active[l].get((b["slice"], wi))
                    qs.append(cell[0] if cell else 0.0); wl.append(wi)
                pts.append(p); blob_of.append(bi); charges.append(qs); wires.append(wl)
    return np.array(pts), np.array(blob_of), np.array(charges), np.array(wires)


# --------------------------------------------------------------------------------------------
# graph + Steiner
# --------------------------------------------------------------------------------------------
def base_graph(det, pts, blob_of, blobs):
    n = len(pts)
    tree = cKDTree(pts)
    R = 1.5 * math.hypot(det.nt * det.ttw, det.pitch.max())
    rows, cols, vals = [], [], []
    for i, j in tree.query_pairs(R):
        same = blob_of[i] == blob_of[j]
        ds = abs(blobs[blob_of[i]]["slice"] - blobs[blob_of[j]]["slice"])
        if same or ds <= 2 * det.nt:
            d = float(np.linalg.norm(pts[i] - pts[j])) + 1e-6
            rows += [i, j]; cols += [j, i]; vals += [d, d]
    G = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    ncomp, lab = connected_components(G, directed=False)
    base_graph.ncomp_raw = ncomp
    # Borůvka bridging (the spirit of connect_graph's MST over components): every round, each
    # component adds its single shortest edge to any OTHER component; repeat until connected.
    while ncomp > 1:
        extra = {}
        for c in range(ncomp):
            a = np.where(lab == c)[0]; b = np.where(lab != c)[0]
            tb = cKDTree(pts[b]); dd, kk = tb.query(pts[a])
            m = int(np.argmin(dd))
            i, j = int(a[m]), int(b[kk[m]])
            key = (min(i, j), max(i, j))
            extra[key] = float(dd[m]) + 1e-6
        for (i, j), d in extra.items():
            rows += [i, j]; cols += [j, i]; vals += [d, d]
        G = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        G.sum_duplicates()
        ncomp, lab = connected_components(G, directed=False)
    return G


def adjacency(G):
    return [G.indices[G.indptr[i]:G.indptr[i + 1]] for i in range(G.shape[0])]


def calc_charge_wcp(qs, cut):
    flags = [q > cut or q == 0 for q in qs]
    nz = [q for q in qs if q != 0]
    ch = math.sqrt(sum(q * q for q in nz) / len(nz)) if len(nz) > 1 else 0.0
    return all(flags), ch


def find_terminals(det, pts, blob_of, charges, G, adj, o: Opts):
    thr = det.term_thr
    terms = set()
    for bi in np.unique(blob_of):
        idx = np.where(blob_of == bi)[0]
        chg = {}
        cand = []
        for i in idx:
            ok, c = calc_charge_wcp(charges[i], thr)
            chg[i] = c
            if c > thr and ok:
                cand.append((c, i))
        if not cand:
            continue
        cand.sort(reverse=True)                                # std::greater<pair<double,size_t>>
        peaks, nonpeaks = set(), set()
        for c, i in cand:
            nb = set(adj[i]) - {i}
            if not peaks:
                peaks.add(i)
                for j in nb:
                    if j in chg and c > chg[j]:
                        nonpeaks.add(j)
                continue
            if i in peaks or i in nonpeaks:
                continue
            ins = True
            for j in nb:
                if j not in chg:
                    continue
                if c > chg[j]:
                    nonpeaks.add(j)
                elif c < chg[j]:
                    ins = False; break
            if ins:
                peaks.add(i)
        if len(peaks) > 1:                                     # merge directly connected peaks
            pv = sorted(peaks)
            parent = list(range(len(pv)))
            def fnd(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]; a = parent[a]
                return a
            for a in range(len(pv)):
                for b in range(a + 1, len(pv)):
                    if G[pv[a], pv[b]] != 0:
                        parent[fnd(a)] = fnd(b)
            groups = {}
            for a in range(len(pv)):
                groups.setdefault(fnd(a), []).append(pv[a])
            peaks = set()
            for g in groups.values():
                cm = pts[g].mean(axis=0)
                peaks.add(min(g, key=lambda k: (np.sum((pts[k] - cm) ** 2), k)))
        terms |= peaks
    # P3b thinning
    R = o.min_sep if o.min_sep is not None else det.min_sep
    if o.min_sep_pitch > 0:
        R = o.min_sep_pitch * det.pitch.max()
    if R > 0 and terms:
        order = sorted(terms, key=lambda i: (calc_charge_wcp(charges[i], thr)[1], i), reverse=True)
        kept = []
        for i in order:
            if all(np.sum((pts[i] - pts[k]) ** 2) >= R * R for k in kept):
                kept.append(i)
        terms = set(kept)
    # P4 extremes (main axis)
    cm = pts.mean(axis=0); u = np.linalg.svd(pts - cm, full_matrices=False)[2][0]
    proj = (pts - cm) @ u
    e0, e1 = int(np.argmin(proj)), int(np.argmax(proj))
    terms |= {e0, e1}
    return terms, (e0, e1)


def steiner(det, pts, charges, G, terms, ends):
    tl = sorted(terms)
    dist, pred, src = dijkstra(G, directed=False, indices=tl, min_only=True, return_predecessors=True)
    Gc = G.tocoo()
    best = {}
    for a, b, w in zip(Gc.row, Gc.col, Gc.data):
        if a >= b:
            continue
        sa, sb = src[a], src[b]
        if sa == sb or sa < 0 or sb < 0:
            continue
        key = (min(sa, sb), max(sa, sb)); tot = dist[a] + w + dist[b]
        if key not in best or tot < best[key][0]:
            best[key] = (tot, a, b)
    edges = set()
    for _, a, b in best.values():
        edges.add((min(a, b), max(a, b)))
        for e in (a, b):
            cur = e
            while src[cur] != cur and pred[cur] >= 0:
                p = pred[cur]; edges.add((min(p, cur), max(p, cur))); cur = p
    verts = sorted(set(v for e in edges for v in e) | set(tl))
    q = {v: calc_charge_wcp(charges[v], 4000.0)[1] for v in verts}
    rows, cols, vals = [], [], []
    for a, b in edges:
        wgeo = G[a, b]
        f = 0.8 + 0.4 * (0.5 * 1e4 / (q[a] + 1e4) + 0.5 * 1e4 / (q[b] + 1e4))
        rows += [a, b]; cols += [b, a]; vals += [wgeo * f, wgeo * f]
    n = len(pts)
    S = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    d, pr = dijkstra(S, directed=False, indices=ends[0], return_predecessors=True)
    path = []; cur = ends[1]
    if not np.isfinite(d[cur]):
        return verts, S, [ends[0], ends[1]]
    while cur != ends[0] and cur >= 0:
        path.append(cur); cur = pr[cur]
    path.append(ends[0])
    return verts, S, path[::-1]


def centroid_positions(det, pts, blob_of, blobs, active, idx):
    """lever B1: per terminal, the charge centroid of its slice in each plane (+-2 wires around the
    terminal's own wire), then the least-squares (y,z) of the three plane coordinates weighted by
    charge.  x unchanged."""
    out = {}
    for i in idx:
        ts = blobs[blob_of[i]]["slice"]
        y, z = pts[i, 1], pts[i, 2]
        A = np.zeros((2, 2)); bb = np.zeros(2); ok = 0
        for l in range(3):
            w0 = cround(wire_cont(det, l, y, z))
            num = den = 0.0
            for wv in range(w0 - 2, w0 + 3):
                c = active[l].get((ts, wv))
                if c:
                    num += c[0] * wv; den += c[0]
            if den <= 0:
                continue
            wc = num / den
            g = det.pdir[l] / det.pitch[l]; off = det.pdir[l] @ det.c0[l] / det.pitch[l]
            A += den * np.outer(g, g); bb += den * g * (wc + off); ok += 1
        if ok >= 2:
            yz = np.linalg.solve(A, bb)
            out[i] = np.array([pts[i, 0], yz[0], yz[1]])
    return out


# --------------------------------------------------------------------------------------------
# track fitting
# --------------------------------------------------------------------------------------------
def calc_ranges(angles, rem, mind, pitch):
    cuv = abs(math.cos(angles[0] - angles[1])); cuw = abs(math.cos(angles[0] - angles[2])); cvw = abs(math.cos(angles[1] - angles[2]))
    s = cuv + cuw + cvw
    au = rem[0] * s - cuv * (mind[1] * pitch[1]) ** 2 / cvw - cuw * (mind[2] * pitch[2]) ** 2 / cvw
    av = rem[1] * s - cuv * (mind[0] * pitch[0]) ** 2 / cuw - cvw * (mind[2] * pitch[2]) ** 2 / cuw
    aw = rem[2] * s - cuw * (mind[0] * pitch[0]) ** 2 / cuv - cvw * (mind[1] * pitch[1]) ** 2 / cuv
    return [max(au, 0.0), max(av, 0.0), max(aw, 0.0)]


class Fitter:
    def __init__(self, det, pts, blob_of, blobs, G, active, steiner_verts, S, o: Opts, spts=None):
        self.det, self.o = det, o
        self.pts, self.blob_of, self.blobs, self.active = pts, blob_of, blobs, active
        self.spts = pts if spts is None else spts                # the cloud steiner_verts index into
        self.adj = adjacency(G)
        self.tree = cKDTree(pts)
        self.sv = np.array(steiner_verts)
        self.stree = cKDTree(self.spts[self.sv])
        self.sadj = adjacency(S)
        self.cells = [{k: (v[0], charge_err(v[0])) for k, v in active[l].items()} for l in range(3)]
        self.fit_DT = det.DT if o.fit_DT is None else o.fit_DT
        self.fit_c = det.c if o.fit_c is None else np.array(o.fit_c)

    # --- helpers ---------------------------------------------------------------------------
    def tind(self, x):
        return cround(x / self.det.ttw)

    def cur_ts(self, x):
        return int(math.floor(self.tind(x) / self.det.nt) * self.det.nt)

    def nlevel(self, adj, start, n):
        found = {start}; front = {start}
        for _ in range(n):
            nxt = set()
            for v in front:
                for u in adj[v]:
                    if u not in found:
                        found.add(u); nxt.add(u)
            front = nxt
        return found

    def _window(self, l, cw_int, cw_cont, half):
        c = cw_cont if self.o.assoc_cont_center else cw_int
        return range(cround(c - half), cround(c + half) + 1)

    # --- form_point_association -------------------------------------------------------------
    def associate(self, p, dis_cut):
        det, o = self.det, self.o
        assoc = [set(), set(), set()]
        d, ci = self.tree.query(p)
        cw_cont = [wire_cont(det, l, p[1], p[2]) for l in range(3)]
        cw = [cround(c) for c in cw_cont]
        cts = self.cur_ts(p[0])
        floor = [o.assoc_floor_pitch * det.pitch[l] for l in range(3)]
        if d < dis_cut:
            nb = self.nlevel(self.adj, int(ci), 3)
            bl = sorted(set(int(self.blob_of[v]) for v in nb))
            max_ts = [0.0, 0.0, 0.0]
            for b in bl:
                st = self.blobs[b]["strips"]; ts = self.blobs[b]["slice"]
                for l in range(3):
                    if st[l][0] - 1 <= cw[l] < st[l][1] + 1:
                        max_ts[l] = max(max_ts[l], abs(ts - cts))
            dcl = [dis_cut] * 3
            for l in range(3):
                if max_ts[l] * det.ttw * 1.2 < dcl[l]:
                    dcl[l] = max_ts[l] * det.ttw * 1.2
                dcl[l] = max(dcl[l], floor[l])
            for b in bl:
                st = self.blobs[b]["strips"]; ts = self.blobs[b]["slice"]
                rem = [dcl[l] ** 2 - ((cts - ts) * det.ttw) ** 2 for l in range(3)]
                if (max(rem) > 0) and abs(cts - ts) <= 20:
                    mind = []
                    for l in range(3):
                        if cw[l] < st[l][0]:
                            mind.append(st[l][0] - cw[l])
                        elif cw[l] < st[l][1]:
                            mind.append(0)
                        else:
                            mind.append(cw[l] - st[l][1] + 1)
                    rg = calc_ranges(det.angle, rem, mind, det.pitch)
                    if min(rg) > 0:
                        for l in range(3):
                            half = math.sqrt(rg[l]) / det.pitch[l]
                            for j in self._window(l, cw[l], cw_cont[l], half):
                                assoc[l].add((ts, j))
        # Steiner branch
        ds, si = self.stree.query(p)
        if ds < dis_cut:
            sv0 = int(self.sv[si]); q0 = self.spts[sv0]
            scw_cont = [wire_cont(det, l, q0[1], q0[2]) for l in range(3)]
            scw = [cround(c) for c in scw_cont]
            sts = self.cur_ts(q0[0])
            nb = self.nlevel(self.sadj, sv0, 3)
            mtw = {}
            for v in nb:
                if v not in set(self.sv.tolist()) and v != sv0:
                    pass
                qv = self.spts[v]
                vts = self.cur_ts(qv[0])
                ws = [cround(wire_cont(det, l, qv[1], qv[2])) for l in range(3)]
                if vts not in mtw:
                    mtw[vts] = [[ws[l], ws[l] + 1] for l in range(3)]
                else:
                    for l in range(3):
                        mtw[vts][l][0] = min(mtw[vts][l][0], ws[l]); mtw[vts][l][1] = max(mtw[vts][l][1], ws[l] + 1)
            mts = [0.0, 0.0, 0.0]
            for vts, rgw in mtw.items():
                for l in range(3):
                    for wv in range(rgw[l][0], rgw[l][1]):
                        if abs(wv - scw[l]) * det.pitch[l] <= dis_cut:
                            mts[l] = max(mts[l], abs(vts - sts))
            dcl = [dis_cut] * 3
            for l in range(3):
                if mts[l] * det.ttw * 1.2 < dcl[l]:
                    dcl[l] = mts[l] * det.ttw * 1.2
                dcl[l] = max(dcl[l], floor[l])
            for vts, rgw in mtw.items():
                rem = [dcl[l] ** 2 - ((sts - vts) * det.ttw) ** 2 for l in range(3)]
                if max(rem) > 0 and abs(sts - vts) <= 20:
                    mind = []
                    for l in range(3):
                        lo, hi = rgw[l]
                        if scw[l] < lo:
                            mind.append(lo - scw[l])
                        elif scw[l] >= hi:
                            mind.append(scw[l] - hi + 1)
                        else:
                            mind.append(0)
                    rg = calc_ranges(det.angle, rem, mind, det.pitch)
                    if min(rg) > 0:
                        for l in range(3):
                            half = math.sqrt(rg[l]) / det.pitch[l]
                            c = scw_cont[l] if o.assoc_cont_center else scw[l]
                            for j in range(cround(c - half), cround(c + half) + 1):
                                assoc[l].add((vts, j))
        if not any(assoc):
            for l in range(3):
                assoc[l].add((cts, cw[l]))
        return assoc, cw, self.tind(p[0])

    # --- examine_point_association ------------------------------------------------------------
    def examine(self, p, assoc, cw, tind, end_point, charge_cut=2000.0):
        saved = [set(), set(), set()]; qty = [0.0, 0.0, 0.0]
        for l in range(3):
            for k in assoc[l]:
                c = self.cells[l].get(k)
                if c and c[0] > charge_cut:
                    saved[l].add(k)
            qty[l] = len(saved[l]) / len(assoc[l]) if assoc[l] else 0.0
        empty = [len(s) == 0 for s in saved]
        if sum(empty) == 2:
            for l in range(3):
                if empty[l]:
                    saved[l].add((tind, cw[l]))
        return saved, qty

    # --- form_map -------------------------------------------------------------------------------
    def form_map(self, ps, ef=0.6, mf=0.9):
        n = len(ps)
        dist = [float(np.linalg.norm(ps[i + 1] - ps[i])) for i in range(n - 1)]
        keep, maps = [], []
        for i in range(n):
            if n == 1:
                dc = 4 / 3 * mf * CM
            elif i == 0:
                dc = min(dist[0] * ef, 4 / 3 * ef * CM)
            elif i == n - 1:
                dc = min(dist[-1] * ef, 4 / 3 * ef * CM)
            else:
                dc = min(max(dist[i - 1] * mf, dist[i] * mf), 4 / 3 * mf * CM)
            assoc, cw, tind = self.associate(ps[i], dc)
            saved, qty = self.examine(ps[i], assoc, cw, tind, i < 2 or i >= n - 2)
            if sum(qty) > 0:
                keep.append(ps[i]); maps.append((saved, qty))
        return keep, maps

    # --- trajectory_fit -------------------------------------------------------------------------
    def traj_fit(self, ps, maps, method):
        det, o = self.det, self.o
        n = len(ps)
        cell2pts = [dict(), dict(), dict()]
        for i, (saved, _) in enumerate(maps):
            for l in range(3):
                for k in saved[l]:
                    cell2pts[l].setdefault(k, []).append(i)
        fac = [dict(), dict(), dict()]
        for l in range(3):
            for k, lst in cell2pts[l].items():
                if method == 1:
                    for i in lst:
                        fac[l][(k, i)] = 1.0 / len(lst)
                else:
                    ds = o.div_sigma
                    if o.div_sigma_pitch > 0:
                        ds = max(ds, o.div_sigma_pitch * det.pitch[l])
                    vals = []
                    for i in lst:
                        ct = ps[i][0] / det.ttw
                        cw = wire_cont(det, l, ps[i][1], ps[i][2])
                        vals.append(math.exp(-0.5 * (((ct - k[0]) * det.ttw) ** 2 + ((cw - k[1]) * det.pitch[l]) ** 2) / ds ** 2))
                    s = sum(vals)
                    for i, v in zip(lst, vals):
                        fac[l][(k, i)] = v / s if s > 0 else 0.0
        out = []
        for i in range(n):
            saved, qty = maps[i]
            A = np.zeros((2, 2)); b = np.zeros(2); sx = 0.0; sxt = 0.0
            for l in range(3):
                g = det.pdir[l] / det.pitch[l]; off = det.pdir[l] @ det.c0[l] / det.pitch[l]
                for k in saved[l]:
                    c = self.cells[l].get(k)
                    q, e = (c if c else (100.0, 1000.0))
                    if q < 100.0:
                        q, e = 100.0, 1000.0
                    s = q / e * fac[l].get((k, i), 1.0)
                    if qty[l] < 0.5:
                        s *= (qty[l] / 0.5) if qty[l] != 0 else 0.05
                    wgt = abs(s) ** o.fit_weight_pow
                    A += wgt * np.outer(g, g); b += wgt * g * (k[1] + off)
                    sx += wgt; sxt += wgt * k[0]
            p0 = ps[i]
            yz0 = p0[1:]
            try:
                yz = yz0 + np.linalg.pinv(A) @ (b - A @ yz0)
            except np.linalg.LinAlgError:
                yz = yz0
            x = (sxt / sx) * det.ttw if sx > 0 else p0[0]
            out.append(np.array([x, yz[0], yz[1]]))
        # skip_trajectory_point + area smoothing
        fine, tmp = [], []
        skip_count = 0
        for i in range(n):
            p = out[i].copy()
            skip = self._skip(p, i, ps, fine, maps)
            if skip:
                skip_count += 1
                if skip_count <= 3:
                    continue
                skip_count = 0
            tmp.append(ps[i]); fine.append(p)
        for i in range(len(fine)):
            rep = False
            for (a0, a1) in ((-1, 1), (-2, 1), (-1, 2)):
                if i + a0 < 0 or i + a1 >= len(fine):
                    continue
                pa, pb = fine[i + a0], fine[i + a1]
                area1 = _tri_area(pa, fine[i], pb); c = float(np.linalg.norm(pa - pb))
                area2 = _tri_area(pa, tmp[i], pb)
                if area1 > 1.8 * c and area1 > 1.7 * area2:
                    rep = True; break
            if rep:
                fine[i] = tmp[i]
        return fine

    def _charge_sum(self, l, w, t):
        s = 0.0
        for dw in (-1, 0, 1):
            c = self.cells[l].get((t, w + dw))
            if c: s += c[0]
        for dt in (-self.det.nt, self.det.nt):
            c = self.cells[l].get((t + dt, w))
            if c: s += c[0]
        return s

    def _skip(self, p, i, ps, fine, maps):
        det = self.det
        def proj(q):
            t = cround(cround(q[0] / det.ttw) / det.nt) * det.nt if False else cround((q[0] / det.ttw) / det.nt) * det.nt
            return t, [cround(wire_cont(det, l, q[1], q[2])) for l in range(3)]
        t1, w1 = proj(p); t2, w2 = proj(ps[i])
        ratio = 0.0; ratio1 = 1.0
        for l in range(3):
            c1 = self._charge_sum(l, w1[l], t1); c2 = self._charge_sum(l, w2[l], t2)
            if c2 != 0:
                ratio += c1 / c2
                ratio1 *= (c1 / c2) if c1 != 0 else 0.25
            else:
                ratio += 1
        if ratio / 3.0 < 0.97 or ratio1 < 0.75:
            p[:] = ps[i]
        if len(fine) >= 2:
            v1 = fine[-1] - fine[-2]; v2 = p - fine[-1]
            ang = _angle(v1, v2)
            ang1 = 180.0
            if i >= 2:
                ang1 = _angle(ps[i - 1] - ps[i - 2], ps[i] - ps[i - 1])
            qty = maps[i][1]
            dead = sum(1 for q in qty if q <= 0)
            if ang > 45 and dead >= 2:
                return True
            if ang > 160 or ang > ang1 + 90:
                return True
            if i + 1 == len(ps) and ang > 45 and np.linalg.norm(v2) < 5.0:
                return True
        return False

    # --- dQ/dx -------------------------------------------------------------------------------
    def dqdx(self, path, end_ext=3.0):
        det = self.det
        n = len(path)
        if n < 2:
            return np.zeros(n), np.zeros(n)
        rows = []
        for l in range(3):
            rows.append(sorted(self.cells[l].keys()))
        rowidx = [{k: r for r, k in enumerate(rows[l])} for l in range(3)]
        dx = np.zeros(n); prevs, nexts = [], []
        for i in range(n):
            cur = path[i]
            if i == 0:
                d = path[1] - cur; d /= max(np.linalg.norm(d), 1e-9)
                pr = cur - d * end_ext; nx = 0.5 * (cur + path[1])
            elif i == n - 1:
                d = cur - path[i - 1]; d /= max(np.linalg.norm(d), 1e-9)
                nx = cur + d * end_ext; pr = 0.5 * (cur + path[i - 1])
            else:
                pr = 0.5 * (cur + path[i - 1]); nx = 0.5 * (cur + path[i + 1])
            prevs.append(pr); nexts.append(nx)
            dx[i] = np.linalg.norm(cur - pr) + np.linalg.norm(cur - nx)
        R = [dict(), dict(), dict()]  # (row, i) -> value/total_err
        reg = np.zeros((n, 3), bool)
        for i in range(n):
            cur, pr, nx = path[i], prevs[i], nexts[i]
            samples = []
            for j in range(5):
                for (a, b) in ((pr, cur), (nx, cur)):
                    rp = a + (b - a) * (j + 0.5) / 5.0
                    wgt = float(np.linalg.norm(b - a))
                    t = max(50.0, abs(rp[0]) / det.v)
                    sl = math.sqrt(2 * det.DL * t + det.addL ** 2) / det.ttw
                    cT = rp[0] / det.ttw
                    cw = [wire_cont(det, l, rp[1], rp[2]) for l in range(3)]
                    sw = [math.sqrt(2 * self.fit_DT * t + self.fit_c[l] ** 2) / det.pitch[l] for l in range(3)]
                    samples.append((cT, sl, cw, sw, wgt))
            wsum = sum(s[4] for s in samples)
            for l in range(3):
                rel, add = (0.075, 0.0) if l < 2 else (0.05, 300.0)
                c0w, c0t = samples[0][2][l], samples[0][0]
                for k in rows[l]:
                    tb, wb = k
                    if abs(wb - c0w) > 10 or abs(tb - c0t) > 10 * det.nt:
                        continue
                    val = 0.0
                    for (cT, sl, cw, sw, wgt) in samples:
                        if abs(tb - cT) <= 4 * sl and abs(wb - cw[l]) <= 4 * sw[l]:
                            ft = 0.5 * (erf((tb + 0.5 * det.nt - cT) / SQ2 / sl) - erf((tb - 0.5 * det.nt - cT) / SQ2 / sl))
                            fw = 0.5 * (erf((wb + 0.5 - cw[l]) / SQ2 / sw[l]) - erf((wb - 0.5 - cw[l]) / SQ2 / sw[l]))
                            val += ft * fw * wgt
                    val = val / wsum if wsum > 0 else 0.0
                    if val > 0:
                        q, e = self.cells[l][k]
                        te = math.sqrt(e * e + (q * rel) ** 2 + add * add)
                        R[l][(rowidx[l][k], i)] = val / te
                for (cT, sl, cw, sw, wgt) in samples:
                    key = (cround(cT / det.nt) * det.nt, cround(cw[l]))
                    if key not in self.cells[l]:
                        reg[i, l] = True; break
        mats, datas, Ms, overl = [], [], [], []
        for l in range(3):
            nr = len(rows[l])
            rel, add = (0.075, 0.0) if l < 2 else (0.05, 300.0)
            d = np.array([self.cells[l][k][0] / math.sqrt(self.cells[l][k][1] ** 2 + (self.cells[l][k][0] * rel) ** 2 + add * add)
                          for k in rows[l]])
            if R[l]:
                rr, ii, vv = zip(*[(r, i, v) for (r, i), v in R[l].items()])
            else:
                rr, ii, vv = [], [], []
            Rm = coo_matrix((vv, (rr, ii)), shape=(nr, n)).tocsr()
            Mw, ov = _compact(Rm, nr, n, 3 if l < 2 else 2)
            mats.append(Rm); datas.append(d); Ms.append(Mw); overl.append(ov)
        F = np.zeros((n, n))
        for i in range(n):
            w = 0.0
            if reg[i, 0]: w += 0.3
            if reg[i, 1]: w += 0.3
            if reg[i, 2]: w += 0.9
            cw_ = (0.15, 0.15, 0.45)
            for l in range(3):
                pv, nv = overl[l][i]
                if i == 0:
                    if nv > 0.5: w += cw_[l] * (2 * nv - 1) ** 2
                elif i == n - 1:
                    if pv > 0.5: w += cw_[l] * (2 * pv - 1) ** 2
                else:
                    if pv + nv > 1.0: w += cw_[l] * (pv + nv - 1) ** 2
            dn = lambda k: (dx[k] + 0.01) / 6.0
            if i == 0:
                F[0, 0] = -w / dn(0); F[0, 1] = w / dn(1)
            elif i == n - 1:
                F[i, i] = -w / dn(i); F[i, i - 1] = w / dn(i - 1)
            else:
                F[i, i] = -2 * w / dn(i); F[i, i + 1] = w / dn(i + 1); F[i, i - 1] = w / dn(i - 1)
        F *= self.o.lam
        A = F.T @ F; b = np.zeros(n)
        for l in range(3):
            Rm = mats[l].toarray(); Mw = Ms[l]
            A += Rm.T @ (Mw[:, None] * Rm); b += Rm.T @ (Mw * datas[l])
        try:
            dQ = np.linalg.solve(A + 1e-12 * np.eye(n), b)
        except np.linalg.LinAlgError:
            dQ = np.linalg.lstsq(A, b, rcond=None)[0]
        return dQ, dx

    # --- do_single_tracking -----------------------------------------------------------------
    def track(self, seed_pts):
        ps = organize_orig_path(seed_pts, 12.0, 6.0)
        ps, maps = self.form_map(ps)
        if len(ps) < 2:
            return None
        ps = self.traj_fit(ps, maps, 1)
        if len(ps) < 2:
            return None
        ps = organize_ps_path(ps, 6.0, 3.0)
        ps, maps = self.form_map(ps)
        if len(ps) < 2:
            return None
        ps = self.traj_fit(ps, maps, 2)
        ps = organize_ps_path(ps, 6.0, 0.0)
        if self.o.final_fill_charge_test:
            keep, _ = self.form_map(ps)
            if len(keep) > 1:
                ps = keep
        if len(ps) < 2:
            return None
        dQ, dx = self.dqdx(ps)
        return np.array(ps), dQ, dx


def _compact(Rm, nr, n, cut):
    Rc = Rm.tocsc(); Rr = Rm.tocsr()
    count2d = np.diff(Rr.indptr)
    avg = np.zeros(n); flag3 = np.zeros(n, bool)
    for i in range(n):
        rs = Rc.indices[Rc.indptr[i]:Rc.indptr[i + 1]]; vs = Rc.data[Rc.indptr[i]:Rc.indptr[i + 1]]
        if len(rs):
            avg[i] = np.sum(count2d[rs] * vs) / np.sum(vs); flag3[i] = np.any(count2d[rs] > 2)
    M = np.ones(nr)
    for r in range(nr):
        cs = Rr.indices[Rr.indptr[r]:Rr.indptr[r + 1]]; vs = Rr.data[Rr.indptr[r]:Rr.indptr[r + 1]]
        if len(cs) == 0:
            continue
        s1 = np.sum(avg[cs] * vs); s2 = np.sum(vs)
        if np.any(flag3[cs]) and s1 > cut * s2:
            M[r] = (1.0 / (s1 / s2 - cut + 1)) ** 2
    ov = []
    sets = [set(Rc.indices[Rc.indptr[i]:Rc.indptr[i + 1]].tolist()) for i in range(n)]
    for i in range(n):
        s0 = len(sets[i])
        if s0 == 0:
            ov.append((0.0, 0.0)); continue
        pv = len(sets[i] & sets[i - 1]) / s0 if i > 0 else 0.0
        nv = len(sets[i] & sets[i + 1]) / s0 if i + 1 < n else 0.0
        ov.append((pv, nv))
    return M, ov


def _tri_area(a, b, c):
    x = np.linalg.norm(a - b); y = np.linalg.norm(c - b); z = np.linalg.norm(a - c)
    s = (x + y + z) / 2
    return math.sqrt(max(s * (s - x) * (s - y) * (s - z), 0.0))


def _angle(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 <= 0 or m2 <= 0:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(v1 @ v2) / (m1 * m2)))))


def organize_orig_path(pts, low, endlim):
    pts = [np.asarray(p, float) for p in pts]
    out = []
    p1 = pts[0]; dis1 = 0.0; p2 = p1
    for q in pts:
        p2 = q; dis1 = np.linalg.norm(p1 - p2)
        if dis1 > low: break
    if dis1 != 0:
        out.append(p1 + (p1 - p2) / dis1 * endlim)
    for p in pts:
        dis = np.linalg.norm(p - out[-1]) if out else low
        if dis < 0.8 * low:
            continue
        elif dis < 1.6 * low:
            out.append(p)
        else:
            k = cround(dis / low); ps = out[-1]
            for j in range(k):
                out.append(ps + (p - ps) / k * (j + 1))
    p1 = pts[-1]; dis1 = 0.0
    for q in reversed(pts):
        p2 = q; dis1 = np.linalg.norm(p1 - p2)
        if dis1 > low: break
    if dis1 != 0:
        out.append(p1 + (p1 - p2) / dis1 * endlim)
    return out


def organize_ps_path(pts, low, endlim):
    ps = [np.asarray(p, float) for p in pts]
    if len(ps) == 0:
        return ps
    out = []
    p1 = ps[0]; dis1 = 0.0
    for q in ps:
        p2 = q; dis1 = np.linalg.norm(p1 - p2)
        if dis1 > low: break
    if dis1 > low:
        out.append(p1 + (p1 - p2) / dis1 * endlim)
    for p in ps:
        dis = np.linalg.norm(p - out[-1]) if out else np.linalg.norm(p - ps[-1])
        if dis < 0.8 * low:
            continue
        elif dis < 1.6 * low:
            out.append(p)
        else:
            k = cround(dis / low); psv = out[-1]
            for j in range(k):
                out.append(psv + (p - psv) / k * (j + 1))
    if endlim != 0:
        p1 = ps[-1]; dis1 = 0.0
        for q in reversed(ps):
            p2 = q; dis1 = np.linalg.norm(p1 - p2)
            if dis1 > low: break
        if dis1 != 0:
            out.append(p1 + (p1 - p2) / dis1 * endlim)
    elif out:
        if np.linalg.norm(ps[-1] - out[-1]) >= 4.5:
            out.append(ps[-1])
    if len(out) <= 1:
        out = ps
    return out


# --------------------------------------------------------------------------------------------
# one event end to end + metrics
# --------------------------------------------------------------------------------------------
def run_event(det, p0, p1, o: Opts, seed=0, noise=300.0, thr=900.0, dqdx_true=5000.0, fluct=0.0):
    rng = np.random.default_rng(seed)
    P, active = deposit(det, p0, p1, dqdx=dqdx_true, noise=noise, thr=thr, rng=rng, fluct=fluct)
    blobs = tile(det, active)
    if not blobs:
        return None
    pts, blob_of, charges, _ = sample(det, blobs, active, o)
    if len(pts) < 10:
        return None
    G = base_graph(det, pts, blob_of, blobs)
    adj = adjacency(G)
    # the cluster cloud the fit associates against: production sampling when the sampling levers
    # are scoped to the Steiner stage only
    if o.sampler_scope == "steiner":
        prod = replace(o, min_step=Opts().min_step, half_pitch=False, x_mid=False)
        cpts, cblob_of, _, _ = sample(det, blobs, active, prod)
        cG = base_graph(det, cpts, cblob_of, blobs)
    else:
        cpts, cblob_of, cG = pts, blob_of, G
    terms, ends = find_terminals(det, pts, blob_of, charges, G, adj, o)
    pts_path = pts
    if o.term_centroid:
        moved = centroid_positions(det, pts, blob_of, blobs, active, sorted(terms))
        pts_path = pts.copy()
        for i, q in moved.items():
            pts_path[i] = q
    verts, S, path = steiner(det, pts, charges, G, terms, ends)
    if o.seed == "truth":
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        e0 = p0 + d * 20.0; e1 = p1 - d * 20.0
        k = int(np.linalg.norm(e1 - e0) / 3.0)
        seed_pts = [e0 + (e1 - e0) * j / k for j in range(k + 1)]
    else:
        seed_pts = [pts_path[i] for i in path]
    fit = Fitter(det, cpts, cblob_of, blobs, cG, active, verts, S, o, spts=pts)
    res = fit.track(seed_pts)
    if res is None:
        return None
    fp, dQ, dx = res
    return dict(fit=fp, dQ=dQ, dx=dx, seed=np.array(seed_pts), npts=len(pts), nterm=len(terms),
                nblob=len(blobs), truth=(p0, p1), dqdx_true=dqdx_true)


def metrics(det, ev, end_cut=30.0):
    p0, p1 = ev["truth"]; d = (p1 - p0) / np.linalg.norm(p1 - p0)
    fp = ev["fit"]; dQ = ev["dQ"]; dx = ev["dx"]
    s = (fp - p0) @ d
    L = np.linalg.norm(p1 - p0)
    inner = (s > end_cut) & (s < L - end_cut)
    if inner.sum() < 10:
        return None
    perp = (fp - p0) - np.outer(s, d)
    dist = np.linalg.norm(perp, axis=1)
    out = dict(n=int(inner.sum()), res_med=float(np.median(dist[inner])), res_p90=float(np.percentile(dist[inner], 90)))
    # split: the drift (x) offset of the point from the true line at the same (y,z)-projected
    # arc length is dominated by the slice-start convention (a constant); the transverse part
    # is the residual in the plane perpendicular to both d and x.
    xhat = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(d, xhat)
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.array([0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(d, t1)                                          # the (x-containing) perpendicular
    a1 = perp @ t1; a2 = perp @ t2
    out["res_t_med"] = float(np.median(np.abs(a1[inner])))       # purely transverse to drift
    out["res_t_p90"] = float(np.percentile(np.abs(a1[inner]), 90))
    out["x_bias"] = float(np.median(perp[inner, 0]))              # signed mean drift offset (mm)
    out["res_x_rms"] = float(np.std(a2[inner]))                   # drift-plane scatter about its bias
    for l, nm in enumerate("uvw"):
        pl = np.array([det.pdir[l][0], det.pdir[l][1]])
        comp = perp[:, 1:] @ pl
        out["res_" + nm] = float(np.sqrt(np.mean(comp[inner] ** 2)) / det.pitch[l])
        wc = np.array([wire_cont(det, l, q[1], q[2]) for q in fp])
        fr = np.mod(wc[inner], 1.0)
        out["snap_" + nm] = float(np.mean(np.minimum(fr, 1 - fr) < 0.15) / 0.30)
    # seed deviation
    sd = ev["seed"]; ss = (sd - p0) @ d; sin = (ss > end_cut) & (ss < L - end_cut)
    if sin.sum() > 3:
        sp = np.linalg.norm((sd - p0) - np.outer(ss, d), axis=1)
        out["seed_med"] = float(np.median(sp[sin])); out["seed_p90"] = float(np.percentile(sp[sin], 90))
    # the data census's zig-zag instrument (d101_census.py chord_dev): distance of each fitted point
    # from the chord of its own +-3 cm neighbourhood, so toy and data are read with one ruler
    Ls = np.r_[0, np.cumsum(np.linalg.norm(np.diff(fp, axis=0), axis=1))]
    dev = np.full(len(fp), np.nan)
    for i in range(len(fp)):
        j0 = np.searchsorted(Ls, Ls[i] - 30.0); j1 = min(len(fp) - 1, np.searchsorted(Ls, Ls[i] + 30.0))
        if j0 >= i or j1 <= i:
            continue
        c = fp[j1] - fp[j0]; nc = np.linalg.norm(c)
        if nc < 1e-6:
            continue
        u = c / nc; v = fp[i] - fp[j0]
        dev[i] = np.linalg.norm(v - (v @ u) * u)
    out["dev_med"] = float(np.nanmedian(dev[inner])); out["dev_p90"] = float(np.nanpercentile(dev[inner], 90))
    dqdx = dQ / np.maximum(dx, 1e-6) / ev["dqdx_true"]           # ratio to truth
    r = dqdx[inner]
    med = np.median(r)
    out["dqdx_med"] = float(med)
    out["dqdx_cv"] = float(1.4826 * np.median(np.abs(r - med)) / med) if med > 0 else float("nan")
    out["dqdx_rms"] = float(np.std(r) / med) if med > 0 else float("nan")
    out["dip"] = float(np.mean(r < 0.5 * med))
    out["dip_truth"] = float(np.mean(r < 0.6))
    x = r - r.mean()
    out["acf1"] = float(np.sum(x[1:] * x[:-1]) / np.sum(x * x)) if np.sum(x * x) > 0 else float("nan")
    dd = dist[inner]
    out["r_res_q"] = float(np.corrcoef(dd, r)[0, 1]) if np.std(dd) > 0 and np.std(r) > 0 else float("nan")
    out["len_ratio"] = float(np.sum(dx[inner]) / (s[inner].max() - s[inner].min() + 1e-9))
    return out


if __name__ == "__main__":
    det = load_detector(sys.argv[1] if len(sys.argv) > 1 else "pdvd")
    p0, p1 = make_track(det, 100.0, 60.0, 30.0, rng=np.random.default_rng(1))
    ev = run_event(det, p0, p1, Opts(), seed=1)
    print({k: v for k, v in ev.items() if k in ("npts", "nterm", "nblob")})
    print(metrics(det, ev))
