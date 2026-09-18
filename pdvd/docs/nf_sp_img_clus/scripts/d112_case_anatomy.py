#!/usr/bin/env python3
"""doc pdvd/112 -- two of the owner's spots taken apart vertex by vertex: why the persisted STM seed leaves the image.

Read-only, on the doc-111 round-2 dump arms (d111hst / d111vst: WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1, identical
to production, doc 111 sec 11.4).  Inputs per case: the event trace (STGR/STGV/STGE/STMRP/STMPATH), tracking-stm.root
(T_rec_charge = the persisted fit rows with their 2-D coordinates, T_proj_data = the cluster's 2-D charge cells) and
mabc-pr.zip clustering-global (the image).

Record, walks and seed as d111s_spot_anatomy.py (duplicated): the persisted round-2 record through the window, the
STMRP walks that built its seed (crawl1 + crawl2, else rough), the seed joined to the dumped steiner_pc vertices.

Per Steiner vertex: class 3live / 1blank / 1dead / 2blank+ and role terminal / extreme / interior (as
d112_terminal_planes.py), ridge offset (d111_stage_attrib.Ridge on the image).

Deviated stretches: maximal runs of window seed vertices > 1 cm off the ridge.  Each run is closed by its anchors, the
nearest seed vertex <= 1 cm before and after it.  Between the two anchors four routes on the dumped steiner_graph:
  seed       the persisted seed's own sub-route
  shortest   Dijkstra under the dumped weights (equals the seed unless the sub-route spans the crawl point)
  on-image   Dijkstra using only edges whose straight segment never leaves the ridge by > 1 cm (edge_offfrac == 0)
  oracle     Dijkstra under w * (1 + 100 f_off), the doc-111 round-2 image oracle
each with its dumped-weight cost, geometric length, cost per cm, length after the doc-111 +-2 cm running mean (the route without its
lattice staircase) and maximum ridge offset.

2-D: the vertices are projected into each plane's (time slice, wire rank) with d110_spot_figs.local_project (a local
affine map fitted on the fit rows, which carry both 3-D and 2-D coordinates); a projected vertex "sits on charge" when
T_proj_data has a cell with q > 0 at its rounded slice and within +-0 / +-1 wire of its rounded wire.

Outputs:
  <out>_<case>_2d.png      U / V / W cells with the Steiner vertices by class, terminals, seed, deviated stretches
  <out>_<case>_frame.png   ridge frame (s, e2) / (s, e3): vertices and terminals by class; edges at the stretches by
                           source; seed; on-image alternative
  <out>_<case>_offset.png  seed ridge offset vs arc length, by class, blank plane annotated
  <out>_<case>_walk.tsv    per seed point in the window (cell_* own-wire/+-1-wire charge cell; band_* charge band width)
  <out>_cases.txt          per case: checks, stretches, route comparison, window terminals

Usage: d112_case_anatomy.py --out figs/112
"""
import argparse, collections, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import d111s_common as C
from d111_spot_frame import frame
from d111s_graph_census import edge_offfrac
import d110_spot_figs as D
from d111s_wiggle_split import smooth
from d112_terminal_planes import vertex_class, vertex_role, rms_check, CLASSES, ROLES
A = C.A

CASES = [
    ("h1", "pdhd", "029107_16", 108, (75.3, 438.3, 450.1), 12.0, "d111hst"),
    ("v3", "pdvd", "039349_20", 25, (-270.2, -243.7, 36.0), 12.0, "d111vst"),
]
CLS_COL = ("tab:green", "tab:red", "tab:blue", "tab:orange")
SRC_COL = {0: "tab:olive", 1: "tab:brown", 2: "tab:purple"}
PLANE = "UVW"
THR = 1.0


def logd_for(arm):
    """doc 114: the dump dir of an arm -- /home/xqian/tmp/d111 for the doc-111/112 arms (unchanged), else the round dir
    named by the arm's prefix (d113*, d114*, ...)."""
    d = f"/home/xqian/tmp/d111/arm_{arm}"
    if os.path.isdir(d):
        return d
    return f"/home/xqian/tmp/{arm[:4]}/arm_{arm}"


def walk_masked(sb, a, b, w, mask):
    M = csr_matrix((np.r_[w[mask], w[mask]], (np.r_[sb.E_s[mask], sb.E_t[mask]], np.r_[sb.E_t[mask], sb.E_s[mask]])),
                   shape=(sb.nv, sb.nv))
    dist, pred = dijkstra(M, directed=False, indices=a, return_predecessors=True)
    if not np.isfinite(dist[b]):
        return None
    p = [b]
    while p[-1] != a:
        p.append(pred[p[-1]])
    return p[::-1]


def route_stats(sb, R, path):
    if path is None:
        return None
    cost = sb.path_cost(path)
    L = 0.0
    for u, v in zip(path[:-1], path[1:]):
        L += sb.E_len[sb.edge_index[(min(u, v), max(u, v))]]
    X, _ = C.polyline_samples(sb.V[path], 0.3)
    mo = float(np.nanmax(R.offset(X))) if len(X) else float(R.offset(sb.V[path]).max())
    _, Sm = smooth(sb.V[path])                   # doc 111 round 2 bow/jitter split: +-2 cm running mean
    Ls = float(np.linalg.norm(np.diff(Sm, axis=0), axis=1).sum()) if len(Sm) > 1 else 0.0
    return dict(path=path, cost=cost, length=L, smooth_length=Ls, maxoff=mo, nv=len(path))


def project_u(t, sel, S):
    """U wire rank of points S.  The fit rows' pu is not a clean geometric projection (a +-1 wire scatter and an x term
    about a (y, z) plane; pv, pw and pt close exactly), so d110's local affine map does not close on U.  U is the mirror
    of V in z on both detectors (PDHD +-35.7 deg, PDVD CRP +-30 deg): the V gradient is fitted exactly on the rows, U takes
    (dV/dy, -dV/dz), and the offset is the median of pu - gradient . (y, z) over the rows.  The time slice comes from
    the pt local map.  Returns (w, s, ok, model) with model = (gy, gz, c, rms of pu about the model)."""
    idx = np.where(sel)[0]
    Y = np.c_[t["y"][idx], t["z"][idx], np.ones(len(idx))]
    cv = np.linalg.lstsq(Y, t["pv"][idx], rcond=None)[0]
    gy, gz = cv[0], -cv[1]
    c0 = float(np.median(t["pu"][idx] - gy * t["y"][idx] - gz * t["z"][idx]))
    rms = float(np.std(t["pu"][idx] - gy * t["y"][idx] - gz * t["z"][idx] - c0))
    _, s, ok = D.local_project(t, sel, "pv", S)
    w = gy * S[:, 1] + gz * S[:, 2] + c0
    return w, s, ok, (gy, gz, c0, rms)


def dedupe(seq):
    out = []
    for x in seq:
        if not out or int(x) != out[-1]:
            out.append(int(x))
    return out


def cell_lookup(blk, lo, hi):
    m = (blk["ch"] >= lo) & (blk["ch"] < hi) & (blk["q"] > 0)
    cells = {}
    for c, s, q in zip(blk["ch"][m].astype(int), blk["ts"][m].astype(int), blk["q"][m]):
        cells[(int(c), int(s))] = float(q)
    arr = np.array(list(cells.keys()), float).reshape(-1, 2)
    return cells, arr


def one_case(case, out, L):
    name, det, ev, cl, click, R, arm = case
    click = np.array(click)
    d = f"{C.IMG}/{det}/work/{ev}_{arm}"
    tr = C.parse_trace(f"{logd_for(arm)}/evt_{ev}.log.gz")
    t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(
        ["x", "y", "z", "cluster_id", "pass", "pu", "pv", "pw", "pt"], library="np")
    img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    mi = img[1] == cl
    P0, w0 = img[0][mi], np.clip(img[2][mi], 1, None)
    Rg = A.Ridge(img[0][mi], img[2][mi])
    blocks = [b for b in tr["blocks"] if b["cl"] == cl and "seed" in b["st"]]
    rec = None
    for ps in np.unique(t["pass"][t["cluster_id"] == cl]):
        m = (t["cluster_id"] == cl) & (t["pass"] == ps)
        F = np.c_[t["x"][m], t["y"][m], t["z"][m]]
        if np.min(np.linalg.norm(F - click, axis=1)) > R:
            continue
        bl = [b for b in blocks if b["rnd"] == "r2" and "final" in b["st"] and len(b["st"]["final"]) == len(F)
              and np.max(np.abs(b["st"]["final"] - F)) < 0.01]
        if bl:
            rec = (ps, F, bl[-1]); break
    if rec is None:
        L.append(f"## {name}: no persisted record through the window"); return
    ps, F, b2 = rec
    sb = C.graph_for(tr, cl, b2["order"])
    r1 = [b for b in blocks if b["rnd"] == "r1" and b["dir"] == b2["dir"] and b["order"] < b2["order"]][-1]
    crawl = [r for r in C.walks_for(tr, cl, b2["order"], after=r1["order"]) if r[2] in ("crawl1", "crawl2")]
    rough = [r for r in C.walks_for(tr, cl, r1["order"]) if r[2] == "rough"]
    walks = crawl[-2:] if len(crawl) >= 2 else rough[-1:]
    crawl_pt = walks[0][4] if len(walks) == 2 else None
    seed = b2["st"]["seed"]
    j = C.join_vertices(sb, seed)
    cls, zero, dead = vertex_class(sb)
    role = vertex_role(sb)
    rok = rms_check(sb, zero)
    voff = Rg.offset(sb.V)
    offfrac = edge_offfrac(sb, Rg)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seed, axis=0), axis=1))]
    i_click = int(np.argmin(np.linalg.norm(seed - click, axis=1)))
    arc = arc - arc[i_click]
    inw = np.linalg.norm(seed - click, axis=1) < R
    Xs, _ = C.polyline_samples(seed[inw], 0.3)
    seed_max = float(np.nanmax(Rg.offset(Xs)))
    on_term = np.where(sb.V_term & ~sb.V_ext & (voff <= THR))[0]

    L.append(f"## {name}: {det} {ev} cl{cl} arm {arm}; record pass {ps}, walks {'+'.join(w[2] for w in walks)}"
             f"{'' if crawl_pt is None else f' (crawl point vertex {crawl_pt})'}")
    L.append(f"- graph: {sb.nv} Steiner vertices, {len(sb.E_s)} edges, {int((sb.V_term & ~sb.V_ext).sum())} terminals "
             f"(+{int(sb.V_ext.sum())} extremes); STGC {sb.header}")
    L.append(f"- check join: {int((j >= 0).sum())} of {len(j)} seed points are dumped vertices")
    L.append(f"- check anchor: seed max ridge offset in the window {seed_max:.2f} cm (figs/111s_spot.tsv)")
    L.append(f"- check charge identity (dumped q == RMS over nonzero planes): {int(rok.sum())} of {sb.nv} vertices")

    # ---- deviated stretches
    wi = np.where(inw)[0]
    runs, cur = [], []
    for i in wi:
        if j[i] >= 0 and voff[j[i]] > THR:
            if cur and i != cur[-1] + 1:
                runs.append(cur); cur = []
            cur.append(i)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    stretch_vertices = set()
    alt_routes = []
    L.append("")
    L.append("| stretch | seed points | anchors (vertex, offset) | seed max off | vertices: term 1blank / term other / interior 1blank / interior 2blank+ / interior other | route | cost | length (cm) | cost per cm | smoothed length (cm) | max off (cm) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, run in enumerate(runs):
        a = max([i for i in range(run[0]) if j[i] >= 0 and voff[j[i]] <= THR], default=0)
        b = min([i for i in range(run[-1] + 1, len(seed)) if j[i] >= 0 and voff[j[i]] <= THR], default=len(seed) - 1)
        va, vb = int(j[a]), int(j[b])
        sub = dedupe(j[a:b + 1])
        vs = set(sub)
        stretch_vertices |= vs
        ins = [v for v in sub if voff[v] > THR]
        cnt = (sum(1 for v in ins if role[v] == 0 and cls[v] == 1), sum(1 for v in ins if role[v] == 0 and cls[v] != 1),
               sum(1 for v in ins if role[v] != 0 and cls[v] == 1), sum(1 for v in ins if role[v] != 0 and cls[v] == 3),
               sum(1 for v in ins if role[v] != 0 and cls[v] not in (1, 3)))
        spans = crawl_pt is not None and crawl_pt in vs and crawl_pt not in (va, vb)
        routes = [("seed", route_stats(sb, Rg, sub))]
        p, _ = sb.walk(va, vb)
        routes.append(("shortest", route_stats(sb, Rg, p)))
        routes.append(("on-image", route_stats(sb, Rg, walk_masked(sb, va, vb, sb.E_w, offfrac == 0))))
        p, _ = sb.walk(va, vb, sb.E_w * (1.0 + 100.0 * offfrac))
        routes.append(("oracle", route_stats(sb, Rg, p)))
        alt_routes.append((k, routes))
        smax = max(voff[v] for v in ins)
        for r_i, (rn, st) in enumerate(routes):
            head = (f"| S{k} | {run[0]}-{run[-1]} | {va} ({voff[va]:.2f}) -> {vb} ({voff[vb]:.2f}){' spans crawl point' if spans else ''} "
                    f"| {smax:.2f} | {' / '.join(str(c) for c in cnt)} ") if r_i == 0 else "| | | | | "
            if st is None:
                L.append(head + f"| {rn} | none | | | | |")
            else:
                ratio = f" ({st['cost'] / routes[0][1]['cost']:.3f}x seed)" if r_i > 0 else ""
                L.append(head + f"| {rn} | {st['cost']:.3f}{ratio} | {st['length']:.2f} | {st['cost'] / st['length']:.3f} | {st['smooth_length']:.2f} | {st['maxoff']:.2f} |")

    # ---- per seed point table
    rows = ["i\tarc_cm\tvid\toff_cm\trole\tclass\tblank\tqu\tqv\tqw\tdead_uvw\tq_dump\tq_rms_ok\tin_src\tin_base\tin_len\tin_w\t"
            "near_onridge_term_cm\tstretch\tcell_u\tcell_v\tcell_w\tblank_dw\tblank_dt\tband_u\tband_v\tband_w"]
    # 2-D projection of every vertex near the click (and of the seed points)
    selfit = (t["cluster_id"] == cl) & (np.linalg.norm(np.c_[t["x"], t["y"], t["z"]] - click, axis=1) < 3 * R)
    near_v = np.where(np.linalg.norm(sb.V - click, axis=1) < 1.3 * R)[0]
    blk = D.proj_block(f"{C.IMG}/{det}/work", f"{ev}_{arm}", cl)
    base = D.DETS[det]["base"]
    proj = {}
    agree = collections.Counter()
    u_model = None
    for pl, key in enumerate(D.PLANES):
        lo, hi = base[pl], (base[pl + 1] if pl < 2 else 10 ** 7)
        if key == "pu":
            w, s, ok, u_model = project_u(t, selfit, sb.V[near_v])
        else:
            w, s, ok = D.local_project(t, selfit, key, sb.V[near_v])
        cells, arr = cell_lookup(blk, lo, hi)
        proj[pl] = dict(w=dict(zip(near_v.tolist(), w)), s=dict(zip(near_v.tolist(), s)), ok=dict(zip(near_v.tolist(), ok)),
                        cells=cells, arr=arr, lo=lo, hi=hi)
        for v, wv, sv, okv in zip(near_v, w, s, ok):
            if not okv:
                agree[("proj_fail", pl)] += 1
                continue
            rw, rs = int(np.round(wv)), int(np.round(sv))
            has0 = (rw, rs) in cells
            has1 = any((rw + dw, rs) in cells for dw in (-1, 0, 1))
            if zero[v, pl] and not dead[v, pl] and cls[v] == 1:
                agree["blank_n"] += 1; agree["blank_nocell0"] += int(not has0); agree["blank_nocell1"] += int(not has1)
                agree[("blank_n", pl)] += 1; agree[("blank_nocell0", pl)] += int(not has0)
                if voff[v] > THR:
                    agree["blank_off_n"] += 1; agree["blank_off_nocell0"] += int(not has0)
            elif not zero[v, pl]:
                agree["live_n"] += 1; agree["live_cell0"] += int(has0); agree["live_cell1"] += int(has1)
                agree[("live_n", pl)] += 1; agree[("live_cell0", pl)] += int(has0)

    def cell_flag(v, pl):
        """own-wire / +-1-wire charge cell at the vertex's projected slice: e.g. 0/1; -1 when not projected."""
        P = proj[pl]
        if v not in P["ok"] or not P["ok"][v]:
            return -1
        rw, rs = int(np.round(P["w"][v])), int(np.round(P["s"][v]))
        return f"{int((rw, rs) in P['cells'])}/{int(any((rw + dw, rs) in P['cells'] for dw in (-1, 0, 1)))}"

    def blank_reach(v):
        pls = [p for p in range(3) if zero[v, p] and not dead[v, p]]
        if cls[v] != 1 or not pls:
            return "", ""
        P = proj[pls[0]]
        if v not in P["ok"] or not P["ok"][v] or len(P["arr"]) == 0:
            return "", ""
        dd = P["arr"] - np.array([P["w"][v], P["s"][v]])
        k = int(np.argmin((dd ** 2).sum(axis=1)))
        return f"{dd[k, 0]:.1f}", f"{dd[k, 1]:.1f}"

    def band_width(v, pl):
        """width (wires) of the contiguous charge run at the vertex's projected slice: the run holding its own wire, else
        the nearest run within +-5 wires (the band the blank plane misses); 0 when the slice has none nearby."""
        P = proj[pl]
        if v not in P["ok"] or not P["ok"][v]:
            return np.nan
        rw, rs = int(np.round(P["w"][v])), int(np.round(P["s"][v]))
        c0 = next((rw + d for d in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5) if (rw + d, rs) in P["cells"]), None)
        if c0 is None:
            return 0
        lo_, hi_ = c0, c0
        while (lo_ - 1, rs) in P["cells"]:
            lo_ -= 1
        while (hi_ + 1, rs) in P["cells"]:
            hi_ += 1
        return hi_ - lo_ + 1

    widths = collections.defaultdict(list)
    reach = collections.Counter()
    prev = None
    for i in wi:
        v = int(j[i])
        if v < 0:
            continue
        e = sb.edge_index.get((min(prev, v), max(prev, v))) if prev is not None and prev != v else None
        dn = float(np.min(np.linalg.norm(sb.V[on_term] - sb.V[v], axis=1))) if len(on_term) else np.nan
        blank = "".join(PLANE[p] for p in range(3) if zero[v, p])
        dw, dt = blank_reach(v)
        if dw != "" and voff[v] > THR:
            reach["n"] += 1
            reach["in_disc"] += int(float(dw) ** 2 + float(dt) ** 2 <= 9.0)
        st = next((f"S{k}" for k, run in enumerate(runs) if run[0] <= i <= run[-1]), "")
        bw = [band_width(v, p) for p in range(3)]
        if cls[v] == 1 and voff[v] > THR:
            for p in range(3):
                widths["off1blank_blank" if zero[v, p] else "off1blank_live"].append(bw[p])
        elif cls[v] == 0 and voff[v] <= THR:
            widths["on3live"].extend(bw)
        q3 = sb.R_q[sb.V_old[v]]
        rows.append("\t".join(str(x) for x in (
            i, f"{arc[i]:.2f}", v, f"{voff[v]:.2f}", ROLES[role[v]], CLASSES[cls[v]], blank or "-",
            f"{q3[0]:.0f}", f"{q3[1]:.0f}", f"{q3[2]:.0f}", "".join(str(int(x)) for x in dead[v]), f"{sb.V_q[v]:.0f}", int(rok[v]),
            C.SRC_NAME[sb.E_src[e]] if e is not None else "-", C.BASE_NAME[sb.E_base[e]] if e is not None else "-",
            f"{sb.E_len[e]:.2f}" if e is not None else "-", f"{sb.E_w[e]:.3f}" if e is not None else "-",
            f"{dn:.2f}", st, cell_flag(v, 0), cell_flag(v, 1), cell_flag(v, 2), dw, dt,
            *[("" if not np.isfinite(x) else int(x)) for x in bw])))
        prev = v
    open(f"{out}_{name}_walk.tsv", "w").write("\n".join(rows) + "\n")

    L.append("")
    L.append(f"- check 2-D projection: vertices within 1.3R projected: {len(near_v)}; local map failed U/V/W "
             f"{agree[('proj_fail', 0)]}/{agree[('proj_fail', 1)]}/{agree[('proj_fail', 2)]}; U = mirror of V, "
             f"gradient ({u_model[0]:.4f}, {u_model[1]:.4f}) wires/cm, fit rows' pu scatter about it {u_model[3]:.2f} wire rms")
    L.append(f"- check blank plane = no charge cell: 1blank planes {agree['blank_n']}, no cell at own wire "
             f"{agree['blank_nocell0']} ({100 * agree['blank_nocell0'] / max(agree['blank_n'], 1):.1f} %), none within +-1 wire "
             f"{agree['blank_nocell1']} ({100 * agree['blank_nocell1'] / max(agree['blank_n'], 1):.1f} %)")
    L.append("  - per plane U/V/W, no cell at own wire: " + " / ".join(
        f"{agree[('blank_nocell0', p)]} of {agree[('blank_n', p)]}" for p in range(3))
        + f"; the 1blank planes of vertices > 1 cm off the ridge: {agree['blank_off_nocell0']} of {agree['blank_off_n']}")
    L.append(f"- check live plane = charge cell: live planes {agree['live_n']}, cell at own wire {agree['live_cell0']} "
             f"({100 * agree['live_cell0'] / max(agree['live_n'], 1):.1f} %), within +-1 wire {agree['live_cell1']} "
             f"({100 * agree['live_cell1'] / max(agree['live_n'], 1):.1f} %); per plane U/V/W own wire: " + " / ".join(
             f"{agree[('live_cell0', p)]} of {agree[('live_n', p)]}" for p in range(3)))
    med = lambda x: f"{np.nanmedian(x):.1f}" if len(x) else "n/a"
    L.append(f"- charge band width (wires, contiguous cells at the vertex's slice): off-ridge 1blank seed vertices, live planes "
             f"median {med(widths['off1blank_live'])} (n {len(widths['off1blank_live'])}), blank plane (nearest band) median "
             f"{med(widths['off1blank_blank'])} (n {len(widths['off1blank_blank'])}); on-ridge 3live seed vertices, all planes "
             f"median {med(widths['on3live'])} (n {len(widths['on3live'])})")
    L.append(f"- bound: off-ridge 1blank seed vertices whose blank cell lies within the +-3 (wire, slice) disc of a charge "
             f"cell of the cluster on that plane: {reach['in_disc']} of {reach['n']}")

    # ---- window terminals
    tw = np.where(sb.V_term & (np.linalg.norm(sb.V - click, axis=1) < R))[0]
    L.append("")
    L.append("| window terminals | <= 1 cm | 1-2 cm | > 2 cm |")
    L.append("|---|---|---|---|")
    for c, cn in enumerate(CLASSES):
        for rr, rn in ((0, "terminal"), (1, "extreme")):
            sel = tw[(cls[tw] == c) & (role[tw] == rr)]
            if len(sel) == 0:
                continue
            o = voff[sel]
            L.append(f"| {rn} {cn} | {int((o <= 1).sum())} | {int(((o > 1) & (o <= 2)).sum())} | {int((o > 2).sum())} |")
    seed_v = set(int(x) for x in j[inw] if x >= 0)
    ont = [v for v in tw if voff[v] <= THR and role[v] == 0]
    L.append(f"- on-ridge terminals in the window: {len(ont)}; on the seed: {sum(1 for v in ont if v in seed_v)}")
    offt = [v for v in tw if voff[v] > THR and role[v] == 0]
    L.append(f"- off-ridge terminals in the window: {len(offt)} (1blank {sum(1 for v in offt if cls[v] == 1)}); on the seed: "
             f"{sum(1 for v in offt if v in seed_v)} (1blank {sum(1 for v in offt if v in seed_v and cls[v] == 1)})")
    L.append("")

    # ---- figures
    fig_2d(name, det, ev, cl, sb, j, inw, runs, near_v, proj, cls, role, zero, dead, voff, out)
    fig_frame(name, det, ev, cl, sb, P0, w0, click, R, seed, j, inw, runs, stretch_vertices, alt_routes, cls, role, voff, out)
    fig_offset(name, det, ev, cl, sb, arc, j, inw, runs, cls, role, zero, voff, out)


def fig_2d(name, det, ev, cl, sb, j, inw, runs, near_v, proj, cls, role, zero, dead, voff, out):
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.5))
    wi = np.where(inw)[0]
    for pl in range(3):
        ax = axs[pl]; P = proj[pl]
        sv = [int(j[i]) for i in wi if j[i] >= 0 and P["ok"].get(int(j[i]), False)]
        if not sv:
            ax.set_title(f"{PLANE[pl]}: no projection"); continue
        W = np.array([P["w"][v] for v in sv]); S = np.array([P["s"][v] for v in sv])
        w0, w1, s0, s1 = W.min() - 6, W.max() + 6, S.min() - 6, S.max() + 6
        cells = P["arr"]
        if len(cells):
            qv = np.array([P["cells"][(int(c), int(s))] for c, s in cells])
            m = (cells[:, 0] >= w0) & (cells[:, 0] <= w1) & (cells[:, 1] >= s0) & (cells[:, 1] <= s1)
            ax.scatter(cells[m, 1], cells[m, 0], c=np.log10(np.clip(qv[m], 100, None)), marker="s", s=60, cmap="Greys",
                       vmin=2.3, vmax=5.0, zorder=1)
        vv = [v for v in near_v if P["ok"].get(int(v), False)]
        vv = np.array([v for v in vv if w0 <= P["w"][v] <= w1 and s0 <= P["s"][v] <= s1], int)
        if len(vv):
            vw = np.array([P["w"][v] for v in vv]); vs = np.array([P["s"][v] for v in vv])
            for c in range(4):
                mm = (cls[vv] == c) & (role[vv] == 2)
                ax.scatter(vs[mm], vw[mm], s=14, color=CLS_COL[c], zorder=3)
                mm = (cls[vv] == c) & (role[vv] != 2)
                ax.scatter(vs[mm], vw[mm], s=70, marker="^", color=CLS_COL[c], edgecolor="k", lw=0.6, zorder=4)
            mm = (cls[vv] == 1) & (role[vv] == 0) & (voff[vv] > THR)
            ax.scatter(vs[mm], vw[mm], s=190, marker="^", facecolor="none", edgecolor="red", lw=1.6, zorder=5)
        ax.plot(S, W, "-", color="k", lw=1.0, zorder=2)
        for run in runs:
            rv = [int(j[i]) for i in range(max(run[0] - 1, 0), run[-1] + 2) if j[i] >= 0 and P["ok"].get(int(j[i]), False)]
            if len(rv) >= 2:
                ax.plot([P["s"][v] for v in rv], [P["w"][v] for v in rv], "-", color="magenta", lw=6, alpha=0.35, zorder=2)
        ax.set_xlim(s0, s1); ax.set_ylim(w0, w1)
        ax.set_xlabel("time slice"); ax.set_ylabel(f"{PLANE[pl]} wire (rank)")
        ax.set_title(f"{PLANE[pl]} plane: grey = this cluster's charge cells;\na vertex on a white cell sees no charge on this plane", fontsize=9)
    h = [Line2D([], [], ls="", marker="o", color=CLS_COL[c], label=f"Steiner vertex {CLASSES[c]}") for c in range(4)]
    h += [Line2D([], [], ls="", marker="^", color="w", markeredgecolor="k", label="terminal / extreme (colour = class)"),
          Line2D([], [], ls="", marker="^", color="w", markeredgecolor="red", markersize=11, label="terminal, 1blank, > 1 cm off"),
          Line2D([], [], color="k", label="seed (persisted STM, round 2)"),
          Line2D([], [], color="magenta", lw=6, alpha=0.35, label="seed > 1 cm off the image ridge")]
    axs[0].legend(handles=h, fontsize=7, loc="best")
    fig.suptitle(f"{name}: {det} {ev} cl{cl} -- the three wire planes around the click", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}_{name}_2d.png", dpi=100); plt.close(fig)


def fig_frame(name, det, ev, cl, sb, P0, w0, click, R, seed, j, inw, runs, stretch_vertices, alt_routes, cls, role, voff, out):
    near = np.linalg.norm(P0 - click, axis=1) < R
    c, e1, e2, e3 = frame(P0[near], w0[near])
    pr = lambda X: ((X - c) @ e1, np.c_[(X - c) @ e2, (X - c) @ e3])
    Vw = np.where(np.linalg.norm(sb.V - click, axis=1) < R)[0]
    fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey="row")
    s_i, T_i = pr(P0[near])
    s_v, T_v = pr(sb.V[Vw])
    ks = np.linalg.norm(seed - click, axis=1) < 1.3 * R
    s_s, T_s = pr(seed[ks])
    for r in range(2):
        ax = axs[r, 0]
        ax.scatter(s_i, T_i[:, r], s=5, c="0.75", label="image")
        for cc in range(4):
            mm = (cls[Vw] == cc) & (role[Vw] == 2)
            ax.scatter(s_v[mm], T_v[mm, r], s=8, color=CLS_COL[cc], label=f"vertex {CLASSES[cc]}")
            mm = (cls[Vw] == cc) & (role[Vw] != 2)
            ax.scatter(s_v[mm], T_v[mm, r], s=60, marker="^", color=CLS_COL[cc], edgecolor="k", lw=0.6)
        mm = (cls[Vw] == 1) & (role[Vw] == 0) & (voff[Vw] > THR)
        ax.scatter(s_v[mm], T_v[mm, r], s=170, marker="^", facecolor="none", edgecolor="red", lw=1.5)
        ax = axs[r, 1]
        ax.scatter(s_i, T_i[:, r], s=5, c="0.8")
        for k in range(len(sb.E_s)):
            u, v = sb.E_s[k], sb.E_t[k]
            if u in stretch_vertices or v in stretch_vertices:
                s_e, T_e = pr(sb.V[[u, v]])
                ax.plot(s_e, T_e[:, r], "-", color=SRC_COL[sb.E_src[k]], lw=0.7, alpha=0.7)
        ax.plot(s_s, T_s[:, r], "-o", color="k", ms=2.5, lw=1.2)
        for run in runs:
            X = seed[max(run[0] - 1, 0):run[-1] + 2]
            s_x, T_x = pr(X)
            ax.plot(s_x, T_x[:, r], "-", color="magenta", lw=6, alpha=0.3)
        for k, routes in alt_routes:
            st = dict(routes).get("on-image")
            if st is not None:
                s_x, T_x = pr(sb.V[st["path"]])
                ax.plot(s_x, T_x[:, r], "--", color="deepskyblue", lw=2.0)
        mm = (cls[Vw] == 1) & (role[Vw] == 0) & (voff[Vw] > THR)
        ax.scatter(s_v[mm], T_v[mm, r], s=170, marker="^", facecolor="none", edgecolor="red", lw=1.5)
        for jj in range(2):
            axs[r, jj].set_ylabel(["e2 (cm)", "e3 (cm, drift side)"][r])
    for jj in range(2):
        axs[1, jj].set_xlabel("s along the image PCA (cm)")
    axs[0, 0].set_title(f"{name} {det} {ev} cl{cl}: image, Steiner vertices and terminals (triangles) by class", fontsize=9)
    axs[0, 1].set_title("seed (black; magenta > 1 cm off), edges at the stretches (olive path, brown connect, purple same-blob),\n"
                        "dashed blue = cheapest route between the same anchors that never leaves the ridge", fontsize=9)
    h = [Line2D([], [], ls="", marker="o", color="0.75", label="image")]
    h += [Line2D([], [], ls="", marker="o", color=CLS_COL[cc], label=f"{CLASSES[cc]}") for cc in range(4)]
    h += [Line2D([], [], ls="", marker="^", color="w", markeredgecolor="red", markersize=11, label="1blank terminal > 1 cm off")]
    axs[0, 0].legend(handles=h, fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(f"{out}_{name}_frame.png", dpi=105); plt.close(fig)


def fig_offset(name, det, ev, cl, sb, arc, j, inw, runs, cls, role, zero, voff, out):
    wi = [i for i in np.where(inw)[0] if j[i] >= 0]
    x = np.array([arc[i] for i in wi]); v = np.array([int(j[i]) for i in wi])
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(x, voff[v], "-", color="0.6", lw=1)
    for cc in range(4):
        mm = (cls[v] == cc) & (role[v] == 2)
        ax.scatter(x[mm], voff[v][mm], s=18, color=CLS_COL[cc], label=f"interior {CLASSES[cc]}", zorder=3)
        mm = (cls[v] == cc) & (role[v] != 2)
        ax.scatter(x[mm], voff[v][mm], s=80, marker="^", color=CLS_COL[cc], edgecolor="k", lw=0.6, label=f"terminal {CLASSES[cc]}", zorder=4)
    for xi, vi in zip(x, v):
        if cls[vi] in (1, 3) and voff[vi] > THR:
            ax.annotate("".join(PLANE[p] for p in range(3) if zero[vi, p]), (xi, voff[vi]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7)
    ax.axhline(THR, color="k", ls=":", lw=0.8)
    ax.set_xlabel("arc length along the seed from the click (cm)"); ax.set_ylabel("ridge offset (cm)")
    ax.set_title(f"{name} {det} {ev} cl{cl}: seed vertices; letters = the zero-charge plane(s) of an off-ridge vertex", fontsize=9)
    h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{out}_{name}_offset.png", dpi=105); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="h1,v3")
    a = ap.parse_args()
    want = set(a.cases.split(","))
    L = ["# doc pdvd/112 case anatomy (read-only; dump arms d111hst / d111vst)", ""]
    for case in CASES:
        if case[0] in want:
            one_case(case, a.out, L)
    txt = "\n".join(L) + "\n"
    open(f"{a.out}_cases.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
