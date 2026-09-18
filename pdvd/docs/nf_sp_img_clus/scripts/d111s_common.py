#!/usr/bin/env python3
"""doc pdvd/111 round 2 -- shared readers for the Steiner-graph dump (WCT_STEINER_GRAPH_DUMP) and the STM path trace.

Trace lines read (stdout of a d111_run_arms.sh arm, gzipped per event):
  STGC <ident> <call> nret=.. nbase_v=.. nbase_e=.. ncomp_closely=.. ncomp_ctpc=.. ncomp_base=.. nterm=.. nextreme=.. nsv=.. nse=.. classify=..
  STGR <ident> <x> <y> <z> <slice> <qu> <qv> <qw> [<bi> <wu> <wv> <ww> <uu> <uv> <uw>]   the retiled cloud (SteinerGrapher.cxx
        steiner_graph_dump); the bracketed seven are doc 114's appended fields (blob major index, wire index, charge unc per plane)
  STGB <ident> <bi> <apa> <face> <slice> <umin> <umax> <vmin> <vmax> <wmin> <wmax> <max_type> <min_type> <max_int> <min_int>  (doc 114)
  STGV <ident> <i> <x> <y> <z> <old> <term> <extreme> <q> <m02> <m61>
  STGE <ident> <s> <t> <len> <w> <src> <base> <base_w> <n ok3 ok2 ok1 live1 @0.2cm/ch0> <n ok3 ok2 ok1 live1 @0.6cm/ch1>
  STMRP <ident> <rough|crawl1|crawl2> <from> <to> <npath>          TaggerCheckSTM do_rough_path / adjust_rough_path
  STMPATHN <cl>/<dir>/<r1|r2> <stage> <n>  +  STMPATH <tag> <stage> <i> <x> <y> <z>   (TrackFitting.cxx stm_path_dump)

Every item carries its stream position, so "the graph a walk ran on" is the last STGC block of that cluster before the
walk, and "the walk behind a seed" is the last STMRP of that cluster before the seed header.
"""
import collections, gzip, re, sys
sys.dont_write_bytecode = True
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts")
import d111_stage_attrib as A      # Ridge, closest_on_polyline, bee_layer (doc 111 round 1, unchanged)

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
DOC = IMG + "/pdvd/docs/nf_sp_img_clus"
SRC_CODE = {"path": 0, "connect": 1, "sbt": 2}
BASE_CODE = {"closely_same": 0, "closely_other": 1, "ctpc": 2, "mst": 3, "none": 4}
SRC_NAME = {v: k for k, v in SRC_CODE.items()}
BASE_NAME = {v: k for k, v in BASE_CODE.items()}


class SteinerBlock:
    """One steiner_graph_dump call: the retiled cloud, the steiner_pc vertices and the steiner_graph edges."""

    def __init__(self, ident, call, order, header):
        self.ident, self.call, self.order, self.header = ident, call, order, header
        self._R, self._V, self._E, self._B = [], [], [], []

    def finish(self):
        # doc 114: STGR carries 7 fields (docs 111-113) or 14 (blob index, wire index and charge
        # uncertainty per plane appended last); STGB is the per-blob line (doc 114 only).
        R = np.array(self._R, float)
        ncol = len(self._R[0]) if len(self._R) else 7
        R = R.reshape(-1, ncol)
        self.R = R[:, 0:3]; self.R_slice = R[:, 3].astype(int); self.R_q = R[:, 4:7]
        if ncol == 14:
            self.R_blob = R[:, 7].astype(int); self.R_w = R[:, 8:11].astype(int); self.R_unc = R[:, 11:14]
        else:
            self.R_blob = self.R_w = self.R_unc = None
        B = np.array(self._B, float).reshape(-1, 14)
        # bi apa face slice umin umax vmin vmax wmin wmax max_type min_type max_int min_int
        self.B = B.astype(int) if len(B) else None
        V = np.array(self._V, float).reshape(-1, 10)
        order = np.argsort(V[:, 0], kind="stable")
        V = V[order]
        self.V = V[:, 1:4]; self.V_old = V[:, 4].astype(int); self.V_term = V[:, 5].astype(bool)
        self.V_ext = V[:, 6].astype(bool); self.V_q = V[:, 7]; self.V_m02 = V[:, 8].astype(int); self.V_m61 = V[:, 9].astype(int)
        self.nv = len(V)
        if len(V) and not np.array_equal(V[:, 0].astype(int), np.arange(len(V))):
            raise ValueError(f"STGV indices of ident {self.ident} are not 0..n-1")
        E = np.array(self._E, float).reshape(-1, 17)
        self.E_s = E[:, 0].astype(int); self.E_t = E[:, 1].astype(int); self.E_len = E[:, 2]; self.E_w = E[:, 3]
        self.E_src = E[:, 4].astype(int); self.E_base = E[:, 5].astype(int); self.E_bw = E[:, 6]
        self.S02 = E[:, 7:12].astype(int); self.S61 = E[:, 12:17].astype(int)      # n ok3 ok2 ok1 live1
        self.edge_index = {}
        for k, (s, t) in enumerate(zip(self.E_s, self.E_t)):
            self.edge_index[(min(s, t), max(s, t))] = k
        del self._R, self._V, self._E, self._B
        return self

    def csr(self, w=None):
        """symmetric weight matrix; boost never holds a parallel edge (add_edge is guarded), so no duplicates."""
        w = self.E_w if w is None else w
        n = self.nv
        M = csr_matrix((np.r_[w, w], (np.r_[self.E_s, self.E_t], np.r_[self.E_t, self.E_s])), shape=(n, n))
        return M

    def walk(self, src, dst, w=None):
        """Dijkstra path src -> dst (vertex list) and its cost under w; None when unreachable."""
        dist, pred = dijkstra(self.csr(w), directed=False, indices=src, return_predecessors=True)
        if not np.isfinite(dist[dst]):
            return None, np.inf
        path = [dst]
        while path[-1] != src:
            path.append(pred[path[-1]])
        return path[::-1], float(dist[dst])

    def path_cost(self, path, w=None):
        w = self.E_w if w is None else w
        c = 0.0
        for a, b in zip(path[:-1], path[1:]):
            k = self.edge_index.get((min(a, b), max(a, b)))
            if k is None:
                return np.nan
            c += w[k]
        return c


SPLICE = re.compile(r"\[\d\d:\d\d:\d\d\.\d{3}\]")


def parse_trace(path, want_stg=True):
    """-> dict(stg=[SteinerBlock], rp=[(order, ident, kind, from, to, npath)], blocks=[path blocks], repaired=n).

    stdout and stderr share the per-event log.  A spdlog stderr line ("[hh:mm:ss.mmm] I [clus] ...") can land in the
    middle of a buffered dump line; the dump line then continues on the next line.  Such a line is cut at the timestamp
    and rejoined with its continuation (counted in `repaired`)."""
    stg, rp, blocks, cur = [], [], [], {}
    sb = None
    fill = None
    pending = None
    repaired = 0
    with gzip.open(path, "rt", errors="replace") as fh:
        for order, line in enumerate(fh):
            if pending is not None:
                line = pending + line
                pending = None
                repaired += 1
            if line[:1] == "S":
                m = SPLICE.search(line)
                if m and m.start() > 0:          # the split can fall anywhere, even after "ST"
                    pending = line[:m.start()]
                    continue
            c = line[:4]
            if c == "STGR" and sb is not None:
                f = line.split()
                if len(f) >= 16:
                    sb._R.append((float(f[2]), float(f[3]), float(f[4]), int(f[5]), float(f[6]), float(f[7]), float(f[8]),
                                  int(f[9]), int(f[10]), int(f[11]), int(f[12]), float(f[13]), float(f[14]), float(f[15])))
                else:
                    sb._R.append((float(f[2]), float(f[3]), float(f[4]), int(f[5]), float(f[6]), float(f[7]), float(f[8])))
            elif c == "STGB" and sb is not None:
                f = line.split()
                sb._B.append(tuple(int(x) for x in f[2:16]))
            elif c == "STGV" and sb is not None:
                f = line.split()
                sb._V.append((int(f[2]), float(f[3]), float(f[4]), float(f[5]), int(f[6]), int(f[7]), int(f[8]),
                              float(f[9]), int(f[10]), int(f[11])))
            elif c == "STGE" and sb is not None:
                f = line.split()
                sb._E.append((int(f[2]), int(f[3]), float(f[4]), float(f[5]), SRC_CODE[f[6]], BASE_CODE[f[7]], float(f[8]),
                              *[int(x) for x in f[9:19]]))
            elif c == "STGC":
                if sb is not None:
                    stg.append(sb.finish())
                f = line.split()
                hdr = dict(kv.split("=") for kv in f[3:])
                sb = SteinerBlock(int(f[1]), int(f[2]), order, {k: int(v) for k, v in hdr.items()}) if want_stg else None
            elif c == "STMR" and line.startswith("STMRP "):
                f = line.split()
                rp.append((order, int(f[1]), f[2], int(f[3]), int(f[4]), int(f[5])))
            elif c == "STMP":
                if line.startswith("STMPATHN "):
                    f = line.split(); tag, stage = f[1], f[2]
                    if stage not in ("seed", "final"):
                        fill = None; continue
                    if stage == "seed" or tag not in cur:
                        p = tag.split("/")
                        b = dict(tag=tag, cl=int(p[0]) if p[0].lstrip("-").isdigit() else -1, dir=p[1] if len(p) > 1 else "?",
                                 rnd=p[2] if len(p) > 2 else "?", st={}, order=order)
                        blocks.append(b); cur[tag] = b
                    b = cur[tag]; b["st"][stage] = []; fill = (b, stage, tag)
                elif line.startswith("STMPATH ") and fill is not None:
                    f = line.split()
                    if f[1] == fill[2] and f[2] == fill[1]:
                        fill[0]["st"][fill[1]].append((float(f[4]), float(f[5]), float(f[6])))
    if sb is not None:
        stg.append(sb.finish())
    for b in blocks:
        b["st"] = {k: np.array(v, float).reshape(-1, 3) for k, v in b["st"].items()}
    return dict(stg=stg, rp=rp, blocks=blocks, repaired=repaired)


def graph_for(tr, cl, order):
    """the last Steiner dump of cluster cl before stream position `order`."""
    best = None
    for sb in tr["stg"]:
        if sb.ident == cl and sb.order < order and (best is None or sb.order > best.order):
            best = sb
    return best


def walks_for(tr, cl, order, after=-1):
    """STMRP lines of cluster cl in (after, order), in stream order."""
    return [r for r in tr["rp"] if r[1] == cl and after < r[0] < order]


def join_vertices(sb, P, tol=1e-3):
    """steiner_pc index of each point of polyline P (cm); -1 where no vertex lies within tol (print precision 6 digits)."""
    from scipy.spatial import cKDTree
    if sb is None or sb.nv == 0 or len(P) == 0:
        return np.full(len(P), -1)
    d, j = cKDTree(sb.V).query(P)
    tol_eff = np.maximum(tol, 5e-6 * np.abs(P).max(axis=1))    # 6 significant digits on |x| up to ~600 cm
    return np.where(d <= tol_eff, j, -1)


def polyline_samples(P, step=0.3):
    """points every ~step cm along polyline P with the length each represents."""
    if len(P) < 2:
        return np.zeros((0, 3)), np.zeros(0)
    out, wts = [], []
    for a, b in zip(P[:-1], P[1:]):
        L = np.linalg.norm(b - a)
        n = max(1, int(round(L / step)))
        f = (np.arange(n) + 0.5) / n
        out.append(a + f[:, None] * (b - a)); wts.append(np.full(n, L / n))
    return np.concatenate(out), np.concatenate(wts)


def resample(P, step=1.2):
    """polyline resampled at a fixed arc-length step (organize_orig_path spacing), ends kept."""
    if len(P) < 2:
        return P.copy()
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] <= 0:
        return P[:1].copy()
    q = np.r_[np.arange(0, s[-1], step), s[-1]]
    return np.c_[[np.interp(q, s, P[:, k]) for k in range(3)]].T


def wiggle_share(P, step=1.2, thr=0.5):
    """share of interior points of the step-resampled polyline whose distance from the chord of their two neighbours > thr cm."""
    Q = resample(P, step)
    if len(Q) < 3:
        return 0, 0
    a, b, c = Q[:-2], Q[1:-1], Q[2:]
    ac = c - a; L = np.linalg.norm(ac, axis=1)
    cr = np.linalg.norm(np.cross(b - a, ac), axis=1)
    d = np.where(L > 0, cr / np.where(L > 0, L, 1), np.linalg.norm(b - a, axis=1))
    return int((d > thr).sum()), len(d)
