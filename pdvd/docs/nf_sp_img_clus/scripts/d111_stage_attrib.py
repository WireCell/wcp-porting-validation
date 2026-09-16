#!/usr/bin/env python3
"""doc pdvd/111 A4 -- at which stage of the STM trajectory chain does a fitted row leave the image?

Reads, per event of a trace arm (d111_run_arms.sh with WCT_STM_PATH_DEBUG=1 WCT_TRAJ_ASSOC_DEBUG=1):
  * the gzipped stdout trace  $LOGD/evt_<run6>_<evt>.log.gz
      STMPATHN/STMPATH <cl>/<fwd|bwd>/<r1|r2> <stage> ...   (TrackFitting.cxx stm_path_dump, doc pdhd/11)
      TRAJASSOC ...                                         (TrackFitting.cxx trajectory_fit, doc pdvd/111)
  * tracking-stm.root T_rec_charge (the persisted fit) and T_proj_data (the cluster's measured cells);
  * mabc-pr.zip clustering-global (the image) and steiner_graph-global (the Steiner cloud, tagged clusters only).

The persisted pass is ONE do_single_tracking call (the "r2" block): adjust_rough_path returns an empty path unless it
crawls (TaggerCheckSTM.cxx:1612-1616), and the caller then refits from the SAME Steiner Dijkstra path (:3773); when it
crawls, the r2 seed is two Dijkstra walks on the same graph.  So the chain behind every persisted row is
    seed (Steiner Dijkstra path) -> org1 (1.2 cm fill) -> fit1 -> org2 (0.6 cm resample) -> fit2 -> org3[f] (fill) -> final
and the r1 block only decides the crawl.  Each persisted record (cluster, pass) is matched to the r2 block whose
`final` stage equals its T_rec_charge rows (count equal, max |dx| < 0.01 cm).

Metrics (all fit-independent):
  * ridge offset d(X): the image is voxelised per cluster (1 cm); each occupied voxel carries the charge-weighted
    centroid and principal axis of the image points within RLOC cm of it; d(X) = distance of X from the axis line
    of the voxel nearest to X (plus the voxel distance when X is > RLOC from every voxel);
  * per stage S, the location corresponding to a final row F is the closest point of S's polyline to F; d_S = d(that
    point).  A final row is DEVIATED when d_final > THR (1 cm).  Its ORIGIN is the earliest stage of the contiguous run
    of stages with d_S > THR that ends at `final` (so a deviation the fit removed and re-made is attributed to where it
    was re-made);
  * seed sub-classes: `crawl_reroute` (the round-2 seed at the location is > THR from the round-1 Steiner path:
    adjust_rough_path's crawl re-routed it), `long_edge` (the seed segment carrying the closest point is > LONG_EDGE cm:
    a straight chord across the cloud or a bridge), `cloud_has_ridge` (the Steiner cloud of that cluster, where available, holds a point within 0.5 cm of the
    ridge within 1.5 cm of the location -- the path walked past on-ridge cloud points) vs `cloud_off` (it does not);
  * fit sub-classes, from the TRAJASSOC record of the entry fit call nearest to the location: `one_plane_<U|V|W>` (the
    solve is off but dropping that plane puts it back within THR), `multi_plane` (no single-plane drop does),
    `revert`/`skip` (post-solve steps);
  * the transfer table: d_final binned by d_seed over every persisted row (does the fit recover a displaced seed?);
  * the replay check: every kept TRAJASSOC `final` position equals the fit1/fit2 STMPATH position of its call.

Usage:
  d111_stage_attrib.py --det pdhd --arm d111htr [--logd /home/xqian/tmp/d111/arm_d111htr] [--events a,b] \
      --out figs/111_attrib_pdhd      # writes <out>.txt (census) and <out>_rows.tsv.gz (deviated rows)
"""
import argparse, collections, glob, gzip, json, os, re, sys, zipfile
import numpy as np
import uproot
from scipy.spatial import cKDTree

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
BASE = {"pdhd": (0, 3200, 6400), "pdvd": (0, 3808, 7616)}
THR = 1.0          # cm
RLOC = 3.0         # cm
VOX = 1.0          # cm
CHAIN = ["seed", "org1", "fit1", "org2", "fit2", "org3", "final"]
ANGLE_BINS = [0, 30, 50, 65, 75, 85, 90.01]
LONG_EDGE = 3.0    # cm: a seed segment longer than this is a straight bridge across the cloud


# ------------------------------------------------------------------ parsing
def parse_trace(path):
    """-> list of blocks in stream order: dict(tag, cl, dir, rnd, stages{name: (n,3)}, calls{method: call_id})
    and assoc{call_id: list of TRAJASSOC field arrays}."""
    blocks, cur, assoc = [], {}, collections.defaultdict(list)
    fill = None      # (block, stage, n_expected)
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("STMPATHN "):
                f = line.split()
                if len(f) < 4:
                    continue
                tag, stage, n = f[1], f[2], int(f[3])
                if stage == "seed" or tag not in cur:
                    parts = tag.split("/")
                    b = dict(tag=tag, cl=int(parts[0]) if parts[0].lstrip("-").isdigit() else -1,
                             dir=parts[1] if len(parts) > 1 else "?", rnd=parts[2] if len(parts) > 2 else "?",
                             stages={}, calls={}, order=len(blocks))
                    blocks.append(b); cur[tag] = b
                b = cur[tag]
                b["stages"][stage] = []
                fill = (b, stage)
            elif line.startswith("STMPATH "):
                f = line.split()
                if fill is not None and len(f) >= 7 and f[1] == fill[0]["tag"] and f[2] == fill[1]:
                    fill[0]["stages"][fill[1]].append((float(f[4]), float(f[5]), float(f[6])))
            elif line.startswith("TRAJASSOC "):
                f = line.split()
                if len(f) < 48:
                    continue
                tag, method, call = f[1], int(f[2]), int(f[3])
                if tag in cur:
                    cur[tag]["calls"].setdefault(method, call)
                    if cur[tag]["calls"][method] != call:     # a second call of the same method on a stale tag
                        cur[tag]["calls"][method] = call
                vals = [np.nan if v == "nan" else float(v) for v in f[4:]]
                assoc[call].append(vals)
    for b in blocks:
        b["stages"] = {k: np.array(v, float).reshape(-1, 3) for k, v in b["stages"].items()}
        if "org3f" in b["stages"]:
            b["stages"]["org3"] = b["stages"]["org3f"]
    return blocks, {k: np.array(v, float) for k, v in assoc.items()}


# TRAJASSOC column indices after the 4 leading fields (tag method call dropped): i cluster apa face kept skip rev nan
A_I, A_CL, A_KEPT, A_SKIP, A_REV = 0, 1, 4, 5, 6
A_INIT, A_SOL, A_FIN, A_LOO = 8, 11, 14, 17      # LOO: U 17..19, V 20..22, W 23..25


# ------------------------------------------------------------------ image ridge field
class Ridge:
    def __init__(self, P, q):
        self.ok = len(P) >= 5
        if not self.ok:
            return
        w = np.clip(q, 1, None)
        key = np.floor(P / VOX).astype(np.int64)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        W = np.bincount(inv, weights=w)
        C = np.c_[[np.bincount(inv, weights=w * P[:, k]) for k in range(3)]].T / W[:, None]
        self.vc = C
        tree = cKDTree(C)
        self.cen = np.zeros_like(C); self.ax = np.zeros_like(C)
        for j, nb in enumerate(tree.query_ball_point(C, RLOC)):
            ww = W[nb]; X = C[nb]
            c = np.average(X, axis=0, weights=ww)
            if len(nb) >= 2:
                _, _, vt = np.linalg.svd((X - c) * np.sqrt(ww / ww.sum())[:, None], full_matrices=False)
                a = vt[0]
            else:
                a = np.array([1.0, 0, 0])
            self.cen[j] = c; self.ax[j] = a
        self.tree = tree

    def offset(self, X):
        X = np.atleast_2d(X)
        if not self.ok or len(X) == 0:
            return np.full(len(X), np.nan)
        dv, j = self.tree.query(X)
        V = X - self.cen[j]
        t = np.sum(V * self.ax[j], axis=1)
        perp = np.linalg.norm(V - t[:, None] * self.ax[j], axis=1)
        return np.where(dv > RLOC, np.maximum(perp, dv), perp)


def closest_on_polyline(P, X):
    """closest point on polyline P (n,3) to each X (m,3); also the length of the segment it lies on."""
    if len(P) == 0:
        return np.full_like(X, np.nan), np.full(len(X), np.nan)
    if len(P) == 1:
        return np.repeat(P, len(X), axis=0), np.zeros(len(X))
    tree = cKDTree(P)
    _, j = tree.query(X)
    best = np.full_like(X, np.nan); bd = np.full(len(X), np.inf); seg = np.zeros(len(X))
    for k in (-1, 0):
        a = np.clip(j + k, 0, len(P) - 2)
        A, B = P[a], P[a + 1]
        AB = B - A; L2 = np.sum(AB * AB, axis=1)
        t = np.where(L2 > 0, np.sum((X - A) * AB, axis=1) / np.where(L2 > 0, L2, 1), 0)
        t = np.clip(t, 0, 1)
        Q = A + t[:, None] * AB
        d = np.linalg.norm(Q - X, axis=1)
        upd = d < bd
        best[upd] = Q[upd]; bd[upd] = d[upd]; seg[upd] = np.sqrt(L2[upd])
    return best, seg


def fit_response(match, assoc, R, Fi):
    """what the two fit calls did at a location whose deviation came from the seed:
    `recovered_reverted` -- a solve put the point within THR of the ridge and the area smoothing reverted it (or
    skip_trajectory_point dropped it); `recovered_kept` -- a solve put it within THR and kept it (a later stage re-made the
    offset); `partial` -- a solve moved it >= 0.3 cm toward the ridge but not within THR; `no_move` -- neither solve
    moved it 0.3 cm toward the ridge (the association window saw no ridge charge); `no_assoc`."""
    best = None
    for meth, prev in ((1, "org1"), (2, "org2")):
        call = match["calls"].get(meth)
        A_ = assoc.get(call)
        P = match["stages"].get(prev)
        if A_ is None or len(A_) == 0 or P is None or len(P) == 0:
            continue
        Q, _ = closest_on_polyline(P, Fi[None, :])
        jj = int(np.argmin(np.linalg.norm(A_[:, A_INIT:A_INIT + 3] - Q[0], axis=1)))
        rec = A_[jj]
        dinit = R.offset(rec[A_INIT:A_INIT + 3])[0]
        dsol = R.offset(rec[A_SOL:A_SOL + 3])[0]
        if dsol <= THR and (rec[A_REV] == 1 or rec[A_SKIP] == 1):
            return "recovered_reverted"
        if dsol <= THR:
            best = "recovered_kept"
        elif dinit - dsol >= 0.3 and best is None:
            best = "partial"
        elif best is None:
            best = "no_move"
    return best or "no_assoc"


# ------------------------------------------------------------------ per event
def bee_layer(zpath, layer):
    with zipfile.ZipFile(zpath) as z:
        name = f"data/0/0-{layer}.json"
        if name not in z.namelist():
            return None
        d = json.loads(z.read(name))
    if not d.get("x"):
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0)
    q = np.array(d["q"], float) if "q" in d else np.ones(len(d["x"]))
    return np.c_[d["x"], d["y"], d["z"]], np.array(d["cluster_id"]), q


def onq_pm1(cells, pu, pt):
    return np.array([any((int(round(w)) + dd, int(round(s))) in cells for dd in (-1, 0, 1)) for w, s in zip(pu, pt)])


def event(det, arm, pre, logd, out_rows, acc):
    d = f"{IMG}/{det}/work/{pre}_{arm}"
    lg = f"{logd}/evt_{pre}.log.gz"
    if not (os.path.exists(lg) and os.path.exists(f"{d}/tracking-stm.root")):
        acc["events_missing"] += 1
        return
    blocks, assoc = parse_trace(lg)
    f = uproot.open(f"{d}/tracking-stm.root")
    t = f["T_rec_charge"].arrays(library="np")
    img = bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    stc = bee_layer(f"{d}/mabc-pr.zip", "steiner_graph-global")
    # measured cells per cluster/plane
    pd = f["T_proj_data"].arrays(library="np")
    cells = collections.defaultdict(lambda: [set(), set(), set()])
    for cid, ch, ts, q in zip(pd["cluster_id"][0], pd["channel"][0], pd["time_slice"][0], pd["charge"][0]):
        ch = np.asarray(ch).astype(int); ts = np.asarray(ts).astype(int); m = np.asarray(q) > 0
        pl = np.searchsorted(np.array(BASE[det][1:]), ch, side="right")
        for p in range(3):
            mm = m & (pl == p)
            cells[int(cid) // 10][p].update(zip(ch[mm].tolist(), ts[mm].tolist()))
    acc["events"] += 1
    # replay check over every call of every block
    for b in blocks:
        for method, call in b["calls"].items():
            st = b["stages"].get(f"fit{method}")
            A = assoc.get(call)
            if st is None or A is None:
                continue
            kept = A[A[:, A_KEPT] == 1]
            if len(kept) != len(st):
                acc["replay_size_mismatch"] += 1
                continue
            dmax = np.max(np.abs(kept[:, A_FIN:A_FIN + 3] - st)) if len(st) else 0.0
            acc["replay_calls"] += 1
            acc["replay_max_dev_cm"] = max(acc["replay_max_dev_cm"], float(dmax))
            if dmax > 1e-3 * 5:
                acc["replay_bad"] += 1
    r2 = collections.defaultdict(list)
    for b in blocks:
        if b["rnd"] == "r2" and "final" in b["stages"]:
            r2[b["cl"]].append(b)
    ridges = {}
    for (cl, ps) in sorted(set(zip(t["cluster_id"].tolist(), t["pass"].tolist()))):
        m = (t["cluster_id"] == cl) & (t["pass"] == ps)
        F = np.c_[t["x"][m], t["y"][m], t["z"][m]]
        acc["records"] += 1
        match = None
        for b in r2.get(cl, []):
            Pf = b["stages"]["final"]
            if len(Pf) == len(F) and np.max(np.abs(Pf - F)) < 0.01:
                match = b
        if match is None:
            acc["records_unmatched"] += 1
            continue
        acc["records_matched"] += 1
        if cl not in ridges:
            mi = img[1] == cl
            ridges[cl] = Ridge(img[0][mi], img[2][mi])
        R = ridges[cl]
        if not R.ok:
            acc["records_no_image"] += 1
            continue
        n = len(F)
        dfin = R.offset(F)
        dst = {}; seglen = None
        for s in CHAIN:
            P = match["stages"].get(s)
            if P is None or len(P) == 0:
                dst[s] = np.full(n, np.nan); continue
            Q, sl = closest_on_polyline(P, F)
            dst[s] = R.offset(Q)
            if s == "seed":
                seglen = sl; seedQ = Q
                acc["rows_long_edge"] += int((sl > LONG_EDGE).sum()); acc["rows_seg_known"] += n
        # direction and angle to drift from the final rows
        D = np.zeros_like(F)
        for i in range(n):
            a, bb = max(0, i - 3), min(n - 1, i + 3)
            v = F[bb] - F[a]; nv = np.linalg.norm(v); D[i] = v / nv if nv > 0 else (1, 0, 0)
        ang = np.degrees(np.arccos(np.clip(np.abs(D[:, 0]), 0, 1)))
        # step length for length weighting
        step = np.r_[np.linalg.norm(np.diff(F, axis=0), axis=1), 0]
        # on-charge +-1 wire on the final rows
        cc = cells[cl]
        off = np.zeros(n, int)
        for p, key in enumerate(("pu", "pv", "pw")):
            off += (~onq_pm1(cc[p], t[key][m], t["pt"][m])).astype(int)
        q = t["q"][m]
        acc["rows"] += n; acc["len"] += step.sum()
        acc["rows_off1"] += int((off >= 1).sum())
        # transfer table: every row with a finite seed offset
        for i in range(n):
            if np.isfinite(dst["seed"][i]) and np.isfinite(dfin[i]):
                acc["transfer"].append((dst["seed"][i], dfin[i], dst["fit1"][i], ang[i]))
        dev = dfin > THR
        acc["rows_dev"] += int(dev.sum()); acc["len_dev"] += float(step[dev].sum())
        # validation of the 3-D ridge metric against the independent 2-D cell test and the Bee hole test
        acc["dev_off1"] += int(((off >= 1) & dev).sum()); acc["nondev_off1"] += int(((off >= 1) & ~dev).sum())
        acc["dev_qneg"] += int(((q < 0) & dev).sum()); acc["nondev_qneg"] += int(((q < 0) & ~dev).sum())
        acc["rows_dev2"] += int((dfin > 2 * THR).sum())
        # Steiner cloud of this cluster (tagged clusters only)
        cloud = None
        if stc is not None and len(stc[0]):
            mc = stc[1] == cl
            if mc.sum() >= 3:
                cloud = stc[0][mc]
        calls = match["calls"]
        # the round-1 block of the same (cluster, direction) that precedes the matched round-2 block: without a crawl the
        # round-2 seed IS the round-1 seed; a crawl re-routes it through the crawl end point (adjust_rough_path)
        r1s = [b for b in blocks if b["cl"] == cl and b["dir"] == match["dir"] and b["rnd"] == "r1"
               and b["order"] < match["order"] and "seed" in b["stages"]]
        r1seed = r1s[-1]["stages"]["seed"] if r1s else None
        if r1seed is not None and "seed" in match["stages"]:
            s2 = match["stages"]["seed"]
            acc["records_rerouted"] += int(not (len(s2) == len(r1seed) and np.allclose(s2, r1seed)))
        for i in np.where(dev)[0]:
            ds = [dst[s][i] for s in CHAIN]
            k = len(CHAIN) - 1
            while k > 0 and np.isfinite(ds[k - 1]) and ds[k - 1] > THR:
                k -= 1
            origin = CHAIN[k]
            sub = ""
            if origin == "seed":
                acc["seed_seglen"].append(float(seglen[i]) if seglen is not None else np.nan)
                if seglen is not None and seglen[i] > LONG_EDGE:
                    sub = "long_edge"
                elif cloud is None:
                    sub = "cloud_na"
                else:
                    near = cloud[np.linalg.norm(cloud - seedQ[i], axis=1) < 1.5]
                    sub = "cloud_has_ridge" if len(near) and np.nanmin(R.offset(near)) < 0.5 else "cloud_off"
            if origin in ("seed", "org1") and r1seed is not None:
                Q2, _ = closest_on_polyline(match["stages"]["seed"], F[i:i + 1])
                Q1, _ = closest_on_polyline(r1seed, Q2)
                if np.linalg.norm(Q1[0] - Q2[0]) > THR:
                    sub = "crawl_reroute"         # the round-2 seed left the round-1 Steiner path here
            if origin in ("seed", "org1"):
                acc["seed_fitresp"][(sub, fit_response(match, assoc, R, F[i]))] += 1
            if origin in ("fit1", "fit2"):
                call = calls.get(int(origin[-1]))
                A = assoc.get(call)
                if A is None or len(A) == 0:
                    sub = "no_assoc"
                else:
                    Pst = match["stages"][origin]
                    Q, _ = closest_on_polyline(Pst, F[i:i + 1])
                    jj = int(np.argmin(np.linalg.norm(np.where(np.isfinite(A[:, A_FIN:A_FIN + 3]),
                                                                A[:, A_FIN:A_FIN + 3], A[:, A_SOL:A_SOL + 3]) - Q[0], axis=1)))
                    rec = A[jj]
                    if rec[A_REV] == 1:
                        sub = "revert"
                    elif rec[A_SKIP] == 1:
                        sub = "skip"
                    else:
                        dsol = R.offset(rec[A_SOL:A_SOL + 3])[0]
                        dinit = R.offset(rec[A_INIT:A_INIT + 3])[0]
                        dloo = R.offset(rec[A_LOO:A_LOO + 9].reshape(3, 3))
                        good = [pl for pl in range(3) if dloo[pl] <= THR]
                        if dsol <= THR:
                            sub = "solve_on_then_moved"
                        elif len(good) == 1:
                            sub = "one_plane_" + "UVW"[good[0]]
                        elif len(good) >= 2:
                            sub = "one_plane_any"
                        else:
                            sub = "multi_plane"
                        if np.isfinite(dinit) and dinit > THR:
                            sub += "+init_off"
            ab = np.searchsorted(ANGLE_BINS, ang[i], side="right") - 1
            acc["origin"][origin] += 1
            acc["origin_len"][origin] += step[i]
            acc["origin_sub"][(origin, sub)] += 1
            acc["origin_angle"][(ab, origin)] += 1
            acc["angle_dev"][ab] += 1
            out_rows.append(f"{pre}\t{cl}\t{ps}\t{i}\t{F[i,0]:.2f}\t{F[i,1]:.2f}\t{F[i,2]:.2f}\t{ang[i]:.1f}\t"
                            f"{q[i]:.0f}\t{off[i]}\t" + "\t".join(f"{v:.2f}" for v in ds) + f"\t{origin}\t{sub}")
        for i in range(n):
            ab = np.searchsorted(ANGLE_BINS, ang[i], side="right") - 1
            acc["angle_rows"][ab] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--logd")
    ap.add_argument("--events", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logd = a.logd or f"/home/xqian/tmp/d111/arm_{a.arm}"
    evs = sorted(os.path.basename(p)[:-len(a.arm) - 1] for p in glob.glob(f"{IMG}/{a.det}/work/*_{a.arm}"))
    if a.events:
        evs = [e for e in evs if e in set(a.events.split(","))]
    acc = collections.defaultdict(int)
    for k in ("origin", "origin_len", "origin_sub", "origin_angle", "angle_dev", "angle_rows", "seed_fitresp"):
        acc[k] = collections.Counter()
    acc["transfer"] = []; acc["replay_max_dev_cm"] = 0.0; acc["seed_seglen"] = []
    rows = []
    for pre in evs:
        event(a.det, a.arm, pre, logd, rows, acc)
        print(f"  {pre}: records {acc['records']} matched {acc['records_matched']} rows {acc['rows']} dev {acc['rows_dev']}",
              file=sys.stderr, flush=True)
    L = []
    P = L.append
    P(f"# doc pdvd/111 A4 stage attribution: det={a.det} arm={a.arm} events={acc['events']} (missing trace/root {acc['events_missing']})")
    P(f"# THR={THR} cm, RLOC={RLOC} cm, VOX={VOX} cm; chain {CHAIN}")
    P("")
    P("## replay check (TRAJASSOC kept `fin` == STMPATH fit1/fit2 of the same call)")
    P(f"calls compared {acc['replay_calls']}, size mismatches {acc['replay_size_mismatch']}, calls with max|d| > 0.005 cm {acc['replay_bad']}, "
      f"max |d| {acc['replay_max_dev_cm']:.4f} cm")
    P("")
    P("## persisted records")
    P(f"records {acc['records']}, matched to an r2 trace block {acc['records_matched']}, unmatched {acc['records_unmatched']}, "
      f"no image {acc['records_no_image']}; round-2 seed re-routed by the crawl (differs from the round-1 seed) in "
      f"{acc['records_rerouted']}")
    rows_n = acc["rows"]
    P(f"rows {rows_n}, length {acc['len']/100:.1f} m; ridge offset > {THR} cm: {acc['rows_dev']} ({100*acc['rows_dev']/max(rows_n,1):.1f} %, "
      f"{acc['len_dev']/100:.1f} m); > {2*THR} cm: {acc['rows_dev2']} ({100*acc['rows_dev2']/max(rows_n,1):.1f} %); "
      f"off-charge +-1 wire in >= 1 plane: {acc['rows_off1']} ({100*acc['rows_off1']/max(rows_n,1):.1f} %)")
    nd, nn_ = acc["rows_dev"], rows_n - acc["rows_dev"]
    P(f"validation: off-charge (>= 1 plane, +-1 wire) among deviated rows {100*acc['dev_off1']/max(nd,1):.1f} % vs non-deviated "
      f"{100*acc['nondev_off1']/max(nn_,1):.1f} %; Bee-dropped q<0 among deviated {100*acc['dev_qneg']/max(nd,1):.1f} % vs non-deviated "
      f"{100*acc['nondev_qneg']/max(nn_,1):.1f} %; share of all q<0 rows that are deviated "
      f"{100*acc['dev_qneg']/max(acc['dev_qneg']+acc['nondev_qneg'],1):.1f} %")
    P("")
    P(f"## origin of deviated rows (ridge offset > {THR} cm): earliest stage of the off run that ends at `final`")
    tot = max(acc["rows_dev"], 1)
    P("| origin | rows | share | length (m) |")
    P("|---|---|---|---|")
    for s in CHAIN:
        P(f"| {s} | {acc['origin'][s]} | {100*acc['origin'][s]/tot:.1f} % | {acc['origin_len'][s]/100:.2f} |")
    P("")
    P("## sub-classes")
    P("| origin | sub-class | rows | share of origin |")
    P("|---|---|---|---|")
    for (o, sb), nn in sorted(acc["origin_sub"].items(), key=lambda kv: (CHAIN.index(kv[0][0]), -kv[1])):
        P(f"| {o} | {sb or '-'} | {nn} | {100*nn/max(acc['origin'][o],1):.1f} % |")
    P("")
    sl = np.array(acc["seed_seglen"], float)
    if len(sl):
        P(f"seed-born rows: length of the seed segment at the location p50 {np.nanmedian(sl):.2f} cm, p90 {np.nanpercentile(sl,90):.2f} cm, "
          f"> {LONG_EDGE:g} cm {100*np.nanmean(sl > LONG_EDGE):.1f} %; ALL rows on a seed segment > {LONG_EDGE:g} cm: "
          f"{100*acc['rows_long_edge']/max(acc['rows_seg_known'],1):.1f} %")
        P("")
    # the fit-born sub-classes collapsed: a single-plane drop fixes the solve / no single drop does / post-solve steps
    for o in ("fit1", "fit2"):
        tot_o = max(acc["origin"][o], 1)
        grp = collections.Counter()
        for (oo, sb), nn in acc["origin_sub"].items():
            if oo != o:
                continue
            head = sb.split("+")[0]
            key = ("one_plane" if head.startswith("one_plane") else head)
            grp[key] += nn
        P(f"{o} sub-classes collapsed: " + ", ".join(f"{k} {100*v/tot_o:.1f} %" for k, v in grp.most_common())
          + f"; one_plane rows are {100*grp['one_plane']/max(acc['rows_dev'],1):.1f} % of all deviated rows")
    P("")
    P("## seed-born deviations (origin seed or org1): what the fit did there, by seed sub-class")
    resp = ["recovered_reverted", "recovered_kept", "partial", "no_move", "no_assoc"]
    P("| seed sub-class | rows | " + " | ".join(resp) + " |")
    P("|---|---|" + "---|" * len(resp))
    subs = sorted({k[0] for k in acc["seed_fitresp"]})
    for sb in subs + ["(all)"]:
        cnt = {r: (sum(v for (s_, r_), v in acc["seed_fitresp"].items() if r_ == r) if sb == "(all)" else acc["seed_fitresp"][(sb, r)]) for r in resp}
        nn = sum(cnt.values())
        P(f"| {sb or '-'} | {nn} | " + " | ".join(f"{cnt[r]} ({100*cnt[r]/max(nn,1):.0f} %)" for r in resp) + " |")
    P("")
    P("## by angle to drift (deg): deviated share of rows, and origin split within the bin")
    P("| angle | rows | deviated | " + " | ".join(CHAIN) + " |")
    P("|---|---|---|" + "---|" * len(CHAIN))
    for ab in range(len(ANGLE_BINS) - 1):
        nr, nd = acc["angle_rows"][ab], acc["angle_dev"][ab]
        cellsx = [f"{100*acc['origin_angle'][(ab, s)]/max(nd,1):.0f} %" for s in CHAIN]
        P(f"| {ANGLE_BINS[ab]:.0f}-{min(ANGLE_BINS[ab+1],90):.0f} | {nr} | {100*nd/max(nr,1):.1f} % | " + " | ".join(cellsx) + " |")
    P("")
    P("## transfer: final ridge offset binned by the seed's ridge offset at the same location (all rows)")
    T = np.array(acc["transfer"]) if acc["transfer"] else np.zeros((0, 4))
    P("| seed offset (cm) | rows | fit1 offset p50 | final offset p50 | final p90 | final > 1 cm |")
    P("|---|---|---|---|---|---|")
    edges = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (T[:, 0] >= lo) & (T[:, 0] < hi)
        if m.sum() == 0:
            continue
        P(f"| {lo:g}-{'inf' if hi > 1e8 else f'{hi:g}'} | {int(m.sum())} | {np.nanmedian(T[m,2]):.2f} | {np.median(T[m,1]):.2f} | "
          f"{np.percentile(T[m,1],90):.2f} | {100*np.mean(T[m,1] > THR):.1f} % |")
    txt = "\n".join(L) + "\n"
    open(a.out + ".txt", "w").write(txt)
    with gzip.open(a.out + "_rows.tsv.gz", "wt") as g:
        g.write("event\tcluster\tpass\trow\tx\ty\tz\tangle\tq\tnoff_planes\t" + "\t".join(f"d_{s}" for s in CHAIN) + "\torigin\tsub\n")
        g.write("\n".join(rows) + ("\n" if rows else ""))
    print(txt)


if __name__ == "__main__":
    main()
