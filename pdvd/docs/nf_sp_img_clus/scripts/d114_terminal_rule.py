#!/usr/bin/env python3
"""doc pdvd/114 sec 5-6 -- why do Steiner terminals form off the main path, and what would a refined admission rule do?

Reads the doc-114 dump (WCT_STEINER_GRAPH_DUMP with the STGR bi/wu/wv/ww/uu/uv/uw fields and the STGB blob lines),
one Steiner dump per cluster with a persisted STM record (the last dump before the record's round-2 fit), and
replays Phase 1 of create_steiner_tree offline:

  find_steiner_terminals runs find_peak_point_indices once per BLOB (SteinerGrapher.cxx:995-1044, :803-986):
    candidacy   calc_charge_wcp(cut) > cut and quality, where quality = every plane's charge > cut OR == 0 and the
                charge is the RMS over the NONZERO planes (0 with fewer than two) -- Facade_Cluster.cxx:1031-1114,
                disable_dead_mix_cell = false (CreateSteinerGraph.cxx);
    neighbours  the 1-hop adjacency of ctpc_ref_pid restricted to the blob's own points, i.e. the in-blob edges of
                connect_graph_closely_pid: two points are adjacent when their wire indices on the blob's max-type and
                min-type planes are within max_wire_interval / min_wire_interval (connect_graph_closely.cxx:530-609);
    peaks       candidates in decreasing (charge, index) order; the first is a peak; a candidate with a higher-charge
                neighbour is not; peaks that are adjacent are merged by connected components, keeping the one
                nearest the component centroid.
  The replay is checked against the dumped terminals: every dumped non-extreme terminal must be a replayed peak
  (Phases 2, 3 and 3b only remove).

Then, for every dumped non-extreme terminal, its 3-D position against the cluster's image (ridge offset > THR = off
the ridge; nearest image point) and, when it is off, WHY the blob's peak landed there:
    ranking      the blob also holds an on-ridge 3live candidate that lost the charge ordering to this point
    ghost_blob   the blob holds no on-ridge 3live candidate, but one exists within R_NEAR in the retiled cloud
    isolated     no on-ridge 3live candidate within R_NEAR (the population any refinement must keep: gap jumps)
split by the terminal's class (3live / 1blank / 1dead / 2blank+; a 3live off-ridge terminal is on charge in all three
planes by construction, so its 3-D distance to the image says whether it is a wide image or a point the image lacks).

Finally the candidate rules are simulated per blob, with the same replay, against the baseline replay:
    wcp         today
    rank3       eligibility unchanged; the peak score is the RMS over the non-dead planes with a blank (zero, not
                dead) plane counted as 0 -- a two-plane point no longer outranks a three-plane neighbour
    nearby<R>   eligibility: a candidate with a zero plane is dropped when a 3live candidate lies within R cm
                (kd query over the retiled cloud); where nothing three-plane is near, it stays: gap jumps kept
    deadonly    a zero plane passes only when it is dead (doc 112 sec 8's blunt form, the control)
    prefer3     blob-scoped: a candidate with a zero plane is eligible only in a blob that holds no 3live candidate;
                a blob whose candidates are all blank keeps them (gap jumps kept; every candidate-bearing blob still
                yields a peak, so coverage cannot drop)
    rank3+nearby<R>, prefer3+nearby<R>
For each rule: off-ridge / on-ridge peaks, on-ridge peaks removed without a replacement peak within R_REP, peaks
removed inside the catalogue's GAP stretches (from --stretches), and the largest peak-free run along each fitted
record (peaks within 2 cm of the fit polyline projected on its arc).

Outputs: <out>.txt (all tables), <out>_terminals.tsv.gz (every dumped terminal with its attribution).

Usage: d114_terminal_rule.py --det pdhd --arm d114hbase [--logd ...] [--jobs 12] [--stretches figs/114_support_pdhd_stretches.tsv] \
           [--cut 500] --out figs/114_rule_pdhd
"""
import argparse, collections, csv, glob, gzip, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import uproot
from multiprocessing import Pool
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import d111s_common as C
from d112_terminal_planes import vertex_class, vertex_role, CLASSES
A = C.A

THR = 1.0
R_NEAR = 1.5
R_REP = 1.5
R_GAP = 2.0
R_RUN = 3.0
DEAD_THR = 1e10
RULES = ["wcp", "rank3", "prefer3", "nearby1.0", "nearby1.5", "nearby2.0", "deadonly", "rank3+nearby1.5",
         "prefer3+nearby1.0", "prefer3+nearby1.5"]
SPOTS = {"h1": ("pdhd", "029107_16", 108, np.array((75.3, 438.3, 450.1)), 12.0),
         "v3": ("pdvd", "039349_20", 25, np.array((-270.2, -243.7, 36.0)), 12.0)}


def wcp(q, cut):
    """calc_charge_wcp, disable_dead_mix_cell=false, vectorised: -> (quality, charge)."""
    nz = q != 0
    flag = (q > cut) | ~nz
    n = nz.sum(axis=1)
    ss = (q * q * nz).sum(axis=1)
    charge = np.where(n > 1, np.sqrt(ss / np.maximum(n, 1)), 0.0)
    return flag.all(axis=1), charge


def peaks_in_blob(pidx, score, adj_pairs):
    """find_peak_point_indices on one blob.  pidx: candidate point indices (int array); score: per-point score (full
    array); adj_pairs: set of (i, j) adjacent candidate pairs within the blob (both orders).  -> list of peak indices."""
    if len(pidx) == 0:
        return []
    order = sorted(pidx.tolist(), key=lambda i: (-score[i], -i))
    peak, non = set(), set()
    nb = collections.defaultdict(list)
    for i, j in adj_pairs:
        nb[i].append(j)
    for i in order:
        if i in peak or i in non:
            continue
        ok = True
        for j in nb.get(i, ()):
            if score[i] > score[j]:
                non.add(j)
            elif score[i] < score[j]:
                ok = False
                break
        if ok:
            peak.add(i)
    if len(peak) <= 1:
        return sorted(peak)
    pk = sorted(peak)
    pos = {i: k for k, i in enumerate(pk)}
    rows, cols = [], []
    for i in pk:
        for j in nb.get(i, ()):
            if j in pos:
                rows.append(pos[i]); cols.append(pos[j])
    if not rows:
        return pk
    M = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(pk), len(pk)))
    ncomp, lab = connected_components(M, directed=False)
    return [int(c) for c in lab_pick(pk, lab, ncomp)]


_XYZ = None


def lab_pick(pk, lab, ncomp):
    out = []
    P = _XYZ[pk]
    for c in range(ncomp):
        m = lab == c
        cen = P[m].mean(axis=0)
        d = np.linalg.norm(P[m] - cen, axis=1)
        out.append(np.array(pk)[m][int(np.argmin(d))])
    return out


def blob_adjacency(pidx, W, mt, nt, mi, ni):
    """in-blob adjacency among the candidate indices pidx (connect_graph_closely_pid phase 1)."""
    if len(pidx) < 2:
        return []
    wm = W[pidx, mt]; wn = W[pidx, nt]
    out = []
    for a in range(len(pidx)):
        m = (np.abs(wm - wm[a]) <= mi) & (np.abs(wn - wn[a]) <= ni)
        m[a] = False
        for b in np.where(m)[0]:
            out.append((int(pidx[a]), int(pidx[b])))
    return out


def one(args):
    global _XYZ
    det, arm, pre, logd, cut, stretches = args
    d = f"{C.IMG}/{det}/work/{pre}_{arm}"
    lg = f"{logd}/evt_{pre}.log.gz"
    acc = collections.Counter()
    term_rows = []
    runs = collections.defaultdict(list)
    if not (os.path.exists(lg) and os.path.exists(f"{d}/tracking-stm.root")):
        acc["events_missing"] += 1
        return pre, acc, term_rows, runs
    tr = C.parse_trace(lg)
    t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(["x", "y", "z", "cluster_id", "pass"], library="np")
    img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    if img is None:
        acc["events_no_image"] += 1
        return pre, acc, term_rows, runs
    acc["events"] += 1
    blocks = [b for b in tr["blocks"] if "seed" in b["st"]]
    recs = collections.defaultdict(list)
    for (cl, ps) in sorted(set(zip(t["cluster_id"].tolist(), t["pass"].tolist()))):
        m = (t["cluster_id"] == cl) & (t["pass"] == ps)
        F = np.c_[t["x"][m], t["y"][m], t["z"][m]]
        gidx = np.where(m)[0]          # global T_rec_charge row of each record row (the census's row_a / row_b)
        b2 = None
        for b in blocks:
            if b["cl"] == cl and b["rnd"] == "r2" and "final" in b["st"] and len(b["st"]["final"]) == len(F) \
                    and np.max(np.abs(b["st"]["final"] - F)) < 0.01:
                b2 = b
        if b2 is not None:
            recs[cl].append((ps, F, b2["order"], gidx))
    gap_rows = collections.defaultdict(list)
    for r in stretches:
        if r["event"] == pre and r["class"].startswith("GAP"):
            gap_rows[(int(r["cluster"]), int(r["pass"]))].append((int(r["row_a"]), int(r["row_b"])))
    for cl, rl in sorted(recs.items()):
        sb = C.graph_for(tr, cl, max(o for _, _, o, _ in rl))
        if sb is None or sb.nv == 0 or sb.R_blob is None or sb.B is None:
            acc["clusters_no_dump"] += 1
            continue
        mi_ = img[1] == cl
        R = A.Ridge(img[0][mi_], img[2][mi_])
        if not R.ok:
            acc["clusters_no_image"] += 1
            continue
        acc["clusters"] += 1
        tree_img = cKDTree(img[0][mi_])
        P, Q, W, U, BI = sb.R, sb.R_q, sb.R_w, sb.R_unc, sb.R_blob
        _XYZ = P
        npt = len(P)
        # dead per plane: the point's own vertex mask when it is a vertex, else the nearest vertex within 0.7 cm (m61)
        dead = np.zeros((npt, 3), bool)
        vm = sb.V_m61
        vdead = np.c_[[(vm >= 0) & (((vm >> (3 + p)) & 1) > 0) for p in range(3)]].T
        dead[sb.V_old] = vdead
        vt = cKDTree(sb.V)
        dv, jv = vt.query(P)
        near = (dv <= 0.7)
        dead[near] = dead[near] | vdead[jv[near]]
        zero = Q < 0.5
        sent = U >= DEAD_THR
        blank = zero & ~dead
        nzero = zero.sum(axis=1)
        cls = np.zeros(npt, int)
        one_ = nzero == 1
        cls[one_ & ~(zero & dead).any(axis=1)] = 1
        cls[one_ & (zero & dead).any(axis=1)] = 2
        cls[nzero >= 2] = 3
        qual, charge = wcp(Q, cut)
        cand = qual & (charge > cut)
        voff = R.offset(P)
        dimg, _ = tree_img.query(P)
        onr = voff <= THR
        # blob table
        B = sb.B
        binfo = {int(b[0]): b for b in B}
        # rules: eligibility + score
        scores, eligs = {}, {}
        eligs["wcp"] = cand; scores["wcp"] = charge
        nd = (~dead).sum(axis=1)
        ss3 = (Q * Q * ~zero).sum(axis=1)
        scores["rank3"] = np.where(nd > 0, np.sqrt(ss3 / np.maximum(nd, 1)), 0.0); eligs["rank3"] = cand
        live3 = cand & (nzero == 0)
        t3 = cKDTree(P[live3]) if live3.any() else None
        # the C++ form of prefer3 / nearby tests "a plane at charge 0" (no dead-registry lookup); a dead plane is a
        # zero plane too.  deadonly keeps the dead-aware blank definition (its whole point).
        hasblank = zero.any(axis=1)
        for rr in (1.0, 1.5, 2.0):
            e = cand.copy()
            if t3 is not None:
                bc = np.where(cand & hasblank)[0]
                if len(bc):
                    cnt = t3.query_ball_point(P[bc], rr, return_length=True)
                    e[bc[cnt > 0]] = False
            eligs[f"nearby{rr}"] = e; scores[f"nearby{rr}"] = charge
        eligs["deadonly"] = cand & ~blank.any(axis=1); scores["deadonly"] = charge
        eligs["rank3+nearby1.5"] = eligs["nearby1.5"]; scores["rank3+nearby1.5"] = scores["rank3"]
        # prefer3: blob-scoped -- a candidate with a blank plane is eligible only in a blob that holds no 3live
        # candidate; a blob whose candidates are all blank keeps them all (gap jumps kept, coverage unchanged)
        e = cand.copy()
        for bi_ in np.unique(BI[cand & (nzero == 0)]):
            sel_ = np.where((BI == bi_) & cand & hasblank)[0]
            e[sel_] = False
        eligs["prefer3"] = e; scores["prefer3"] = charge
        eligs["prefer3+nearby1.0"] = e & eligs["nearby1.0"]; scores["prefer3+nearby1.0"] = charge
        eligs["prefer3+nearby1.5"] = e & eligs["nearby1.5"]; scores["prefer3+nearby1.5"] = charge
        # per blob replay
        order = np.argsort(BI, kind="stable")
        bsorted = BI[order]
        starts = np.r_[0, np.flatnonzero(np.diff(bsorted)) + 1, len(bsorted)]
        peaks = {r: [] for r in RULES}
        blob_of = {}
        blob_cands = {}
        for k in range(len(starts) - 1):
            pts = order[starts[k]:starts[k + 1]]
            bi = int(bsorted[starts[k]])
            b = binfo.get(bi)
            if b is None:
                acc["points_no_blob"] += len(pts); continue
            mt, nt, mi, ni = int(b[10]), int(b[11]), int(b[12]), int(b[13])
            allc = pts[cand[pts]]
            if len(allc) == 0:
                continue
            acc["blobs_cand"] += 1
            blob_cands[bi] = allc
            adj_all = blob_adjacency(allc, W, mt, nt, mi, ni)
            for r in RULES:
                sel = allc[eligs[r][allc]]
                if len(sel) == 0:
                    continue
                if len(sel) == len(allc):
                    adj = adj_all
                else:
                    ss = set(sel.tolist())
                    adj = [(i, j) for i, j in adj_all if i in ss and j in ss]
                pk = peaks_in_blob(sel, scores[r], adj)
                peaks[r].extend(pk)
                if r == "wcp":
                    for i in pk:
                        blob_of[i] = bi
        # ---- replay check against the dumped terminals
        dumped = set(sb.V_old[sb.V_term & ~sb.V_ext].tolist())
        base = set(peaks["wcp"])
        acc["term_dumped"] += len(dumped); acc["term_dumped_in_replay"] += len(dumped & base)
        acc["replay_peaks"] += len(base); acc["replay_peaks_dumped"] += len(base & dumped)
        acc["replay_per_blob"] += len(base); acc["replay_blobs"] += acc["blobs_cand"] * 0  # placeholder
        # ---- attribution of dumped terminals
        onr3 = np.where(cand & (nzero == 0) & onr)[0]
        t_onr3 = cKDTree(P[onr3]) if len(onr3) else None
        for i in sorted(dumped):
            bi = int(BI[i])
            c = int(cls[i])
            off = not onr[i]
            key = ("term", CLASSES[c], "off" if off else "on")
            acc[key] += 1
            attr = "-"
            if off:
                bc = blob_cands.get(bi)
                in_blob = bool(bc is not None and np.any(onr[bc] & (nzero[bc] == 0)))
                near3 = bool(t_onr3 is not None and t_onr3.query_ball_point(P[i], R_NEAR, return_length=True) > 0)
                attr = "ranking" if in_blob else ("ghost_blob" if near3 else "isolated")
                acc[("attr", CLASSES[c], attr)] += 1
                if c == 0:
                    acc[("off3live", "img<=1" if dimg[i] <= THR else "img>1")] += 1
                if c == 1:
                    bp = int(np.where(blank[i])[0][0])
                    acc[("blank_unc", "sentinel" if sent[i, bp] else ("zero_unc" if U[i, bp] == 0 else "other"))] += 1
            else:
                if c == 1:
                    bp = int(np.where(blank[i])[0][0])
                    acc[("blank_unc_on", "sentinel" if sent[i, bp] else ("zero_unc" if U[i, bp] == 0 else "other"))] += 1
            term_rows.append((pre, cl, i, bi, f"{P[i,0]:.2f}", f"{P[i,1]:.2f}", f"{P[i,2]:.2f}", CLASSES[c],
                              "".join("1" if not zero[i, p] else ("D" if dead[i, p] else "0") for p in range(3)),
                              f"{voff[i]:.2f}", f"{dimg[i]:.2f}", f"{charge[i]:.0f}", attr, int(i in base)))
        # ---- rule metrics vs the baseline replay
        base_arr = np.array(sorted(base), int)
        for r in RULES:
            pk = np.array(sorted(set(peaks[r])), int)
            acc[("rule", r, "peaks")] += len(pk)
            acc[("rule", r, "off")] += int((~onr[pk]).sum()) if len(pk) else 0
            acc[("rule", r, "on")] += int(onr[pk].sum()) if len(pk) else 0
            acc[("rule", r, "off_1blank")] += int(((~onr[pk]) & (cls[pk] == 1)).sum()) if len(pk) else 0
            removed = np.setdiff1d(base_arr, pk)
            added = np.setdiff1d(pk, base_arr)
            acc[("rule", r, "removed")] += len(removed); acc[("rule", r, "added")] += len(added)
            acc[("rule", r, "removed_on")] += int(onr[removed].sum()) if len(removed) else 0
            acc[("rule", r, "removed_off")] += int((~onr[removed]).sum()) if len(removed) else 0
            if len(removed) and len(pk):
                tp = cKDTree(P[pk])
                dd, _ = tp.query(P[removed])
                acc[("rule", r, "removed_on_norep")] += int((onr[removed] & (dd > R_REP)).sum())
                acc[("rule", r, "removed_off_norep")] += int((~onr[removed] & (dd > R_REP)).sum())
            elif len(removed):
                acc[("rule", r, "removed_on_norep")] += int(onr[removed].sum())
                acc[("rule", r, "removed_off_norep")] += int((~onr[removed]).sum())
            # GAP stretches and largest peak-free run per record
            for ps, F, _, gidx in rl:
                for (ra, rb) in gap_rows.get((cl, ps), []):
                    la, lb = int(np.searchsorted(gidx, ra)), int(np.searchsorted(gidx, rb))
                    if la >= len(gidx) or gidx[la] != ra or lb >= len(gidx) or gidx[lb] != rb:
                        acc["gap_rows_unmapped"] += 1
                        continue
                    seg = F[la:lb + 1]
                    if len(seg) == 0:
                        continue
                    # amendment 3 (Q3'): coverage ALONG the stretch -- the largest run of the anchor-to-anchor
                    # polyline without a peak within R_RUN, base vs rule, robust to a peak moving sideways
                    segA = F[max(la - 1, 0):min(lb + 1, len(F) - 1) + 1]
                    if len(segA) > 1:
                        arcA = np.r_[0, np.cumsum(np.linalg.norm(np.diff(segA, axis=0), axis=1))]
                        tA = cKDTree(segA)
                        def lrun(idx_):
                            if len(idx_) == 0:
                                return float(arcA[-1])
                            dq, jq = tA.query(P[idx_])
                            on_ = dq <= R_RUN
                            if not on_.any():
                                return float(arcA[-1])
                            s_ = np.sort(np.r_[0, arcA[jq[on_]], arcA[-1]])
                            return float(np.max(np.diff(s_)))
                        g0, g1 = lrun(base_arr), lrun(pk)
                        acc[("rule", r, "gap_run_base")] += g0; acc[("rule", r, "gap_run")] += g1
                        acc[("rule", r, "gap_n")] += 1
                        acc[("rule", r, "gap_grow2")] += int(g1 - g0 > 2.0)
                    ts = cKDTree(seg)
                    if len(base_arr):
                        db, _ = ts.query(P[base_arr]); nb = int((db <= R_GAP).sum())
                    else:
                        nb = 0
                    if len(pk):
                        dk, _ = ts.query(P[pk]); nk = int((dk <= R_GAP).sum())
                    else:
                        nk = 0
                    acc[("rule", r, "gap_peaks_base")] += nb; acc[("rule", r, "gap_peaks")] += nk
                    if r == "wcp":
                        acc["gap_stretches"] += 1
                if len(F) > 1 and len(pk):
                    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(F, axis=0), axis=1))]
                    Qp, _ = A.closest_on_polyline(F, P[pk])
                    dpk = np.linalg.norm(Qp - P[pk], axis=1)
                    on = dpk <= R_RUN
                    if on.any():
                        tf = cKDTree(F)
                        _, jf = tf.query(Qp[on])
                        s = np.sort(np.r_[0, arc[jf], arc[-1]])
                        gap = float(np.max(np.diff(s)))
                    else:
                        gap = float(arc[-1])
                    runs[r].append(gap)
        # ---- spots
        for sname, (sdet, sev, scl, click, win) in SPOTS.items():
            if sdet != det or sev != pre or scl != cl:
                continue
            inw = np.linalg.norm(P - click, axis=1) < win
            for r in RULES:
                pk = np.array(sorted(set(peaks[r])), int)
                pw = pk[inw[pk]] if len(pk) else pk
                acc[("spot", sname, r, "on")] += int(onr[pw].sum()) if len(pw) else 0
                acc[("spot", sname, r, "off")] += int((~onr[pw]).sum()) if len(pw) else 0
                acc[("spot", sname, r, "off_1blank")] += int(((~onr[pw]) & (cls[pw] == 1)).sum()) if len(pw) else 0
            dw = np.array(sorted(dumped), int); dw = dw[inw[dw]]
            acc[("spot", sname, "dumped", "on")] += int(onr[dw].sum()); acc[("spot", sname, "dumped", "off")] += int((~onr[dw]).sum())
    return pre, acc, term_rows, runs


def pct(a, b):
    return f"{100.0 * a / b:.1f} %" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--logd"); ap.add_argument("--events"); ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--cut", type=float, default=500.0)
    ap.add_argument("--stretches")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logd = a.logd or f"/home/xqian/tmp/d114/arm_{a.arm}"
    evs = a.events.split(",") if a.events else sorted(os.path.basename(p)[:-len(f"_{a.arm}")]
                                                      for p in glob.glob(f"{C.IMG}/{a.det}/work/*_{a.arm}"))
    stretches = []
    if a.stretches:
        with open(a.stretches) as fh:
            stretches = [r for r in csv.DictReader(fh, delimiter="\t") if r["det"] == a.det]
    with Pool(a.jobs) as pool:
        res = pool.map(one, [(a.det, a.arm, e, logd, a.cut, stretches) for e in evs], chunksize=1)
    acc = collections.Counter(); rows = []; runs = collections.defaultdict(list)
    for pre, ac, r, ru in res:
        acc.update(ac); rows.extend(r)
        for k, v in ru.items():
            runs[k].extend(v)
    with gzip.open(f"{a.out}_terminals.tsv.gz", "wt") as fh:
        fh.write("event cluster point blob x y z class planes_uvw d_ridge d_img charge attribution in_replay\n".replace(" ", "\t"))
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    L = [f"# doc pdvd/114 terminal rule replay: det={a.det} arm={a.arm} cut={a.cut:.0f} events={acc['events']} (missing "
         f"{acc['events_missing']}); clusters with a record and a dump {acc['clusters']} (no dump {acc['clusters_no_dump']}, "
         f"no image {acc['clusters_no_image']}); candidate-bearing blobs {acc['blobs_cand']}; GAP stretches seen {acc['gap_stretches']}",
         f"# R_NEAR {R_NEAR} cm, R_REP {R_REP}, R_GAP {R_GAP}, R_RUN {R_RUN}, THR {THR}", ""]
    L.append("## replay check")
    L.append(f"- dumped non-extreme terminals {acc['term_dumped']}; of them replayed Phase-1 peaks {acc['term_dumped_in_replay']} "
             f"({pct(acc['term_dumped_in_replay'], acc['term_dumped'])})")
    L.append(f"- replayed Phase-1 peaks {acc['replay_peaks']}; of them dumped terminals {acc['replay_peaks_dumped']} "
             f"({pct(acc['replay_peaks_dumped'], acc['replay_peaks'])}); the rest were removed by Phases 2 / 3 / 3b or the "
             f"replay differs; peaks per candidate-bearing blob {acc['replay_peaks'] / max(acc['blobs_cand'], 1):.3f}")
    L.append("")
    L.append("## dumped terminals by class and side of the ridge (THR 1 cm), and the attribution of the off-ridge ones")
    L.append("| class | on ridge | off ridge | off share | ranking (an on-ridge 3live candidate in the same blob lost) | ghost blob (none in the blob, one within R_NEAR) | isolated (none within R_NEAR) |")
    L.append("|---|---|---|---|---|---|---|")
    for c in CLASSES:
        on = acc[("term", c, "on")]; off = acc[("term", c, "off")]
        if on + off == 0:
            continue
        L.append(f"| {c} | {on} | {off} | {pct(off, on + off)} | {acc[('attr', c, 'ranking')]} ({pct(acc[('attr', c, 'ranking')], off)}) | "
                 f"{acc[('attr', c, 'ghost_blob')]} ({pct(acc[('attr', c, 'ghost_blob')], off)}) | {acc[('attr', c, 'isolated')]} "
                 f"({pct(acc[('attr', c, 'isolated')], off)}) |")
    L.append(f"- 3live off-ridge terminals: within 1 cm of an image point {acc[('off3live', 'img<=1')]}, farther {acc[('off3live', 'img>1')]}")
    L.append(f"- 1blank off-ridge terminals, the blank plane's uncertainty: sentinel (1e12, painted or dead) {acc[('blank_unc', 'sentinel')]}, "
             f"zero (channel absent from the activity) {acc[('blank_unc', 'zero_unc')]}, other {acc[('blank_unc', 'other')]}; "
             f"on-ridge 1blank: sentinel {acc[('blank_unc_on', 'sentinel')]}, zero {acc[('blank_unc_on', 'zero_unc')]}, other {acc[('blank_unc_on', 'other')]}")
    L.append("")
    L.append("## rules, replayed per blob (baseline = the wcp replay)")
    L.append("| rule | peaks | on ridge | off ridge | off 1blank | removed on / no replacement within R_REP | removed off / no replacement | added | peaks in GAP stretches (base -> rule) | GAP stretches: mean largest peak-free run base -> rule (cm), grown > 2 cm | largest peak-free run per record p50 / p90 (cm) | records whose run grows > 2 cm |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    base_runs = np.array(runs["wcp"]) if runs["wcp"] else np.zeros(0)
    for r in RULES:
        rr = np.array(runs[r]) if runs[r] else np.zeros(0)
        grow = int((rr - base_runs > 2.0).sum()) if len(rr) == len(base_runs) and len(rr) else -1
        L.append(f"| {r} | {acc[('rule', r, 'peaks')]} | {acc[('rule', r, 'on')]} | {acc[('rule', r, 'off')]} | {acc[('rule', r, 'off_1blank')]} | "
                 f"{acc[('rule', r, 'removed_on')]} / {acc[('rule', r, 'removed_on_norep')]} | {acc[('rule', r, 'removed_off')]} / "
                 f"{acc[('rule', r, 'removed_off_norep')]} | {acc[('rule', r, 'added')]} | {acc[('rule', r, 'gap_peaks_base')]} -> "
                 f"{acc[('rule', r, 'gap_peaks')]} | "
                 f"{acc[('rule', r, 'gap_run_base')] / max(acc[('rule', r, 'gap_n')], 1):.2f} -> "
                 f"{acc[('rule', r, 'gap_run')] / max(acc[('rule', r, 'gap_n')], 1):.2f}, {acc[('rule', r, 'gap_grow2')]} of {acc[('rule', r, 'gap_n')]} | "
                 f"{np.percentile(rr, 50):.2f} / {np.percentile(rr, 90):.2f} | {grow} |"
                 if len(rr) else f"| {r} | {acc[('rule', r, 'peaks')]} | | | | | | | | | n/a | |")
    L.append("")
    L.append("## the owner's spot (window terminals on / off the ridge; off 1blank)")
    for sname in SPOTS:
        if any(k[0] == "spot" and k[1] == sname for k in acc):
            L.append(f"- {sname}: dumped {acc[('spot', sname, 'dumped', 'on')]} / {acc[('spot', sname, 'dumped', 'off')]}; " + "; ".join(
                f"{r} {acc[('spot', sname, r, 'on')]} / {acc[('spot', sname, r, 'off')]} ({acc[('spot', sname, r, 'off_1blank')]})" for r in RULES))
    open(f"{a.out}.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
