#!/usr/bin/env python3
"""doc pdvd/111 Phase C -- trajectory-quality metrics of a knob arm against its baseline arm (same pctrees, same pin),
on the common set of persisted STM-fit records (event, cluster, pass) present in both arms.  Read-only.

Metrics (definitions frozen in figs/111_pred.txt):
  P1  share of rows whose ridge offset d > 1 cm (d111_stage_attrib.Ridge: charge-weighted local image axis, 3 cm);
  P2  share of rows off-charge (no measured cell of the cluster within +-1 wire on the row's own slice) in >= 1 plane;
  P3  Bee-visible holes (runs of >= 3 consecutive rows with q < 0, the Bee client cut, doc 110) per 10 m of fit;
  P4  image coverage: image charge within 1.5 cm of the fit polyline, over the image charge within 1.5 cm of EITHER
      arm's fit of that record (one denominator for both arms, so a shortcut or a truncation shows as a loss);
  P5  the owner's spots: ridge-offset max within the spot window (h1, h2 on PDHD 029107_16; v1-v3 on PDVD 039349_20);
  P6' clean-record no-regression: in records whose base arm has no row > 1 cm, the arm's share of rows > 1 cm;
  R   median over common events of the per-event PR wall (pr_resource_*.txt) arm / base;
  also: rows > 2 cm, median dQ/dx (ke/cm), rows with chord wiggle > 1 cm, record/row counts, records only in one arm.

Usage:
  d111_eval.py --det pdhd --base d111hoff --arm d111hsr10 [--out figs/111_eval_pdhd_sr10.txt]
"""
import argparse, collections, glob, json, os, sys, zipfile
import numpy as np
import uproot
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d111_stage_attrib import Ridge, closest_on_polyline, bee_layer, onq_pm1, BASE  # noqa: E402

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
SPOTS = {
    "pdhd": [("h1", "029107_16", 108, (75.3, 438.3, 450.1), 12.0), ("h2", "029107_16", 106, (182.0, 537.4, 401.8), 24.0)],
    "pdvd": [("v1", "039349_20", 80, (313.7, 95.9, 124.9), 12.0), ("v2", "039349_20", 80, (94.5, 30.7, 165.0), 12.0),
             ("v3", "039349_20", 25, (-270.2, -243.7, 36.0), 12.0)],
}


def records(d):
    t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(library="np")
    out = {}
    for cl, ps in sorted(set(zip(t["cluster_id"].tolist(), t["pass"].tolist()))):
        m = (t["cluster_id"] == cl) & (t["pass"] == ps)
        out[(cl, ps)] = {k: t[k][m] for k in ("x", "y", "z", "q", "pu", "pv", "pw", "pt")}
    return out, t


def cells_of(d, det):
    pd = uproot.open(f"{d}/tracking-stm.root")["T_proj_data"].arrays(library="np")
    cells = collections.defaultdict(lambda: [set(), set(), set()])
    for cid, ch, ts, q in zip(pd["cluster_id"][0], pd["channel"][0], pd["time_slice"][0], pd["charge"][0]):
        ch = np.asarray(ch).astype(int); ts = np.asarray(ts).astype(int); m = np.asarray(q) > 0
        pl = np.searchsorted(np.array(BASE[det][1:]), ch, side="right")
        for p in range(3):
            mm = m & (pl == p)
            cells[int(cid) // 10][p].update(zip(ch[mm].tolist(), ts[mm].tolist()))
    return cells


def wall(d):
    for p in glob.glob(f"{d}/pr_resource_*.txt"):
        for tok in open(p).read().split():
            if tok.startswith("wall_s="):
                return float(tok.split("=")[1])
    return None


def holes(q, minrows=3):
    n, i, k = 0, 0, len(q)
    while i < k:
        if q[i] < 0:
            j = i
            while j < k and q[j] < 0:
                j += 1
            n += (j - i) >= minrows; i = j
        else:
            i += 1
    return n


def wiggle(F, k=3):
    out = np.zeros(len(F))
    for i in range(k, len(F) - k):
        a, b = F[i - k], F[i + k]; nn = np.linalg.norm(b - a)
        if nn > 0:
            u = (b - a) / nn; v = F[i] - a; out[i] = np.linalg.norm(v - (v @ u) * u)
    return out


def dist_to_polyline(P, X):
    if len(P) == 0 or len(X) == 0:
        return np.full(len(X), np.inf)
    Q, _ = closest_on_polyline(P, X)
    return np.linalg.norm(Q - X, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    evA = {os.path.basename(p)[:-len(a.base) - 1]: p for p in glob.glob(f"{IMG}/{a.det}/work/*_{a.base}")}
    evB = {os.path.basename(p)[:-len(a.arm) - 1]: p for p in glob.glob(f"{IMG}/{a.det}/work/*_{a.arm}")}
    common = sorted(e for e in set(evA) & set(evB)
                    if os.path.exists(f"{evA[e]}/tracking-stm.root") and os.path.exists(f"{evB[e]}/tracking-stm.root"))
    S = {k: collections.Counter() for k in ("A", "B")}
    walls = []
    dq = {"A": [], "B": []}
    only = collections.Counter()
    spot_rows = []
    for ev in common:
        RA, _ = records(evA[ev]); RB, _ = records(evB[ev])
        CA, CB = cells_of(evA[ev], a.det), cells_of(evB[ev], a.det)
        img = bee_layer(f"{evA[ev]}/mabc-pr.zip", "clustering-global")
        only["A"] += len(set(RA) - set(RB)); only["B"] += len(set(RB) - set(RA))
        ridges = {}
        wa, wb = wall(evA[ev]), wall(evB[ev])
        if wa and wb:
            walls.append(wb / wa)
        for key in sorted(set(RA) & set(RB)):
            cl = key[0]
            if cl not in ridges:
                mi = img[1] == cl
                ridges[cl] = (Ridge(img[0][mi], img[2][mi]), img[0][mi], np.clip(img[2][mi], 1, None))
            R, Pimg, Wimg = ridges[cl]
            if not R.ok:
                continue
            FA = np.c_[RA[key]["x"], RA[key]["y"], RA[key]["z"]]; FB = np.c_[RB[key]["x"], RB[key]["y"], RB[key]["z"]]
            near_either = (dist_to_polyline(FA, Pimg) < 1.5) | (dist_to_polyline(FB, Pimg) < 1.5)
            clean = bool(np.all(R.offset(FA) <= 1))
            den = Wimg[near_either].sum()
            for lab, rec, F, C in (("A", RA[key], FA, CA), ("B", RB[key], FB, CB)):
                s = S[lab]
                n = len(F)
                d = R.offset(F)
                step = np.r_[np.linalg.norm(np.diff(F, axis=0), axis=1), 0]
                off = np.zeros(n, int)
                for p, k in enumerate(("pu", "pv", "pw")):
                    off += (~onq_pm1(C[cl][p], rec[k], rec["pt"])).astype(int)
                s["records"] += 1; s["rows"] += n; s["len_cm"] += step.sum()
                s["dev1"] += int((d > 1).sum()); s["dev2"] += int((d > 2).sum())
                if clean:
                    s["clean_rows"] += n; s["clean_dev1"] += int((d > 1).sum())
                s["off1"] += int((off >= 1).sum()); s["qneg"] += int((rec["q"] < 0).sum())
                s["holes"] += holes(rec["q"]); s["wig1"] += int((wiggle(F) > 1).sum())
                cov = Wimg[near_either & (dist_to_polyline(F, Pimg) < 1.5)].sum()
                s["cov_num"] += cov; s["cov_den"] += den
                dx = np.where(step > 0, step, np.nan)
                dq[lab].append((rec["q"] + 1000) * 10 / 1000.0 / np.r_[dx[:-1], np.nan][:n] if n > 1 else np.array([]))
        for name, sev, scl, click, R_ in SPOTS.get(a.det, []):
            if sev != ev:
                continue
            click = np.array(click)
            row = [name]
            for lab, RR in (("A", RA), ("B", RB)):
                mi = img[1] == scl
                Rg = Ridge(img[0][mi], img[2][mi])
                best = None
                for (cl, ps), rec in RR.items():
                    if cl != scl:
                        continue
                    F = np.c_[rec["x"], rec["y"], rec["z"]]
                    w = np.linalg.norm(F - click, axis=1) < R_
                    if w.sum() == 0:
                        continue
                    dd = Rg.offset(F[w]); qq = rec["q"][w]
                    val = (float(dd.max()), float(np.percentile(dd, 90)), int((qq < 0).sum()), int(w.sum()), ps)
                    best = val if best is None or val[0] > best[0] else best
                row.append(best)
            spot_rows.append(row)
    L = []
    P = L.append
    P(f"# doc pdvd/111 Phase C eval ({a.det}): base A={a.base}  arm B={a.arm}; common events {len(common)}")
    P(f"records only in A {only['A']}, only in B {only['B']} (tag churn; excluded)")
    P("")
    P("| metric | A (base) | B (arm) | B/A - 1 |")
    P("|---|---|---|---|")
    def rel(x, y):
        return f"{100*(y/x-1):+.1f} %" if x else "n/a"
    A_, B_ = S["A"], S["B"]
    P(f"| common records / rows / length (m) | {A_['records']} / {A_['rows']} / {A_['len_cm']/100:.1f} | {B_['records']} / {B_['rows']} / {B_['len_cm']/100:.1f} | |")
    for lab, key, den in (("P1 rows ridge offset > 1 cm", "dev1", "rows"), ("rows ridge offset > 2 cm", "dev2", "rows"),
                          ("P2 rows off-charge (>= 1 plane, +-1 wire)", "off1", "rows"), ("rows q < 0 (Bee-dropped)", "qneg", "rows"),
                          ("rows chord wiggle > 1 cm", "wig1", "rows")):
        fa, fb = A_[key] / max(A_[den], 1), B_[key] / max(B_[den], 1)
        P(f"| {lab} | {100*fa:.2f} % ({A_[key]}) | {100*fb:.2f} % ({B_[key]}) | {rel(fa, fb)} |")
    ha, hb = A_["holes"] / max(A_["len_cm"] / 1000, 1e-9), B_["holes"] / max(B_["len_cm"] / 1000, 1e-9)
    P(f"| P3 Bee holes (>= 3 rows q<0) per 10 m | {ha:.2f} ({A_['holes']}) | {hb:.2f} ({B_['holes']}) | {rel(ha, hb)} |")
    ca, cb = A_["cov_num"] / max(A_["cov_den"], 1), B_["cov_num"] / max(B_["cov_den"], 1)
    P(f"| P4 image coverage (charge within 1.5 cm) | {100*ca:.2f} % | {100*cb:.2f} % | {100*(cb-ca):+.2f} pt |")
    P(f"| P6' clean records (base max offset <= 1 cm): rows > 1 cm | {100*A_['clean_dev1']/max(A_['clean_rows'],1):.2f} % ({A_['clean_rows']} rows) | {100*B_['clean_dev1']/max(B_['clean_rows'],1):.2f} % ({B_['clean_rows']} rows) | |")
    P(f"| R median per-event wall ratio B/A | | {np.median(walls) if walls else float('nan'):.3f} ({len(walls)} events) | |")
    for lab in ("A", "B"):
        arr = np.concatenate([x for x in dq[lab] if len(x)]) if dq[lab] else np.array([np.nan])
        S[lab]["dqdx_med"] = np.nanmedian(arr)
    P(f"| dQ/dx median (ke/cm, row dQ over its step) | {S['A']['dqdx_med']:.1f} | {S['B']['dqdx_med']:.1f} | {rel(S['A']['dqdx_med'], S['B']['dqdx_med'])} |")
    if spot_rows:
        P("")
        P("## P5 spots: ridge-offset max / p90 (cm), q<0 rows, rows in window, pass")
        P("| spot | A | B |")
        P("|---|---|---|")
        for name, va, vb in spot_rows:
            f = lambda v: "n/a" if v is None else f"{v[0]:.2f} / {v[1]:.2f}, q<0 {v[2]}, n {v[3]}, pass {v[4]}"
            P(f"| {name} | {f(va)} | {f(vb)} |")
    txt = "\n".join(L) + "\n"
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
