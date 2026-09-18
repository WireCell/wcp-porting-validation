#!/usr/bin/env python3
"""doc pdvd/115 -- the 2-D measurement against the projected 3-D trajectory, per fitted row and wire plane, for a
knob arm against its base arm on the common persisted STM-fit records (event, cluster, pass).  Read-only.

The fit writes every row's projection into each plane as a FRACTIONAL global wire rank (pu, pv, pw; deliberately
not truncated, PdvdMagnifyTrackingVisitor.cxx) and a drift time slice (pt); the measured 2-D cells of the cluster
are integer (channel rank, time slice) with a charge (T_proj_data).  Every earlier row-side test (onq_pm1, P2)
rounds first and returns a boolean.  This script keeps the fraction: per row and plane,

    d_exact  = w - (channel of the cluster's nearest charged cell ON the row's rounded slice)   [wire units, signed]
    d_pm1    = the same over the slices s-1, s, s+1 (the smallest |d|)
    noslice  = the cluster has no charged cell on that slice in that plane (d_exact undefined)

A plane is DEAD at the row when the nearest dumped Steiner vertex within DEAD_R carries that plane's dead bit
(the doc-114 census recipe; the dump of the same arm is read); dead planes are excluded from every share.

W is the clean plane on both detectors (pt and pw close exactly, doc pdhd/24); U and V carry a known ~1-wire
systematic on PDHD (wrapped planes) and a rounding offset on PDVD (doc 110), so their shares are stated with it.

Reported per arm on the common records, and paired per record (arm - base):
    per plane: live rows, noslice share, median |d_exact|, share |d_exact| > 1 wire, > 2 wires, the +-1-slice
               versions, and OFF = noslice or |d_exact| > 1 (the metric-free "not on the measured charge");
    rows OFF in >= 1 live plane (exact, and the +-1-slice / +-1-wire tolerant form == the doc-111 P2);
    R2D = W-plane share of live rows with |d_exact| > 1 wire (the doc-115 clause);
    paired: records where the arm's W-plane |d| > 1 share is lower / higher / equal (+-0.005), row-weighted mean
    difference.

Usage:
  d115_proj_resid.py --det pdhd --base d115hoff --arm d115hbwa [--jobs 8] [--events a,b] --out figs/115_resid_pdhd_bwa
Writes <out>.txt, <out>.json (the numbers the verdict reads) and <out>_rows.tsv.gz.
"""
import argparse, collections, glob, gzip, json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import uproot
from multiprocessing import Pool
from scipy.spatial import cKDTree
import d111s_common as C
A = C.A

DEAD_R = 1.5
PLANES = "UVW"
KEYS = ("pu", "pv", "pw")


def logd_for(arm):
    for root in ("/home/xqian/tmp/d115", "/home/xqian/tmp/d114", "/home/xqian/tmp/d113"):
        d = f"{root}/arm_{arm}"
        if os.path.isdir(d):
            return d
    return f"/home/xqian/tmp/d115/arm_{arm}"


def load_cells(f, det):
    """cluster -> plane -> slice -> sorted channel array (charged cells only)."""
    pd = f["T_proj_data"].arrays(library="np")
    cells = collections.defaultdict(lambda: [collections.defaultdict(list) for _ in range(3)])
    for cid, ch, ts, q in zip(pd["cluster_id"][0], pd["channel"][0], pd["time_slice"][0], pd["charge"][0]):
        ch = np.asarray(ch).astype(int); ts = np.asarray(ts).astype(int); m = np.asarray(q) > 0
        pl = np.searchsorted(np.array(A.BASE[det][1:]), ch, side="right")
        cl = int(cid) // 10
        for p in range(3):
            mm = m & (pl == p)
            for c, s in zip(ch[mm].tolist(), ts[mm].tolist()):
                cells[cl][p][s].append(c)
    out = {}
    for cl, planes in cells.items():
        out[cl] = [{s: np.array(sorted(set(v))) for s, v in pl.items()} for pl in planes]
    return out


def nearest_d(chans, w):
    if chans is None or len(chans) == 0:
        return np.nan
    k = np.searchsorted(chans, w)
    best = np.inf
    for j in (k - 1, k):
        if 0 <= j < len(chans):
            d = w - chans[j]
            if abs(d) < abs(best):
                best = d
    return best


def dead_bits(tr, cl):
    """(kd-tree of the last dumped Steiner vertices of cluster cl, their dead mask (n,3)) or None."""
    best = None
    for sb in tr["stg"]:
        if sb.ident == cl and sb.nv and (best is None or sb.order > best.order):
            best = sb
    if best is None:
        return None
    m = best.V_m61
    dead = np.c_[[(m >= 0) & (((m >> (3 + p)) & 1) > 0) for p in range(3)]].T
    return cKDTree(best.V), dead


def rows_of(det, work, logd, pre, keys):
    """per record key -> per row per plane (d_exact, d_pm1, dead); None when the arm lacks the event."""
    d = f"{work}/{pre}"
    lg = f"{logd}/evt_{pre}.log.gz"
    if not os.path.exists(f"{d}/tracking-stm.root"):
        return None
    f = uproot.open(f"{d}/tracking-stm.root")
    t = f["T_rec_charge"].arrays(library="np")
    cells = load_cells(f, det)
    tr = C.parse_trace(lg) if os.path.exists(lg) else dict(stg=[])
    out = {}
    dcache = {}
    for (cl, ps) in keys:
        m = (t["cluster_id"] == cl) & (t["pass"] == ps)
        if not m.any():
            continue
        n = int(m.sum())
        F = np.c_[t["x"][m], t["y"][m], t["z"][m]]
        q = t["q"][m]
        s0 = np.round(t["pt"][m]).astype(int)
        de = np.full((n, 3), np.nan); dp = np.full((n, 3), np.nan)
        pl = cells.get(cl, [{}, {}, {}])
        for p, key in enumerate(KEYS):
            w = t[key][m]
            for i in range(n):
                de[i, p] = nearest_d(pl[p].get(s0[i]), w[i])
                best = np.inf
                for ds in (-1, 0, 1):
                    dd = nearest_d(pl[p].get(s0[i] + ds), w[i])
                    if np.isfinite(dd) and abs(dd) < abs(best):
                        best = dd
                dp[i, p] = best if np.isfinite(best) else np.nan
        dead = np.zeros((n, 3), bool)
        if cl not in dcache:
            dcache[cl] = dead_bits(tr, cl)
        if dcache[cl] is not None:
            vtree, dm = dcache[cl]
            dv, jv = vtree.query(F)
            near = dv <= DEAD_R
            dead[near] = dm[jv[near]]
        out[(cl, ps)] = dict(de=de, dp=dp, dead=dead, q=q, n=n)
    return out


def one(args):
    det, base, arm, logb, loga, pre = args
    wb = f"{C.IMG}/{det}/work"
    kb = f"{pre}_{base}"; ka = f"{pre}_{arm}"
    try:
        tb = uproot.open(f"{wb}/{kb}/tracking-stm.root")["T_rec_charge"].arrays(["cluster_id", "pass"], library="np")
        ta = uproot.open(f"{wb}/{ka}/tracking-stm.root")["T_rec_charge"].arrays(["cluster_id", "pass"], library="np")
    except Exception:
        return pre, None
    keys = sorted(set(zip(tb["cluster_id"].tolist(), tb["pass"].tolist())) & set(zip(ta["cluster_id"].tolist(), ta["pass"].tolist())))
    RB = rows_of(det, wb, logb, kb, keys); RA = rows_of(det, wb, loga, ka, keys)
    if RB is None or RA is None:
        return pre, None
    rows = []
    recs = []
    for k in keys:
        if k not in RB or k not in RA:
            continue
        rec = {"key": k}
        for lab, R in (("A", RB[k]), ("B", RA[k])):
            live = ~R["dead"]
            de, dp = R["de"], R["dp"]
            per = {}
            for p in range(3):
                L = live[:, p]
                nl = int(L.sum())
                ns = int((L & ~np.isfinite(de[:, p])).sum())
                nsp = int((L & ~np.isfinite(dp[:, p])).sum())
                ad = np.abs(de[L, p]); adp = np.abs(dp[L, p])
                per[p] = dict(live=nl, noslice=ns, noslice_pm1=nsp,
                              absd=ad[np.isfinite(ad)].tolist(),
                              gt1=int((ad > 1.0).sum()), gt2=int((ad > 2.0).sum()),
                              gt1_pm1=int((adp > 1.0).sum()), gt2_pm1=int((adp > 2.0).sum()),
                              off=int((ns + (ad > 1.0).sum())), off_pm1=int((nsp + (adp > 1.0).sum())))
            offrow = np.zeros(R["n"], bool); offrow_pm1 = np.zeros(R["n"], bool); offrow_pm1w = np.zeros(R["n"], bool)
            for p in range(3):
                L = live[:, p]
                offrow |= L & (~np.isfinite(de[:, p]) | (np.abs(de[:, p]) > 1.0))
                offrow_pm1 |= L & (~np.isfinite(dp[:, p]) | (np.abs(dp[:, p]) > 1.0))
                offrow_pm1w |= L & (~np.isfinite(dp[:, p]) | (np.abs(dp[:, p]) > 1.5))
            rec[lab] = dict(n=R["n"], per=per, offrow=int(offrow.sum()), offrow_pm1=int(offrow_pm1.sum()),
                            offrow_pm1w=int(offrow_pm1w.sum()), qneg=int((R["q"] < 0).sum()))
            for i in range(R["n"]):
                rows.append((pre, k[0], k[1], i, lab, *["%.3f" % de[i, p] if np.isfinite(de[i, p]) else "nan" for p in range(3)],
                             *["%.3f" % dp[i, p] if np.isfinite(dp[i, p]) else "nan" for p in range(3)],
                             "".join("1" if R["dead"][i, p] else "0" for p in range(3))))
        recs.append(rec)
    return pre, dict(recs=recs, rows=rows)


def agg(recs, lab):
    S = collections.Counter(); absd = {p: [] for p in range(3)}
    for rec in recs:
        r = rec[lab]
        S["records"] += 1; S["rows"] += r["n"]
        S["offrow"] += r["offrow"]; S["offrow_pm1"] += r["offrow_pm1"]; S["offrow_pm1w"] += r["offrow_pm1w"]; S["qneg"] += r["qneg"]
        for p in range(3):
            for kk in ("live", "noslice", "noslice_pm1", "gt1", "gt2", "gt1_pm1", "gt2_pm1", "off", "off_pm1"):
                S[f"{kk}_{p}"] += r["per"][p][kk]
            absd[p].extend(r["per"][p]["absd"])
    out = dict(records=S["records"], rows=S["rows"])
    sh = lambda a, b: (a / b) if b else float("nan")
    for p in range(3):
        pl = PLANES[p]
        out[f"live_{pl}"] = S[f"live_{p}"]
        out[f"noslice_{pl}"] = sh(S[f"noslice_{p}"], S[f"live_{p}"])
        out[f"med_absd_{pl}"] = float(np.median(absd[p])) if absd[p] else float("nan")
        out[f"gt1_{pl}"] = sh(S[f"gt1_{p}"], S[f"live_{p}"])
        out[f"gt2_{pl}"] = sh(S[f"gt2_{p}"], S[f"live_{p}"])
        out[f"gt1pm1_{pl}"] = sh(S[f"gt1_pm1_{p}"], S[f"live_{p}"])
        out[f"off_{pl}"] = sh(S[f"off_{p}"], S[f"live_{p}"])
        out[f"offpm1_{pl}"] = sh(S[f"off_pm1_{p}"], S[f"live_{p}"])
    out["offrow"] = sh(S["offrow"], S["rows"]); out["offrow_pm1"] = sh(S["offrow_pm1"], S["rows"])
    out["offrow_pm1w"] = sh(S["offrow_pm1w"], S["rows"]); out["qneg"] = sh(S["qneg"], S["rows"])
    out["R2D"] = out["gt1_W"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--logd-base"); ap.add_argument("--logd-arm")
    ap.add_argument("--events")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logb = a.logd_base or logd_for(a.base); loga = a.logd_arm or logd_for(a.arm)
    evB = {os.path.basename(p)[:-len(a.base) - 1] for p in glob.glob(f"{C.IMG}/{a.det}/work/*_{a.base}")}
    evA = {os.path.basename(p)[:-len(a.arm) - 1] for p in glob.glob(f"{C.IMG}/{a.det}/work/*_{a.arm}")}
    evs = sorted(evB & evA)
    if a.events:
        evs = [e for e in evs if e in set(a.events.split(","))]
    with Pool(a.jobs) as pool:
        res = pool.map(one, [(a.det, a.base, a.arm, logb, loga, pre) for pre in evs])
    recs, rows, missing = [], [], 0
    for pre, r in res:
        if r is None:
            missing += 1; continue
        recs.extend(r["recs"]); rows.extend(r["rows"])
    SA, SB = agg(recs, "A"), agg(recs, "B")
    # paired per record: W-plane share |d| > 1 (live rows)
    better = worse = same = 0; wsum = 0.0; wn = 0
    for rec in recs:
        pa, pb = rec["A"]["per"][2], rec["B"]["per"][2]
        if pa["live"] == 0 or pb["live"] == 0:
            continue
        fa, fb = pa["gt1"] / pa["live"], pb["gt1"] / pb["live"]
        if fb < fa - 0.005: better += 1
        elif fb > fa + 0.005: worse += 1
        else: same += 1
        wsum += (fb - fa) * pa["live"]; wn += pa["live"]
    paired = dict(better=better, worse=worse, same=same, mean_diff=(wsum / wn) if wn else float("nan"))
    with open(a.out + ".txt", "w") as fh:
        P = lambda s="": fh.write(s + "\n")
        P(f"# doc pdvd/115 2-D residual: det={a.det} base={a.base} arm={a.arm} events={len(evs)} (missing {missing}); "
          f"common records {SA['records']}; rows base {SA['rows']} arm {SB['rows']}; DEAD_R {DEAD_R} cm")
        P("# d_exact = fractional projected wire rank - nearest charged cell of the cluster on the row's slice (wire units);")
        P("# d_pm1 = the same over slices s-1..s+1; OFF = no cell on the slice or |d| > 1; dead planes excluded")
        P()
        P("| plane | arm | live rows | no cell on slice | median abs d | abs d > 1 | abs d > 2 | +-1 slice: abs d > 1 | OFF (exact) | OFF (+-1 slice) |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for pl in PLANES:
            for lab, S in (("base", SA), ("arm", SB)):
                P(f"| {pl} | {lab} | {S[f'live_{pl}']} | {100*S[f'noslice_{pl}']:.2f} % | {S[f'med_absd_{pl}']:.3f} | "
                  f"{100*S[f'gt1_{pl}']:.2f} % | {100*S[f'gt2_{pl}']:.2f} % | {100*S[f'gt1pm1_{pl}']:.2f} % | "
                  f"{100*S[f'off_{pl}']:.2f} % | {100*S[f'offpm1_{pl}']:.2f} % |")
        P()
        P("| row metric | base | arm | rel |")
        P("|---|---|---|---|")
        for kk, name in (("R2D", "R2D: W-plane live rows with abs d_exact > 1 wire"),
                         ("offrow", "rows OFF in >= 1 live plane (exact slice, > 1 wire)"),
                         ("offrow_pm1", "rows OFF in >= 1 live plane (+-1 slice, > 1 wire)"),
                         ("offrow_pm1w", "rows OFF in >= 1 live plane (+-1 slice, > 1.5 wire ~ doc-111 P2)"),
                         ("qneg", "rows with q < 0 (the Bee cut)")):
            rel = (SB[kk] - SA[kk]) / SA[kk] if SA[kk] else float("nan")
            P(f"| {name} | {100*SA[kk]:.2f} % | {100*SB[kk]:.2f} % | {100*rel:+.1f} % |")
        P()
        P(f"paired per record, W-plane share abs d > 1: arm lower {paired['better']}, higher {paired['worse']}, "
          f"equal (+-0.005) {paired['same']}; row-weighted mean difference {100*paired['mean_diff']:+.3f} points")
    json.dump(dict(det=a.det, base=a.base, arm=a.arm, events=len(evs), A=SA, B=SB, paired=paired), open(a.out + ".json", "w"), indent=1)
    with gzip.open(a.out + "_rows.tsv.gz", "wt") as fh:
        fh.write("event cluster pass row arm dU dV dW dU_pm1 dV_pm1 dW_pm1 dead_uvw\n".replace(" ", "\t"))
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(open(a.out + ".txt").read())


if __name__ == "__main__":
    main()
