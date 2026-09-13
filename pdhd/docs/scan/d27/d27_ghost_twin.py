#!/usr/bin/env python3
"""doc pdhd/27 sec 3 -- causal control for the front-face distance defect on PDHD.

    STM_SCAN_RECORD=<smx27 record> python3 d27_ghost_twin.py > ghost_twin.txt     (writes ghost_twin.json)


2-D cell point (Facade_Grouping.cxx:831): x = time2drift(face)(time * tick), y = pitch[plane] * (wire + 0.5) + center.
A centre is put through the same conversion at an integer (tick, wire) (convert_3Dpoint_time_ch).  So within one
(apa, face, plane) every recorded distance is d = hypot(s * (t - tc), p * (w - wc)), offsets and signs cancel, and x is
common to the three planes of one face (same tc).

1. p per plane from the geometry file; s fitted on W cells.
2. TWIN: trilaterate (tc, wc) per (item, face, plane) from the recorded d_stop / d_ctl; residuals must be ~0 -- on the
   muon's face AND on the other face (i.e. the 152 cm numbers are exactly the face-0 lattice distances).
3. CORRECTION: every U/V cell is re-measured on the NEAREST of its channel's wires on the muon's face, against the
   muon-face centre (tc from W; wc from the U/V cells already recorded on the muon's face -- on APA1/3 only the ones
   any_within put there).  Region and control are rebuilt with the C++ sum rule and plane rule.  A role-1 cell that
   was never inside a radius under the recorded distance has own_blob "not computed" (0): "fix" counts it as own,
   "fixlo" does not -- bounds, since on APA0/2 85-88 % of role-1 cells in radius are own.
4. NEGATIVE CONTROL: cells recorded on the muon's face on a channel with ONE muon-face wire must not move (> 0.01 cm).
   GHOST CHECK: cells recorded on the muon's face within 30 cm must not be moved to another segment.
"""
import bz2, json, glob, os, sys, collections
import numpy as np, uproot
from scipy.optimize import least_squares
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM
R = 10.0; W8 = np.array([0.25, 0.25, 1.0]); SW = 0.04; KMEV = 41.3035e-6
ARM = "h26q2dprod"

g = json.load(bz2.open("/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunehd-wires-larsoft-v1.json.bz2"))["Store"]
c2w = collections.defaultdict(list); pitch = {}; NW = {}
for a in g["anodes"]:
    a = a["Anode"]
    for fi in a["faces"]:
        f = g["faces"][fi]["Face"]
        for pi in f["planes"]:
            pl = g["planes"][pi]["Plane"]
            NW[(a["ident"], f["ident"], pl["ident"])] = len(pl["wires"])
            cen = []
            for idx, wi in enumerate(pl["wires"]):
                w = g["wires"][wi]["Wire"]
                c2w[(a["ident"], w["channel"])].append((f["ident"], pl["ident"], idx))
                t, h = g["points"][w["tail"]]["Point"], g["points"][w["head"]]["Point"]
                cen.append(((t["y"] + h["y"]) / 2, (t["z"] + h["z"]) / 2, h["y"] - t["y"], h["z"] - t["z"]))
            cen = np.array(cen); wd = cen[:, 2:].mean(0); wd /= np.linalg.norm(wd); pd = np.array([-wd[1], wd[0]])
            steps = np.diff(cen[:, :2] @ pd)
            pitch[(a["ident"], f["ident"], pl["ident"])] = float(np.median(np.abs(steps))) / 10.0   # mm -> cm
P = {pl: np.median([v for k, v in pitch.items() if k[2] == pl]) for pl in range(3)}
print("pitch [cm] per plane (median over apa/face):", {k: round(v, 4) for k, v in P.items()},
      "spread", round(max(pitch.values()) - min(pitch.values()), 5))


def combine(s, n):
    s = np.asarray(s, float); w = W8.copy(); w[np.asarray(n) == 0] = 0
    mn, mx = int(np.argmin(s)), int(np.argmax(s))
    md = [i for i in range(3) if i not in (mn, mx)][0] if mn != mx else 1
    if mn == mx: mn, mx = 0, 2
    ov = (w * s).sum() / w.sum() if w.sum() > 0 else 0.0
    asy = abs(s[md] - s[mx]) / (s[md] + s[mx]) if s[md] + s[mx] > 0 else 0.0
    if asy > SW and w[md] + w[mn] > 0: ov = (w[md] * s[md] + w[mn] * s[mn]) / (w[md] + w[mn])
    return ov


def trilat(t, w, d, s, p, tc0=None):
    """-> (tc, wc, rms residual cm).  tc fixed when tc0 is given."""
    if len(d) < 2: return np.nan, np.nan, np.nan
    i0 = int(np.argmin(d))
    if tc0 is None:
        best = None
        for sgn_t in (-1, 1):
            for sgn_w in (-1, 1):
                x0 = [t[i0] + sgn_t * d[i0] / s * 0.5, w[i0] + sgn_w * d[i0] / p * 0.5]
                r = least_squares(lambda v: np.hypot(s * (t - v[0]), p * (w - v[1])) - d, x0)
                if best is None or r.cost < best.cost: best = r
        res = np.hypot(s * (t - best.x[0]), p * (w - best.x[1])) - d
        return best.x[0], best.x[1], float(np.sqrt(np.mean(res ** 2)))
    best = None
    for sgn in (-1, 1):
        r = least_squares(lambda v: np.hypot(s * (t - tc0), p * (w - v[0])) - d, [w[i0] + sgn * d[i0] / p * 0.5])
        if best is None or r.cost < best.cost: best = r
    res = np.hypot(s * (t - tc0), p * (w - best.x[0])) - d
    return tc0, best.x[0], float(np.sqrt(np.mean(res ** 2)))


its, _ = BM.items("pdhd", ARM, "strict"); judged = {x[0]: x for x in its}
CB = ["cluster_id", "apa", "plane", "face", "wire", "channel", "time", "charge", "pred_mu", "pred_all", "xshared", "own_blob", "d_stop_cm", "d_ctl_cm", "role"]
TB = ["cluster_id", "is_stm", "michel_found", "michel_q2d_region", "michel_q2d_ctl", "michel_ke_q2d_region", "michel_ke_q2d_ctl"]
items = []
for fn in sorted(glob.glob(f"{IMG}/pdhd/work/*_{ARM}/tracking-pr.root")):
    ev = os.path.basename(os.path.dirname(fn))[: -len(ARM) - 1]
    u = uproot.open(fn)
    if "T_stm_michel_2d" not in u: continue
    t = u["T_stm_michel"].arrays(TB, library="np"); C = u["T_stm_michel_2d"].arrays(CB, library="np")
    for i in range(len(t["cluster_id"])):
        c = int(t["cluster_id"][i]); key = f"{ev}/{c}"
        if not (int(t["is_stm"][i]) == 1 and int(t["michel_found"][i]) == 1): continue
        m = C["cluster_id"] == c
        cc = {k: C[k][m] for k in CB}
        x = judged.get(key)
        hm = bool(x and x[1] in BM.STOP and x[2] in BM.MICHEL_KINDS)
        items.append(dict(key=key, hm=hm, C=cc, q_reg=float(t["michel_q2d_region"][i]), q_ctl=float(t["michel_q2d_ctl"][i]),
                          ke_reg=float(t["michel_ke_q2d_region"][i]), ke_ctl=float(t["michel_ke_q2d_ctl"][i])))
print(f"is_stm & michel_found candidates (every APA, no truth) {len(items)}; hand Michel (APA0 strict) {sum(it['hm'] for it in items)}")

# ---- s from W cells on a handful of items -----------------------------------------------------------
def w_groups(it):
    """-> (apa, face, mask of W cells on that apa and face) for the dominant (apa, face) of the W role-1 cells, or None"""
    C = it["C"]; s = C["plane"] == 2
    k = collections.Counter(zip(C["apa"][s & (C["role"] == 1)].tolist(), C["face"][s & (C["role"] == 1)].tolist())).most_common(1)
    if not k: return None
    apa, face = k[0][0]
    return apa, face, s & (C["face"] == face) & (C["apa"] == apa)
items = [it for it in items if w_groups(it) is not None]
print(f"  with W role-1 cells: {len(items)}")
cal = items[:12]
scan = []
for s_try in np.arange(0.070, 0.090, 0.0002):
    tot = 0.0
    for it in cal:
        apa, face, s = w_groups(it); C = it["C"]
        _, _, rms = trilat(C["time"][s].astype(float), C["wire"][s].astype(float), C["d_stop_cm"][s], s_try, pitch[(apa, face, 2)])
        tot += rms
    scan.append((tot, s_try))
S_T = min(scan)[1]
print(f"s = cm per tick fitted on W cells of {len(cal)} items: {S_T:.4f} (sum of rms {min(scan)[0]:.4f} cm)")

# ---- per item: twin, correction ---------------------------------------------------------------------
out = []
for it in items:
    C = it["C"]; apa, face, sW = w_groups(it)
    T = C["time"].astype(float); Wi = C["wire"].astype(float)
    r = dict(key=it["key"], hm=it["hm"], apa=apa, face=face, ke_reg=it["ke_reg"], ke_ctl=it["ke_ctl"],
             n_other_apa=int((C["apa"] != apa).sum()))
    tc = {}
    for nm, dc in (("stop", "d_stop_cm"), ("ctl", "d_ctl_cm")):
        tcW, wcW, rmsW = trilat(T[sW], Wi[sW], C[dc][sW], S_T, pitch[(apa, face, 2)]); tc[nm] = tcW; r[f"rmsW_{nm}"] = rmsW
        for pl in (0, 1):
            sm = (C["plane"] == pl) & (C["face"] == face) & (C["apa"] == apa)
            so = (C["plane"] == pl) & (C["face"] != face) & (C["apa"] == apa)
            r[f"rms_mu_{nm}_{pl}"] = trilat(T[sm], Wi[sm], C[dc][sm], S_T, pitch[(apa, face, pl)], tc0=tcW)[2] if sm.sum() >= 2 else np.nan
            r[f"rms_other_{nm}_{pl}"] = trilat(T[so], Wi[so], C[dc][so], S_T, pitch[(apa, 1 - face, pl)])[2] if so.sum() >= 3 else np.nan
    # corrected distances: every U/V cell of this apa re-measured on the NEAREST of its channel's wires on the muon's
    # face.  That covers both forms of the defect -- the other face (APA1/3) and the first of two segments on the
    # muon's face (every APA; d27_wrap_face.py sec 5).
    # NEGATIVE CONTROL: a cell recorded on the muon's face whose channel has ONE muon-face wire must come back unchanged.
    # GHOST CHECK: a cell recorded on the muon's face within 30 cm that the nearest-wire rule moves to another segment.
    dnew = {"stop": C["d_stop_cm"].copy(), "ctl": C["d_ctl_cm"].copy()}
    ok = True; neg_n = neg_bad = ghost = 0; n_changed = {"stop": 0, "ctl": 0}
    for pl in (0, 1):
        sa = (C["plane"] == pl) & (C["apa"] == apa)
        if not sa.any(): continue
        sm = sa & (C["face"] == face)
        for nm, dc in (("stop", "d_stop_cm"), ("ctl", "d_ctl_cm")):
            if sm.sum() < 2: ok = False; continue
            _, wc, rms = trilat(T[sm], Wi[sm], C[dc][sm], S_T, pitch[(apa, face, pl)], tc0=tc[nm])
            if not np.isfinite(rms) or rms > 0.05: ok = False; continue
            nwp = NW[(apa, face, pl)]
            for j in np.where(sa)[0]:
                # the geometry file lists face-1 wires in the opposite order to the index the chain uses: every
                # recorded face-1 cell (W included) carries n - 1 - file index, every face-0 cell the file index
                alts = [(w if face == 0 else nwp - 1 - w) for (f_, p_, w) in c2w[(apa, int(C["channel"][j]))] if f_ == face and p_ == pl]
                if not alts:
                    dnew[nm][j] = -1.0; continue
                ds = [np.hypot(S_T * (T[j] - tc[nm]), pitch[(apa, face, pl)] * (a - wc)) for a in alts]
                k = int(np.argmin(ds)); rec = float(C[dc][j]); on = int(C["face"][j]) == face
                if on and len(alts) == 1:
                    neg_n += 1; neg_bad += int(abs(ds[k] - rec) > 0.01)
                if on and 0 <= rec <= 30 and alts[k] != int(Wi[j]): ghost += 1
                if abs(ds[k] - rec) > 0.01: n_changed[nm] += 1
                dnew[nm][j] = ds[k]
    r["corrected_ok"] = ok; r["neg_n"] = neg_n; r["neg_bad"] = neg_bad; r["ghost"] = ghost
    r["n_changed_stop"] = n_changed["stop"]; r["n_changed_ctl"] = n_changed["ctl"]
    # sums: chain rule (own != 0) ; a role-1 cell that was never in a region under the recorded distance has own_blob 0
    # "not computed" -- assume own for role-1 cells (the fit predicts the muon there), and say how often that is needed
    pmu = np.maximum(C["pred_mu"], 0); xs = C["xshared"] == 1
    con = np.where(xs, np.maximum(C["pred_all"] - pmu, 0), C["charge"] - pmu)
    for tag, dd in (("rec", {"stop": C["d_stop_cm"], "ctl": C["d_ctl_cm"]}), ("fix", dnew), ("fixlo", dnew)):
        for nm in ("stop", "ctl"):
            inr = (dd[nm] >= 0) & (dd[nm] <= R)
            was_in = (C["d_stop_cm"] >= 0) & (C["d_stop_cm"] <= R) | (C["d_ctl_cm"] >= 0) & (C["d_ctl_cm"] <= R)
            # fix: a role-1 cell whose own_blob was never computed counts as own; fixlo: it does not (lower bound)
            own = (C["own_blob"] != 0) | ((C["role"] == 1) & ~was_in) if tag == "fix" else (C["own_blob"] != 0)
            sel = inr & own
            Ssum = [con[sel & (C["plane"] == p)].sum() for p in range(3)]; N = [int((sel & (C["plane"] == p)).sum()) for p in range(3)]
            r[f"{tag}_{nm}"] = max(combine(Ssum, N), 0) * KMEV; r[f"{tag}_{nm}_N"] = N
            if tag == "fix": r[f"assumed_own_{nm}"] = int((inr & (C["role"] == 1) & ~was_in).sum())
    out.append(r)

A = lambda k, s: np.array([x[k] for x in out if s(x)], float)
print("\n=== TWIN (recorded distances reproduced from the (tick, wire) lattice), rms residual cm, median / max")
for lab, s in (("muon face 0 (APA0/APA2)", lambda x: x["face"] == 0), ("muon face 1 (APA1/APA3)", lambda x: x["face"] == 1)):
    print(f"  [{lab}] n {len(A('rmsW_stop', s))}")
    for k in ("rmsW_stop", "rmsW_ctl", "rms_mu_stop_0", "rms_mu_stop_1", "rms_other_stop_0", "rms_other_stop_1", "rms_other_ctl_0"):
        v = A(k, s); v = v[np.isfinite(v)]
        if len(v): print(f"    {k:18s} n {len(v):3d} median {np.median(v):.4f} max {v.max():.4f}")
print("  chain vs offline recorded-distance sums: region max |diff| %.2e MeV, control %.2e MeV" % (
    max(abs(x["rec_stop"] - x["ke_reg"]) for x in out), max(abs(x["rec_ctl"] - x["ke_ctl"]) for x in out)))

print("\n=== CORRECTION (U/V distances recomputed on the nearest muon-face wire of each channel)")
print(f"  NEGATIVE CONTROL: cells recorded on the muon's face on a one-muon-face-wire channel: {sum(x['neg_n'] for x in out)} "
      f"(both centres), changed by more than 0.01 cm: {sum(x['neg_bad'] for x in out)}")
print(f"  GHOST CHECK: cells recorded on the muon's face within 30 cm moved to another segment: {sum(x['ghost'] for x in out)}")
print(f"  cells whose region distance changed: {sum(x['n_changed_stop'] for x in out)}, control distance: {sum(x['n_changed_ctl'] for x in out)}")
for lab, s in (("muon on APA0/APA2 (face 0), corrected", lambda x: x["face"] == 0 and x["corrected_ok"]),
               ("muon on APA1/APA3 (face 1), corrected", lambda x: x["face"] == 1 and x["corrected_ok"]),
               ("NOT correctable (fewer than 2 muon-face U/V anchor cells)", lambda x: not x["corrected_ok"])):
    for hm_lab, hs in (("every is_stm Michel", lambda x: True), ("hand Michel", lambda x: x["hm"])):
        ss = lambda x: s(x) and hs(x)
        n = len(A("rec_stop", ss))
        if not n: continue
        d_reg = A("fix_stop", ss) - A("rec_stop", ss); d_ctl = A("fix_ctl", ss) - A("rec_ctl", ss)
        nU = np.array([x["fix_ctl_N"][0] for x in out if ss(x)]); nU0 = np.array([x["rec_ctl_N"][0] for x in out if ss(x)])
        print(f"  [{lab} | {hm_lab}] n {n}: region {np.median(A('rec_stop', ss)):5.1f} -> {np.median(A('fix_stop', ss)):5.1f} MeV (items changed {int((np.abs(d_reg) > 1e-9).sum())}, "
              f"median shift {np.median(d_reg):+.1f}) | control {np.median(A('rec_ctl', ss)):5.1f} -> {np.median(A('fix_ctl', ss)):5.1f} (changed {int((np.abs(d_ctl) > 1e-9).sum())}) | "
              f"control U cells p50 {np.median(nU0):.0f} -> {np.median(nU):.0f} | cells whose own_blob had to be assumed p50 region {np.median(A('assumed_own_stop', ss)):.0f} ctl {np.median(A('assumed_own_ctl', ss)):.0f}")
print("\n=== PDHD hand Michel, pooled (every correctable item corrected), median MeV")
hm = [x for x in out if x["hm"]]
for tag in ("rec", "fix", "fixlo"):
    reg = np.array([x[f"{tag}_stop"] if x["corrected_ok"] else x["rec_stop"] for x in hm])
    ctl = np.array([x[f"{tag}_ctl"] if x["corrected_ok"] else x["rec_ctl"] for x in hm])
    print(f"  {tag:6s} n {len(hm)}: region {np.median(reg):5.1f} (above 52.8: {int((reg > 52.83).sum())}) | control {np.median(ctl):5.1f} (above 10: {int((ctl > 10).sum())})")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghost_twin.json"), "w"), default=float)
