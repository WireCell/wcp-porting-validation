#!/usr/bin/env python3
"""doc pdhd/25 sec 3 -- where PDHD's remaining stopper misses sit, and what the obvious levers would
buy, sized OFFLINE on production's own payload (no arm, no knob).

    python3 d25_misses.py [--pdhd h23conf] [--pdvd p96vprod]

Reuses d25_bragg_michel.py's reader, truth and APA assignment (and its self-gates).

The re-verdict model -- every gate below is the C++'s, read from source, not guessed:
  pre-P1 bits   = reject_bits | topology_cleared_bits   (P1 runs LAST, on final bits:
                  CheckSTM_Michel.cxx:4440-4451)
  no_bragg      = contrast < bragg_contrast_min * contrast_expected       (:3083)
  shape_flat    = ks_mu + ks_margin >= ks_flat                            (:3121)
  P1 clear      = michel_found AND conn_type in {1,2} AND ke >= ke_min AND len >= len_min
                  -> clears no_bragg|shape_flat (|profile_sparse)         (StmMichelFunctions.cxx:772)
  is_stm        = (bits == 0)                                             (asserted on the payload)
Items where the Bragg-peak geometric fallback or the wide anchor fired are NOT re-modelled (those
paths re-derive the bits from a second profile reading); their recorded bits are carried unchanged.
The model must reproduce production's recorded bits EXACTLY on every modelled item before any sweep
is printed; P1's clear must reproduce topology_cleared_bits on every item.

Sweeps are PREDICTIONS for a single-knob change on a FIXED payload.  They cannot see a knob that
moves the fit or the candidate pool, and docs pdhd/21-23 showed combined arms break naive twins.
"""
import argparse, collections, glob, json, math, os, re, sys
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d25_bragg_michel as Q

IMG = Q.IMG
TK = "/nfs/data/1/xqian/toolkit-dev/toolkit/clus/src/CheckSTM_Michel.cxx"
CFG = {"pdhd": IMG + "/pdhd/work/029107_17_h25cfg0/.wct-pr_h25cfg0.json",
       "pdvd": IMG + "/pdvd/work/039252_0_h25vcfg/.wct-pr_h25vcfg.json"}
BITS = ["no_chain", "stop_unmatched", "no_bragg", "shape_flat", "not_muon_pid",
        "continuation", "stop_near_boundary", "vertex_hadron", "short",
        "profile_sparse", "plateau_off_mip", "stop_into_dead", "cluster_not_track",
        "profile_geometry"]                       # census_lib.STM_BITS, validated on 303/303 rows (doc 21)
B = {n: 1 << i for i, n in enumerate(BITS)}
names = lambda bits: "+".join(n for i, n in enumerate(BITS) if int(bits) >> i & 1) or "-"


def cxx_default(member):
    m = re.search(rf"\b{member}\{{([^}}]*)\}}", open(TK).read())
    return float(m.group(1))


def prod_params(det):
    """production constants from the COMPILED config; absent keys fall back to the C++ default."""
    sys.argv_saved = sys.argv
    d = json.load(open(CFG[det])); bag = None
    def walk(o):
        nonlocal bag
        if isinstance(o, dict):
            if "ks_margin" in o and "max_candidates" in o: bag = o
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(d)
    g = lambda k, member: float(bag[k]) if k in bag else cxx_default(member)
    return dict(ks_margin=g("ks_margin", "m_ks_margin"),
                contrast_min=g("bragg_contrast_min", "m_bragg_contrast_min"),
                p1=bool(bag.get("topology_stop_evidence", False)),
                ke_min=g("topology_michel_ke_min", "m_topology_michel_ke_min"),
                len_min=g("topology_michel_len_min_cm", "m_topology_michel_len_min_cm"),
                sparse=bool(bag.get("topology_clears_sparse", False)),
                mip=float(bag["mip_dqdx"]))


def modelled(d):
    return not (int(d["bragg_anchor_fallback"]) or int(d["wide"]))


def p1_clear(bits, d, P):
    if not P["p1"] or not int(d["michel_found"]) or int(d["michel_conn_type"]) not in (1, 2):
        return 0
    ke, ln = float(d["michel_ke_best"]), float(d["michel_len"])
    if not (math.isfinite(ke) and math.isfinite(ln)):
        return 0
    clr = 0
    if ke >= P["ke_min"] and ln >= P["len_min"]:
        clr = B["no_bragg"] | B["shape_flat"] | (B["profile_sparse"] if P["sparse"] else 0)
    # HYPOTHETICAL, NO SUCH KNOB EXISTS: a strong Michel also clears plateau_off_mip
    pc = P.get("plateau_clear")
    if pc and ke >= pc[0] and ln >= pc[1]:
        clr |= B["plateau_off_mip"] | B["no_bragg"] | B["shape_flat"]
    return bits & clr


def reverdict(d, P):
    pre = int(d["reject_bits"]) | int(d["topology_cleared_bits"])
    bits = pre
    if modelled(d):
        if float(d["contrast_expected"]) > 0 and not (pre & B["profile_sparse"]):
            bits &= ~B["no_bragg"]
            if float(d["contrast"]) < P["contrast_min"] * float(d["contrast_expected"]):
                bits |= B["no_bragg"]
        if not (float(d["ks_mu"]) == 0 and float(d["ks_flat"]) == 0):
            bits &= ~B["shape_flat"]
            if float(d["ks_mu"]) + P["ks_margin"] >= float(d["ks_flat"]):
                bits |= B["shape_flat"]
    pre_model = bits
    bits &= ~p1_clear(bits, d, P)
    return pre_model, bits


def self_check(det, its, P):
    bad_bits = [x[0] for x in its if modelled(x[4])
                and reverdict(x[4], P)[0] != (int(x[4]["reject_bits"]) | int(x[4]["topology_cleared_bits"]))]
    bad_p1 = [x[0] for x in its
              if p1_clear(int(x[4]["reject_bits"]) | int(x[4]["topology_cleared_bits"]), x[4], P)
              != int(x[4]["topology_cleared_bits"])]
    bad_stm = [x[0] for x in its if (int(x[4]["reject_bits"]) == 0) != (int(x[4]["is_stm"]) == 1)]
    n_mod = sum(1 for x in its if modelled(x[4]))
    print(f"MODEL CHECK {det}: params {P}")
    print(f"  modelled {n_mod}/{len(its)}; pre-P1 bits reproduced on {n_mod - len(bad_bits)}/{n_mod}"
          f"{' MISMATCH ' + str(bad_bits[:8]) if bad_bits else ''}")
    print(f"  P1 clear reproduces topology_cleared_bits on {len(its) - len(bad_p1)}/{len(its)}"
          f"{' MISMATCH ' + str(bad_p1[:8]) if bad_p1 else ''}")
    print(f"  is_stm == (reject_bits == 0) on {len(its) - len(bad_stm)}/{len(its)}"
          f"{' MISMATCH ' + str(bad_stm[:8]) if bad_stm else ''}")
    if bad_bits or bad_p1 or bad_stm:
        sys.exit("model does not reproduce production -- refusing to print any sweep")


def score(its, P, base=None):
    c = collections.Counter(); gained, newfp, lost = [], [], []
    for k, v, kind, src, d in its:
        hs = v in Q.STOP; cs = reverdict(d, P)[1] == 0
        c["TP" if hs and cs else "FN" if hs else "FP" if cs else "TN"] += 1
        if base is not None:
            was = int(d["is_stm"]) == 1
            if cs and not was: (gained if hs else newfp).append(k)
            if was and not cs and hs: lost.append(k)
    TP, FP, FN, TN = c["TP"], c["FP"], c["FN"], c["TN"]
    return TP, FP, FN, TN, gained, newfp, lost


def line(label, r, show_items=False):
    TP, FP, FN, TN, g, f, l = r
    s = (f"  {label:34s} {TP:3d}/{FP:2d}/{FN:3d}/{TN:3d}  purity {TP/max(1,TP+FP):.3f} "
         f"eff {TP/max(1,TP+FN):.3f}  (+{len(g)} TP, +{len(f)} FP, -{len(l)} TP)")
    if show_items and (g or f or l):
        s += f"\n      gained {' '.join(sorted(g))}\n      new FP {' '.join(sorted(f)) or '-'}"
        if l: s += f"\n      lost   {' '.join(sorted(l))}"
    return s


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def profile_noise(det, arm, keys, mip):
    """per candidate, over the muon chain (role 1) with rr in [20, 60] cm and q above the chain's own
    0.15-MIP live cut: robust relative scatter 1.4826*MAD/median, and the median neighbour-to-neighbour
    relative jump |q_i - q_{i+1}| / median.  -> {key: (n, scatter, jump)}"""
    out = {}
    want = collections.defaultdict(set)
    for k in keys:
        e, c = k.split("/"); want[e].add(int(c))
    for f in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/tracking-pr.root")):
        evt = os.path.basename(os.path.dirname(f)).replace("_" + arm, "")
        if evt not in want: continue
        p = uproot.open(f)["T_stm_michel_pts"].arrays(["cluster_id", "role", "rr", "L", "q"], library="np")
        for c in want[evt]:
            m = (p["cluster_id"] == c) & (p["role"] == 1)
            rr, L, q = p["rr"][m], p["L"][m], p["q"][m]
            o = np.argsort(L); rr, q = rr[o], q[o]
            s = (rr >= 20) & (rr <= 60) & (q > 0.15 * mip)
            if s.sum() < 10: continue
            qs = q[s]; med = np.median(qs)
            out[f"{evt}/{c}"] = (int(s.sum()), 1.4826 * np.median(np.abs(qs - med)) / med,
                                 np.median(np.abs(np.diff(qs))) / med)
    return out


def q(a, p): return np.percentile(np.asarray(a), p) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdhd", default="h23conf")
    ap.add_argument("--pdvd", default="p96vprod")
    a = ap.parse_args()
    Php, Pvd = prod_params("pdhd"), prod_params("pdvd")
    hd = {pop: Q.items("pdhd", a.pdhd, pop)[0] for pop in ("strict", "majority", "all")}
    vd = Q.items("pdvd", a.pdvd, "all")[0]
    # the arms must be the committed production censuses (same gate as d25_bragg_michel)
    for lab, its, want in (("pdhd strict", hd["strict"], (77, 0, 28, 69)),
                           ("pdhd majority", hd["majority"], (79, 1, 30, 70)),
                           ("pdvd", vd, (242, 8, 34, 262))):
        got = Q.census(its)[0]
        print(f"GATE {lab}: {got} want {want} -> {'PASS' if got == want else 'FAIL'}")
        if got != want: sys.exit("gate failed")
    self_check("pdhd strict", hd["strict"], Php)
    self_check("pdhd majority", hd["majority"], Php)
    self_check("pdvd", vd, Pvd)

    # ---- 1. the misses, decomposed ------------------------------------------------------
    print("\n==== 1. MISSED hand stoppers, by reject bits (after P1) ====")
    for lab, its in (("PDHD strict", hd["strict"]), ("PDVD", vd)):
        miss = [x for x in its if x[1] in Q.STOP and int(x[4]["is_stm"]) == 0]
        c = collections.Counter(names(x[4]["reject_bits"]) for x in miss)
        only_shape = sum(1 for x in miss if int(x[4]["reject_bits"]) & ~(B["no_bragg"] | B["shape_flat"]) == 0)
        mf = sum(1 for x in miss if int(x[4]["michel_found"]) == 1)
        print(f"  {lab}: {len(miss)} misses; only no_bragg/shape_flat: {only_shape}; carry michel_found=1: {mf}")
        for k, n in c.most_common(): print(f"      {k:45s} {n}")

    # ---- 2. how well the shape quantities separate stoppers from through-goers --------
    print("\n==== 2. SEPARATION: AUC of the chain's own shape quantities, hand stoppers vs hand THRU ====")
    print("  (0.5 = no separation, 1.0 = perfect; ks_gap = ks_flat - ks_mu, cr = contrast / contrast_expected)")
    for lab, its in (("PDHD strict", hd["strict"]), ("PDHD majority", hd["majority"]), ("PDVD", vd)):
        S = [x[4] for x in its if x[1] in Q.STOP]; T = [x[4] for x in its if x[1] == "THRU"]
        ks = lambda L: [float(d["ks_flat"]) - float(d["ks_mu"]) for d in L if not (float(d["ks_mu"]) == 0 and float(d["ks_flat"]) == 0)]
        cr = lambda L: [float(d["contrast"]) / float(d["contrast_expected"]) for d in L if float(d["contrast_expected"]) > 0]
        pm = lambda L, mip: [float(d["plateau_med"]) / mip for d in L if float(d["plateau_med"]) > 0]
        mip = Php["mip"] if lab.startswith("PDHD") else Pvd["mip"]
        print(f"  {lab:14s} n_stop {len(S):3d} n_thru {len(T):3d}   AUC ks_gap {auc(ks(S), ks(T)):.3f}   "
              f"AUC contrast/expected {auc(cr(S), cr(T)):.3f}")
        print(f"      stoppers: ks_gap p25/50/75 {q(ks(S),25):.3f}/{q(ks(S),50):.3f}/{q(ks(S),75):.3f}   "
              f"c/e {q(cr(S),25):.2f}/{q(cr(S),50):.2f}/{q(cr(S),75):.2f}   plateau/mip {q(pm(S,mip),25):.2f}/{q(pm(S,mip),50):.2f}/{q(pm(S,mip),75):.2f}")
        print(f"      THRU    : ks_gap p25/50/75 {q(ks(T),25):.3f}/{q(ks(T),50):.3f}/{q(ks(T),75):.3f}   "
              f"c/e {q(cr(T),25):.2f}/{q(cr(T),50):.2f}/{q(cr(T),75):.2f}   plateau/mip {q(pm(T,mip),25):.2f}/{q(pm(T,mip),50):.2f}/{q(pm(T,mip),75):.2f}")

    # ---- 3. profile noise ---------------------------------------------------------------
    print("\n==== 3. PROFILE NOISE on the muon plateau (rr 20-60 cm, live points) ====")
    N = {}
    N["hd"] = profile_noise("pdhd", a.pdhd, [x[0] for x in hd["majority"]], Php["mip"])
    N["vd"] = profile_noise("pdvd", a.pdvd, [x[0] for x in vd], Pvd["mip"])
    for lab, its, nz in (("PDHD strict", hd["strict"], N["hd"]), ("PDVD", vd, N["vd"])):
        for cls, sel in (("stopper TP", lambda x: x[1] in Q.STOP and int(x[4]["is_stm"]) == 1),
                         ("stopper MISS", lambda x: x[1] in Q.STOP and int(x[4]["is_stm"]) == 0),
                         ("THRU", lambda x: x[1] == "THRU")):
            v = [nz[x[0]] for x in its if sel(x) and x[0] in nz]
            if not v: continue
            sc = [t[1] for t in v]; jp = [t[2] for t in v]; npt = [t[0] for t in v]
            print(f"  {lab:12s} {cls:13s} n={len(v):3d}  scatter p25/50/75 {q(sc,25):.3f}/{q(sc,50):.3f}/{q(sc,75):.3f}"
                  f"   jump {q(jp,25):.3f}/{q(jp,50):.3f}/{q(jp,75):.3f}   pts {int(q(npt,50))}")

    # ---- 4. offline lever sweeps -------------------------------------------------------
    print("\n==== 4. OFFLINE SWEEPS (prediction on production's payload; one lever at a time) ====")
    SWEEP = [("production", {})]
    SWEEP += [(f"P1 ke_min {ke:g} len_min {ln:g}", dict(ke_min=ke, len_min=ln))
              for ke, ln in ((10, 2), (10, 1.5), (10, 1), (7, 3), (5, 3), (5, 2), (5, 1.5), (3, 3), (3, 1.5), (0, 0))]
    SWEEP += [("P1 + clears_sparse", dict(sparse=True))]
    SWEEP += [(f"ks_margin {m:g}", dict(ks_margin=m)) for m in (-0.03, -0.04, -0.05, -0.06, -0.08, -0.10, -0.12, -0.15, -0.20)]
    SWEEP += [(f"bragg_contrast_min {c:g}", dict(contrast_min=c)) for c in (0.5, 0.4, 0.3)]
    SWEEP += [("ks -0.05 + contrast 0.5", dict(ks_margin=-0.05, contrast_min=0.5)),
              ("ks -0.05 + P1 ke 5 len 1.5", dict(ks_margin=-0.05, ke_min=5, len_min=1.5)),
              ("ks -0.10 + P1 ke 5 len 1.5", dict(ks_margin=-0.10, ke_min=5, len_min=1.5)),
              ("HYPO strong Michel clears plateau (20 MeV, 10 cm)", dict(plateau_clear=(20.0, 10.0))),
              ("HYPO ks -0.10 + P1 5/1.5 + plateau 20/10", dict(ks_margin=-0.10, ke_min=5, len_min=1.5, plateau_clear=(20.0, 10.0)))]
    for lab, its, P0, show in (("PDHD APA0 strict", hd["strict"], Php, True),
                               ("PDHD APA0 majority", hd["majority"], Php, False),
                               ("PDHD all four APAs (APA0 INCLUDED, reference only)", hd["all"], Php, False),
                               ("PDVD (with a candidate)", vd, Pvd, False)):
        print(f"\n  --- {lab} ---")
        for name, ov in SWEEP:
            if lab.startswith("PDVD") and name.startswith("P1") and ov.get("ke_min", 99) >= P0["ke_min"] and "sparse" not in ov:
                pass
            P = dict(P0); P.update(ov)
            print(line(name, score(its, P, base=True), show_items=show and name != "production"))

    # ---- 5. the PDHD miss table ----------------------------------------------------------
    print("\n==== 5. PDHD APA0-strict misses, one row each ====")
    print("  APA0-majority items are excluded here; apa = the majority APA of the role-1 points")
    print("  apa key            verdict     kind      bits                    mf conn  keMeV  mlen  muon  plat/mip  c/e   ks_mu  ks_flat gap    dead/prof  scatter  nearest lever")
    for k, v, kind, src, d in sorted(hd["strict"], key=lambda x: x[0]):
        if v not in Q.STOP or int(d["is_stm"]) == 1: continue
        bits = int(d["reject_bits"]); gap = float(d["ks_flat"]) - float(d["ks_mu"])
        ce = float(d["contrast"]) / float(d["contrast_expected"]) if float(d["contrast_expected"]) > 0 else float("nan")
        shape_only = bits & ~(B["no_bragg"] | B["shape_flat"]) == 0
        mf = int(d["michel_found"]); conn = int(d["michel_conn_type"])
        if shape_only and mf and conn in (1, 2):
            lever = "P1 floor (Michel found, below KE/len floor)"
        elif bits == B["shape_flat"] and gap > -0.05:
            lever = "ks_margin (shape_flat only, gap > -0.05)"
        elif shape_only and not mf:
            lever = "no Michel found, profile unreadable"
        elif bits & B["plateau_off_mip"]:
            lever = "plateau scale (doc 21 sec 5: no window separates)"
        else:
            lever = "other"
        nz = N["hd"].get(k)
        print(f"  {int(d['apa_major']):3d} {k:14s} {v:11s} {str(kind):9s} {names(bits):23s} {mf:2d} {conn:4d} {float(d['michel_ke_best']):6.1f} {float(d['michel_len']):5.1f} "
              f"{float(d['muon_len']):5.0f} {float(d['plateau_med'])/Php['mip']:8.2f} {ce:5.2f} {float(d['ks_mu']):6.3f} {float(d['ks_flat']):7.3f} {gap:6.3f} "
              f"{int(d['n_dead_pts']):4d}/{int(d['n_profile_pts']):4d} {nz[1] if nz else float('nan'):8.3f}  {lever}")

    # ---- 6. why ks_margin saturates: what holds the THRU items shape_flat rejects ----------
    print("\n==== 6. THRU items with shape_flat in their pre-P1 bits: accepted if shape_flat never fired? ====")
    for lab, its, P0 in (("PDHD APA0 strict", hd["strict"], Php), ("PDHD all four APAs", hd["all"], Php),
                         ("PDVD", vd, Pvd)):
        thru = [x for x in its if x[1] == "THRU"]
        P2 = dict(P0); P2["ks_margin"] = -10.0            # shape_flat can never fire
        sf = [x for x in thru if reverdict(x[4], P0)[0] & B["shape_flat"]]
        held = collections.Counter(names(reverdict(x[4], P2)[1]) for x in sf)
        acc = sum(1 for x in sf if reverdict(x[4], P2)[1] == 0)
        gaps = [float(x[4]["ks_flat"]) - float(x[4]["ks_mu"]) for x in sf]
        print(f"  {lab}: THRU {len(thru)}, carrying shape_flat {len(sf)}, accepted with shape_flat removed {acc}; "
              f"ks_gap {min(gaps):.3f} .. {max(gaps):.3f}; unmodelled {sum(1 for x in sf if not modelled(x[4]))}")
        for k, n in held.most_common(): print(f"      still set: {k:45s} {n}")


if __name__ == "__main__":
    main()
