#!/usr/bin/env python3
"""doc pdvd/106 sec 2-6 -- how the two trajectory levers move the PDHD muon's END (read-only, no chain run).

Doc 104 sec 3 showed, at point level, that some of A1's Michel false positives are muon charge that A1 truncated
off its own chain and re-labelled.  It showed that on FIVE hand-picked cases.  This script asks whether that is a
property of the levers or a tail of a few events, over every STM candidate in all 61 PDHD events, WITHOUT using
any hand label -- every number here is arm-vs-arm geometry, so the scan-provenance question (were smx27/smx28
labelled on the old fit?) cannot touch it.

THE FOUR ARMS (all on disk, 61/61 events each, same pctrees so (event, cluster_id) is a stable join -- doc 101
sec 6.5 "idset_compare.txt"; doc 102 built its data arms on the doc 101 pctrees):
    A0  d101hnew   production: 'stepped' sampler, no fit knobs
    K   d101hkf    fit knobs only      (fit_weight_pow 1.5 + assoc_cont_center 1)
    S   d102hocs   retile sampler only ('charge_stepped')
    A1  d102hcs    both levers -- the cell the doc 103 grade calls A1
Doc 103's per-lever Michel purity is K 0.880 / S 0.906 / A1 0.895 against A0 0.946: NOT additive, "both" beats
"fit knobs alone".  So all three levers are reported against A0 separately and no additivity is assumed.

WHAT THE BRANCHES MEAN (CheckSTM_Michel.cxx):
    tagger_stop_pt  :1558  the STM tagger's own fit end, fx/fy/fz[istop]
    stop_pt         :1559  initialised TO tagger_stop_pt, then refined to the graph stop vertex at :2908/:3069
    stop_dis        :2909  |stop_pt - tagger_stop_pt|, so it is 0 for a record that never reached the refinement
    muon_len        :3085  prof.total_length -- the profiled chain's length, struct default 0 (:1152)
    n_retreat/retreat_len :2975  chain segments retreated off the fit's far end (doc pdvd/57) -- the code's OWN
                          name for the mechanism doc 104 sec 3 measured by hand
    contrast        StmMichelFunctions.cxx:337  tail_med/plateau_med in FIXED residual-range windows
    expected        :339   the muon dQ/dx model at the same rr, so it barely moves -> (contrast - expected) is a
                          damage measure.  BUT short_track (:320) halves the plateau window, so its flips are
                          tracked separately or the comparison changes meaning underneath you.

TWO TRAPS IN THE CLASSIFIER ITSELF, both found by checking a case instead of trusting a bin (sec 4):
  A. END SWAP.  029107_16/46 has A1's stop 1.78 cm from A0's ENTRY and A1's entry 1.16 cm from A0's stop: the same
     track, read in the opposite direction.  A forward projection scores that as "truncated by 264.76 cm", which is
     meaningless.  A swap is a worse defect than truncation -- it puts the Bragg peak at the wrong end -- so it is
     detected first and reported as its own class.
  B. AN ADVANCE IS INVISIBLE TO A FORWARD PROJECTION.  Projecting A1's stop onto A0's chain bounds the arc by
     muon_len(A0), so the offset can never be positive and "advanced" would always read 0.  029107_19/111 (doc 104
     sec 3.1: "its muon grew", 108.6 -> 121.8) projects to arc = 108.60 = exactly muon_len(A0) with perp 13.88 and
     would be binned "off-trajectory".  Advance is therefore measured by the REVERSE projection, A0's stop onto the
     lever's chain.  Both directions are computed for every pair.

TWO TRAPS IN THE POPULATION:
  1. DO NOT intersect all four arms.  The candidate set moves (doc 101 sec 6.5: PDHD 341->320, 70 ids lost, 49
     gained, only 10 of 61 events with an identical id set).  A four-way intersection keeps exactly the clusters
     the levers did not disturb -- it is selected AGAINST the effect and would report "no truncation".  Each lever
     is compared on its OWN pairwise intersection with A0, and lost/gained are reported as strata, not dropped.
  2. muon_len is 0 for a candidate that early-rejects before :3085.  Differencing 0 against a real length
     fabricates a 100 % truncation.  Pairs need muon_len > 0 in BOTH arms; the per-arm drop counts are printed
     because an asymmetric drop is itself a result.

    python3 d106_end_geometry.py > figs/106_end_geometry.txt
"""
import glob, os, sys
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
DET = "pdhd"
ARMS = [("A0", "d101hnew"), ("K", "d101hkf"), ("S", "d102hocs"), ("A1", "d102hcs")]
LEVERS = ["K", "S", "A1"]
STEP_CM = 0.6          # role-1 profile sampling; the nearest-vertex projection is exact to half of this
PERP_CM = 0.6          # one PDHD wire pitch -- the same scale doc 104 sec 3 used for "on the chain"
ARC_CM = 3.0           # below this an end move is not called a truncation or an advance
SWAP_CM = 10.0         # ends this close to the OTHER arm's opposite end = the chain was read backwards.  10 cm,
                       # not 1, because chain[0] is a median 0.67 / p90 2.0 / max 8.4 cm from entry_pt (below).

REC = ["cluster_id", "is_stm", "michel_found", "muon_len", "n_profile_pts", "n_chain_segs",
       "stop_x", "stop_y", "stop_z", "tagger_stop_x", "tagger_stop_y", "tagger_stop_z",
       "entry_x", "entry_y", "entry_z", "stop_dis", "n_retreat", "retreat_len", "n_split", "split_len",
       "plateau_med", "tail_med", "contrast", "contrast_expected", "bragg_valid", "short_track", "has_pass"]


def load(arm):
    """(records, chains) keyed '<run>_<evt>/<cid>'.  chains are the role-1 profile points, ordered entry->stop."""
    rec, chain = {}, {}
    for d in sorted(glob.glob(f"{IMG}/{DET}/work/*_{arm}")):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            f = uproot.open(f"{d}/tracking-pr.root")
            t = f["T_stm_michel"].arrays(REC, library="np")
            p = f["T_stm_michel_pts"].arrays(["cluster_id", "role", "L", "x", "y", "z"], library="np")
        except Exception:
            continue
        for i in range(len(t["cluster_id"])):
            k = f"{ev}/{int(t['cluster_id'][i])}"
            rec[k] = {c: float(t[c][i]) for c in REC if c != "cluster_id"}
            m = (p["cluster_id"] == int(t["cluster_id"][i])) & (p["role"] == 1)
            if m.sum() >= 2:
                o = np.argsort(p["L"][m])            # L is arc length from the entry; role 1 is already ordered
                chain[k] = (np.stack([p["x"][m][o], p["y"][m][o], p["z"][m][o]], 1), p["L"][m][o])
    return rec, chain


def project(pt, ch):
    """Nearest vertex of chain ch to pt -> (arc length along ch, perpendicular distance).  Quantised to STEP_CM/2."""
    xyz, L = ch
    d = np.linalg.norm(xyz - np.asarray(pt), axis=1)
    j = int(np.argmin(d))
    return float(L[j]), float(d[j])


def geom(a0, rl, c0, cl):
    """Full end geometry of one pair.  Returns dict with the swap test and BOTH projections.

    fwd  = lever's stop onto A0's chain   -> off_f < 0 means the stop retreated along A0's track (truncation)
    rev  = A0's stop onto the lever chain -> off_r < 0 means the lever's chain runs PAST A0's stop (advance)
    """
    s0 = np.array([a0["stop_" + c] for c in "xyz"]); e0 = np.array([a0["entry_" + c] for c in "xyz"])
    s1 = np.array([rl["stop_" + c] for c in "xyz"]); e1 = np.array([rl["entry_" + c] for c in "xyz"])
    g = {"dstop": float(np.linalg.norm(s1 - s0)), "dentry": float(np.linalg.norm(e1 - e0)),
         "s1e0": float(np.linalg.norm(s1 - e0)), "e1s0": float(np.linalg.norm(e1 - s0))}
    g["arc_f"], g["perp_f"] = project(s1, c0)
    g["off_f"] = g["arc_f"] - a0["muon_len"]
    g["arc_r"], g["perp_r"] = project(s0, cl)
    g["off_r"] = g["arc_r"] - rl["muon_len"]
    # The swap test uses the record's OWN entry/stop branches, not the profile polyline, because the two do not
    # share an origin: measured over 104 A0 candidates, |chain[-1] - stop_pt| is 0.000 at p50 AND p90 (the profile
    # ends exactly at the stop), but |chain[0] - entry_pt| is 0.67 at p50, 2.01 at p90 and 8.38 at most -- the
    # profile starts AFTER the entry point.  A perpendicular test near arc 0 therefore measures against the wrong
    # reference and silently drops real swaps; an earlier version of this script lost 3 of 4 that way.
    g["swap"] = ((g["s1e0"] + g["e1s0"] < g["dstop"] + g["dentry"])
                 and max(g["s1e0"], g["e1s0"]) < SWAP_CM and g["dstop"] > ARC_CM)
    return g


def classify(g):
    """Priority cascade; each pair lands in exactly one class.  Order matters -- see traps A and B."""
    if g["swap"]:
        return "swap"
    if g["perp_f"] < PERP_CM and g["off_f"] < -ARC_CM:
        return "trunc"
    if g["perp_r"] < PERP_CM and g["off_r"] < -ARC_CM:
        return "adv"
    if g["dstop"] <= ARC_CM:
        return "same"
    return "off"


def q(v, ps=(10, 25, 50, 75, 90)):
    if not len(v):
        return "        --        "
    return " ".join(f"{np.percentile(v, p):7.2f}" for p in ps)


def dist(a, b, pre):
    return float(np.linalg.norm([a[pre + c] - b[pre + c] for c in "xyz"]))


def main():
    D = {lab: load(arm) for lab, arm in ARMS}
    for lab, arm in ARMS:
        n = len(D[lab][0])
        print(f"# {lab:2s} = {arm:9s}  candidates {n:4d}   with a profile {len(D[lab][1]):4d}"
              f"   muon_len>0 {sum(1 for r in D[lab][0].values() if r['muon_len'] > 0):4d}")
    A0, C0 = D["A0"]

    # ---- V1: the five doc 104 sec 3.1 cases must come back with their published A0 -> A1 lengths -------------
    V1 = {"029107_28/109": (595.9, 544.8), "029107_27/39": (62.1, 40.2), "029107_19/111": (108.6, 121.8)}
    A1r = D["A1"][0]
    for k, (l0, l1) in V1.items():
        g0, g1 = A0[k]["muon_len"], A1r[k]["muon_len"]
        assert abs(g0 - l0) < 0.1 and abs(g1 - l1) < 0.1, f"V1 FAILED {k}: got {g0:.1f}->{g1:.1f} want {l0}->{l1}"
    print(f"# V1 ok: doc 104 sec 3.1 lengths reproduce for {len(V1)} cases from these arms")

    print("\n\n=== 1. candidate churn: each lever against A0, pairwise (never a four-way intersection) ===")
    print(f"{'lever':6s} {'|A0|':>5s} {'|L|':>5s} {'shared':>7s} {'lost':>6s} {'(is_stm)':>9s} {'(michel)':>9s}"
          f" {'gained':>7s} {'(is_stm)':>9s}  lost muon_len p50/p90")
    for lv in LEVERS:
        R = D[lv][0]
        lost = [k for k in A0 if k not in R]
        gain = [k for k in R if k not in A0]
        ll = [A0[k]["muon_len"] for k in lost if A0[k]["muon_len"] > 0]
        print(f"{lv:6s} {len(A0):5d} {len(R):5d} {len(set(A0) & set(R)):7d} {len(lost):6d}"
              f" {sum(1 for k in lost if A0[k]['is_stm'] > 0):9d} {sum(1 for k in lost if A0[k]['michel_found'] > 0):9d}"
              f" {len(gain):7d} {sum(1 for k in gain if R[k]['is_stm'] > 0):9d}"
              f"   {np.median(ll) if ll else -1:7.1f} {np.percentile(ll, 90) if ll else -1:7.1f}")
    print("  A cluster that VANISHES is a worse end defect than one that shortens: doc 101 sec 6.5 checked that no")
    print("  lost id is fitted anywhere in the knob-on file, so these are gone, not renumbered.")

    pairs = {}
    print("\n\n=== 2. paired length change, on each lever's own intersection with A0, muon_len>0 in both ===")
    print(f"{'lever':6s} {'pairs':>6s} {'drop0':>6s} {'dropL':>6s} | ratio L/A0  p10     p25     p50     p75     p90"
          f" | {'sum ratio':>9s} {'d<-20cm':>8s} {'d<-5cm':>7s} {'d>+5cm':>7s}")
    for lv in LEVERS:
        R = D[lv][0]
        sh = sorted(set(A0) & set(R))
        d0 = sum(1 for k in sh if A0[k]["muon_len"] <= 0)
        dL = sum(1 for k in sh if R[k]["muon_len"] <= 0)
        ks = [k for k in sh if A0[k]["muon_len"] > 0 and R[k]["muon_len"] > 0]
        pairs[lv] = ks
        rat = np.array([R[k]["muon_len"] / A0[k]["muon_len"] for k in ks])
        dl = np.array([R[k]["muon_len"] - A0[k]["muon_len"] for k in ks])
        s0 = sum(A0[k]["muon_len"] for k in ks)
        print(f"{lv:6s} {len(ks):6d} {d0:6d} {dL:6d} |            {q(rat)} | "
              f"{sum(R[k]['muon_len'] for k in ks) / s0:9.3f} {int((dl < -20).sum()):8d} {int((dl < -5).sum()):7d}"
              f" {int((dl > 5).sum()):7d}")
    print("  drop0/dropL = shared ids whose muon_len is 0 in A0 / in the lever (early reject before :3085).")

    print("\n\n=== 3. where the move is: displacement of each end (cm), paired, p50 / p90 ===")
    print(f"{'lever':6s} | {'|d stop|':>17s} | {'|d tagger_stop|':>17s} | {'|d entry|':>17s} |"
          f" {'stop_dis A0':>12s} {'stop_dis L':>11s}")
    for lv in LEVERS:
        R, ks = D[lv][0], pairs[lv]
        ds = np.array([dist(R[k], A0[k], "stop_") for k in ks])
        dt = np.array([dist(R[k], A0[k], "tagger_stop_") for k in ks])
        de = np.array([dist(R[k], A0[k], "entry_") for k in ks])
        print(f"{lv:6s} | {np.median(ds):7.2f} {np.percentile(ds, 90):9.2f} | {np.median(dt):7.2f} "
              f"{np.percentile(dt, 90):9.2f} | {np.median(de):7.2f} {np.percentile(de, 90):9.2f} |"
              f" {np.median([A0[k]['stop_dis'] for k in ks]):12.2f} {np.median([R[k]['stop_dis'] for k in ks]):11.2f}")

    G = {lv: {k: geom(A0[k], D[lv][0][k], C0[k], D[lv][1][k])
              for k in pairs[lv] if k in C0 and k in D[lv][1]} for lv in LEVERS}

    # V3: sections 2 and 7 quote len(pairs[lv]); sections 4/7 quote len(G[lv]).  They must be the SAME population or
    # the doc reports two denominators as one.  G additionally needs a role-1 chain in both arms, so this asserts
    # that every muon_len>0 record has a profile -- structural, not a coincidence of these arms.
    for lv in LEVERS:
        assert set(G[lv]) == set(pairs[lv]), \
            f"V3 FAILED {lv}: pairs {len(pairs[lv])} vs chains {len(G[lv])}, missing {set(pairs[lv]) - set(G[lv])}"
    print("# V3 ok: every muon_len>0 pair has a role-1 chain in both arms, so all sections share one denominator")

    # self-check: the cascade must not hide a pair that satisfies BOTH end tests (chains would have to cross)
    for lv in LEVERS:
        both = [k for k, g in G[lv].items()
                if not g["swap"] and g["perp_f"] < PERP_CM and g["off_f"] < -ARC_CM
                and g["perp_r"] < PERP_CM and g["off_r"] < -ARC_CM]
        assert not both, f"V2 FAILED {lv}: truncated AND advanced and not a swap, cascade order decides it: {both}"
    print("# V2 ok: outside the swap class no pair fires both end tests, so the cascade order is not a hidden choice")

    print("\n\n=== 4. DIRECTED: which end moved, and did it move ALONG the old track? (the headline) ===")
    print(f"  perpendicular offset of the lever's stop from A0's own chain, all pairs (cm): p25/p50/p75/p90")
    for lv in LEVERS:
        v = np.array([g["perp_f"] for g in G[lv].values()])
        print(f"    {lv:3s} n={len(v):4d}  {q(v, (25,50,75,90))}   <{PERP_CM} cm: {int((v<PERP_CM).sum())}"
              f" ({(v<PERP_CM).mean():.2f})")
    print(f"\n  classes (one per pair, priority: swap > truncated > advanced > same end > off-trajectory)")
    print(f"{'lever':6s} {'pairs':>6s} | {'swap':>8s} {'truncated':>11s} {'advanced':>10s} {'same end':>10s}"
          f" {'off-traj':>10s} | trunc depth p50/p90/max | adv p50/p90")
    for lv in LEVERS:
        c = {x: [] for x in ("swap", "trunc", "adv", "same", "off")}
        for k, g in G[lv].items():
            c[classify(g)].append(k)
        n = len(G[lv])
        td = np.array([-G[lv][k]["off_f"] for k in c["trunc"]])
        ad = np.array([-G[lv][k]["off_r"] for k in c["adv"]])
        print(f"{lv:6s} {n:6d} | {len(c['swap']):4d} {len(c['swap'])/n:4.2f}"
              f" {len(c['trunc']):6d} {len(c['trunc'])/n:4.2f} {len(c['adv']):5d} {len(c['adv'])/n:4.2f}"
              f" {len(c['same']):5d} {len(c['same'])/n:4.2f} {len(c['off']):5d} {len(c['off'])/n:4.2f} | "
              + (f"{np.median(td):7.2f} {np.percentile(td,90):7.2f} {td.max():7.2f}" if len(td) else "      --      ")
              + " | " + (f"{np.median(ad):6.2f} {np.percentile(ad,90):6.2f}" if len(ad) else "   --"))
    print("  'off-trajectory' is NOT necessarily a worse fit: these levers move the fit everywhere, so a stop that")
    print("  sits >1 wire pitch off A0's polyline may simply be the same end on a slightly different curve.")

    print("\n\n=== 4c. reconciling sec 2 with sec 4: WHERE the net shortening lives ===")
    print("  Sec 2 says A1 shortens 5:1 (107 pairs below -5 cm vs 21 above +5).  Sec 4 says truncation and advance")
    print("  are nearly balanced.  Both are true only if the net loss sits OUTSIDE the two end classes -- so here is")
    print("  the signed length change per class, which is instrument-free (no projection enters it).")
    print(f"{'lever':6s} {'class':>10s} {'n':>5s} {'d muon_len p50':>15s} {'p10':>8s} {'p90':>8s} {'sum d (cm)':>12s}"
          f" {'share of net':>13s}")
    for lv in LEVERS:
        R = D[lv][0]
        tot = sum(R[k]["muon_len"] - A0[k]["muon_len"] for k in G[lv])
        for cn in ("swap", "trunc", "adv", "same", "off"):
            ks = [k for k, g in G[lv].items() if classify(g) == cn]
            if not ks:
                continue
            dl = np.array([R[k]["muon_len"] - A0[k]["muon_len"] for k in ks])
            print(f"{lv:6s} {cn:>10s} {len(ks):5d} {np.median(dl):15.2f} {np.percentile(dl,10):8.2f}"
                  f" {np.percentile(dl,90):8.2f} {dl.sum():12.1f} {dl.sum()/tot if tot else 0:13.2f}")
        print(f"{lv:6s} {'ALL':>10s} {len(G[lv]):5d} {'':15s} {'':8s} {'':8s} {tot:12.1f} {1.0:13.2f}")

    print("\n\n=== 4d. is the 'same end' shortening zig-zag removal?  (local arc/chord over a 6 cm window) ===")
    print("  A chain that loses length while BOTH ends stay put has to be getting straighter.  Local arc/chord is")
    print("  the wiggle: over a window of WIN points it is (arc along the chain) / (straight distance end to end),")
    print("  so 1.000 is a straight segment and larger is zig-zag.  Whole-track tortuosity is dominated by real")
    print("  multiple scattering, so it is reported beside the local number, not instead of it.")
    WIN = 10                                        # 10 x 0.6 cm = a 6 cm window

    def wiggle(ch):
        """(local arc/chord over WIN points, whole-track tortuosity) or None if the chain is too short."""
        xyz, L = ch
        if len(L) < WIN + 2:
            return None
        c = np.linalg.norm(xyz[WIN:] - xyz[:-WIN], axis=1)
        ok = c > 1e-6
        e2e = float(np.linalg.norm(xyz[-1] - xyz[0]))
        if not ok.sum() or e2e <= 1e-6:
            return None
        return float(np.median((L[WIN:] - L[:-WIN])[ok] / c[ok])), float(L[-1] / e2e)

    print(f"{'lever':6s} {'class':>10s} {'n':>5s} {'short':>6s} | {'local arc/chord A0':>18s} {'lever':>8s}"
          f" {'delta p50':>10s} {'paired d p50':>13s} | {'tortuosity A0':>13s} {'lever':>8s} {'delta':>9s}")
    for lv in LEVERS:
        for cn in ("same", "trunc", "adv", "off", "swap"):
            ks = [k for k, g in G[lv].items() if classify(g) == cn]
            rows2, nshort = [], 0
            for k in ks:                       # a track enters ONLY if BOTH arms give a usable chain (no silent drop)
                wa, wb = wiggle(C0[k]), wiggle(D[lv][1][k])
                if wa is None or wb is None:
                    nshort += 1
                    continue
                rows2.append((wa, wb))
            if not rows2:
                continue
            a0w = np.array([r[0][0] for r in rows2]); lvw = np.array([r[1][0] for r in rows2])
            a0t = np.array([r[0][1] for r in rows2]); lvt = np.array([r[1][1] for r in rows2])
            print(f"{lv:6s} {cn:>10s} {len(rows2):5d} {nshort:6d} | {np.median(a0w):18.4f} {np.median(lvw):8.4f}"
                  f" {np.median(lvw) - np.median(a0w):10.4f} {np.median(lvw - a0w):13.4f}"
                  f" | {np.median(a0t):13.4f} {np.median(lvt):8.4f} {np.median(lvt - a0t):9.4f}")
    print("  'short' = tracks excluded because one arm's chain is under 12 points; they are counted, never dropped.")
    print("  'paired d p50' is the median of the PER-TRACK difference, which is the one to read: a difference of")
    print("  medians can hide a shift that every track shares.")
    print("\n  CLOSURE: if straightening is the WHOLE story for the 'same end' class, then per track")
    print("  d(muon_len) should equal d(tortuosity) x (end-to-end distance).  Predicted vs observed, p50:")
    for lv in LEVERS:
        pr, ob = [], []
        for k, g in G[lv].items():
            if classify(g) != "same":
                continue
            wa, wb = wiggle(C0[k]), wiggle(D[lv][1][k])
            if wa is None or wb is None:
                continue
            e2e = float(np.linalg.norm(C0[k][0][-1] - C0[k][0][0]))
            pr.append((wb[1] - wa[1]) * e2e)
            ob.append(D[lv][0][k]["muon_len"] - A0[k]["muon_len"])
        pr, ob = np.array(pr), np.array(ob)
        res = ob - pr
        print(f"    {lv:3s} n={len(pr):4d}  predicted {np.median(pr):7.2f} cm   observed {np.median(ob):7.2f} cm"
              f"   unexplained residual p50 {np.median(res):7.2f} cm  (p10 {np.percentile(res,10):6.2f},"
              f" p90 {np.percentile(res,90):6.2f})")

    print("\n\n=== 4b. the end swaps: the same track read backwards (worst class, Bragg peak at the wrong end) ===")
    for lv in LEVERS:
        sw = [k for k, g in G[lv].items() if g["swap"]]
        print(f"  {lv:3s}: {len(sw)} -> " + (", ".join(sw) if sw else "none"))
    print("\n  STABILITY of the attribution against SWAP_CM (the class is defined by a constant, so it must be")
    print("  shown that S > K is not an artifact of choosing 10 cm).  Reversal count per lever:")
    print(f"    {'SWAP_CM':>8s} " + " ".join(f"{lv:>5s}" for lv in LEVERS))
    for cm in (3.0, 5.0, 10.0, 15.0, 25.0):
        row = []
        for lv in LEVERS:
            row.append(sum(1 for g in G[lv].values()
                           if (g["s1e0"] + g["e1s0"] < g["dstop"] + g["dentry"])
                           and max(g["s1e0"], g["e1s0"]) < cm and g["dstop"] > ARC_CM))
        print(f"    {cm:8.0f} " + " ".join(f"{n:5d}" for n in row) + ("   <- used" if cm == SWAP_CM else ""))

    sw1 = [(k, g) for k, g in G["A1"].items() if g["swap"]]
    if sw1:
        print(f"  {'key':>16s} {'len A0':>8s} {'len A1':>8s} {'arc_f':>7s} {'perp_f':>7s} {'|s1-e0|':>8s} {'|e1-s0|':>8s}"
              f" {'mich A0':>8s} {'mich A1':>8s} {'stm A0':>7s} {'stm A1':>7s}")
        for k, g in sw1:
            print(f"  {k:>16s} {A0[k]['muon_len']:8.1f} {D['A1'][0][k]['muon_len']:8.1f} {g['arc_f']:7.2f}"
                  f" {g['perp_f']:7.2f} {g['s1e0']:8.2f} {g['e1s0']:8.2f}"
                  f" {int(A0[k]['michel_found']):8d} {int(D['A1'][0][k]['michel_found']):8d}"
                  f" {int(A0[k]['is_stm']):7d} {int(D['A1'][0][k]['is_stm']):7d}")

    print("\n\n=== 5. the code's own retreat bookkeeping (CheckSTM_Michel.cxx:2975, doc pdvd/57) ===")
    print(f"{'lever':6s} {'pairs':>6s} | {'n_retreat>0 A0':>14s} {'n_retreat>0 L':>13s} {'newly retreating':>16s}"
          f" | retreat_len L-A0 p50/p90 | {'n_split>0 A0':>12s} {'n_split>0 L':>11s}")
    for lv in LEVERS:
        R, ks = D[lv][0], pairs[lv]
        dr = np.array([R[k]["retreat_len"] - A0[k]["retreat_len"] for k in ks])
        print(f"{lv:6s} {len(ks):6d} | {sum(1 for k in ks if A0[k]['n_retreat'] > 0):14d}"
              f" {sum(1 for k in ks if R[k]['n_retreat'] > 0):13d}"
              f" {sum(1 for k in ks if R[k]['n_retreat'] > 0 and A0[k]['n_retreat'] == 0):16d}"
              f" | {np.median(dr):11.2f} {np.percentile(dr, 90):8.2f} |"
              f" {sum(1 for k in ks if A0[k]['n_split'] > 0):12d} {sum(1 for k in ks if R[k]['n_split'] > 0):11d}")
    print("  If retreat_len barely moves, the doc-57 retreat step is NOT the site that shortens these muons, and a")
    print("  fix aimed at it would miss.  The shortening happens upstream, in the fit and the chain it builds.")

    print("\n\n=== 6. does the end move corrupt the stopping dQ/dx?  (contrast = tail_med/plateau_med) ===")
    print(f"{'lever':6s} {'both valid':>10s} | {'bragg lost':>10s} {'gained':>7s} | {'short_track flips':>17s}"
          f" | {'d(contrast) p50':>15s} | {'d(contrast-exp) p50/p90':>23s} | {'plateau_med rel p50/p90':>23s}")
    for lv in LEVERS:
        R, ks = D[lv][0], pairs[lv]
        bv = [k for k in ks if A0[k]["bragg_valid"] > 0 and R[k]["bragg_valid"] > 0 and A0[k]["plateau_med"] > 0]
        dc = np.array([R[k]["contrast"] - A0[k]["contrast"] for k in bv])
        dx = np.array([(R[k]["contrast"] - R[k]["contrast_expected"])
                       - (A0[k]["contrast"] - A0[k]["contrast_expected"]) for k in bv])
        dp = np.array([R[k]["plateau_med"] / A0[k]["plateau_med"] - 1.0 for k in bv])
        print(f"{lv:6s} {len(bv):10d} | {sum(1 for k in ks if A0[k]['bragg_valid']>0 and R[k]['bragg_valid']==0):10d}"
              f" {sum(1 for k in ks if A0[k]['bragg_valid']==0 and R[k]['bragg_valid']>0):7d}"
              f" {sum(1 for k in ks if A0[k]['short_track'] != R[k]['short_track']):17d}"
              f" | {np.median(dc):15.4f} | {np.median(dx):11.4f} {np.percentile(dx,90):11.4f}"
              f" | {np.median(dp):11.4f} {np.percentile(dp,90):11.4f}")
    print("  contrast is the Bragg SHAPE (a ratio, self-normalising); plateau_med rel is the absolute dQ/dx SCALE.")
    print("  They answer different questions -- docs pdhd/16, 17, 29, 50 calibrate on the scale.")

    print("\n\n=== 7. split by whether the lever tagged a Michel (reco flag, no hand label involved) ===")
    print(f"{'lever':6s} {'stratum':>18s} {'n':>5s} | {'swap':>7s} {'trunc':>10s} {'adv':>9s}"
          f" | trunc depth p50/p90 | d muon_len p50/p10")
    for lv in LEVERS:
        R = D[lv][0]
        for nm, sel in (("michel in L", lambda k: R[k]["michel_found"] > 0),
                        ("no michel in L", lambda k: R[k]["michel_found"] == 0),
                        ("is_stm in A0", lambda k: A0[k]["is_stm"] > 0)):
            ks = [k for k in G[lv] if sel(k)]
            cl = [classify(G[lv][k]) for k in ks]
            td = np.array([-G[lv][k]["off_f"] for k, c in zip(ks, cl) if c == "trunc"])
            dl = np.array([R[k]["muon_len"] - A0[k]["muon_len"] for k in ks])
            n = max(len(ks), 1)
            print(f"{lv:6s} {nm:>18s} {len(ks):5d} | {cl.count('swap'):3d} {cl.count('swap')/n:4.2f}"
                  f" {cl.count('trunc'):5d} {cl.count('trunc')/n:4.2f} {cl.count('adv'):4d} {cl.count('adv')/n:4.2f} | "
                  + (f"{np.median(td):8.2f} {np.percentile(td,90):8.2f}" if len(td) else "      --        ")
                  + f"  | {np.median(dl) if len(dl) else -9:8.2f} {np.percentile(dl,10) if len(dl) else -9:8.2f}")

    print("\n\n=== 8. every A1 pair that truncated, swapped or advanced (the tail doc 104 sec 3 sampled by hand) ===")
    R = D["A1"][0]
    rowsx = []
    for k, g in G["A1"].items():
        c = classify(g)
        if c in ("trunc", "swap", "adv"):
            rowsx.append((c, -g["off_f"] if c == "trunc" else (-g["off_r"] if c == "adv" else g["dstop"]), k, g))
    rowsx.sort(key=lambda r: -r[1])
    print(f"{'class':>6s} {'move':>8s} {'key':>16s} {'len A0':>8s} {'len A1':>8s} {'perp_f':>7s} {'|dstop|':>8s}"
          f" {'mich A0':>8s} {'mich A1':>8s} {'stm A0':>7s} {'stm A1':>7s}")
    for c, m, k, g in rowsx:
        print(f"{c:>6s} {m:8.2f} {k:>16s} {A0[k]['muon_len']:8.1f} {R[k]['muon_len']:8.1f} {g['perp_f']:7.2f}"
              f" {g['dstop']:8.2f} {int(A0[k]['michel_found']):8d} {int(R[k]['michel_found']):8d}"
              f" {int(A0[k]['is_stm']):7d} {int(R[k]['is_stm']):7d}")
    print(f"  ({len(rowsx)} of {len(G['A1'])} A1 pairs; doc 104 sec 3.1's five cases were hand-picked from this set)")


if __name__ == "__main__":
    main()
