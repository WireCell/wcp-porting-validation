#!/usr/bin/env python3
"""Is there ONE drift velocity + flash re-assignment that makes every PDVD
cathode-crossing pair land on a BRIGHT, cathode-light-consistent flash?

Builds on crosser_flash_velocity_evt298567.py.  For each crosser pair (apa 0 +
apa 4) the cathode-meeting geometry depends only on pi = t_flash * v, so at a
trial velocity v each flash meets at a definite distance d(v); the LIGHT (the
measured cathode X-ARAPUCA pattern vs the ~v-invariant predicted template)
breaks the (t,v) degeneracy.  Cathode KS is v-INDEPENDENT (a per-flash quantity),
so it is precomputed once and the velocity scan only recomputes the cheap
KDTree meeting.

For a grid of trial velocities we assign each pair to its best meeting flash
(d <= D_CUT, |x_mid| <= XMID_CUT) minimizing cathode KS, and score how many
pairs land on a bright (cathode PE >= PE_MIN) light-consistent (KS <= KS_MAX)
flash.  A common velocity, if it exists, maximizes that count across pairs.

Analysis only.  A velocity change is stop-and-ask; findings are reported, not
applied.  Run from pdvd/:
    python3 ql_light_calib/crosser_common_velocity.py [calib.json ...]
Default input: the evt298567 _vcal dump (ql_scan port-5017 display).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PDVD, "ql_display"))
sys.path.insert(0, HERE)
import find_crossers as fc
from ablib_gold import GridLib, predict, ks_dis, VUV_EFF

LIB = os.path.join(PDVD, "photlib", "pdvd-photlib-vis-v5-175nm.json")
CATH = list(range(4, 12))
WIN_US = 200.0
D_CUT = 25.0
XMID_CUT = 15.0
PE_MIN = 300.0      # "bright" cathode
KS_MAX = 0.40       # cathode-pattern consistent (model flatness residual ~0.35)
V_SCAN = np.linspace(0.150, 0.166, 161)   # -5.4% .. +4.7% of 0.1586


def meeting(c0, c4, t, v):
    rel = (c0.sign_offset - c4.sign_offset) * t * v
    Q = c0.P.copy(); Q[:, 0] += rel
    dist, jn = c4.tree3.query(Q, k=1)
    i = int(np.argmin(dist))
    x0 = c0.P[i, 0] + c0.sign_offset * t * v
    x4 = c4.P[jn[i], 0] + c4.sign_offset * t * v
    return float(dist[i]), 0.5 * (x0 + x4)


def best_pi(c0, c4, t_nom, v_nom):
    """product pi* = t*v that minimizes the pair meeting distance d
    (coarse scan then local refine; d(pi) is smooth)."""
    def scan(lo, hi, n):
        grid = np.linspace(lo, hi, n)
        ds = [meeting(c0, c4, t_nom, pi / t_nom)[0] for pi in grid]
        return grid, np.array(ds)
    g, ds = scan(t_nom * v_nom * 0.85, t_nom * v_nom * 1.15, 61)
    i = int(np.argmin(ds))
    lo = g[max(0, i - 1)]; hi = g[min(len(g) - 1, i + 1)]
    g2, ds2 = scan(min(lo, hi), max(lo, hi), 41)
    return float(g2[int(np.argmin(ds2))])


def cath_pred(d, gid, apa, uid, lib, live):
    b = {"apa": apa, "flash_gid": gid, "main_cluster": uid, "other_clusters": []}
    try:
        return predict(d, b, lib, live)
    except KeyError:
        return np.zeros(len(VUV_EFF))


def cath_ks(meas, tmpl, live):
    """cathode-only KS, dropping channels the per-flash saturation veto zeroed
    (measured ~0 where the template predicts substantial light -- e.g. gid61's
    railed opdet 10). Prior doc: bright-module cathode streams can veto to 0."""
    idx = [j for j in CATH if live[j] and not (meas[j] < 1.0 and tmpl[j] > 30.0)]
    if not idx:
        return 1.0
    m, p = meas[np.array(idx)], tmpl[np.array(idx)]
    if m.sum() <= 0 or p.sum() <= 0:
        return 1.0
    return float(np.abs(np.cumsum(m) / m.sum() - np.cumsum(p) / p.sum()).max())


def crosser_pairs(d):
    """reproduce find_crossers selection to get the (gid_pick, u0, u4) list."""
    import io
    import contextlib
    argv = sys.argv
    # call find_crossers.main via its selection is awkward; re-derive minimally
    best = {}
    for j, b in enumerate(d["bundles"]):
        uid = b["main_cluster"]
        if uid == 3999999 or uid not in d["cluster_by_uid"]:
            continue
        k = (b["flash_gid"], uid)
        if k not in best or b["ks_dis"] < d["bundles"][best[k]]["ks_dis"]:
            best[k] = j
    by_gid = {}
    for (gid, uid) in best:
        by_gid.setdefault(gid, {0: [], 4: []})[d["cluster_by_uid"][uid]["apa"]].append(uid)
    pair_gids = {}
    for gid, sides in by_gid.items():
        for u0 in sides[0]:
            for u4 in sides[4]:
                pair_gids.setdefault((u0, u4), []).append(gid)
    return pair_gids


def analyze(path):
    d = fc.load(path)
    for k in list(d["geometry"]):
        d["geometry"][str(k)] = d["geometry"][k]
    v_nom = d["drift_speed"]
    lib = GridLib(LIB)
    live = np.array([o["active"] and not o.get("auto_masked", False) for o in d["opdets"]])
    live_c = live.copy()
    live_c[[i for i in range(len(live)) if i not in CATH]] = False
    fby = d["flash_by_gid"]
    flashes = sorted(d["flashes"], key=lambda f: f["time"])
    ftime = np.array([f["time"] for f in flashes])
    fpe = {f["gid"]: np.array(f["pe"]) for f in d["flashes"]}
    event = os.path.basename(path)[len("calib-"):-len(".json")]

    # reproduce the display crosser pairs, keep those found by find_crossers cuts
    pair_gids = crosser_pairs(d)
    clus = {}
    def C(uid):
        if uid not in clus:
            clus[uid] = fc.Clus(d, uid)
        return clus[uid]

    pairs = []
    for (u0, u4), gids in pair_gids.items():
        c0, c4 = C(u0), C(u4)
        if c0.length < 50 or c4.length < 50:
            continue
        dyz_min, _ = c4.tree_yz.query(c0.P[:, 1:], k=1)
        if dyz_min.min() > 25.0:
            continue
        # pick display flash = eligible shared flash minimizing d at v_nom
        elig = []
        for gid in gids:
            dd, xm = meeting(c0, c4, fby[gid]["time"], v_nom)
            if abs(xm) <= 10.0:
                elig.append((dd, gid))
        if not elig:
            continue
        elig.sort()
        if elig[0][0] > D_CUT:
            continue
        gpick = elig[0][1]
        # angle gate (collinear) -- reuse find_crossers helper
        a_glob = fc.fold_angle(c0.gax, c4.gax)
        if not (a_glob <= 10.0):   # global-PCA collinearity (cheap, robust)
            continue
        pairs.append((gpick, u0, u4, c0, c4))

    print(f"\n########## {event}  (v_nom={v_nom:.4f}, {len(pairs)} crossers) ##########")

    # per-pair light table: for each candidate flash, its OWN-T0 predicted
    # cathode template + saturation-aware cathode KS (both v-INDEPENDENT).
    light = {}     # (gpick,u0,u4) -> (predCath_at_pick, {gid:(t,cathMeasPE,ks,predCathPE)})
    for (gpick, u0, u4, c0, c4) in pairs:
        tp = (cath_pred(d, gpick, 0, u0, lib, live)
              + cath_pred(d, gpick, 4, u4, lib, live))
        t_nom = fby[gpick]["time"]
        rows = {}
        for k in np.where(np.abs(ftime - t_nom) <= WIN_US)[0]:
            g = flashes[k]["gid"]; meas = fpe[g]
            tg = (cath_pred(d, g, 0, u0, lib, live)
                  + cath_pred(d, g, 4, u4, lib, live))       # template at THIS flash's T0
            rows[g] = (flashes[k]["time"], meas[CATH].sum(),
                       cath_ks(meas, tg, live), tg[CATH].sum())
        light[(gpick, u0, u4)] = (tp[CATH].sum(), rows)

    def assign(v):
        """for each pair, the flash that jointly satisfies bright (measCath>=
        PE_MIN) + cathode-pattern consistent (KS<=KS_MAX) + meeting (d<=D_CUT,
        |xmid|<=XMID_CUT); among those pick the smallest meeting distance.  All
        three gates are needed -- the flash field is dense, so no single gate is
        unique (dim flashes fluke small d or normalized KS)."""
        out = {}
        for (gpick, u0, u4, c0, c4) in pairs:
            _, rows = light[(gpick, u0, u4)]
            best = None
            for g, (t, cpe, ks, pce) in rows.items():
                if cpe < PE_MIN or ks > KS_MAX:
                    continue
                dd, xm = meeting(c0, c4, t, v)
                if dd <= D_CUT and abs(xm) <= XMID_CUT:
                    if best is None or dd < best[4]:
                        best = (g, t, ks, cpe, dd, xm, pce)
            out[(gpick, u0, u4)] = best
        return out

    def nfix(a):
        return sum(1 for b in a.values() if b)

    score = np.array([nfix(assign(v)) for v in V_SCAN])
    v_best = V_SCAN[int(np.argmax(score))]
    v_fr = 0.153      # field-response drift speed (user: FR computed at 1.53 mm/us)
    print(f"velocity scan: best common v = {v_best:.4f} ({(v_best/v_nom-1)*100:+.2f}%) "
          f"fixes {score.max()}/{len(pairs)}; nominal {v_nom:.4f} fixes "
          f"{nfix(assign(v_nom))}/{len(pairs)}; FR v=0.153 fixes {nfix(assign(v_fr))}/{len(pairs)}")

    # hand-scan self-check: the 4 KNOWN_298567 crossers must keep their picked flash
    if event == "evt298567":
        a = assign(v_nom)
        for (gpick, u0, u4, c0, c4) in pairs:
            if gpick in fc.KNOWN_298567:
                b = a[(gpick, u0, u4)]
                got = b[0] if b else None
                print(f"   self-check gid{gpick}: light-best=gid{got} "
                      f"{'OK' if got == gpick else 'DIFFERS'}")

    # production auto-select map: (flash_gid, uid) that QLMatching auto_selected
    autosel = {(b["flash_gid"], b["main_cluster"]) for b in d["bundles"]
               if b.get("auto_selected")}

    def anode_end_u(c, t, v):
        """anode-most u (min drift-coord from the anode) of a half at flash t --
        a full anode->cathode crosser has u_min ~ 0; a time/velocity error shifts
        it by v*dt.  Uses the half's own geometry."""
        g = d["geometry"][c.apa]
        u = g["s"] * (c.P[:, 0] + c.sign_offset * t * v - g["anode_x"])
        return float(np.min(u))

    # per-crosser: bright-consistent flash + implied velocity, uniqueness of the
    # match (n_rivals), whether production auto_selected the pick vs the match,
    # and the anode-end shift between pick and match (user's anode cross-check).
    a = assign(v_nom)
    recs = []
    print(f"\n  --- per-crosser bright match (nominal assign) ---")
    for (gpick, u0, u4, c0, c4) in pairs:
        t_nom = fby[gpick]["time"]
        _, rows = light[(gpick, u0, u4)]
        # n_rivals = distinct flashes passing bright+KS+meeting at nominal v
        rivals = 0
        for gg, (tt, cpe2, ks2, pce2) in rows.items():
            if cpe2 >= PE_MIN and ks2 <= KS_MAX:
                ddx, xmx = meeting(c0, c4, tt, v_nom)
                if ddx <= D_CUT and abs(xmx) <= XMID_CUT:
                    rivals += 1
        b = a[(gpick, u0, u4)]
        if not b:
            print(f"   c{u0}+c{u4} pick=gid{gpick}  (no bright cathode-consistent flash)")
            continue
        g, t, ks, cpe, dd, xm, pce = b
        pistar = best_pi(c0, c4, fby[g]["time"], v_nom)
        v_imp = pistar / t
        wrong = abs(t - t_nom) > 5.0
        pick_as = (gpick, u0) in autosel or (gpick, u4) in autosel
        match_as = (g, u0) in autosel or (g, u4) in autosel
        # anode-end shift: longer half, at the pick vs the matched flash
        cbig = c0 if c0.length >= c4.length else c4
        au_pick = anode_end_u(cbig, t_nom, v_nom)
        au_match = anode_end_u(cbig, t, v_nom)
        recs.append(dict(event=event, u0=u0, u4=u4, gpick=gpick, gmatch=g,
                         t_true=t, dt=t - t_nom, measCath=cpe, predCath=pce,
                         ks=ks, d_nom=dd, v_imp=v_imp, wrong=wrong,
                         len0=c0.length, len4=c4.length, n_rivals=rivals,
                         pick_autosel=pick_as, match_autosel=match_as,
                         au_pick=au_pick, au_match=au_match))
        print(f"   c{u0}+c{u4} pick=gid{gpick} -> gid{g} dt={t-t_nom:+5.0f} "
              f"measCath={cpe:6.0f} KS={ks:.3f} d@nom={dd:4.1f} "
              f"v_imp={v_imp:.4f} nrival={rivals} "
              f"pickAS={int(pick_as)} matchAS={int(match_as)} "
              f"anodeU pick={au_pick:+.0f}/match={au_match:+.0f} "
              f"{'WRONG-FLASH' if wrong else 'pick-ok'}")
    return recs


def summarize(all_recs, v_nom=0.1586):
    wf = [r for r in all_recs if r["wrong"]]
    ok = [r for r in all_recs if not r["wrong"]]
    print("\n\n================ RUN-WIDE SUMMARY ================")
    print(f"crossers total {len(all_recs)}; pick already on bright flash {len(ok)}; "
          f"WRONG-FLASH (pick dim, bright flash elsewhere) {len(wf)}")
    # uniqueness of the bright match (advisor: gates the "81%" claim)
    if wf:
        nr = np.array([r["n_rivals"] for r in wf])
        uniq = int((nr == 1).sum())
        print(f"  WRONG-FLASH match uniqueness: {uniq}/{len(wf)} have exactly ONE "
              f"bright+consistent+meeting flash (n_rivals: median {int(np.median(nr))}, "
              f"max {nr.max()})")
        # find_crossers pick vs production QLMatching auto_select
        pa = sum(1 for r in wf if r["pick_autosel"])
        ma = sum(1 for r in wf if r["match_autosel"])
        print(f"  of {len(wf)} wrong-flash: production QLMatching auto_selected the "
              f"DIM pick in {pa}, the BRIGHT match in {ma}")
        # anode-end shift pick->match (user's cross-check): a real time offset
        # would move it ~v*dt; report the spread
        dau = np.array([r["au_match"] - r["au_pick"] for r in wf])
        print(f"  anode-end u shift (match - pick): median {np.median(dau):+.1f} cm "
              f"(= v*dt); |anode_u| at match median "
              f"{np.median([abs(r['au_match']) for r in wf]):.0f} cm")
    for name, sub in (("ALL", all_recs), ("WRONG-FLASH", wf), ("pick-ok", ok)):
        if not sub:
            continue
        v = np.array([r["v_imp"] for r in sub])
        w = np.array([abs(r["t_true"]) for r in sub])
        # |t|-weighted mean/median of implied velocity
        order = np.argsort(v)
        vs, ws = v[order], w[order]
        cw = np.cumsum(ws)
        wmed = vs[np.searchsorted(cw, cw[-1] / 2)]
        wmean = float((v * w).sum() / w.sum())
        print(f"  {name:12s} n={len(sub):3d}  implied v: |t|-wtd mean {wmean:.4f} "
              f"({(wmean/v_nom-1)*100:+.2f}%)  |t|-wtd median {wmed:.4f}  "
              f"range [{v.min():.4f},{v.max():.4f}]")
    # dt of wrong-flash matches (is the time offset consistent?)
    if wf:
        dt = np.array([r["dt"] for r in wf])
        print(f"  WRONG-FLASH dt to pick: mean {dt.mean():+.0f} median "
              f"{np.median(dt):+.0f} us  range [{dt.min():+.0f},{dt.max():+.0f}]")
    # velocity-error vs time-offset: regress dt = (a-1)*t_pick + b.
    #   pure velocity error -> b~0, v_true = v_nom/(1+slope)
    #   pure constant time offset -> slope~0, b = the offset
    # exclude best_pi rail hits (|dv|>=14%: unreliable short/foul pairs)
    good = [r for r in all_recs if abs(r["v_imp"]/v_nom - 1) < 0.14]
    tp = np.array([r["t_true"] - r["dt"] for r in good])   # picked-flash time
    dd = np.array([r["dt"] for r in good])
    if len(tp) >= 3:
        A = np.vstack([tp, np.ones_like(tp)]).T
        (slope, inter), *_ = np.linalg.lstsq(A, dd, rcond=None)
        v_from_slope = v_nom / (1 + slope)
        print(f"\n  dt-vs-t_pick regression (n={len(tp)}, rail hits excluded):")
        print(f"    slope   {slope:+.4f}  -> velocity error reading v_true={v_from_slope:.4f} "
              f"({(v_from_slope/v_nom-1)*100:+.2f}%)")
        print(f"    intercept {inter:+.1f} us  -> constant-time-offset reading")
        # which model dominates? compare residual to slope-only and offset-only
        r_full = dd - (slope * tp + inter)
        r_off = dd - np.median(dd)               # constant offset only
        r_vel = dd - (slope * tp)                # velocity only (b=0)
        rms = lambda x: float(np.sqrt(np.mean(x**2)))
        print(f"    RMS resid: full {rms(r_full):.0f}  offset-only {rms(r_off):.0f}  "
              f"velocity-only {rms(r_vel):.0f} us")


def main():
    args = sys.argv[1:] or [os.path.join(PDVD, "work", "039252_0_vcal",
                                         "calib-evt298567.json")]
    all_recs = []
    for p in args:
        all_recs += analyze(p)
    if len(args) > 1:
        summarize(all_recs)
        csv = os.path.join(HERE, "crosser_flash_records.csv")
        cols = ["event", "u0", "u4", "gpick", "gmatch", "t_true", "dt",
                "measCath", "predCath", "ks", "d_nom", "v_imp", "wrong",
                "len0", "len4", "n_rivals", "pick_autosel", "match_autosel",
                "au_pick", "au_match"]
        with open(csv, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in all_recs:
                fh.write(",".join(str(r[c]) for c in cols) + "\n")
        print(f"\nwrote {csv} ({len(all_recs)} crossers)")


if __name__ == "__main__":
    main()
