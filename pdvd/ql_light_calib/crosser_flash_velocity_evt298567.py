#!/usr/bin/env python3
"""Are the low-PE cathode-crossing matches in evt298567 paired to the wrong
flash?  Test the user's hypothesis: the true flash is a nearby BRIGHTER one, and
the cathode-meeting geometry is restored by a drift-velocity change.

Physics.  A cathode crosser is two clusters (apa 0 bottom + apa 4 top) that at
the true T0 meet at the cathode (x~0).  The T0 correction is a rigid x-shift
sign_offset*t_flash*v per half, so the two halves' RELATIVE shift is -2*t*v and
their meeting geometry depends only on the product pi = t_flash * v (x_mid, the
common-mode, is v-independent).  Hence a different flash t_k reproduces the same
cathode meeting at v_k = pi*/t_k, i.e. dv/v = pi*/(t_k*v_nom) - 1 ~ -dt/|t|.  The
geometry is therefore DEGENERATE along (t,v); the LIGHT breaks it.

Because the PDVD photon library is a 10 cm trilinear grid, a ~50 us (~8 cm) T0
shift moves points < 1 cell, so the PREDICTED cathode pattern is ~invariant
across neighbor flashes (verified below).  A cathode crosser must produce STRONG
cathode light (X-ARAPUCAs sit on the cathode), so the correct flash is the one
whose MEASURED cathode pattern matches that prediction.

Per crosser pair we report, for every flash in a +/- window:
  measured total & cathode PE, cathode-KS vs the (invariant) predicted template,
  the meeting distance d and the velocity v_k / dv that restores cathode meeting.

Analysis only -- reads the _vcal dump, changes no config / matcher / velocity.
A velocity change is stop-and-ask; a wrong-looking number is report-don't-tune.

Run from pdvd/:  python3 ql_light_calib/crosser_flash_velocity_evt298567.py
Input: work/039252_0_vcal/calib-evt298567.json  (the ql_scan port-5017 display)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PDVD, "ql_display"))
sys.path.insert(0, HERE)
import find_crossers as fc          # Clus, load, closest-approach helpers
from ablib_gold import GridLib, predict, ks_dis, VUV_EFF

CAL = os.path.join(PDVD, "work", "039252_0_vcal", "calib-evt298567.json")
LIB = os.path.join(PDVD, "photlib", "pdvd-photlib-vis-v5-175nm.json")
CATH = list(range(4, 12))          # cathode X-ARAPUCA opdets (x~0), ablib CH_GROUP
WIN_US = 250.0                     # flash search half-window
V_NOM = None                       # filled from dump

# the 6 display crossers (pair u0/u4 at the picked gid) from find_crossers _vcal
PAIRS = [(24, 29, 4000091), (36, 107, 4000131), (42, 36, 4000062),
         (47, 5, 4000065), (61, 142, 4000057), (83, 81, 4000075)]


def meeting(c0, c4, t, v):
    """closest-approach d (cm) and cathode-plane midpoint x_mid at flash time t,
    drift v -- find_crossers geometry, parametrized by v."""
    rel = (c0.sign_offset - c4.sign_offset) * t * v
    Q = c0.P.copy()
    Q[:, 0] += rel
    dist, jn = c4.tree3.query(Q, k=1)
    i = int(np.argmin(dist))
    x0 = c0.P[i, 0] + c0.sign_offset * t * v
    x4 = c4.P[jn[i], 0] + c4.sign_offset * t * v
    return float(dist[i]), 0.5 * (x0 + x4)


def best_pi(c0, c4, t_nom, v_nom):
    """product pi* = t*v minimizing meeting d (scan around the nominal pick)."""
    pi_nom = t_nom * v_nom
    grid = pi_nom * np.linspace(0.90, 1.10, 401)
    ds = []
    for pi in grid:
        # hold t at t_nom, vary v = pi/t_nom so t*v = pi
        d, _ = meeting(c0, c4, t_nom, pi / t_nom)
        ds.append(d)
    return float(grid[int(np.argmin(ds))]), float(np.min(ds))


def cath_pred(d, gid, apa, uid, lib, live):
    b = {"apa": apa, "flash_gid": gid, "main_cluster": uid, "other_clusters": []}
    try:
        return predict(d, b, lib, live)
    except KeyError:
        return np.zeros(len(VUV_EFF))


def main():
    global V_NOM
    d = fc.load(CAL)
    # fc.load rewrites geometry to int keys; ablib.predict wants str keys -- keep both
    for k in list(d["geometry"]):
        d["geometry"][str(k)] = d["geometry"][k]
    V_NOM = d["drift_speed"]
    lib = GridLib(LIB)
    live = np.array([o["active"] and not o.get("auto_masked", False)
                     for o in d["opdets"]])
    live_c = live.copy()
    live_c[[i for i in range(len(live)) if i not in CATH]] = False
    flashes = sorted(d["flashes"], key=lambda f: f["time"])
    ftime = np.array([f["time"] for f in flashes])
    fpe = {f["gid"]: np.array(f["pe"]) for f in d["flashes"]}
    fby = d["flash_by_gid"]

    print(f"# {CAL}\n# v_nom = {V_NOM} cm/us, cathode opdets {CATH}, "
          f"window +/-{WIN_US} us\n")

    for gid, u0, u4 in PAIRS:
        c0, c4 = fc.Clus(d, u0), fc.Clus(d, u4)
        t_nom = fby[gid]["time"]
        pistar, dmin = best_pi(c0, c4, t_nom, V_NOM)
        v_at_nomflash = pistar / t_nom               # v that best-meets at gid
        # invariant predicted cathode template (at the current flash)
        tmpl = cath_pred(d, gid, 0, u0, lib, live) + cath_pred(d, gid, 4, u4, lib, live)
        tmpl_c = tmpl.copy(); tmpl_c[[i for i in range(len(tmpl)) if i not in CATH]] = 0
        print(f"== pair c{u0}(bot)+c{u4}(top)  display gid{gid} "
              f"t={t_nom:.1f}us len {c0.length:.0f}/{c4.length:.0f}cm ==")
        print(f"   pi* = {pistar:.2f} (v/t) -> best-meet d={dmin:.1f}cm; "
              f"predicted cathode PE (template) = {tmpl_c.sum():.0f}")
        # candidate flashes in window
        sel = np.where(np.abs(ftime - t_nom) <= WIN_US)[0]
        rows = []
        for k in sel:
            f = flashes[k]; g = f["gid"]; t = f["time"]
            meas = fpe[g]
            v_k = pistar / t
            dv = v_k / V_NOM - 1.0
            d_nom, xmid = meeting(c0, c4, t, V_NOM)      # meeting at NOMINAL v
            ks_c = ks_dis(meas, tmpl_c, live_c)
            # predicted template recomputed AT this flash (invariance check)
            tk = (cath_pred(d, g, 0, u0, lib, live)
                  + cath_pred(d, g, 4, u4, lib, live))
            tk_c = tk[CATH].sum()
            rows.append((g, t, t - t_nom, meas.sum(), meas[CATH].sum(),
                         ks_c, v_k, dv * 100, d_nom, xmid, tk_c))
        rows.sort(key=lambda r: r[1])
        print("   gid     t_us    dt   totPE  cathPE  cathKS   v_k    dv%  "
              "d@vnom xmid  predC@k")
        for (g, t, dt, tot, cath, ks_c, v_k, dvp, d_nom, xmid, tkc) in rows:
            mark = " <=picked" if g == gid else ""
            star = " *BRIGHT" if cath > 300 else ""
            print(f"   {g:4d} {t:8.1f} {dt:6.1f} {tot:7.0f} {cath:7.0f} "
                  f"{ks_c:6.3f} {v_k:6.4f} {dvp:6.2f} {d_nom:6.1f} {xmid:5.1f} "
                  f"{tkc:7.0f}{mark}{star}")
        print()


if __name__ == "__main__":
    main()
