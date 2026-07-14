#!/usr/bin/env python3
"""Worked numeric trace of the PDVD light-flash -> charge-readout time chain.

Read-only.  Follows ONE matched flash's time through every term the current
Q-L matching chain applies -- from the light-window-relative flash time, through
the per-crate trigger offset, into the drift-x placement -- for BOTH a Bottom
(BDE) and a Top (TDE) cluster of a single cathode-crossing cosmic.  Its purpose
is to make the whole light<->charge registration visible in one place so any
unaccounted overall time shift has nowhere to hide.

The arithmetic here IS the code's arithmetic (match/src/QLMatching.cxx):
    flash_x_offset = sign_offset * (flash_time + trigger_offset) * drift_speed
    x_drift        = x_raw + flash_x_offset
    u              = s * (x_drift - anode_x)        # u=0 anode, u=u_cathode cathode
The calib dump already writes the flash time offset-folded, per crate:
    f["time"]  = (get_time() + trigger_offset_bot) / us     # bottom/BDE clock
    f["time1"] = (get_time() + trigger_offset_top) / us     # top/TDE clock
so this script reads f["time"]/f["time1"] directly and only re-does the
sign*t*v step, reproducing what QLMatching did internally.

Companion doc: pdvd-light-charge-time-chain.md
Repro:
    cd wcp-porting-img/pdvd
    python3 docs/qlmatch/trace_light_charge_time.py
    # or point at another dump / flash:
    python3 docs/qlmatch/trace_light_charge_time.py work/039252_6/calib-evt298651.json 88
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEF_DUMP = "work/039252_6/calib-evt298651.json"
DEF_GID = 88
# The gid=88 crosser in this dump: bottom uid=33 (2494 pts) + top uid=4000166
# (1138 pts) -- one cosmic crossing the cathode, its two halves read out by the
# two independent crates (BDE bottom, TDE top) but matched to ONE shared flash.
OUT_PNG = "docs/qlmatch/trace_light_charge_298651_gid88.png"


def side_of(apa):
    return 0 if apa < 4 else 4


def main(dump=DEF_DUMP, gid=DEF_GID):
    d = json.load(open(dump))
    v = d["drift_speed"]                       # cm/us
    TO = d["trigger_offsets_us"]               # [bottom(BDE), top(TDE)] us
    geo = {int(k): val for k, val in d["geometry"].items()}
    clu = {c["uid"]: c for c in d["clusters"]}
    fl = next(f for f in d["flashes"] if f["gid"] == gid)

    off_bot, off_top = TO[0], TO[1]
    t_bot, t_top = fl["time"], fl["time1"]     # already offset-folded, us
    get_time = t_bot - off_bot                 # light-window-relative, us
    dlt = t_top - t_bot                        # crate skew = off_top - off_bot

    print("=" * 78)
    print("PDVD light -> charge time-chain trace   dump=%s  flash gid=%d" % (dump, gid))
    print("=" * 78)
    print("drift_speed          v   = %.6f cm/us  (= %.4f mm/us)" % (v, v * 10))
    print("   (this dump's value -- the single-crosser estimate, NOT a canonical convention)")
    print("trigger_offsets_us  [BDE bottom, TDE top] = [%.3f, %.3f] us" % (off_bot, off_top))
    print()
    print("-- the flash time, decomposed --")
    print("  get_time()            = %10.3f us   (light full-stream window relative)" % get_time)
    print("  + offset_bot (%9.3f) = %10.3f us   f[\"time\"]  = BOTTOM (BDE) charge axis"
          % (off_bot, t_bot))
    print("  + offset_top (%9.3f) = %10.3f us   f[\"time1\"] = TOP    (TDE) charge axis"
          % (off_top, t_top))
    print("  crate skew  Delta = time1 - time = %+.3f us  ->  Delta*v = %+.3f cm"
          % (dlt, dlt * v))
    print("  (Delta == offset_top - offset_bot == %+.3f us  -- the ONLY top/bottom"
          % (off_top - off_bot))
    print("   time difference.  It is EVENT-SPECIFIC: the two crates' charge windows float")
    print("   per-event on independent 64-sample frame boundaries, up to ~32 us apart.)")
    print()

    rows = []

    def trace(uid, apa, clock):
        c = clu[uid]
        g = geo[side_of(apa)]
        x = np.array(c["x"])
        t = fl[clock]
        xoff = g["sign_offset"] * t * v
        xd = x + xoff
        u = g["s"] * (xd - g["anode_x"])
        # Robust endpoints: min/max over thousands of points is one stray ghost.
        # p1 = anode end, p99 = cathode end.
        u_an, u_ca = np.percentile(u, 1.0), np.percentile(u, 99.0)
        crate = "BDE bottom" if apa < 4 else "TDE top"
        print("-- %s half: uid=%d apa=%d  n=%d  (clock=%s)" % (crate, uid, apa, c["npoints"], clock))
        print("     sign_offset=%+d  s=%+.1f  anode_x=%.2f  u_cathode=%.2f"
              % (g["sign_offset"], g["s"], g["anode_x"], g["u_cathode"]))
        print("     flash_x_offset = sign*t*v = %+d * %.3f * %.6f = %+.3f cm"
              % (g["sign_offset"], t, v, xoff))
        print("     x_raw  [min,max] = [%8.2f, %8.2f] cm" % (x.min(), x.max()))
        print("     x_drift[min,max] = [%8.2f, %8.2f] cm  (x_raw + flash_x_offset)"
              % (xd.min(), xd.max()))
        print("     u  robust[p1,p99]= [%8.2f, %8.2f]     (anode u=0 .. cathode u=%.2f)"
              % (u_an, u_ca, g["u_cathode"]))
        print("     u  raw  [min,max]= [%8.2f, %8.2f]     (incl. stray ghost points)"
              % (u.min(), u.max()))
        print()
        rows.append(dict(crate=crate, uid=uid, n=c["npoints"], clock=clock, t=t,
                         xoff=xoff, u=u, u_an=u_an, u_ca=u_ca,
                         z=np.array(c["z"]), ucath=g["u_cathode"]))
        return u

    ub = trace(33, 0, "time")
    ut = trace(4000166, 4, "time1")

    # Wrong-clock demonstration: use the bottom clock for the top half.
    g = geo[4]
    x = np.array(clu[4000166]["x"])
    u_wrong = g["s"] * (x + g["sign_offset"] * t_bot * v - g["anode_x"])
    ut_ca = np.percentile(ut, 99.0)
    uw_ca = np.percentile(u_wrong, 99.0)
    print("-- top/bottom is REAL: use the WRONG (bottom) clock for the top half --")
    print("     top-half cathode-end u (p99):  correct(time1)=%.2f   wrong(time)=%.2f   miss=%+.2f cm (=Delta*v)"
          % (ut_ca, uw_ca, ut_ca - uw_ca))
    print()

    # Self-consistency vs the code: a small at_cathode fragment must land at u~u_cathode.
    f137 = next((f for f in d["flashes"] if f["gid"] == 137), None)
    if f137 is not None and 4 in clu:
        c = clu[4]
        g = geo[0]
        x = np.array(c["x"])
        uf = g["s"] * (x + g["sign_offset"] * f137["time"] * v - g["anode_x"])
        print("-- self-consistency: gid137 at_cathode fragment (bot uid4, n=%d) --" % c["npoints"])
        print("     this script's u = [%.2f, %.2f]  vs  u_cathode = %.2f"
              % (uf.min(), uf.max(), g["u_cathode"]))
        print("     -> reproduces the code's at_cathode flag; the arithmetic matches QLMatching.")
        print()

    # Compact summary table (robust p1/p99 endpoints).
    print("-- summary (robust p1/p99 endpoints) --")
    print("  %-11s %6s %9s %9s %11s %11s" %
          ("crate", "n", "clock/us", "xoff/cm", "u_anode_end", "u_cath_end"))
    for r in rows:
        print("  %-11s %6d %9.2f %+9.2f %11.2f %11.2f" %
              (r["crate"], r["n"], r["t"], r["xoff"], r["u_an"], r["u_ca"]))
    print()
    print("  Every term above is trigger_offset (folded into time/time1) and drift_speed --")
    print("  the flash-side registration.  What this arithmetic does NOT contain, and where")
    print("  an unaccounted overall light<->charge shift could therefore hide, is entirely on")
    print("  the CHARGE side, in how x_raw itself is anchored in time:")
    print("    (a) the WCT frame tick 0 vs the DAQ charge_{bde,tde}_window_start that")
    print("        offset_{bot,top} reference -- if these disagree, a CONSTANT shift enters")
    print("        (same-sign on both crates), and a BOTTOM-vs-TOP difference would come from")
    print("        the RDTimeStamp unit inconsistency (TDE ns-epoch vs BDE 16ns-DTS ticks).")
    print("    (b) the SP tick-shift (ctoffset + intrinsic FR origin/speed), a fixed per-crate")
    print("        registration of the deconvolved charge -- NOT derived from any flash.")
    print("  These are testable at the anode: an anode-touching track's charge end must land")
    print("  at u=0 (see the next-check in the companion doc, and pdvd-anode-time-consistency")
    print("  section 8.14).  Note: a velocity mismatch (v here vs the true drift speed) also")
    print("  shifts x -- that is an x error, not a time offset; keep the two separate.")

    # ---- PNG: both halves in the common drift coordinate u (0=anode, u_cathode=cathode) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"BDE bottom": "#1f77b4", "TDE top": "#d62728"}
    for r in rows:
        ax.scatter(r["u"], r["z"], s=3, c=colors[r["crate"]], alpha=0.5,
                   label="%s (uid %d, n=%d)" % (r["crate"], r["uid"], r["n"]))
    ucath = rows[0]["ucath"]
    ax.axvline(0.0, color="gray", ls="--", lw=1.0, label="anode (u=0)")
    ax.axvline(ucath, color="green", ls="--", lw=1.0, label="cathode (u=%.1f)" % ucath)
    ax.set_xlabel("u  (drift coordinate, cm;  0 = anode, %.1f = cathode)" % ucath)
    ax.set_ylabel("z (cm)")
    ax.set_title("gid=88 crosser placed by the matched flash: BDE uses time, TDE uses time1\n"
                 "(Delta = time1 - time = %+.2f us = %+.2f cm crate skew)" % (dlt, dlt * v))
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print("\nwrote", OUT_PNG)


if __name__ == "__main__":
    dump = sys.argv[1] if len(sys.argv) > 1 else DEF_DUMP
    gid = int(sys.argv[2]) if len(sys.argv) > 2 else DEF_GID
    main(dump, gid)
