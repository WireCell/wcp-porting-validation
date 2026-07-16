#!/usr/bin/env python3
"""Why the bright 274.4 us flash loses the full-gap crosser top:22 (run 039252 evt 298567).

Read-only.  The ANODE-side counterpart of analyze_cathode_containment.py: it
reproduces the Q-L matcher's anode-containment test for cluster top:22 against the
two candidate flashes, and shows that the 2 cm anode pull is what flips which flash
is admissible.

The arithmetic IS the code's (match/src/QLMatching.cxx):
    flash_x_offset = sign_offset * flash_time * drift_speed   # time already offset-folded
    u              = s * (x_raw + flash_x_offset - anode_x)   # u=0 anode, u=u_cathode cathode
    contained  <=>  first_u > anode_ext1 - anode_ext1_margin  # anode floor  (:3727)
                 && last_u  < u_cathode + cathode_ext1        # cathode ceiling
                 && last_u > 0 && first_u < u_cathode
The calib dump writes the flash time offset-folded PER CRATE: f["time"] carries the
BOTTOM/BDE offset, f["time1"] the TOP/TDE offset (commit e587f357).  top:22 is apa 4
=> read f["time1"].  Do NOT add d["trigger_offsets_us"] on top -- it is already in.

Validation gate baked in: flash 70 MUST come out contained (the dump marks its
bundle contained=True).  If it does not, the offset convention above is being
misapplied -- fix that before reading anything into flash 71.

Companion doc: pdvd-cathode-containment-flash-demotion.md sec 11
Repro:
    cd wcp-porting-img/pdvd
    python3 docs/qlport/scripts/analyze_anode_containment.py
    # or another dump:
    python3 docs/qlport/scripts/analyze_anode_containment.py <path-to-calib-evt298567.json>
"""
import sys
import json
import numpy as np

DEF_DUMP = "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd/work/039252_0_spcov/calib-evt298567.json"

UID = 4000022          # top:22 -- the full-gap cathode crosser under study
APA = 4                # top volume
PULL_US = 13.507       # PDVD_QL_EXTRA_OFFSET_US production default = 2.0 cm toward anode

ANODE_EXT1 = -2.0      # cm, C++ default (QLMatching.h) -- NOT overridden for PDVD
ANODE_EXT2 = 4.0       # cm, anode flag-window OUTER edge
CATHODE_EXT1 = 2.0     # cm, PDVD production (runner default; C++ default is 1.2)
MARGIN_OLD = 1.0       # cm, pre-2026-07-16 hard-coded slack  => floor -3.0
MARGIN_NEW = 2.0       # cm, PDVD production anode_ext1_margin => floor -4.0


def main(dump=DEF_DUMP):
    d = json.load(open(dump))
    v = d["drift_speed"]                                    # cm/us
    g = {int(k): x for k, x in d["geometry"].items()}[APA]
    clu = {c["uid"]: c for c in d["clusters"]}
    fl = {f["gid"]: f for f in d["flashes"]}

    if UID not in clu:
        sys.exit("cluster uid %d not in %s" % (UID, dump))
    x = np.array(clu[UID]["x"])
    uc = g["u_cathode"]
    cath_ceiling = uc + CATHODE_EXT1

    # the two candidate flashes, located by time so a gid renumber cannot mislead
    def gid_at(t):
        c = [gg for gg, f in fl.items() if abs(f["time"] - t) < 0.5]
        return c[0] if c else None
    g70, g71 = gid_at(243.99), gid_at(274.43)
    if g70 is None or g71 is None:
        sys.exit("expected flashes at t=244.0/274.4 us not found in %s" % dump)

    def u_of(gid, extra_us=0.0):
        t = fl[gid]["time1"] + extra_us                     # apa 4 => TDE clock
        return g["s"] * (x + g["sign_offset"] * t * v - g["anode_x"])

    print("=" * 94)
    print("PDVD anode-containment: full-gap crosser top:22 vs the 244.0 / 274.4 us flashes")
    print("dump=%s" % dump)
    print("=" * 94)
    print("drift_speed v = %.6f cm/us   u_cathode = %.2f cm   cluster npoints = %d"
          % (v, uc, len(x)))
    u70 = u_of(g70)
    print("cluster u-span = %.3f cm = %.1f%% of the drift gap  -> a full-gap crosser has"
          % (np.ptp(u70), 100 * np.ptp(u70) / uc))
    print("almost no timing freedom, so a 2 cm systematic decides the match.\n")

    # ---- validation gate: flash 70 must be contained -------------------------
    fu70, lu70 = u70.min(), u70.max()
    ok70 = (fu70 > ANODE_EXT1 - MARGIN_OLD) and (lu70 < cath_ceiling)
    print("VALIDATION GATE: flash %d (t=%.1f) contained under the OLD floor? %s   %s"
          % (g70, fl[g70]["time"], "YES" if ok70 else "NO",
             "(dump says contained=True -- convention OK)" if ok70
             else "*** convention misapplied, stop ***"))
    if not ok70:
        sys.exit(1)
    print()

    # ---- the cut table -------------------------------------------------------
    print("%-34s %9s %9s   %s" % ("", "first_u", "last_u", "verdict"))
    print("-" * 94)
    for margin in (MARGIN_OLD, MARGIN_NEW):
        floor = ANODE_EXT1 - margin
        for gid in (g70, g71):
            u = u_of(gid)
            fu, lu = u.min(), u.max()
            bad = []
            if not fu > floor:
                bad.append("ANODE fail by %.3f cm" % (floor - fu))
            if not lu < cath_ceiling:
                bad.append("CATHODE fail by %.3f cm" % (lu - cath_ceiling))
            print("margin %.1f (floor %+.1f)  flash %-3d %+9.3f %+9.3f   %s %s"
                  % (margin, floor, gid, fu, lu,
                     "CONTAINED" if not bad else "DROPPED", "; ".join(bad)))
        print()

    # ---- headroom / admissible window ---------------------------------------
    head_anode = fu70 - (ANODE_EXT1 - MARGIN_OLD)
    head_cath = cath_ceiling - lu70
    dt = fl[g71]["time1"] - fl[g70]["time1"]
    print("At flash %d, under the OLD floor:" % g70)
    print("  anode headroom  %.3f cm (%.2f us)   cathode headroom %.3f cm (%.2f us)"
          % (head_anode, head_anode / v, head_cath, head_cath / v))
    print("  => total admissible flash-time window %.3f cm = %.2f us"
          % (head_anode + head_cath, (head_anode + head_cath) / v))
    print("  flash %d is %+.2f us later = %+.3f cm -> overshoots by %.3f cm (%.2f us)\n"
          % (g71, dt, dt * v, dt * v - head_anode, (dt * v - head_anode) / v))

    # ---- the 2 cm pull is what flips the verdict -----------------------------
    print("The 2 cm anode pull (PDVD_QL_EXTRA_OFFSET_US=%.3f us = %.3f cm) flips it:"
          % (PULL_US, PULL_US * v))
    print("%-30s %9s %9s   %s" % ("", "first_u", "last_u", "verdict (OLD floor -3.0)"))
    print("-" * 94)
    for gid in (g70, g71):
        for pull_on in (True, False):
            u = u_of(gid, 0.0 if pull_on else -PULL_US)
            fu, lu = u.min(), u.max()
            bad = []
            if not fu > ANODE_EXT1 - MARGIN_OLD:
                bad.append("ANODE fail by %.3f cm" % (ANODE_EXT1 - MARGIN_OLD - fu))
            if not lu < cath_ceiling:
                bad.append("CATHODE fail by %.3f cm" % (lu - cath_ceiling))
            print("flash %-3d pull=%-3s              %+9.3f %+9.3f   %s %s"
                  % (gid, "ON" if pull_on else "OFF", fu, lu,
                     "CONTAINED" if not bad else "DROPPED", "; ".join(bad)))
    print()

    # ---- the code's own verdict ---------------------------------------------
    print("Cross-check -- the dump writes a bundle ONLY if contained:")
    for gid in (g70, g71):
        b = [x for x in d["bundles"] if x["flash_gid"] == gid
             and (x["main_cluster"] == UID or UID in x.get("other_clusters", []))]
        print("  flash %-3d (t=%7.1f, PE=%8.0f) x cluster %d : %s"
              % (gid, fl[gid]["time"], fl[gid]["total_PE"], UID,
                 ("bundle PRESENT (contained=%s, auto_selected=%s)"
                  % (b[0]["contained"], b[0]["auto_selected"])) if b
                 else "NO BUNDLE -> code judged it uncontained"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEF_DUMP)
