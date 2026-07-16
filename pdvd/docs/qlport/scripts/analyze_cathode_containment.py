#!/usr/bin/env python3
"""Why two bright cathode-crosser flashes are demoted to dim ones (run 039252 evt 298567).

Read-only.  For the two hand-scan pairs in evt 298567 that were matched to
suspiciously DIM flashes, this reproduces the Q-L matcher's cathode-containment
test on the nearby BRIGHT flash and measures, per TPC half, how far the
cathode-end drift coordinate is pushed PAST the cathode -- the quantity that
`require_containment` gates on.

The arithmetic IS the code's (match/src/QLMatching.cxx):
    flash_x_offset = sign_offset * (flash_time + trigger_offset) * drift_speed
    x_drift        = x_raw + flash_x_offset
    u              = s * (x_drift - anode_x)        # u=0 anode, u=u_cathode cathode
    contained  <=>  last_u < u_cathode + cathode_ext1      (cathode_ext1 = +1.2 cm)
The calib dump writes the flash time offset-folded, per crate: f["time"] carries
the BOTTOM/BDE offset, f["time1"] the TOP/TDE offset, so a bottom half reads
f["time"] and a top half reads f["time1"] (commit e587f357).  This script only
re-does the sign*t*v step, reproducing what QLMatching did internally.

Two independent signals per (half, flash):
  1. recomputed cathode-end overshoot past u_cathode (this script);
  2. whether the code kept that (cluster, flash) as a candidate bundle in the
     dump -- the dump writes a bundle ONLY if contained (QLMatching.cxx:2821-2827),
     so absence == the code's own containment verdict (modulo the marginal
     caveat in the companion doc).

Companion doc: pdvd-cathode-containment-flash-demotion.md
Repro:
    cd wcp-porting-img/pdvd
    python3 docs/qlport/scripts/analyze_cathode_containment.py
    # or another dump:
    python3 docs/qlport/scripts/analyze_cathode_containment.py work/039252_0/calib-evt298567.json
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEF_DUMP = "work/039252_0/calib-evt298567.json"
OUT_PNG = "docs/qlport/pics/cathode_containment_298567.png"

CATHODE_EXT1 = 1.2   # cm, QLMatching.h:238 (PDVD default; jsonnet sets only cathode_ext2)
CATHODE_FACE = 3.0   # cm, the |x|=3 cm active cathode face (params.jsonnet:33); the user's "3 cm"

# The two hand-scan pairs.  Each: (label, cluster uid, clock key, current flash gid,
# candidate/suspected-bright flash gid).  Bottom (apa 0) uid==ident and reads "time";
# top (apa 4) uid==4000000+ident and reads "time1".
PAIRS = [
    ("pair 1  (flash 38 dim <- top:60,bot:50 ; suspected bright flash 37)",
     [("top:60", 4000060, "time1"), ("bot:50", 50, "time")], 38, 37),
    ("pair 2  (flash 42 dim <- bot:8,top:63 ; suspected bright flash 41)",
     [("bot:8", 8, "time"), ("top:63", 4000063, "time1")], 42, 41),
]


def side_of(apa):
    return 0 if apa < 4 else 4


def main(dump=DEF_DUMP):
    d = json.load(open(dump))
    v = d["drift_speed"]                                   # cm/us
    geo = {int(k): val for k, val in d["geometry"].items()}
    clu = {c["uid"]: c for c in d["clusters"]}
    fl = {f["gid"]: f for f in d["flashes"]}

    # Candidate-bundle universe as written by the code (contained bundles only).
    cand = set()
    for b in d["bundles"]:
        cand.add((b["main_cluster"], b["flash_gid"]))
        for oc in b.get("other_clusters", []):
            cand.add((oc, b["flash_gid"]))

    def clusters_on_flash(gid):
        s = set()
        for b in d["bundles"]:
            if b["flash_gid"] == gid:
                s.add(b["main_cluster"])
                s.update(b.get("other_clusters", []))
        return s

    print("=" * 92)
    print("PDVD cathode-containment flash demotion   dump=%s" % dump)
    print("=" * 92)
    print("drift_speed v = %.6f cm/us    cathode_ext1 = +%.1f cm (containment tol past cathode)"
          % (v, CATHODE_EXT1))
    print("u_cathode = %.2f cm ;  cathode_in = u_cathode + cathode_ext1 = %.2f cm"
          % (geo[0]["u_cathode"], geo[0]["u_cathode"] + CATHODE_EXT1))
    print("the user's \"3 cm\" is the |x|=3 cm active cathode FACE that DEFINES u_cathode, not the tol.")
    print()

    rows = []   # for the PNG

    def analyze(label, uid, clock, gid, kind):
        c = clu[uid]
        g = geo[side_of(c["apa"])]
        uc = g["u_cathode"]
        x = np.array(c["x"])
        t = fl[gid][clock]
        u = g["s"] * (x + g["sign_offset"] * t * v - g["anode_x"])
        us = np.sort(u)
        rawmax = us[-1]
        p995 = np.percentile(u, 99.5)
        cathode_in = uc + CATHODE_EXT1
        n_out = int((u > cathode_in).sum())
        over_face = rawmax - uc                     # past the cathode itself
        over_tol = rawmax - cathode_in              # past the +1.2 cm containment tol
        present = (uid, gid) in cand
        verdict = "CONTAINED" if present else "DROPPED (not a candidate)"
        pe = fl[gid]["total_PE"]
        print("  %-7s  flash gid%-3d t=%9.2f PE=%8.1f  clock=%s" % (label, gid, fl[gid]["time"], pe, clock))
        print("     cathode-end u:  rawmax=%7.2f  p99.5=%7.2f   (u_cathode=%.2f)" % (rawmax, p995, uc))
        print("     past cathode face : %+6.2f cm   (vs the |x|=3 cm face figure)" % over_face)
        print("     past +1.2 cm tol  : %+6.2f cm   %d pts beyond cathode_in (%.2f%%)"
              % (over_tol, n_out, 100.0 * n_out / len(u)))
        print("     dump bundle for (uid %d, gid %d): %s -> %s" % (uid, gid, present, verdict))
        print()
        rows.append(dict(label=label, uid=uid, gid=gid, kind=kind, u=u, uc=uc,
                         z=np.array(c["z"]), rawmax=rawmax, over_tol=over_tol, present=present))

    for title, halves, cur, canfl in PAIRS:
        print("-" * 92)
        print(title)
        print("-" * 92)
        for label, uid, clock in halves:
            analyze(label, uid, clock, cur, "current")     # dim / current match
            analyze(label, uid, clock, canfl, "candidate")  # bright / suspected match
        # not-a-global-drop census
        ncur = len(clusters_on_flash(cur))
        ncan = len(clusters_on_flash(canfl))
        print("     census: flash gid%d has %d candidate clusters ; gid%d has %d -- both widely used,"
              % (cur, ncur, canfl, ncan))
        print("     so the missing halves above are CLUSTER-SPECIFIC (containment is the only per-(flash,cluster) gate).")
        print()

    # ---- PNG: the two crossers in the common drift coord u, current vs candidate flash ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    col = {"current": "#1f77b4", "candidate": "#d62728"}
    mark = {True: "o", False: "x"}
    for ax, (title, halves, cur, canfl) in zip(axes, PAIRS):
        prs = [r for r in rows if r["gid"] in (cur, canfl)]
        uc = prs[0]["uc"]
        for r in prs:
            lab = "%s  gid%d (%s)  %s" % (
                r["label"], r["gid"], r["kind"],
                "kept" if r["present"] else "DROPPED")
            ax.scatter(r["u"], r["z"], s=4, c=col[r["kind"]], alpha=0.45,
                       marker=mark[r["present"]], label=lab)
        ax.axvline(0.0, color="gray", ls=":", lw=1.0)
        ax.axvline(uc, color="green", ls="--", lw=1.2, label="cathode u=%.1f" % uc)
        ax.axvline(uc + CATHODE_EXT1, color="orange", ls="--", lw=1.2,
                   label="cathode_in=+1.2cm (%.1f)" % (uc + CATHODE_EXT1))
        ax.axvline(uc + CATHODE_FACE, color="purple", ls=":", lw=1.0,
                   label="user 3cm (%.1f)" % (uc + CATHODE_FACE))
        ax.set_xlim(uc - 60, uc + 12)
        ax.set_xlabel("u  (drift coord, cm; 0=anode, %.1f=cathode)" % uc)
        ax.set_ylabel("z (cm)")
        ax.set_title(title.split("(")[0].strip() +
                     "\ndim gid%d vs bright gid%d" % (cur, canfl), fontsize=9)
        ax.legend(loc="lower left", fontsize=6.5)
        ax.grid(alpha=0.25)
    fig.suptitle("evt 298567: bright flash pushes a cathode-crosser half PAST the cathode -> require_containment drops it",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=120)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEF_DUMP)
