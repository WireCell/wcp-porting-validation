#!/usr/bin/env python3
"""doc pr/141 item 4 -- the type-admission scan set, and its yield.

READ-ONLY.  Reads the production arm work-pr140r2-off-* (239 events) and writes
only its --tsv.

doc pr/139 sec 22.5 left one lead at n = 2: the splitter fires on non-EM-typed
objects with purity 0.000 (1 fire, 0 confirmed cuts), so an EM-only restriction
looks free -- but two negatives cannot set a rule.  Tonight's sec 4 added a
second question of exactly the same shape at n = 1: K20's 40 cm "trackish"
bound refuses a mu-typed object (283713's 23011, 57.3 cm) that the owner's scan
says IS the second gamma.

**Both are questions about which particle TYPES may enter the pi0 machinery**,
and both need a targeted sample rather than a threshold picked off one event.
This script builds that sample, in two strata, and -- the point -- prices each
stratum before anyone scans it.

  S-MU   mu-typed (|pdg| = 13) showers that K20 refuses: not flagged as
         showers, length in the 40-80 cm band (40 = the shipped bound; 80 = a
         plausible ceiling), conn_type 2 or 3 so they belong to the
         DISCONNECTED pool, and above 30 MeV.  For each, the script asks the
         question that decides the knob's worth: pair it with every EM shower
         in its event under the finder's own vertex-ray geometry and report
         whether ANY pairing lands in the (100,160) window.  A candidate that
         cannot make an in-window pair is worthless however it is typed.

  S-PI   pi-typed (|pdg| = 211) showers big enough for the splitter to
         consider (>= 3 segments and >= 100 MeV), which is the population sec
         22.5's EM-only restriction would exclude.

    python3 scripts/pr141_typeset.py --tsv docs/pr/pr141-typeset.tsv
"""
import argparse, glob, json, math, os, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = (100.0, 160.0)
# A mu-typed object's stored kine_charge is the TRACK-hypothesis energy; the
# same charge under the shower hypothesis is larger (different recombination
# and fudge).  Measured on the one object where both numbers exist -- 283713's
# 23011, dump kine_charge 168.4 MeV vs the scan label's shower-hypothesis
# 320.9 MeV -- the ratio is 1.906.  The viewer's own note quotes 1.66 for the
# same effect, so this is a per-object quantity and 1.9 is used here only to
# SCREEN, never to price anything.
HYP = 1.906
ARMS = ["work-pr140r2-off-mcp1k", "work-pr140r2-off-mcp2k",
        "work-pr140r2-off-ncpi0", "work-pr140r2-off-nuecc48"]


def ang(u, v):
    nu = math.sqrt(sum(x * x for x in u)); nv = math.sqrt(sum(x * x for x in v))
    if nu <= 0 or nv <= 0:
        return None
    c = sum(u[i] * v[i] for i in range(3)) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--mu-len", type=float, nargs=2, default=(40.0, 80.0))
    args = ap.parse_args()

    smu, spi, nev = [], [], 0
    for arm in ARMS:
        for f in sorted(glob.glob(os.path.join(SX, arm, "pr_evt*", "calib-pr-evt*.json"))):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            nev += 1
            ev = int(d.get("meta", {}).get("eventNo") or 0)
            mv = d.get("main_vertex") or {}
            v = [mv.get("x"), mv.get("y"), mv.get("z")]
            if v[0] is None:
                continue
            shw = d.get("showers") or []
            em = [s for s in shw if abs(int(s.get("particle_id") or 0)) == 11
                  and (s.get("kine_charge") or 0) >= 20.0]

            for s in shw:
                pdg = abs(int(s.get("particle_id") or 0))
                E = s.get("kine_charge") or 0.0
                ln = s.get("total_length") or 0.0
                ct = s.get("start_connection_type")
                st = s.get("start") or {}
                p = [st.get("x"), st.get("y"), st.get("z")]
                if None in p:
                    continue

                if (pdg == 13 and ct in (2, 3) and E >= 30.0
                        and args.mu_len[0] <= ln <= args.mu_len[1]):
                    best = best_h = None
                    for t in em:
                        tp = t["start"]
                        th = ang([p[i] - v[i] for i in range(3)],
                                 [tp["x"] - v[0], tp["y"] - v[1], tp["z"] - v[2]])
                        if th is None:
                            continue
                        E2 = t.get("kine_charge") or 0.0
                        sf = math.sin(math.radians(th) / 2) ** 2
                        m = math.sqrt(4 * E * E2 * sf)
                        mh = math.sqrt(4 * E * HYP * E2 * sf)
                        if WIN[0] <= m <= WIN[1] and (best is None or
                                                      abs(m - 135) < abs(best[2] - 135)):
                            best = (int(t["id"]), E2, m, th)
                        if WIN[0] <= mh <= WIN[1] and (best_h is None or
                                                       abs(mh - 135) < abs(best_h[2] - 135)):
                            best_h = (int(t["id"]), E2, mh, th)
                    smu.append(dict(stratum="S-MU", arm=arm.split("-")[-1], event=ev,
                                    shower=int(s["id"]), pdg=s.get("particle_id"),
                                    E="%.1f" % E, length="%.1f" % ln, conn=ct,
                                    nseg=s.get("num_segments"),
                                    partner=best[0] if best else "",
                                    partner_E="%.1f" % best[1] if best else "",
                                    mass="%.1f" % best[2] if best else "",
                                    theta="%.1f" % best[3] if best else "",
                                    inwindow=bool(best),
                                    partner_h=best_h[0] if best_h else "",
                                    mass_h="%.1f" % best_h[2] if best_h else "",
                                    theta_h="%.1f" % best_h[3] if best_h else "",
                                    inwindow_h=bool(best_h)))

                if pdg == 211 and (s.get("num_segments") or 0) >= 3 and E >= 100.0:
                    spi.append(dict(stratum="S-PI", arm=arm.split("-")[-1], event=ev,
                                    shower=int(s["id"]), pdg=s.get("particle_id"),
                                    E="%.1f" % E, length="%.1f" % ln, conn=ct,
                                    nseg=s.get("num_segments"),
                                    partner="", partner_E="", mass="", theta="",
                                    inwindow="", partner_h="", mass_h="",
                                    theta_h="", inwindow_h=""))

    hit = [r for r in smu if r["inwindow"]]
    hit_h = [r for r in smu if r["inwindow_h"]]
    print("events read: %d" % nev)
    print("\n=== S-MU: mu-typed, conn 2/3, E >= 30 MeV, length %.0f-%.0f cm "
          "(the band K20's 40 cm bound refuses) ===" % tuple(args.mu_len))
    print("  candidates: %d" % len(smu))
    print("  of which an in-window pairing EXISTS with some EM shower: %d (%.0f%%)"
          % (len(hit), 100.0 * len(hit) / max(1, len(smu))))
    print("  of which in-window under the SHOWER hypothesis (x%.3f): %d (%.0f%%)"
          % (HYP, len(hit_h), 100.0 * len(hit_h) / max(1, len(smu))))
    for r in sorted(hit_h, key=lambda r: abs(float(r["mass_h"]) - 135)):
        print("    SHOWER-HYP  %-8s evt %-7d sh %-7d E=%7s (x%.2f) len=%6s "
              "-> %s theta=%s  m=%s"
              % (r["arm"], r["event"], r["shower"], r["E"], HYP, r["length"],
                 r["partner_h"], r["theta_h"], r["mass_h"]))
    for r in sorted(hit, key=lambda r: abs(float(r["mass"]) - 135))[:20]:
        print("    %-8s evt %-7d sh %-7d E=%7s len=%6s nseg=%-3s -> %s "
              "(E=%s) theta=%s  m=%s"
              % (r["arm"], r["event"], r["shower"], r["E"], r["length"],
                 r["nseg"], r["partner"], r["partner_E"], r["theta"], r["mass"]))
    print("\n=== S-PI: pi-typed, >= 3 seg, >= 100 MeV "
          "(what an EM-only splitter would exclude) ===")
    print("  candidates: %d" % len(spi))
    for r in sorted(spi, key=lambda r: -float(r["E"]))[:15]:
        print("    %-8s evt %-7d sh %-7d E=%7s len=%6s nseg=%s conn=%s"
              % (r["arm"], r["event"], r["shower"], r["E"], r["length"],
                 r["nseg"], r["conn"]))

    if args.tsv:
        hdr = ["stratum", "arm", "event", "shower", "pdg", "E", "length",
               "conn", "nseg", "partner", "partner_E", "theta", "mass",
               "inwindow", "partner_h", "theta_h", "mass_h", "inwindow_h"]
        with open(args.tsv, "w") as fh:
            fh.write("\t".join(hdr) + "\n")
            for r in smu + spi:
                fh.write("\t".join(str(r[h]) for h in hdr) + "\n")
        print("\nwrote %s (%d rows)" % (args.tsv, len(smu) + len(spi)))


if __name__ == "__main__":
    main()
