#!/usr/bin/env python3
"""doc pr/136 #3 -- is the energy-ladder feedback worth chasing?

THE HYPOTHESIS (charter sec 5 item 2).  examine_showers' cross-cluster merge
ladder keys its cone width to the ABSORBING shower's own energy
(NeutrinoShowerClustering.cxx:4999-5013):

    Eshower > 800 MeV  -> tmp_angle < 30
    Eshower > 360 MeV  -> tmp_angle < 25
    Eshower > 250 MeV  -> tmp_angle < 15
    Eshower > 150 MeV  -> tmp_angle < 10   (and < 18 for a weak-dir conn-1 seg)
    Eshower > 100 MeV  -> tmp_angle < 10   (only when sg_length < 25 cm)

with `Eshower = kine_best ?: kine_charge` (:4869).  A shower that is already
short of charge therefore falls to a NARROWER cone and absorbs less, which
keeps it short: self-reinforcing, and there is no knob anywhere on the ladder.
kine_shower_fudge_factor DIVIDES kine_charge, so the 0.84 -> 0.86 production
flip moved every Eshower by 2.3% and could push showers across an edge.

THE KILLING MEASUREMENT, which is what this script is.  The hypothesis only
matters if showers actually SIT near the edges.  If the population is far from
150/250/360/800, no plausible energy error changes any tier and the whole idea
is dead without writing a knob.

THE CAVEAT, stated because it bounds the answer.  The ladder is evaluated
inside examine_showers, right after the second kinematics pass; the dump
carries the FINAL kine_best/kine_charge, after examine_showers itself,
examine_shower_1, the fragment merges and the pi0 passes have grown showers
further.  So this is a proxy: it answers "how close is the population to an
edge", not "which shower was on which side at ladder time".  A shower that grew
after examine_showers reads HIGHER here than the ladder saw, which makes this
census CONSERVATIVE for the feedback direction (deficient showers read too
high, so the near-edge count is if anything understated).

READ-ONLY.

    scripts/pr136_tier_census.py --manifest98 em117-136f086probe98-manifest.tsv \\
        --manifest141 em114c-136f086probe141-manifest.tsv --fudge 0.86
"""
import argparse, csv, glob, importlib.util, json, os, statistics as st, sys

SD = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(SD)
ED = os.path.join(SX, "em_display")

EDGES = [100.0, 150.0, 250.0, 360.0, 800.0]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


S = _load("em117_score", os.path.join(ED, "em117_score.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest98", default="em117-136f086probe98-manifest.tsv")
    ap.add_argument("--manifest141", default="em114c-136f086probe141-manifest.tsv")
    ap.add_argument("--fudge", type=float, default=0.86,
                    help="the fudge in force on this arm (for the flip-crossing test)")
    ap.add_argument("--prev-fudge", type=float, default=0.84,
                    help="the previous production fudge, to count edge crossings")
    ap.add_argument("--tsv", default="docs/pr/pr136-tier-census.tsv")
    a = ap.parse_args()

    rows = []
    for man, setname in ((a.manifest98, "98"), (a.manifest141, "141")):
        mp = man if os.path.isabs(man) else os.path.join(ED, man)
        if not os.path.exists(mp):
            print("[warn] no manifest %s" % mp); continue
        for ev, mrow in sorted(S.load_manifest(mp).items()):
            dump = S.load_dump(mrow["dump"])
            if not dump:
                continue
            for sh in dump.get("showers") or ():
                if int(sh.get("particle_id", 0)) != 11:
                    continue      # the ladder's cache keeps EM electrons only
                kb = float(sh.get("kine_best") or 0.0)
                kc = float(sh.get("kine_charge") or 0.0)
                E = kb if kb != 0 else kc
                if E <= 0:
                    continue
                # nearest edge, and the fractional distance to it
                d = min(((abs(E - e), e) for e in EDGES))
                # where the SAME charge would land at the previous fudge:
                # kine_charge is divided by the fudge, so E' = E * fudge/prev
                Eprev = E * a.fudge / a.prev_fudge
                tier = lambda x: sum(1 for e in EDGES if x > e)
                rows.append(dict(set=setname, sample=mrow.get("sample", ""), event=ev,
                                 shower=int(sh["id"]), conn=sh.get("start_connection_type", -1),
                                 kine_best=round(kb, 2), kine_charge=round(kc, 2),
                                 E=round(E, 2), nearest_edge=d[1],
                                 d_MeV=round(d[0], 2), d_frac=round(d[0] / d[1], 4),
                                 tier=tier(E), tier_prev_fudge=tier(Eprev),
                                 flip_crossed=int(tier(E) != tier(Eprev))))

    n = len(rows)
    print("ENERGY-LADDER TIER CENSUS  (doc pr/136 #3)")
    print("  EM showers (particle_id 11) with E>0: %d over %d events"
          % (n, len({r["event"] for r in rows})))
    print("  edges %s MeV;  fudge in force %.2f (previous %.2f)"
          % ("/".join("%g" % e for e in EDGES), a.fudge, a.prev_fudge))
    if not n:
        return 1
    print("\nDISTANCE TO THE NEAREST TIER EDGE")
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 25), (25, 50), (50, 1e9)):
        k = [r for r in rows if lo <= r["d_MeV"] < hi]
        print("  within %-9s %5d  %5.1f%%   (charge-side: %d are BELOW their edge)"
              % (("%g-%g MeV" % (lo, hi)) if hi < 1e9 else ">50 MeV",
                 len(k), 100.0 * len(k) / n,
                 sum(1 for r in k if r["E"] < r["nearest_edge"])))
    print("\n  median distance to the nearest edge: %.1f MeV" % st.median(r["d_MeV"] for r in rows))
    print("  tier occupancy (0 = below 100 MeV, 5 = above 800):")
    for t in range(6):
        k = [r for r in rows if r["tier"] == t]
        print("     tier %d  %5d  %5.1f%%" % (t, len(k), 100.0 * len(k) / n))

    nc = sum(r["flip_crossed"] for r in rows)
    print("\nTHE 0.84 -> 0.86 FLIP TEST -- how many showers changed tier when the")
    print("  production fudge moved?  %d of %d = %.2f%%" % (nc, n, 100.0 * nc / n))
    if nc:
        print("  %-8s %-8s %-9s %-9s %s" % ("event", "shower", "E@0.86", "E@0.84", "tier"))
        for r in sorted((r for r in rows if r["flip_crossed"]), key=lambda x: -x["E"])[:12]:
            print("  %-8d %-8d %-9.1f %-9.1f %d -> %d"
                  % (r["event"], r["shower"], r["E"], r["E"] * a.fudge / a.prev_fudge,
                     r["tier_prev_fudge"], r["tier"]))

    near = [r for r in rows if r["d_MeV"] < 10]
    print("\nREADING.  %d of %d showers (%.1f%%) sit within 10 MeV of a tier edge."
          % (len(near), n, 100.0 * len(near) / n))
    print("  The %.1f%% energy step of the %.2f->%.2f flip moves a shower across an edge"
          % (100.0 * abs(a.fudge - a.prev_fudge) / a.prev_fudge, a.prev_fudge, a.fudge))
    print("  only if it sits within that fraction of one, and the flip test above is")
    print("  the direct count: %d of %d.  Judge the hypothesis on those two numbers,"
          % (nc, n))
    print("  not on the plausibility of the mechanism.")
    below = sum(1 for r in rows if r["flip_crossed"] and r["tier"] < r["tier_prev_fudge"])
    print("  Direction: %d of the %d crossings are DOWNWARD (the fudge divides, so the"
          % (below, nc))
    print("  0.86 flip made those showers absorb through a NARROWER cone).")

    o = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
    with open(o, "w", newline="") as fh:
        w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nwrote %s (%d rows)" % (o, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
