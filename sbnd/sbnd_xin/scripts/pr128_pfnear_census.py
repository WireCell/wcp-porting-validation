#!/usr/bin/env python3
"""doc pr/128 -- choose the class-A operating point from the C++'s own tape.

WCT_PFNEAR_DEBUG=1 makes fill_bee_pf_tree print one line per candidate from
the SAME code path the knob uses, so census and knob cannot use different
definitions (the pr/127 WCT_SCCC_DEBUG pattern; the tape is byte-neutral and
the gate on the tape arms proves it -- 478/478 archives identical).

  PFNEAR track seg=.. cluster=.. pdg=.. score=.. len_cm=.. ke_mev=..
        gap_cm=.. cand_end_cm=.. ref_end_cm=.. kink_deg=.. d_mainvtx_cm=..
        verdict=..
  PFNEAR conn4 shower_id=.. node=.. cluster=.. pdg=.. ke_mev=.. len_cm=..
        nseg=.. gap_cm=.. verdict=..

The question this answers: does the CONTINUATION geometry separate the class
the owner wants rescued (a daughter split into another cluster, joining the
candidate end-to-end and running straight on) from the class the owner ruled
out (a cosmic brushing the far end of a displayed track)?  SBND 18255-72786
is the reference negative: three objects from two large off-vertex clusters,
+1151 MeV on a 701 MeV candidate.

Repro:
  ./scripts/pr128_pfnear_census.py 'work-pr128r1-dbg98-*' 'work-pr128r1-dbg141-*' \
      --tsv docs/pr/pr128-pfnear-census.tsv
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)

RX_TRACK = re.compile(
    r"PFNEAR track seg=(?P<seg>-?\d+) cluster=(?P<cluster>\S+) pdg=(?P<pdg>-?\d+) "
    r"score=(?P<score>[-\d.]+) len_cm=(?P<len>[-\d.]+) ke_mev=(?P<ke>[-\d.]+) "
    r"gap_cm=(?P<gap>[-\d.]+) cand_end_cm=(?P<cend>[-\d.]+) ref_end_cm=(?P<rend>[-\d.]+) "
    r"kink_deg=(?P<kink>[-\d.]+) d_mainvtx_cm=(?P<dmv>[-\d.]+) verdict=(?P<verdict>\w+)")
RX_CONN4 = re.compile(
    r"PFNEAR conn4 shower_id=(?P<sid>-?\d+) node=(?P<node>-?\d+) cluster=(?P<cluster>\S+) "
    r"pdg=(?P<pdg>-?\d+) ke_mev=(?P<ke>[-\d.]+) len_cm=(?P<len>[-\d.]+) nseg=(?P<nseg>\d+) "
    r"gap_cm=(?P<gap>[-\d.]+) verdict=(?P<verdict>\w+)")

# (end_tol_cm, kink_deg, gap_cm) points to score.  The shipped default is
# marked in the output.
POINTS = [
    (1e9, 1e9, 5.0),    # proximity only -- the 72786 failure
    (20.0, 45.0, 5.0),
    (10.0, 30.0, 5.0),  # C++ default
    (10.0, 30.0, 10.0),
    (5.0, 20.0, 5.0),
    (5.0, 10.0, 5.0),
    (2.0, 10.0, 3.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--tsv")
    args = ap.parse_args()

    tracks, conn4 = [], []
    for pat in args.roots:
        for root in sorted(glob.glob(os.path.join(SX, pat)) or glob.glob(pat)):
            for lg in sorted(glob.glob(os.path.join(root, "pr_evt*", "stdout.log"))):
                ev = int(os.path.basename(os.path.dirname(lg))[len("pr_evt"):])
                seen_t, seen_c = set(), set()
                for line in open(lg, errors="replace"):
                    m = RX_TRACK.search(line)
                    if m:
                        d = m.groupdict()
                        key = int(d["seg"])
                        if key in seen_t:
                            continue          # the PR chain renders the tree twice
                        seen_t.add(key)
                        tracks.append(dict(event=ev, seg=key, cluster=d["cluster"],
                                           pdg=int(d["pdg"]), score=float(d["score"]),
                                           length=float(d["len"]), ke=float(d["ke"]),
                                           gap=float(d["gap"]), cend=float(d["cend"]),
                                           rend=float(d["rend"]), kink=float(d["kink"]),
                                           dmv=float(d["dmv"])))
                        continue
                    m = RX_CONN4.search(line)
                    if m:
                        d = m.groupdict()
                        key = int(d["node"])
                        if key in seen_c:
                            continue
                        seen_c.add(key)
                        conn4.append(dict(event=ev, node=key, cluster=d["cluster"],
                                          pdg=int(d["pdg"]), ke=float(d["ke"]),
                                          length=float(d["len"]), nseg=int(d["nseg"]),
                                          gap=float(d["gap"]), keep=d["verdict"] == "KEEP"))

    evs = {r["event"] for r in tracks} | {r["event"] for r in conn4}
    print("%d event(s) with a tape\n" % len(evs))

    # ---------------- class A operating points ----------------
    print("CLASS A -- cross-cluster tracks that pass pdg/length/straight-long.")
    print("  %d candidate(s) in %d event(s) reached the geometry test.\n"
          % (len(tracks), len({r['event'] for r in tracks})))
    print("  %-26s %7s %7s %9s   %s" % ("(end_tol, kink, gap)", "cands", "events", "sum_KE", "worst single event"))
    for etol, kink, gap in POINTS:
        sel = [r for r in tracks
               if r["gap"] <= gap and r["cend"] <= etol and r["rend"] <= etol and r["kink"] <= kink]
        per = {}
        for r in sel:
            per[r["event"]] = per.get(r["event"], 0.0) + r["ke"]
        worst = max(per.items(), key=lambda kv: kv[1]) if per else (0, 0.0)
        lab = "proximity only" if etol > 1e8 else "%.0fcm / %.0fdeg / %.0fcm" % (etol, kink, gap)
        mark = "   <= C++ default" if (etol, kink, gap) == (10.0, 30.0, 5.0) else ""
        print("  %-26s %7d %7d %9.1f   evt %d +%.1f MeV%s"
              % (lab, len(sel), len(per), sum(r["ke"] for r in sel), worst[0], worst[1], mark))

    print("\n  every candidate, sorted by kink (the discriminator):")
    print("  %-8s %-8s %-6s %8s %8s %7s %8s %8s %8s %9s"
          % ("event", "seg", "pdg", "len_cm", "ke_mev", "gap", "cand_end", "ref_end", "kink", "d_mainvtx"))
    for r in sorted(tracks, key=lambda r: r["kink"]):
        flag = "  <-- 72786 cosmic" if r["event"] == 72786 else ""
        print("  %-8d %-8d %-6d %8.1f %8.1f %7.2f %8.2f %8.2f %8.1f %9.1f%s"
              % (r["event"], r["seg"], r["pdg"], r["length"], r["ke"], r["gap"],
                 r["cend"], r["rend"], r["kink"], r["dmv"], flag))

    # ---------------- class B ----------------
    # The tape's own verdict needs the knob ON (keep = knob && gap <= cut), so on
    # a knob-OFF census arm it is always "skip".  Recompute the keep set from the
    # gap column, which the tape emits either way.
    KEEP_GAP = 20.0
    keep = [r for r in conn4 if 0 <= r["gap"] <= KEEP_GAP]
    print("\nCLASS B -- conn-4 showers, gap to the main cluster.  (These arms ran "
          "knob-OFF, so the keep set is recomputed from the gap column at the "
          "%.0f cm default, not read from the tape verdict.)" % KEEP_GAP)
    print("  %d conn-4 shower(s) in %d event(s), %.0f MeV total"
          % (len(conn4), len({r['event'] for r in conn4}), sum(r["ke"] for r in conn4)))
    for lab, lo, hi in (("<5 cm", 0, 5), ("5-20", 5, 20), ("20-50", 20, 50),
                        ("50-150", 50, 150), ("150+", 150, 1e9)):
        sub = [r for r in conn4 if lo <= r["gap"] < hi]
        print("    %-8s: %4d shower(s) %8.1f MeV" % (lab, len(sub), sum(r["ke"] for r in sub)))
    print("  KEEP at the %.0f cm default: %d shower(s) in %d event(s), %.1f MeV"
          % (KEEP_GAP, len(keep), len({r["event"] for r in keep}), sum(r["ke"] for r in keep)))
    print("  %-8s %-8s %-6s %9s %8s %8s" % ("event", "node", "pdg", "ke_mev", "len_cm", "gap_cm"))
    for r in sorted(keep, key=lambda r: -r["ke"]):
        print("  %-8d %-8d %-6d %9.1f %8.1f %8.2f"
              % (r["event"], r["node"], r["pdg"], r["ke"], r["length"], r["gap"]))

    if args.tsv:
        out = args.tsv if os.path.isabs(args.tsv) else os.path.join(SX, args.tsv)
        with open(out, "w") as fh:
            fh.write("class\tevent\tid\tcluster\tpdg\tscore\tlength_cm\tke_mev\tgap_cm\t"
                     "cand_end_cm\tref_end_cm\tkink_deg\td_mainvtx_cm\n")
            for r in sorted(tracks, key=lambda r: (r["event"], r["seg"])):
                fh.write("A\t%d\t%d\t%s\t%d\t%.3f\t%.1f\t%.1f\t%.2f\t%.2f\t%.2f\t%.1f\t%.1f\n"
                         % (r["event"], r["seg"], r["cluster"], r["pdg"], r["score"],
                            r["length"], r["ke"], r["gap"], r["cend"], r["rend"],
                            r["kink"], r["dmv"]))
            for r in sorted(conn4, key=lambda r: (r["event"], r["node"])):
                fh.write("B\t%d\t%d\t%s\t%d\t\t%.1f\t%.1f\t%.2f\t\t\t\t\n"
                         % (r["event"], r["node"], r["cluster"], r["pdg"],
                            r["length"], r["ke"], r["gap"]))
        print("\nwrote %s (%d rows)" % (out, len(tracks) + len(conn4)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
