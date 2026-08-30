#!/usr/bin/env python3
"""doc pr/131 -- the pr/128 denominator, measured.

doc pr/128 shipped four PF/kine completeness knobs and then recorded, in its
own "Still open" list, that it could not state a fraction:

    "Every completeness claim so far is anecdotal because the same_cluster
     gate hides the audit line as well as the pools, so there is NO
     DENOMINATOR: we cannot say what fraction of near-candidate reconstructed
     charge is missing from kine_reco_Enu, only that these 13 objects
     existed.  Measuring that denominator (debug-gated, no knob, no predicate
     filter) is the prerequisite for deciding whether more completeness
     rounds are worth running."

This script consumes the census that measures it: the `PFDENOM` / `PFDENOM_SUM`
lines emitted by MultiAlgBlobClustering.cxx under WCT_PFDENOM_DEBUG (stderr
only, no knob, no gate relaxed).

Five mutually exclusive buckets on the SUM line, because conn4_skip_segs is
tested BEFORE same_cluster in every pool, so "conn-4" and "cross-cluster"
overlap and a single class string would not add up:

    drawn    in used_segs, not conn-4          -- the PF tree shows it
    conn4    in conn4_skip_segs                -- pr/128 class B; SPLIT below
             by main-cluster membership, because the owner already drew that
             line when approving 105074 (pr/128): main-cluster membership is a
             sufficient admission rule, vertex reachability is NOT required.
             conn-4 material INSIDE the main cluster is the candidate's;
             conn-4 in a genuinely distant cluster (conn-4 means >80 cm,
             NeutrinoShowerClustering.cxx:3733) is correctly skipped.
    extra    a late pool drew it without       -- pr/93 confident-track,
             inserting into used_segs             pr/123 guard-freed,
                                                  pr/128 near-cross-cluster
    audited  unclaimed, SAME cluster           -- the pr/65 audit line names it
    hidden   unclaimed, CROSS cluster          -- named NOWHERE (the gap)

CURRENCY WARNING, stated once and repeated in the output.  Segment KE
(`particle_info()->kinetic_energy()`) is NOT the currency of
`kine_reco_Enu` -- Enu carries shower charge-based energy, the pr/101 mass
rules and long-muon range KE.  Every ratio computed inside the bucket family
is like-for-like (segment KE throughout); the Enu column is printed BESIDE
it as a scale reference and is explicitly NOT a denominator.

READ-ONLY over the census arms and the calib dumps.

Repro:
  cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  ./scripts/pr131_denominator.py <arm-glob> [<arm-glob>...] > docs/pr/pr131-denominator.txt
"""
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)

DEFAULT_ARMS = ["work-pr131-denom98-*", "work-pr131-denom141-*"]
NEAR_CUTS = [0.5, 2, 5, 10, 20, 50]

RE_SUM = re.compile(r"PFDENOM_SUM nseg=(\d+)((?: \w+=[\d.]+/[\d.-]+/[\d.-]+)+)")
RE_KV = re.compile(r"(\w+)=([\d.]+)/([\d.-]+)/([\d.-]+)")
RE_SEG = re.compile(r"PFDENOM (\S.*)$")
MULTICALL = {}


def parse_seg(line):
    d = {}
    for tok in line.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            d[k] = float(v) if ("." in v or v.lstrip("-").isdigit()) else v
        except ValueError:
            d[k] = v
    return d


def scan_arm(pat):
    """-> {event: (sums, [segrows], enu)}"""
    out = {}
    for d in sorted(glob.glob(os.path.join(SX, pat))):
        for ed in sorted(glob.glob(os.path.join(d, "pr_evt*"))):
            ev = int(os.path.basename(ed).replace("pr_evt", ""))
            log = os.path.join(ed, "stdout.log")
            if not os.path.exists(log):
                continue
            # fill_bee_pf_tree runs once per evaluated candidate, and 3 of the
            # 239 events evaluate two.  Both calls' nodes accumulate into the
            # SAME Bee bundle (`if (out_particles)`), so the totals ADD -- but
            # a segment reachable from both graphs must not be counted twice,
            # so per-segment rows are de-duplicated on seg id.
            sums, segs, seen, ncall = None, [], set(), 0
            for ln in open(log, errors="replace"):
                if "PFDENOM_SUM" in ln:
                    m = RE_SUM.search(ln)
                    if m:
                        ncall += 1
                        one = {k: (float(n), float(k2), float(l))
                               for k, n, k2, l in RE_KV.findall(m.group(2))}
                        if sums is None:
                            sums = one
                        else:
                            for k, v in one.items():
                                a = sums.get(k, (0.0, 0.0, 0.0))
                                sums[k] = (a[0] + v[0], a[1] + v[1], a[2] + v[2])
                elif "PFDENOM seg=" in ln:
                    r = parse_seg(ln)
                    key = (r.get("seg"), r.get("bucket"))
                    if key in seen:
                        continue
                    seen.add(key)
                    segs.append(r)
            if sums is None:
                continue
            MULTICALL[ev] = ncall
            enu = None
            cal = os.path.join(ed, "calib-pr-evt%d.json" % ev)
            if os.path.exists(cal):
                try:
                    enu = float((json.load(open(cal)).get("kine") or {})
                                .get("kine_reco_Enu") or 0.0)
                except Exception:
                    enu = None
            out[ev] = (sums, segs, enu)
    return out


def main():
    pats = sys.argv[1:] or DEFAULT_ARMS
    ev = {}
    for p in pats:
        ev.update(scan_arm(p))
    if not ev:
        sys.exit("no PFDENOM census found under: %s\n"
                 "  (a probe that emits nothing is indistinguishable from a probe\n"
                 "   that is not wired -- check WCT_PFDENOM_DEBUG was set and that\n"
                 "   pf_orphan_audit_only/pf_shower_vertex_barrier are on in the arm)"
                 % ", ".join(pats))

    BK = ["drawn", "conn4", "extra", "audited", "hidden"]
    # conn-4 splits on main-cluster membership (owner line, pr/128 / 105074).
    # The per-segment lines already carry xclus, so this needs no re-run.
    c4 = {"conn4_main": [0.0, 0.0, 0.0], "conn4_far": [0.0, 0.0, 0.0]}
    for _, (_, segs, _) in ev.items():
        for r in segs:
            if r.get("bucket") != "conn4":
                continue
            k = "conn4_far" if r.get("xclus") == 1 else "conn4_main"
            c4[k][0] += 1
            c4[k][1] += r.get("ke_mev", 0.0)
            c4[k][2] += r.get("len_cm", 0.0)
    tot = {b: [0.0, 0.0, 0.0] for b in BK}
    for _, (sums, _, _) in ev.items():
        for b in BK:
            n, k, l = sums.get(b, (0.0, 0.0, 0.0))
            tot[b][0] += n; tot[b][1] += k; tot[b][2] += l

    print("=" * 78)
    print("pr/131 -- THE pr/128 DENOMINATOR, over %d events" % len(ev))
    print("=" * 78)
    mc = sorted(e for e, n in MULTICALL.items() if n > 1)
    print("  fill_bee_pf_tree calls: 1 per event except %d (%s), which evaluate"
          % (len(mc), ", ".join(str(x) for x in mc)))
    print("  two candidates; both bundles accumulate into the same PF output, so")
    print("  their sums ADD and per-segment rows are de-duplicated on seg id.")
    print("\nCURRENCY: every number in the bucket family is segment KE (MeV).")
    print("kine_reco_Enu is a DIFFERENT quantity (shower charge energy + pr/101")
    print("mass rules + long-muon range) and is printed only as a scale")
    print("reference -- never as the denominator of a bucket ratio.\n")

    kall = sum(tot[b][1] for b in BK)
    nall = sum(tot[b][0] for b in BK)
    print("  %-9s %8s %14s %8s %12s %8s" % ("bucket", "nseg", "KE MeV", "%KE", "len cm", "%len"))
    lall = sum(tot[b][2] for b in BK)
    for b in BK:
        n, k, l = tot[b]
        print("  %-9s %8.0f %14.1f %7.2f%% %12.1f %7.2f%%"
              % (b, n, k, 100 * k / kall if kall else 0, l, 100 * l / lall if lall else 0))
    print("  %-9s %8.0f %14.1f" % ("TOTAL", nall, kall))
    print("\n  conn-4 split on MAIN-CLUSTER membership (owner line, pr/128 / 105074:")
    print("  main-cluster membership admits, vertex reachability is not required):")
    for k in ("conn4_main", "conn4_far"):
        n, kk, l = c4[k]
        print("    %-11s %8.0f %14.1f %7.2f%% of all KE %12.1f cm"
              % (k, n, kk, 100 * kk / kall if kall else 0, l))
    print("    -> conn4_main is INSIDE near-candidate charge by that line;")
    print("       conn4_far is correctly skipped (conn-4 means >80 cm).")

    unc = tot["audited"][1] + tot["hidden"][1]
    # The owner-grounded near-candidate universe: everything the candidate has
    # a claim on = drawn + extra + audited + hidden + conn4_main.  conn4_far
    # is excluded by the same owner line, not by convenience.
    universe = (tot["drawn"][1] + tot["extra"][1] + tot["audited"][1]
                + tot["hidden"][1] + c4["conn4_main"][1])
    missing = tot["audited"][1] + tot["hidden"][1] + c4["conn4_main"][1]
    print("\n--- THE HEADLINE ---")
    print("  Of all reconstructed segment KE in the PR graph, the fraction that")
    print("  reaches no PF output and no audit line at all (bucket `hidden`):")
    print("      %.1f MeV of %.1f MeV = %.2f%%"
          % (tot["hidden"][1], kall, 100 * tot["hidden"][1] / kall if kall else 0))
    print("  Of the UNCLAIMED population only (audited + hidden), the hidden share:")
    print("      %.1f of %.1f MeV = %.1f%%  (%.0f of %.0f segments)"
          % (tot["hidden"][1], unc, 100 * tot["hidden"][1] / unc if unc else 0,
             tot["hidden"][0], tot["audited"][0] + tot["hidden"][0]))
    print("\n  THE NUMBER doc pr/128 ASKED FOR -- of the near-candidate")
    print("  reconstructed segment KE (drawn + extra + audited + hidden +")
    print("  conn4_main; conn4_far excluded by the owner's own >80 cm line),")
    print("  the share that reaches NO PF output:")
    print("      %.1f MeV of %.1f MeV = %.2f%%"
          % (missing, universe, 100 * missing / universe if universe else 0))
    print("      of which hidden (counted nowhere at all) %.1f MeV = %.2f%%"
          % (tot["hidden"][1], 100 * tot["hidden"][1] / universe if universe else 0))

    print("\n--- NEAR-CANDIDATE cut sweep (dmin = gap to what the tree draws) ---")
    print("  %6s %25s %8s %25s" % ("cut cm", "hidden within cut", "in evts", "audited within cut"))
    for c in NEAR_CUTS:
        h = [(e, r) for e, (_, segs, _) in ev.items() for r in segs
             if r.get("bucket") == "hidden" and 0 <= r.get("dmin_cm", -1) <= c]
        a = [r for _, (_, segs, _) in ev.items() for r in segs
             if r.get("bucket") == "audited" and 0 <= r.get("dmin_cm", -1) <= c]
        print("  %6.1f %8d seg %10.1f MeV %8d %8d seg %10.1f MeV"
              % (c, len(h), sum(r.get("ke_mev", 0) for _, r in h),
                 len({e for e, _ in h}),
                 len(a), sum(r.get("ke_mev", 0) for r in a)))

    print("\n--- per-event: where the hidden charge is ---")
    rows = []
    for e, (sums, segs, enu) in ev.items():
        hk = sums.get("hidden", (0, 0, 0))[1]
        if hk <= 0:
            continue
        near = sum(r.get("ke_mev", 0) for r in segs
                   if r.get("bucket") == "hidden" and 0 <= r.get("dmin_cm", -1) <= 10)
        rows.append((hk, e, sums, near, enu))
    rows.sort(reverse=True)
    nev_hidden = len(rows)
    print("  %d of %d events carry ANY hidden KE" % (nev_hidden, len(ev)))
    print("  %8s %11s %11s %12s %10s" % ("event", "hidden MeV", "<=10cm MeV", "reco_Enu MeV", "hid/Enu"))
    for hk, e, sums, near, enu in rows[:20]:
        print("  %8d %11.1f %11.1f %12s %9s"
              % (e, hk, near, "%.1f" % enu if enu else "-",
                 "%.1f%%" % (100 * hk / enu) if enu else "-"))
    if len(rows) > 20:
        print("  ... %d more" % (len(rows) - 20))

    print("\n--- what the hidden population IS (all events) ---")
    hid = [r for _, (_, segs, _) in ev.items() for r in segs if r.get("bucket") == "hidden"]
    by = collections.Counter()
    kby = collections.Counter()
    for r in hid:
        by[int(r.get("pdg", 0))] += 1
        kby[int(r.get("pdg", 0))] += r.get("ke_mev", 0)
    print("  pdg by count  : %s" % dict(by.most_common()))
    print("  pdg by KE MeV : %s" % {k: round(v, 1) for k, v in kby.most_common()})
    lens = sorted((r.get("len_cm", 0) for r in hid), reverse=True)
    if lens:
        print("  length cm: max %.1f  median %.1f  n(>10cm) %d  n(>50cm) %d"
              % (lens[0], lens[len(lens) // 2],
                 sum(1 for l in lens if l > 10), sum(1 for l in lens if l > 50)))
    nod = sum(1 for r in hid if r.get("dirsign") == 0)
    nof = sum(1 for r in hid if r.get("nfits") == 0)
    print("  would ALSO be dropped by the display filters: dirsign==0 %d, empty fits %d"
          % (nod, nof))
    gf = [r for r in hid if r.get("gf") == 1]
    print("  carrying kPass4GuardFreed but NOT drawn: %d (%.1f MeV)"
          % (len(gf), sum(r.get("ke_mev", 0) for r in gf)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
