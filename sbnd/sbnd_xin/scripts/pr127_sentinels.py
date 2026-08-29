#!/usr/bin/env python3
"""doc pr/127 sec 5 -- sentinel assertions for shipped, owner-approved fixes.

Why this exists: doc pr/127 found that the pr/93 round-4 fix for SBND
18264-137238 (straight_cont_cross_cluster) stopped firing ~2026-08-21 when
the Q/L era changed under it, and stayed dead in production for ~10 days.
NOTHING in the standing validation bar could see it: a knob A/B compares two
arms that both have the fix on, the event carries no vtx105 label, and in the
pr/125 Bee pair its row read "no-op verified" (true of the flip, silent about
the event).

A sentinel is the missing check: for each shipped fix, the named event it was
shipped for, and a property of the CURRENT production arm that must still
hold.  Cheap (reads arms already on disk), and it fails loudly.

Assertion kinds (energy-valued ones read the PF node text "name  NNN MeV",
so they tolerate the few-MeV drift every round produces and still catch the
structural loss a dead fix causes):
  pf_node_ge   <name> <MeV>  some PF node of that particle name is >= MeV
  pf_node_lt   <name> <MeV>  EVERY PF node of that name is < MeV
  pf_contains  <text>        PF node text contains this substring
  pf_absent    <text>        ... does not
  log_absent   <text>        the per-event run log does NOT contain <text>
  shower_max_ge <|pdg|> <MeV>  max kine_best over calib showers of that |pdg|
  enu_between  <lo> <hi>     kine_reco_Enu in [lo, hi] MeV
  nshowers     <lo> <hi>     len(calib showers) in [lo, hi]
  log_contains <text>        any per-event log in the arm contains this text

Usage:
  ./scripts/pr127_sentinels.py --arms 'work-pr127r1-flipS98-*' 'work-pr127r1-flipS141-*'
  ./scripts/pr127_sentinels.py --arms 'work-pr125r1-flipK5*'          # expect the 137238 FAIL
  ./scripts/pr127_sentinels.py --list

Exit code: 0 = every applicable sentinel PASSed, 1 = at least one FAILed.
A sentinel whose event is not present in the given arms is reported SKIP and
does not fail the run.

RE-BASELINE WARNING: the energy thresholds below are absolute MeV, so a round
that changes the ENERGY SCALE rather than the structure will move them all.
The queued `kine_shower_fudge_factor` 0.80 -> 0.84 flip (doc pr/126) is
exactly that: ~5% on every EM energy.  52693's `pf_node_lt e- 175` sits at
158 today (166 after), and 69314's shower count is exposed from the other
side because K5's own gate is `kine < 10 MeV` -- a 5% bump pushes borderline
crumbs over the cap and the count rises.  When that flip lands, re-measure
and re-baseline these numbers; a FAIL in that round is churn, not regression.
"""
import argparse
import glob
import json
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)

# (event, doc, what shipped, [(kind, *args), ...])
# Thresholds are set from the measured post-fix value and the measured
# pre-fix value, and placed between them -- so the sentinel fails when the
# fix dies, not when the number drifts a few MeV.
SENTINELS = [
    (137238, "pr/93 r4 + pr/127", "sccc demote+bridge: the cross-cluster muon body joins PF",
     # pre-fix: no mu- at all under the stem (the body is in cluster 7 and invisible);
     # post-fix: mu- 207 (body) + mu- 66 (continuation).
     [("pf_node_ge", "mu-", 150.0), ("log_contains", "sccc demote")]),
    (37112, "pr/125 K3", "samevtx track-frag absorb: gamma + proton stem = ONE shower",
     # pre-fix: gamma 549 + a separate 469 MeV proton-typed shower.
     [("shower_max_ge", 11, 700.0), ("pf_node_lt", "proton", 400.0)]),
    (69314, "pr/125 K5", "satellite absorb: the vertex-connected crumbs fold in",
     # pre-fix 25 showers, post-fix 18.
     [("nshowers", 14, 20)]),
    (94392, "pr/125 guard", "pass3_cone track guard: the 305 MeV fake electron is gone",
     [("pf_node_lt", "e-", 250.0)]),
    (52693, "pr/125 guard", "pass3_cone track guard: 183 -> 153 MeV e- + released muon",
     [("pf_node_lt", "e-", 175.0)]),
    (77328, "pr/125 guard", "pass3_cone track guard: 180 MeV EM shower -> re-rooted proton",
     [("pf_node_lt", "e-", 130.0)]),
    (173819, "pr/125 guard", "pass3_cone track guard: 301 MeV EM shower -> re-rooted proton",
     [("pf_node_lt", "e-", 200.0)]),
    (171572, "pr/123 r2", "pf_orphan_guard_freed: the guard-freed muon returns as a PF root",
     [("log_contains", "pf-orphan-guard-freed"), ("pf_node_ge", "mu-", 250.0)]),
    (393505, "pr/123 r2", "pf_orphan_guard_freed: same, second firing event",
     [("log_contains", "pf-orphan-guard-freed")]),
    (348471, "pr/93 r4 + pr/121", "detach_track_stem: no 719 MeV proton-typed aggregate",
     [("pf_node_lt", "proton", 600.0), ("shower_max_ge", 11, 300.0)]),
    (315167, "pr/93 r4", "orphan confident track: the 150.7cm proton is counted + shown",
     [("pf_node_ge", "proton", 500.0)]),   # not in the standard manifests -> SKIP
    # ---- doc pr/128 (PF/kine completeness).  Both knobs carry a distance or
    # angle threshold tuned to measured geometry, the exposure class this
    # registry exists for -- and pr/128's own census found a pr/123 knob had
    # started overlapping a new pool, so overlap is watched here too.
    (105074, "pr/128 class B", "conn4_near: the two main-cluster pdg-13 showers are shown + counted",
     # pre-fix: both invisible (pr/74 conn3_unreachable stamped them conn-4 at
     # anchor_dis 119 cm); post-fix mu- 215 + mu- 162, Enu 1188.7 -> 1565.8.
     [("pf_node_ge", "mu-", 180.0), ("log_contains", "pr128 pf-conn4-near: KEEP")]),
    (55740, "pr/128 class A", "near cross-cluster track: the 123.1cm muon joins PF + kine",
     # pre-fix: absent from both (cross-cluster, score 100, so neither the pr/93
     # confident-track class nor the pr/123 flag class reaches it).
     # post-fix mu- 300 MeV under a pseudo-neutron; Enu 488.5 -> 894.7.
     [("pf_node_ge", "mu-", 250.0), ("log_contains", "pr128 pf-orphan-near-cross-cluster")]),
    (72786, "pr/128 class A", "CONTROL: the continuation terms keep the cosmics OUT",
     # This event's four candidates come from two large off-vertex clusters
     # (148 and 102 cm extent, 35 and 77 cm off the vertex).  On proximity
     # alone they were admitted and put +1151 MeV on a 701 MeV candidate.  A
     # future loosening that re-admits them fires this line.
     # The legit maximum mu- here is 238.8 MeV; the four cosmics are 268.9,
     # 281.7 and 344.0 MeV, so 250 sits between the two populations.
     [("log_absent", "pr128 pf-orphan-near-cross-cluster"), ("pf_node_lt", "mu-", 250.0)]),

    # ---- doc pr/130: the doc-84 long-muon / MCS family.  Every one of these
    # is a shipped, SBND-PRODUCTION-ON fix whose target event is in NO standard
    # manifest (em114 / em114c), so before this round the whole family was
    # invisible to every gate we run -- the exact exposure doc pr/127 was
    # written about.  These events only execute if the arm list includes
    # work-sent<TAG>-mcp{1,2}k from scripts/pr130_sentinel_arm.sh; otherwise
    # they report SKIP, which reads like a pass.
    # Post-fix values measured 2026-08-29 in work-sent130-*; pre-fix values are
    # doc 84's own, and each threshold sits between the two.
    (497311, "doc 84 r1", "long-muon range fallback: the 332.8 cm unbroken muon gets a range KE",
     # pre-fix 508.5, post-fix 766.3 -- and note this is the SHOWER kine_best,
     # not the PF node (which reads 473 MeV here; they are different quantities).
     [("shower_max_ge", 13, 700.0),
      ("log_contains", "long_muon_range_empty_chain_fallback: recomputed kinematics")]),
    (66366, "doc 84 r1", "OUTCOME ONLY -- the 300.6 cm chain / 692.2 MeV result",
     # CAVEAT, measured doc pr/130: this entry does NOT guard
     # long_muon_stub_bridge_len.  Re-run with the knob at its legacy 6.0
     # (compiled-config proof: 7.5 in prod vs 6 in work-sent130neg2-*, and
     # members_geometry / cathode_bridge / range_fallback all off as well) the
     # event STILL reads nseg_chain=4 L_cm=300.6 range=692.2 -- identical.  So
     # doc 84 r1's P3 rescue is now produced by another route on its own target
     # event, and no event currently guards that knob.  Kept as an outcome
     # sentinel (it still catches a regression of the RESULT from any cause);
     # see doc pr/130 Part 2 "the masked knob".
     [("pf_node_ge", "mu-", 650.0), ("log_contains", "nseg_chain=4 L_cm=300.6")]),
    (313847, "doc 84 r2", "long_muon_members_geometry: out-of-chain muon members join the range",
     # pre-fix 547.5, post-fix 602.2.
     [("pf_node_ge", "mu-", 575.0)]),
    (281595, "doc 84 r2", "members_geometry: the 221 cm continuation beats the 13.9 cm stub",
     # pre-fix 750.1, post-fix 808.5.
     [("pf_node_ge", "mu-", 780.0)]),
    (53793, "doc 84 r2", "long_muon_cathode_bridge: two half-showers become one muon",
     # pre-fix two halves 367 + 528, post-fix 912.9 -- 700 is above the larger half.
     [("pf_node_ge", "mu-", 700.0), ("log_contains", "long_muon_cathode_bridge:")]),
    (177536, "doc 84 r2.1", "cathode bridge + BFS main-vertex guard + start re-seat",
     # pre-fix 679, post-fix 907.0.  The proton line guards round 2.1 itself:
     # before the guard the bridge BFS walked THROUGH the main vertex and ate
     # the proton prong.
     [("pf_node_ge", "mu-", 800.0), ("pf_node_ge", "proton", 100.0)]),
    (77978, "doc 84 r2.1", "same, the owner's scan event (proton prong must survive)",
     # pre-fix 337.5, post-fix 388.2.
     [("pf_node_ge", "mu-", 365.0), ("pf_node_ge", "proton", 100.0)]),
    (172794, "doc 84 r4", "long_muon_cathode_bridge_lever 5 -> 15 cm",
     # The accept line is the primary assertion -- it is binary and exact.
     # post-fix chain 298.7 cm / range 687.8 MeV.
     [("log_contains", "long_muon_cathode_bridge: absorb bare chain into sid=1"),
      ("pf_node_ge", "mu-", 600.0)]),
    (347890, "doc 84 r4", "long_muon_cathode_bridge_track_partner (far half mis-PID'd 211)",
     # pre-fix 429.0 (measured in the knobs-off control), post-fix 488.7.
     # The first draft used 400 here and the negative control PASSED it at
     # 429 -- only the log line caught the dead knob.  460 sits between.
     [("log_contains", "long_muon_cathode_bridge: merge shower sid=5"),
      ("pf_node_ge", "mu-", 460.0)]),
    (67026, "doc 84 r4", "long_muon_cathode_bridge_short_gap 8 cm (gap-angle waived, gap>0)",
     # post-fix chain 325.2 cm / range 748.6 MeV.
     [("log_contains", "long_muon_cathode_bridge: absorb bare chain into sid=0"),
      ("pf_node_ge", "mu-", 650.0)]),

    # ---- doc pr/130: pr/124's headline win had no sentinel either.  It is in
    # em114c, so unlike the doc-84 family it does execute under the standard
    # 141 arms -- it was simply never registered.
    (406125, "pr/124 A", "shower_pass4_prune_gap2=25 gap-band tier-2 prune (qF1 0.097 -> 1.000)",
     # The metric that moved is an EM label score with no calib scalar, so the
     # assertion is that the prune still SHEDS -- the re-seed line is emitted
     # only when a band component is actually taken out.
     [("log_contains", "pr124 pass4_prune2:"),
      ("log_contains", "band component(s) re-seeded")]),
]


def pf_texts(arm, event):
    z = os.path.join(arm, "pr_evt%d" % event, "mabc-pr.zip")
    if not os.path.exists(z):
        return None
    out = []

    def walk(n):
        out.append(n.get("text", ""))
        for c in n.get("children", []):
            walk(c)

    with zipfile.ZipFile(z) as zf:
        for root in json.loads(zf.read("data/0/0-mc.json")):
            walk(root)
    return out


def calib(arm, event):
    p = os.path.join(arm, "pr_evt%d" % event, "calib-pr-evt%d.json" % event)
    return json.load(open(p)) if os.path.exists(p) else None


def find_arm(arms, event):
    for pat in arms:
        for root in sorted(glob.glob(os.path.join(SX, pat)) or glob.glob(pat)):
            if os.path.isdir(os.path.join(root, "pr_evt%d" % event)):
                return root
    return None


def pf_energies(arm, event, name):
    """MeV values of every PF node rendered as '<name>  <NNN> MeV'."""
    out = []
    for t in pf_texts(arm, event) or []:
        m = re.match(r"(\S+)\s+(-?\d+(?:\.\d+)?)\s+MeV\s*$", t.strip())
        if m and m.group(1) == name:
            out.append(float(m.group(2)))
    return out


def check(arm, event, assertion):
    kind = assertion[0]
    if kind in ("pf_node_ge", "pf_node_lt"):
        vals = pf_energies(arm, event, assertion[1])
        hit = any(v >= assertion[2] for v in vals)
        ok = hit if kind == "pf_node_ge" else not hit
        return ok, "%s %s %.0f MeV (seen: %s)" % (
            kind, assertion[1], assertion[2],
            ", ".join("%.0f" % v for v in sorted(vals, reverse=True)[:4]) or "none")
    if kind == "shower_max_ge":
        j = calib(arm, event)
        vals = [s["kine_best"] for s in (j or {}).get("showers", [])
                if abs(s.get("particle_id", 0)) == assertion[1]]
        mx = max(vals) if vals else None
        ok = mx is not None and mx >= assertion[2]
        return ok, "max |pdg|=%d shower %s want >= %.0f MeV" % (
            assertion[1], "%.1f" % mx if mx is not None else "none", assertion[2])
    if kind in ("pf_contains", "pf_absent"):
        texts = pf_texts(arm, event) or []
        hit = any(assertion[1] in t for t in texts)
        ok = hit if kind == "pf_contains" else not hit
        return ok, "%s '%s' (%d PF nodes)" % (kind, assertion[1], len(texts))
    if kind == "enu_between":
        j = calib(arm, event)
        e = j["kine"]["kine_reco_Enu"] if j else None
        ok = e is not None and assertion[1] <= e <= assertion[2]
        return ok, "Enu=%s want [%.0f, %.0f]" % (
            "%.1f" % e if e is not None else "?", assertion[1], assertion[2])
    if kind == "nshowers":
        j = calib(arm, event)
        n = len(j["showers"]) if j else None
        ok = n is not None and assertion[1] <= n <= assertion[2]
        return ok, "showers=%s want [%d, %d]" % (n, assertion[1], assertion[2])
    if kind in ("log_contains", "log_absent"):
        pat = os.path.join(arm, "pr_evt%d" % event, "*.log")
        hit = any(assertion[1] in open(f, errors="replace").read()
                  for f in glob.glob(pat))
        ok = hit if kind == "log_contains" else not hit
        return ok, "%s '%s'" % (kind, assertion[1])
    return False, "unknown assertion %s" % kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["work-pr127r1-flipS98-*",
                                                  "work-pr127r1-flipS141-*"])
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for ev, doc, what, _ in SENTINELS:
            print("%-8d %-18s %s" % (ev, doc, what))
        return 0

    nfail = nskip = npass = 0
    for ev, doc, what, assertions in SENTINELS:
        arm = find_arm(args.arms, ev)
        if not arm:
            print("SKIP %-8d %-18s (no arm has this event)" % (ev, doc))
            nskip += 1
            continue
        verdicts = [check(arm, ev, a) for a in assertions]
        ok = all(v[0] for v in verdicts)
        print("%s %-8d %-18s %s" % ("PASS" if ok else "FAIL", ev, doc, what))
        print("        arm=%s" % os.path.basename(arm))
        for (v, msg), a in zip(verdicts, assertions):
            print("        [%s] %s" % ("ok" if v else "XX", msg))
        npass += 1 if ok else 0
        nfail += 0 if ok else 1
    print("\n%d PASS, %d FAIL, %d SKIP" % (npass, nfail, nskip))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
