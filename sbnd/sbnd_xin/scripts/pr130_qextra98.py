#!/usr/bin/env python3
"""doc pr/130 item 1b part 6 -- the 98-set's own 22 condemned segments.

Part 1 of pr130-qextra-refresh.md measured the affirmative q_extra pool on
BOTH manifests, then analysed only the 141-set's 22 segments and left one
sentence about the other side: "The 98-set adds a further 22 segments /
7.056e6 on its own labels."  Those 22 were never looked at.  This script
looks at them, and asks the three questions Parts 1-3 asked of the 141-set:

  1. WHAT are they -- length / pdg / cluster / absorber, and how does that
     distribution compare to the 141-set's?  (The 141-set's pool is led by a
     110 cm pdg-13 muon; if the 98-set's is not, the two pools are not the
     same failure and a fix for one cannot be assumed to reach the other.)
  2. Is any of them REACHABLE by a shipped guard?  Every guard in this
     campaign's over-clustering family is length-gated -- pr/123's
     shower_pass4_track_guard_len (>50 cm AND track-like), pr/130's two new
     seats (50 / 15 cm), and the walk-add guard's own
     segment_is_straight_long_track floor (>10 cm, and pdg-11 exempt unless
     em_straight_min_len is passed, which only examine_shower_1 does).  So
     the length distribution IS the reachability answer.
  3. Is the charge really EXTRA to the event, or does a sibling shower in the
     same event want it?  em117_score.py scores each labelled shower
     independently, so a segment the scanner moved from shower A to shower B
     lands in A's `extra` AND in B's `miss`.  That is a re-home, not an
     over-cluster, and counting it as q_extra inflates the pool.  Run on BOTH
     sets so the comparison is like-for-like (an asymmetry claimed on one set
     alone would just be a property of the scorer).

Then asks what it COSTS: two of the three events are ncpi0, the scanner's
note on both says the condemned charge is a second gamma, and the dump
carries the reconstructed pi0 pairing (`kine_pio_*`).  So the consequence of
the merge is readable without any truth file.

Also re-runs Part 2's label-store-vs-live-run attribution check, which was
only ever done for the 141-set: `marks_detail[...]["absorbed_by"]` is what the
label store recorded at scan time, and the census arms on disk say what the
current binary actually does.

READ-ONLY over em_labels/ (M13), the calib dumps and the census stdout logs.
No arms launched, no knobs, nothing shipped.

Repro:
  cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  ./scripts/pr130_qextra98.py > docs/pr/pr130-qextra-98set.txt
"""
import collections
import copy
import csv
import glob
import importlib.util
import os
import json
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
EMD = os.path.join(SX, "em_display")

# Identical crossoff list to pr130_qmiss_rank.py / pr130_qextra_rank.py so the
# three rankings stay strictly comparable.
ADJUDICATED = {318769, 415278, 283515, 179369}

SETS = [("98", "em117-pr130q98-manifest.tsv", "emscan-0827",
         "docs/pr/pr130-98-score-prod.tsv", "emprep-pr130q98",
         ["work-pr130r1-probe98-*"]),
        ("141", "em114c-pr130q141-manifest.tsv", "emscan-0828-agent5",
         "docs/pr/pr130-141-score-prod.tsv", "emprep-pr130q141",
         ["work-pr130r1-probe141-*"])]

# Every length-gated decline this campaign has shipped, with the seat it sits
# at.  A condemned segment is "reachable" by one of these only if it arrives
# at that seat AND clears the floor.
# em_exempt: the seat never declines a pdg-11 segment however long it is.
# The three *_len guards test |pdg| in {13,211,2212} or a MIP-flat dQ/dx, and
# the walk-add guard early-outs on |pdg|==11 unless em_straight_min_len > 0 --
# which only the examine_shower_1 call site passes (PRShower.cxx:826-834).
SHIPPED_GUARDS = [
    ("pr123 shower_pass4_track_guard_len", "pass4_angle", 50.0, True),
    ("pr130 shower_pass4_prox_guard_len", "pass4_proximity", 50.0, True),
    ("pr130 shower_pass3_backfill_guard_len", "pass3_cone", 15.0, True),
    ("pr40r6 shower_absorb_track_guard", "*walk_add*", 10.0, True),
]

RE_ADD = re.compile(r"SHOWER_ABSORB ADD shower_start_seg=(\d+) seg=(\d+)")
RE_DIRECT = re.compile(r"SHOWER_ABSORB DIRECT site=(\S+) shower_start_seg=(\d+) seg=(\d+)")
RE_GEOM = re.compile(r"SHOWER_ABSORB PASS4_GEOM seg=(\d+) .*?tier=(\d+)")
# The walk-add ADD line carries no site of its own; the site is announced by
# the `SHOWER_ABSORB site=<seat> shower_start_seg=<id>` line that opens the
# walk.  Without this the label store's "from_vertices (walk_add)" would be
# compared against a bare "walk_add" and every row would read as MOVED.
RE_SITE = re.compile(r"SHOWER_ABSORB site=(\S+) shower_start_seg=(\d+)")


def load_scorer():
    cwd = os.getcwd()
    os.chdir(EMD)
    try:
        spec = importlib.util.spec_from_file_location("s117", "em117_score.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(cwd)


def census_log(ev, arms):
    for pat in arms:
        for d in sorted(glob.glob(os.path.join(SX, pat))):
            p = os.path.join(d, "pr_evt%d" % ev, "stdout.log")
            if os.path.exists(p):
                return p
    return None


def live_absorb(path):
    """-> {(seg, shower_start_seg): site}, {seg: pass4 tier}.  Every admit line
    names its own shower, so the (seg, shower) pairing is never inferred from
    running context; only the SEAT of a walk-add comes from the walk's opening
    line, and that line names the same shower."""
    sites, tiers = {}, {}
    seat = {}
    if not path:
        return sites, tiers
    for ln in open(path, errors="replace"):
        m = RE_SITE.search(ln)
        if m:
            seat[int(m.group(2))] = m.group(1)
            continue
        m = RE_DIRECT.search(ln)
        if m:
            sites[(int(m.group(3)), int(m.group(2)))] = m.group(1) + " (direct)"
            continue
        m = RE_ADD.search(ln)
        if m:
            shw = int(m.group(1))
            sites.setdefault((int(m.group(2)), shw),
                             "%s (walk_add)" % seat.get(shw, "?"))
            continue
        m = RE_GEOM.search(ln)
        if m:
            tiers[int(m.group(1))] = int(m.group(2))
    return sites, tiers


def pio_report(ev, arms, s117, man, prepdir, ins, condemned, shw):
    """What the reconstruction did with the pi0 in this event, and whether its
    chosen partner gamma is a fragment the scanner said belongs to the merged
    shower.  Read straight off the dump -- no truth file, no fit."""
    dump = s117.load_dump(man[ev]["dump"])
    k = dump.get("kine") or {}
    showers = [x for x in (dump.get("showers") or []) if isinstance(x, dict)]
    by_e = {round(float(x.get("kine_charge") or 0.0), 3): x for x in showers}
    parts = []
    for key in ("kine_pio_energy_1", "kine_pio_energy_2"):
        e = k.get(key)
        if e is None:
            continue
        m = min(showers, key=lambda x: abs(float(x.get("kine_charge") or 0.0) - float(e)),
                default=None)
        parts.append((float(e), m.get("id") if m else None))
    return k, parts, showers


# The two guard seats pr/130 Part 4 shipped and Part 5 flipped SBND ON.  Part
# 4 measured a 10-event blast radius that does not name any of these three,
# but a cited list is weaker than a diff, and the arms are on disk.
FLIP_ARMS = [("work-pr130r1-g1off-%s", "work-pr130r1-gs1on-%s")]


def strip_timers(d):
    """A raw cmp on these dumps reports a difference that is only a wall-clock
    field (vertex_scoreboard.dual_chain.off_ms).  M2 in miniature: judge the
    content, not the file."""
    d = copy.deepcopy(d)
    for v in (d.get("vertex_scoreboard") or {}).values():
        if isinstance(v, dict):
            for t in [t for t in v if t.endswith("_ms")]:
                v.pop(t)
    return d


def mirror_qmiss(tag, manf, labtag, tsv, prepdir, s117):
    """The mirror of Q3.  Q3 asked whether an affirmative q_extra segment is
    also some sibling shower's `miss`.  The same asymmetry could exist on the
    other side -- an affirmative q_miss segment that a SIBLING shower holds,
    which is again a re-home rather than a drop.  Left as an open item by the
    first cut of this doc; measured here so the corrected pool numbers have no
    loose end.  -> (n, q, n_total, q_total)"""
    cwd = os.getcwd()
    os.chdir(EMD)
    try:
        man = s117.load_manifest(manf)
        labs = s117.load_labels(labtag)
        ins_by, _ = marks(labs)
        srows = list(csv.DictReader(open(os.path.join(SX, tsv)), delimiter="\t"))
        extra_by = {}
        for r in srows:
            extra_by[(int(r["event"]), int(r["shower"]))] = \
                {int(x) for x in (r["extra"] or "").split(",") if x.strip()}
        n = nt = 0
        q = qt = 0.0
        per = collections.defaultdict(lambda: [0, 0.0])
        for r in srows:
            ev = int(r["event"])
            if ev in ADJUDICATED:
                continue
            shw = int(r["shower"])
            dump = s117.load_dump(man[ev]["dump"])
            if dump is None:
                continue
            _, seginfo, _ = s117.digest_dump(dump, s117.load_prep(ev, prepdir))
            ins = ins_by.get((ev, shw), set())
            for sg in [int(x) for x in (r["miss"] or "").split(",") if x.strip()]:
                if sg not in ins:
                    continue                       # weak miss, not affirmative
                nt += 1
                qt += seginfo.get(sg, {}).get("charge", 0.0)
                held = [b for (e2, b), ex in extra_by.items()
                        if e2 == ev and b != shw and sg in ex]
                if held:
                    n += 1
                    q += seginfo.get(sg, {}).get("charge", 0.0)
                    per[ev][0] += 1
                    per[ev][1] += seginfo.get(sg, {}).get("charge", 0.0)
        return n, q, nt, qt, dict(per)
    finally:
        os.chdir(cwd)


def marks(labs):
    ins, outs = {}, {}
    for ev, rec in labs.items():
        for shw, det in ((rec.get("em") or {}).get("marks_detail") or {}).items():
            m = det.get("marked") or {}
            ins[(ev, int(shw))] = {int(s) for s, v in m.items() if v.get("kind") == "in"}
            outs[(ev, int(shw))] = {int(s): v for s, v in m.items() if v.get("kind") == "out"}
    return ins, outs


def collect(tag, manf, labtag, tsv, prepdir, arms, s117):
    """-> rows (one per affirmative-q_extra segment) + per-event note."""
    cwd = os.getcwd()
    os.chdir(EMD)
    try:
        man = s117.load_manifest(manf)
        labs = s117.load_labels(labtag)
        ins_by, outs_by = marks(labs)
        srows = list(csv.DictReader(open(os.path.join(SX, tsv)), delimiter="\t"))
        # miss lists per (event, shower) -- for the cross-shower re-home check.
        miss_by = {}
        for r in srows:
            miss_by[(int(r["event"]), int(r["shower"]))] = \
                {int(x) for x in (r["miss"] or "").split(",") if x.strip()}
        out = []
        notes = {}
        for r in srows:
            ev = int(r["event"])
            if ev in ADJUDICATED:
                continue
            shw = int(r["shower"])
            dump = s117.load_dump(man[ev]["dump"])
            if dump is None:
                continue
            _, seginfo, _ = s117.digest_dump(dump, s117.load_prep(ev, prepdir))
            outs = outs_by.get((ev, shw), {})
            aff = [int(x) for x in (r["extra"] or "").split(",") if x.strip() and int(x) in outs]
            if not aff:
                continue
            notes[ev] = ((labs[ev].get("note") or "").strip(),
                         labs[ev].get("sample"), labs[ev].get("bee_url"))
            # The census names the shower by its RECONSTRUCTED root.  The
            # label key is the SCAN-TIME root, and em117_score.py's --cross-run
            # matching already recorded where that shower now lives; evt 142421
            # re-rooted 7010 -> 108104.  Use the matched id or every row of a
            # re-rooted shower reads as "(not in census)".
            live_key = int(r["matched"]) if (r.get("matched") or "").strip() else shw
            sites, tiers = live_absorb(census_log(ev, arms))
            shw_cl = seginfo.get(shw, {}).get("cluster")
            for s in aff:
                info = seginfo.get(s, {})
                mk = outs[s]
                # Does another SCORED shower of the same event want it?
                wanted_by = [b for (e2, b), miss in miss_by.items()
                             if e2 == ev and b != shw and s in miss]
                wanted_in = [b for b in wanted_by if s in ins_by.get((ev, b), set())]
                out.append(dict(
                    set=tag, event=ev, shower=shw, shw_cl=shw_cl, seg=s,
                    cluster=info.get("cluster"), length=info.get("length", 0.0),
                    pdg=info.get("pdg"), charge=info.get("charge", 0.0),
                    dist=float(mk.get("dist") or 0.0), angle=float(mk.get("angle") or 0.0),
                    label_site=mk.get("absorbed_by") or "(own root/extent)",
                    live_site=sites.get((s, live_key), "(not in census)"),
                    live_key=live_key,
                    tier=tiers.get(s), wanted_by=wanted_by, wanted_in=wanted_in))
        return out, notes
    finally:
        os.chdir(cwd)


def main():
    s117 = load_scorer()
    allrows, allnotes = {}, {}
    for tag, manf, labtag, tsv, prepdir, arms in SETS:
        allrows[tag], allnotes[tag] = collect(tag, manf, labtag, tsv, prepdir, arms, s117)

    rows = allrows["98"]
    notes = allnotes["98"]
    tot = sum(r["charge"] for r in rows)

    print("=" * 78)
    print("THE 98-SET's 22 CONDEMNED SEGMENTS  (affirmative q_extra: explicit")
    print("OUT mark on a scanned shower, reconstruction still holds it)")
    print("=" * 78)
    print("  %d segments / %.3e over %d events" % (len(rows), tot, len({r['event'] for r in rows})))
    print("  NOTE the denominator: emscan-0827 carries only 29 OUT marks in total,")
    print("  so this pool is nearly the whole OUT population of an IN-heavy scan")
    print("  (246 IN / 29 OUT = 11%), not a survey.  And with only 3 contributing")
    print("  events, ANY 'top-N holds 100%' concentration figure is vacuous.")

    print("\n--- the scanner's own words on each event (primary evidence) ---")
    for ev in sorted(notes, key=lambda e: -sum(r["charge"] for r in rows if r["event"] == e)):
        note, sample, bee = notes[ev]
        q = sum(r["charge"] for r in rows if r["event"] == ev)
        print("  %8d  %-8s q=%.3e  note: %s" % (ev, sample, q, note or "(none)"))
        print("            %s" % bee)

    print("\n--- per-segment (label store vs LIVE census on the arms on disk) ---")
    print("  %8s %8s %8s %7s %6s %6s %10s %5s %5s  %-28s %-28s %s"
          % ("event", "shower", "seg", "cluster", "len_cm", "pdg", "charge",
             "d_cm", "ang", "label absorbed_by", "live census", "tier"))
    agree = 0
    for r in sorted(rows, key=lambda r: (-sum(x["charge"] for x in rows if x["event"] == r["event"]),
                                         r["event"], -r["charge"])):
        norm = lambda s: s.replace(" (walk_add)", "").replace(" (direct)", "").replace("(own root/extent)", "own-root")
        ok = norm(r["label_site"]) == norm(r["live_site"]) or \
            (r["label_site"].startswith("(own") and r["live_site"] == "(not in census)")
        agree += ok
        print("  %8d %8d %8d %7s %6.1f %6s %10.3e %5.0f %5.0f  %-28s %-28s %s%s"
              % (r["event"], r["shower"], r["seg"], r["cluster"], r["length"], r["pdg"],
                 r["charge"], r["dist"], r["angle"], r["label_site"], r["live_site"],
                 r["tier"] if r["tier"] is not None else "-", "" if ok else "   <-- MOVED"))
    print("  attribution agreement label-store vs live run: %d / %d" % (agree, len(rows)))

    print("\n--- is the CONDEMNED segment the shower's own reconstructed root? ---")
    for tag in ("98", "141"):
        rs = allrows[tag]
        roots = sorted({(r["event"], r["live_key"]) for r in rs
                        if r["seg"] == r["live_key"]})
        print("  %s-set: %d of %d contributing events have their reco shower ROOTED on a"
              " condemned segment: %s"
              % (tag, len(roots), len({r["event"] for r in rs}),
                 ", ".join("evt %d root %d" % t for t in roots) or "(none)"))

    print("\n--- Q1  what are they: length / pdg, both sets side by side ---")
    for tag in ("98", "141"):
        rs = allrows[tag]
        lens = sorted((r["length"] for r in rs), reverse=True)
        qmed = sorted(rs, key=lambda r: -r["charge"])[0]
        print("  %s-set: n=%d  max_len=%.1f cm  median_len=%.1f cm  n(>10cm)=%d n(>15cm)=%d n(>50cm)=%d"
              % (tag, len(rs), lens[0], lens[len(lens) // 2],
                 sum(1 for l in lens if l > 10), sum(1 for l in lens if l > 15),
                 sum(1 for l in lens if l > 50)))
        pdgs = collections.Counter(r["pdg"] for r in rs)
        qpdg = collections.Counter()
        for r in rs:
            qpdg[r["pdg"]] += r["charge"]
        print("        pdg by count %s ; by charge %s"
              % (dict(pdgs), {k: "%.2e" % v for k, v in sorted(qpdg.items(), key=lambda kv: -kv[1])}))
        print("        largest single item: seg %d, %.1f cm, pdg %s, %.3e (%.1f%% of its pool)"
              % (qmed["seg"], qmed["length"], qmed["pdg"], qmed["charge"],
                 100 * qmed["charge"] / sum(r["charge"] for r in rs)))
        # pdg on a sub-cm segment is not a particle claim
        tiny = [r for r in rs if r["length"] < 2.0 and str(r["pdg"]) in ("13", "2212", "211")]
        print("        track-pdg segments SHORTER than 2 cm (pdg is not meaningful there): %d"
              % len(tiny))

    print("\n--- Q2  reachability: can any SHIPPED length-gated guard see them? ---")
    for name, seat, floor, em in SHIPPED_GUARDS:
        print("  %-42s seat %-16s floor >%.0f cm%s"
              % (name, seat, floor, "  (pdg-11 exempt)" if em else ""))
    print("  A segment is REACHABLE only if it arrives at that seat, clears the")
    print("  floor, AND is not exempt.  Length alone is not enough -- but length")
    print("  alone is already enough to rule most of these out.")
    for tag in ("98", "141"):
        rs = allrows[tag]
        hit, len_only = [], []
        for r in rs:
            site = r["live_site"] if r["live_site"] != "(not in census)" else r["label_site"]
            for name, seat, floor, em in SHIPPED_GUARDS:
                at = (seat == "*walk_add*" and "walk_add" in site) or seat in site
                if not (at and r["length"] > floor):
                    continue
                if em and str(r["pdg"]) == "11":
                    len_only.append((r["event"], r["seg"], r["length"], name))
                else:
                    hit.append((r["event"], r["seg"], r["length"], name))
        print("  %s-set: %d of %d segments REACHABLE by a shipped guard at their own seat"
              % (tag, len(hit), len(rs)))
        for h in hit:
            print("        evt %d seg %d %.1f cm -> %s" % h)
        for h in len_only:
            print("        (length-only, pdg-11 exempt) evt %d seg %d %.1f cm at %s" % h)

    print("\n--- Q3b the MIRROR: is any affirmative q_miss segment held by a")
    print("        SIBLING shower in the same event (a re-home, not a drop)? ---")
    for tag, manf, labtag, tsv, prepdir, _ in SETS:
        n, q, nt, qt, per = mirror_qmiss(tag, manf, labtag, tsv, prepdir, s117)
        print("  %s-set: %d of %d affirmative q_miss segments (%.3e of %.3e, %.1f%%)"
              % (tag, n, nt, q, qt, 100 * q / qt if qt else 0))
        for ev, (cnt, cq) in sorted(per.items(), key=lambda kv: -kv[1][1]):
            print("        evt %d: %d segs / %.3e" % (ev, cnt, cq))

    print("\n--- Q2b which guards/prunes were ALREADY ON in the arm that produced")
    print("        these 22 segments (compiled config, not the jsonnet) ---")
    for ev in sorted(notes, key=lambda e: -sum(r["charge"] for r in rows if r["event"] == e))[:1]:
        for pat in SETS[0][5]:
            for d in sorted(glob.glob(os.path.join(SX, pat))):
                cf = os.path.join(d, "pr_evt%d" % ev, ".wct-cfg-evt%d.json" % ev)
                if not os.path.exists(cf):
                    continue
                cfg = json.load(open(cf))
                for n in cfg:
                    if n.get("type") != "TaggerCheckNeutrino":
                        continue
                    da = n.get("data") or {}
                    hit = {k: v for k, v in sorted(da.items())
                           if ("prune" in k or "guard" in k) and "shower" in k}
                    print("  %s" % cf)
                    for k, v in hit.items():
                        print("    %-42s %s" % (k, v))
                break
            else:
                continue
            break

    print("\n--- Q3  is the charge EXTRA, or does a sibling shower want it? ---")
    for tag in ("98", "141"):
        rs = allrows[tag]
        dbl = [r for r in rs if r["wanted_by"]]
        dbl_in = [r for r in rs if r["wanted_in"]]
        qd = sum(r["charge"] for r in dbl)
        qdi = sum(r["charge"] for r in dbl_in)
        t = sum(r["charge"] for r in rs)
        print("  %s-set: %d of %d segments (%.3e, %.1f%% of the pool) are ALSO in another"
              % (tag, len(dbl), len(rs), qd, 100 * qd / t if t else 0))
        print("        scored shower's `miss` list in the SAME event"
              " -- of those, %d (%.3e, %.1f%%) carry an explicit IN mark there."
              % (len(dbl_in), qdi, 100 * qdi / t if t else 0))
        for r in sorted(dbl, key=lambda r: -r["charge"]):
            print("        evt %d seg %d %.3e : OUT of shower %d, missing from %s%s"
                  % (r["event"], r["seg"], r["charge"], r["shower"],
                     ",".join(str(b) for b in r["wanted_by"]),
                     " (explicit IN)" if r["wanted_in"] else ""))
    print("\n--- Q4  what the merge COSTS: the reconstructed pi0 in each event ---")
    cwd = os.getcwd()
    os.chdir(EMD)
    try:
        man = s117.load_manifest(SETS[0][1])
        labs = s117.load_labels(SETS[0][2])
        ins_by, _ = marks(labs)
        for ev in sorted(notes, key=lambda e: -sum(r["charge"] for r in rows if r["event"] == e)):
            shw = [r["shower"] for r in rows if r["event"] == ev][0]
            # the reco root, not the scan-time root (142421: 7010 -> 108104)
            live = [r["live_key"] for r in rows if r["event"] == ev][0]
            k, parts, showers = pio_report(ev, SETS[0][5], s117, man, SETS[0][4],
                                           ins_by, rows, shw)
            print("  evt %d (%s): kine_pio_flag=%s  mass=%.1f MeV  angle=%.1f deg"
                  % (ev, notes[ev][1], k.get("kine_pio_flag"),
                     float(k.get("kine_pio_mass") or 0.0), float(k.get("kine_pio_angle") or 0.0)))
            ins = ins_by.get((ev, shw), set())
            for e, sid in parts:
                tag = []
                if sid in (shw, live):
                    tag.append("<- THE OVER-CLUSTERED SHOWER")
                if sid in ins:
                    tag.append("<- rooted on a segment the scanner marked IN for shower %d" % shw)
                print("        gamma %8.1f MeV = shower %-8s %s" % (e, sid, " ".join(tag)))
            print("        reco Enu = %.1f MeV over %d showers"
                  % (float(k.get("kine_reco_Enu") or 0.0), len(showers)))
            sib = sorted({b for r in rows if r["event"] == ev for b in r["wanted_by"]})
            if sib:
                paired = {sid for _, sid in parts}
                for b in sib:
                    print("        the shower the scanner says wants that charge is %d -- %s"
                          % (b, "in the pi0 pairing" if b in paired
                             else "NOT in the pi0 pairing"))
    finally:
        os.chdir(cwd)

    print("\n--- Q5  do the SHIPPED (Part 4/5, SBND production ON) guard seats")
    print("        change anything on these three events? ---")
    for ev in sorted(notes, key=lambda e: -sum(r["charge"] for r in rows if r["event"] == e)):
        sample = notes[ev][1]
        done = False
        for offpat, onpat in FLIP_ARMS:
            off = os.path.join(SX, offpat % sample, "pr_evt%d" % ev, "calib-pr-evt%d.json" % ev)
            on = os.path.join(SX, onpat % sample, "pr_evt%d" % ev, "calib-pr-evt%d.json" % ev)
            if not (os.path.exists(off) and os.path.exists(on)):
                continue
            a, b = strip_timers(json.load(open(off))), strip_timers(json.load(open(on)))
            bad = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
            print("  evt %d (%s): %s   [%s vs %s]"
                  % (ev, sample,
                     "IDENTICAL -- the shipped seats do not touch this event"
                     if not bad else "DIFFERS in %s" % bad,
                     offpat % sample, onpat % sample))
            done = True
        if not done:
            print("  evt %d (%s): no arm pair on disk" % (ev, sample))

    return 0


if __name__ == "__main__":
    sys.exit(main())
