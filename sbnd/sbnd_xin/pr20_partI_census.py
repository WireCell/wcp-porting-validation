#!/usr/bin/env python3
"""doc pr/20 Part I S10: the demoted-main funnel census and per-drop impact table.

Reads two PR-chain arms built by run_pr_chain_batch.sh on the SAME Q/L root:

  off_root   every Part I knob OFF                  -- the physics baseline
  on_root    P2+P3+P4 ON, cosmic_companion_min_length = 0 (no floor)
             -- the envelope arm: because P4's test is `L >= floor`, every
             companion that any floor could drop is dropped here, so this one
             run enumerates the entire actionable population.

WHY A FLOOR CANNOT BE READ OFF A BARE LENGTH HISTOGRAM
------------------------------------------------------
`cosmic_companion_min_length` is a safety valve, not a tuning knob: it keeps a
SHORT cosmic-tagged companion in the neutrino even when the tagger convicts it,
so a mis-tagged neutrino daughter is never silently lost.  Choosing it therefore
needs the per-drop IMPACT (did the event's label / energy / score move, and was
the move attributable?), not just how long the dropped things were.  This script
produces both, keyed the same way.

THE FUNNEL (the numbers that matter, in order)
----------------------------------------------
  flagged    parts the outer un-merge marked Flags::demoted_main         (P2)
  admitted   of those, the ones TGM/STM/FC actually evaluated            (P3)
             -- the scope filter and the beam-window gate cut here, and they
             cut hard: on evt 59003 the beam gate alone takes 8 -> 1.
  convicted  of those, the ones a cosmic tagger tagged TGM or STM
  dropped    of those, the ones that were a companion of the SELECTED main
             and so were actually removed from the neutrino                (P4)

`dropped` <= `convicted` by construction: P4 only ever sees companions of the
one selected main.  That is not a coverage gap in this census -- it is exactly
the population P4 can ever act on.

LOG LINES PARSED (all emitted at the runner's default -L debug)
---------------------------------------------------------------
  <ClusteringUnmergeBundle:pr> cluster N: demoted_main (K blob(s) ...)   flagged
  visit: TaggerCheckTGM: evaluate_demoted_mains: N demoted main(s) added admitted
  visit: TaggerCheck{TGM,STM,FC}: cluster N -> {TGM,STM,FC}=...          verdicts
  TaggerCheckNeutrino: selected main cluster N (t0 .., L .., M associated)
  TaggerCheckNeutrino: companion cluster N (L .., TGM=.. STM=..) dropped ...

IDENT SPACES DO NOT JOIN.  The un-merge's `cluster 1801: demoted_main` idents
are pre-rebuild composites; the taggers' `cluster 26` idents are the PR job's
own.  So `flagged` is a COUNT only, and the identity of the admitted demoted
mains is recovered the one way that is sound: the set of cluster idents the
taggers evaluated in the ON arm minus the set they evaluated in the OFF arm.
Both arms run the same binary on the same input, so that difference is exactly
what `evaluate_demoted_mains` added.  It is cross-checked against the P3 INFO
count and any mismatch is reported per event rather than smoothed over.

Repro:
  ./pr20_partI_census.py work-pi5cens-proff work-pi5cens-pron0 \\
      --off-scores /home/xqian/tmp/pr20exec/s10_scores_off.tsv \\
      --on-scores  /home/xqian/tmp/pr20exec/s10_scores_on0.tsv \\
      --out /home/xqian/tmp/pr20exec/s10
"""
import argparse
import concurrent.futures as cf
import os
import re
import sys

RE_FLAGGED = re.compile(r"cluster (\d+): demoted_main \((\d+) blob")
RE_ADMITTED = re.compile(
    r"TaggerCheck(TGM|STM|FC): evaluate_demoted_mains: (\d+) demoted main\(s\) added")
# The arrow is U+2192, written by SPDLOG_LOGGER_INFO in the three taggers.
#
# These deliberately do NOT anchor on the "TaggerCheckXXX: " prefix.  WCT log
# lines tear mid-word when threads interleave, and it happens to these very
# lines -- e.g. evt 62583 carries
#     MABC timing: done took 11cluster 30 → TGM=false
# where the verdict survived but the prefix did not, and evt 65267 carries
#     TaggerCheckTGM: cluster 23 → TGMing took 0.017927 ms
# where the prefix survived but the verdict did not.  Matching on the
# "cluster N → KEY=" tail recovers the first kind; the second kind is recovered
# by cross-reading the STM line (below) and, failing that, is reported as an
# unknown verdict rather than silently counted as a non-conviction.
RE_TGM = re.compile(r"cluster (\d+) \S+ TGM=(\w+)$")
RE_STM = re.compile(r"cluster (\d+) \S+ STM=(\d+) TGM=(\d+)")
RE_FC = re.compile(r"cluster (\d+) \S+ FC=(\w+)")
RE_SEL = re.compile(
    r"selected main cluster (\d+) \(t0 (\S+) us, L (\S+) cm, (\d+) associated\)")
RE_DROP = re.compile(
    r"companion cluster (\d+) \(L (\S+) cm, TGM=(\d+) STM=(\d+)\) dropped")


def parse_log(path):
    """Pull the Part I funnel evidence out of one event's PR log."""
    out = {
        "flagged": 0, "flagged_blobs": [], "admitted": {},
        "tgm": {}, "stm": {}, "stm_tgm": {}, "fc": {},
        "sel": None, "drops": [],
    }
    if not os.path.exists(path):
        return None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = RE_FLAGGED.search(line)
            if m:
                out["flagged"] += 1
                out["flagged_blobs"].append(int(m.group(2)))
                continue
            m = RE_ADMITTED.search(line)
            if m:
                out["admitted"][m.group(1)] = int(m.group(2))
                continue
            m = RE_TGM.search(line)
            if m:
                out["tgm"][int(m.group(1))] = m.group(2) == "true"
                continue
            m = RE_STM.search(line)
            if m:
                # TaggerCheckSTM logs both FLAGS after TGM and STM have run
                # (TaggerCheckSTM.cxx:498-501), so this one line is a complete,
                # redundant source for both verdicts -- the tear-proofing.
                out["stm"][int(m.group(1))] = m.group(2) == "1"
                out["stm_tgm"][int(m.group(1))] = m.group(3) == "1"
                continue
            m = RE_FC.search(line)
            if m:
                out["fc"][int(m.group(1))] = m.group(2) == "true"
                continue
            m = RE_SEL.search(line)
            if m:
                out["sel"] = (int(m.group(1)), float(m.group(2)),
                              float(m.group(3)), int(m.group(4)))
                continue
            m = RE_DROP.search(line)
            if m:
                out["drops"].append((int(m.group(1)), float(m.group(2)),
                                     m.group(3) == "1", m.group(4) == "1"))
    return out


def _parse_pair(task):
    """One event's two logs.  Module-level so ProcessPoolExecutor can pickle it.

    A 1000-event pair is 2000 log files and tens of GB of debug text; parsing
    them serially dominates this script's wall time, and the events are wholly
    independent, so the pool is a straight win.  Ordering is not relied on --
    results are keyed by event and sorted afterwards.
    """
    off_root, on_root, evt = task
    o = parse_log(os.path.join(off_root, "pr_evt%d" % evt, "wct_pr_evt%d.log" % evt))
    n = parse_log(os.path.join(on_root, "pr_evt%d" % evt, "wct_pr_evt%d.log" % evt))
    return evt, o, n


def load_scores(path):
    """pr_scores_table.py TSV -> {event: {col: value}}."""
    if not path or not os.path.exists(path):
        return {}
    rows = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) != len(hdr):
                continue
            d = dict(zip(hdr, f))
            try:
                rows[int(d["event"])] = d
            except (KeyError, ValueError):
                pass
    return rows


def fnum(d, key):
    try:
        return float(d[key])
    except (KeyError, TypeError, ValueError):
        return None


def delta(a, b):
    if a is None or b is None:
        return ""
    return "%.4f" % (b - a)


def events_of(root):
    evts = []
    for name in os.listdir(root):
        if name.startswith("pr_evt"):
            try:
                evts.append(int(name[6:]))
            except ValueError:
                pass
    return sorted(evts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("off_root")
    ap.add_argument("on_root")
    ap.add_argument("--off-scores", default=None)
    ap.add_argument("--on-scores", default=None)
    ap.add_argument("--out", default=None, help="prefix for _funnel.tsv / _drops.tsv")
    ap.add_argument("-j", "--jobs", type=int, default=min(24, os.cpu_count() or 1),
                    help="log-parsing processes (default: min(24, ncpu); 1 = serial)")
    args = ap.parse_args()

    off_scores = load_scores(args.off_scores)
    on_scores = load_scores(args.on_scores)

    evts = sorted(set(events_of(args.off_root)) & set(events_of(args.on_root)))
    only_off = sorted(set(events_of(args.off_root)) - set(evts))
    only_on = sorted(set(events_of(args.on_root)) - set(evts))
    if only_off or only_on:
        print("WARNING: %d event(s) only in OFF, %d only in ON -- census uses the "
              "%d in both" % (len(only_off), len(only_on), len(evts)), file=sys.stderr)

    tasks = [(args.off_root, args.on_root, e) for e in evts]
    if args.jobs == 1:
        parsed = [_parse_pair(t) for t in tasks]
    else:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            parsed = list(pool.map(_parse_pair, tasks, chunksize=8))
    parsed.sort(key=lambda r: r[0])

    funnel = []
    drops = []
    mismatches = []
    for evt, o, n in parsed:
        if o is None or n is None:
            continue

        # The demoted mains, identified the only sound way: what the ON arm's
        # taggers evaluated that the OFF arm's did not.  Taken as the UNION over
        # the three taggers -- all three evaluate the same cluster set, so a
        # torn line in one of them cannot lose a cluster from the census.
        def seen(d):
            return set(d["tgm"]) | set(d["stm"]) | set(d["fc"])
        added = sorted(seen(n) - seen(o))
        admitted_info = n["admitted"].get("TGM")
        if admitted_info is not None and admitted_info != len(added):
            mismatches.append((evt, admitted_info, len(added)))

        # TGM verdict: prefer the STM line's flag readback (complete by
        # construction), fall back to TGM's own line, and treat "neither
        # survived the log" as UNKNOWN rather than as a non-conviction.
        def tgm_of(c):
            if c in n["stm_tgm"]:
                return n["stm_tgm"][c]
            return n["tgm"].get(c)
        convicted, unknown = [], []
        for c in added:
            t, s = tgm_of(c), n["stm"].get(c)
            if t or s:
                convicted.append(c)
            elif t is None and s is None:
                unknown.append(c)

        od, nd = off_scores.get(evt, {}), on_scores.get(evt, {})
        for cid, ln, tgm, stm in n["drops"]:
            drops.append({
                "event": evt, "cluster": cid, "L_cm": ln,
                "TGM": int(tgm), "STM": int(stm),
                "was_demoted": int(cid in added),
                "label_off": od.get("event_label", ""),
                "label_on": nd.get("event_label", ""),
                "label_moved": int(bool(od.get("event_label")) and
                                   od.get("event_label") != nd.get("event_label")),
                "dEnu_MeV": delta(fnum(od, "kine_reco_Enu_MeV"),
                                  fnum(nd, "kine_reco_Enu_MeV")),
                "dnumu": delta(fnum(od, "numu_score"), fnum(nd, "numu_score")),
                "dnue": delta(fnum(od, "nue_score"), fnum(nd, "nue_score")),
                "n_drops_this_evt": len(n["drops"]),
            })

        funnel.append({
            "event": evt,
            "flagged": n["flagged"],
            "admitted": len(added),
            "admitted_info": admitted_info if admitted_info is not None else "",
            "convicted": len(convicted),
            "verdict_unknown": len(unknown),
            "dropped": len(n["drops"]),
            "sel_main_off": o["sel"][0] if o["sel"] else "",
            "sel_main_on": n["sel"][0] if n["sel"] else "",
            "sel_L_off": "%.1f" % o["sel"][2] if o["sel"] else "",
            "sel_L_on": "%.1f" % n["sel"][2] if n["sel"] else "",
            "n_assoc_off": o["sel"][3] if o["sel"] else "",
            "n_assoc_on": n["sel"][3] if n["sel"] else "",
            "label_off": od.get("event_label", ""),
            "label_on": nd.get("event_label", ""),
        })

    if args.out:
        write_tsv(args.out + "_funnel.tsv", funnel)
        write_tsv(args.out + "_drops.tsv", drops)
        print("wrote %s_funnel.tsv (%d rows) and %s_drops.tsv (%d rows)"
              % (args.out, len(funnel), args.out, len(drops)))

    report(funnel, drops, mismatches)


def write_tsv(path, rows):
    if not rows:
        with open(path, "w") as fh:
            fh.write("")
        return
    cols = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")


def quantiles(xs, qs=(0, 5, 25, 50, 75, 95, 100)):
    if not xs:
        return {}
    s = sorted(xs)
    out = {}
    for q in qs:
        i = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
        out[q] = s[i]
    return out


def report(funnel, drops, mismatches):
    n_evt = len(funnel)
    print("\n=== doc pr/20 Part I S10 -- demoted-main funnel, %d events ===" % n_evt)
    tot = {k: sum(int(r[k]) for r in funnel)
           for k in ("flagged", "admitted", "convicted", "verdict_unknown", "dropped")}
    n_evt_flagged = sum(1 for r in funnel if r["flagged"])
    n_evt_admitted = sum(1 for r in funnel if r["admitted"])
    n_evt_dropped = sum(1 for r in funnel if r["dropped"])
    print("  flagged   %6d  (on %d/%d events)" % (tot["flagged"], n_evt_flagged, n_evt))
    print("  admitted  %6d  (on %d/%d events)  <- scope + beam-window gate"
          % (tot["admitted"], n_evt_admitted, n_evt))
    print("  convicted %6d   TGM or STM" % tot["convicted"])
    if tot["verdict_unknown"]:
        print("  (verdict unreadable on %d admitted cluster(s) -- torn log line, "
              "counted as NEITHER convicted nor cleared)" % tot["verdict_unknown"])
    print("  dropped   %6d  (on %d/%d events)  <- companions of the selected main"
          % (tot["dropped"], n_evt_dropped, n_evt))

    if mismatches:
        print("\n  WARNING: P3 INFO count != set-difference count on %d event(s):"
              % len(mismatches))
        for evt, info, got in mismatches[:10]:
            print("    evt %d: INFO says %d, set-difference says %d" % (evt, info, got))

    if not drops:
        print("\n  no drops -- nothing for the floor to arbitrate.")
        return

    lens = [d["L_cm"] for d in drops]
    q = quantiles(lens)
    print("\n=== dropped-companion lengths (cm), n=%d ===" % len(lens))
    print("  min %.2f | p5 %.2f | p25 %.2f | median %.2f | p75 %.2f | p95 %.2f | max %.2f"
          % (q[0], q[5], q[25], q[50], q[75], q[95], q[100]))
    print("\n  cumulative count at or above a candidate floor:")
    for f in (0, 1, 2, 3, 5, 10, 15, 20, 30, 50):
        keep = sum(1 for x in lens if x >= f)
        evts_f = len({d["event"] for d in drops if d["L_cm"] >= f})
        print("    floor %5.1f cm -> %4d drop(s) on %3d event(s)" % (f, keep, evts_f))

    moved = [d for d in drops if d["label_moved"]]
    print("\n=== impact ===")
    print("  events whose beam label moved: %d" % len({d["event"] for d in moved}))
    for d in moved[:20]:
        print("    evt %d cluster %d L %.1f cm: %s -> %s"
              % (d["event"], d["cluster"], d["L_cm"], d["label_off"], d["label_on"]))
    single = [d for d in drops if d["n_drops_this_evt"] == 1 and d["dEnu_MeV"]]
    if single:
        print("\n  single-drop events (the energy delta is attributable to ONE"
              " companion), n=%d:" % len(single))
        single.sort(key=lambda d: d["L_cm"])
        print("    %8s %8s %10s %12s %10s" % ("event", "L_cm", "dEnu_MeV", "dnumu", "dnue"))
        for d in single[:40]:
            print("    %8d %8.2f %10s %12s %10s"
                  % (d["event"], d["L_cm"], d["dEnu_MeV"], d["dnumu"], d["dnue"]))


if __name__ == "__main__":
    main()
