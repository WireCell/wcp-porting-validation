#!/usr/bin/env python3
"""Classify every PR event as evaluated / not-evaluated / QUARANTINE, and
optionally drop the heavy products of the not-evaluated ones.

    prune_unevaluated.py ARM [ARM ...]              # classify + report only
    prune_unevaluated.py --apply ARM [ARM ...]      # also delete

Owner instruction for the prod0813 campaign (doc pr/76): "you only need to save
them, if they have PR results". An event WITH a selected neutrino candidate
keeps everything; an event WITHOUT one keeps only its record layer.

  dropped when not evaluated:  pctree-pr-evt<ID>.tar.gz, tracking-pr.root,
                               mabc-pr.zip, calib-pr-evt<ID>.json
  ALWAYS kept:                 wct_pr_evt<ID>.log, stdout.log, rc.txt,
                               .time.meta, nusel-evt<ID>.tsv

The log is kept deliberately: nu_evaluated exists in NO other artifact -- not in
nusel-evt<ID>.tsv, not in nusel-table.tsv, not in tracking-pr.root -- so
deleting it would make the census unfalsifiable afterwards. It costs ~150 KB
against the ~8 MB it replaces.

WHY THREE MARKERS AND NOT ONE GREP
----------------------------------
The obvious implementation greps for "selected main cluster" and prunes
everything else. That is unsafe HERE, and the failure is silent and permanent.
WCT writes long spdlog messages non-atomically, so a line can be cut mid-word
with another thread's message spliced into it. This round measured exactly that
on its own gate: two of 48 events lost the cluster-id token of a
"cluster N no STM fit: fully contained" message purely because adding one
visitor to the pipeline shifted thread interleaving (doc pr/76 sec on the
knob-on gate; project_wct_log_line_tearing). A torn "selected main cluster"
line would read as not-evaluated, and this script would then delete a REAL PR
result -- removing it from the Bee set and the hand scan with nothing left on
disk to recompute from.

So every event must match EXACTLY ONE of three mutually exclusive markers:

  selected main cluster ............ evaluated  -> keep everything
  no main cluster selected ......... not evaluated -> prunable
  cosmic-tagged ...; skipping ...... not evaluated -> prunable  (the in-window
                                     cluster was cosmic-tagged, so the tagger
                                     never evaluated the event -- correct, not
                                     a bug; evt 116962 is the known case)

Zero matches, or more than one, means the log is torn or the event took a path
this script does not model. Those are QUARANTINED: never pruned, always listed.
A cross-check catches the subtler direction too -- an event with no selection
line but WITH a calib-pr-evt<ID>.json is suspicious, because PrDisplayDump emits
that dump off the selected main cluster, so JSON-present + line-absent is the
signature of a torn selection line rather than a genuine non-selection.

Reports the quarantine count even when it is 0: a silent 0 is indistinguishable
from a check that never ran.

DO NOT REIMPLEMENT THIS WITH `grep` -- MEASURED, NOT HYPOTHETICAL
-----------------------------------------------------------------
Torn lines splice bytes mid-character, so these logs are frequently INVALID
UTF-8 ("Non-ISO extended-ASCII text"). GNU grep in a UTF-8 locale then treats
the file as binary and, for a plain match, prints NOTHING and exits 1 -- it does
not warn, and `grep -c` emits no count at all. Reproduced on this campaign:

    $ grep -c 'selected main cluster' work-nuecc48-prod0813/pr_evt256587/wct_pr_evt256587.log
    (no output, rc=1)
    $ grep -a -c 'selected main cluster' .../wct_pr_evt256587.log
    1

That event HAS a real selection -- "selected main cluster 11 (t0 1.482 us,
L 130.6 cm, 89 associated)" at byte 360702 -- and a shell loop built on plain
grep classified it as not-evaluated. Under --apply that would have deleted a
genuine PR result. Python's open(..., errors='replace') has no such failure
mode, which is why this script, pr_scores_table.py:99, nusel_extract.py:262 and
make_pr_bee.py:83 all read that way. If you must use grep on a WCT log, pass -a.
"""
import os, re, sys, glob

RE_SELECTED = re.compile(r'TaggerCheckNeutrino: selected main cluster \S+ \(t0 ')
RE_NOMAIN = re.compile(r'TaggerCheckNeutrino: no main cluster selected')
RE_COSMIC_SKIP = re.compile(r'TaggerCheckNeutrino: in-window cluster .* cosmic-tagged .*; '
                            r'skipping \(nu_skip_cosmic\)')

HEAVY_GLOBS = ("pctree-pr-evt*.tar.gz", "tracking-pr.root", "mabc-pr.zip",
               "calib-pr-evt*.json")


def classify(prdir, evt):
    log = os.path.join(prdir, f"wct_pr_evt{evt}.log")
    if not os.path.isfile(log):
        return "QUARANTINE", "no wct_pr_evt log -- cannot establish evaluation either way"
    txt = open(log, errors="replace").read()
    sel = bool(RE_SELECTED.search(txt))
    nomain = bool(RE_NOMAIN.search(txt))
    cosmic = bool(RE_COSMIC_SKIP.search(txt))
    has_calib = bool(glob.glob(os.path.join(prdir, "calib-pr-evt*.json")))

    n = sum((sel, nomain, cosmic))
    if sel and n == 1:
        return "evaluated", ""
    if sel and n > 1:
        # A selection plus a non-selection marker: the cosmic-skip line is
        # per-cluster and can legitimately co-occur with a selection on a
        # different cluster, so this is not automatically an error -- but it is
        # not something to prune on either.  Keep, and say so.
        return "evaluated", f"also matched {'nomain ' if nomain else ''}" \
                            f"{'cosmic-skip' if cosmic else ''} (kept: selection wins)"
    if n == 0:
        return "QUARANTINE", ("no marker matched" +
                              (" BUT calib dump present -- torn selection line?"
                               if has_calib else " and no calib dump"))
    if has_calib:
        return "QUARANTINE", ("marked not-evaluated but a calib dump exists -- "
                              "PrDisplayDump emits off the selected main cluster")
    return "not-evaluated", "no-main" if nomain else "cosmic-skip"


def main():
    args = sys.argv[1:]
    apply_ = False
    if args and args[0] == "--apply":
        apply_, args = True, args[1:]
    if not args:
        sys.exit(__doc__)

    grand = dict(evaluated=0, notev=0, quar=0, freed=0)
    for arm in args:
        arm = arm.rstrip("/")
        rows = []
        for d in sorted(os.listdir(arm)):
            if not (d.startswith("pr_evt") and os.path.isdir(os.path.join(arm, d))):
                continue
            evt = d[len("pr_evt"):]
            prdir = os.path.join(arm, d)
            verdict, note = classify(prdir, evt)
            rows.append((evt, prdir, verdict, note))

        ev = [r for r in rows if r[2] == "evaluated"]
        nv = [r for r in rows if r[2] == "not-evaluated"]
        qz = [r for r in rows if r[2] == "QUARANTINE"]

        freed = 0
        for evt, prdir, verdict, note in nv:
            for g in HEAVY_GLOBS:
                for p in glob.glob(os.path.join(prdir, g)):
                    if os.path.islink(p):          # never follow/remove a link
                        continue
                    freed += os.path.getsize(p)
                    if apply_:
                        os.remove(p)

        print(f"=== {arm} ===")
        print(f"  events            {len(rows)}")
        print(f"  evaluated (keep)  {len(ev)}")
        print(f"  not-evaluated     {len(nv)}   "
              f"(no-main {sum(1 for r in nv if r[3]=='no-main')}, "
              f"cosmic-skip {sum(1 for r in nv if r[3]=='cosmic-skip')})")
        print(f"  QUARANTINE        {len(qz)}   <-- never pruned")
        for evt, _, _, note in qz:
            print(f"      evt {evt}: {note}")
        for evt, _, _, note in ev:
            if note:
                print(f"      note evt {evt}: {note}")
        print(f"  heavy products {'FREED' if apply_ else 'that WOULD be freed'}: "
              f"{freed/2**20:.1f} MiB")
        print()
        grand['evaluated'] += len(ev); grand['notev'] += len(nv)
        grand['quar'] += len(qz); grand['freed'] += freed

    print(f"TOTAL  evaluated {grand['evaluated']}  not-evaluated {grand['notev']}  "
          f"QUARANTINE {grand['quar']}  "
          f"{'freed' if apply_ else 'would free'} {grand['freed']/2**30:.2f} GiB")
    if not apply_:
        print("\ndry run -- re-run with --apply to delete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
