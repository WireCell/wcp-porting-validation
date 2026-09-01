"""doc 91 -- the COUNT-driven retire round, 2026-09-01b.

Fork of plan_20260901.py by verbatim `cp` (cmp-verified before editing), then
edited block by block.  ASSERT 15 and 16 are new.

WHY THIS ROUND EXISTS, in the owner's words: "The sbnd_xin directory still have
a lot of work* directory, do we need all of them?  Can we retire some and go
minimum?  I understand that they do not take much disk space, but it is just
difficult to look at them."  So the metric is DIR COUNT, not bytes: 101 -> 52.
Bytes are a side effect (~7.0 GiB) and are not the reason for any decision here.

Then, on being shown the classification: "Peer is done, we can remove or retire
them.  We want to keep the latest production though."  That released doc 90's
nine arms, which the 09-01 round had protected by prefix while that round was
live, and pinned work-*-prod0901b as untouchable.

THREE THINGS THIS ROUND MEASURED RATHER THAN INHERITED
------------------------------------------------------
1. The citation census the 09-01 round used was CIRCULAR.  It counted a name as
   "cited" when a previous retire PLANNER named it -- i.e. an arm was protected
   because it had been protected.  scripts/retire/ is excluded here.  It was
   also NAME-EXACT, so it reported docs=0 for all 28 doc-87 arms, which cite
   themselves as templates ("work-87knob-def-").  A census that returns 0 for a
   cited arm is indistinguishable from one that works; both faults are fixed.

2. Two "hardcoded default arm" protections were MIS-READS.  em_display/
   prep_pr121.py:15 and prep_pr117.py:17,28 are docstring USAGE lines; neither
   file has an argparse default naming an arm.  Checked with
   `grep -n "default=" em_display/prep_pr11{7,21}.py` -> no arm name.

3. work-*-prod0901b contains all 30 pr127_sentinels.py events itself.  The
   suite finds an event by scanning arm roots for pr_evt<N>/, and 30/30
   resolve, so it runs against production.  work-sent130-{mcp1k,mcp2k} existed
   only because production used to be an event subset.  ASSERT 15 re-derives
   this on every run rather than trusting today's reading.

THE OPEN FINDING THAT SHAPED THE KEEP SET
-----------------------------------------
Run against production the sentinel suite is NOT green: 27 PASS, 6 FAIL.
Three of the six -- 47212 (pr/120 backward-stem guard), 173819 (pr/125
pass3_cone guard) and 406125 (pr/124 gap-band prune) -- still PASS in the arms
where the fix was validated.  For 406125 the knob is still ON in
wct-pr-perevt.jsonnet and the C++ log line still exists; it simply no longer
fires.  That is the doc pr/127 failure mode (a shipped fix dying silently)
recurring, and it is NOT diagnosed here -- CLAUDE.md sec 5.7: report, do not
tune.  Its consequence for THIS round is concrete: the arms that can adjudicate
it must not be deleted while it is open, which is what ASSERT 16 enforces.

work-pr134-f086-* is the minimal sufficient baseline (four failures in -mcp2k,
one in -mcp1k, one in -nuecc48) AND the only one at the same EM scale as
production -- cfg kine_shower_fudge_factor = 0.86, which doc pr/135 flipped and
which f086 is.  That matters because pr127_sentinels.py's own header warns that
an energy-scale change moves every absolute-MeV threshold: a baseline at 0.80
could not separate "the fix died" from "the scale moved".

WHAT IS DIFFERENT FROM 09-01, STRUCTURALLY
------------------------------------------
* No production arm is released and no successor gate is run, so ASSERT 14
  degenerates to "gate not required" and passes vacuously.  ASSERT 15 and 16
  exist so this round is not guarded only by asserts that cannot fire -- the
  lesson from the 09-01 sweep interlock that was "passing" on a glob matching
  zero arms.  ASSERT 16 re-derives the failing set by RUNNING the suite; a
  hardcoded list would go stale the moment a fix is repaired and would then
  protect arms for a reason that no longer holds.
* PR_PROVENANCE and CAMPAIGN_IMG are empty: the r1qlmc/r2mc sim chain is
  released IN FULL (10 dirs), so closure holds in both directions instead of
  being preserved by exempting a link.
* The tier file name is tierA_20260901b.txt.  The 09-01 header warns that this
  name is INTERPOLATED and a missed rename makes the dry run target the
  previous round's already-deleted list and report dirs=0.  Renamed and checked.

TWO DIRS IN THE REMOVAL SET WERE NOT ARMS
-----------------------------------------
work-nuecc48-prsmoke2 held three git-tracked runner scripts and no event data;
its one consumer (scripts/repro_track_pid_evt172230.py:4) already pointed at a
subdir that no longer exists.  work-stmcamp-d66new held nothing but
nusel_labels/d66flip/ -- the tree's ONLY nusel label store -- and its comparison
partner work-stmcamp-d66old was retired rounds ago.  Both were emptied by
`git mv` BEFORE the round (to scripts/legacy/ and ./nusel_labels/d66flip/
respectively, with d66_scan_score.py repointed to --root .), so ASSERT 3 sees no
tracked file in the removal set and no record is deleted.
"""

import os, re, json, subprocess, collections, sys, filecmp, csv, glob, time

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.path.join(ROOT, "scripts", "retire")
STATE = os.environ.get("RETIRE_STATE", os.path.join(SCR, "state-20260901b"))
os.makedirs(STATE, exist_ok=True)
os.chdir(ROOT)

if os.path.exists(os.path.join(STATE, "removed.tsv")) and not os.environ.get("RETIRE_REPLAN"):
    sys.stderr.write(
        f"REFUSING: {STATE}/removed.tsv exists -- this round has already run (M13).\n"
        f"Fork with a new date/state for a new round; RETIRE_REPLAN=1 to override.\n")
    sys.exit(3)

dirs = sorted(d for d in os.listdir('.')
              if (d.startswith('work') or d.startswith('void'))
              and os.path.isdir(d) and not os.path.islink(d))

# ---------------------------------------------------------------- KEEP
# Prefixes whose every arm is kept.  Used ONLY for the open peer round: the
# peer was still creating arms while this ran, so an enumerated list would be
# stale by construction.  Everything else in KEEP is named individually.
KEEP_PREFIX = {
    # doc 91.  TWO families, and both are held by a measurement made in THIS
    # round rather than inherited from the last one.
    #
    # work-pr134-f086-* is the minimal sufficient baseline for every one of the
    # six sentinels that FAIL against production (137238 173819 393505 47212
    # 292643 406125): -mcp2k holds four, -mcp1k the fifth, -nuecc48 the sixth.
    # It is also the ONLY surviving baseline at the SAME EM scale as production
    # -- cfg kine_shower_fudge_factor = 0.86, flipped by doc pr/135, and f086
    # is that point -- so a PASS-here / FAIL-there split cannot be the
    # energy-scale drift pr127_sentinels.py's own header warns about.  Deleting
    # this family would leave the open regression unadjudicable.
    'work-pr134-f086': 'doc 91 sec 3 -- scale-matched baseline for all 6 failing sentinels',
    #
    # The sentinel NEGATIVE controls.  work-sent130-{mcp1k,mcp2k} (the POSITIVE
    # arms) ARE released this round: work-*-prod0901b was measured to contain
    # all 30 sentinel events itself, so the suite no longer needs them to run.
    # The neg arms stay because they are the only on-disk proof the registry
    # CAN fail -- the one property a green suite cannot demonstrate about
    # itself, and the lesson this tree learned the hard way when a guard
    # "passed" on an empty match set.
    'work-sent130neg': 'sentinel registry negative controls -- the proof the suite can fail',
}

KEEP_WHY = {
    # --- the latest production, 4 -----------------------------------------
    # Owner, 2026-09-01: "We want to keep the latest production though."
    # ref/prod-2026-09-01b, save_in_scope ON, full output.  UNCHANGED by this
    # round: doc 91 releases no production arm and runs no successor gate.
    'work-nuecc48-prod0901b': 'THE latest production, pinned operating point, 48 evts',
    'work-ncpi0-prod0901b':   'THE latest production, 19 evts',
    'work-mcp1k-prod0901b':   'THE latest production, 1000 evts',
    'work-mcp2k-prod0901b':   'THE latest production, 2000 evts',
    # --- campaign INPUT: the Q/L roots every re-run reads, 4 ---------------
    # The PR job's ONLY input (TensorFileSource).  Releasing these would mean
    # redoing imaging + Q/L from reco1 before any PR chain could run again.
    'work-nuecc48-grp0825':  'doc 81 stage A (imaging + Q/L), 48 evts -- campaign input',
    'work-ncpi0-grp0825':    'doc 81 stage A, 19 evts -- campaign input',
    'work-mcp1k-grp0825':    'doc 81 stage A, 1000 evts -- campaign input',
    'work-mcp2k-grp0825':    'doc 81 stage A, 2000 evts -- campaign input',
    # --- the vertex-label epoch: 878 label files point into these, 4 -------
    'work-vtx105-base-nuecc48': 'calib-pr dumps behind the vtxscan-vtx105-* label epoch, 47 evts',
    'work-vtx105-base-ncpi0':   'calib-pr dumps behind the current label epoch, 19 evts',
    'work-vtx105-base-mcp1k':   'calib-pr dumps behind the current label epoch, 407 evts',
    'work-vtx105-base-mcp2k':   'calib-pr dumps behind the current label epoch, 581 evts',
    # --- EM label-store backing: 251 em_labels source paths, 6 -------------
    # Measured this round by resolving every `source` string in em_labels/,
    # not by citation count: these six are the arms those JSONs name.
    'work-em114-mcp1k':   'probe-root fallback for the original 94, em_display/prep_em_scan.py:275',
    'work-em114-mcp2k':   'probe-root fallback, same chain',
    'work-em114-ncpi0':   'probe-root fallback, same chain',
    'work-em114-nuecc48': 'probe-root fallback, same chain',
    'work-em114c-mcp1k':  '--parse-probes source for em114c-manifest.tsv (141-set); 54 label refs',
    'work-em114c-mcp2k':  '--parse-probes source for the 141-set display manifest; 93 label refs',
    # --- doc pr/130's census + emscan-0829 label source, 6 -----------------
    'work-pr130r1-probe98-mcp1k':   'dump source for both pr130q manifests + pr130_launder_scan.py census',
    'work-pr130r1-probe98-mcp2k':   'dump source for both pr130q manifests',
    'work-pr130r1-probe98-ncpi0':   'dump source for both pr130q manifests',
    'work-pr130r1-probe98-nuecc48': 'dump source for both pr130q manifests',
    'work-pr130r1-probe141-mcp1k':  '141-set half of the pr130q dump source',
    'work-pr130r1-probe141-mcp2k':  '141-set half of the pr130q dump source',
    # --- the sentinel registry's worked FAILING case, 6 --------------------
    'work-pr125r1-flipK598-mcp1k':   'scripts/pr127_sentinels.py:31 -- the worked FAILING sentinel case (137238)',
    'work-pr125r1-flipK598-mcp2k':   'the negative control; a registry with no failing case cannot be trusted',
    'work-pr125r1-flipK598-ncpi0':   'the negative control',
    'work-pr125r1-flipK598-nuecc48': 'the negative control',
    'work-pr125r1-flipK5141-mcp1k':  '141-set half of the negative control',
    'work-pr125r1-flipK5141-mcp2k':  '141-set half of the negative control',
    # --- doc 87: the half the successor gate did NOT re-establish, 9 -------
    # SPLIT, not kept or released as a block.  doc 89's successor gate compared
    # prod0901 (which PREDATES the doc-87 knobs, toolkit ddce7430) against
    # prod0901b (knobs present, at their defaults) on all 3067 events and found
    # every shared product identical.  That IS doc 87 sec 6.1's "knobs at their
    # defaults change nothing" and sec 1.4's "the merge changes nothing", at
    # 3067 events instead of 482 -- so the twelve gate arms go, and the gate
    # TSV in state-20260901/successor-gate.tsv keeps the claim re-checkable.
    # These nine are the part that gate could NOT speak to:
    #   knob-min / knob-sup : sec 6.2's minimal-output table.  prod0901b ran
    #     FULL output, so nothing in doc 89 exercises output suppression.
    #     These are still the only minimal-mode arms in the tree.
    #   grp-*  : sec 6.4 group-mode.  prod0901b is per-event mode; a different
    #     output layout, not a different setting of the same one.
    #   flip-* : sec 4.6 flip-equivalence at the ncpi0/nuecc48 scale.
    'work-87knob-min-ncpi0':   'doc 87 sec 6.2 -- minimal-output arm; NOT re-established by doc 89',
    'work-87knob-min-nuecc48': 'doc 87 sec 6.2 -- minimal-output arm',
    'work-87knob-sup-ncpi0':   'doc 87 sec 6.2 -- suppressed-output arm',
    'work-87knob-sup-nuecc48': 'doc 87 sec 6.2 -- suppressed-output arm',
    'work-87grp-def-ncpi0':    'doc 87 sec 6.4 -- group-mode default; prod0901b is per-event mode',
    'work-87grp-min-ncpi0':    'doc 87 sec 6.4 -- group-mode minimal',
    'work-87grp-nopct-ncpi0':  'doc 87 sec 6.4 -- group-mode, pctree suppressed',
    'work-87flip-ncpi0':       'doc 87 sec 4.6 -- flip-equivalence (shipped jsonnet == env-forced)',
    'work-87flip-nuecc48':     'doc 87 sec 4.6 -- flip-equivalence',
    # --- non-reproducible / record dirs, 3 --------------------------------
    'work-tfix388-r9':       'doc pr/28 sec.15.9 -- NOT reproducible from any surviving input',
    'work-probe178410a':     'the ONLY proof mcp2k evt 178410 SIGSEGV is non-deterministic; 17 MB',
    # CORRECTION to the 08-31/09-01 PROTECTED entry, which said bare work/
    # holds work/nusel_labels/.  It does not, and never did in this tree -- it
    # holds pr137_sheets/ only.  The tree's single nusel label store lived
    # inside work-stmcamp-d66new/ and was git mv'd to ./nusel_labels/d66flip/
    # by this round before that dir was released.
    'work':                  'bare work/ -- holds pr137_sheets/ (doc pr/137 scan record, M13)',
}

# Fold the prefix families in.  (This loop was DROPPED by the first splice of
# this fork and the trial planner run caught it: work-87*, work-sent130* and
# work-pr134-f086-* -- 40 arms, 4.9 G, including the sentinel registry's
# negative controls -- had silently landed in the removal set.  ASSERT 7 and
# ASSERT 11 both refused, which is exactly what they are for.)
for d in dirs:
    for pfx, why in KEEP_PREFIX.items():
        if d.startswith(pfx):
            KEEP_WHY.setdefault(d, why)

KEEP = set(KEEP_WHY)

# Provenance edges checked by ASSERT 8: PR arm -> the Q/L root its own log says
# it read.  RE-DERIVED from each arm's log on every run, so a stale hardcode
# cannot pass silently.  Unchanged from 08-25b -- this round retires no Q/L
# root, so no SUCCESSOR substitution is needed and none is declared.
PR_PROVENANCE = {
    # EMPTY this round.  The only two edges the 09-01 planner carried
    # (work-vfr1qlmc-cbr3on -> work-r1qlmc-vfcbr3on and the r2mc twin) name
    # arms that are BOTH released here, so the edge is retired with them
    # rather than left pointing at a gap.  ASSERT 8 degenerates to a no-op and
    # says so; it is not disarmed.
}
# ---- ASSERT 14: the successor gate ---------------------------------------
# prod0901 may only be released because prod0901b reproduces it on every
# SHARED product.  This is not an inference from the five gated 308-event
# links -- it is measured at 3067 events by scripts/doc89_successor_gate.py,
# which writes state-20260901/successor-gate.tsv.  Checked by ROW COUNT, never
# `[ -s ]` (the 08-25b lesson), and re-derived by the driver at deletion time.
SUCCESSOR = {
    # EMPTY.  doc 91 is a COUNT-driven round, not a production rebase: every
    # production arm stays (owner: "We want to keep the latest production
    # though").  With no superseded arm in the removal set ASSERT 14 reports
    # "gate not required" and passes -- which is why ASSERT 15 and 16 below
    # exist, so this round is not guarded only by asserts that cannot fire.
}
SUCCESSOR_GATE = os.path.join(STATE, "successor-gate.tsv")
SUCCESSOR_ROWS = 3067

LIVE_MANIFESTS = {
    # RETIRED from LIVE 2026-09-01b: em117-117onK1-manifest.tsv (98 rows) is
    # superseded by em117-134f08698-manifest.tsv below -- the SAME 98 events
    # at the 0.86 EM scale that is now production (doc pr/135).  Keeping both
    # would pin the onK1 arms to display a superseded energy scale.  It moves
    # to ASSERT 11's STATED COST list, where the breakage is printed by name.
    'em_display/em117-pr130q98-manifest.tsv':      98,   # doc pr/130 census source (kept)
    'em_display/em114c-pr130q141-manifest.tsv':  141,
    'em_display/em117-134f08698-manifest.tsv':    98,   # NEW: the 0.86 production point
    'em_display/em114c-134f086141-manifest.tsv': 141,
}


# ---- ASSERT 13: the peer session's own verified dependency list -----------
# Sent by the concurrently-running session after it grepped its scripts, the
# manifests its scorers read and its score-script defaults -- not from memory.
# Encoded as an assert rather than trusted as prose, for the same reason the
# 08-23 hand-read of the open doc's arm table became ASSERT 12 that round.
# NOTE: the peer also named work-pr127r1-flipS{98,141}-* as a must-keep.  Those
# arms DO NOT EXIST on disk and are not created or removed by this round --
# scripts/pr127_sentinels.py:30's docstring reference was already stale before
# today.  Recorded in PROTECTED.txt block 3; NOT silently repaired, the peer
# owns that file.
# doc 89: there IS a peer session this round, and it is running.  Recorded here
# so ASSERT 13 states the dependency rather than leaving it to ASSERT 12's
# accident of timing -- ASSERT 12 only sees an arm that is being written in its
# window, while this is a standing claim about the round.
PEER_DEPS = []
# doc 90's nine arms (work-90pi0-r1..r5, -r6probe, -wide-mcp1k, -wide-nuecc48,
# work-90rep-A) were PROTECTED by prefix on 09-01 while that round was live.
# The owner released them by name on 2026-09-01: "Peer is done, we can remove
# or retire them."  They are in the removal set, and doc 90's determinism claim
# (140 pi0 events, 0 differences) becomes text-plus-archived-records; the
# repeat arms themselves do not survive.  STATED, not absorbed.
_PEER_DEPS_OLD = ['work-90pi0-r1', 'work-90pi0-r2', 'work-90pi0-r3', 'work-90pi0-r4']

PEER_STALE = ['work-pr127r1-flipS98', 'work-pr127r1-flipS141']

# ---- ASSERT 12: concurrent-writer safety ----------------------------------
# The tree is NOT quiet.  A removal-set dir touched within this window, or
# named by a live process, refuses the round.  Deliberately generous: the
# peer's per-event runs are minutes long, and a stale mtime is the whole point.
FRESH_WINDOW_S = 3600

# ---- ASSERT 10: script literals into the removal set, acknowledged --------
# Every name here is a hardcoded path in a .py/.sh that stops resolving after
# this round.  A removal-set name NOT listed here refuses the round.
ACK_BROKEN_REFS = {
    # DISPOSITION RULE, unchanged from 08-25b: a script that RECORDS how a
    # finished round was run keeps naming the arm that round actually read.
    # Repointing those would make them claim a provenance they did not have.
    # LIVE defaults get the opposite treatment -- they are in KEEP, above.
    'work-pr118r1-dbg-mcp1k':   'em_display 98-set arm builder, doc pr/118 CLOSED (two-tier merge SBND ON)',
    'work-pr118r1-dbg-mcp2k':   'em_display 98-set arm builder, doc pr/118 CLOSED',
    'work-pr118r1-dbg-ncpi0':   'em_display 98-set arm builder, doc pr/118 CLOSED',
    'work-pr118r1-dbg-nuecc48': 'em_display 98-set arm builder, doc pr/118 CLOSED',
    'work-pr119r1-dbgA-mcp1k':   'em_display 98-set arm builder, doc pr/119 CLOSED (measured dead)',
    'work-pr119r1-dbgA-mcp2k':   'em_display 98-set arm builder, doc pr/119 CLOSED',
    'work-pr119r1-dbgA-ncpi0':   'em_display 98-set arm builder, doc pr/119 CLOSED',
    'work-pr119r1-dbgA-nuecc48': 'em_display 98-set arm builder, doc pr/119 CLOSED',
    'work-pr120r1-dbgA-mcp1k':   'em_display 98-set arm builder, doc pr/120 CLOSED (backward-stem guard SBND ON)',
    'work-pr120r1-dbgA-mcp2k':   'em_display 98-set arm builder, doc pr/120 CLOSED',
    'work-pr120r1-dbgA-ncpi0':   'em_display 98-set arm builder, doc pr/120 CLOSED',
    'work-pr120r1-dbgA-nuecc48': 'em_display 98-set arm builder, doc pr/120 CLOSED',
    # --- added 2026-08-31: the pr/129-132 rounds are CLOSED and these are
    # their RECORD scripts (how the round was run) or comments citing where
    # a measured value came from.  Disposition rule unchanged: a script that
    # records a finished round keeps naming the arm that round actually read.
    'work-pr129gf-nokine-mcp2k': 'scripts/pr129_guardfreed_arm.sh:19 -- doc pr/129 CLOSED (pointing test)',
    'work-pr129probe-mcp1k':     'scripts/pr129_probe_arms.sh:29 -- doc pr/129 CLOSED',
    'work-pr129probe-mcp2k':     'scripts/pr129_probe_arms.sh:25 -- doc pr/129 CLOSED',
    'work-pr129r1-on98-mcp2k':   'pr127_sentinels.py:161 COMMENT recording where a rendered value was measured',
    'work-pr130-47212-guardoff': 'pr127_sentinels.py:163 COMMENT, the guard-OFF counterpart of the same note',
    'work-pr130-qx1-mcp1k':      'scripts/pr130_qextra_attrib.py:35 -- doc pr/130 CLOSED (q_extra attribution)',
    'work-pr130-qx1-mcp2k':      'scripts/pr130_qextra_attrib.py:35 -- doc pr/130 CLOSED',
    'work-pr132-r4off-mcp2k':    'scripts/pr132_vtx_census.py:25 -- doc pr/132 CLOSED (vertex census default)',
    # --- added 2026-08-31b with the scan-time dump release ---------------
    'work-mcp1k-prod0825':       'scripts/multi/smoke_rc.sh:8 -- a smoke-runner default; the scan-time PR arm it names is released this pass',
    # --- added 2026-09-01 (doc 89).  Disposition rule unchanged: a script
    # that RECORDS how a finished round was run keeps naming the arm that
    # round actually read; repointing it would make it claim a provenance it
    # did not have.  The prod0830/prod0901/empre0901 entries are the same
    # rule applied to a SUPERSEDED baseline -- prod0901b is what a fresh run
    # should read, and doc 89 sec 7 says so where these scripts are listed.
    'work-mcp1k-empre0901': 'scripts/pr142_analyze.sh:41 -- the doc pr/142 "before" arm; doc COMPLETE, tables in products/empre0901/',
    'work-mcp1k-prod0830': 'scripts/bee/build_d86_bee.sh:22 + d85r2 -- doc 85/86 RECORD scripts; prod0830 is superseded by prod0901b',
    'work-mcp1k-prod0901': 'scripts/pr142_analyze.sh:41 -- doc pr/142 RECORD script; superseded by prod0901b (interlock 7)',
    'work-mcp2k-prod0830': 'scripts/bee/build_d86_bee.sh:22 -- doc 86 video-showcase record script',
    'work-mcp2k-prod0901': 'scripts/cfg/fire_census.py:33 -- doc 77 fire-census default; superseded by prod0901b',
    'work-ncpi0-prod0830': 'scripts/bee/build_d86_bee.sh:22 -- doc 86 record script',
    'work-ncpi0-prod0901': 'scripts/cfg/fire_census.py:34 -- doc 77 fire-census default; superseded by prod0901b',
    'work-nuecc48-empre0901': 'scripts/pr142_analyze.sh:44 -- the doc pr/142 "before" arm; doc COMPLETE',
    'work-nuecc48-prod0830': 'scripts/bee/build_d86_bee.sh:22 -- doc 86 record script',
    'work-nuecc48-prod0901': 'scripts/pr142_analyze.sh:44 -- doc pr/142 record script; superseded by prod0901b',
    'work-pr140r2-off-mcp1k': 'scripts/pr141_massfail.py:60 -- doc pr/141 CLOSED, its own baseline arm',
    'work-pr140r2-off-mcp2k': 'scripts/pr141_massfail.py:61 -- doc pr/141 CLOSED',
    'work-pr140r2-off-ncpi0': 'scripts/pr141_massfail.py:59 -- doc pr/141 CLOSED',
    'work-pr140r2-off-nuecc48': 'scripts/pr141_pidset.py:47 -- doc pr/141 CLOSED',
    'work-pr141dbg-pair-mcp1k': 'scripts/pr141_pairtape.py:30 -- doc pr/141 CLOSED, single-event debug arm',
    'work-pr141dbg-pair-mcp2k': 'scripts/pr141_pairtape.py:31 -- doc pr/141 CLOSED',
    'work-pr141dbg-pair2-mcp1k': 'scripts/pr141_pairtape.py:33 -- doc pr/141 CLOSED',
    'work-pr141dbg-pair2-mcp2k': 'scripts/pr141_pairtape.py:34 -- doc pr/141 CLOSED',
    'work-pr141dbg-pair2-ncpi0': 'scripts/pr141_pairtape.py:29 -- doc pr/141 CLOSED',
    # --- added 2026-09-01b (doc 91).  DISPOSITION RULE UNCHANGED, but with a
    # correction to the 09-01 planner: it kept work-em114c-prodnow-* and
    # work-pr117r1-onK1-mcp1k on the stated ground that they were "hardcoded
    # default arm, em_display/prep_pr121.py:15" / "prep_pr117.py:17,28".  That
    # was a MIS-READ.  Both files were re-checked this round: neither has an
    # argparse default naming an arm -- every hit is a docstring USAGE line or
    # a comment.  `grep -n "default=" em_display/prep_pr11{7,21}.py` matches
    # no arm name.  So these fall under the record-script rule after all, and
    # the LIVE-default exemption never applied to them.
    #
    # SUCCESSOR for the display baseline: the 98-set and 141-set displays now
    # come from em117-134f08698-manifest.tsv / em114c-134f086141-manifest.tsv,
    # backed by the kept work-pr134-f086-* arms at the 0.86 production scale.
    'work-em114c-prodnow-mcp1k': 'em_display/prep_pr121.py:15 USAGE line; 141-set display now from the f086 manifest',
    'work-em114c-prodnow-mcp2k': 'em_display/prep_pr121.py:15 USAGE line; superseded by the f086 manifest',
    'work-pr117r1-onK1-mcp1k':   'em_display/prep_pr117.py:17,28 USAGE line + a regex comment; superseded by em117-134f08698',
    'work-em114-probe3':         'em_display/run_em114_probe.sh:26 COMMENT recording doc pr/114 sec 3; its prod0825 comparator was released 08-31b, so the proof was already text-only',
    'work-nuecc48-prsmoke2':     'scripts/repro_track_pid_evt172230.py:4 -- ALREADY broken: that path names nupr_evt172230_mipvote_on/, a subdir that no longer exists. The 3 runner scripts it held were git mv-d to scripts/legacy/',
    'work-r1qlmc-prod0813':      'scripts/pr83r2_census.py:15 COMMENT recording a 2026-08-17 equivalence check -- a record of a finished round',
    'work-r2mc-prod0813':        'scripts/pr83r2_census.py:15 COMMENT, same check',
    'work-stmcamp-d66new':       'scripts/analysis/stm/d66_flip_report.py:15 USAGE line naming d66old, itself retired rounds ago. Its nusel_labels/d66flip store was git mv-d to ./nusel_labels/d66flip and d66_scan_score.py repointed to --root .',
}

REF_ROOTS = ['scripts', 'vtx_rules', 'dl_vtx_training', 'em_display']


def read_protected(path):
    out = set()
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        out.update(line.split("\t")[0].split())
    return out


PROT_LISTED = read_protected(os.path.join(SCR, "PROTECTED.txt"))
RELEASED = sorted(PROT_LISTED - KEEP)


def tier(d):
    return 'KEEP' if d in KEEP else 'A'


def group(d):
    for pat, g in ((r'^work-d84r1', 'doc84-r1'),
                   (r'^work-d84r2', 'doc84-r2'),
                   (r'^work-d84r3', 'doc84-r3'),
                   (r'^work-d84r4', 'doc84-r4'),
                   (r'^work-(mcp1k|mcp2k)-mcs80', 'doc80-mcs'),
                   (r'^work-em114', 'doc114-emdisplay'),
                   (r'^work-em11[567]', 'doc115-117-emscan'),
                   (r'^work-pr117r1', 'pr117-family'),
                   (r'^work-pr118r1', 'pr118-family'),
                   (r'^work-pr119r1', 'pr119-family'),
                   (r'^work-pr120r1', 'pr120-family'),
                   (r'^work-pr121r1', 'pr121-family'),
                   (r'^work-pr122', 'pr122-family'),
                   (r'^work-pr123r1', 'pr123-family'),
                   (r'^work-pr124r1', 'pr124-family'),
                   (r'^work-pr125r1', 'pr125-family'),
                   (r'^work-pr126', 'pr126-family'),
                   (r'^work-pr127r1', 'pr127-family'),
                   (r'^work-pr128r1', 'pr128-family'),
                   (r'^work-vtx', 'vtx-family'),
                   (r'^work-probe', 'probe-arms'),
                   (r'^work-vf', 'valfast-transient')):
        if re.match(pat, d):
            return g
    return 'other'


bytier = collections.defaultdict(list)
for d in dirs:
    bytier[tier(d)].append(d)

K, TA = bytier['KEEP'], bytier['A']
R = sorted(TA)
ARCHIVE = sorted(TA)
assert len(dirs) == len(R) + len(K), "classes do not partition work*"
missing_keep = sorted(KEEP - set(dirs))

# ---------------------------------------------------------------- footprint
# HEAVY = a reproducible data product; everything else is the record layer and
# is archived verbatim.  Carried from 08-25 with ONE addition, and the addition
# is the reason this round ran a census instead of reusing the list on trust.
#
# .groups/g<N>.tar.gz -- the group INPUT archive of a group-mode PR run: a
# bundle of Q/L pctrees handed to the run, i.e. copied data, not a record.  The
# 08-25 round could justify "HEAVY unchanged" with a census finding ZERO
# unclassified file above 5 MiB; that was true of ITS removal set and is false
# here, because none of its arms was group-mode and doc 84 round 3's census
# arms are.  Unclassified, they put 4.96 GiB of duplicated pctree data into the
# record tar.  PROVEN duplicates, not assumed:
# verify_group_dupes_20260829.py checked all 188 archives member by member
# against the surviving grp0825 Q/L roots -- 1231656/1231656 byte-identical
# (state-20260829/group-dupes.tsv).  Matched on the FULL PATH, not the
# basename: a stray g1.tar.gz outside a .groups/ dir must stay a record.
HEAVY = (re.compile(r'^pctree.*\.tar\.gz$'), re.compile(r'^mabc.*\.zip$'),
         re.compile(r'^calib(-pr)?-evt.*\.json(\.gz)?$'), re.compile(r'.*\.npz$'),
         re.compile(r'^clusters-apa.*\.tar\.gz$'),
         re.compile(r'^opflash_apa.*\.tar\.gz$'),
         re.compile(r'^tracking-pr\.root$'),
         re.compile(r'^oc56scan-evt.*\.jsonl$'))
HEAVY_PATH = (re.compile(r'(^|/)\.groups/g\d+\.tar\.gz$'),)


def is_heavy(f, path=None):
    if any(p.match(f) for p in HEAVY):
        return True
    return path is not None and any(p.search(path) for p in HEAVY_PATH)


RECORD_DIR = re.compile(r'^(nusel_labels|ql_labels|decisions.*)$')

Rset_for_labels = set(R)
per = {}
label_hits = []
for d in dirs:
    tot = keep = nk = nh = 0
    for cur, sub, files in os.walk(d):
        sub[:] = [s for s in sub if not os.path.islink(os.path.join(cur, s))]
        for s in sub:
            if RECORD_DIR.match(s) and d in Rset_for_labels:
                label_hits.append(os.path.join(cur, s))
        for f in files:
            p = os.path.join(cur, f)
            if os.path.islink(p):
                continue
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            tot += sz
            if is_heavy(f, p):
                nh += 1
            else:
                keep += sz
                nk += 1
    per[d] = dict(tot=tot, keep=keep, nk=nk, nh=nh)

# ---------------------------------------------------------------- report
print("=== RETIREMENT ROUND 2026-08-31b (scan-time dumps + superseded pi0 reference points) ===")
print(f"universe {len(dirs)} work* dirs -> KEEP {len(K)}, remove {len(R)}")
if missing_keep:
    print(f"!! KEEP names not on disk: {missing_keep}")
print("\n[KEEP]")
for d in sorted(K):
    print(f"    {d:34s} {per.get(d,{}).get('tot',0)/2**20:8.0f} MB  {KEEP_WHY[d]}")
print(f"    {'--- KEEP TOTAL':34s} {sum(per[d]['tot'] for d in K)/2**30:8.2f} GiB")
print(f"\n[REMOVE] {len(R)} dirs, {sum(per[d]['tot'] for d in R)/2**30:.2f} GiB")
for d in sorted(R, key=lambda x: -per[x]['tot'])[:25]:
    print(f"    {d:34s} {per[d]['tot']/2**20:8.0f} MB  [{group(d)}]")
if len(R) > 25:
    print(f"    ... and {len(R)-25} more (full list: tierA_20260901.txt)")

print(f"\n[RELEASED from PROTECTED.txt] {len(RELEASED)}")
print("    " + (" ".join(RELEASED) if RELEASED else "(none)"))

print("\n=== ARCHIVE FOOTPRINT ===")
stat = collections.defaultdict(lambda: [0, 0, 0])
for d in ARCHIVE:
    s = stat[group(d)]
    s[0] += per[d]['tot']; s[1] += per[d]['keep']; s[2] += per[d]['nk']
for g in sorted(stat):
    t, k, nk = stat[g]
    print(f"{g:22s} total {t/2**30:6.2f} GiB  archive {k/2**20:8.1f} MiB ({nk} files)  "
          f"reclaim {(t-k)/2**30:6.2f} GiB")
T = sum(per[d]['tot'] for d in R)
Kb = sum(stat[g][1] for g in stat)
print(f"{'TOTAL':22s} total {T/2**30:6.2f} GiB  archive {Kb/2**20:8.1f} MiB  "
      f"reclaim {(T-Kb)/2**30:6.2f} GiB")

# ---------------------------------------------------------------- asserts
fail = 0

print("\n=== ASSERT 1: no real SP frame is lost -- source dirs survive locally ===")
SP_SOURCES = {
    'mcp1k (1000 data)':  'input_files_reco1/staged-mcp2025c-1000evt',
    'mcp2k (2000 data)':  'input_files_reco1/staged-mcp2025c-2nd-2000evt',
    'nuecc48 (48 data)':  'input_files_reco1/extracted-2025fall-48evt-fsprod',
    'ncpi0 (19 data)':    'input_files_reco1/extracted-ncpi0',
    'r1qlmc (10 sim)':    'input_files_reco1/extracted-r1ql-f1',
    'r2mc (13 sim)':      'input_files_reco1/extracted-r2patrec-f1',
}
for label, src in sorted(SP_SOURCES.items()):
    ok = os.path.isdir(src) and not os.path.islink(src) and bool(os.listdir(src))
    n = len(os.listdir(src)) if os.path.isdir(src) else 0
    print(f"      {'OK ' if ok else '!! '} {label:20s} {src}  ({n} entries)")
    if not ok:
        fail += 1
print("    No imaging SP layer drops this round -- there is no Phase 4 thinning.")

print("\n=== ASSERT 2: every hand-scan / label record has a verified archive copy (M13) ===")
LABROOT = os.path.join(ROOT, "archive", "records", "labels")


def tree_identical(a, b):
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    if mismatch or errors:
        return False
    return all(tree_identical(os.path.join(a, d), os.path.join(b, d))
               for d in cmp.common_dirs)


if not label_hits:
    print("0 label dirs in the removal set -- PASS (strict form)")
    print("    (as of 2026-09-01b NO label dir lives inside work-* at all.  The")
    print("     last one -- work-stmcamp-d66new/nusel_labels/d66flip/, the tree's")
    print("     ONLY nusel label store -- was git mv-d to ./nusel_labels/d66flip/")
    print("     by this round, 22 tracked files, BEFORE its dir entered the")
    print("     removal set.  em_labels/, vertex_labels/, overclustering_labels/")
    print("     and now nusel_labels/ are all top-level and outside work-*, so")
    print("     this assert is strictly clean rather than clean-because-the-dir-")
    print("     is-in-KEEP as it was on 09-01.)")
else:
    for p in sorted(label_hits):
        rel = os.path.relpath(p, ROOT)
        dst = os.path.join(LABROOT, rel)
        nsrc = sum(len(f) for _, _, f in os.walk(p))
        if not os.path.isdir(dst):
            print(f"  !! NO ARCHIVE COPY: {rel}")
            fail += 1
        elif not tree_identical(p, dst):
            print(f"  !! ARCHIVE COPY DIFFERS: {rel}")
            fail += 1
        else:
            ntag = len(os.listdir(p))
            print(f"  OK  {rel:38s} {ntag:2d} tags, {nsrc:3d} files -> archive copy verified")

print("\n=== ASSERT 3: no git-tracked file inside the removal set ===")
tracked = subprocess.run(['git', '-C', '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img',
                           'ls-files', '-z', '--'] + ['sbnd/sbnd_xin/' + d for d in R],
                          capture_output=True, text=True).stdout.split('\0')
tracked = [t for t in tracked if t]
if not tracked:
    print("0 -- PASS")
else:
    fail += 1
    for t in tracked[:50]:
        print(f"  !! {t}")

print("\n=== ASSERT 4: dangling-link dry run ===")
Rset = set(R)
bad = collections.Counter()
nlinks = 0
top = [e for e in sorted(os.listdir('.')) if e not in Rset and e != '.git']
for root in top:
    if os.path.islink(root):
        m = re.search(r'sbnd_xin/(work[^/]*)', os.readlink(root))
        if m and m.group(1) in Rset:
            bad[(root, m.group(1))] += 1
        continue
    if not os.path.isdir(root):
        continue
    for cur, sub, files in os.walk(root, followlinks=False):
        for name in sub + files:
            p = os.path.join(cur, name)
            if os.path.islink(p):
                nlinks += 1
                m = re.search(r'sbnd_xin/(work[^/]*)', os.readlink(p))
                if m and m.group(1) in Rset:
                    bad[(root, m.group(1))] += 1
print(f"({nlinks} symlinks outside removal set, hidden dirs included)")
if not bad:
    print("0 -- PASS")
else:
    fail += 1
    for (s, t), c in bad.most_common():
        print(f"  !! {c:6d}  {s} -> {t}")

print("\n=== ASSERT 5: every KEEP name exists and is non-empty ===")
bad_keep = [d for d in KEEP if not (os.path.isdir(d) and os.listdir(d))]
print("0 -- PASS" if not bad_keep else f"  !! {bad_keep}")
if bad_keep:
    fail += 1

print("\n=== ASSERT 6: overclustering_labels archived + git-tracked (carried from 08-11) ===")
occ_src = os.path.join(ROOT, "overclustering_labels")
occ_dst = os.path.join(LABROOT, "overclustering_labels")
if not os.path.isdir(occ_dst):
    print("  !! NO ARCHIVE COPY at archive/records/labels/overclustering_labels")
    fail += 1
elif not tree_identical(occ_src, occ_dst):
    print("  !! ARCHIVE COPY DIFFERS from live overclustering_labels/")
    fail += 1
else:
    gitcount = subprocess.run(
        ['git', '-C', '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img', 'ls-files',
         'sbnd/sbnd_xin/overclustering_labels'],
        capture_output=True, text=True).stdout.strip().splitlines()
    if len(gitcount) < 200:
        print(f"  !! only {len(gitcount)} files git-tracked under overclustering_labels (expect >=230)")
        fail += 1
    else:
        print(f"  OK  archive copy verified identical, {len(gitcount)} files committed to git")

print("\n=== ASSERT 6b (NEW): em_labels -- the pi0 hand-scan record -- is protected ===")
# Found by this round, not assumed: em_labels/ had ZERO git-tracked files (the
# repo .gitignore has `*.json` at line 2) and no archive copy, while it holds
# the literal input to the owner's stated next move.  Same two-part test as
# ASSERT 6: verified archive copy AND committed to git.  No em_labels byte is
# retired by this round -- this is the record layer it declines to leave broken.
em_src = os.path.join(ROOT, "em_labels")
em_dst = os.path.join(LABROOT, "em_labels")
nj = len(glob.glob(os.path.join(em_src, '*', '*.json')))
if not os.path.isdir(em_dst):
    print(f"  !! NO ARCHIVE COPY at archive/records/labels/em_labels ({nj} label JSONs live)")
    fail += 1
elif not tree_identical(em_src, em_dst):
    print("  !! ARCHIVE COPY DIFFERS from live em_labels/")
    fail += 1
else:
    g = subprocess.run(['git', '-C', '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img',
                        'ls-files', 'sbnd/sbnd_xin/em_labels'],
                       capture_output=True, text=True).stdout.strip().splitlines()
    if len(g) < nj:
        print(f"  !! only {len(g)} of {nj} em_labels JSONs git-tracked "
              f"(`*.json` is gitignored -- needs `git add -f`, M9)")
        fail += 1
    else:
        print(f"  OK  archive copy verified identical, {len(g)} files committed "
              f"({nj} label JSONs across {len(os.listdir(em_src))} scan tags)")

print("\n=== ASSERT 7: PROTECTED.txt is honoured, not merely printed ===")
prot_missing = sorted(PROT_LISTED - KEEP)
if not prot_missing:
    print(f"0 -- PASS  ({len(PROT_LISTED)} PROTECTED.txt names, all in KEEP)")
else:
    fail += 1
    print(f"  !! PROTECTED.txt names NOT in KEEP (would be swept): {prot_missing}")


def _group_provenance(pr_arm, expected_src):
    """Confirm a group-mode PR arm was built from expected_src, by evidence."""
    import hashlib, tarfile
    gdir = os.path.join(pr_arm, ".groups")
    if not os.path.isdir(gdir):
        return None
    for bl in sorted(glob.glob(os.path.join(gdir, "g*-build.log"))):
        with open(bl, errors="replace") as fh:
            m = re.search(r"group provenance: ql_root=(\S+)", fh.read())
            if m:
                return os.path.basename(m.group(1).rstrip("/"))
    tars = sorted(glob.glob(os.path.join(gdir, "g*.tar.gz")))
    if not tars or not os.path.isdir(expected_src):
        return None
    try:
        t = tarfile.open(tars[0])
        got = {ti.name: hashlib.sha256(t.extractfile(ti).read()).hexdigest()
               for ti in t if ti.isfile()}
    except Exception:
        return None
    if not got:
        return None
    want = {}
    for evt in sorted(os.listdir(expected_src)):
        if not evt.startswith("ql_evt"):
            continue
        f = os.path.join(expected_src, evt, "pctree-evt%s.tar.gz" % evt[len("ql_evt"):])
        if not os.path.exists(f):
            continue
        try:
            st = tarfile.open(f)
            for ti in st:
                if ti.isfile():
                    want[ti.name] = hashlib.sha256(st.extractfile(ti).read()).hexdigest()
        except Exception:
            return None
    if want and all(want.get(k) == v for k, v in got.items()):
        return os.path.basename(expected_src.rstrip("/"))
    return None


print("\n=== ASSERT 8: KEEP is closed under PR-arm provenance ===")
# THE ROUND-LEVEL FACT COMES FIRST, and it is checked, not assumed.  The
# per-arm log grep can only ever say "this arm names a root that survives";
# what the assert actually protects is "no kept arm's input is being deleted".
# If the removal set contains NO Q/L root (ql_evt<N>/) and NO imaging root
# (evt<N>/) at all, that property holds for every kept arm unconditionally,
# whether or not its own log happens to record provenance.
#
# This matters concretely: work-{ncpi0,nuecc48}-prod0825 record NO provenance
# in either supported form.  Their .groups/g*-build.log predate the
# "group provenance: ql_root=" line make_group_pctree.py writes today (mcp1k's
# and mcp2k's have it), and their g*.tar.gz were cleaned up after the build, so
# the byte-identity fallback has nothing to read either.  That is a pre-existing
# gap in those two arms' records, NOT something this round causes, and it is
# reported as UNVERIFIED rather than silently downgraded.  If any root were in
# the removal set, unverifiable would remain a hard FAIL -- the relaxation is
# gated on the round-level fact below, which is the whole point.
ROOTS_IN_R = sorted(d for d in R if os.path.isdir(d) and
                    any(e.startswith('ql_evt') or (e.startswith('evt') and e[3:].isdigit())
                        for e in os.listdir(d)))
if ROOTS_IN_R:
    print(f"  !! the removal set contains {len(ROOTS_IN_R)} Q/L or imaging root(s): "
          f"{ROOTS_IN_R[:8]} -- per-arm provenance MUST be proven for every kept arm")
else:
    print("  OK  removal set contains 0 Q/L roots (ql_evt<N>/) and 0 imaging roots "
          "(evt<N>/): no kept arm's input can be deleted by this round")
prov_fail = []
prov_unverified = []
for pr_arm, expected_src in sorted(PR_PROVENANCE.items()):
    if pr_arm not in KEEP:
        prov_fail.append((pr_arm, expected_src, "arm itself not in KEEP"))
        continue
    evt_dirs = sorted(e for e in os.listdir(pr_arm) if e.startswith('pr_evt')) \
        if os.path.isdir(pr_arm) else []
    found_src = None
    for evt in evt_dirs[:1]:
        for f in os.listdir(os.path.join(pr_arm, evt)):
            if not f.endswith('.log'):
                continue
            with open(os.path.join(pr_arm, evt, f), errors='replace') as fh:
                m = re.search(r'(work-[a-zA-Z0-9-]+)/ql_evt', fh.read())
                if m:
                    found_src = m.group(1)
                    break
        break
    if found_src is None:
        found_src = _group_provenance(pr_arm, expected_src)
    if found_src is None:
        why = "no ql_evt provenance line, and no group-input evidence (g*.tar.gz cleaned up)"
        if ROOTS_IN_R:
            prov_fail.append((pr_arm, expected_src, why))
        else:
            prov_unverified.append((pr_arm, expected_src, why))
    elif found_src != expected_src:
        prov_fail.append((pr_arm, expected_src, f"log actually reads {found_src}"))
    elif found_src not in KEEP:
        prov_fail.append((pr_arm, expected_src,
                          f"{found_src} not in KEEP and no SUCCESSOR declared this round"))
    else:
        print(f"  OK  {pr_arm:34s} -> {found_src} (in KEEP)")
for arm, exp, why in prov_unverified:
    print(f"  UNVERIFIED  {arm:26s} -> expected {exp} (in KEEP): {why}")
if not prov_fail:
    n_ok = len(PR_PROVENANCE) - len(prov_unverified)
    print(f"0 -- PASS  ({n_ok}/{len(PR_PROVENANCE)} PR arms proven; {len(prov_unverified)} "
          f"UNVERIFIED but harmless -- 0 roots in the removal set)")
else:
    fail += 1
    for arm, exp, why in prov_fail:
        print(f"  !! {arm} (expected src {exp}): {why}")

print("\n=== ASSERT 9: KEEP is closed FORWARD over the campaign input set ===")
# doc 91: the r1qlmc/r2mc sim chain is released IN FULL -- both imaging hubs,
# both Q/L hubs, both prod0813 PR arms, both -vfcbr3on Q/L roots and both
# vf*-cbr3on out-roots.  Closure is preserved by releasing the whole chain,
# not by exempting a link: neither sample is in the production four, nothing
# has read them since 2026-08-17, and PROTECTED.txt already records that the
# bokeh pin they were held for had lapsed.
CAMPAIGN_IMG = {}
CAMPAIGN_OUT = {
    'work-nuecc48-grp0825': ('ql_evt',   48), 'work-ncpi0-grp0825': ('ql_evt',   19),
    'work-mcp1k-grp0825':   ('ql_evt', 1000), 'work-mcp2k-grp0825': ('ql_evt', 2000),
    'work-vtx105-base-nuecc48': ('pr_evt',  47),
    'work-vtx105-base-ncpi0':   ('pr_evt',  19),
    'work-vtx105-base-mcp1k':   ('pr_evt', 407),
    'work-vtx105-base-mcp2k':   ('pr_evt', 581),
}
GRP_IMG = {'work-nuecc48-grp0825': 48, 'work-ncpi0-grp0825': 19,
           'work-mcp1k-grp0825': 1000, 'work-mcp2k-grp0825': 2000}
CAMPAIGN_INPUT = {
    'input_files_reco1/staged-mcp2025c-1000evt':      1001,
    'input_files_reco1/staged-mcp2025c-2nd-2000evt':  2001,
    'input_files_reco1/extracted-2025fall-48evt-fsprod': None,
    'input_files_reco1/extracted-ncpi0':                 None,
}
a9_fail = 0
for hub, want in sorted(CAMPAIGN_IMG.items()):
    if hub not in KEEP:
        print(f"  !! {hub} is NOT in KEEP -- the campaign would silently skip its events")
        a9_fail += 1
        continue
    got = len([e for e in os.listdir(hub) if e.startswith('evt')]) if os.path.isdir(hub) else 0
    if got != want:
        print(f"  !! {hub}: {got} evt* dirs, expected {want}")
        a9_fail += 1
    else:
        print(f"  OK  {hub:20s} {got:5d} evt* dirs, in KEEP")
for src, want in sorted(CAMPAIGN_INPUT.items()):
    ok = os.path.isdir(src) and bool(os.listdir(src))
    if not ok:
        print(f"  !! {src}: missing or empty")
        a9_fail += 1
        continue
    emap = os.path.join(src, 'entry_event_map.tsv')
    if want is None:
        print(f"  OK  {src}  ({len(os.listdir(src))} entries, no entry_event_map.tsv expected)")
    elif not os.path.exists(emap):
        print(f"  !! {src}: entry_event_map.tsv missing (run_ql_batch.sh:90-97 needs it)")
        a9_fail += 1
    else:
        n = sum(1 for _ in open(emap))
        if n != want:
            print(f"  !! {emap}: {n} lines, expected {want}")
            a9_fail += 1
        else:
            print(f"  OK  {src}  entry_event_map.tsv {n} lines")
for arm, (pfx, want) in sorted(CAMPAIGN_OUT.items()):
    if arm not in KEEP:
        print(f"  !! {arm} is NOT in KEEP -- would be swept")
        a9_fail += 1
        continue
    got = len([e for e in os.listdir(arm) if e.startswith(pfx)]) if os.path.isdir(arm) else 0
    if got != want:
        print(f"  !! {arm}: {got} {pfx}* dirs, expected {want}")
        a9_fail += 1
    else:
        print(f"  OK  {arm:34s} {got:5d} {pfx}* dirs, in KEEP")
for arm, want in sorted(GRP_IMG.items()):
    if arm not in KEEP:
        print(f"  !! {arm} is NOT in KEEP -- the imaging layer would be lost outright")
        a9_fail += 1
        continue
    got = len([e for e in os.listdir(arm)
               if e.startswith('evt') and e[3:].isdigit()]) if os.path.isdir(arm) else 0
    if got != want:
        print(f"  !! {arm}: {got} evt* dirs, expected {want}")
        a9_fail += 1
    else:
        print(f"  OK  {arm:34s} {got:5d} evt* dirs (imaging layer), in KEEP")
if not a9_fail:
    print(f"0 -- PASS  ({len(CAMPAIGN_IMG)} imaging hubs + {len(CAMPAIGN_INPUT)} SP inputs + "
          f"{len(CAMPAIGN_OUT)} kept arms, counts verified, all in KEEP)")
else:
    fail += 1

print("\n=== ASSERT 10: every script literal into the removal set is acknowledged ===")
lit = collections.defaultdict(list)
scan_files = []
for rt in REF_ROOTS:
    for cur, sub, files in os.walk(rt):
        sub[:] = [s for s in sub if s != '__pycache__' and not os.path.islink(os.path.join(cur, s))]
        scan_files += [os.path.join(cur, f) for f in files if f.endswith(('.py', '.sh'))]
scan_files += [f for f in os.listdir('.') if f.endswith(('.py', '.sh')) and os.path.isfile(f)]
scan_files = [f for f in scan_files if not f.startswith('scripts/retire/')]
for f in scan_files:
    try:
        with open(f, errors='replace') as fh:
            for i, line in enumerate(fh, 1):
                for m in re.findall(r'work-[a-zA-Z0-9_-]+', line):
                    if m in Rset:
                        lit[m].append(f"{f}:{i}")
    except (OSError, IsADirectoryError):
        continue
unacked = sorted(set(lit) - set(ACK_BROKEN_REFS))
for name in sorted(lit):
    mark = 'ACK ' if name in ACK_BROKEN_REFS else '!!  '
    print(f"  {mark}{name:34s} {len(lit[name]):3d} ref(s), e.g. {lit[name][0]}")
if not unacked:
    print(f"0 -- PASS  ({len(lit)} removal-set names referenced by scripts, all acknowledged; "
          f"{len(scan_files)} files scanned)")
else:
    fail += 1
    print(f"  !! {len(unacked)} removal-set name(s) hardcoded in a script but NOT in "
          f"ACK_BROKEN_REFS: {unacked}")
    print("     Add each to ACK_BROKEN_REFS with the cost, or move it to KEEP.")

print("\n=== ASSERT 11 (NEW): the live hand-scan / pi0 manifests still resolve ===")
# Row counts and per-row path resolution, never `[ -s ]`.  These are the
# manifests the owner's next move reads.  Broken manifests OUTSIDE this set are
# a stated cost, listed below by name so the number is written down before the
# round runs rather than discovered by a failing scan weeks later.
a11_fail = 0
survive = set(K)


def manifest_arms(path):
    arms = collections.Counter()
    n = 0
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            p = (r.get('dump') or '').strip()
            if not p or '/' not in p:
                continue
            n += 1
            arms[p.split('/')[0]] += 1
    return n, arms


for m, want in sorted(LIVE_MANIFESTS.items()):
    if not os.path.exists(m):
        print(f"  !! {m}: MISSING")
        a11_fail += 1
        continue
    n, arms = manifest_arms(m)
    broken = sorted(a for a in arms if a in Rset)
    if n != want:
        print(f"  !! {m}: {n} rows, expected {want}")
        a11_fail += 1
    elif broken:
        print(f"  !! {m}: {n} rows but these arms are in the REMOVAL set: {broken}")
        a11_fail += 1
    else:
        print(f"  OK  {os.path.basename(m):42s} {n:4d} rows -> {sorted(arms)}")
if not a11_fail:
    print(f"0 -- PASS  ({len(LIVE_MANIFESTS)} live manifests, every dump path lands in KEEP)")
else:
    fail += 1
dead = []
for m in sorted(glob.glob('em_display/*manifest*.tsv')):
    if m in LIVE_MANIFESTS:
        continue
    n, arms = manifest_arms(m)
    b = sorted(a for a in arms if a in Rset)
    if b:
        dead.append((m, n, b))
print(f"\n    STATED COST: {len(dead)} closed-round manifests stop resolving --")
for m, n, b in dead:
    print(f"      {os.path.basename(m):42s} {n:4d} rows  {b[0]}{' +%d' % (len(b)-1) if len(b) > 1 else ''}")

print("\n=== ASSERT 12 (NEW): concurrent-writer safety ===")
# A peer session is mid-gate.  Prior rounds ran against a quiet tree and could
# afford a blanket refusal (M5); here that would either block the round or
# invite ALLOW_LIVE_JOBS=yes, which disarms the interlock outright.  So: a live
# process may run, but not one that names a REMOVAL-set dir, and no removal-set
# dir may have been written recently.  Re-derived at deletion time too
# (retire_20260901.sh interlock 6) -- plan-time evidence expires.
a12_fail = 0
psout = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
live_hits = sorted({d for d in R if d in psout})
if live_hits:
    print(f"  !! a live process names {len(live_hits)} removal-set dir(s): {live_hits[:8]}")
    a12_fail += 1
else:
    nwc = sum(1 for ln in psout.splitlines() if 'wire-cell ' in ln and 'sbnd_xin' in ln)
    print(f"  OK  {nwc} live sbnd_xin wire-cell process(es), none names a removal-set dir")
now = time.time()
fresh = sorted((d, now - os.path.getmtime(d)) for d in R
               if os.path.exists(d) and now - os.path.getmtime(d) < FRESH_WINDOW_S)
if fresh:
    print(f"  !! {len(fresh)} removal-set dir(s) written in the last "
          f"{FRESH_WINDOW_S//60} min: {[(d, int(a)) for d, a in fresh[:8]]}")
    a12_fail += 1
else:
    youngest = min((now - os.path.getmtime(d)) for d in R if os.path.exists(d))
    print(f"  OK  youngest removal-set dir was written {youngest/3600:.1f} h ago "
          f"(window {FRESH_WINDOW_S//60} min)")
if not a12_fail:
    print("0 -- PASS  (no live writer touches the removal set)")
else:
    fail += 1

print("\n=== ASSERT 13 (NEW): the peer session's declared dependencies are in KEEP ===")
a13 = sorted(d for d in PEER_DEPS if d not in KEEP)
absent = sorted(d for d in PEER_DEPS if not os.path.isdir(d))
if a13:
    fail += 1
    print(f"  !! peer-declared dependencies NOT in KEEP: {a13}")
elif absent:
    fail += 1
    print(f"  !! peer-declared dependencies not on disk: {absent}")
else:
    print(f"0 -- PASS  ({len(PEER_DEPS)} peer-declared arms, all in KEEP and on disk)")
stale = [p for p in PEER_STALE if not glob.glob(p + '*')]
print(f"    NOTE: {len(stale)} peer-named arm(s) do not exist and predate this round "
      f"(scripts/pr127_sentinels.py:30): {stale}")

print("\n=== ASSERT 14 (NEW): the successor gate licenses the prod0901 release ===")
# prod0901 is in the removal set ONLY because prod0901b reproduces it on every
# SHARED product.  The evidence is a per-event TSV, checked by ROW COUNT --
# a header-only or short file is non-empty and would pass an existence test
# while proving nothing.  Re-derived by the driver at deletion time too, since
# a proof written before the last event finished is not a proof about the arm.
a14_bad = 0
in_R = sorted(o for o in SUCCESSOR if o in set(R))
if not in_R:
    print("0 -- PASS  (no superseded arm is in the removal set; gate not required)")
else:
    for old, new in sorted(SUCCESSOR.items()):
        if old in set(R) and new not in KEEP:
            print(f"  !! {old} is being removed but its successor {new} is not in KEEP")
            a14_bad += 1
        if old in set(R) and not os.path.isdir(new):
            print(f"  !! successor {new} is not on disk")
            a14_bad += 1
    if not os.path.exists(SUCCESSOR_GATE):
        print(f"  !! {SUCCESSOR_GATE} MISSING -- run scripts/doc89_successor_gate.py first")
        a14_bad += 1
    else:
        rows = ok = 0
        for ln in open(SUCCESSOR_GATE):
            if ln.startswith('#') or ln.startswith('sample\t'):
                continue
            f = ln.rstrip('\n').split('\t')
            if len(f) < 3:
                continue
            rows += 1
            if f[-1] == 'OK':
                ok += 1
        if rows != SUCCESSOR_ROWS or ok != SUCCESSOR_ROWS:
            print(f"  !! successor gate incomplete: {rows} rows, {ok} OK, "
                  f"expected {SUCCESSOR_ROWS}/{SUCCESSOR_ROWS}")
            a14_bad += 1
        else:
            print(f"  successor gate   {ok}/{SUCCESSOR_ROWS} events reproduce on every shared product")
    if a14_bad:
        fail += 1
    else:
        print(f"0 -- PASS  ({len(in_R)} superseded arm(s) released on measured evidence)")


print("\n=== ASSERT 15+16 (NEW): the sentinel suite survives, and so does every ===")
print("===              arm that still passes a sentinel production fails    ===")
# Both checks live in scripts/retire/sentinel_guard_20260901b.py so that this
# assert and the driver's interlock 8 cannot drift apart -- one implementation,
# two callers.  Read that file's docstring for why it evaluates ONE ARM AT A
# TIME (pr127_sentinels.py picks the first arm holding the event, so a combined
# run reports the wrong verdict) and for the two earlier readings it corrects.
sys.path.insert(0, SCR)
import sentinel_guard_20260901b as sg                       # noqa: E402
try:
    ok16 = sg.report(dirs, set(K))
except Exception as exc:                                    # noqa: BLE001
    print(f"  !! sentinel guard could not run: {exc} (failing closed)")
    ok16 = False
if ok16:
    print("0 -- PASS  (sentinel coverage intact; no regression loses its last passing arm)")
else:
    fail += 1


# ---------------------------------------------------------------- emit
with open(os.path.join(SCR, "tierA_20260901b.txt"), "w") as fh:
    fh.write("\n".join(sorted(TA)) + "\n")
json.dump({'A': sorted(TA), 'D': [], 'R': R, 'ARCHIVE': ARCHIVE,
           'SUCCESSOR': SUCCESSOR, 'FROZEN': {},
           'KEEP': sorted(K), 'KEEP_WHY': KEEP_WHY, 'HUB': [], 'POSTBUILD': [],
           'KEEP_PREFIX': KEEP_PREFIX,
           'PROTECTED': sorted(KEEP & PROT_LISTED), 'RELEASED': RELEASED,
           'PR_PROVENANCE': PR_PROVENANCE,
           'ACK_BROKEN_REFS': ACK_BROKEN_REFS,
           'LIVE_MANIFESTS': LIVE_MANIFESTS,
           'PEER_DEPS': PEER_DEPS,
           'planned_at': now,
           'SUCCESSOR_GATE': SUCCESSOR_GATE, 'SUCCESSOR_ROWS': SUCCESSOR_ROWS,
           'script_refs': {k: v for k, v in lit.items()},
           'per': per, 'cites': {},
           'group': {d: group(d) for d in dirs}},
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)
print(f"\nremoval set: {len(R)} dirs -> scripts/retire/tierA_20260901b.txt")
print(f"survivors: {len(K)}")
print(f"state: {STATE}/plan.json")
print("\nOVERALL: " + ("PASS -- all asserts clean" if not fail
                       else f"FAIL -- {fail} assert(s) tripped, do not proceed"))
sys.exit(0 if not fail else 1)
