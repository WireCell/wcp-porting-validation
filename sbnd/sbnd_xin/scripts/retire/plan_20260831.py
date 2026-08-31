#!/usr/bin/env python3
"""Retirement round 2026-08-31 -- release the CLOSED pi0 campaign (docs
pr/129-134) and the superseded prod0825 production pair.  169 G -> ~80 G.

Fork of plan_20260829.py.  All 13 asserts carried unchanged; only this
round's configuration differs.  Two structural differences from 08-29:

  * THE TREE IS QUIET AGAIN.  08-29 ran beside a peer session mid-gate and
    had to refine interlock 2 for it; this round's only live wire-cell
    processes are THIS session's own work-pr134-f086-* campaign, which is
    in KEEP.  PEER_DEPS is therefore empty -- not because the guard was
    dropped, but because there is no peer.  ASSERT 12 still runs and still
    refuses if any live process names a removal-set dir.
  * THE pi0 EPOCH IS NO LONGER AN OPEN ROUND.  08-29 carved it OUT of the
    two fattest families because the owner's next move needed it.  That
    move is now DONE and SHIPPED (docs pr/132-135, chain flipped to SBND
    production 2026-08-31), so the campaign's 361 intermediate arms are
    exactly what this round releases.

Owner scope, verbatim (2026-08-31):

    "we would like to do a clean up for the sbnd_xin directory.  We can
     retire the intermediate debug files work*, and leave the latest
     production, as well as the scan results that we will use as a metric
     to proceed. ... we can go back to say 50 G etc."

Three constraints, each shaping a different part of KEEP:

  1. "leave the latest production" -> work-*-prod0830 (all four samples,
     1000/2000/19/48 events) stays, and the pi0 chain's own production
     point work-pr134-f086-* (the 0.86 EM-scale campaign this round ran)
     becomes the new metric baseline.  prod0825 is RELEASED: prod0830
     supersedes it sample-for-sample, and the 08-29 round explicitly left
     that call to "next round", which is this one.
  2. "the scan results that we will use as a metric to proceed" -> the
     hand-scan LABELS (em_labels/, 249 JSONs, git-tracked since 08-29),
     the vtx105 vertex-label epoch, the display roots the label manifests
     resolve through (em114*, em114c*, pr117r1-onK1-*), and the 50-pi0
     manifest's dump source (pr124r1-onA*).  The next campaign -- the
     upstream charge/tail deficit, doc pr/135 sec 9 -- scores with the
     gamma ledger over these labels, so they are its literal input.
  3. "retire the intermediate debug files work*" -> pr132 (204 arms),
     pr133 (79), pr134 (78 minus the 12 kept), pr129/pr130/pr131 (the
     peer's now-closed rounds, minus the sentinel suite).

WHAT MAKES THE pi0 RELEASE SAFE, AND IT IS NOT AN ASSUMPTION.  The three
arms this round keeps out of pr133/pr134 are the two ENDS of the shipped
A/B plus the new scale point, and the gates already proved the rest are
byte-redundant with them:

    work-pr134-off9-*  ==  work-pr133-flipchk-*   (gate PASS 478/478)
    work-pr134-k24b-*  ==  work-pr134-flip2-*     (gate PASS 478/478)

so releasing off9/k24b loses no byte that flipchk/flip2 do not still hold.
Every other arm in those families is an intermediate iteration whose
numbers live in the round docs and their committed TSVs.

WHAT THIS ROUND DELIBERATELY DOES NOT DO.  It does not reach the owner's
"say 50 G" figure, and the shortfall is named rather than closed by
deleting inputs: work-*-grp0825 (26 G) is the Q/L campaign INPUT that
every future PR re-run reads, and work-*-prod0830 (14 G) is the latest
production the owner asked to keep.  Together with archive/records (14 G,
the record layer of every past round) they are ~54 G of the ~80 G floor.
Going below that means releasing a campaign input or the production
baseline -- an owner decision, offered in the round report, not taken here.
"""
import os, re, json, subprocess, collections, sys, filecmp, csv, glob, time

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.path.join(ROOT, "scripts", "retire")
STATE = os.environ.get("RETIRE_STATE", os.path.join(SCR, "state-20260831"))
os.makedirs(STATE, exist_ok=True)
os.chdir(ROOT)

if os.path.exists(os.path.join(STATE, "removed.tsv")) and not os.environ.get("RETIRE_REPLAN"):
    sys.stderr.write(
        f"REFUSING: {STATE}/removed.tsv exists -- this round has already run (M13).\n"
        f"Fork with a new date/state for a new round; RETIRE_REPLAN=1 to override.\n")
    sys.exit(3)

dirs = sorted(d for d in os.listdir('.')
              if d.startswith('work') and os.path.isdir(d) and not os.path.islink(d))

# ---------------------------------------------------------------- KEEP
# Prefixes whose every arm is kept.  Used ONLY for the open peer round: the
# peer was still creating arms while this ran, so an enumerated list would be
# stale by construction.  Everything else in KEEP is named individually.
KEEP_PREFIX = {
    # This session's own live campaign: the 0.86 EM-scale production arms
    # were still being written while this plan ran, so an enumerated list
    # would be stale by construction (the 08-29 lesson, one session later).
    'work-pr134-f086': 'doc pr/135 -- the 0.86 EM-scale PRODUCTION point; the next round\'s metric baseline',
    'work-sent130':    'the sentinel suite guarding every shipped pr/127-134 flip, incl. negative controls',
}

KEEP_WHY = {
    # --- the latest production, all four samples, 4 -----------------------
    # prod0830 SUPERSEDES prod0825 sample-for-sample (1000/2000/19/48
    # events each, verified on disk this round), which is why prod0825 is
    # released below -- the call 08-29 deferred to "next round".
    'work-nuecc48-prod0830': 'doc 85 -- latest full production (PR), 48 evts',
    'work-ncpi0-prod0830':   'doc 85 -- latest full production, 19 evts',
    'work-mcp1k-prod0830':   'doc 85 -- latest full production, 1000 evts',
    'work-mcp2k-prod0830':   'doc 85 -- latest full production, 2000 evts',
    # --- prod0825: NOT releasable after all, 4 ---------------------------
    # This round OPENED intending to release it (08-29 deferred the call to
    # "next round") on the ground that prod0830 supersedes it sample-for-
    # sample.  ASSERT 11 refused: the BASE hand-scan display manifests --
    # em114-manifest.tsv (98 rows) and em114c-manifest-agent5.tsv (5) --
    # resolve their dump paths into prod0825, not prod0830.  prod0830
    # supersedes it as the PRODUCTION baseline; it does not replace it as
    # the dump source the hand-scan metric was built against, and the owner
    # asked to keep exactly that metric.  Superseding it properly means
    # rebuilding those manifests against prod0830 first -- additive work,
    # not a deletion.
    'work-nuecc48-prod0825': 'em114-manifest.tsv dump source (ASSERT 11); doc 81 stage B',
    'work-ncpi0-prod0825':   'em114-manifest.tsv dump source (ASSERT 11)',
    'work-mcp1k-prod0825':   'em114-manifest.tsv + em114c-manifest-agent5.tsv dump source (ASSERT 11)',
    'work-mcp2k-prod0825':   'em114-manifest.tsv + em114c-manifest-agent5.tsv dump source (ASSERT 11)',
    # --- campaign INPUT: the Q/L roots every re-run reads, 4 --------------
    # NOT a record: the next campaign (doc pr/135 sec 9, the upstream
    # charge/tail deficit) re-runs the PR chain many times and these are
    # what it runs FROM.  Releasing them would mean redoing imaging + Q/L.
    'work-nuecc48-grp0825':  'doc 81 stage A (imaging + Q/L), 48 evts -- campaign input',
    'work-ncpi0-grp0825':    'doc 81 stage A, 19 evts -- campaign input',
    'work-mcp1k-grp0825':    'doc 81 stage A, 1000 evts -- campaign input',
    'work-mcp2k-grp0825':    'doc 81 stage A, 2000 evts -- campaign input',
    # --- the two SIM samples and their hubs: only copies, 10 --------------
    'work-img-r1qlmc':       'imaging hub; ONLY copy of this sim sample (no grp0825 counterpart)',
    'work-img-r2mc':         'imaging hub; ONLY copy of this sim sample (no grp0825 counterpart)',
    'work-r1qlmc-cb0805':    'SIM Q/L hub; ASSERT 8 input to work-r1qlmc-prod0813',
    'work-r2mc-cb0805':      'SIM Q/L hub; ASSERT 8 input to work-r2mc-prod0813',
    'work-r1qlmc-prod0813':  'only PR product for this sim sample; PROTECTED.txt',
    'work-r2mc-prod0813':    'only PR product for this sim sample; PROTECTED.txt',
    'work-r1qlmc-vfcbr3on':  'post-flip Q/L for r1qlmc; input to work-vfr1qlmc-cbr3on',
    'work-r2mc-vfcbr3on':    'post-flip Q/L for r2mc; input to work-vfr2mc-cbr3on',
    'work-vfr1qlmc-cbr3on':  'latest PR out-root for r1qlmc',
    'work-vfr2mc-cbr3on':    'latest PR out-root for r2mc',
    # --- the vertex-label epoch: the movers METRIC, 4 ---------------------
    # pr90_movers.py --tags vtx105 is the ADVERSE gate every round runs;
    # these dumps are what the vtx105 labels were scanned against.
    'work-vtx105-base-nuecc48': 'calib-pr dumps behind the vtxscan-vtx105-* label epoch, 47 evts',
    'work-vtx105-base-ncpi0':   'calib-pr dumps behind the current label epoch, 19 evts',
    'work-vtx105-base-mcp1k':   'calib-pr dumps behind the current label epoch, 407 evts',
    'work-vtx105-base-mcp2k':   'calib-pr dumps behind the current label epoch, 581 evts',
    # --- git-tracked / not reproducible, 4 --------------------------------
    'work-tfix388-r9':       'doc pr/28 sec.15.9 -- NOT reproducible from any surviving input',
    'work-stmcamp-d66new':   'git-tracked nusel_labels/ hand-scan state (M13)',
    'work-nuecc48-prsmoke2': '3 git-tracked runner scripts',
    'work-probe178410a':     'the ONLY proof mcp2k evt 178410 SIGSEGV is non-deterministic; 17 MB',
    # --- the hand-scan metric layer the NEXT round reads, 12 --------------
    # Owner: "the scan results that we will use as a metric to proceed".
    # pr126-pi0-manifest.tsv (50 hand pi0) resolves into pr124r1-onA*, and
    # the whole pr125r1 family carries the doc pr/126 sec 4f census AND the
    # sentinel registry's worked FAILING case (pr127_sentinels.py:31) --
    # a registry with no failing case cannot be trusted.
    'work-pr124r1-onA98-mcp1k':    'dump source, 50 hand-paired pi0 (pr126-pi0-manifest.tsv) + 98-set census',
    'work-pr124r1-onA98-mcp2k':    'dump source, pi0 manifest + 98-set census',
    'work-pr124r1-onA98-ncpi0':    'dump source, pi0 manifest + 98-set census',
    'work-pr124r1-onA98-nuecc48':  'dump source, pi0 manifest + 98-set census',
    'work-pr124r1-onA141v2-mcp1k': 'the 141-set half of the pi0 manifest',
    'work-pr124r1-onA141v2-mcp2k': 'the 141-set half of the pi0 manifest',
    'work-pr125r1-flipchk98-mcp1k':   'doc pr/126 sec 4f census at the pr/125 production point',
    'work-pr125r1-flipchk98-mcp2k':   'doc pr/126 sec 4f census',
    'work-pr125r1-flipchk98-ncpi0':   'doc pr/126 sec 4f census',
    'work-pr125r1-flipchk98-nuecc48': 'doc pr/126 sec 4f census',
    'work-pr125r1-flipchk141-mcp1k':  'the 141-set half of the sec 4f census',
    'work-pr125r1-flipchk141-mcp2k':  'the 141-set half of the sec 4f census',
    # --- the sentinel registry's NEGATIVE control, 6 ----------------------
    'work-pr125r1-flipK598-mcp1k':   'scripts/pr127_sentinels.py:31 -- the worked FAILING sentinel case (137238)',
    'work-pr125r1-flipK598-mcp2k':   'the negative control; a registry with no failing case cannot be trusted',
    'work-pr125r1-flipK598-ncpi0':   'the negative control',
    'work-pr125r1-flipK598-nuecc48': 'the negative control',
    'work-pr125r1-flipK5141-mcp1k':  '141-set half of the negative control',
    'work-pr125r1-flipK5141-mcp2k':  '141-set half of the negative control',
    # --- hand-scan DISPLAY defaults + probe layer, 13 ---------------------
    # Hardcoded default arm names in the tools that BUILD the hand-scan
    # manifests; a live default, not a closed-round record.
    'work-em114c-prodnow-mcp1k': 'hardcoded default arm, em_display/prep_pr121.py:15 (141-set display)',
    'work-em114c-prodnow-mcp2k': 'hardcoded default arm, em_display/prep_pr121.py:15',
    'work-pr117r1-onK1-mcp1k':   'hardcoded default arm, em_display/prep_pr117.py:17,28 (98-set display)',
    'work-pr117r1-onK1-mcp2k':   'completes the 98-set display baseline; em117-117onK1-manifest.tsv 98/98',
    'work-pr117r1-onK1-ncpi0':   'completes the 98-set display baseline',
    'work-pr117r1-onK1-nuecc48': 'completes the 98-set display baseline',
    'work-em114-mcp1k':   'probe-root fallback for the original 94, em_display/prep_em_scan.py:275',
    'work-em114-mcp2k':   'probe-root fallback, same chain',
    'work-em114-ncpi0':   'probe-root fallback, same chain',
    'work-em114-nuecc48': 'probe-root fallback, same chain',
    'work-em114-probe3':  'doc pr/114 sec 3 probe-equivalence proof (3 evts vs prod0825, byte-equal)',
    'work-em114c-mcp1k':  '--parse-probes source for em114c-manifest.tsv (141-set); run_em114c_probe.sh:72',
    'work-em114c-mcp2k':  '--parse-probes source for the 141-set display manifest',
    # --- doc pr/130's non-reproducible census source, 6 ------------------
    # The rest of pr129*/pr130* is RELEASED this round (their PROTECTED.txt
    # block states the condition: "these go when doc pr/130 is reported and
    # closed, in the same pass that releases the pr128r1-on gate label" --
    # both now true).  These six are the exception their own line argues
    # for: "the 239-event blast-radius result is not reproducible without
    # them".  That is a non-reproducibility claim, not a "the doc already
    # has the numbers" claim, and PROTECTED.txt's rule keeps it.  0.8 G.
    'work-pr130r1-probe98-mcp1k':   'dump source for both pr130q manifests + pr130_launder_scan.py census',
    'work-pr130r1-probe98-mcp2k':   'dump source for both pr130q manifests',
    'work-pr130r1-probe98-ncpi0':   'dump source for both pr130q manifests',
    'work-pr130r1-probe98-nuecc48': 'dump source for both pr130q manifests',
    'work-pr130r1-probe141-mcp1k':  '141-set half of the pr130q dump source',
    'work-pr130r1-probe141-mcp2k':  '141-set half of the pr130q dump source',
    # --- the two ENDS of the SHIPPED pi0 A/B, 8 ---------------------------
    # The owner scanned this A/B on Bee and flipped production on it.  The
    # gates make the rest of pr133/pr134 redundant with these two:
    #   work-pr134-off9-* == work-pr133-flipchk-*  (PASS 478/478)
    #   work-pr134-k24b-* == work-pr134-flip2-*    (PASS 478/478)
    # so off9/k24b are released while both ENDS stay byte-for-byte.
    'work-pr133-flipchk-mcp1k':   'the OFF end of the shipped pi0 A/B (pre-chain production point)',
    'work-pr133-flipchk-mcp2k':   'the OFF end of the shipped A/B',
    'work-pr133-flipchk-ncpi0':   'the OFF end of the shipped A/B',
    'work-pr133-flipchk-nuecc48': 'the OFF end of the shipped A/B',
    'work-pr134-flip2-mcp1k':   'the ON end: the 0.84 production point doc pr/134 sec 14 + pr/135 sec 8 were measured on',
    'work-pr134-flip2-mcp2k':   'the ON end of the shipped A/B',
    'work-pr134-flip2-ncpi0':   'the ON end of the shipped A/B',
    'work-pr134-flip2-nuecc48': 'the ON end of the shipped A/B',
}

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
    'work-nuecc48-prod0825': 'work-nuecc48-grp0825',
    'work-ncpi0-prod0825':   'work-ncpi0-grp0825',
    'work-mcp1k-prod0825':   'work-mcp1k-grp0825',
    'work-mcp2k-prod0825':   'work-mcp2k-grp0825',
    'work-vfr1qlmc-cbr3on': 'work-r1qlmc-vfcbr3on',
    'work-vfr2mc-cbr3on':   'work-r2mc-vfcbr3on',
}
SUCCESSOR = {}          # none this round: no Q/L root retires

# ---- ASSERT 11: manifests that MUST still resolve, with their row counts --
# Row counts, never `[ -s ]` or mere existence (the 08-25b lesson).  These are
# the manifests the pi0 round and the peer's scorers read; every dump path in
# each must land in a KEEP arm after the round.
LIVE_MANIFESTS = {
    'em_display/pr126-pi0-manifest.tsv':          50,   # the 50 hand-paired pi0
    'em_display/em117-117onK1-manifest.tsv':      98,   # the 98-set display baseline
    'em_display/em114c-manifest.tsv':            141,   # base 141-set display
    'em_display/em114-manifest.tsv':              98,   # base 98-set display
    'em_display/em116confirm-manifest.tsv':       30,
    'em_display/em114c-manifest-agent5.tsv':       5,
    'em_display/em117-125flipchk98-manifest.tsv': 98,   # doc pr/126 sec 4f census
    'em_display/em114c-125flipchk141-manifest.tsv': 141,
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
PEER_DEPS = []   # no peer session this round: the tree is quiet again (see
                 # the header).  ASSERT 13 degenerates to a no-op, but ASSERT
                 # 12 still refuses if any live process names a removal dir.

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
print("=== RETIREMENT ROUND 2026-08-31 (the CLOSED pi0 campaign; production + metric stay) ===")
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
    print(f"    ... and {len(R)-25} more (full list: tierA_20260829.txt)")

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
    print("    (the tree's only live label dir inside work-* is")
    print("     work-stmcamp-d66new/nusel_labels, in KEEP; em_labels/,")
    print("     vertex_labels/ and overclustering_labels/ are outside work-*)")
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
CAMPAIGN_IMG = {'work-img-r1qlmc': 10, 'work-img-r2mc': 13}
CAMPAIGN_OUT = {
    'work-r1qlmc-vfcbr3on': ('ql_evt',  10), 'work-r2mc-vfcbr3on': ('ql_evt',  13),
    'work-nuecc48-grp0825': ('ql_evt',   48), 'work-ncpi0-grp0825': ('ql_evt',   19),
    'work-mcp1k-grp0825':   ('ql_evt', 1000), 'work-mcp2k-grp0825': ('ql_evt', 2000),
    'work-nuecc48-prod0825': ('pr_evt',   48), 'work-ncpi0-prod0825': ('pr_evt',   19),
    'work-mcp1k-prod0825':   ('pr_evt', 1000), 'work-mcp2k-prod0825': ('pr_evt', 2000),
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
# (retire_20260831.sh interlock 6) -- plan-time evidence expires.
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

# ---------------------------------------------------------------- emit
with open(os.path.join(SCR, "tierA_20260831.txt"), "w") as fh:
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
           'script_refs': {k: v for k, v in lit.items()},
           'per': per, 'cites': {},
           'group': {d: group(d) for d in dirs}},
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)
print(f"\nremoval set: {len(R)} dirs -> scripts/retire/tierA_20260831.txt")
print(f"survivors: {len(K)}")
print(f"state: {STATE}/plan.json")
print("\nOVERALL: " + ("PASS -- all asserts clean" if not fail
                       else f"FAIL -- {fail} assert(s) tripped, do not proceed"))
sys.exit(0 if not fail else 1)
