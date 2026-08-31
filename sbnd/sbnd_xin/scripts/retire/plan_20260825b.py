#!/usr/bin/env python3
"""Retirement round 2026-08-25b -- release the SUPERSEDED HALF of the stage-A
gate and the pre-flip PR baseline.  144 G -> ~104 G, 12 arms, ~40 G.

Fork of plan_20260825.py (same day, second pass -- precedent
retire_20260819b.sh).  Asserts 1-10 carried; TWO new asserts, 11 and 12, and
one carried assert (8) gains an explicit SUCCESSOR rule.  KEEP shrinks 38 -> 26.

Owner scope, asked and given directly:

    "I assume we can safely retire [work-img-{4 samples}, work-*-ql0819,
     work-*-prod0823], and recover the disk"

which is the three-row candidate table this round was planned from.  Each row
has a different justification and a different cost, so they are NOT one
decision:

  1. work-img-{nuecc48,ncpi0,mcp1k,mcp2k} (~19 G).  doc 81 sec 7 proved these
     byte-identical to the imaging half of work-<s>-grp0825, 24536/24536.
     Pure duplication as long as grp0825 stays -- and grp0825 is PROTECTED.
     work-img-{r1qlmc,r2mc} are NOT duplicates (no grp0825 arm exists for
     either sim sample) and stay in KEEP.
  2. work-<s>-ql0819 (~10.3 G).  The Q/L half of the same gate, same proof.
  3. work-<s>-prod0823 (~10.2 G).  Explicitly PRE-flip (doc 81 sec 4),
     superseded by prod0825.  PROTECTED.txt still carries an active line for
     it; this round writes the RELEASED line.

WHAT MAKES THIS ROUND DIFFERENT FROM EVERY EARLIER ONE.  Rows 1 and 2 are the
two halves of ONE gate reference, and this round deletes BOTH.  Doc 81 sec 7's
"PASS 24536/24536" would stop being re-checkable the moment that happens --
the same failure interlock 4 of the 08-25 round was invented to prevent, but
one layer up: there the reference was a PR arm, here it is the whole stage-A
reference side.  So:

  * scripts/retire/hash_manifest_stagea_20260825b.py freezes it -- all four
    samples, all 8 products per event, member-content rollups, into
    state-20260825b/hashes/stagea-<s>.tsv.  ASSERT 11 checks the ROW COUNT
    (152 + 384 + 8000 + 16000 = 24536), never mere existence: a
    header-only .tsv passes `[ -s ]` and would be a vacuous PASS with 29 G
    deleted behind it.
  * hash_manifest_pr_20260825b.py freezes prod0823 the same way.  Note in the
    doc, and repeated here so nobody mis-reads it later: that one preserves NO
    live gate.  prod0825 is at a different operating point, so there is no
    byte-identity claim between them to keep checkable.  It buys a future
    revert-reproduction check.  Insurance, not gate preservation.

A tool cannot be shared between the two shapes and the failure would be
SILENT.  hash_manifest_20260825.py matches `pr_evt(\\d+)$`; stage-A arms are
`evt<N>/` + `ql_evt<N>/`, so it would walk zero events and write a header-only
file.  That is why there are two freeze scripts, not one with a flag.

ASSERT 8 GAINS A SUCCESSOR RULE.  work-vtx105-base-* is PROTECTED and stays,
and its own logs name work-<s>-ql0819 as the Q/L root it read -- a root this
round deletes.  Under the carried assert that is an automatic FAIL, and
suppressing it by hand is exactly the kind of silent weakening this file
exists to prevent.  Instead SUCCESSOR maps each retiring root to
work-<s>-grp0825 and the assert accepts the substitution ONLY when the
successor is in KEEP *and* that sample's frozen stage-A manifest is complete.
The substitution is sound because the products are proven identical, not
because they are similar.

ASSERT 12 (new).  Two live tools default to an arm this round removes:
scripts/multi/repro_ql_nondet.sh:55 (doc 82's reproducer, closed TODAY) and
scripts/multi/ql_legacy_gate.sh:21.  ASSERT 10 would flag them as unacked
literals, which is the wrong disposition -- they must be REPOINTED at
grp0825, not acknowledged as broken.  ASSERT 12 checks the repoint actually
happened before the round may run.  This is the 08-23 lesson made mechanical:
ASSERT 10 catches hardcoded paths, but only a hand-read of the OPEN doc's arm
table catches the rest, so the hand-read result gets encoded as an assert.

COSTS, stated before the round rather than discovered after it:
  * doc pr/104-pr/111's A/B references against the pr/104 production epoch
    become fully text-only -- prod0823 is the last on-disk carrier of it.
  * sp-frames.tar.bz2 -- NOT a cost, checked rather than assumed.  This was
    written up as a loss first and the tar was then read: HEAVY has no pattern
    matching sp-frames, so heavy_class() returns None and all 2067 files
    (mcp2k 2000, nuecc48 48, ncpi0 19) are archived verbatim into
    stagea-refside-20260825b/imaging-hubs/.  They are preserved, not dropped.
    That is also most of the record tree's 2.9 G, so the NET recovery is ~36 G,
    not the 39 GiB the driver reports as removed.
  * work-{nuecc48,ncpi0}-ql0819's ql_evt*/calib-evt*.json Q/L dumps (146 + 53
    MB) go.  grp0825 does not carry them.  Same precedent: mcp1k/mcp2k were
    thinned of theirs in the 08-11 round and have 0 today.
  * the five closed-round arm scripts (pr107/108/109/112) stop resolving.
    ACKed below, not repointed: they are records of how a finished round was
    run, and rewriting them would misrepresent it.  grp0825 is the drop-in
    successor for any FUTURE re-run -- its ql_evt<N>/ layout carries every
    product ql0819's does except calib-evt/wct_ql logs.

This round is HYGIENE, not pressure relief: /nfs/data/1 is at 75% with 881 G
free before it runs.  Nothing was at risk; ~40 G is simply no longer earning
its keep.

--- inherited header, 2026-08-23 round ---

Fork of plan_20260819b.py.  tier() UNCHANGED (KEEP-only, two classes).
Asserts 1-9 carried; ONE new assert (10) added -- see below.  KEEP shrinks
36 -> 38 names but the CONTENT turns over almost completely: the pr/95
prod0819 PR baseline is released and the pr/104 production epoch takes its
place.

Why this round exists.  The owner asked (2026-08-23) to "go back to a minimal
state with the latest production available (QLMatching, and PR)".  Since the
08-19 pass-2 round left 36 dirs / 54 G, docs pr/98-104 (four production flips)
and pr/105-111 (vertex-strategy, dQ/dx, exclusion and DL-vertex studies)
regrew the tree to 455 work-* dirs / 188 G, with another 15 G outside work-*.
Every one of those arms is a leg of an A/B whose verdict is already text in
its doc.

THIS ROUND RUNS *BEFORE* A CAMPAIGN, like 08-19 pass 1.  Owner, asked directly
about the one gap this exposes (mcp2k has no full-coverage PR product at the
pr/104 production epoch -- work-pr104-on4-mcp2k covers 15 of 2000 events):

    "we can drop this, and redo the production for the samples, so we keep
     the latest PR production for all samples"

So the four work-*-prod0819 PR arms are RELEASED and a fresh full-coverage PR
campaign at current HEAD replaces them.  KEEP must therefore stay closed
FORWARD over that campaign's input set -- the six imaging hubs and the four
-ql0819 Q/L roots -- which is ASSERT 9, carried from 08-19 pass 1 and
re-pointed at this round's arms and their REAL event counts (see the note on
CAMPAIGN_OUT below; a stale hardcode there is how a short arm passes silently).

KEEP is 38 names in seven groups:

  1. Campaign INPUT (8): the six work-img-* imaging hubs, plus the two SIM
     Q/L hubs work-{r1qlmc,r2mc}-cb0805.  The three DATA cb0805 hubs are
     RELEASED this round (owner) -- superseded by -ql0819.
  2. Latest production QLMatching (4): work-{nuecc48,ncpi0,mcp1k,mcp2k}-ql0819,
     48/19/1000/2000 events, bare production at toolkit fd6a116d.  These are
     NOT released with the prod0819 PR arms -- they are the re-run's input.
  3. Latest production PR (8): work-pr104-on4-* (the product) and
     work-pr104-flipchk-* (the only on-disk proof the shipped post-flip
     jsonnet reproduces it with no env).  Superseded the moment the re-run
     lands; release them in that follow-up pass, not before.
  4. The current vertex-label epoch (4): work-vtx105-base-*.  1756 of
     vertex_labels/vtxscan-vtx105-*'s `source` entries resolve into these
     four arms and every older epoch's dumps (harv3, prod0813, vtx100) is
     already gone.  Owner chose to keep this one.
  5. doc pr/111's live inputs (4): work-vtx106-harv-{base,nofitx}-nuecc48 and
     work-vtx106-cne-{on,off}-nuecc48.
     NOT a citation inference -- five scripts/pr111_*.py files hardcode these
     absolute paths as the fit_exclusion ON/OFF pair.  pr/111 is the OPEN
     round.
  6. The two SIM samples (6): work-{r1qlmc,r2mc}-{prod0813,vfcbr3on} and
     work-vf{r1qlmc,r2mc}-cbr3on.  Neither sim sample is in the data campaign,
     so these remain their latest and only products.  290 MB.
  7. Git-tracked / not reproducible (4): work-tfix388-r9, work-stmcamp-d66new,
     work-nuecc48-prsmoke2, work-probe178410a.

ASSERT 10 (new).  Two prior rounds discovered *after* the fact that a script
hardcoded an arm they had just deleted (vtx_rules/baselines.py's
deployed_dump_path(), scripts/analysis/pr57/oc56_truth.py's DEFAULT_ARMS).
The safety net has always been a citation check over docs/, which does not
see a path literal in a .py.  ASSERT 10 greps scripts/, vtx_rules/,
dl_vtx_training/ and the top-level *.py/*.sh for every work-* literal, and
REFUSES if any name in the removal set is not in ACK_BROKEN_REFS below.  The
point is not to save the arm -- it is that every broken reference is a cost
written down in the round's doc section before the round runs, never a
surprise weeks later.

NOT retired and NOT thinned: dl_vtx_training (67 M, 0 *.pth), vertex_labels/,
overclustering_labels/, archive/ (the record layer, M13), input_files*/ (the
SP sources ASSERT 1 checks), bee/ (backs uploaded doc links).  Those four are
~15 G and are the floor this round cannot go below.

Writes scripts/retire/tierA_20260825.txt and state-20260825/plan.json.
Read-only w.r.t. work-*.
"""
import os, re, json, subprocess, collections, sys, filecmp

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.path.join(ROOT, "scripts", "retire")
STATE = os.environ.get("RETIRE_STATE", os.path.join(SCR, "state-20260825b"))
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
KEEP_WHY = {
    # --- campaign INPUT, 4 ------------------------------------------------
    # The four DATA imaging hubs are RELEASED this round: doc 81 sec 7 proved
    # each byte-identical to the imaging half of its grp0825 arm, which stays.
    # The two SIM hubs are NOT duplicates -- no grp0825 arm exists for either
    # sim sample -- and are the only copy, so they stay.  Spelled out because
    # a glob over 'work-img-*' would sweep them and interlock 3 only catches
    # that if KEEP names them explicitly.
    'work-img-r1qlmc':       'imaging hub; ONLY copy of this sim sample (no grp0825 counterpart)',
    'work-img-r2mc':         'imaging hub; ONLY copy of this sim sample (no grp0825 counterpart)',
    'work-r1qlmc-cb0805':    'SIM Q/L hub; ASSERT 8 input to work-r1qlmc-prod0813',
    'work-r2mc-cb0805':      'SIM Q/L hub; ASSERT 8 input to work-r2mc-prod0813',
    # --- doc 81 PRODUCTS: the new baseline for MCS and PR, 8 --------------
    'work-nuecc48-grp0825':  'doc 81 stage A (imaging + Q/L), 48 evts, per-event layout',
    'work-ncpi0-grp0825':    'doc 81 stage A, 19 evts',
    'work-mcp1k-grp0825':    'doc 81 stage A, 1000 evts',
    'work-mcp2k-grp0825':    'doc 81 stage A, 2000 evts',
    'work-nuecc48-prod0825': 'doc 81 stage B (PR) at the CURRENT operating point, 48 evts',
    'work-ncpi0-prod0825':   'doc 81 stage B, 19 evts',
    'work-mcp1k-prod0825':   'doc 81 stage B, 1000 evts',
    'work-mcp2k-prod0825':   'doc 81 stage B, 2000 evts',
    # --- the PRE-flip PR record: RELEASED this round -----------------------
    # The 08-25 round kept work-<s>-prod0823 as "the only on-disk product at
    # the pre-flip config".  That is still true and it is no longer a reason
    # to keep it: its remaining job -- revert-proof reference -- was already
    # exercised and the result is text in doc 81 sec 4.  Owner released it
    # explicitly.  hash_manifest_pr_20260825b.py freezes it first (ASSERT 11),
    # which is insurance for a future revert-reproduction, NOT preservation of
    # a live gate: prod0825 is at a different operating point, so there was
    # never a byte-identity claim between the two.
    # --- the current vertex-label epoch, 4 --------------------------------
    'work-vtx105-base-nuecc48': 'calib-pr dumps behind the vtxscan-vtx105-* label epoch, 47 evts',
    'work-vtx105-base-ncpi0':   'calib-pr dumps behind the current label epoch, 19 evts',
    'work-vtx105-base-mcp1k':   'calib-pr dumps behind the current label epoch, 407 evts',
    'work-vtx105-base-mcp2k':   'calib-pr dumps behind the current label epoch, 581 evts',
    # --- the two SIM samples: not in the data campaign, 6 -----------------
    'work-r1qlmc-prod0813':  'only PR product for this sim sample; PROTECTED.txt',
    'work-r2mc-prod0813':    'only PR product for this sim sample; PROTECTED.txt',
    'work-r1qlmc-vfcbr3on':  'post-flip Q/L for r1qlmc; input to work-vfr1qlmc-cbr3on',
    'work-r2mc-vfcbr3on':    'post-flip Q/L for r2mc; input to work-vfr2mc-cbr3on',
    'work-vfr1qlmc-cbr3on':  'latest PR out-root for r1qlmc',
    'work-vfr2mc-cbr3on':    'latest PR out-root for r2mc',
    # --- git-tracked / not reproducible, 4 --------------------------------
    'work-tfix388-r9':       'doc pr/28 sec.15.9 -- NOT reproducible from any surviving input',
    'work-stmcamp-d66new':   'git-tracked nusel_labels/ hand-scan state (M13)',
    'work-nuecc48-prsmoke2': '3 git-tracked runner scripts',
    'work-probe178410a':     'the ONLY proof mcp2k evt 178410 SIGSEGV is non-deterministic; 6.7 MB',
}
KEEP = set(KEEP_WHY)

# Provenance edges checked by ASSERT 8: PR arm -> the Q/L root its own log
# says it read.  RE-DERIVED from each arm's log on every run, so a stale
# hardcode cannot pass silently.  Re-pointed this round: the prod0819 edges
# are gone with their arms; the pr/104, vtx105 and vtx106 edges are new.
PR_PROVENANCE = {
    'work-nuecc48-prod0825': 'work-nuecc48-grp0825',
    'work-ncpi0-prod0825':   'work-ncpi0-grp0825',
    'work-mcp1k-prod0825':   'work-mcp1k-grp0825',
    'work-mcp2k-prod0825':   'work-mcp2k-grp0825',
    # the four prod0823 edges are gone with their arms this round.
    'work-vtx105-base-nuecc48': 'work-nuecc48-ql0819',
    'work-vtx105-base-ncpi0':   'work-ncpi0-ql0819',
    'work-vtx105-base-mcp1k':   'work-mcp1k-ql0819',
    'work-vtx105-base-mcp2k':   'work-mcp2k-ql0819',
    'work-vfr1qlmc-cbr3on': 'work-r1qlmc-vfcbr3on',
    'work-vfr2mc-cbr3on':   'work-r2mc-vfcbr3on',
}

# ---- ASSERT 8's SUCCESSOR rule (new this round) ---------------------------
# work-vtx105-base-* is PROTECTED and stays; its own per-event logs name
# work-<s>-ql0819, a root this round deletes.  Under the carried assert that is
# an automatic FAIL, and hand-suppressing it is precisely the silent weakening
# these asserts exist to prevent.  So the substitution is declared, and it is
# accepted ONLY when both halves of its justification hold at run time:
#   (a) the successor is itself in KEEP, and
#   (b) that sample's frozen stage-A manifest is COMPLETE (ASSERT 11's count).
# (b) is what makes this sound rather than merely convenient: doc 81 sec 7
# proved the Q/L products byte-identical member-for-member, and the frozen
# manifest is that proof surviving the deletion.  A successor without a
# complete manifest is refused.
SUCCESSOR = {
    'work-nuecc48-ql0819': 'work-nuecc48-grp0825',
    'work-ncpi0-ql0819':   'work-ncpi0-grp0825',
    'work-mcp1k-ql0819':   'work-mcp1k-grp0825',
    'work-mcp2k-ql0819':   'work-mcp2k-grp0825',
}

# ---- ASSERT 11: the frozen manifests, and their REQUIRED row counts -------
# Row counts, never `[ -s ]`.  stage A is 8 products/event (4 imaging npz +
# 3 mabc + 1 pctree) and the four samples sum to doc 81 sec 7's own 24536.
# prod0823 is 3 products/event (mabc-pr.zip, pctree-pr, tracking-pr.root).
FROZEN = {
    'stagea-nuecc48.tsv':        48 * 8,
    'stagea-ncpi0.tsv':          19 * 8,
    'stagea-mcp1k.tsv':        1000 * 8,
    'stagea-mcp2k.tsv':        2000 * 8,
    'work-nuecc48-prod0823.tsv':  48 * 3,
    'work-ncpi0-prod0823.tsv':    19 * 3,
    'work-mcp1k-prod0823.tsv':  1000 * 3,
    'work-mcp2k-prod0823.tsv':  2000 * 3,
}
STAGEA_TOTAL = 24536

# ---- ASSERT 12: live tools that must be REPOINTED, not acknowledged -------
# Each entry: file -> (a literal that must be GONE, a literal that must be
# PRESENT).  These two default to a retiring arm; the fix is to repoint them at
# grp0825 (proven identical), not to record them as broken.  doc 82's
# reproducer closed TODAY and is the freshest thing this round could break.
REPOINTED = {
    'scripts/multi/repro_ql_nondet.sh': ('work-${_s}-ql0819', 'work-${_s}-grp0825'),
    'scripts/multi/ql_legacy_gate.sh':  ('work-img-mcp1k', 'work-mcp1k-grp0825'),
}

# ---- ASSERT 10: script literals into the removal set, acknowledged --------
# Every name here is a hardcoded path in a .py/.sh that stops resolving after
# this round.  Each is a stated cost in docs/work-tags.md's 2026-08-23
# section.  A removal-set name NOT listed here refuses the round.
ACK_BROKEN_REFS = {
    # --- 08-25b round: the stage-A reference side and the pre-flip PR arms --
    # DISPOSITION NOTE.  Everything below is ACKed, never repointed, on one
    # rule: a script that RECORDS how a finished round was run keeps naming the
    # arm that round actually read.  Repointing those at grp0825 would make
    # them claim a provenance the round did not have.  The two LIVE tools get
    # the opposite treatment -- see REPOINTED / ASSERT 12.
    # For any FUTURE re-run, work-<s>-grp0825 is the drop-in successor: its
    # ql_evt<N>/ layout carries every product ql0819's does except
    # calib-evt*.json and wct_ql_evt*.log.
    'work-img-mcp1k':
        'scripts/multi/ql_legacy_gate.sh IMGBASE default -- REPOINTED at '
        'work-mcp1k-grp0825 this round (ASSERT 12), so the literal that remains is '
        'the doc-string/comment mention only',
    'work-img-mcp2k':
        'comment-level mentions only; the imaging npz are byte-identical inside '
        'work-mcp2k-grp0825 (doc 81 sec 7, frozen in stagea-mcp2k.tsv)',
    'work-img-ncpi0':
        'comment-level mentions only; imaging npz live on in work-ncpi0-grp0825',
    'work-img-nuecc48':
        'comment-level mentions only; imaging npz live on in work-nuecc48-grp0825',
    'work-nuecc48-ql0819':
        'scripts/pr107_arms.sh, pr108_testA.sh, pr109_sbnd_arms.sh, pr112_arms.sh, '
        'pr112_dual_arms.sh -- all CLOSED rounds, kept as the record of what they '
        'read.  Q/L products frozen in stagea-nuecc48.tsv and byte-identical inside '
        'work-nuecc48-grp0825',
    'work-ncpi0-ql0819':
        'scripts/pr107_arms.sh + pr112 arm builders (closed rounds); products '
        'frozen in stagea-ncpi0.tsv, live in work-ncpi0-grp0825',
    'work-mcp1k-ql0819':
        'scripts/pr107_arms.sh + pr112 arm builders and scripts/manifests/numu50.txt '
        "(a comment naming the sample's arm); frozen in stagea-mcp1k.tsv, live in "
        'work-mcp1k-grp0825',
    'work-mcp2k-ql0819':
        'scripts/manifests/numu50.txt comment + pr112 arm builders; frozen in '
        'stagea-mcp2k.tsv, live in work-mcp2k-grp0825',
    'work-nuecc48-prod0823':
        'PRE-flip PR baseline, released by owner.  COST: doc pr/104-pr/111 A/B '
        'references against the pr/104 epoch become text-only -- this was its last '
        'on-disk carrier.  Frozen in work-nuecc48-prod0823.tsv',
    'work-ncpi0-prod0823':   'PRE-flip PR baseline; frozen in work-ncpi0-prod0823.tsv',
    'work-mcp1k-prod0823':   'PRE-flip PR baseline; frozen in work-mcp1k-prod0823.tsv',
    'work-mcp2k-prod0823':   'PRE-flip PR baseline; frozen in work-mcp2k-prod0823.tsv',
    # --- doc 81 round: the pr/112 and vtx106 families are released ---------
    'work-pr112i-off-nuecc48':
        'scripts/pr112_dual_eval.py default arm; doc pr/112 is CLOSED (snapD2 shipped '
        'SBND production ON) and its numbers are tables in that doc',
    'work-pr112i-snapD2-nuecc48':
        'scripts/pr112_dual_eval.py default arm.  NOTE: the snapD2 family is what doc 81 '
        'gated the new prod0825 arms against (96/96 + 48/48, 38/38 + 19/19); releasing it '
        'makes that PASS text-only, which the owner accepted explicitly ("release all of '
        'pr112 once the new arms pass").  work-<s>-prod0825 is the reference from here on',
    'work-vtx106-harv-base-nuecc48':
        'hardcoded by scripts/pr112_pair.py and the five pr111_*.py; doc pr/111 is CLOSED '
        '(conclusion: keep fit_exclusion ON) and its numbers are text in that doc',
    'work-vtx106-harv-nofitx-nuecc48':
        'the OFF leg of the same pair, same scripts, same closed round',
    # doc pr/95 baseline, released by the owner in favour of the re-run
    'work-nuecc48-prod0819': 'released; docs/pr/95 + pr/98-104 A/B references become text-only',
    'work-ncpi0-prod0819':   'released with the baseline',
    'work-mcp1k-prod0819':   'released with the baseline',
    'work-mcp2k-prod0819':   'released with the baseline; mcp2k has no 2000-evt PR product until the re-run lands',
    # the pr/96 family: its subject (the prod0819 arms) is gone
    'work-pr96gate-mcp2k':   'mixed-binary equivalence proof for prod0819; subject released',
    'work-pr96gate-nuedisp': 'same proof, pr_display leg',
    'work-pr96gate-disp':    'ambiguous-ownership arm kept at 08-19; the pr/96 round is closed (doc pr/96)',
    'work-pr96-dbg1-mcp2k':  'doc pr/96 debug arm, round CLOSED (evts 70084 residual documented)',
    'work-pr96-dbg2-mcp2k':  'doc pr/96 debug arm, round CLOSED',
    'work-pr96-dbg3-mcp2k':  'doc pr/96 debug arm, round CLOSED',
    'work-pr96-fx1-mcp2k':   'doc pr/96 fix arm, round CLOSED',
    # the three DATA Q/L hubs, superseded by ql0819 (owner)
    'work-mcp1k-cb0805':     "scripts/analysis/pr48/backtoback_census.py's 445-dump census stops being re-runnable; doc pr/48 numbers survive as text",
    'work-nuecc48-cb0805':   'scripts/runners/run_pr_geom_arm_dl.sh pctree pin stops resolving',
    'work-ncpi0-cb0805':     'superseded by work-ncpi0-ql0819',
    # doc pr/102, released by the owner (r2 shipped len_admit=30)
    'work-pr102-head-mcp1k':       'doc pr/102 after-epoch arm; round shipped',
    'work-pr102r2-offfull-mcp1k':  'doc pr/102 r2 census/ledger arm; round shipped',
    'work-pr102r2-onfull-mcp1k':   'doc pr/102 r2 census/ledger arm; round shipped',
    'work-pr102r2-on-nuecc48':     'doc pr/102 r2 validation arm; round shipped',
    'work-pr102r2-on-ncpi0':       'doc pr/102 r2 validation arm; round shipped',
    'work-pr102r2-onA-mcp1k':      'doc pr/102 r2 validation arm; round shipped',
    'work-pr102r2-onA-mcp2k':      'doc pr/102 r2 validation arm; round shipped',
    # closed rounds whose scripts name their own arms
    'work-pr99r2-on3-mcp1k':   'doc pr/99 r2 arm; round SHIPPED and pushed',
    'work-pr99r2-on3-mcp2k':   'doc pr/99 r2 arm; round SHIPPED',
    'work-pr99r2-on3-ncpi0':   'doc pr/99 r2 arm; round SHIPPED',
    'work-pr99r2-on3-nuecc48': 'doc pr/99 r2 arm; round SHIPPED',
    'work-vtx100-base-mcp1k':   'doc pr/104 census + vtxscan-vtx100-* label epoch; superseded by the vtx105 epoch (KEEP)',
    'work-vtx100-base-mcp2k':   'doc pr/104 census source; superseded by vtx105',
    'work-vtx100-base-ncpi0':   'superseded by vtx105',
    'work-vtx100-base-nuecc48': 'superseded by vtx105',
    'work-pr107-off-nuecc48':      'doc pr/107 gate arm; round documented, push held by owner -- numbers are text in the doc',
    'work-pr108-off1-nuecc48':     'doc pr/108 Test A arm; round CLOSED (wcp 0e38ad0)',
    'work-pr108-assoccheck-nuecc48': 'doc pr/108 association-check arm; CLOSED',
    'work-pr108-dqdump-on-nuecc48':  'doc pr/108 dQ/dx dump arm; CLOSED, dumps quoted in the doc',
    'work-pr108-dqdump-off-nuecc48': 'doc pr/108 dQ/dx dump arm; CLOSED',
    'work-pr108-dqdump2-on-nuecc48': 'doc pr/108 dQ/dx dump arm; CLOSED',
    'work-pr108-dqdump2-off-nuecc48':'doc pr/108 dQ/dx dump arm; CLOSED',
    'work-pr109-on-nuecc48':         'doc pr/109 arm; round CLOSED and PUSHED (tk a46b0ddb+b5c9f43a)',
    'work-pr109-off-nuecc48':        'doc pr/109 arm; CLOSED',
    'work-pr109e-on-nuecc48':        'doc pr/109 sec 8.3.1 arm; CLOSED, every number re-quoted in the doc',
    'work-pr109e-off-nuecc48':       'doc pr/109 sec 8.3.1 arm; CLOSED',
    'work-pr109f-on-fbcoff-nuecc48': 'doc pr/109 fit_blob_coverage leg; CLOSED',
    'work-pr109f-off-fbcoff-nuecc48':'doc pr/109 fit_blob_coverage leg; CLOSED',
}

REF_ROOTS = ['scripts', 'vtx_rules', 'dl_vtx_training']


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
    for pat, g in ((r'^work-img-', 'imaging-hubs'),
                   (r'^work-.*-cb0805$', 'ql-hubs'),
                   (r'^work-.*-(ql|prod)0819$', 'prod0819-baseline'),
                   (r'^work-.*-prod0813$', 'prod0813-arms'),
                   (r'^work-pr96', 'pr96-family'),
                   (r'^work-pr98', 'pr98-family'),
                   (r'^work-pr99', 'pr99-family'),
                   (r'^work-pr101', 'pr101-family'),
                   (r'^work-pr102', 'pr102-family'),
                   (r'^work-pr103', 'pr103-family'),
                   (r'^work-pr104', 'pr104-family'),
                   (r'^work-pr107', 'pr107-family'),
                   (r'^work-pr108', 'pr108-family'),
                   (r'^work-pr109', 'pr109-family'),
                   (r'^work-vtx100', 'vtx100-family'),
                   (r'^work-vtx105', 'vtx105-family'),
                   (r'^work-vtx106', 'vtx106-family'),
                   (r'^work-scan-prodflip', 'scan-prodflip-arms'),
                   (r'^work-probe', 'probe-arms'),
                   (r'^work-vf', 'valfast-transient'),
                   (r'^work-(mcp1kall|ncpi0|nuecc48|r1qlmc|r2mc)-vfcbr3', 'vfcbr3-nusel-roots'),
                   (r'^work-nuecc48', 'nuecc48-arms'),
                   (r'^work-ncpi0', 'ncpi0-arms'),
                   (r'^work-mcp2k', 'mcp2k-arms'),
                   (r'^work-mcp1k', 'mcp1k-arms'),
                   (r'^work-(r1ql|r2mc|r2patrec)', 'mc-sample-arms')):
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
HEAVY = (re.compile(r'^pctree.*\.tar\.gz$'), re.compile(r'^mabc.*\.zip$'),
         re.compile(r'^calib(-pr)?-evt.*\.json(\.gz)?$'), re.compile(r'.*\.npz$'),
         re.compile(r'^clusters-apa.*\.tar\.gz$'),
         re.compile(r'^opflash_apa.*\.tar\.gz$'),
         re.compile(r'^tracking-pr\.root$'),
         re.compile(r'^oc56scan-evt.*\.jsonl$'))


def is_heavy(f):
    return any(p.match(f) for p in HEAVY)


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
            if is_heavy(f):
                nh += 1
            else:
                keep += sz
                nk += 1
    per[d] = dict(tot=tot, keep=keep, nk=nk, nh=nh)

# ---------------------------------------------------------------- report
print("=== RETIREMENT ROUND 2026-08-23 (minimal state at the latest production) ===")
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
    print(f"    ... and {len(R)-25} more (full list: tierA_20260825.txt)")

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
    print("    (the tree's only live label dir is work-stmcamp-d66new/nusel_labels, in KEEP;")
    print("     sbnd_xin/vertex_labels/ and overclustering_labels/ are outside work-*)")
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

print("\n=== ASSERT 7: PROTECTED.txt is honoured, not merely printed ===")
prot_missing = sorted(PROT_LISTED - KEEP)
if not prot_missing:
    print(f"0 -- PASS  ({len(PROT_LISTED)} PROTECTED.txt names, all in KEEP)")
else:
    fail += 1
    print(f"  !! PROTECTED.txt names NOT in KEEP (would be swept): {prot_missing}")


def _group_provenance(pr_arm, expected_src):
    """Confirm a group-mode PR arm was built from expected_src, by evidence.

    Prefer the line make_group_pctree.py records; otherwise require that every
    member of the arm's first group input archive is byte-identical to a member
    of the expected source root's per-event pctrees.
    """
    import glob, hashlib, tarfile
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
prov_fail = []
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
        # doc 81: a GROUP-mode PR arm's per-event log is a slice of the group log
        # and never names an ql_evt path.  Fall back to evidence: the group input
        # archive's members must be exactly the per-event pctrees of the expected
        # source root.  That is a stronger check than the log grep, not a weaker one.
        found_src = _group_provenance(pr_arm, expected_src)
    if found_src is None:
        prov_fail.append((pr_arm, expected_src, "no ql_evt provenance line, and no group-input evidence"))
    elif found_src != expected_src:
        prov_fail.append((pr_arm, expected_src, f"log actually reads {found_src}"))
    elif found_src not in KEEP:
        # SUCCESSOR rule (new this round, see the dict above): the named root is
        # retiring, but a byte-identical successor survives.  Accepted only with
        # a COMPLETE frozen manifest -- the surviving form of the proof.
        succ = SUCCESSOR.get(found_src)
        samp = found_src[len('work-'):].rsplit('-', 1)[0] if found_src.startswith('work-') else None
        man = os.path.join(STATE, 'hashes', f'stagea-{samp}.tsv')
        rows = sum(1 for ln in open(man) if not ln.startswith('#')) \
            if os.path.exists(man) else 0
        want = FROZEN.get(f'stagea-{samp}.tsv')
        if succ is None:
            prov_fail.append((pr_arm, expected_src,
                              f"{found_src} not in KEEP and no SUCCESSOR declared"))
        elif succ not in KEEP:
            prov_fail.append((pr_arm, expected_src,
                              f"successor {succ} is itself not in KEEP"))
        elif want is None or rows != want:
            prov_fail.append((pr_arm, expected_src,
                              f"successor {succ} declared but frozen manifest "
                              f"{os.path.basename(man)} has {rows} rows, need {want}"))
        else:
            print(f"  OK  {pr_arm:34s} -> {found_src} RETIRED, successor {succ} "
                  f"in KEEP ({rows} frozen rows)")
    else:
        print(f"  OK  {pr_arm:34s} -> {found_src} (in KEEP)")
if not prov_fail:
    print(f"0 -- PASS  ({len(PR_PROVENANCE)} PR arms, every Q/L input confirmed in KEEP)")
else:
    fail += 1
    for arm, exp, why in prov_fail:
        print(f"  !! {arm} (expected src {exp}): {why}")

print("\n=== ASSERT 9: KEEP is closed FORWARD over the campaign input set ===")
# run_ql_batch.sh:51-53 writes "rc=91 ... no-imaging" and EXITS 0 when
# $IMGBASE/evt<N> is absent, so a missing or thinned imaging hub degrades an
# arm to a short one rather than erroring.  Counts, not existence.
# The four DATA imaging hubs are RELEASED this round, so the forward-closure
# property changes SHAPE rather than weakening: stage A for the data samples is
# now work-<s>-grp0825 itself, which carries the same imaging npz (proven
# identical, and frozen) in the same evt<N>/ layout.  It is checked below in
# CAMPAIGN_OUT with its real event counts.  Only the two SIM hubs remain as
# separate imaging inputs, because neither sim sample has a grp0825 arm.
CAMPAIGN_IMG = {
    'work-img-r1qlmc':   10,
    'work-img-r2mc':     13,
}
# The Q/L roots the 2026-08-23 PR re-run will read, plus every PR arm kept --
# with its REAL count, not the sample's nominal one (pr104-on4-mcp2k is 15,
# flipchk-mcp1k is 26, the vtx105 arms are the labeled universe 47/19/407/581).
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
# The imaging half that moves from work-img-<s> into grp0825 this round.
# CAMPAIGN_OUT already counts each grp0825 arm's ql_evt<N> dirs; this counts its
# evt<N> dirs, so the imaging layer the retiring hubs used to hold is verified
# present at full coverage before those hubs are deleted -- not assumed from the
# fact that the Q/L half is there.
GRP_IMG = {
    'work-nuecc48-grp0825':   48, 'work-ncpi0-grp0825': 19,
    'work-mcp1k-grp0825':   1000, 'work-mcp2k-grp0825': 2000,
}
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
        print(f"  !! {arm}: {got} evt* dirs, expected {want} "
              f"-- imaging coverage short, do NOT delete work-img-*")
        a9_fail += 1
    else:
        print(f"  OK  {arm:34s} {got:5d} evt* dirs (imaging layer), in KEEP")
reachable = [c for c in CAMPAIGN_INPUT if c.split('/')[0].startswith('work')]
if reachable:
    print(f"  !! campaign input inside the work* universe: {reachable}")
    a9_fail += 1
if not a9_fail:
    print(f"0 -- PASS  ({len(CAMPAIGN_IMG)} imaging hubs + {len(CAMPAIGN_INPUT)} SP inputs + "
          f"{len(CAMPAIGN_OUT)} kept arms, counts verified, all in KEEP)")
else:
    fail += 1

print("\n=== ASSERT 10 (NEW): every script literal into the removal set is acknowledged ===")
# vtx_rules/baselines.py and scripts/analysis/pr57/oc56_truth.py each broke
# this way in an earlier round, discovered weeks later.  docs/ citation checks
# do not see a path literal in a .py.
lit = collections.defaultdict(list)
scan_files = []
for rt in REF_ROOTS:
    for cur, sub, files in os.walk(rt):
        sub[:] = [s for s in sub if s != '__pycache__' and not os.path.islink(os.path.join(cur, s))]
        scan_files += [os.path.join(cur, f) for f in files
                       if f.endswith(('.py', '.sh', '.json', '.tsv', '.txt'))]
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

print("\n=== ASSERT 11 (NEW): the frozen manifests are COMPLETE, by row count ===")
# Existence is not the test.  A header-only .tsv passes `[ -s ]`, and the
# reference behind it would be gone -- an M1-shaped vacuous PASS with 29 G
# deleted.  So: exact row counts, and stage A must sum to doc 81 sec 7's own
# 24536 archives.
a11_fail, stagea_rows = 0, 0
HASHDIR = os.path.join(STATE, "hashes")
for fn, want in sorted(FROZEN.items()):
    p = os.path.join(HASHDIR, fn)
    if not os.path.exists(p):
        print(f"  !! {fn:30s} MISSING -- run the freeze before planning")
        a11_fail += 1
        continue
    got = sum(1 for ln in open(p) if not ln.startswith('#'))
    if fn.startswith('stagea-'):
        stagea_rows += got
    if got != want:
        print(f"  !! {fn:30s} {got} rows, expected {want}")
        a11_fail += 1
    else:
        print(f"  OK  {fn:30s} {got:6d} rows")
if stagea_rows != STAGEA_TOTAL:
    print(f"  !! stage-A rows total {stagea_rows}, expected {STAGEA_TOTAL} "
          f"(doc 81 sec 7's own archive count) -- the gate has NOT been reproduced")
    a11_fail += 1
else:
    print(f"  OK  stage-A total {stagea_rows} == doc 81 sec 7's 24536 archives")
if not a11_fail:
    print(f"0 -- PASS  ({len(FROZEN)} manifests, every row count exact)")
else:
    fail += 1

print("\n=== ASSERT 12 (NEW): live tools repointed, not merely acknowledged ===")
# ASSERT 10 would classify these as unacked literals and let an ACK entry
# silence them.  That is the wrong disposition: they are LIVE defaults, and the
# correct fix is to repoint them at the surviving byte-identical arm.  doc 82's
# reproducer closed today; it is the freshest thing this round could break.
a12_fail = 0
for f, (gone, present) in sorted(REPOINTED.items()):
    if not os.path.exists(f):
        print(f"  !! {f}: missing")
        a12_fail += 1
        continue
    txt = open(f, errors='replace').read()
    if gone in txt:
        print(f"  !! {f}: still defaults to {gone!r} -- repoint it at {present!r}")
        a12_fail += 1
    elif present not in txt:
        print(f"  !! {f}: {gone!r} removed but {present!r} not found -- check the repoint")
        a12_fail += 1
    else:
        print(f"  OK  {f:38s} -> {present}")
if not a12_fail:
    print(f"0 -- PASS  ({len(REPOINTED)} live tools repointed at surviving arms)")
else:
    fail += 1

# ---------------------------------------------------------------- emit
with open(os.path.join(SCR, "tierA_20260825b.txt"), "w") as fh:
    fh.write("\n".join(sorted(TA)) + "\n")
json.dump({'A': sorted(TA), 'D': [], 'R': R, 'ARCHIVE': ARCHIVE,
           'SUCCESSOR': SUCCESSOR, 'FROZEN': FROZEN,
           'KEEP': sorted(K), 'KEEP_WHY': KEEP_WHY, 'HUB': [], 'POSTBUILD': [],
           'PROTECTED': sorted(KEEP & PROT_LISTED), 'RELEASED': RELEASED,
           'PR_PROVENANCE': PR_PROVENANCE,
           'ACK_BROKEN_REFS': ACK_BROKEN_REFS,
           'script_refs': {k: v for k, v in lit.items()},
           'per': per, 'cites': {},
           'group': {d: group(d) for d in dirs}},
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)
print(f"\nremoval set: {len(R)} dirs -> scripts/retire/tierA_20260825b.txt")
print(f"survivors: {len(K)}")
print(f"state: {STATE}/plan.json")
print("\nOVERALL: " + ("PASS -- all asserts clean" if not fail
                       else f"FAIL -- {fail} assert(s) tripped, do not proceed"))
sys.exit(0 if not fail else 1)
