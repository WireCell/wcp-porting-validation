#!/usr/bin/env python3
"""
Cleanup ROUND J 2026-09-21b -- planner for sbnd_xin + pdvd + pdhd.  Doc sbnd_xin/121.

OWNER INSTRUCTION, verbatim (2026-09-21):
  "Now, can you do one round of cleanup for the pdvd pdhd, sbnd_xin and ~/tmp
   directory? We want to keep the latest production as well as the relevant input
   files. We can retire the intermidiate files to save some disks. We have done
   several round of this in the past. Please plan and execute."
  ... "after, please update the relevant md file, commit and push"

Same keep test as docs 104-114 (substrate | latest production | hand-scan source |
OPEN | live | PROTECTED), re-derived for the state of 2026-09-21.  Fork of
plan_20260919.py; what THIS round changes:

  1. THE SAMPLE GRAMMAR GREW THE ROUND-3 MC TAGS -- the dangerous one, see SAMP below.
     78 of the 146 sbnd arms are work-r3nue-* / work-r3cv-* / work-r3off-*.  Under round
     H's SAMP every docs 115-120 arm would have been unprotectable by name and fallen
     straight to tier 1.  Round A's "the round grammar grew an h" repeating, failing OPEN.
  2. sbnd GETS A SUBSTRATE LIST (round H ran it empty): `d115`, the round-3 stage A
     layer.  The owner's "relevant input files"; M11.
  3. SBND PRODUCTION GREW BY FIVE, re-read from primary sources.  Two of them are the
     value-first catches this round turned on: `d116tfull` is doc 118's gate REFERENCE
     (the current production output) and `d116s0rep` is doc 116's noise-floor CONTROL.
     Both sit inside a d116 family whose other three members are released -- a
     name-adjacency read would have swept both.
  4. THE d116 HOLD IS DROPPED on pdvd and pdhd (doc pdvd/116 closed, committed, pushed),
     and `d119`/`d120` are held on sbnd instead (both landed 2026-09-21).
  5. mg18* RELEASED BY OWNER NAME (see G18), the mg10 precedent.
  6. LIVENESS pins remote_head_20260921b.txt = f04d8937, VERIFIED equal to the remote by
     `ls-remote` over https + the gh credential helper before it was written.  Note that
     today is the worst case for live_tokens(): docs 115-120 are ALL committed and pushed,
     so it reports nothing live and every hold above is typed by hand on purpose.
  7. NO LIVE PEER: ListAgents shows five peer sessions, all offline or idle.  The
     d111/d112/d113/d103/d108 hold in sweep_tmp (round D's, for the pdvd doc-113 peer)
     narrows to d119/d120 for the same reason.

This script RETIRES NOTHING.  It prints a plan and writes tier files.

--- round H's header follows ---
Cleanup ROUND H 2026-09-19 -- planner for sbnd_xin + pdvd + pdhd.  Doc sbnd_xin/111.

OWNER INSTRUCTION, verbatim (2026-09-16):
  "Hi, it is time to clean up some disks for ./pdvd ./pdhd ./sbnd_xin and ~/tmp
   directories. We have accumulated quite a bit intermediate debug files (e.g.
   work*) direcotries. We want to retire some to reduce the disk space usage. We
   have done this multiple times. Note, we want to keep the input and output files
   for the production setting for these experiments, but want to retire the
   intermediate test files. Please act on this and write the md file, commit and
   push."
Same keep test as docs 104-106 (substrate | latest production | hand-scan source |
OPEN | live), re-derived for the state of 2026-09-16.  Fork of plan_20260913.py;
what THIS round changes:

  1. PDVD PRODUCTION MOVED TWICE, read from primary source.
     * doc pdvd/100 sec 7.5 (FLIPPED 2026-09-13): SP top gain 0.889, imaging tag
       `pvdimg` = hard links of `p98von`'s 16 imaging archives (doc 100 line 422);
       `p98von` still holds the production SP frames those archives were imaged from.
     * doc qlmatch/29 sec 7 (FLIPPED 2026-09-13): LASSO boundary weight 0.1 as the
       runner default; F1 arm `q29flip` = production clustering + Q/L, no overrides.
     * doc pdvd/103 sec 14.4 (APPLIED 2026-09-15, toolkit 8fc6070e): both trajectory
       levers; `d103vflip` is the flipped production PR on `p100flip`'s pctrees and
       the source of the production Bee set (doc 110 sec 1); `d103vprod1` is the one
       event re-run through the applied tree with no overrides; `d103v1` is the graded
       A1 cell `d103vflip` reproduces.  Docs 111/112 run SRC=d103vflip.
     p96vprod/p96vscope are SUPERSEDED production (p96vprod survives as a hand-scan
     source).  Their PROTECTED line moves to RETIRED.
  2. PDHD PRODUCTION MOVED.  doc pdvd/108 (2026-09-15, owner override of a D2):
     `d108hflip` is the flipped config with no TLA on `d102hcs`'s pctrees, gated
     equal to `d102hcs` (the graded A1).  doc pdvd/109: `d109hstm` = the display-scope
     proof, source of Bee set c3165b38.  h28prod/h28wl/h28off/h28cfg* are SUPERSEDED
     production and its OFF side; the OFF side of the doc 108 flip is its graded A0
     `d101hnew` (== h28prod on T_stm_michel, doc 103 sec 10), kept.
  3. sbnd production unchanged in reconstruction: `work-*-d102m`/`d102mpr` (doc 109
     line 512: "stay at eacacafe"); `work-*-d109prod` is the only arm run on the
     current reference ref/prod-2026-09-14 (doc 109 Repro lines 61-62), 30 MB, kept.
  4. LIVE PEER, confirmed by message 2026-09-16 13:0x: session
     "steiner-path-3d-deviation" plans doc pdvd/113 and will write d113* arms and
     ~/tmp/d113; it reads d111*/d112* and asked to hold PDHD d101hnew, PDVD d103v1 and
     d101vnew.  Prefixes d111/d112/d113 are held in every tree.
  5. OPEN: `p101q` -- doc pdvd/100 sec 8.7, QtoL 0.0783 "for the owner's decision",
     recommended next step an owner look at `p100flip` and `p101q` dumps side by side.
  6. SUBSTRATE RE-DERIVED (lexical first-hop census 2026-09-16).  New hubs: pvdimg
     11883 inbound, p98von 4097, p100flip 2190.  d16vnu/d16hnu are DROPPED: round C
     kept them as the PR runners' SRC tag, but production is now re-run with
     SRC=d103vflip (PDVD) / SRC=d108hflip (PDHD) (doc 111 Repro lines 141-142).
  7. LIVENESS IS DERIVED AGAINST THE PUSHED REMOTE HEAD, not local main.  Local main
     (30121ad8) is nine commits behind the remote (1ef0e67f), and those nine commits
     are exactly docs 110-112 and pr/149; against local main every one of them reads
     untracked, so their arms would all look live.  A private index read from
     LIVE_REF (default: the sha in remote_head_20260916.txt) is compared instead.
  8. RELEASE BYTES ARE SET-RELATIVE in the work trees too: pvdimg's archives are hard
     links of p98von's, so a dir's du is not what its removal frees.

This script RETIRES NOTHING.  It prints a plan and writes tier files.

--- round C's header follows ---
--- round B's header follows ---
Cleanup round 2026-09-12 ROUND B -- planner for sbnd_xin + pdvd + pdhd.  Doc 105 sec 13.

ROUND B, on the owner's instruction after round A executed (2026-09-12):
  "These can be cleaned up: sbnd d145np (9.85 GiB) ...; pdvd's old flip-evidence
   arms (~4.6 GiB): they need their PROTECTED.txt lines retired first; mg10 and
   the wt-merge/wt-premerge worktrees (~8 GiB)"
Fork of plan_20260912.py with exactly three config changes, each the owner's:
  * sbnd keep_arms: work-*-d145np removed (pr/148's named input, released).
  * open_prefix: mg10 removed from all three trees (the merge validation round).
  * pdvd substrate: d42fit removed -- listed from the 09-10 census; the
    transitive closure below re-derives whether any KEPT arm still borrows from
    it, and pulls it back (reported) if one does.
The matching PROTECTED.txt lines are moved to RETIRED in the same change.  Keep
test, interlocks and the live prefixes h26/h27/h28/p97 are round A's, unchanged.

--- round A's header follows ---
Cleanup round 2026-09-12 -- planner for sbnd_xin + pdvd + pdhd.  Doc 105.

OWNER INSTRUCTION, verbatim:
  "it is time to clean up a bit the pdhd, pdvd, sbnd_xin and ~/tmp directories
   to save some disk space.  We have done this multiple times, please do it
   like what we did before."
So the KEEP TEST IS DOC 104's, unchanged (the owner's 09-10 words still govern:
"keep the latest production, but for work* directory that are intermediate
results, we can retire them"; "save the hand scan results for STM ... PDVD and
PDHD").  An arm is kept only if it is
    substrate | latest production | a hand-scan source | live | OPEN.
Read plan_20260910.py's header for why each of those exists.  Fork of
plan_20260910.py; what THIS round changes:

  1. PRODUCTION MOVED, read from primary source, not the `prod` substring.
     PDVD = p96vprod (doc pdvd/96: "the flipped file with NO TLA", confirmed
     against its measurement arm p96vscope -- the pair is one claim).
     PDHD = h26q2dprod (doc pdhd/26 sec 1.2 line 121: "PDHD production =
     h26q2dprod (identical to h26conf on every verdict)"; h26conf is the
     additive-gate baseline, the other half of that claim).
     sbnd = work-*-d102m / d102mpr, unchanged since 09-08.
     p79vprod (09-10's PDVD production) is SUPERSEDED and its PROTECTED.txt
     line is moved to RETIRED, deliberately.

  2. THE ROUND GRAMMAR GREW AN `h`.  PDHD rounds since doc pdhd/18 name arms
     h18b, h22conf, h25kr ...  Every arm-token regex here said (?:p|d), so
     live_tokens() could not see an h-round as live and INTERLOCK 14 could not
     map h22conf to doc pdhd/22.  Both now accept h.  (INTERLOCK 14 failing
     closed would have refused the round; live_tokens failing open would have
     released a live h-round -- the dangerous half.)

  3. THE HAND-SCAN SET GREW, and half of it has no prep set.  smx5..smx9
     (pdvd) and smx18..smx27 + own19..own26 (pdhd) were all scanned after
     09-10.  scan_arms_20260913.py now also reads docs/scan records, because
     smx25/smx26 put h25k/h25kr/h25r movers in front of the scanners through
     prep-pdhd-smx19 re-use rather than a prep dir of their own.

  4. LIVE: the peer round that wrote doc pdhd/26 at 18:24 today is still
     running (~/tmp/h27 written 18:24, h27cfg* work dirs, its transcript
     written 18:31).  h26*/h27* are protected by PREFIX before the first plan
     run -- live_tokens() cannot see ~/tmp, and doc pdhd/26 is committed.

  5. A CONFIRM-TIME RE-PLAN MUST NOT OVERWRITE THE PLAN-TIME RECORD.  On
     2026-09-11 05:00 the 09-10 driver's INTERLOCK A re-ran the planner for
     `1 pdhd`, which rewrote keep_pdvd_20260910.txt (a committed record, +120
     p82vprod lines) and every prebroken_*_20260910.txt baseline.  With
     PLAN_SUFFIX set (the driver sets it) every output of this script goes to
     <name>.<suffix>.txt and the plan-time files are never touched.

This script RETIRES NOTHING.  It prints a plan and writes tier files.
"""
import os, re, sys, json, time, subprocess, collections, glob
G87 = "doc 87 sec 6.4's group-mode arms (sbnd PROTECTED line 291, RETIRED 2026-09-13); docs/87_production-output-minimization.md is committed, but a family named 87grp-* carries no d/p/h round prefix for INTERLOCK 14 to map"
G = "the owner released the undocumented master-merge validation round by name on 2026-09-12 (doc 105 sec 13); its ~/tmp/mg10 record layer is frozen in cleanup-20260913/tmp-tier1"
G18 = "ROUND I: the owner released the undocumented 2026-09-18 master-merge validation round (toolkit b93673ef/d2646110) by name on 2026-09-21 -- 'Release all of it' -- exactly as the mg10 precedent (doc 105 sec 13) requires; the merge is long since on master and the toolkit has moved four commits past it.  ~/tmp/mg18 (17.5 GiB, four CMake build trees) goes in the same round's sweep"

R     = "/home/xqian/toolkit-dev/wcp-porting-img"
STAMP = "20260921b"
SUFFIX = ('.' + os.environ["PLAN_SUFFIX"]) if os.environ.get("PLAN_SUFFIX") else ""
HERE  = os.path.dirname(os.path.abspath(__file__))
EVT   = re.compile(r"^(\d{6})_(\d+)(?:_(.+))?$")
# ROUND I: the sample grammar grew the round-3 MC tags.  78 of the 146 sbnd arms are
# work-r3nue-* / work-r3cv-* / work-r3off-* (doc 115's samples, and every docs 116-120
# arm built on them).  With r3* unknown, arm_token("work-r3cv-d116tfull") returns
# "r3cv-d116tfull": production=["d116tfull"] cannot match it, open_prefix=("d119",)
# cannot match it (the dir starts "work-", the token starts "r3cv-"), and `a in LIVE`
# cannot match it -- so EVERY docs 115-120 arm would be unprotectable by name and fall
# straight to tier 1.  This is round A's "the round grammar grew an h" (header item 2 of
# plan_20260912.py) repeating, in the failing-OPEN direction.
SAMP  = re.compile(r"-(mcp1k|mcp2k|ncpi0|nuecc48|dbg25a|r3cv|r3nue|r3off)(?=-|$)|^(mcp1k|mcp2k|ncpi0|nuecc48|dbg25a|r3cv|r3nue|r3off)-")

def scan_arms(det):
    """The hand-scan sources, re-derived here rather than pasted."""
    p = os.path.join(HERE, f"scan_arms_{STAMP}.json")
    if not os.path.exists(p):
        sys.exit(f"missing {p} -- run: python3 scan_arms_{STAMP}.py --json={p}")
    return list(json.load(open(p)).get(det, []))

TREES = {
 "sbnd": dict(
   root=f"{R}/sbnd/sbnd_xin", work=f"{R}/sbnd/sbnd_xin", unit="siblingdir",
   # ROUND I: sbnd gets a substrate list for the first time; round H ran it EMPTY.
   # `d115` is the round-3 STAGE A layer -- the per-group SP `g*/frames-dnn.tar.bz2`
   # archives (8.6 MB each) that every docs 116-119 stage-B arm was built on, and the
   # ONLY copy on our disk: the reco1 inputs are reached through the untracked symlink
   # `xin-round3-samples` -> /nfs/data/1/yuhw/2025-fall-prod-sample/ (doc 115 Repro), which
   # is not ours.  This is the owner's "relevant input files", and CLAUDE.md M11.
   substrate=["d115"],
   # ROUND I (2026-09-21).  Production re-derived from primary sources, NOT inherited.
   # Round H's four stay, and the docs 115-119 campaign adds four more:
   #  * stage A  `work-<s>-d102m`  -- doc 109 line 512 "stay at eacacafe"; unchanged since 09-08.
   #  * stage B  `work-<s>-pr150s0` -- still the newest FULL four-sample stage-B output set.
   #    Production's *operating point* has moved three times since (ref/prod-2026-09-20,
   #    -21c, -21d), but MEASURED: no d118/d119 arm exists on mcp1k/mcp2k/ncpi0/nuecc48, so
   #    no newer full set exists, and sentinels_tolerant.py --arms 'work-*-pr150s0' (21 PASS
   #    / 0 FAIL / 2 OPEN / 7 INERT, baseline re-run 2026-09-21 11:00) still resolves here.
   #  * `d102mpr` is the SUPERSEDED stage B (cit 175, 56 in scripts).  KEPT again:
   #    feedback_gate_source_arm_retired -- a superseded OUTPUT arm can still be the
   #    INPUT a published gate resolves into, and a newer arm is never a substitute.
   #  * `d109prod` -- the only arm on ref/prod-2026-09-14 (doc 109 Repro 61-62), 30 MB.
   #  * `d116tfull` -- ROUND I, VALUE-FIRST.  This is the CURRENT PRODUCTION OUTPUT, not a
   #    sweep point.  Doc 118 (Status: FLIPPED, toolkit 675fd266) line 5: the flip is "doc
   #    116's `tfull`"; its decisive gate G1 compares every product of the flipped default
   #    AGAINST `work-r3*-d116tfull`, archive members by content -- 4 call sites in
   #    scripts/d118/hash_gate.py, 3 in stageB_flip.sh, plus d119's stageB_tail/stageB_alloc.
   #    A name-adjacency read would have swept it with the rest of the d116 family.
   #  * `d116s0rep` -- ROUND I, the same catch on the other side: the NOISE-FLOOR CONTROL.
   #    Doc 116's cell table line 114: "`s0rep` | none | production re-run: the noise floor",
   #    compiled-config IDENTICAL to the no-TLA baseline (4ee005c935dac7df, doc 116 sec 2).
   #    It is the `arm_before` of doc 119 round 0's own Repro line.  Releasing it while
   #    keeping `d118fliprep` -- the same kind of arm -- would have been incoherent.
   #  * `d118flip`/`d118fliprep` -- the flip arm on ref/prod-2026-09-20 and its determinism
   #    control (doc 118 Repro; root_gate.py --arms d118flip d118fliprep "control").
   #  * `d115pr` -- the published truth baseline docs 116/117 grade every rung against
   #    (numuCC 69.5/86.4, nueCC 40.9 %).  OWNER'S CALL 2026-09-21: keep.
   # ROUND J: `d102mpr` LEAVES production.  Round H and round I both kept it on
   # feedback_gate_source_arm_retired -- a superseded OUTPUT arm can still be the INPUT a
   # published gate resolves into.  Two rounds of cooling-off have passed, the owner released
   # it by name on 2026-09-21 from a menu that stated the cost ("a published gate that
   # resolves into it could no longer be replayed"), and INTERLOCK 9 re-checks that no LIVE
   # manifest still resolves into it before anything is removed.  10.26 GiB.
   production=["d102m","pr150s0","d109prod",
               "d116tfull","d116s0rep","d118flip","d118fliprep","d115pr"],
   scan_arms=[],
   flip_evidence=[],
   # ROUND I: docs 115-118 are closed AND pushed, so live_tokens() sees none of them -- which
   # is exactly why the hold below is typed by hand.  Doc 120 and ref/prod-2026-09-21d landed
   # TODAY at 10:22 and doc 119's round 5 also landed today, so both rounds are held one round.
   # The hold is spelled BOTH ways because arm_token() strips the sample: a dir is
   # `work-r3nue-d119tail` while its token is `d119tail`.
   # This also holds `work-r3nue-d119flip-torn` (a torn-write arm, ~1 GiB) -- releasing it
   # would need an exception to the prefix rule, which is not worth 1 GiB.  Next round's
   # first release.
   open_prefix=("work-d103","work-d104","work-d149","work-d150",
                "work-r3nue-d119","work-r3cv-d119","work-r3off-d119",
                "work-r3nue-d120","work-r3cv-d120","work-r3off-d120",
                "d119","d120"),
   keep_arms=[
     # the sentinel suite's negative-control layer and the 31/31 witness.
     "work-s144pos-mcp1k","work-s144pos-mcp2k","work-s144pos-nuecc48",
     "work-s144posleg-mcp1k","work-s144posleg-mcp2k","work-s144posleg-nuecc48",
     "work-s144neg-dvtx","work-s144neg-prox","work-s144neg-gf","work-s144neg-sccc",
     "work-s144neg-memgeo","work-s144neg-backg","work-s144neg-cone","work-s144neg-bfill",
     # INTERLOCK 9: em_display's manifests still RESOLVE into these.
     "work-pr134-f086-mcp1k","work-pr134-f086-mcp2k",
     "work-pr134-f086-ncpi0","work-pr134-f086-nuecc48",
     # ROUND H, value-first (feedback_audit_value_first_not_name_adjacency):
     # pr150csp3bw carries 225 vertex_labels -- it is a HAND-SCAN SOURCE, not a sweep point.
     # pr150tcsp3bw is doc pr/150's most-cited arm (155 doc hits) at 1.47 GiB.
     # pr150g16new / d113g16off are doc 113's G1 gate pair (80 MB each).
     "pr150csp3bw","pr150tcsp3bw","pr150g16new","d113g16off",
   ],
   uncited_ok={},
   tier2={}, tier3={}, tier4={},
 ),

 "pdvd": dict(
   root=f"{R}/pdvd", work=f"{R}/pdvd/work", unit="armsuffix",
   # ROUND H: d103vflip joins the substrate -- it is the SRC the doc-116 flip arm was built on
   # (doc 116 sec 10.0: ARM=d116vflip SRC=d103vflip) and carries 138 doc citations.
   substrate=["pvdimg","p98von","p100flip","d51vclus","keep","d27fresh","d41prov","d39r2prov","(bare)","d103vflip"],
   # PRODUCTION MOVED 2026-09-18 (doc pdvd/116 sec 10, owner flip; toolkit f9665bea).
   # Shipped VALUES: prefer3 + tree+path alpha 0.5, stm_proton_muon_guard true,
   # michel_min_kink_deg 20, michel_max_len_cm 30.  The arm carrying them with NO TLA is
   # `d116vflip` (sec 10.0 (d)); `d116vr2` is the graded R2 cell it reproduces (also a
   # hand-scan source via own116v, so scan_arms holds it independently).
   # d103vprod1 = the SUPERSEDED applied-tree production, kept one round (1.25 GiB).
   # ROUND J: `d103vprod1` LEAVES production -- the superseded applied-tree production,
   # kept one round by round H and another by round I.  Owner released it 2026-09-21.  1.25 GiB.
   production=["d116vflip","d116vr2","q29flip","q29stm"],
   scan_arms=scan_arms("pdvd"),
   flip_evidence=[],
   # ROUND I: the d116 hold is DROPPED -- doc pdvd/116 closed 2026-09-18, is committed and
   # pushed, and its production arms (d116vflip/d116vr2) are named in `production` above, so
   # they no longer need a prefix to survive.
   # But `d119` IS held, matching sbnd's open_prefix.  The first plan run released
   # d119vnu/d119vr3 here while holding every d119 arm on sbnd -- doc sbnd_xin/119 is ONE
   # cross-detector round whose rounds 3 and 5 were gated on PDHD and PDVD too, and whose
   # round 5 (the allocator flip, ref/prod-2026-09-21c) landed TODAY.  Holding it on one
   # detector and releasing it on the other two is incoherent; the hold costs 2.38 GiB here
   # and 1.82 on pdhd, 2.9 % of the round.
   open_prefix=("d119",),
   keep_arms=["magnify","ql_scores","ql_labels","stm_michel_labels",
              "d08_scan_labels","d08pv_scan_labels",
              "p101q",            # OPEN: doc pdvd/100 sec 8.7 QtoL, owner decision
              "d103v1","d101vnew","d103v0"],
              # ROUND I: `mg18vbase`/`mg18vfix` (the 2026-09-18 master-merge validation round,
              # toolkit b93673ef/d2646110) are REMOVED from this list.  Round H held them
              # because the round is undocumented, so INTERLOCK 14 cannot map it to a doc by
              # round number, and the mg10 precedent (doc 105 sec 13) is that the OWNER
              # releases a merge-validation round by name.  The owner did, 2026-09-21:
              # "Release all of it".  They are carried in `uncited_ok` below so INTERLOCK 14
              # records the ruling instead of failing closed on it.
   uncited_ok={"mg18vbase": G18, "mg18vfix": G18},
   tier2={}, tier3={}, tier4={},
 ),

 "pdhd": dict(
   root=f"{R}/pdhd", work=f"{R}/pdhd/work", unit="armsuffix",
   # ROUND H: d108hflip joins the substrate -- SRC of the doc-116 flip arm (sec 10.0), 88 cits.
   substrate=["d51hclus","d09","d09ctl","d09ctl2","stm0","(bare)","wcc","d108hflip"],
   # PRODUCTION MOVED 2026-09-18 (doc pdvd/116 sec 10): same values as PDVD.
   # `d116hflip` = flipped config, no TLA; `d116hr2` = the graded R2 cell (own116h scan source).
   production=["d116hflip","d116hr2","d102hcs","d109hstm"],
   scan_arms=scan_arms("pdhd"),
   flip_evidence=[],
   # ROUND I: the d116 hold is DROPPED, `d119` held -- same reasons as pdvd above.
   open_prefix=("d119",),
   keep_arms=["stmw","allpd","ql_labels","stm_scan_labels","stm_michel_labels",
              "d08_scan_labels","d05_scan_labels",
              "d101hnew"],
              # ROUND I: `mg18hbase`/`mg18hfix` REMOVED -- see the pdvd note above; the
              # owner released the 09-18 merge-validation round by name on 2026-09-21.
   uncited_ok={"mg18hbase": G18, "mg18hfix": G18},
   tier2={}, tier3={}, tier4={},
 ),
}

def load_citations():
    p = os.path.join(HERE, f"cit_{STAMP}.json")
    t = os.path.join(HERE, f"toks_{STAMP}.json")
    for f in (p, t):
        if not os.path.exists(f):
            sys.exit(f"missing {f} -- run:  python3 toks_{STAMP}.py && "
                     f"python3 cit_{STAMP}.py toks_{STAMP}.txt cit_{STAMP}.json")
    d = json.load(open(p))
    toks = {k.split("\t", 1)[1]: v["toks"] for k, v in json.load(open(t)).items()}
    return d["count"], d.get("who", {}), toks

def protected_lines():
    """Union of every PROTECTED.txt.  Returns (exact names, prefixes).

    A line ending in '*' is a PREFIX.  The 09-08 matcher compared
    `arm == line.strip('*')`, which matched nothing, so every `d51v*`-shaped
    line in these files protected exactly zero dirs and the round was relying
    on its own inline open_prefix instead.  Honouring them is strictly the
    safe direction; the lines that should no longer protect anything were
    moved to each file's RETIRED section rather than left to fail silently."""
    names, prefixes = set(), set()
    for p in (f"{R}/sbnd/sbnd_xin/scripts/retire/PROTECTED.txt",
              f"{R}/pdhd/scripts/retire/PROTECTED.txt",
              f"{R}/pdvd/scripts/retire/PROTECTED.txt"):
        if not os.path.exists(p): continue
        for line in open(p):
            line = line.strip()
            if not line or line.startswith("#"): continue
            for tok in line.split("\t")[0].split():
                (prefixes if tok.endswith("*") else names).add(tok.rstrip("*"))
    return names, prefixes

DOCROOTS = ("pdvd/docs", "pdhd/docs", "sbnd/sbnd_xin/docs",
            "pdvd/scripts", "pdhd/scripts", "sbnd/sbnd_xin/scripts")

def live_tokens():
    """Arm families named by a record that is NOT YET COMMITTED.

    THE RULE THIS REPLACES was a hand-typed open_prefix, and on 2026-09-10 it
    was already wrong: doc pdvd/82 was written at 21:13 and its p82v* arms at
    21:26, while the list said p79/p80/p81.  A live round is exactly a round
    whose record is still uncommitted, so derive it instead of typing it --
    that is self-maintaining and cannot go stale between rounds.

    Untracked file  -> every arm token in it is live.
    Modified file   -> only tokens on ADDED lines (a tracked doc names its
                       whole campaign's history; the new lines are the live part).
    Binary files are skipped: pdhd/docs/scripts/ currently holds a stray ELF
    executable, which `strings` would happily mine for false arm tokens."""
    toks, files = set(), []
    # round D: against the PUSHED remote head through a private index (header item 7).
    ref = os.environ.get("LIVE_REF") or open(os.path.join(HERE, f"remote_head_{STAMP}.txt")).read().strip()
    idx = f"/home/xqian/tmp/cleanup-{STAMP}/live_index{SUFFIX}"
    os.makedirs(os.path.dirname(idx), exist_ok=True)
    genv = dict(os.environ, GIT_INDEX_FILE=idx)
    if subprocess.run(["git","-C",R,"read-tree",ref], env=genv).returncode != 0:
        sys.exit(f"live_tokens: git read-tree {ref} failed -- fetch the remote first")
    subprocess.run(["git","-C",R,"update-index","-q","--refresh"], env=genv,
                   capture_output=True)
    unt = subprocess.run(["git","-C",R,"ls-files","--others","--exclude-standard","--"] + list(DOCROOTS),
                         env=genv, capture_output=True, text=True).stdout.splitlines()
    mod = subprocess.run(["git","-C",R,"diff","--name-only","--diff-filter=AM","--"] + list(DOCROOTS),
                         env=genv, capture_output=True, text=True).stdout.splitlines()
    g = [f"?? {p}" for p in unt] + [f" M {p}" for p in mod]
    print(f"# live_tokens: reference {ref[:12]}, {len(unt)} untracked + {len(mod)} modified path(s) under DOCROOTS")
    # {0,14}: {1,14} could not match a 4-character arm such as d53v (doc 105).
    # round D: q29*/pr149* are real round grammars too (qlmatch doc 29, sbnd pr/149).
    ARMRE = re.compile(r"\b(?:p|d|h|q|pr)[0-9]{1,3}[a-z][a-z0-9]{0,14}\b")
    for line in g:
        st, _, path = line[:2], None, line[3:].strip()
        # THE ROUND MUST NOT READ ITS OWN OUTPUT.  tier1_*.txt, keep_*.txt and
        # toks.txt all live under scripts/retire and name EVERY arm in the
        # tree; on the first run they made 154 pdvd families look "live" and
        # cut the release from 119.93 GiB to 11.40.  This is doc 91's
        # "protected because protected" and the 09-05/09-06 recurrence of it --
        # cit_*.py carries --exclude-dir=retire for exactly this reason.
        if "/retire/" in f"/{path}" or path.rstrip("/").endswith("/retire"):
            continue
        # ...AND NOT ITS OWN DOCS.  2026-09-12, measured: the only thing marking
        # p82vprod "live" was an uncommitted edit to doc 104 (its sec 15 names the
        # arms the 09-11 re-plan wrote), and doc 105 will name every arm it
        # releases.  A cleanup doc names arms in order to RELEASE them; reading it
        # as liveness is doc 91's "protected because protected" on the doc layer.
        if re.search(r"(^|/)[0-9]*_?[^/]*(cleanup|retire)[^/]*$", path):
            continue
        full = os.path.join(R, path)
        paths = []
        if os.path.isdir(full):
            for cur, _s, fs in os.walk(full): paths += [os.path.join(cur, f) for f in fs]
        else:
            paths = [full]
        for fp in paths:
            try:
                with open(fp, "rb") as fh: head = fh.read(4096)
                if b"\0" in head: continue            # binary: skip
                if st.strip() == "??":
                    text = open(fp, errors="ignore").read()
                else:
                    text = subprocess.run(["git","-C",R,"diff","-U0","--",
                                           os.path.relpath(fp, R)],
                                          env=genv, capture_output=True, text=True).stdout
                    text = "\n".join(l[1:] for l in text.splitlines()
                                     if l.startswith("+") and not l.startswith("+++"))
            except OSError:
                continue
            hits = set(ARMRE.findall(text))
            if hits: files.append((path, sorted(hits)[:6]))
            toks |= hits
    return toks, files

LIVE, LIVE_FILES = live_tokens()
CIT, WHO, TOKS = load_citations()
PROT, PROTPFX  = protected_lines()
fails          = []

def prot_hit(d, a):
    if d in PROT or (a and a in PROT): return True
    for p in PROTPFX:
        if d.startswith(p) or (a and a.startswith(p)): return True
    return False

def check(tree, n, ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  INTERLOCK {n}: {msg}")
    if not ok: fails.append((tree, n))

def du_kb(paths):
    out = {}
    for i in range(0, len(paths), 400):
        r = subprocess.run(["du","-sk"]+paths[i:i+400], capture_output=True, text=True).stdout
        for l in r.splitlines():
            kb, p = l.split("\t", 1); out[os.path.basename(p)] = int(kb)
    return out

def arm_token(d, unit):
    if unit == "armsuffix":
        m = EVT.match(d)
        return None if not m else (m.group(3) or "(bare)")
    return SAMP.sub("", d[5:]).strip("-") if d.startswith("work-") else None

def owner_of(linkpath):
    """Owning arm dir of a symlink TARGET, by grammar match on the normalised
    path parts.  Never relpath against a ROOT constant -- that was vacuous for
    absolutely-spelled links because /nfs/data/1/... is itself a symlink."""
    full = os.path.normpath(os.path.join(os.path.dirname(linkpath),
                                         os.readlink(linkpath)))
    own = None
    for part in full.split(os.sep):
        if EVT.match(part) or part.startswith("work-"): own = part
    return own

def plan_tree(tree, cfg):
    print(f"\n{'='*78}\n== {tree}   ({cfg['work']})\n{'='*78}")
    WORK, unit = cfg["work"], cfg["unit"]
    entries = sorted(d for d in os.listdir(WORK)
                     if os.path.isdir(os.path.join(WORK, d))
                     and not os.path.islink(os.path.join(WORK, d)))
    parsed    = {d: arm_token(d, unit) for d in entries}
    universe  = {d: a for d, a in parsed.items() if a is not None}
    out_scope = sorted(d for d, a in parsed.items() if a is None)

    SCAN = set(cfg["scan_arms"])
    keep_names = (set(cfg["substrate"]) | set(cfg["production"]) | SCAN
                  | set(cfg["flip_evidence"]) | set(cfg["keep_arms"]))
    def is_open(d, a):
        if d.startswith(cfg["open_prefix"]) or (a or "").startswith(cfg["open_prefix"]):
            return True
        return bool(a) and a in LIVE        # derived: named by an uncommitted record
    keep_dirs = {d for d, a in universe.items()
                 if d in keep_names or a in keep_names or is_open(d, a)}
    for d, a in universe.items():
        if prot_hit(d, a): keep_dirs.add(d)

    cand  = {d for d in universe} - keep_dirs
    rest  = sorted(cand)

    def cited(d):
        return max([CIT.get(d, 0), CIT.get(universe[d] or "", 0)]
                   + [CIT.get(t, 0) for t in TOKS.get(d, ())])
    # REPORTED, NOT OBEYED.  See header item 1.
    tier1 = list(rest)
    tier2 = tier3 = tier4 = []

    # --- transitive closure: keeping a dir means keeping its substrate -----
    relset = set(tier1)
    for _ in range(12):
        pull = set()
        for d in keep_dirs:
            for cur, sub, files in os.walk(os.path.join(WORK, d)):
                for e in files + sub:
                    fp = os.path.join(cur, e)
                    if os.path.islink(fp):
                        o = owner_of(fp)
                        if o in relset: pull.add(o)
        if not pull: break
        keep_dirs |= pull; relset -= pull
    closure = sorted(set(tier1) - relset)
    tier1 = [d for d in tier1 if d in relset]
    if closure:
        print(f"        closure: +{len(closure)} dirs pulled back as substrate of a kept dir")
    KEEP = sorted(keep_dirs)

    def ndirs(name):
        return sum(1 for d, a in universe.items() if d == name or a == name)

    # ---- INTERLOCK 1: substrate present
    short = {a: ndirs(a) for a in cfg["substrate"] if ndirs(a) == 0}
    check(tree, 1, not short, f"substrate present ({short or 'all present'})")

    # ---- INTERLOCK 2: no KEPT symlink may resolve into a releasing dir
    dangle = []
    for d in KEEP:
        for cur, sub, files in os.walk(os.path.join(WORK, d)):
            for e in files + sub:
                fp = os.path.join(cur, e)
                if os.path.islink(fp) and owner_of(fp) in relset:
                    dangle.append(f"{d}: {os.path.relpath(fp, WORK)} -> {owner_of(fp)}")
            if len(dangle) > 5: break
    check(tree, 2, not dangle, f"no kept symlink resolves into a releasing dir "
          f"({len(dangle)} would dangle{'; e.g. ' + dangle[0] if dangle else ''})")

    # ---- INTERLOCK 3: live-WRITER guard (mtime double-sample + scoped ps)
    def snap(dirs):
        out = {}
        for d in dirs:
            acc = []
            for cur, sub, files in os.walk(os.path.join(WORK, d)):
                for f in files:
                    try: acc.append(os.path.getmtime(os.path.join(cur, f)))
                    except OSError: pass
                if len(acc) > 3000: break
            out[d] = (len(acc), max(acc) if acc else 0)
        return out
    sample = tier1[::max(1, len(tier1)//120)] if tier1 else []
    before = snap(sample)
    ps  = subprocess.run(["ps","-eo","cmd"], capture_output=True, text=True).stdout
    busy = [l for l in ps.splitlines()
            if re.search(r"wire-cell|run_pr_evt|run_clus_evt|run_img_evt|wcsonnet", l)
            and cfg["root"] in l and f"plan_{STAMP}" not in l and "grep" not in l]
    time.sleep(12)
    moved = [d for d in sample if before[d] != snap([d])[d]]
    check(tree, 3, not moved and not busy,
          f"no live writer ({len(moved)} of {len(sample)} sampled dirs moved, "
          f"{len(busy)} tree-scoped wire-cell procs)")

    # ---- INTERLOCK 4: pre-existing broken symlinks recorded BEFORE the round
    pre = subprocess.run(["find", cfg["root"], "-xtype", "l"],
                         capture_output=True, text=True).stdout.split()
    open(os.path.join(HERE, f"prebroken_{tree}_{STAMP}{SUFFIX}.txt"), "w").write("\n".join(pre))
    check(tree, 4, True, f"pre-existing broken symlinks recorded: {len(pre)} "
                         f"(the post-state number only means something against this)")

    # ---- INTERLOCK 5: PROTECTED.txt names nothing in the release
    hit = sorted({d for d in tier1 if prot_hit(d, universe[d])})
    check(tree, 5, not hit, f"PROTECTED.txt clear of the release ({hit[:4] or 'clear'})")

    # ---- INTERLOCK 7: nothing being released is a record/label dir
    RECORD = ("labels","decisions","ql_labels","stm_scan_labels","stm_michel_labels",
              "snap","sweep","em_labels","vertex_labels","scan_labels","state-","archive")
    bad = [d for d in tier1 if any(k in d for k in RECORD)]
    check(tree, 7, not bad, f"no record/label dir in the release ({bad or 'clear'})")

    # ---- INTERLOCK 8: never delete THROUGH a symlink
    thru = [d for d in tier1 if os.path.islink(os.path.join(WORK, d))]
    check(tree, 8, not thru, f"no target is itself a symlink ({thru or 'clear'})")

    # ---- INTERLOCK 9: no arm a LIVE manifest still RESOLVES into
    man = set()
    for pat in ("em_display/*.tsv","em_display/*.txt","*/manifest*.tsv","docs/scan/*.tsv",
                "*_scan/*.tsv","*_labels/*.json","*_labels/*/*.json",
                "*_scan/prep*/*.json","*_scan/*.py"):
        for f in glob.glob(os.path.join(cfg["root"], pat)):
            try: t = open(f, errors="ignore").read()
            except OSError: continue
            for mm in re.finditer(r"work-[A-Za-z0-9_.-]+|[0-9]{6}_[0-9]+_[A-Za-z0-9-]+", t):
                man.add(mm.group(0))
    resolved = {a for a in man if os.path.isdir(os.path.join(WORK, a))}
    inrel    = sorted(resolved & relset)
    check(tree, 9, not inrel, f"manifests name {len(man)} arms, {len(resolved)} exist, "
          f"{len(inrel)} in the release ({inrel[:4] or 'none'})")

    # ---- INTERLOCK 11: the production and substrate sets must RESOLVE
    unres = sorted(n for n in cfg["substrate"] + cfg["production"] + cfg["scan_arms"]
                   if not any(d == n or universe[d] == n for d in universe))
    check(tree, 11, not unres,
          f"every substrate/production/scan name resolves ({unres or 'all resolve'})")

    # ---- INTERLOCK 13 (NEW): the hand scans keep their SOURCE arm.
    # Re-derived here from the scan tooling, not read from the tier config, so
    # that a config typo cannot quietly drop a scan.  The owner asked for the
    # STM scans by name; this is that request made mechanical.
    sa = set(cfg["scan_arms"])
    lost = sorted(a for a in sa
                  if any(universe[d] == a for d in tier1))
    missing = sorted(a for a in sa if ndirs(a) == 0)
    check(tree, 13, not lost and not missing,
          f"hand-scan sources kept: {sorted(sa) or 'none for this tree'}"
          f"{'; IN RELEASE: ' + str(lost) if lost else ''}"
          f"{'; DOES NOT RESOLVE: ' + str(missing) if missing else ''}")

    # ---- INTERLOCK 14 (NEW): an UNCITED family may not be released silently.
    # Citation no longer keeps an arm (header item 1), which removes the only
    # thing that used to make an undocumented family visible.  So the burden
    # flips: a cited family releases on its doc; an UNCITED one must be shown
    # to belong to a round whose doc IS committed.  Derived from the family's
    # own round number, not from a hand-written excuse -- d11fvoff releases
    # because docs/11_*.md exists and is committed, and if no such doc existed
    # the family would be an undocumented round and this would refuse.
    # round D: pr<N> (sbnd docs/pr/<N>_*.md) and q<N> (pdvd docs/qlmatch/<N>_*.md) map by number too.
    RNUM = re.compile(r"^(?:pr|d|p|h|q)(\d{1,3})")
    committed_docs = set()
    for droot in (f"{R}/pdvd/docs", f"{R}/pdhd/docs", f"{R}/sbnd/sbnd_xin/docs"):
        for cur, _s, fs in os.walk(droot):
            for f in fs:
                m = re.match(r"^(\d{1,3})[_-]", f)
                if m and f.endswith(".md"): committed_docs.add(int(m.group(1)))
    unc = collections.Counter()
    for d in tier1:
        if not cited(d): unc[universe[d]] += 1
    ungrounded, excused = [], []
    for a in unc:
        m = RNUM.match(a or "")
        if m and int(m.group(1)) in committed_docs:
            continue
        # round B: the config's uncited_ok hook was present since 09-06 and never
        # read.  Wired now, EXACT names only, and every use is printed below so a
        # PASS says which families passed on an owner ruling rather than a doc.
        if a in cfg.get("uncited_ok", {}):
            excused.append(a); continue
        ungrounded.append(a)
    ungrounded = sorted(ungrounded)
    check(tree, 14, not ungrounded,
          f"{len(unc)} uncited families in the release; each maps to a committed "
          f"doc by round number ({len(ungrounded)} do not: {ungrounded[:6] or 'none'})"
          f"{'; excused by owner ruling: ' + str(sorted(excused)) if excused else ''}")

    # ---- INTERLOCK 15 (NEW): the RECORD must outlive the BYTES.
    # The whole licence for this round is that a committed doc replaces the
    # per-event products.  So no released family may be named by a record that
    # is still uncommitted -- that arm's doc could still change, or be a live
    # round's.  This is the negative half of INTERLOCK 16.
    leak = sorted({universe[d] for d in tier1 if universe[d] in LIVE})
    check(tree, 15, not leak,
          f"nothing released is named by an uncommitted record ({leak or 'clear'})")

    # ---- INTERLOCK 16 (NEW): show what the live derivation actually found,
    # so a PASS above is readable rather than vacuous (doc 91's "protected
    # because protected" and the 09-05 zero-fires shape).
    mine = sorted({a for a in LIVE if any(universe[d] == a for d in universe)})
    check(tree, 16, True,
          f"live-by-uncommitted-record: {len(LIVE)} tokens from "
          f"{len(LIVE_FILES)} file(s); {len(mine)} resolve in this tree "
          f"({mine[:8] or 'none'})")

    # ------------------------------------------------------------- report --
    sz = du_kb([os.path.join(WORK, d) for d in tier1 + KEEP])
    t1kb = sum(sz.get(d, 0) for d in tier1)
    keepkb = sum(sz.get(d, 0) for d in KEEP)
    # round D: set-relative -- an inode frees bytes only if every link lies in the release.
    ino = {}
    for d in tier1:
        for cur, subs, fs in os.walk(os.path.join(WORK, d)):
            subs[:] = [x for x in subs if not os.path.islink(os.path.join(cur, x))]
            for f in fs:
                try: st = os.lstat(os.path.join(cur, f))
                except OSError: continue
                if not (st.st_mode & 0o170000 == 0o100000): continue
                k = (st.st_dev, st.st_ino)
                c, n, b = ino.get(k, (0, st.st_nlink, st.st_blocks * 512))
                ino[k] = (c + 1, n, b)
    freed_gib  = sum(b for c, n, b in ino.values() if c >= n) / 2**30
    shared_gib = sum(b for c, n, b in ino.values() if c < n) / 2**30
    print(f"  set-relative: FREED {freed_gib:.2f} GiB (every link inside the release); "
          f"{shared_gib:.2f} GiB shared with a kept link frees nothing")
    print(f"\n  universe {len(universe)} dirs | KEEP {len(KEEP)} = {keepkb/1048576:.2f} GiB"
          f" | RELEASE {len(tier1)} = {t1kb/1048576:.2f} GiB"
          f" | out-of-scope (untouched) {len(out_scope)}")
    byarm = collections.Counter()
    for d in tier1: byarm[universe[d]] += sz.get(d, 0)
    print(f"  --- RELEASE ---   {'arm':<24}{'dirs':>6}{'GiB':>9}{'cited':>7}")
    for a, kb in byarm.most_common(80):
        n = sum(1 for d in tier1 if universe[d] == a)
        print(f"                    {a:<24}{n:>6}{kb/1048576:>9.2f}{CIT.get(a,0):>7}")
    if len(byarm) > 80: print(f"                    ... and {len(byarm)-80} more families")
    tf = os.path.join(HERE, f"tier1_{tree}_{STAMP}{SUFFIX}.txt")
    with open(tf, "w") as fh:
        for d in tier1: fh.write(os.path.join(WORK, d) + "\n")
    print(f"  tier 1 file: {tf}  ({len(tier1)} lines, {t1kb/1048576:.2f} GiB)")
    kf = os.path.join(HERE, f"keep_{tree}_{STAMP}{SUFFIX}.txt")
    with open(kf, "w") as fh:
        for d in KEEP: fh.write(os.path.join(WORK, d) + "\n")
    return t1kb

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in TREES] or list(TREES)
    g1 = 0
    for t in want: g1 += plan_tree(t, TREES[t])
    print(f"\n{'='*78}\nGRAND TOTAL release {g1/1048576:.2f} GiB")
    print(f"interlock failures: {fails or 'NONE'}")
    print("\nThis script retired nothing.  Review the tier files, then run the "
          "retire driver (CONFIRM=yes).")
    sys.exit(1 if fails else 0)
