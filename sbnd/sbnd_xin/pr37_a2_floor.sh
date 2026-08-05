#!/bin/bash
# doc pr/37 sec.2.5 (A2) -- re-measure the valfast A/A' floor at toolkit HEAD
# 2457320d, and decide whether the T_tagger/T_kine gate can be promoted from
# INFORMATIONAL to HARD before the 24-knob population campaign.
#
# WHY.  valfast/README.md demotes gate 4 to INFORMATIONAL on a MEASURED A/A'
# taken 2026-08-02: nueCC48 came back 9/47 multiset-identical, mcp1k 536/572,
# with a named value flip (shw_sp_lol_1_v_angle 32.5 -> 131.8 on r1qlmc evt 16),
# and concludes "an exact comparison would false-DIFF ~half the events".
# doc pr/37 sec.6.2 measures the OPPOSITE at 2457320d on nueCC48 -- 48/48
# exact-order identical over every branch of every tree, twice (off48 vs off48b,
# repeatA vs repeatB).  Between those dates pr/36 F4 swapped four address-
# ordered float sums in NeutrinoTaggerNuE.cxx to index-ordered sets and
# tagger_ordered_segment_sets shipped ON.  The two measurements used DIFFERENT
# input roots (pinned work-nuecc48-nuf vs work-nuecc48-prod0803), so this is
# not a controlled pair -- hence this run rather than an inference.
#
# THREE ARMS, not two:
#   vf37a  setarch -R   } matched address layout  -> reproduces the README's
#   vf37b  setarch -R   }                            own experimental condition
#   vf37c  ASLR ON      -> vs vf37a is the CROSS-LAYOUT floor, which is the
#                          property the campaign actually needs: production
#                          arms are not run under setarch.
# A clean a-vs-b with a dirty a-vs-c would mean the instability is real but
# layout-keyed (M4), and gate 4 may be promoted only for setarch-pinned arms.
#
# PR_EXTRA_STAGES=pr_display appends PrDisplayDump so calib-pr-evt<ID>.json
# exists: it is the ONLY outlet for pr/36 F1's match_isFC (pr/36 sec.10.9), one
# of the 24 knobs the campaign must gate, and valfast/*.sh reference it nowhere.
# That stage is read-only -- doc pr/26's own gate is that an arm run with it
# hashes identically to one without -- and BOTH arms of every pair here carry
# it, so it cannot manufacture a difference.
#
# PR-tail mode on the pinned hubs: both arms share the ql_root byte-for-byte,
# which is what makes an archive diff attributable (valfast/README.md).
#
# Nothing is deleted and no existing arm is written into (CLAUDE.md M13); the
# three tags are new.  No build happens here -- libWireCellClus.so's mtime is
# printed before, between and after every arm, and any change voids the run
# rather than being reinterpreted (M1).
set -u
cd "$(dirname "$0")" || exit 2
LOGDIR=${LOGDIR:?set LOGDIR to a scratch dir}
LIB=/nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so

stamp() { echo "### $1 lib=$(ls --time-style=full-iso -la $LIB | awk '{print $6,$7}')  load=$(cut -d' ' -f1 /proc/loadavg)"; }

stamp "BEFORE-vf37a"
PR_EXTRA_STAGES=pr_display setarch x86_64 -R \
    ./valfast/run_valfast.sh vf37a -j 8 > "$LOGDIR/vf37a.log" 2>&1
echo "### vf37a rc=$?"

stamp "BETWEEN-a-b"
PR_EXTRA_STAGES=pr_display setarch x86_64 -R \
    ./valfast/run_valfast.sh vf37b -j 8 > "$LOGDIR/vf37b.log" 2>&1
echo "### vf37b rc=$?"

stamp "BETWEEN-b-c"
# no setarch: ASLR on, the cross-layout arm
PR_EXTRA_STAGES=pr_display \
    ./valfast/run_valfast.sh vf37c -j 8 > "$LOGDIR/vf37c.log" 2>&1
echo "### vf37c rc=$?"

stamp "AFTER-vf37c"
echo DONE
