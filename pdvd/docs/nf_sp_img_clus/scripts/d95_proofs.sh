#!/usr/bin/env bash
# doc pdvd/95 -- the compiled-config proofs for flipping the region-based Michel charge into
# PDVD production.  Four NEW keys, none present in the compiled config before:
#   michel_q2d             false -> true   (the master switch; C++ default false)
#   michel_q2d_cells       false -> true   (the per-cell table; the owner asked for it in production)
#   michel_q2d_region_cm     0.0 -> <R>    (the region radius; 0 = off)
#   michel_q2d_region_ctl_cm -1.0 -> -1.0  (the body control: INERT, left at the C++ default)
#
# On the control key -- REVISED after the measurement (this replaces an earlier argument in this
# file that the control was a diagnostic not wanted in production).  The body control turned out
# to be the thing that makes the region energy a CHECKABLE number rather than just a number: a
# cluster whose control reads high is one the muon subtraction fits poorly everywhere, so its
# stop reading is inflated too.  It flags 13 % of found Michels and 35 % of through-going items,
# and it is what distinguished "the estimator is inflating here" from "the hand scan under-called
# a Michel" on the loud STM_ONLY items (doc 95 sec 4.6).  A consumer cannot recover it after the
# fact without re-running, so it ships.
#
# On proof A's FORM, which differs from doc pdvd/93's and must not be quietly relabelled.  The
# house form is "PRE + the MEASURED arm's TLA vs POST unset -> 0 lines".  Here the measurement arm
# ran michel_q2d_region_cm = 40 (the DIAGNOSTIC radius, deliberately wide so every radius could be
# swept offline from one arm) while production carries a tight radius.  The two key sets therefore
# differ in exactly one value, and proof A below proves only that the file compiles to
# PRODUCTION's keys.  The remaining claim -- that those keys were measured -- is closed by
# d95_confirm.py, which predicts the production radius's scalars from the wide arm's own cell
# table (exact: the offline sum reproduces the C++ on 596/596 candidates) and checks the
# confirmation arm reproduces them item by item.  Stated here so nobody reads proof A as carrying
# more than it does.
#
#   A  PRE + the arm's TLA (control forced off) vs POST unset -> 0 lines
#   B  POST + every key forced back to its C++ initializer vs PRE -> one line per key, each a key
#      present at its C++ default vs absent.  The inert-key form of docs 58 / 61 / 82 / 93.
#      The initializers are grepped from the source and printed, so the reader can see the
#      component reads the same number either way.
#   C  PRE vs POST -> exactly the keys we set
#   D  PDHD untouched: pdhd/wct-pr-perevt.jsonnet unchanged against git HEAD and carrying no
#      michel_q2d key at all.  PDHD stays OFF -- doc pdvd/81 sec 8a measured 45 % of its Michels
#      leaning on the cross-shared fitted substitution (bridged 0.82, charge-only 0.55 of the
#      all-measured reading) against 22 % / 1.000 on PDVD, and there is no owner hand-scan there.
#
# Fork by duplication (CLAUDE.md M10) of d93_proofs.sh; the keys and the TLA changed.
set -u
: "${PRE_EVT:?set PRE_EVT to the saved pre-flip wct-pr-perevt.jsonnet}"
R=${R:?set R to the region radius being flipped, e.g. 15.0}
CTL=${CTL:--1.0}
TLA=${TLA:-"michel_q2d:true,michel_q2d_cells:true,michel_q2d_region_cm:$R,michel_q2d_region_ctl_cm:$CTL"}
OFF=${OFF:-"michel_q2d:false,michel_q2d_cells:false,michel_q2d_region_cm:0.0,michel_q2d_region_ctl_cm:-1.0"}
D=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
H=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd
SRC=/home/xqian/toolkit-dev/toolkit/clus/src/CheckSTM_Michel.cxx
W=/home/xqian/tmp/p95/proofs; mkdir -p $W
PJ='["switch_scope","flag_mains","unmerge_assoc","steiner","fiducialutils","tagger_check_tgm","tagger_check_stm","tagger_check_fc","protect_bundle","steiner_refresh","check_stm_michel","tracking_visitor","pr_display"]'
comp() {   # out src [extra-args...]
    local out=$1 src=$2; shift 2
    rm -f $W/$out.json
    (cd $D && wcsonnet -P $D -A input=x.tar.gz -A output_dir=$W -S run=39252 -S subrun=0 -S event=1 \
        -S pipeline_names="$PJ" "$@" -o $W/$out.json "$src") > $W/$out.log 2>&1
    local rc=$?
    echo "  compile $out rc=$rc size=$(stat -c %s $W/$out.json 2>/dev/null)"
    if [ $rc != 0 ] || [ ! -s $W/$out.json ]; then echo "ABORT: $out did not compile (see $W/$out.log)"; exit 2; fi
}
pj() { python3 -m json.tool --sort-keys $W/$1.json; }
comp pre      "$PRE_EVT"
comp pre_on   "$PRE_EVT" -S "stm_michel_extra={$TLA}"
comp post     wct-pr-perevt.jsonnet
comp post_off wct-pr-perevt.jsonnet -S "stm_michel_extra={$OFF}"
# NOTE the wording, which differs from doc pdvd/93's on purpose.  There, proof A's TLA was the
# MEASURED arm's, so 0 lines meant "production runs exactly what the arm ran".  Here the
# measurement arm ran the wide diagnostic radius, so 0 lines means only "the file compiles to
# PRODUCTION's four keys" -- a claim about the config, not about the measurement.  The other
# half, that those keys are the reading this doc measured, is closed by d95_confirm.py against
# the wide arm's own cell table.  Do not relabel this line as the doc-93 form.
echo "A  PRE + production's key set vs POST unset: $(diff <(pj pre_on) <(pj post) | grep -c '^[<>]') lines  (0 == the flipped file compiles to exactly those keys; d95_confirm.py closes the 'and they were measured' half)"
echo "B  POST + every key forced back vs PRE:"; diff <(pj pre) <(pj post_off) | grep '^[<>]'
echo "   C++ initializers:"; grep -o 'm_michel_q2d[a-z_]*{[^}]*}' $SRC | sort -u | sed 's/^/     /'
echo "C  PRE vs POST:"; diff <(pj pre) <(pj post) | grep '^[<>]'
echo "   keys in the compiled POST: $(grep -o '"michel_q2d[a-z_]*" *: *[a-z0-9.-]*' $W/post.json | sort -u | tr '\n' ' ')"
git -C $H/.. diff --quiet HEAD -- pdhd/wct-pr-perevt.jsonnet; echo "D  PDHD file vs git HEAD: $([ $? = 0 ] && echo unchanged || echo CHANGED)" \
    "| michel_q2d lines in PDHD: $(grep -c 'michel_q2d' $H/wct-pr-perevt.jsonnet)"
echo PROOFS_DONE
