#!/usr/bin/env bash
# doc pdvd/96 -- the compiled-config proofs for flipping the region's SCOPE into PDVD production.
# ONE new key, not present in the compiled config before:
#   michel_q2d_region_scope  0 -> <SCOPE>   (0 = off / every cell in radius; C++ default 0)
#
# The other four michel_q2d keys are ALREADY in production from doc pdvd/95 and are unchanged
# here, so proof C must show exactly one added line.  If it shows more, this round has moved a
# key it did not intend to and the flip does not stand.
#
# On proof A's FORM.  The house form is "PRE + the MEASURED arm's TLA vs POST unset -> 0 lines".
# As in doc 95 that form is only partly available, and for the same reason plus one more:
#   * the measurement arm ran michel_q2d_region_cm = 40 (the DIAGNOSTIC radius, wide so every
#     scope could be swept offline from one arm) while production carries the tight R = 10;
#   * and if the pre-registered rule selects scope 2, the arm's scope 1 differs from production's
#     scope as well.
# So proof A proves only that the file compiles to PRODUCTION's key set.  The remaining claim --
# that those keys are the reading this doc measured -- is closed by d96_confirm.py, which predicts
# production's scalars from the measurement arm's own cell table (exact arithmetic, no tolerance)
# and checks the confirmation arm reproduces them item by item.  Do not relabel this line.
#
#   A  PRE + production's key set vs POST unset -> 0 lines
#   B  POST + every key forced back to its C++ initializer vs PRE -> one line per key, each a key
#      present at its C++ default vs absent.  The inert-key form of docs 58 / 61 / 82 / 93 / 95.
#   C  PRE vs POST -> exactly the ONE key this round sets
#   D  PDHD untouched: pdhd/wct-pr-perevt.jsonnet unchanged against git HEAD and carrying no
#      michel_q2d key at all.  PDHD stays OFF -- doc pdvd/81 sec 8a measured 45 % of its Michels
#      leaning on the cross-shared fitted substitution against 22 % on PDVD, and there is still no
#      owner hand-scan of this chain there.
#
# Fork by duplication (CLAUDE.md M10) of d95_proofs.sh; the key set and the TLA changed.
set -u
: "${PRE_EVT:?set PRE_EVT to the saved pre-flip wct-pr-perevt.jsonnet}"
SCOPE=${SCOPE:?set SCOPE to the scope being flipped, e.g. 1}
R=${R:-10.0}
CTL=${CTL:-35.0}
TLA=${TLA:-"michel_q2d:true,michel_q2d_cells:true,michel_q2d_region_cm:$R,michel_q2d_region_ctl_cm:$CTL,michel_q2d_region_scope:$SCOPE"}
OFF=${OFF:-"michel_q2d:false,michel_q2d_cells:false,michel_q2d_region_cm:0.0,michel_q2d_region_ctl_cm:-1.0,michel_q2d_region_scope:0"}
D=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
H=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd
SRC=/home/xqian/toolkit-dev/toolkit/clus/src/CheckSTM_Michel.cxx
W=/home/xqian/tmp/p96/proofs; mkdir -p $W
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
echo "A  PRE + production's key set vs POST unset: $(diff <(pj pre_on) <(pj post) | grep -c '^[<>]') lines  (0 == the flipped file compiles to exactly those keys; d96_confirm.py closes the 'and they were measured' half)"
echo "B  POST + every key forced back vs PRE:"; diff <(pj pre) <(pj post_off) | grep '^[<>]'
echo "   C++ initializers:"; grep -o 'm_michel_q2d[a-z_0-9]*{[^}]*}' $SRC | sort -u | sed 's/^/     /'
echo "C  PRE vs POST:"; diff <(pj pre) <(pj post) | grep '^[<>]'
echo "   (this round sets ONE key -- more than one added line here is a defect, not a pass)"
echo "   keys in the compiled POST: $(grep -o '"michel_q2d[a-z_0-9]*" *: *[a-z0-9.-]*' $W/post.json | sort -u | tr '\n' ' ')"
git -C $H/.. diff --quiet HEAD -- pdhd/wct-pr-perevt.jsonnet; echo "D  PDHD file vs git HEAD: $([ $? = 0 ] && echo unchanged || echo CHANGED)" \
    "| michel_q2d lines in PDHD: $(grep -c 'michel_q2d' $H/wct-pr-perevt.jsonnet)"
echo PROOFS_DONE
