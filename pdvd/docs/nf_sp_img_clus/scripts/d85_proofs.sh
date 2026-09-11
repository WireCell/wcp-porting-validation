#!/usr/bin/env bash
# doc pdvd/85 -- the compiled-config proofs for the stop_gamma_require_stm flip in
# pdvd/wct-pr-perevt.jsonnet.  PRE_EVT = a copy of the file BEFORE the flip; VAL = the
# flipped value, true (the arm it came from ran `-S stm_michel_extra={stop_gamma_require_stm:true}`).
#   A  PRE + `-S stm_michel_extra={stop_gamma_require_stm:VAL}` (the arm's config) vs POST unset -> 0 lines
#   B  POST + `-S stm_michel_extra={stop_gamma_require_stm:false}` (the C++ default forced back) vs PRE
#      -> exactly the one key, present false vs absent
#   C  PRE vs POST -> exactly the one key
#   D  PDHD untouched: the key is not in the compiled PDHD config
# Fork by duplication (CLAUDE.md M10) of d84_proofs.sh; only the key, the off value and W changed.
set -u
: "${PRE_EVT:?}"; : "${VAL:?}"
D=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
H=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd
W=/home/xqian/tmp/p85/proofs; mkdir -p $W
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
comp pre_on   "$PRE_EVT" -S "stm_michel_extra={stop_gamma_require_stm:$VAL}"
comp post     wct-pr-perevt.jsonnet
comp post_off wct-pr-perevt.jsonnet -S 'stm_michel_extra={stop_gamma_require_stm:false}'
echo "A  PRE + extra ON vs POST unset: $(diff <(pj pre_on) <(pj post) | grep -c '^[<>]') lines"
echo "B  POST + extra 0 vs PRE:"; diff <(pj pre) <(pj post_off) | grep '^[<>]'
echo "C  PRE vs POST:"; diff <(pj pre) <(pj post) | grep '^[<>]'
echo "   stop_gamma_require_stm in the compiled POST: $(grep -c '"stop_gamma_require_stm"' $W/post.json) key(s): $(grep -o '"stop_gamma_require_stm" *: *[a-z]*' $W/post.json | sort -u)"
echo "D  PDHD: stop_gamma_require_stm in pdhd/wct-pr-perevt.jsonnet: $(grep -c stop_gamma_require_stm $H/wct-pr-perevt.jsonnet) line(s)"
echo PROOFS_DONE
