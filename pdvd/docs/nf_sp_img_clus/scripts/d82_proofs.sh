#!/usr/bin/env bash
# doc pdvd/82 -- the compiled-config proofs for the split_kink_min_deg flip in
# pdvd/wct-pr-perevt.jsonnet.  PRE_EVT = a copy of the file BEFORE the flip.
#   A  PRE + `-S stm_michel_extra={split_kink_min_deg:10.0}` (the arm's config) vs POST unset -> 0 lines
#   B  POST + `-S stm_michel_extra={split_kink_min_deg:15.0}` (the C++ default forced back) vs PRE
#      -> exactly the one key, present at 15 vs absent (a value knob has no key-suppression form)
#   C  PRE vs POST -> exactly the one key
#   D  the doc-82 knobs themselves are NOT in the shipped config (not flipped)
# Fork by duplication (CLAUDE.md M10) of d79_proofs.sh; only the key changed.
set -u
: "${PRE_EVT:?}"
D=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
W=/home/xqian/tmp/p82/proofs; mkdir -p $W
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
comp pre_on   "$PRE_EVT" -S 'stm_michel_extra={split_kink_min_deg:10.0}'
comp post     wct-pr-perevt.jsonnet
comp post_off wct-pr-perevt.jsonnet -S 'stm_michel_extra={split_kink_min_deg:15.0}'
echo "A  PRE + extra ON vs POST unset: $(diff <(pj pre_on) <(pj post) | grep -c '^[<>]') lines"
echo "B  POST + extra 15 vs PRE:"; diff <(pj pre) <(pj post_off) | grep '^[<>]'
echo "C  PRE vs POST:"; diff <(pj pre) <(pj post) | grep '^[<>]'
echo "   split_kink_min_deg in the compiled POST: $(grep -c '"split_kink_min_deg"' $W/post.json) key(s): $(grep -o '"split_kink_min_deg": [0-9.]*' $W/post.json | sort -u)"
echo "D  the doc-82 knobs are NOT shipped: stop_tail_peak_frac $(grep -c '"stop_tail_peak_frac"' $W/post.json), stop_tail_peak_kink_min_deg $(grep -c '"stop_tail_peak_kink_min_deg"' $W/post.json), michel_collinear_split $(grep -o '"michel_collinear_split": [a-z]*' $W/post.json | sort -u)"
echo PROOFS_DONE
