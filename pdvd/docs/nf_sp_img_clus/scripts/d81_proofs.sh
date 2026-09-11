#!/usr/bin/env bash
# doc pdvd/81 -- the compiled-config proofs for the michel_q2d / michel_q2d_cells
# flip in <det>/wct-pr-perevt.jsonnet.  DET=pdvd|pdhd; PRE_EVT = a copy of the
# file BEFORE the flip.
#   A  PRE + `-S stm_michel_extra={michel_q2d:true,michel_q2d_cells:true}` (the ON arm's config)
#      vs POST unset                                                            -> 0 lines
#   B  POST + `-S stm_michel_extra={michel_q2d:false,michel_q2d_cells:false}` vs PRE
#      -> exactly the two keys, present at false vs absent (a bag value has no key-suppression
#         form; the C++ default is false, so the compiled OFF behaviour is the PRE behaviour)
#   C  PRE vs POST                                                              -> exactly the two keys
# Fork of d79_proofs.sh (the compile command, PDVD's 13-stage production pipeline; PDHD's
# pipeline_names taken from its own file).
set -u
: "${PRE_EVT:?}"; DET=${DET:-pdvd}
D=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/$DET
W=/home/xqian/tmp/p81/proofs_$DET; mkdir -p $W
# both runners' PIPE_NU (pdvd/run_pr_evt.sh:130, pdhd/run_pr_evt.sh:129) are this same 13-stage list
PJ='["switch_scope","flag_mains","unmerge_assoc","steiner","fiducialutils","tagger_check_tgm","tagger_check_stm","tagger_check_fc","protect_bundle","steiner_refresh","check_stm_michel","tracking_visitor","pr_display"]'
RUN=39252; [ "$DET" = pdhd ] && RUN=29107
comp() {   # out src [extra-args...]
    local out=$1 src=$2; shift 2
    rm -f $W/$out.json
    (cd $D && wcsonnet -P $D -A input=x.tar.gz -A output_dir=$W -S run=$RUN -S subrun=0 -S event=1 \
        -S pipeline_names="$PJ" "$@" -o $W/$out.json "$src") > $W/$out.log 2>&1
    local rc=$?
    echo "  compile $out rc=$rc size=$(stat -c %s $W/$out.json 2>/dev/null)"
    if [ $rc != 0 ] || [ ! -s $W/$out.json ]; then echo "ABORT: $out did not compile (see $W/$out.log)"; exit 2; fi
}
pj() { python3 -m json.tool --sort-keys $W/$1.json; }
comp pre      "$PRE_EVT"
comp pre_on   "$PRE_EVT" -S 'stm_michel_extra={michel_q2d:true,michel_q2d_cells:true}'
comp post     wct-pr-perevt.jsonnet
comp post_off wct-pr-perevt.jsonnet -S 'stm_michel_extra={michel_q2d:false,michel_q2d_cells:false}'
echo "A  PRE + extra ON vs POST unset: $(diff <(pj pre_on) <(pj post) | grep -c '^[<>]') lines"
echo "B  POST + extra false vs PRE:"; diff <(pj pre) <(pj post_off) | grep '^[<>]'
echo "C  PRE vs POST:"; diff <(pj pre) <(pj post) | grep '^[<>]'
echo "   michel_q2d keys in the compiled POST: $(grep -o '"michel_q2d[a-z_]*": [a-z]*' $W/post.json | sort | uniq -c | tr '\n' ' ')"
echo PROOFS_DONE
