#!/usr/bin/env bash
# doc pdvd/88 -- the compiled-config proofs for the flip of doc 87's floored stop-local keep
# and doc 88's reachable stop snap in pdvd/wct-pr-perevt.jsonnet (four keys:
# stop_local_residual_cm, _min_points, _min_len_cm, stop_snap_reachable).
# PRE_EVT = a copy of the file BEFORE the flip; TLA = the arm's extra bag (p88v5fr's).
#   A  PRE + `-S stm_michel_extra={TLA}` (the arm's config) vs POST unset -> 0 lines
#   B  POST + `-S stm_michel_extra={stop_local_residual_cm:0.0}` (the radius forced back to its
#      C++ default) vs PRE -> the radius key (present 0 vs absent), the two floor keys and
#      stop_snap_reachable, all INERT at radius 0: the floors are read only inside
#      `if (m_stop_local_residual_cm > 0)`, and with no keep n_kept_near_stop_main is 0, which
#      is the reachable snap's gate (doc 88's p88vr: the knob alone == production)
#   C  PRE vs POST -> exactly the four keys
#   D  PDHD untouched: none of the keys is in pdhd/wct-pr-perevt.jsonnet
# Fork by duplication (CLAUDE.md M10) of doc 87's unrun d87_proofs.sh (itself a fork of
# d85_proofs.sh); the fourth key, W and the default TLA changed.
set -u
: "${PRE_EVT:?}"
TLA=${TLA:-"stop_local_residual_cm:5.0,stop_local_residual_min_points:5,stop_local_residual_min_len_cm:5.0,stop_snap_reachable:true"}
D=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
H=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd
W=/home/xqian/tmp/p88/proofs; mkdir -p $W
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
comp post_off wct-pr-perevt.jsonnet -S 'stm_michel_extra={stop_local_residual_cm:0.0}'
echo "A  PRE + extra ON vs POST unset: $(diff <(pj pre_on) <(pj post) | grep -c '^[<>]') lines"
echo "B  POST + radius 0 vs PRE:"; diff <(pj pre) <(pj post_off) | grep '^[<>]'
echo "C  PRE vs POST:"; diff <(pj pre) <(pj post) | grep '^[<>]'
echo "   keys in the compiled POST: $(grep -o '"stop_\(local_residual[a-z_]*\|snap_reachable\)" *: *[0-9a-z.]*' $W/post.json | sort -u | tr '\n' ' ')"
echo "D  PDHD: stop_local_residual / stop_snap_reachable in pdhd/wct-pr-perevt.jsonnet: $(grep -c 'stop_local_residual\|stop_snap_reachable' $H/wct-pr-perevt.jsonnet) line(s)"
echo PROOFS_DONE
