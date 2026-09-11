#!/usr/bin/env bash
# doc pdvd/90 -- the compiled-config proofs for the flip of doc 89's 0-FP bundle in
# pdvd/wct-pr-perevt.jsonnet: topology_michel_ke_min 3.0 (a NEW key; C++ default 10, absent
# from the compiled config before) and plateau_mip_hi 1.6 -> 2.0 (a value edit).
# PRE_EVT = a copy of the file BEFORE the flip; TLA = the arm's extra bag (p90vb's).
#   A  PRE + `-S stm_michel_extra={TLA}` (the arm's config) vs POST unset -> 0 lines
#   B  POST + `-S stm_michel_extra={topology_michel_ke_min:10.0,plateau_mip_hi:1.6}` (both forced
#      back) vs PRE -> exactly one line: topology_michel_ke_min present at 10 vs absent.  That is
#      the inert-key form of docs 58 / 61 / 82: the value equals the C++ member initializer
#      (m_topology_michel_ke_min{10.0}, checked here), so the component reads the same number.
#   C  PRE vs POST -> exactly the two keys
#   D  PDHD untouched: pdhd/wct-pr-perevt.jsonnet unchanged against git HEAD, no
#      topology_michel_ke_min in it, its plateau_mip_hi still 1.6
# Fork by duplication (CLAUDE.md M10) of d88_proofs.sh; the keys, W and the default TLA changed.
set -u
: "${PRE_EVT:?}"
TLA=${TLA:-"topology_michel_ke_min:3.0,plateau_mip_hi:2.0"}
D=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
H=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd
W=/home/xqian/tmp/p90/proofs; mkdir -p $W
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
comp post_off wct-pr-perevt.jsonnet -S 'stm_michel_extra={topology_michel_ke_min:10.0,plateau_mip_hi:1.6}'
echo "A  PRE + extra ON vs POST unset: $(diff <(pj pre_on) <(pj post) | grep -c '^[<>]') lines"
echo "B  POST + both forced back vs PRE:"; diff <(pj pre) <(pj post_off) | grep '^[<>]'
echo "   C++ initializer: $(grep -o 'm_topology_michel_ke_min{[^}]*}' /home/xqian/toolkit-dev/toolkit/clus/src/CheckSTM_Michel.cxx)"
echo "C  PRE vs POST:"; diff <(pj pre) <(pj post) | grep '^[<>]'
echo "   keys in the compiled POST: $(grep -o '"\(topology_michel_ke_min\|plateau_mip_hi\)" *: *[0-9.]*' $W/post.json | sort -u | tr '\n' ' ')"
git -C $H/.. diff --quiet HEAD -- pdhd/wct-pr-perevt.jsonnet; echo "D  PDHD file vs git HEAD: $([ $? = 0 ] && echo unchanged || echo CHANGED)" \
    "| topology_michel_ke_min lines: $(grep -c 'topology_michel_ke_min' $H/wct-pr-perevt.jsonnet)" \
    "| plateau_mip_hi: $(grep -o 'plateau_mip_hi: *[0-9.]*' $H/wct-pr-perevt.jsonnet | tr '\n' ' ')"
echo PROOFS_DONE
