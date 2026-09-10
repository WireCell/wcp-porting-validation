#!/usr/bin/env bash
# doc pdvd/76 (P5) -- the compiled-config proofs for stm_trackfitting_config_file.
#   PRE_PR=<pre-change copy of cfg/pgrapher/experiment/protodunevd/pr.jsonnet>
#   PRE_EVT=<pre-change copy of pdvd/wct-pr-perevt.jsonnet>
# The post-change files are the ones in the tree.  Three proofs:
#   1 unset: the production perevt compile, pre vs post            -> 0 lines
#   1b unset: pr.jsonnet compiled directly through the same perevt with the pre copy on the
#      import path (-P, which wcsonnet searches before WIRECELL_PATH -- proof 0 checks that) vs the tree                                  -> 0 lines
#   2 set (-A stm_trackfitting_config=<abs>): pre+unset vs post+set differ by exactly the two
#      trackfitting_config_file values of tagger_check_stm and check_stm_michel; the
#      tagger_check_neutrino value is unchanged
set -u
: "${PRE_PR:?}" "${PRE_EVT:?}"
D=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
CFG=/nfs/data/1/xqian/toolkit-dev/toolkit/cfg
W=/home/xqian/tmp/p76/proofs; mkdir -p $W/pre_tree/pgrapher/experiment/protodunevd
PJ='["switch_scope","flag_mains","unmerge_assoc","steiner","fiducialutils","tagger_check_tgm","tagger_check_stm","tagger_check_fc","protect_bundle","steiner_refresh","check_stm_michel","tracking_visitor","pr_display"]'
TF=/home/xqian/tmp/p76/tf_same.json
# a shadow import tree holding ONLY the pre-change pr.jsonnet; everything else resolves to the real cfg
cp "$PRE_PR" $W/pre_tree/pgrapher/experiment/protodunevd/pr.jsonnet
comp() {   # out src pre(0/1) [extra-args...]
    local out=$1 src=$2 pre=$3; shift 3
    local jp=""; [ "$pre" = 1 ] && jp="-P $W/pre_tree"
    rm -f $W/$out.json
    (cd $D && wcsonnet $jp -P $D -A input=x.tar.gz -A output_dir=$W -S run=39252 -S subrun=0 -S event=1 \
        -S pipeline_names="$PJ" "$@" -o $W/$out.json "$src") > $W/$out.log 2>&1
    local rc=$?
    echo "  compile $out rc=$rc size=$(stat -c %s $W/$out.json 2>/dev/null)"
    if [ $rc != 0 ] || [ ! -s $W/$out.json ]; then echo "ABORT: $out did not compile (see $W/$out.log)"; exit 2; fi
}
pj() { python3 -m json.tool --sort-keys $W/$1.json; }
# 0 the shadow tree must actually win over WIRECELL_PATH: the post perevt passes the new
#   parameter, so compiling it against the PRE pr.jsonnet must FAIL ("has no parameter")
(cd $D && wcsonnet -P $W/pre_tree -P $D -A input=x.tar.gz -A output_dir=$W -S run=39252 -S subrun=0 -S event=1 \
    -S pipeline_names="$PJ" -o $W/shadow_check.json wct-pr-perevt.jsonnet) > $W/shadow_check.log 2>&1
rc=$?; echo "0  shadow check (post perevt vs PRE pr.jsonnet must fail): rc=$rc $(grep -o 'no parameter[^,]*\|has no parameter[^,]*\|Unknown parameter[^,]*' $W/shadow_check.log | head -1)"
[ $rc = 0 ] && { echo "ABORT: the -P shadow did not take effect; proofs 1/1b would be vacuous"; exit 2; }
comp pre_pre   "$PRE_EVT" 1                       # pre perevt + pre pr.jsonnet   (yesterday's world)
comp pre_post  "$PRE_EVT" 0                       # pre perevt + post pr.jsonnet  (pr.jsonnet's default alone)
comp post_post wct-pr-perevt.jsonnet 0            # post perevt + post pr.jsonnet (today's tree, TLA unset)
comp post_set  wct-pr-perevt.jsonnet 0 -A stm_trackfitting_config=$TF
echo "1  unset, pre vs post (perevt + pr.jsonnet): $(diff <(pj pre_pre) <(pj post_post) | grep -c '^[<>]') lines"
echo "1b unset, pr.jsonnet alone (pre perevt, pre vs post pr.jsonnet): $(diff <(pj pre_pre) <(pj pre_post) | grep -c '^[<>]') lines"
echo "2  set (-A stm_trackfitting_config=$TF) vs unset:"; diff <(pj post_post) <(pj post_set) | grep '^[<>]'
# 2b the production pipeline list has no tagger_check_neutrino; add it to show the third
#    consumer keeps the shared file when the STM file is set
PJN='["switch_scope","flag_mains","unmerge_assoc","steiner","fiducialutils","tagger_check_tgm","tagger_check_stm","tagger_check_fc","protect_bundle","steiner_refresh","check_stm_michel","tagger_check_neutrino","tracking_visitor","pr_display"]'
for v in unset set; do
    extra=""; [ $v = set ] && extra="-A stm_trackfitting_config=$TF"
    (cd $D && wcsonnet -P $D -A input=x.tar.gz -A output_dir=$W -S run=39252 -S subrun=0 -S event=1 \
        -S pipeline_names="$PJN" $extra -o $W/nu_$v.json wct-pr-perevt.jsonnet) > $W/nu_$v.log 2>&1
    echo "  compile nu_$v rc=$? size=$(stat -c %s $W/nu_$v.json 2>/dev/null)"
done
echo "2b with tagger_check_neutrino in the pipeline, set vs unset: $(diff <(pj nu_unset) <(pj nu_set) | grep -c '^[<>]') lines; trackfitting_config_file values when set:"
python3 - <<PY
import json
d=json.load(open("$W/nu_set.json"))
for n in d: 
    if isinstance(n,dict) and "data" in n and isinstance(n["data"],dict) and "trackfitting_config_file" in n["data"]: print("    %-40s %s" % (n.get("type","?")+":"+n.get("name",""), n["data"]["trackfitting_config_file"]))
PY
echo "   trackfitting_config_file values, set:";   grep -n '"trackfitting_config_file"' $W/post_set.json
echo "   trackfitting_config_file values, unset:"; grep -n '"trackfitting_config_file"' $W/post_post.json
echo PROOFS_DONE
