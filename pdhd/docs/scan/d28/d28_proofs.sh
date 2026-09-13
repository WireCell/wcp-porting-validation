#!/usr/bin/env bash
# doc pdhd/28 -- compiled-config proofs for flipping michel_q2d_region_wire_lookup into PDHD production.
# Fork by duplication (CLAUDE.md M10) of ../d26/d26_proofs.sh; d26 untouched.  Compile-only through the production
# runner (PDHD_PR_COMPILE_ONLY=1 run_pr_evt.sh -nu -stm-fit, event 029107_17, pctree symlinks of 029107_17_d51hclus).
#
#   STAGE=pre  (BEFORE the edit)  h28cfg0 = PRE file, no TLA (= production h26q2dprod); h28cfgT = PRE file + tla_wl.txt
#                                 (= what arm h28wl ran); a copy of the PRE file is kept in /home/xqian/tmp/h28.
#   STAGE=post (AFTER the edit)   h28cfg = POST file, no TLA (= what h28prod runs); h28cfgoff = POST file with the key
#                                 forced back to its C++ initializer (false).
#     A  h28cfgT  -> h28cfg     0 / 0 / 0, and the WHOLE compiled config identical once the work tag is renamed
#     B  h28cfg0  -> h28cfgoff  exactly 1 key, at its C++ initializer
#     C  h28cfg0  -> h28cfg     exactly 1 key added, nothing removed or changed
#     D  PDVD's wct-pr-perevt.jsonnet unchanged against git HEAD
# Usage: STAGE=pre bash d28_proofs.sh > cfg_pre.txt 2>&1; (edit) STAGE=post bash d28_proofs.sh > cfg_proofs.txt 2>&1
set -u
I=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
K=$I/pdhd/docs/scan/h22/d22_cfg_keys.py
SRC=/home/xqian/toolkit-dev/toolkit/clus/src/CheckSTM_Michel.cxx
W=/home/xqian/tmp/h28
TLA=$(cat $I/pdhd/docs/scan/d28/tla_wl.txt)
OFF='-S stm_michel_extra={michel_q2d_region_wire_lookup:false}'
STAGE=${STAGE:?pre or post}
cd $I/pdhd
cfg() { echo "$I/pdhd/work/029107_17_$1/.wct-pr_$1.json"; }
stage_dir() {
    local d=work/029107_17_$1
    [ -e $d ] && { echo "ABORT: $d exists (new tags only)"; exit 3; }
    mkdir $d
    ln -s $I/pdhd/work/029107_17_d51hclus/pctree-evt1119.tar.gz $d/
    ln -s $I/pdhd/work/029107_17_d51hclus/pctree-evt1119.tlas $d/
}
if [ "$STAGE" = pre ]; then
    git -C $I diff --quiet -- pdhd/wct-pr-perevt.jsonnet || { echo "ABORT: pdhd/wct-pr-perevt.jsonnet already differs from HEAD"; exit 3; }
    cp $I/pdhd/wct-pr-perevt.jsonnet $W/pre_wct-pr-perevt.jsonnet
    stage_dir h28cfg0; stage_dir h28cfgT
    PDHD_PR_COMPILE_ONLY=1 ./run_pr_evt.sh -nu -stm-fit -s h28cfg0 29107 17 > $W/cfg0.log 2>&1; echo "compile h28cfg0 rc=$?"
    PDHD_PR_COMPILE_ONLY=1 PDHD_PR_TLA="$TLA" ./run_pr_evt.sh -nu -stm-fit -s h28cfgT 29107 17 > $W/cfgT.log 2>&1; echo "compile h28cfgT rc=$?"
    for t in h28cfg0 h28cfgT; do f=$(cfg $t); [ -s "$f" ] && echo "  $t $(stat -c %s $f) bytes" || { echo "ABORT: $f missing"; exit 2; }; done
    echo "  PRE file md5 $(md5sum $W/pre_wct-pr-perevt.jsonnet | cut -c1-12)"
    echo "--- h28cfg0 -> h28cfgT: want exactly the one key"
    python3 $K $(cfg h28cfg0) $(cfg h28cfgT)
    exit 0
fi
stage_dir h28cfg; stage_dir h28cfgoff
PDHD_PR_COMPILE_ONLY=1 ./run_pr_evt.sh -nu -stm-fit -s h28cfg 29107 17 > $W/cfg.log 2>&1; echo "compile h28cfg rc=$?"
PDHD_PR_COMPILE_ONLY=1 PDHD_PR_TLA="$OFF" ./run_pr_evt.sh -nu -stm-fit -s h28cfgoff 29107 17 > $W/cfgoff.log 2>&1; echo "compile h28cfgoff rc=$?"
for t in h28cfg0 h28cfgT h28cfg h28cfgoff; do
    f=$(cfg $t); [ -s "$f" ] || { echo "ABORT: $f missing"; exit 2; }
    echo "  $t $(stat -c %s $f) bytes, mtime $(stat -c %y $f | cut -c1-19)"
done
echo "  PRE file md5 $(md5sum $W/pre_wct-pr-perevt.jsonnet | cut -c1-12) | POST file md5 $(md5sum $I/pdhd/wct-pr-perevt.jsonnet | cut -c1-12)"
echo; echo "--- A: h28cfgT (arm h28wl's config) -> h28cfg (the flipped file): want 0/0/0"
python3 $K $(cfg h28cfgT) $(cfg h28cfg)
python3 - "$(cfg h28cfgT)" "$(cfg h28cfg)" <<'EOF'
import sys
a, b = open(sys.argv[1]).read(), open(sys.argv[2]).read()
b2 = b.replace("h28cfg", "h28cfgT")
print("--- whole compiled config, h28cfg with its work tag renamed h28cfg->h28cfgT, vs h28cfgT:")
print("  identical: %s | bytes %d %d | tag occurrences %d" % (a == b2, len(a), len(b), b.count("h28cfg")))
EOF
echo; echo "--- B: h28cfg0 (PRE) -> h28cfgoff (POST, key forced to its C++ initializer): want exactly 1 key, false"
python3 $K $(cfg h28cfg0) $(cfg h28cfgoff)
echo "  C++ initializer:"; grep -o 'm_michel_q2d_region_wire_lookup{[^}]*}' $SRC | sed 's/^/    /'
echo; echo "--- C: h28cfg0 (PRE) -> h28cfg (POST): want 1 added, 0 removed, 0 changed"
python3 $K $(cfg h28cfg0) $(cfg h28cfg)
echo; echo "--- D: PDVD production file vs git HEAD"
git -C $I diff --quiet -- pdvd/wct-pr-perevt.jsonnet && echo "  pdvd/wct-pr-perevt.jsonnet unchanged" || echo "  *** pdvd/wct-pr-perevt.jsonnet CHANGED ***"
echo; echo "--- the production diff (non-comment lines)"
git -C $I diff -U0 -- pdhd/wct-pr-perevt.jsonnet | grep '^[+-]' | grep -v '^[+-][+-]' | grep -v '^[+-]\s*//' || true
