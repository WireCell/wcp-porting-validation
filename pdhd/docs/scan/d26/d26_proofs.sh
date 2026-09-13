#!/usr/bin/env bash
# doc pdhd/26 part A -- the compiled-config proofs for flipping the region-based Michel energy into PDHD production.
# Compile-only through the production runner (PDHD_PR_COMPILE_ONLY=1 run_pr_evt.sh -nu -stm-fit, event 029107_17, the
# pctree symlinks of 029107_17_d51hclus), keys diffed with h22/d22_cfg_keys.py, exactly as doc pdhd/25 sec 9.
#
#   h27cfg0    PRE file (= production h26conf), no TLA          [compiled BEFORE the edit]
#   h27cfgT    PRE file + tla_q2d.txt  = what arm h26q2d ran    [compiled BEFORE the edit]
#   h27cfg     POST file, no TLA       = what h26q2dprod runs
#   h27cfgoff  POST file + every new key forced back to its C++ initializer
#
#   A  h27cfgT  -> h27cfg     0 / 0 / 0, and the WHOLE compiled config identical once the work tag is renamed
#   B  h27cfg0  -> h27cfgoff  exactly the 5 keys, each at its C++ initializer (inert-key form; the initializers
#                              are grepped from CheckSTM_Michel.cxx and printed beside it)
#   C  h27cfg0  -> h27cfg     exactly the 5 keys added, nothing removed or changed
#   D  PDVD's wct-pr-perevt.jsonnet unchanged against git HEAD
# Usage (after the edit): bash d26_proofs.sh > cfg_proofs.txt 2>&1; echo rc=$?
set -u
I=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
K=$I/pdhd/docs/scan/h22/d22_cfg_keys.py
SRC=/home/xqian/toolkit-dev/toolkit/clus/src/CheckSTM_Michel.cxx
OFF='-S stm_michel_extra={michel_q2d:false,michel_q2d_cells:false,michel_q2d_region_cm:0.0,michel_q2d_region_ctl_cm:-1.0,michel_q2d_region_scope:0}'
cd $I/pdhd
cfg() { echo "$I/pdhd/work/029107_17_$1/.wct-pr_$1.json"; }
for t in h27cfg h27cfgoff; do
    d=work/029107_17_$t
    [ -e $d ] && { echo "ABORT: $d exists (new tags only)"; exit 3; }
    mkdir $d
    ln -s $I/pdhd/work/029107_17_d51hclus/pctree-evt1119.tar.gz $d/
    ln -s $I/pdhd/work/029107_17_d51hclus/pctree-evt1119.tlas $d/
done
PDHD_PR_COMPILE_ONLY=1 ./run_pr_evt.sh -nu -stm-fit -s h27cfg 29107 17 > /home/xqian/tmp/h27/cfg.log 2>&1; echo "compile h27cfg rc=$?"
PDHD_PR_COMPILE_ONLY=1 PDHD_PR_TLA="$OFF" ./run_pr_evt.sh -nu -stm-fit -s h27cfgoff 29107 17 > /home/xqian/tmp/h27/cfgoff.log 2>&1; echo "compile h27cfgoff rc=$?"
for t in h27cfg0 h27cfgT h27cfg h27cfgoff; do
    f=$(cfg $t); [ -s "$f" ] || { echo "ABORT: $f missing"; exit 2; }
    echo "  $t $(stat -c %s $f) bytes, mtime $(stat -c %y $f | cut -c1-19)"
done
echo "  PRE file used for h27cfg0/h27cfgT: /home/xqian/tmp/h27/pre_wct-pr-perevt.jsonnet md5 $(md5sum /home/xqian/tmp/h27/pre_wct-pr-perevt.jsonnet | cut -c1-12) (= git HEAD's pdhd/wct-pr-perevt.jsonnet at the time)"
echo "  POST file md5 $(md5sum $I/pdhd/wct-pr-perevt.jsonnet | cut -c1-12)"

echo; echo "--- A: h27cfgT (arm h26q2d's measured config) -> h27cfg (the flipped file): want 0/0/0"
python3 $K $(cfg h27cfgT) $(cfg h27cfg)
python3 - "$(cfg h27cfgT)" "$(cfg h27cfg)" <<'EOF'
import sys
a, b = open(sys.argv[1]).read(), open(sys.argv[2]).read()
n = b.count("h27cfg\"") + b.count("h27cfg/") + b.count("_h27cfg")
b2 = b.replace("h27cfg", "h27cfgT")
print("--- whole compiled config, h27cfg with its work tag renamed h27cfg->h27cfgT, vs h27cfgT:")
print("  identical: %s | bytes %d %d | tag occurrences %d" % (a == b2, len(a), len(b), b.count("h27cfg")))
EOF
echo; echo "--- B: h27cfg0 (PRE) -> h27cfgoff (POST, keys forced to their C++ initializers): want exactly the 5 keys, at the initializers"
python3 $K $(cfg h27cfg0) $(cfg h27cfgoff)
echo "  C++ initializers:"; grep -o 'm_michel_q2d[a-z_0-9]*{[^}]*}' $SRC | sort -u | sed 's/^/    /'
echo; echo "--- C: h27cfg0 (PRE) -> h27cfg (POST): want 5 added, 0 removed, 0 changed"
python3 $K $(cfg h27cfg0) $(cfg h27cfg)
echo; echo "--- D: PDVD production file vs git HEAD"
git -C $I diff --quiet -- pdvd/wct-pr-perevt.jsonnet && echo "  pdvd/wct-pr-perevt.jsonnet unchanged" || echo "  *** pdvd/wct-pr-perevt.jsonnet CHANGED ***"
echo; echo "--- the production diff (non-comment lines)"
git -C $I diff -U0 -- pdhd/wct-pr-perevt.jsonnet | grep '^[+-]' | grep -v '^[+-][+-]' | grep -v '^[+-]\s*//' || true
