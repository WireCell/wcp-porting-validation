#!/usr/bin/env bash
# doc pdvd/70 sec 10 -- gates and scoring for the P1 arms (run after run_arms.sh).
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/d71
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
cd $IMG

echo "=== 0. completeness: events with both tracking files, per arm"
for a in d71vleg d71voff d71vp1 d71vsp d71hleg d71hoff; do
    d=pdvd; [ "${a:3:1}" = h ] && d=pdhd
    n=$(ls -d $d/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in $d/work/*_$a; do [ -s $e/tracking-stm.root ] && [ -s $e/mabc-pr.zip ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" $d/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  $a dirs $n complete $ok loader-deaths $lt"
done

echo "=== 1. OFF gate: mabc-pr.zip member content + calib-pr json md5, leg (base pin) vs off (P1 pin, knob off)"
for pair in "pdvd d71vleg d71voff" "pdhd d71hleg d71hoff"; do
    set -- $pair; d=$1; A=$2; B=$3
    same=0; diff=0; miss=0; cs=0; cd=0
    for ea in $d/work/*_$A; do
        pre=${ea%_$A}; eb=${pre}_$B
        [ -s $ea/mabc-pr.zip ] && [ -s $eb/mabc-pr.zip ] || { miss=$((miss+1)); continue; }
        ha=$(python3 abtest/hash_archive.py $ea/mabc-pr.zip | cut -d' ' -f1)
        hb=$(python3 abtest/hash_archive.py $eb/mabc-pr.zip | cut -d' ' -f1)
        [ "$ha" = "$hb" ] && same=$((same+1)) || { diff=$((diff+1)); echo "    zip DIFF $(basename $pre)"; }
        ja=$(md5sum $ea/calib-pr-evt*.json | cut -c1-32); jb=$(md5sum $eb/calib-pr-evt*.json | cut -c1-32)
        [ "$ja" = "$jb" ] && cs=$((cs+1)) || { cd=$((cd+1)); echo "    calib DIFF $(basename $pre)"; }
    done
    echo "  $d $A vs $B: mabc-pr.zip same $same diff $diff missing $miss | calib-pr json same $cs diff $cd"
done

echo "=== 2. OFF gate: every T_stm_michel branch (d51g_branch_census)"
python3 $X/d51g_branch_census.py --before "pdvd/work/*_d71vleg" --after "pdvd/work/*_d71voff" \
    --before-arm d71vleg --after-arm d71voff --pts --out $W/g_pdvd > $W/g_pdvd.txt 2>&1; echo "  pdvd rc=$?"
python3 $X/d51g_branch_census.py --before "pdhd/work/*_d71hleg" --after "pdhd/work/*_d71hoff" \
    --before-arm d71hleg --after-arm d71hoff --pts --out $W/g_pdhd > $W/g_pdhd.txt 2>&1; echo "  pdhd rc=$?"

echo "=== 3. prep the PDVD arms"
cd $IMG/pdhd/stm_michel_scan
for a in d71vleg d71voff d71vp1 d71vsp; do
    ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  prep $a rc=$? payloads $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l)"
done
cd $IMG

echo "=== 4. does the doc 70 prediction transfer? d68a3 (old pin) vs d71vleg (merged binary), verdicts"
python3 - <<'EOF'
import glob, json, os
def load(p):
    out = {}
    for fn in glob.glob(os.path.join(p, "smprep-*.json")):
        ev, c = os.path.basename(fn)[7:-5].rsplit("-c", 1)
        out["%s/%s" % (ev, c)] = json.load(open(fn))["verdict"]
    return out
A, B = load("/home/xqian/tmp/d68/prep_d68a3"), load("/home/xqian/tmp/d71/prep_d71vleg")
common = set(A) & set(B)
moved = sorted(k for k in common if A[k] != B[k])
print("  d68a3 %d, d71vleg %d, common %d, only-d68a3 %d, only-d71vleg %d, verdict dicts differing %d"
      % (len(A), len(B), len(common), len(set(A) - set(B)), len(set(B) - set(A)), len(moved)))
for k in moved[:10]:
    print("   ", k, sorted(f for f in set(A[k]) | set(B[k]) if A[k].get(f) != B[k].get(f))[:8])
EOF

echo "=== 5. P1 C++ vs the offline rule, item by item"
python3 $X/d70_p1_check.py --leg $W/prep_d71vleg --on $W/prep_d71vp1 > $W/p1_check.txt 2>&1; echo "  p1 rc=$?"; tail -3 $W/p1_check.txt
python3 $X/d70_p1_check.py --leg $W/prep_d71vleg --on $W/prep_d71vsp --sparse > $W/sp_check.txt 2>&1; echo "  sparse rc=$?"; tail -3 $W/sp_check.txt
python3 $X/d70_p1_check.py --leg $W/prep_d71vleg --on $W/prep_d71voff --off > $W/off_check.txt 2>&1; echo "  off (rule never fires) rc=$?"; tail -2 $W/off_check.txt

echo "=== 6. census on the smx1a+smx3+smx4 record"
cd $IMG/pdhd/stm_michel_scan
for a in d71vleg d71vp1 d71vsp; do
    STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json \
        python3 census_score.py --prep $W/prep_$a --baseline $W/prep_d71vleg --arm $a > $W/score_$a.txt 2>&1
    echo "  census $a rc=$?"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' $W/score_$a.txt | grep census
done
python3 census_score.py --check > $W/check.txt 2>&1; echo "  --check rc=$?: $(tail -1 $W/check.txt)"
echo GATES_DONE
