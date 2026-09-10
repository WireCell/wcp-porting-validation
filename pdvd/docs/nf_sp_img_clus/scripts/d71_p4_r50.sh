#!/usr/bin/env bash
# doc pdvd/71 sec 11 -- the owner's 50 cm radius for P4.
# Run AFTER pdvd/wct-pr-perevt.jsonnet carries michel_gamma_radius_cm: 50.0 and
# with no other PDVD job reading that file: the arm is BARE production (no TLA),
# so it computes exactly what production now computes.  Then prep, census, the
# movers against p4v35 (production before this change, 35 cm) by name, and the
# P4 grading next to the 35 and 60 cm arms of sec 6-8.
#   SKIP_ARM=1 re-runs only the analysis.
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p4
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
cd $IMG

if [ "${SKIP_ARM:-0}" != 1 ]; then
    LOGD=$W/arm_p4v50 ARM=p4v50 DET=pdvd SRC=d16vnu JOBS=${JOBS:-24} PIN=$W/libpin_p4 \
        $X/d53_run_arms.sh > $W/p4v50.out 2>&1
    echo "arm rc=$? (1 = the no-candidate event 039252_11 read as incomplete, as on every arm)"
fi
grep -h 'md5\|complete' $W/p4v50.out

echo "=== 0. completeness and loader deaths"
n=$(ls -d pdvd/work/*_p4v50 | wc -l)
ok=$(for e in pdvd/work/*_p4v50; do [ -s $e/tracking-stm.root ] && [ -s $e/mabc-pr.zip ] && echo 1; done | wc -l)
echo "  p4v50 dirs $n complete $ok loader-deaths $(grep -l 'file too short' pdvd/work/*_p4v50/wct_pr_*.log 2>/dev/null | wc -l)"

echo "=== 1. mabc-pr.zip / calib vs p4v35 (production before this change): the admission widening moves them"
same=0; diff=0
for ea in pdvd/work/*_p4v35; do
    pre=${ea%_p4v35}; eb=${pre}_p4v50
    [ "$(python3 abtest/hash_archive.py $ea/mabc-pr.zip | cut -d' ' -f1)" = "$(python3 abtest/hash_archive.py $eb/mabc-pr.zip | cut -d' ' -f1)" ] \
        && same=$((same+1)) || diff=$((diff+1))
done
echo "  mabc-pr.zip same $same diff $diff"

echo "=== 2. every T_stm_michel branch, p4v35 -> p4v50"
python3 $X/d51g_branch_census.py --before "pdvd/work/*_p4v35" --after "pdvd/work/*_p4v50" \
    --before-arm p4v35 --after-arm p4v50 --split-key n_michel_gammas --out $W/g_v50.txt > /dev/null 2>&1
echo "  rc=$?"; sed -n '1,40p' $W/g_v50.txt

echo "=== 3. prep + census on the smx1a+smx3+smx4 record"
cd $IMG/pdhd/stm_michel_scan
./prep_stm_michel_scan.py --det pdvd --arm p4v50 --outdir $W/prep_p4v50 --sheetdir $W/sheet_p4v50 \
    --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_p4v50.log 2>&1
echo "  prep rc=$? payloads $(ls $W/prep_p4v50/smprep-*.json | wc -l)"
STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_p4v50 --baseline $W/prep_p4v35 --arm p4v50 > $W/score_p4v50.txt 2>&1
echo "  census rc=$?"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' $W/score_p4v50.txt | grep census
cd $IMG

echo "=== 4. verdict movers by name, against the record: 35 -> 50 and 50 -> 60"
STM_SCAN_RECORD=$REC python3 - <<'EOF'
import sys
sys.path.insert(0, "pdhd/stm_michel_scan"); import census_lib as C
R = C.load_record()
A = {n: C.load_payloads("/home/xqian/tmp/p4/prep_" + n, R)[0] for n in ("p4v35", "p4v50", "p4v60")}
for a, b in (("p4v35", "p4v50"), ("p4v50", "p4v60")):
    keys = sorted(set(A[a]) & set(A[b]))
    print("  %s -> %s: common %d, only-%s %s, only-%s %s" % (a, b, len(keys), a, sorted(set(A[a]) - set(A[b])), b, sorted(set(A[b]) - set(A[a]))))
    for f in ("is_stm", "michel_found"):
        mv = [k for k in keys if A[a][k]["verdict"][f] != A[b][k]["verdict"][f]]
        print("    %s moves on %d: %s" % (f, len(mv), " ".join("%s[%s %d->%d]" % (k, R[k]["verdict"] if k in R else "not judged",
                                                                         A[a][k]["verdict"][f], A[b][k]["verdict"][f]) for k in mv)))
EOF

echo "=== 5. P4 grading at 35 / 50 / 60 cm"
STM_SCAN_RECORD=$REC python3 $X/d71_p4_score.py --off $W/prep_p4voff \
    --on p4v35=$W/prep_p4v35:35 --on p4v50=$W/prep_p4v50:50 --on p4v60=$W/prep_p4v60:60 \
    --json $W/score_p4_r50.json > $W/score_p4_r50.txt 2>&1
echo "  score rc=$?"; cat $W/score_p4_r50.txt

echo "=== 6. the record is untouched"
(cd $IMG/pdhd/stm_michel_scan && python3 census_score.py --check 2>&1 | tail -1)
echo R50_DONE
