#!/usr/bin/env bash
# doc pdvd/71 -- gates and scoring for the P4 arms (run after d71_p4_arms.sh).
# Fork of d70_p1_gates.sh.  Output: /home/xqian/tmp/p4/gates.log (tee it).
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p4
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
cd $IMG

echo "=== 0. completeness: events with both tracking files, per arm; loader deaths; pins"
for a in p4vleg p4voff p4v35 p4v60 p4hleg p4hoff; do
    d=pdvd; [ "${a:2:1}" = h ] && d=pdhd
    n=$(ls -d $d/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in $d/work/*_$a; do [ -s $e/tracking-stm.root ] && [ -s $e/mabc-pr.zip ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" $d/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  $a dirs $n complete $ok loader-deaths $lt | $(grep -h 'md5' $W/$a.out | tr '\n' ' ')"
done
echo "  pins now: $(md5sum /home/xqian/tmp/d71/libpin_p1/libWireCellClus.so | cut -c1-12) (P1)  $(md5sum $W/libpin_p4/libWireCellClus.so | cut -c1-12) (P4)"

echo "=== 1. mabc-pr.zip member content + calib-pr json md5"
echo "    (OFF gates: leg (P1 pin) vs off (P4 pin, knob off); NEUTRALITY: off vs v35; EVIDENCE: off vs v60)"
for pair in "pdvd p4vleg p4voff" "pdhd p4hleg p4hoff" "pdvd p4voff p4v35" "pdvd p4voff p4v60"; do
    set -- $pair; d=$1; A=$2; Bb=$3
    same=0; diff=0; miss=0; cs=0; cd=0; cn=0
    for ea in $d/work/*_$A; do
        pre=${ea%_$A}; eb=${pre}_$Bb
        [ -s $ea/mabc-pr.zip ] && [ -s $eb/mabc-pr.zip ] || { miss=$((miss+1)); continue; }
        ha=$(python3 abtest/hash_archive.py $ea/mabc-pr.zip | cut -d' ' -f1)
        hb=$(python3 abtest/hash_archive.py $eb/mabc-pr.zip | cut -d' ' -f1)
        [ "$ha" = "$hb" ] && same=$((same+1)) || { diff=$((diff+1)); [ $diff -le 5 ] && echo "    zip DIFF $(basename $pre)"; }
        ja=$(ls $ea/calib-pr-evt*.json 2>/dev/null | head -1); jb=$(ls $eb/calib-pr-evt*.json 2>/dev/null | head -1)
        if [ -z "$ja" ] && [ -z "$jb" ]; then cn=$((cn+1)); continue; fi
        [ -n "$ja" ] && [ -n "$jb" ] && [ "$(md5sum < $ja)" = "$(md5sum < $jb)" ] && cs=$((cs+1)) || { cd=$((cd+1)); [ $cd -le 5 ] && echo "    calib DIFF $(basename $pre)"; }
    done
    echo "  $d $A vs $Bb: mabc-pr.zip same $same diff $diff missing $miss | calib-pr json same $cs diff $cd absent-on-both $cn"
done

echo "=== 2. every T_stm_michel branch (d51g_branch_census): OFF gates with --pts; off -> v35 and off -> v60 scalars"
python3 $X/d51g_branch_census.py --before "pdvd/work/*_p4vleg" --after "pdvd/work/*_p4voff" \
    --before-arm p4vleg --after-arm p4voff --pts --out $W/g_pdvd.txt > /dev/null 2>&1; echo "  pdvd OFF rc=$?"; grep -E 'matched|BIT-IDENTICAL|NO shared|moved|FLIPS|IDENTICAL point|role labels|geometry changed|NEW|DROPPED' $W/g_pdvd.txt
python3 $X/d51g_branch_census.py --before "pdhd/work/*_p4hleg" --after "pdhd/work/*_p4hoff" \
    --before-arm p4hleg --after-arm p4hoff --pts --out $W/g_pdhd.txt > /dev/null 2>&1; echo "  pdhd OFF rc=$?"; grep -E 'matched|BIT-IDENTICAL|NO shared|moved|FLIPS|IDENTICAL point|role labels|geometry changed|NEW|DROPPED' $W/g_pdhd.txt
python3 $X/d51g_branch_census.py --before "pdvd/work/*_p4voff" --after "pdvd/work/*_p4v35" \
    --before-arm p4voff --after-arm p4v35 --split-key n_michel_gammas --out $W/g_v35.txt > /dev/null 2>&1; echo "  off -> v35 rc=$?"; grep -E 'matched|BIT-IDENTICAL|NO shared|FLIPS|NEW|DROPPED' $W/g_v35.txt
python3 $X/d51g_branch_census.py --before "pdvd/work/*_p4voff" --after "pdvd/work/*_p4v60" \
    --before-arm p4voff --after-arm p4v60 --split-key n_michel_gammas --out $W/g_v60.txt > /dev/null 2>&1; echo "  off -> v60 rc=$?"; cat $W/g_v60.txt | sed -n '1,60p'

echo "=== 3. NEUTRALITY of T_stm_michel_pts, off -> v35: every row of role != 4 identical; role 4 only added"
python3 - <<'EOF'
import glob, os, collections, uproot
def load(arm):
    out = {}
    for d in sorted(glob.glob("pdvd/work/*_" + arm)):
        fp = os.path.join(d, "tracking-pr.root")
        if not os.path.exists(fp): continue
        f = uproot.open(fp)
        if "T_stm_michel_pts" not in f: continue   # no candidate in the event (039252_11), on both arms
        t = f["T_stm_michel_pts"].arrays(library="np")
        ev = os.path.basename(d)[: -len(arm) - 1]
        for i in range(len(t["cluster_id"])):
            k = (ev, int(t["cluster_id"][i]))
            out.setdefault(k, []).append(tuple(float(t[c][i]) for c in ("role", "seg_id", "x", "y", "z", "q", "L", "rr", "q_sup")))
    return out
A, B = load("p4voff"), load("p4v35")
keys = sorted(set(A) | set(B))
bad, n4, c4, a4 = [], 0, 0, 0
for k in keys:
    ra = collections.Counter(r for r in A.get(k, []) if r[0] != 4)
    rb = collections.Counter(r for r in B.get(k, []) if r[0] != 4)
    a4 += sum(1 for r in A.get(k, []) if r[0] == 4)
    m = sum(1 for r in B.get(k, []) if r[0] == 4)
    n4 += m; c4 += m > 0
    if ra != rb: bad.append(k)
print("  candidates with rows: off %d, v35 %d; role != 4 rows identical on %d / %d; differing: %s"
      % (len(A), len(B), len(keys) - len(bad), len(keys), bad[:10] or "none"))
print("  role-4 rows: off %d, v35 %d on %d candidates" % (a4, n4, c4))
EOF

echo "=== 4. prep the PDVD arms (bare production; the pin tranche sheet as before)"
cd $IMG/pdhd/stm_michel_scan
for a in p4vleg p4voff p4v35 p4v60; do
    ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  prep $a rc=$? payloads $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l)"
done

echo "=== 5. census on the smx1a+smx3+smx4 record (baseline p4voff)"
for a in p4vleg p4voff p4v35 p4v60; do
    STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$a --baseline $W/prep_p4voff --arm $a > $W/score_$a.txt 2>&1
    echo "  census $a rc=$?"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' $W/score_$a.txt | grep census
done
cd $IMG

echo "=== 5b. the production census reconciled: p4vleg (bare production, 35 cm admission) vs d71vsp (survey arm, 60 cm)"
STM_SCAN_RECORD=$REC python3 - <<'EOF'
import sys, collections
sys.path.insert(0, "pdhd/stm_michel_scan"); import census_lib as C
R = C.load_record()
A, _ = C.load_payloads("/home/xqian/tmp/d71/prep_d71vsp", R)
B, _ = C.load_payloads("/home/xqian/tmp/p4/prep_p4vleg", R)
keys = sorted(set(A) & set(B))
print("  payloads d71vsp %d, p4vleg %d, common %d, only-d71vsp %s, only-p4vleg %s"
      % (len(A), len(B), len(keys), sorted(set(A) - set(B))[:10], sorted(set(B) - set(A))[:10]))
for f in ("is_stm", "michel_found"):
    mv = [k for k in keys if A[k]["verdict"][f] != B[k]["verdict"][f]]
    print("  %s differs on %d: %s" % (f, len(mv), " ".join("%s[%s %d->%d]" % (k, R[k]["verdict"] if k in R else "-", A[k]["verdict"][f], B[k]["verdict"][f]) for k in mv)))
EOF

echo "=== 6. P4 grading"
STM_SCAN_RECORD=$REC python3 $X/d71_p4_score.py --off $W/prep_p4voff \
    --on p4v35=$W/prep_p4v35:35 --on p4v60=$W/prep_p4v60:60 --json $W/score_p4.json > $W/score_p4.txt 2>&1
echo "  score rc=$?"; cat $W/score_p4.txt

echo "=== 7. the record is untouched"
(cd $IMG/pdhd/stm_michel_scan && python3 census_score.py --check > $W/check.txt 2>&1; echo "  --check rc=$?: $(tail -1 $W/check.txt)")
echo GATES_DONE
