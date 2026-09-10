#!/usr/bin/env bash
# docs pdvd/72 (P3b) + pdvd/73 (P2) -- gates and grading for the arms of
# d72_arms.sh.  Fork of d71_p4_gates.sh.  Output: tee /home/xqian/tmp/p72/gates.log.
#   EXTRA_ARMS="p72v2c ..."   wave-2 PDVD arms, compared with p72voff and prepped
#   SCORE_EXTRA="--on p72v2c=/home/xqian/tmp/p72/prep_p72v2c:kmin=60,kw=5"   their d72_score.py settings
#   SKIP_PREP=1               reuse existing preps
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p72
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
cd $IMG
VA="p72vleg p72voff p72vb60 p72vb90 p72v2ab ${EXTRA_ARMS:-}"

echo "=== 0. completeness: events with both tracking files, per arm; loader deaths; pins"
for a in $VA p72hleg p72hoff; do
    d=pdvd; [ "${a:3:1}" = h ] && d=pdhd
    n=$(ls -d $d/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in $d/work/*_$a; do [ -s $e/tracking-stm.root ] && [ -s $e/mabc-pr.zip ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" $d/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  $a dirs $n complete $ok loader-deaths $lt | $(grep -h 'md5' $W/$a.out 2>/dev/null | tr '\n' ' ')"
done
echo "  pins now: $(md5sum /home/xqian/tmp/p4/libpin_p4/libWireCellClus.so | cut -c1-8) (P4, expect 1af88622)  $(md5sum $W/libpin_p72/libWireCellClus.so | cut -c1-8) (P72, expect 8aa45329)"

echo "=== 1. mabc-pr.zip member content + calib-pr json md5"
echo "    (OFF: leg (P4 pin) vs off (P72 pin); STALE-BASELINE: p4v50 vs p72vleg; P3b NEUTRALITY: off vs vb60 (predicted identical);"
echo "     PLATEAU: vb60 vs vb90 (predicted identical); P2: off vs the P2 arms)"
PAIRS=("pdvd p72vleg p72voff" "pdhd p72hleg p72hoff" "pdvd p4v50 p72vleg" "pdvd p72voff p72vb60" "pdvd p72vb60 p72vb90" "pdvd p72voff p72v2ab")
for x in ${EXTRA_ARMS:-}; do PAIRS+=("pdvd p72voff $x"); done
for pair in "${PAIRS[@]}"; do
    read -r d A Bb <<< "$pair"
    same=0; diff=0; miss=0; cs=0; cdf=0; cn=0; dl=""
    for ea in $d/work/*_$A; do
        pre=${ea%_$A}; eb=${pre}_$Bb
        [ -s $ea/mabc-pr.zip ] && [ -s $eb/mabc-pr.zip ] || { miss=$((miss+1)); continue; }
        ha=$(python3 abtest/hash_archive.py $ea/mabc-pr.zip | cut -d' ' -f1)
        hb=$(python3 abtest/hash_archive.py $eb/mabc-pr.zip | cut -d' ' -f1)
        [ "$ha" = "$hb" ] && same=$((same+1)) || { diff=$((diff+1)); dl="$dl $(basename $pre)"; }
        ja=$(ls $ea/calib-pr-evt*.json 2>/dev/null | head -1); jb=$(ls $eb/calib-pr-evt*.json 2>/dev/null | head -1)
        if [ -z "$ja" ] && [ -z "$jb" ]; then cn=$((cn+1)); continue; fi
        [ -n "$ja" ] && [ -n "$jb" ] && [ "$(md5sum < $ja)" = "$(md5sum < $jb)" ] && cs=$((cs+1)) || cdf=$((cdf+1))
    done
    echo "  $d $A vs $Bb: mabc-pr.zip same $same diff $diff missing $miss | calib-pr json same $cs diff $cdf absent-on-both $cn"
    [ -n "$dl" ] && echo "    zip differs on:$(echo $dl | tr ' ' '\n' | head -12 | tr '\n' ' ')"
done

echo "=== 2. every T_stm_michel branch and point row (d51g_branch_census --pts)"
G="matched|BIT-IDENTICAL|NO shared|moved|FLIPS|IDENTICAL point|role labels|geometry changed|NEW|DROPPED"
cen() {   # det before after out
    python3 $X/d51g_branch_census.py --before "$1/work/*_$2" --after "$1/work/*_$3" \
        --before-arm $2 --after-arm $3 --pts --out $W/$4 > /dev/null 2>&1
    echo "  $1 $2 -> $3 rc=$?"; grep -E "$G" $W/$4
}
cen pdvd p72vleg p72voff g_pdvd.txt
cen pdhd p72hleg p72hoff g_pdhd.txt
cen pdvd p4v50 p72vleg g_stale.txt
cen pdvd p72voff p72vb60 g_b60.txt
cen pdvd p72vb60 p72vb90 g_b90.txt
cen pdvd p72voff p72v2ab g_2ab.txt
for x in ${EXTRA_ARMS:-}; do cen pdvd p72voff $x g_$x.txt; done

echo "=== 3. P3b neutrality in detail, p72voff -> p72vb60: which candidates and which scalar branches differ; point rows by role"
python3 - <<'EOF'
import glob, os, collections, numpy as np, uproot
def load(arm):
    S, PTS = {}, collections.defaultdict(list)
    for d in sorted(glob.glob("pdvd/work/*_" + arm)):
        fp = os.path.join(d, "tracking-pr.root")
        if not os.path.exists(fp): continue
        f = uproot.open(fp); ev = os.path.basename(d)[: -len(arm) - 1]
        if "T_stm_michel" in f:
            t = f["T_stm_michel"].arrays(library="np")
            names = [n for n, v in t.items() if v.dtype != object]
            for i in range(len(t["cluster_id"])):
                S[(ev, int(t["cluster_id"][i]))] = {n: t[n][i].item() for n in names}
        if "T_stm_michel_pts" in f:
            p = f["T_stm_michel_pts"].arrays(library="np")
            for i in range(len(p["cluster_id"])):
                PTS[(ev, int(p["cluster_id"][i]))].append(tuple(float(p[c][i]) for c in ("role", "seg_id", "x", "y", "z", "q", "L", "rr", "q_sup")))
    return S, PTS
(A, PA), (B, PB) = load("p72voff"), load("p72vb60")
keys = sorted(set(A) | set(B))
print("  candidates off %d, vb60 %d, common %d" % (len(A), len(B), len(set(A) & set(B))))
onlyB = sorted(set(n for k in B for n in B[k]) - set(n for k in A for n in A[k]))
print("  branches only on vb60: %s" % onlyB)
for k in keys:
    if k not in A or k not in B: print("  candidate on one side only:", k); continue
    dif = [n for n in A[k] if n in B[k] and not (A[k][n] == B[k][n] or (A[k][n] != A[k][n] and B[k][n] != B[k][n]))]
    ex = {n: B[k][n] for n in onlyB if B[k][n] != 0}
    if dif or ex:
        print("  %s/%d differs: %s | new-branch values %s" % (k[0], k[1], ", ".join("%s %s->%s" % (n, A[k][n], B[k][n]) for n in dif), ex))
bad = []
for k in sorted(set(PA) | set(PB)):
    for role in sorted(set(r[0] for r in PA.get(k, []) + PB.get(k, []))):
        ra = collections.Counter(r for r in PA.get(k, []) if r[0] == role)
        rb = collections.Counter(r for r in PB.get(k, []) if r[0] == role)
        if ra != rb: bad.append("%s/%d role %d (%d -> %d rows)" % (k[0], k[1], role, sum(ra.values()), sum(rb.values())))
print("  point-row groups that differ: %s" % (bad or "none"))
EOF

echo "=== 4. prep the PDVD arms (bare production; the pin tranche sheet as before)"
cd $IMG/pdhd/stm_michel_scan
for a in $VA; do
    if [ "${SKIP_PREP:-0}" = 1 ] && [ -d $W/prep_$a ]; then echo "  prep $a reused"; continue; fi
    ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  prep $a rc=$? payloads $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l)"
done

echo "=== 5. census on the smx1a+smx3+smx4 record (baseline p72voff)"
for a in $VA; do
    STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$a --baseline $W/prep_p72voff --arm $a > $W/score_$a.txt 2>&1
    echo "  census $a rc=$?"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' $W/score_$a.txt | grep census
done
cd $IMG

echo "=== 6. grading (d72_score.py)"
STM_SCAN_RECORD=$REC python3 $X/d72_score.py --off p72voff=$W/prep_p72voff \
    --on p72vb60=$W/prep_p72vb60:kmin=60 --on p72vb90=$W/prep_p72vb90:kmin=90 \
    --on p72v2ab=$W/prep_p72v2ab:kmin=60,mlt=0.15,fsm=60 ${SCORE_EXTRA:-} \
    --json $W/score_p72.json > $W/score_p72.txt 2>&1
echo "  score rc=$?"; cat $W/score_p72.txt

echo "=== 7. the record is untouched"
(cd $IMG/pdhd/stm_michel_scan && python3 census_score.py --check > $W/check.txt 2>&1; echo "  --check rc=$?: $(tail -1 $W/check.txt)")
echo GATES_DONE
