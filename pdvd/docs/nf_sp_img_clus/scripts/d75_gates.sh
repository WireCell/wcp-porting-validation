#!/usr/bin/env bash
# doc pdvd/75 (P1b) -- gates and grading for the arms of d75_arms.sh.  Fork of
# d74_gates.sh.  Output: tee /home/xqian/tmp/p75/gates.log.
#   EXTRA_ARMS="p75vX ..."            wave-2 PDVD arms, compared with p75voff and prepped
#   SCORE_EXTRA="--on p75vX=/home/xqian/tmp/p75/prep_p75vX:fb"   their d75_score.py variant
#   SKIP_PREP=1                        reuse existing preps
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p75
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
cd $IMG
ON="p75vfb p75vfb0 ${EXTRA_ARMS:-}"
VA="p75vleg p75voff $ON"

echo "=== 0. completeness: events with both tracking files, per arm; loader deaths; pins"
for a in $VA p75hleg p75hoff; do
    d=pdvd; [ "${a:3:1}" = h ] && d=pdhd
    n=$(ls -d $d/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in $d/work/*_$a; do [ -s $e/tracking-stm.root ] && [ -s $e/mabc-pr.zip ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" $d/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  $a dirs $n complete $ok loader-deaths $lt | $(grep -h 'md5' $W/$a.out 2>/dev/null | tr '\n' ' ')"
done
echo "  pins now: $(md5sum /home/xqian/tmp/p74/libpin_p74/libWireCellClus.so | cut -c1-8) (P74, expect 566dc517)  $(md5sum $W/libpin_p75/libWireCellClus.so | cut -c1-8) (P75, expect 02557b8d)"
echo "  arm completion markers: $(grep -c '^DONE' $W/arms_wave1.log) DONE lines, ALL_DONE $(grep -c '^ALL_DONE' $W/arms_wave1.log)"

echo "=== 1. mabc-pr.zip member content + calib-pr json md5"
echo "    (OFF: leg (P74 pin) vs off (P75 pin); STALE-BASELINE: p74vprod2 vs p75vleg, p74hoff vs p75hleg; P1b: off vs each ON arm)"
PAIRS=("pdvd p75vleg p75voff" "pdhd p75hleg p75hoff" "pdvd p74vprod2 p75vleg" "pdhd p74hoff p75hleg")
for x in $ON; do PAIRS+=("pdvd p75voff $x"); done
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
    [ -n "$dl" ] && echo "    zip differs on:$(echo $dl | tr ' ' '\n' | head -20 | tr '\n' ' ')"
done

echo "=== 2. every T_stm_michel branch and point row (d51g_branch_census --pts)"
G="matched|BIT-IDENTICAL|NO shared|moved|FLIPS|IDENTICAL point|role labels|geometry changed|NEW|DROPPED"
cen() {   # det before after out
    python3 $X/d51g_branch_census.py --before "$1/work/*_$2" --after "$1/work/*_$3" \
        --before-arm $2 --after-arm $3 --pts --out $W/$4 > /dev/null 2>&1
    echo "  $1 $2 -> $3 rc=$?"; grep -E "$G" $W/$4
}
cen pdvd p75vleg p75voff g_pdvd.txt
cen pdhd p75hleg p75hoff g_pdhd.txt
cen pdvd p74vprod2 p75vleg g_stale.txt
cen pdhd p74hoff p75hleg g_stale_h.txt
for x in $ON; do cen pdvd p75voff $x g_$x.txt; done

echo "=== 3. per ON arm vs p75voff: which candidates and which scalar branches differ; point-row groups by role"
for x in $ON; do
ARM_ON=$x python3 - <<'EOF'
import glob, os, collections, uproot
on = os.environ["ARM_ON"]
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
(A, PA), (B, PB) = load("p75voff"), load(on)
print("  --- %s: candidates off %d, on %d, common %d" % (on, len(A), len(B), len(set(A) & set(B))))
onlyB = sorted(set(n for k in B for n in B[k]) - set(n for k in A for n in A[k]))
print("  branches only on %s: %s" % (on, onlyB))
nd = 0
for k in sorted(set(A) | set(B)):
    if k not in A or k not in B: print("  candidate on one side only:", k); continue
    dif = [n for n in A[k] if n in B[k] and not (A[k][n] == B[k][n] or (A[k][n] != A[k][n] and B[k][n] != B[k][n]))]
    ex = {n: B[k][n] for n in onlyB if B[k][n] != 0}
    if dif or ex:
        nd += 1
        print("  %s/%d differs (%d branches): %s | new-branch values %s" % (k[0], k[1], len(dif), ", ".join(sorted(dif)[:14]) + (" ..." if len(dif) > 14 else ""), ex))
print("  %d candidates differ" % nd)
bad = []
for k in sorted(set(PA) | set(PB)):
    for role in sorted(set(r[0] for r in PA.get(k, []) + PB.get(k, []))):
        ra = collections.Counter(r for r in PA.get(k, []) if r[0] == role)
        rb = collections.Counter(r for r in PB.get(k, []) if r[0] == role)
        if ra != rb: bad.append("%s/%d role %d (%d -> %d rows)" % (k[0], k[1], role, sum(ra.values()), sum(rb.values())))
print("  point-row groups that differ: %s" % (bad or "none"))
EOF
done

echo "=== 4. prep the PDVD arms (bare production; the pin tranche sheet as before)"
cd $IMG/pdhd/stm_michel_scan
for a in $VA; do
    if [ "${SKIP_PREP:-0}" = 1 ] && [ -d $W/prep_$a ]; then echo "  prep $a reused"; continue; fi
    ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  prep $a rc=$? payloads $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l)"
done

echo "=== 5. census on the smx1a+smx3+smx4 record (baseline p75voff)"
for a in $VA; do
    STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$a --baseline $W/prep_p75voff --arm $a > $W/score_$a.txt 2>&1
    echo "  census $a rc=$?"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' $W/score_$a.txt | grep census
done
cd $IMG

echo "=== 6. the prediction (d75_sizing.py on doc 74's p74vprod2, written before launch), then the grading (d75_score.py)"
echo "  pred.json: $(stat -c '%y' $W/pred.json)  ($(python3 -c "import json;d=json.load(open('$W/pred.json'));print({k:len(v['fires']) for k,v in d.items()})"))"
STM_SCAN_RECORD=$REC python3 $X/d75_score.py --off p75voff=$W/prep_p75voff \
    --on p75vfb=$W/prep_p75vfb:fb --on p75vfb0=$W/prep_p75vfb0:fb0 ${SCORE_EXTRA:-} \
    --pred $W/pred.json --json $W/score_p75.json > $W/score_p75.txt 2>&1
echo "  score rc=$?"; cat $W/score_p75.txt

echo "=== 7. the record is untouched"
(cd $IMG/pdhd/stm_michel_scan && python3 census_score.py --check > $W/check.txt 2>&1; echo "  --check rc=$?: $(tail -1 $W/check.txt)")
echo GATES_DONE
