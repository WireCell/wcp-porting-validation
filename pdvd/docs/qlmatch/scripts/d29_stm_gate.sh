#!/bin/bash
# doc qlmatch/29 -- the pre-registered STM gate (d29/prereg.md sec 3.4) on a finished 120-event arm, against p100flip.
#   bash d29_stm_gate.sh q29stm        (after d29_stm_arms.sh; writes docs/qlmatch/d29/stm_gate_<ARM>_*.txt)
# Reuses doc pdvd/99-100 scripts unchanged: d99rw_identity.py, d99rw_census.py, d99_population.py,
# d100_carry_verdicts.py, d100r3_grade.py; closure fit_qtol_crossers.py.  The owner-corrected record is round 2's carry_r2
# (keyed on p99rwon = p100c = p100flip cluster ids); the arm's copy is carried by geometry into /home/xqian/tmp/p29/carry.
set -u
ARM=${1:?arm}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
NS=$PDVD/docs/nf_sp_img_clus/scripts
D=$PDVD/docs/qlmatch/d29
REC=/home/xqian/tmp/p100/carry_r2/latest_on_p99rwon_corrected.json
CAR=/home/xqian/tmp/p29/carry/corrected_on_$ARM.json
mkdir -p /home/xqian/tmp/p29/carry
[ -e "$CAR" ] && { echo "REFUSE existing $CAR" >&2; exit 2; }
cd "$NS" || exit 2
python3 d99rw_identity.py --arm "$ARM" --base p100flip --nt all > "$D/stm_gate_${ARM}_identity.txt" 2>&1; echo "identity rc=$?"
python3 d99rw_census.py --arms p100flip "$ARM" > "$D/stm_gate_${ARM}_census.txt" 2>&1; echo "census rc=$?"
python3 d99_population.py --pairs p100flip:p100flip p100flip:"$ARM" "$ARM":p100flip > "$D/stm_gate_${ARM}_population.txt" 2>&1; echo "population rc=$?"
python3 d100_carry_verdicts.py --record "$REC" --base-arm p99rwon --arm "$ARM" --out "$CAR" > "$D/stm_gate_${ARM}_carry.txt" 2>&1; echo "carry rc=$?"
python3 d100r3_grade.py --arm p100flip:"$REC" --arm "$ARM":"$CAR" > "$D/stm_gate_${ARM}_grade.txt" 2>&1; echo "grade rc=$?"
python3 d99_grade.py --arms p100flip:"$REC" "$ARM":"$CAR" --movers p100flip,"$ARM" > "$D/stm_gate_${ARM}_movers.txt" 2>&1; echo "movers rc=$?"
(cd "$PDVD/ql_light_calib" && python3 fit_qtol_crossers.py --tag "$ARM") > "$D/stm_gate_${ARM}_crossers.txt" 2>&1; echo "closure rc=$?"
tail -1 "$D/stm_gate_${ARM}_identity.txt"
cat "$D/stm_gate_${ARM}_grade.txt"
grep "all " "$D/stm_gate_${ARM}_census.txt"
sed -n '/is_stm -> /,+4p' "$D/stm_gate_${ARM}_population.txt" | grep "all\|->"
grep "GLOBAL\|cathode XA\|PMTs" "$D/stm_gate_${ARM}_crossers.txt"
