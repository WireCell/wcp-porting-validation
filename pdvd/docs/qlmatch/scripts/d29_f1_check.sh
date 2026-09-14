#!/bin/bash
# doc qlmatch/29 sec 7 -- F1 after the PDVD_QL_LASSO_BWEIGHT=0.1 runner-default flip: production with no overrides (q29flip)
# must equal the STM-gate arm run with the override (q29stm) on all 120 events of pdvd/stm/events.txt.
#   bash d29_f1_check.sh            (after ARM=q29flip d29_stm_arms.sh; writes docs/qlmatch/d29/f1_{identity,calib}.txt)
# d99rw_identity.py covers pctree, tlas, PR trees, mabc-pr; the Q/L calib dumps (clusters, flashes, bundles with their LASSO
# strengths; the LASSO settings themselves are not in quality_params) are compared byte for byte here, file lists must match.
set -u
A=${A:-q29flip}; B=${B:-q29stm}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
D=$PDVD/docs/qlmatch/d29
for f in "$D/f1_identity.txt" "$D/f1_calib.txt"; do [ -e "$f" ] && { echo "REFUSE existing $f" >&2; exit 2; }; done
(cd "$PDVD/docs/nf_sp_img_clus/scripts" && python3 d99rw_identity.py --arm "$A" --base "$B" --nt all) > "$D/f1_identity.txt" 2>&1
echo "identity rc=$?"
EVS=$(grep -v '^#' "$PDVD/stm/events.txt" | awk 'NF>=2 {printf "%06d_%s\n", $1, $2}')
{
  echo "# doc qlmatch/29 F1 -- calib dumps $A vs $B, byte for byte (cmp), events of pdvd/stm/events.txt"
  same=0; diff=0; nev=0
  for e in $EVS; do
    nev=$((nev+1))
    la=$(cd "$PDVD/work/${e}_$A" 2>/dev/null && ls calib-evt*.json 2>/dev/null | sort)
    lb=$(cd "$PDVD/work/${e}_$B" 2>/dev/null && ls calib-evt*.json 2>/dev/null | sort)
    if [ -z "$la" ] || [ "$la" != "$lb" ]; then echo "$e: FILE LIST DIFFERS ($(echo $la | wc -w) vs $(echo $lb | wc -w))"; diff=$((diff+1)); continue; fi
    ok=1
    for f in $la; do cmp -s "$PDVD/work/${e}_$A/$f" "$PDVD/work/${e}_$B/$f" || { echo "$e: $f DIFFERS"; ok=0; }; done
    if [ $ok = 1 ]; then same=$((same+1)); else diff=$((diff+1)); fi
  done
  echo "SUMMARY calib dumps $A vs $B: identical $same/$nev, differ $diff"
  # negative control: the pre-flip production arm must differ somewhere, or identity above is vacuous
  C=${C:-p100flip}; cdiff=0
  for e in $EVS; do
    for f in $(cd "$PDVD/work/${e}_$A" 2>/dev/null && ls calib-evt*.json 2>/dev/null); do
      cmp -s "$PDVD/work/${e}_$A/$f" "$PDVD/work/${e}_$C/$f" || { cdiff=$((cdiff+1)); break; }
    done
  done
  echo "CONTROL calib dumps $A vs $C (pre-flip production): events differing $cdiff/$nev (must be > 0)"
} > "$D/f1_calib.txt"
tail -1 "$D/f1_identity.txt"; tail -2 "$D/f1_calib.txt"
