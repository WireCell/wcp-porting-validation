#!/bin/bash
# doc qlmatch/35 F1(c) -- after the ToT runner-default flip: production with no overrides through stm/run_campaign.sh
# (q35flip) must equal the candidate arm run with the overrides (q35tk) on all 120 events of pdvd/stm/events.txt, and
# differ from pre-flip production (q35ctl).  Fork of d29_f1_check.sh (untouched): arm names and output dir changed, and
# the deploy tell added -- the calib dump's quality_params carry ks_sat_tol only when it is > 0.
#   bash d35_f1_check.sh            (writes docs/qlmatch/d35/f1_{identity,calib}.txt)
set -u
A=${A:-q35flip}; B=${B:-q35tk}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
D=$PDVD/docs/qlmatch/d35
for f in "$D/f1_identity.txt" "$D/f1_calib.txt"; do [ -e "$f" ] && { echo "REFUSE existing $f" >&2; exit 2; }; done
(cd "$PDVD/docs/nf_sp_img_clus/scripts" && python3 d99rw_identity.py --arm "$A" --base "$B" --nt all) > "$D/f1_identity.txt" 2>&1
echo "identity rc=$?"
EVS=$(grep -v '^#' "$PDVD/stm/events.txt" | awk 'NF>=2 {printf "%06d_%s\n", $1, $2}')
{
  echo "# doc qlmatch/35 F1 -- calib dumps $A vs $B, byte for byte (cmp), events of pdvd/stm/events.txt"
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
  C=${C:-q35ctl}; cdiff=0
  for e in $EVS; do
    for f in $(cd "$PDVD/work/${e}_$A" 2>/dev/null && ls calib-evt*.json 2>/dev/null); do
      cmp -s "$PDVD/work/${e}_$A/$f" "$PDVD/work/${e}_$C/$f" || { cdiff=$((cdiff+1)); break; }
    done
  done
  echo "CONTROL calib dumps $A vs $C (pre-flip production): events differing $cdiff/$nev (must be > 0)"
  # deploy tell: ks_sat_tol in the dumps' quality_params of the flip arm, not in the pre-flip control
  ka=0; kc=0
  for e in $EVS; do
    grep -q '"ks_sat_tol"' "$PDVD/work/${e}_$A"/calib-evt*.json 2>/dev/null && ka=$((ka+1))
    grep -q '"ks_sat_tol"' "$PDVD/work/${e}_$C"/calib-evt*.json 2>/dev/null && kc=$((kc+1))
  done
  echo "DEPLOY ks_sat_tol in calib dumps: $A $ka/$nev (must be $nev), $C $kc/$nev (must be 0)"
} > "$D/f1_calib.txt"
tail -1 "$D/f1_identity.txt"; tail -3 "$D/f1_calib.txt"
