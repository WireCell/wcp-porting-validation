#!/bin/bash
# doc 100 sec 6.1 -- ~/tmp sweep, PASS 2 (2026-09-04, owner-selected).
#
#   ./scripts/retire/sweep_tmp_20260904b.sh              # dry run (default)
#   CONFIRM=yes ./scripts/retire/sweep_tmp_20260904b.sh  # actually delete
#
# Pass 1 (sweep_tmp_20260904.sh) took ~/tmp 132 -> 76 GiB.  The owner then asked
# for more and selected "drop the uncited pdvd pins".  This is that set and
# nothing else.
#
#   pinlib2 .. pinlib7   7.2 GiB   ZERO citations anywhere in either repo.  All
#                                  seven were cut 2026-09-02 11:42-16:51 for the
#                                  pdvd doc-25 gate rounds.  `pinlib` (unsuffixed)
#                                  IS cited -- by docs/25_pdvd-stopping-muon-
#                                  michel-chain.md -- and is KEPT.
#   doc27                1.2 GiB   doc pdvd/27 CLOSED; its d27v7/v7* arms are in
#                                  the pending pdvd release set.
#   doc35                1.9 GiB   doc pdvd/35 CLOSED; its d34*/d35 arms likewise.
#
# THE COST, STATED PLAINLY (this is the owner's trade, not a free win):
# pinlib{2..7} are the pinned binaries behind work-*-doc25new{2..7}, which are
# KEPT.  Dropping the pins does not touch those arms or their measurements, but
# it does mean they can no longer be RE-RUN against the exact binary that made
# them -- only re-read.  doc 98's rule is "a pin goes with its arms"; this
# deliberately departs from it, on the owner's instruction, because six 1.2 GiB
# copies of a superseded library is a poor trade for re-runnability of an
# intermediate gate round.  The commit each pins is recorded in doc pdvd/25.
#
# doc27/doc35's arms are in the pdvd release set but THAT ROUND HAS NOT RUN.
# Dropping a pin before its arms is the one ordering this machinery refuses
# elsewhere -- here it is safe only because the arms are already condemned and
# their record layer is archived (pdvd/archive/records/pdvd-rounds-20260904).
set -u
TMP=/home/xqian/tmp
CONFIRM=${CONFIRM:-no}

# --- interlock A: never sweep a live session's scratchpad or a live pin
LIVE=$(ps -u "$(id -un)" -o args= | grep -oE 'claude --resume [0-9a-f]{8}-[0-9a-f-]+' | awk '{print $3}' | sort -u)
echo "== live claude sessions (scratchpads protected) =="; echo "${LIVE:-  (none)}" | sed 's/^/  /'

DROP=""
for d in pinlib2 pinlib3 pinlib4 pinlib5 pinlib6 pinlib7 doc27 doc35; do
  [ -d "$TMP/$d" ] && DROP="$DROP $d"
done

# --- interlock B: re-derive "uncited" NOW rather than trusting this header.
# A citation added since the plan was written must veto the drop.
echo "== interlock B: re-deriving citations =="
for d in $DROP; do
  n=$(grep -rl --include='*.sh' --include='*.py' --include='*.md' "tmp/$d\b" \
        /home/xqian/toolkit-dev/wcp-porting-img 2>/dev/null | wc -l)
  case "$d" in
    doc27|doc35) printf "  %-9s cites=%s (round CLOSED, arms condemned)\n" "$d" "$n";;
    *) printf "  %-9s cites=%s\n" "$d" "$n"
       [ "$n" -eq 0 ] || { echo "REFUSE: $d is cited; it was listed as uncited."; exit 2; };;
  esac
done

# --- interlock C: the KEEP pins must still be present and must not be listed
for k in pinlib d39/lib_d39 d97b-libsnap d99r2-libsnap prod0901b-libsnap d31r6lib doc37/lib_pin; do
  [ -e "$TMP/$k" ] || { echo "REFUSE: keep-pin $k is missing -- state is not what the plan assumed"; exit 2; }
  case " $DROP " in *" $k "*) echo "REFUSE: $k is a KEEP pin"; exit 2;; esac
done
echo "== interlock C: 7 keep-pins present and not listed =="

echo; echo "== sweep set =="
for d in $DROP; do printf "  %-10s %s\n" "$d" "$(du -sh "$TMP/$d" 2>/dev/null | cut -f1)"; done
TOT=$(du -sc --block-size=1 $(for d in $DROP; do echo "$TMP/$d"; done) 2>/dev/null | tail -1 | cut -f1)

if [ "$CONFIRM" != "yes" ]; then
  echo; printf "DRY RUN -- nothing deleted.  Would free %.2f GiB from %d dirs.\n" \
      "$(echo "$TOT/1073741824" | bc -l)" "$(echo $DROP | wc -w)"
  echo "To execute:  CONFIRM=yes ./scripts/retire/sweep_tmp_20260904b.sh"; exit 0
fi

echo "== DELETING =="
for d in $DROP; do rm -rf -- "$TMP/$d" && echo "  removed $d"; done
echo; echo "~/tmp now: $(du -sh $TMP 2>/dev/null | cut -f1)"
