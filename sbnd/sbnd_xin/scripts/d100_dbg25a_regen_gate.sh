#!/bin/bash
# doc 100 sec 12 -- prove the regenerated work-dbg25a-ql imaging reproduces the
# arm doc 100 sec 4 deleted, using the archived record layer as the reference.
#
#   ./scripts/d100_dbg25a_regen_gate.sh
#
# TWO INDEPENDENT CHECKS, because either alone is weak:
#   1. npz count + BYTE TOTAL vs the archived manifest.  The npz were a DROPPED
#      heavy class so no per-file hash survives -- 80 files / 105545029 bytes is
#      the only number the record kept.
#   2. per-event wct_img_evt<N>.log content vs the archived logs, which the
#      record layer DID keep in full.  This is the content-level check the byte
#      total cannot give.
#
# WHAT MUST BE NORMALISED AWAY, and why each is not physics:
#   [HH:MM:SS.mmm]  line prefix          -- wall clock
#   TICK: N ms (this: N ms)              -- integer ms counters, run to run
#   the whole `I [ timer ] Timer:` block -- wall/core-sec, and SORTED BY
#                                           DURATION, so its ORDER also varies
# Nothing else is stripped.  Getting this wrong in the safe direction is easy
# and worthless, so the gate carries its own NEGATIVE CONTROL: it compares two
# DIFFERENT events under the same normalisation and fails loudly if they
# compare equal.  A gate that cannot fail proves nothing.
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
QL=$BASE/work-dbg25a-ql
ARC=$BASE/archive/records/campaign-close-20260904/doc95-debug25/work-dbg25a-ql.tar.zst
EVTS="2 4 5 8 10 12 14 16 17 21 22 23 25 28 30 31 34 39 41 44"
REF=$(mktemp -d /home/xqian/tmp/d100gate.XXXXXX)
[ -f "$ARC" ] || { echo "REFUSE: archived record $ARC missing"; exit 2; }
zstd -dc "$ARC" | tar x -C "$REF" || exit 2

norm() { sed -E 's/^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+\] //; s/TICK: [0-9]+ ms \(this: [0-9]+ ms\)/TICK: <t>/g; s/0x[0-9a-f]+//g' "$1" | grep -v 'I \[ timer  \] Timer:'; }

echo "=== check 1: npz count and byte total vs the archived manifest ==="
n=$(find "$QL" -name 'icluster-apa*.npz' | wc -l)
b=$(find "$QL" -name 'icluster-apa*.npz' -printf '%s\n' | awk '{s+=$1} END{print s+0}')
echo "  files $n (want 80)   bytes $b (want 105545029)"
c1=FAIL; [ "$n" -eq 80 ] && [ "$b" -eq 105545029 ] && c1=PASS
echo "  -> $c1"

echo "=== check 2: per-event imaging log content vs the archived logs ==="
same=0; diffn=0
for e in $EVTS; do
  o=$REF/work-dbg25a-ql/evt$e/wct_img_evt$e.log; n2=$QL/evt$e/wct_img_evt$e.log
  if diff -q <(norm "$o") <(norm "$n2") >/dev/null 2>&1; then same=$((same+1)); else diffn=$((diffn+1)); fi
done
echo "  identical $same / 20   differing $diffn"
c2=FAIL; [ "$same" -eq 20 ] && c2=PASS
echo "  -> $c2"

echo "=== negative control: two DIFFERENT events must compare different ==="
if diff -q <(norm "$REF/work-dbg25a-ql/evt2/wct_img_evt2.log") <(norm "$QL/evt4/wct_img_evt4.log") >/dev/null 2>&1
then echo "  -> BROKEN: evt2 == evt4, the normalisation destroyed the signal"; c3=FAIL
else echo "  -> OK: evt2 vs evt4 reported different, the check has power to fail"; c3=PASS; fi

echo "=== dangling links in the protected arm ==="
d=$(find "$BASE/work-dbg25a-d97prodchk" -xtype l | wc -l)
echo "  $d (want 0; was 100)"
c4=FAIL; [ "$d" -eq 0 ] && c4=PASS
echo "  -> $c4"

rm -rf "$REF"
echo
if [ "$c1$c2$c3$c4" = "PASSPASSPASSPASS" ]; then echo "OVERALL: PASS"; exit 0
else echo "OVERALL: FAIL ($c1/$c2/$c3/$c4)"; exit 1; fi
