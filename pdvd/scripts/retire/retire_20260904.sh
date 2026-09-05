#!/bin/bash
# doc pdvd/29 -- the 2026-09-04 retire round (pdvd half).
#
#   ./scripts/retire/retire_20260904.sh              # dry run (default)
#   CONFIRM=yes ./scripts/retire/retire_20260904.sh  # actually delete
#
# Order (non-negotiable):
#   scripts/retire/census_20260904.py  <out.json>   (the arm census)
#   scripts/retire/plan_20260904.py                 (7 interlocks, "OVERALL: PASS")
#   scripts/retire/archive_records_20260904.py      (integrity gate PASS n/n)
#   ./scripts/retire/retire_20260904.sh             (DRY RUN, check dirs/bytes)
#   CONFIRM=yes ./scripts/retire/retire_20260904.sh
#
# OWNER SCOPE, verbatim (2026-09-04): "I would like retire some work* file in
# ./sbnd_xin and ./pdvd directory.  We want to keep the latest production result
# as well as their input."  Asked which depth, the owner chose option A --
# substrate spine + the gate arms of the three shipped flips + the live round.
#
# *** A PEER SESSION IS LIVE IN THIS TREE. ***  A peer Claude session has been
# running since 08:16 and committed 6e7f6350 at 18:07; it wrote d39base dirs at
# 18:06.  The owner's instruction was "plan now, execute after they finish".
# Do NOT run with CONFIRM=yes until the doc-39 round is closed.  Interlock A
# re-runs the full plan at confirm time -- which re-derives the symlink graph
# and the live-writer scan -- so a peer that resumed since the plan was written
# will make this driver REFUSE rather than delete under them.  That is the
# safety net, not a licence to run it early: stage_pr_tag.sh lets the peer
# create a NEW inbound link to any arm at any moment, and a link created
# between the plan and the rm is invisible to both.
#
# WHY THE KEEP SET IS FIVE LEVELS DEEP (see plan_20260904.py's header):
#   keep -> d27fresh -> d28dlfp -> d34base -> d37dloff -> d39*(LIVE)
# `keep` holds the SP+DNNROI frames and d27fresh borrows 960 of them, so `keep`
# IS the imaging input, not a superseded arm.  d27fresh is stage_pr_tag.sh's
# documented default src_tag and carries 9793 inbound links.
set -u
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
STATE=$PDVD/scripts/retire/state-20260904
REC=$PDVD/archive/records/pdvd-rounds-20260904
CONFIRM=${CONFIRM:-no}
cd "$PDVD" || exit 2

[ -f "$STATE/plan.json" ] || { echo "REFUSE: no plan.json -- run plan_20260904.py"; exit 2; }

# --- interlock A: the plan must be PASSing right NOW, not when it was written.
# This is what makes "plan now, execute later" safe: it re-derives the symlink
# graph and re-runs the 20s live-writer window against the current tree.
echo "== interlock A: re-running plan_20260904.py =="
python3 "$PDVD/scripts/retire/plan_20260904.py" > "$STATE/plan-recheck.log" 2>&1
rc=$?
tail -4 "$STATE/plan-recheck.log"
[ $rc -eq 0 ] || { echo "REFUSE: plan interlocks FAIL (see $STATE/plan-recheck.log)"; exit 2; }

# --- interlock B: every retiring arm must have a verified record tar
echo "== interlock B: record layer =="
RETIRE_JOBS=${RETIRE_JOBS:-16} python3 "$PDVD/scripts/retire/archive_records_20260904.py" \
    --verify-only | tail -2
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "REFUSE: record integrity gate FAIL"; exit 2; }

# --- the set
mapfile -t DIRS < <(python3 -c "import json;[print(d) for d in json.load(open('$STATE/plan.json'))['ARCHIVE']]")
nd=${#DIRS[@]}
echo
echo "== retire set: $nd arm dirs =="
python3 -c "
import json,collections
p=json.load(open('$STATE/plan.json'))
c=collections.Counter(p['group'][d] for d in p['ARCHIVE'])
for g,n in sorted(c.items(),key=lambda kv:-kv[1]): print(f'  {g:<20} {n:>5} dirs')"

if [ "$CONFIRM" != "yes" ]; then
  echo
  echo "DRY RUN -- nothing deleted.  Bytes that would be freed:"
  python3 -c "
import json;p=json.load(open('$STATE/plan.json'))
print('  %d dirs   %.2f GiB   (keep %d dirs / %.2f GiB + %d out-of-scope light dirs)'%(
  len(p['ARCHIVE']),p['bytes_retire_kb']/1048576,len(p['KEEP']),
  p['bytes_keep_kb']/1048576,len(p['OUT_OF_SCOPE'])))"
  echo
  echo "*** A PEER SESSION IS LIVE IN pdvd/work -- do not confirm until doc 39 closes. ***"
  echo "To execute:  CONFIRM=yes ./scripts/retire/retire_20260904.sh"
  exit 0
fi

echo "== DELETING =="
n=0
for d in "${DIRS[@]}"; do
  # belt and braces: the grammar, then the keep-arm names, checked per dir
  case "$d" in
    */*|.*|"") echo "REFUSE: '$d' is not a bare work child"; exit 2;;
    *_keep|*_d27fresh|*_d28dlfp|*_d34base|*_d37dloff|*_d31r6e2e|*_perfslide|\
    *_d36on|*_d36off|*_d37on05|*_d37off0|*_d37off1|\
    *_d38h20|*_d38off|*_d38flip20|*_magnify|*_d39*)
      echo "REFUSE: $d is a keep target"; exit 2;;
  esac
  [ -d "work/$d" ] || continue
  rm -rf -- "work/$d"
  n=$((n+1))
done
echo "  removed $n dirs"

echo
echo "== post-state =="
# INTERLOCK 4 recorded 6 PRE-EXISTING broken links (target arm 'stm1', deleted
# outside the machinery some time ago) and proved every one lives inside a
# retiring arm -- so after this round the correct count is ZERO, and any
# non-zero value here is damage THIS round caused.
echo "broken symlinks (expect 0; 6 pre-existing ones lived in retired arms): \
$(find work/ -maxdepth 2 -xtype l 2>/dev/null | wc -l)"
echo "work children: $(ls work/ | wc -l)"
du -sh "$PDVD/work"
