#!/bin/bash
# doc 98 -- the BYTE-driven retire round, 2026-09-02.
#
#   ./scripts/retire/retire_20260902.sh              # dry run (default)
#   CONFIRM=yes ./scripts/retire/retire_20260902.sh  # actually delete
#
# Order (non-negotiable, carried from the 08-31 / 09-01 rounds):
#   scripts/retire/rollup_grp0825_ql_20260902.py    (freeze the ql record)
#   scripts/retire/witness_sentinels_20260902.py    (build work-sent97-*)
#   scripts/retire/plan_20260902.py                 (7 interlocks, "OVERALL: PASS")
#   scripts/retire/archive_records_20260902.py      (integrity gate PASS n/n)
#   ./scripts/retire/retire_20260902.sh             (DRY RUN, check dirs/bytes)
#   CONFIRM=yes ./scripts/retire/retire_20260902.sh
#
# OWNER SCOPE, verbatim (2026-09-02): "the sbnd_xin directory now is exploded
# to 189G, which needs to be clean up.  We should retire the middle work*
# directory, and we only need to save the latest production results for Q/L
# matching as well as PR results."
#
# So: keep work-*-d97fv (Q/L, ref/prod-2026-09-04) and work-*-d97fvpr2 (PR),
# keep the imaging substrate, retire the middle.  62 dirs + the stale Q/L half
# of grp0825 = 125.1 GiB.
#
# THE TWO THINGS THAT MAKE THIS ROUND DIFFERENT FROM ITS DRY RUN LOOKING SAFE:
#
#   1. work-*-d97off2pr and work-*-prod0901b were the ONLY arms where
#      d97_sentinels.py is 31/31 (production is 30/31 -- 105074 / pr-128 class
#      B is an open owner call).  Deleting both would strand that sentinel with
#      nothing to adjudicate against.  work-sent97-* is materialised FIRST and
#      interlock 2 refuses the round unless the suite is 31/31 against the
#      witness alone.
#
#   2. work-*-grp0825 is SPLIT, not deleted: 10478 symlinks across the tree
#      resolve into its evt<N> imaging, so only the ql_evt<N> half goes, and
#      only after its 18402-member hash rollup was frozen into
#      state-20260902/grp0825-ql-rollup.tsv.  A PROTECTED.txt line records that
#      grp0825 is imaging-only from here, so the next round does not read the
#      old KEEP as still covering Q/L.
#
# A PEER SESSION IS LIVE IN THIS TREE (PDVD doc 25, gate round 7).  Its SBND
# gate arms work-*-doc25new* live in sbnd_xin and are protected by prefix;
# interlock C re-derives liveness from mtime + open handles + a tree-scoped
# process scan on every run, and was proven able to fail before being trusted.
set -u
ROOT=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
STATE=$ROOT/scripts/retire/state-20260902
REC=$ROOT/archive/records/campaign-close-20260902
CONFIRM=${CONFIRM:-no}
cd "$ROOT" || exit 2

[ -f "$STATE/plan.json" ] || { echo "REFUSE: no plan.json -- run plan_20260902.py"; exit 2; }

# --- interlock A: the plan must be PASSing right now, not when it was written
echo "== interlock A: re-running plan_20260902.py =="
python3 "$ROOT/scripts/retire/plan_20260902.py" > "$STATE/plan-recheck.log" 2>&1
rc=$?
tail -3 "$STATE/plan-recheck.log"
[ $rc -eq 0 ] || { echo "REFUSE: plan interlocks FAIL (see $STATE/plan-recheck.log)"; exit 2; }

# --- interlock B: every retiring arm must have a verified record
echo "== interlock B: record layer =="
python3 - <<'PY' || exit 2
import json, os, sys, tarfile
SX="/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
STATE=os.path.join(SX,"scripts/retire/state-20260902")
REC=os.path.join(SX,"archive/records/campaign-close-20260902")
plan=json.load(open(os.path.join(STATE,"plan.json")))
bad=[]
for tag in plan["ARCHIVE"]:
    d=os.path.join(REC, plan["group"][tag])
    tgz=os.path.join(d, tag+".tar.gz")
    man=os.path.join(d, tag+".manifest.tsv")
    if not (os.path.exists(tgz) and os.path.exists(man)):
        bad.append((tag,"missing")); continue
    try:
        with tarfile.open(tgz,"r:gz") as tf:
            n=sum(1 for m in tf.getmembers() if m.isfile())
    except Exception as e:
        bad.append((tag,f"unreadable {e}")); continue
    want=0
    for i,ln in enumerate(open(man)):
        f=ln.rstrip("\n").split("\t")
        if i and f[0]=="record" and f[1]=="ARCHIVED": want=int(f[2])
    if n!=want: bad.append((tag,f"tar {n} != manifest {want}"))
print(f"  {len(plan['ARCHIVE'])-len(bad)}/{len(plan['ARCHIVE'])} arms have a verified record tar")
for t,w in bad[:10]: print(f"  BAD {t}: {w}")
sys.exit(1 if bad else 0)
PY

# --- the sets
DIRS=$(python3 -c "import json;print(' '.join(json.load(open('$STATE/plan.json'))['ARCHIVE']))")
QLD=$(python3 -c "import json;print(' '.join(json.load(open('$STATE/plan.json'))['grp0825_ql']))")
nd=$(echo $DIRS | wc -w); nq=$(echo $QLD | wc -w)

echo
echo "== retire set: $nd arm dirs + $nq grp0825 ql_evt dirs =="
echo "$DIRS" | tr ' ' '\n' | sed 's/^/  arm  /'
echo "  ql   work-*-grp0825/ql_evt* ($nq dirs; imaging evt<N> KEPT)"

if [ "$CONFIRM" != "yes" ]; then
  echo
  echo "DRY RUN -- nothing deleted.  Bytes that would be freed:"
  du -sck $DIRS 2>/dev/null | tail -1
  python3 -c "import json;p=json.load(open('$STATE/plan.json'));print('  grp0825 ql half: %.1f GiB'%(p['bytes_ql']/1048576))"
  echo
  echo "To execute:  CONFIRM=yes ./scripts/retire/retire_20260902.sh"
  exit 0
fi

echo "== DELETING =="
for d in $DIRS; do
  case "$d" in
    work-*-d97fv|work-*-d97fvpr2|work-*-grp0825|work-sent97-*|*doc25*)
      echo "REFUSE: $d is a keep target"; exit 2;;
  esac
  if [ -d "$d" ]; then
    rm -rf -- "$d"
    echo "  removed $d"
  fi
done
for d in $QLD; do
  case "$d" in
    *ql_evt*) rm -rf -- "$d";;
    *) echo "REFUSE: $d is not a ql_evt dir"; exit 2;;
  esac
done
echo "  removed $nq grp0825 ql_evt dirs (imaging kept)"

echo
echo "== post-state =="
echo "broken symlinks: $(find work-*/ -maxdepth 1 -xtype l 2>/dev/null | wc -l)"
echo "work dirs: $(ls -d work-*/ 2>/dev/null | wc -l)"
du -sh "$ROOT"
