#!/bin/bash
# doc 100 -- the 2026-09-04 retire round (sbnd_xin half).
#
#   ./scripts/retire/retire_20260904.sh              # dry run (default)
#   CONFIRM=yes ./scripts/retire/retire_20260904.sh  # actually delete
#
# Order (non-negotiable, carried from the 08-31 / 09-01 / 09-02 rounds):
#   scripts/retire/plan_20260904.py                 (8 interlocks, "OVERALL: PASS")
#   scripts/retire/archive_records_20260904.py      (integrity gate PASS n/n)
#   ./scripts/retire/retire_20260904.sh             (DRY RUN, check dirs/bytes)
#   CONFIRM=yes ./scripts/retire/retire_20260904.sh
#
# OWNER SCOPE, verbatim (2026-09-04): "I would like retire some work* file in
# ./sbnd_xin and ./pdvd directory.  We want to keep the latest production result
# as well as their input.  Please also clean up a bit the ~/tmp directory."
# Asked which depth, the owner chose "honour PROTECTED.txt" over the deeper cut.
#
# So: keep work-*-d97fv (Q/L, ref/prod-2026-09-04), work-*-d97fvpr2 (PR), the
# grp0825 imaging substrate, work-*-d99r3prod{,pr} (the ref/prod-2026-09-05 flip
# arm), the sent97 witness, and every PROTECTED.txt line.  Retire the middle:
# 77 dirs / 12.6 GiB.
#
# WHAT MAKES THIS ROUND DIFFERENT FROM ITS DRY RUN LOOKING SAFE:
#
#   1. d99r2wr and d99r2bothpr are the SOURCE side of doc 99 round 3's flip
#      gate (d99r3prod == d99r2wr, d99r3prodpr == d99r2bothpr) -- the evidence
#      that licensed cutting ref/prod-2026-09-05.  A gate's source arm is part
#      of the record, so INTERLOCK 8 refuses the round unless their member
#      hashes were frozen into the archive FIRST.
#
#   2. INTERLOCK 5 already earned its keep this round: it refused
#      work-probe178410a (the ONLY on-disk proof that mcp2k evt 178410's SIGSEGV
#      is nondeterministic -- not re-capturable on demand) and
#      work-dbg25a-d97prodchk (doc 97 sec 9's production-default proof set).
#      Both were on the first draft of RETIRE_LABELS and are now KEEP.
#
#   3. dbg25a/dbg25b are doc 95's debug-25 MC scan set.  They are NOT the pdvd
#      doc-25 family, which is a LIVE PEER round and is protected by prefix
#      (doc25*, doc27warn, d31r7probe, d37sbnd -- a peer Claude session has been
#      running since 08:16 and committed at 18:07).  Interlock 4 re-derives
#      liveness from mtime + open handles + a tree-scoped process scan.
#
#   4. Two sentinel scripts exist and they answer different questions.
#      pr127_sentinels.py is 30 PASS / 0 FAIL on production (doc 98 retired the
#      105074 case there).  d97_sentinels.py still carries 105074, so it is
#      30/1 on production and 31/0 on the work-sent97-* witness -- which is
#      exactly what that witness was carved out to do.  Not a regression.
set -u
ROOT=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
STATE=$ROOT/scripts/retire/state-20260904
REC=$ROOT/archive/records/campaign-close-20260904
CONFIRM=${CONFIRM:-no}
cd "$ROOT" || exit 2

[ -f "$STATE/plan.json" ] || { echo "REFUSE: no plan.json -- run plan_20260904.py"; exit 2; }

# --- interlock A: the plan must be PASSing right now, not when it was written
echo "== interlock A: re-running plan_20260904.py =="
python3 "$ROOT/scripts/retire/plan_20260904.py" > "$STATE/plan-recheck.log" 2>&1
rc=$?
tail -3 "$STATE/plan-recheck.log"
[ $rc -eq 0 ] || { echo "REFUSE: plan interlocks FAIL (see $STATE/plan-recheck.log)"; exit 2; }

# --- interlock B: every retiring arm must have a verified record
echo "== interlock B: record layer =="
python3 - <<'PY' || exit 2
import json, os, sys, tarfile
SX="/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
STATE=os.path.join(SX,"scripts/retire/state-20260904")
REC=os.path.join(SX,"archive/records/campaign-close-20260904")
plan=json.load(open(os.path.join(STATE,"plan.json")))
bad=[]
for tag in plan["ARCHIVE"]:
    d=os.path.join(REC, plan["group"][tag])
    # Mixed codec: recompress_archive_*.py re-encodes tarballs above its size
    # floor to .tar.zst and leaves the rest .tar.gz.  Accept either; BOTH
    # present means an interrupted re-encode and is an error.
    gz=os.path.join(d, tag+".tar.gz"); zst=os.path.join(d, tag+".tar.zst")
    man=os.path.join(d, tag+".manifest.tsv")
    if os.path.exists(gz) and os.path.exists(zst):
        bad.append((tag,"both .tar.gz and .tar.zst (interrupted re-encode)")); continue
    tgz = zst if os.path.exists(zst) else gz
    if not (os.path.exists(tgz) and os.path.exists(man)):
        bad.append((tag,"missing")); continue
    try:
        if tgz.endswith(".zst"):
            import subprocess, io
            raw=subprocess.run(["zstd","-dc",tgz],capture_output=True).stdout
            with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                n=sum(1 for m in tf.getmembers() if m.isfile())
        else:
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
  echo "To execute:  CONFIRM=yes ./scripts/retire/retire_20260904.sh"
  exit 0
fi

echo "== DELETING =="
for d in $DIRS; do
  case "$d" in
    # Sample-ANCHORED on purpose.  The first draft used work-*-d97fv, which
    # also matches work-dbg25a-d97fv and refused a legitimate target: the
    # production arms are exactly work-<one of four samples>-<stage>.
    work-mcp1k-d97fv|work-mcp2k-d97fv|work-ncpi0-d97fv|work-nuecc48-d97fv|\
    work-mcp1k-d97fvpr2|work-mcp2k-d97fvpr2|work-ncpi0-d97fvpr2|work-nuecc48-d97fvpr2|\
    work-mcp1k-grp0825|work-mcp2k-grp0825|work-ncpi0-grp0825|work-nuecc48-grp0825|\
    work-sent97-*|*doc25*|work-*-d99r3prod|work-*-d99r3prodpr|\
    work-*-vtx105-base|work-em114*|work-*-pr134-f086|work-pr130r1-*|\
    work-87*|work-tfix388-r9|work-probe178410a|\
    work-mcp1k-d97prodchk|work-mcp2k-d97prodchk|work-ncpi0-d97prodchk|\
    work-nuecc48-d97prodchk|work-dbg25a-d97prodchk|\
    work-*-doc27warn|work-d31r7probe*|work-d37sbnd*)
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
# NO -maxdepth.  The first version of this line used -maxdepth 1 and reported
# 20 broken links when there were 100: the other 80 sit inside ql_evt<N>/
# subdirs.  A post-state check that is shallower than the damage is worse than
# none, because its small number reads as a small problem.
echo "broken symlinks (any depth): $(find work-*/ -xtype l 2>/dev/null | wc -l)"
echo "work dirs: $(ls -d work-*/ 2>/dev/null | wc -l)"
du -sh "$ROOT"
