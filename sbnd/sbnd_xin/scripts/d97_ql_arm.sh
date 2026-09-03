#!/bin/bash
# doc 97 -- a Q/L-STAGE arm over an existing stage-A root.
#
# Why this exists.  Every A/B harness in this tree (doc94r3_arm.sh,
# stm_campaign/run_round.sh, ...) re-runs only the PR tagger tail on top of a
# frozen Q/L root, because every knob those rounds touched lives in the PR
# chain.  `track_recarve` lives in ClusteringSeparate, which runs inside the
# Q/L job, so this round needs the Q/L stage re-run and the PR chain rebuilt on
# top of it.
#
# The group-mode stage-A driver (run_chain_group.sh --from ql) cannot be used:
# doc 89's retire round pruned the g<K>/ group scratch out of the four
# work-<s>-grp0825 roots, so the group imaging checkpoint is gone.  The
# PER-EVENT imaging (evt<ID>/icluster-apa{0,1}-{active,masked}.npz) survives,
# and doc 81 sec 7 gated group-mode products byte-identical to per-event ones
# (24536/24536), so a per-event Q/L re-run off those npz is the same
# reconstruction.  Proven again for this round: see docs/97 sec 2.
#
# Usage:
#   ROOT=$PWD/work-<sample>-<tag> ./scripts/d97_ql_arm.sh <sample> \
#        [-j N] [-f events.txt] [<evt> ...]
#   sample: ncpi0 | nuecc48 | mcp1k | mcp2k
#   no event list  => every evt<ID> present in the source stage-A root
#
# Env:
#   ROOT      output root (REQUIRED, must not exist -- M13)
#   SRC       stage-A root supplying imaging   (default work-<sample>-grp0825)
#   QL_EXTRA  flags appended to run_ql_evt.sh  (default -save-pctree, which the
#             PR chain needs; word-split on purpose)
#   LIBSNAP   pinned library dir (default ~/tmp/doc94r3b-libsnap; a peer session
#             shares this tree and rebuilds local/lib underneath a long arm)
#   D97_JOBS  concurrency (default 8; CLAUDE.md M5 -- check /proc/loadavg after)
#
# Every event writes $ROOT/.status/<evt> with its rc.  The driver exits non-zero
# unless EVERY requested event has rc=0: batch_summary() in _runlib.sh returns 0
# when at least ONE event succeeded, which is not a safety net.
#
# Internal: ./d97_ql_arm.sh --worker <sample> <evt> <inputdir> <idx>
set -u
cd -P "$(dirname "$0")/.." || exit 1
SX=$PWD
IN=$SX/input_files_reco1
S1K=$IN/staged-mcp2025c-1000evt
S2K=$IN/staged-mcp2025c-2nd-2000evt

# --- one event -------------------------------------------------------------
if [ "${1:-}" = "--worker" ]; then
    smp=$2; evt=$3; indir=$4; idx=$5
    st=$ROOT/.status/$evt
    src=${SRC:-$SX/work-$smp-grp0825}
    if [ ! -d "$src/evt$evt" ]; then
        echo "rc=91 evt=$evt no-imaging in $src" > "$st"; exit 0
    fi
    ln -sfn "$src/evt$evt" "$ROOT/evt$evt"
    export LD_LIBRARY_PATH=${LIBSNAP:-$HOME/tmp/doc94r3b-libsnap}:${LD_LIBRARY_PATH:-}
    t0=$(date +%s)
    ( SBND_WORK_ROOT=$ROOT SBND_INPUT_DIR=$indir \
        setarch x86_64 -R "$SX/run_ql_evt.sh" data ${QL_EXTRA--save-pctree} "$idx"
    ) > "$ROOT/.log/$evt.log" 2>&1
    rc=$?
    z=$ROOT/ql_evt$evt/mabc-all-apa.zip
    # a 0-byte / missing zip is a failure even when wire-cell exited 0
    if [ ! -s "$z" ] && [ "$rc" -eq 0 ]; then rc=93; fi
    nfire=$(grep -ac "Separate track_recarve" "$ROOT/.log/$evt.log" 2>/dev/null); : "${nfire:=0}"
    printf 'rc=%s evt=%s wall=%ss zip=%s fired=%s\n' \
        "$rc" "$evt" "$(( $(date +%s) - t0 ))" "$(stat -c %s "$z" 2>/dev/null || echo 0)" "$nfire" > "$st"
    exit 0
fi

# --- driver ----------------------------------------------------------------
SMP=${1:-}; shift || true
case "$SMP" in
    ncpi0)   INDIR=$IN/extracted-ncpi0 ;;
    nuecc48) INDIR=$IN/extracted-2025fall-48evt-fsprod ;;   # -fsprod: the set work-nuecc48-grp0825 was built from (docs/work-tags.md:227); the bare extracted-2025fall-48evt has a different FrameShiftInfo and changes every flash match
    mcp1k)   INDIR=$S1K ;;
    mcp2k)   INDIR=$S2K ;;
    *) echo "usage: ROOT=... $0 <ncpi0|nuecc48|mcp1k|mcp2k> [-j N] [-f f] [evt...]" >&2; exit 2 ;;
esac
J=${D97_JOBS:-8}
evts=()
while [ $# -gt 0 ]; do
    case "$1" in
        -j) J=$2; shift 2 ;;
        -f) while read -r _e; do [ -n "$_e" ] && evts+=("$_e"); done < "$2"; shift 2 ;;
        *)  evts+=("$1"); shift ;;
    esac
done
[ -n "${ROOT:-}" ] || { echo "REFUSE: set ROOT explicitly (M13)" >&2; exit 2; }
[ -e "$ROOT" ] && { echo "REFUSE: $ROOT exists (M13: never write into an existing label)" >&2; exit 2; }
SRC=${SRC:-$SX/work-$SMP-grp0825}
[ -d "$SRC" ] || { echo "no source stage-A root: $SRC" >&2; exit 2; }
if [ ${#evts[@]} -eq 0 ]; then
    while read -r _d; do evts+=("${_d#evt}"); done < <(cd "$SRC" && ls -d evt* 2>/dev/null)
fi
[ ${#evts[@]} -gt 0 ] || { echo "no events" >&2; exit 2; }
mkdir -p "$ROOT/.status" "$ROOT/.log"

# event -> (input dir, idx).  mcp1k/mcp2k are staged one event per e<entry>
# directory, so idx is always 1.  ncpi0/nuecc48 are ONE bundled archive, so the
# idx is the event's position in run_ql_evt.sh's own listing -- taken once here
# (each call costs a full bz2 listing, 17 s on the 48-event set).
declare -A IDXOF=()
case "$SMP" in
  ncpi0|nuecc48)
    while read -r i e; do IDXOF[$e]=$i; done < <(
        SBND_WORK_ROOT=$ROOT SBND_INPUT_DIR=$INDIR "$SX/run_ql_evt.sh" data 2>/dev/null |
        awk '/^ *[0-9]+ -> /{print $1, $3}')
    ;;
esac

echo "=== doc 97 Q/L arm ==="
echo "  sample=$SMP  root=$ROOT  src=$SRC  jobs=$J  events=${#evts[@]}"
echo "  input=$INDIR"
echo "  QL_EXTRA='${QL_EXTRA--save-pctree}'  libsnap=${LIBSNAP:-$HOME/tmp/doc94r3b-libsnap}"
# $SX/../.. is wcp-porting-img, NOT the toolkit -- this line said "toolkit HEAD"
    # and printed the wcp commit for every arm ever run with it.  Chasing
    # 0fce23d4 in the toolkit (it is a wcp doc-96 commit) is how it was found,
    # while refreshing doc 92.  Both repos are printed now, each named correctly,
    # and the pinned libsnap is what actually decides the physics anyway.
    echo "  wcp HEAD=$(git -C "$SX/../.." rev-parse --short HEAD 2>/dev/null)  toolkit HEAD=$(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse --short HEAD 2>/dev/null)"
echo "  local/lib libWireCellClus.so mtime=$(stat -c %y /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so)"
echo "  started $(date -Is)"
t0=$(date +%s)

work=()
for e in "${evts[@]}"; do
    case "$SMP" in
      mcp1k) i=$(awk -F'\t' -v e="$e" '$4==e{print $1; exit}' "$S1K/entry_event_map.tsv")
             [ -n "$i" ] || { echo "  WARN evt $e not in mcp1k map"; continue; }
             work+=("$e $S1K/e$i 1") ;;
      mcp2k) i=$(awk -F'\t' -v e="$e" '$4==e{print $1; exit}' "$S2K/entry_event_map.tsv")
             [ -n "$i" ] || { echo "  WARN evt $e not in mcp2k map"; continue; }
             work+=("$e $S2K/e$i 1") ;;
      *)     i=${IDXOF[$e]:-}
             [ -n "$i" ] || { echo "  WARN evt $e not in $SMP listing"; continue; }
             work+=("$e $INDIR $i") ;;
    esac
done
export ROOT SRC QL_EXTRA LIBSNAP SMP 2>/dev/null || true
printf '%s\n' "${work[@]}" | \
    xargs -P "$J" -n 3 bash -c 'exec "$0" --worker "'"$SMP"'" "$1" "$2" "$3"' "$SX/scripts/d97_ql_arm.sh"

echo "  finished $(date -Is) ($(( $(date +%s) - t0 ))s)"
nreq=${#work[@]}
nok=$(grep -h . "$ROOT"/.status/* 2>/dev/null | grep -c '^rc=0 ')
nfired=$(grep -h . "$ROOT"/.status/* 2>/dev/null | grep -vc ' fired=0$')
echo "  requested=$nreq  rc=0 on $nok  events with a track_recarve fire: $nfired"
bad=$(grep -h . "$ROOT"/.status/* 2>/dev/null | grep -v '^rc=0 ')
if [ -n "$bad" ] || [ "$nok" -ne "$nreq" ]; then
    echo "  ############ FAILED / MISSING EVENTS -- do not gate on this arm"
    printf '%s\n' "$bad" | sed 's/^/  # /'
    exit 1
fi
exit 0
