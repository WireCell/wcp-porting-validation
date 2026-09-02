#!/bin/bash
# doc 96 sec 5 -- census of ClusteringSeparate's gate quantities over the whole
# doc-95 sample, to answer: is JudgeSeparateDec_2 inert on IN-TIME SBND
# clusters in general, or only on the owner's two events?
#
# Per-event (not `all`): WCT_SEP_DEBUG writes to std::cout, which run_ql_evt.sh
# does NOT fold into the per-event wct_ql_evt<ID>.log, so 20 parallel jobs
# would interleave into one untagged stream.  Sequential, one log per event.
# ~20 s/event => ~9 min for all 25.
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LD_LIBRARY_PATH=$HOME/tmp/doc94r3b-libsnap:${LD_LIBRARY_PATH:-}
export WCT_SEP_DEBUG=1
LOGD=/home/xqian/tmp/d96/census; mkdir -p "$LOGD"

for grp in a b; do
    export SBND_INPUT_DIR=$BASE/input_files_reco1/extracted-dbg25$grp
    export SBND_WORK_ROOT=$BASE/work-dbg25$grp-sepcensus95
    mkdir -p "$SBND_WORK_ROOT"
    # imaging is the production one, symlinked (M11)
    for d in "$BASE/work-dbg25$grp-ql"/evt*; do
        [ -d "$d" ] && ln -sfn "$d" "$SBND_WORK_ROOT/$(basename "$d")"
    done
    n=$(./run_ql_evt.sh mc 2>/dev/null | grep -c '^ *[0-9]* -> ')
    echo "=== group $grp: $n events  $(date -Is)"
    for i in $(seq 1 "$n"); do
        evt=$(./run_ql_evt.sh mc 2>/dev/null | awk -v i="$i" '$1==i{print $3}')
        setarch x86_64 -R ./run_ql_evt.sh mc "$i" > "$LOGD/$grp-evt$evt.log" 2>&1
        echo "  idx $i evt $evt rc=$?  SEPDBG=$(grep -c SEPDBG "$LOGD/$grp-evt$evt.log")"
    done
done
echo "=== done $(date -Is)"
