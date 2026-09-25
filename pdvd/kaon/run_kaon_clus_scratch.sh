#!/bin/bash
# doc pdvd/118 sec 3.3: clustering + Q/L for the top-only run 39305 WITHOUT
# toolkit C++ changes.  Two QLMatching defects block the production config on a
# single (top) joint input (both documented in the doc, both left for the owner):
#   (1) fit_round2 null-deref (QLMatching.cxx:2712) when the post-fit cull
#       removed a cluster's best pair  -> run with PDVD_QL_POSTCULL=0;
#   (2) write_opflash_pc always emits the input-1 "time1" display column when
#       per-side trigger_offsets are set -> pad trigger_offsets to [top, top]
#       (input 0 = the top group; index 1 feeds only that display column).
# Everything else is the production run_clus_evt.sh config (compile-only mode,
# then the same wire-cell invocation: GOGC=off + tcmalloc).
# Usage: ./run_kaon_clus_scratch.sh [evt ...]
KDIR=$(cd "$(dirname "$0")" && pwd)
PDVD=$(dirname "$KDIR")
export PDVD_INPUT_DATA=$PDVD/input_kaon
MAXJ=${KAON_MAX_JOBS:-5}
if [ $# -gt 0 ]; then EVTS=("$@"); else
    mapfile -t EVTS < <(ls -d $PDVD_INPUT_DATA/run039305/evt_* | sed 's/.*evt_//' | sort -n)
fi
one() {
    local e=$1 W=$PDVD/work/039305_$1 J rc
    cd $PDVD
    PDVD_QL_POSTCULL=0 PDVD_CLUS_COMPILE_ONLY=1 PDVD_KEEP_CFG=1 \
        ./run_clus_evt.sh -a 4,5,6,7 -calib -save-pctree 39305 $e || return $?
    J="$W/.wct-clus_a4,5,6,7.json"
    python3 - "$J" <<'PY' || return 7
import json, sys
p = sys.argv[1]; cfg = json.load(open(p)); n = 0
for x in cfg:
    if x.get('type') == 'QLMatching':
        t = x['data']['trigger_offsets']
        assert len(t) == 1 and x['data']['anode_pd_channels'] == [[]], (t, x['data']['anode_pd_channels'])
        x['data']['trigger_offsets'] = [t[0], t[0]]; n += 1
assert n == 1
json.dump(cfg, open(p, 'w'))
print('patched trigger_offsets ->', [t[0], t[0]])
PY
    env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 GOGC=off wire-cell \
        -l stderr -l "$W/wct_clus_039305_${e}_a4,5,6,7.log:debug" -L debug -c "$J"
    rc=$?
    return $rc
}
nfail=0
for e in "${EVTS[@]}"; do
    while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do wait -n || nfail=$((nfail+1)); done
    ( one $e > $KDIR/logs/clusscratch_$e.log 2>&1; rc=$?; echo "rc=$rc" >> $KDIR/logs/clusscratch_$e.log; exit $rc ) &
done
while [ $(jobs -rp | wc -l) -gt 0 ]; do wait -n || nfail=$((nfail+1)); done
echo "run_kaon_clus_scratch: ${#EVTS[@]} events, $nfail failed"
grep -H '^rc=' $(for e in "${EVTS[@]}"; do echo $KDIR/logs/clusscratch_$e.log; done)
exit $nfail
