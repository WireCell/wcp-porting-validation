#!/bin/bash
# doc pdvd/119 knob-ON arm on the top-only run 39305 (the kaon candidates).
# Same scratch settings as doc 118's run_kaon_clus_scratch.sh (post-cull OFF,
# trigger_offsets padded to [top, top] -- the two QLMatching single-side
# defects are still unfixed), but into FRESH dirs:
#   light: work/039305_light<evt>_d119beam   (PDVD_BEAM_LABEL=1)
#   clus : work/039305_<evt>_d119beamscratch (imaging symlinked from the
#          canonical work/039305_<evt>/, which is read, never written)
# Runs on the install prefix PREFIX (arm B of doc 119).
# Usage: PREFIX=<install> ./run_d119_kaon_beam_scratch.sh [evt ...]
set -u
KDIR=$(cd "$(dirname "$0")" && pwd)
PDVD=$(dirname "$KDIR")
PREFIX=${PREFIX:?set PREFIX}
export PATH="$PREFIX/bin:$PATH" LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"
export PDVD_INPUT_DATA=$PDVD/input_kaon
TAG=d119beamscratch
MAXJ=${KAON_MAX_JOBS:-3}
if [ $# -gt 0 ]; then EVTS=("$@"); else
    mapfile -t EVTS < <(ls -d $PDVD_INPUT_DATA/run039305/evt_* | sed 's/.*evt_//' | sort -n)
fi
one() {
    local e=$1 W=$PDVD/work/039305_${1}_$TAG J rc
    cd $PDVD
    if [ ! -s "work/039305_light${e}_d119beam/opflash_pdvd-wct.tar.gz" ]; then
        PDVD_BEAM_LABEL=1 ./run_light_evt.sh -s _d119beam \
            -f $PDVD_INPUT_DATA/light/np02vd_raw_run039305_evt${e}_rawwf.root 39305 $e || return 5
    fi
    [ -e "$W" ] && { echo "$W exists -- refusing (M13)"; return 3; }
    mkdir -p "$W"
    for f in work/039305_$e/clusters-apa-anode?-ms-*.tar.gz work/039305_$e/img-provenance.txt; do ln -s "$PDVD/$f" "$W/"; done
    PDVD_BEAM_LABEL=1 PDVD_LIGHT_SUFFIX=_d119beam PDVD_QL_POSTCULL=0 PDVD_CLUS_COMPILE_ONLY=1 PDVD_KEEP_CFG=1 \
        ./run_clus_evt.sh -s $TAG -a 4,5,6,7 -calib 39305 $e || return $?
    J="$W/.wct-clus_a4,5,6,7.json"
    python3 - "$J" <<'PY' || return 7
import json, sys
p = sys.argv[1]; cfg = json.load(open(p)); n = 0
for x in cfg:
    if x.get('type') == 'QLMatching':
        t = x['data']['trigger_offsets']
        assert len(t) == 1 and x['data']['anode_pd_channels'] == [[]], (t, x['data']['anode_pd_channels'])
        x['data']['trigger_offsets'] = [t[0], t[0]]; n += 1
    if x.get('type') == 'MultiAlgBlobClustering' and 'bee_beam_window_us' in x.get('data', {}):
        print('bee_beam_window_us', x['data']['bee_beam_window_us'])
assert n == 1
json.dump(cfg, open(p, 'w'))
print('patched trigger_offsets ->', [t[0], t[0]])
PY
    setarch x86_64 -R env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 GOGC=off wire-cell \
        -l stderr -l "$W/wct_clus_039305_${e}_a4,5,6,7.log:debug" -L debug -c "$J"
}
mkdir -p $KDIR/logs
nfail=0
for e in "${EVTS[@]}"; do
    while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do wait -n || nfail=$((nfail+1)); done
    ( one $e > $KDIR/logs/d119scratch_$e.log 2>&1; rc=$?; echo "rc=$rc" >> $KDIR/logs/d119scratch_$e.log; exit $rc ) &
done
while [ $(jobs -rp | wc -l) -gt 0 ]; do wait -n || nfail=$((nfail+1)); done
echo "run_d119_kaon_beam_scratch: ${#EVTS[@]} events, $nfail failed"
grep -H '^rc=' $(for e in "${EVTS[@]}"; do echo $KDIR/logs/d119scratch_$e.log; done)
exit $nfail
