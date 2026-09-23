#!/bin/bash
# doc pdvd/117 S6: xtrack sim -> NF -> SP at a known D_L, isochronous tracks (d117_s6_tracks.py), anode 0 face 0.
set -u
O=/home/xqian/tmp/d117/s6sim; PIN=/home/xqian/tmp/d117/libpin
SIMDIR=/home/xqian/toolkit-dev/wcp-porting-img/pdvd_sim
WC=/home/xqian/toolkit-dev/local/bin/wire-cell
export WIRECELL_PATH=$SIMDIR:/home/xqian/toolkit-dev/toolkit/cfg:/home/xqian/toolkit-dev/wire-cell-data
export LD_LIBRARY_PATH=$PIN
if ldd $WC | grep -i wirecell | grep -qv "$PIN"; then echo "REFUSING: libs not from pin" >&2; exit 2; fi
arm() {   # name, extra TLAs...
    local n=$1; shift
    wcsonnet --tla-code "tracks=$(cat $O/tracks_s6_a0.json)" --tla-code anode_index=0 --tla-code lar_drift=1.48073 \
        --tla-str output_prefix=$O/$n "$@" $SIMDIR/wct-sim-xtrack-sp.jsonnet > $O/cfg/$n.json || { echo "$n compile rc=$?"; return; }
    setarch x86_64 -R $WC -l $O/$n.log -L info -c $O/cfg/$n.json > /dev/null 2>&1
    echo "$n rc=$?"
}
export -f arm; export O PIN SIMDIR WC
{
arm DL0_s1  --tla-code lar_DL=0.0001 --tla-code seed=1 &
arm DL0_s2  --tla-code lar_DL=0.0001 --tla-code seed=2 &
arm DL4_s1  --tla-code lar_DL=4.1307 --tla-code seed=1 &
arm DL4_s2  --tla-code lar_DL=4.1307 --tla-code seed=2 &
arm DL8_s1  --tla-code lar_DL=8.2614 --tla-code seed=1 &
arm DL8_s2  --tla-code lar_DL=8.2614 --tla-code seed=2 &
arm DL4_nn  --tla-code lar_DL=4.1307 --tla-code noise=false --tla-code seed=1 &
wait
} | tee $O/rc.txt
# top CRP (doc 117 sec 9.2), run the same way into $O/top with the anode-4 tracks:
#   d117_s6_tracks.py --cfg <anode_index=4 compiled driver> --anode 4 --face 0 --outdir $O/top/f0 ; cp $O/top/f0/*_a4.json $O/top/
#   for each of DL0/DL4/DL8 x seed 1,2: wcsonnet ... --tla-code anode_index=4 --tla-str output_prefix=$O/top/<arm> ... ; wire-cell -c
