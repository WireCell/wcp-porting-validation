#!/bin/bash
# doc pdvd/100 round 2 -- the readout-edge exemption's arms and the OFF gates' arms, in waves (d100/prereg_round2.md sec 2).
#
#   setsid nohup bash d100r2_arms.sh > /home/xqian/tmp/p100/r2_arms.log 2>&1 < /dev/null &
#
# PDVD arms: d100_arms.sh WAVE=pr (PR only on the source arm's pctree, its .tlas copied), PIN=libpin_p100b (Clus ef0822ec,
# the build with readout_edge_defer / readout_edge_require_michel).
#   X2  p100boff  p99rwon  p100c's config                                   == p100c (pin p96) on 120/120
#   X4  p100bg    p99rwon  + stm_readout_edge_guard=false                   \  identical PR branches
#       p100bd    p99rwon  + stm_readout_edge_defer=true                    /  (the deferral IS the guard-off flow)
#       p100bx    p99rwon  + defer + readout_edge_require_michel             the gain-flip candidate with the exemption
#       p100bxp   p99wflip production config + defer + require_michel        what the exemption does to production
# PDHD X3: d53_run_arms.sh DET=pdhd SRC=d16hnu, production config, h100a (pin p96) vs h100b (pin p100b).
# No waf target may run while this is live (d53_run_arms.sh header).  Every arm refuses an existing tag (M13).
set -u
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
L=/home/xqian/tmp/p100
PB=$L/libpin_p100b
PA=/home/xqian/tmp/p96/libpin_p96
C='-S stm_recomb_C=0.8630'
D='-S stm_readout_edge_defer=true'
Q='-S stm_michel_extra={readout_edge_require_michel:true}'

pdvd() {   # arm src tla
    PIN=$PB WAVE=pr ARM=$1 SRC=$2 PR_TLA="$3" bash "$X/d100_arms.sh" > "$L/arm_$1.nohup" 2>&1
    echo "[$(date -Is)] $1 rc=$? $(tail -2 "$L/arm_$1/arm.log" 2>/dev/null | head -1)"
}
pdhd() {   # arm pin
    ARM=$1 DET=pdhd SRC=d16hnu JOBS=6 PIN=$2 LOGD=$L/arm_$1 bash "$X/d53_run_arms.sh" > "$L/arm_$1.nohup" 2>&1
    echo "[$(date -Is)] $1 rc=$? $(grep 'complete ' "$L/arm_$1.nohup" | tail -1)"
}

md5sum "$PB"/libWireCell*.so "$PA"/libWireCell*.so > "$L/r2_pin_md5_before.txt"
echo "[$(date -Is)] wave 1: X2 off identity + X4 guard-off control"
pdvd p100boff p99rwon "$C" &
pdvd p100bg p99rwon "$C -S stm_readout_edge_guard=false" &
wait
echo "[$(date -Is)] wave 2: X4 deferral + the exemption"
pdvd p100bd p99rwon "$C $D" &
pdvd p100bx p99rwon "$C $D $Q" &
wait
echo "[$(date -Is)] wave 3: the exemption on production + PDHD OFF on the old pin"
pdvd p100bxp p99wflip "$D $Q" &
pdhd h100a "$PA" &
wait
echo "[$(date -Is)] wave 4: PDHD OFF on the new pin"
pdhd h100b "$PB"
md5sum "$PB"/libWireCell*.so "$PA"/libWireCell*.so > "$L/r2_pin_md5_after.txt"
cmp -s "$L/r2_pin_md5_before.txt" "$L/r2_pin_md5_after.txt" && echo "[$(date -Is)] pins unchanged" \
    || echo "[$(date -Is)] *** A PIN CHANGED DURING THE ARMS -- every comparison is VOID ***"
echo "[$(date -Is)] done"
