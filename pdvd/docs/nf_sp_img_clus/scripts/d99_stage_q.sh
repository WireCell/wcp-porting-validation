#!/bin/bash
# doc pdvd/99 -- stage a clustering input dir the way production's d51vclus was staged (scripts/stage_ql_tag.sh:
# the 16 imaging archives + img-provenance.txt, NOTHING else), so that run_clus_evt.sh takes the same code path.
#
#   SRC=p98voff DST=p98voffq d99_stage_q.sh
#   SRC=p98von  DST=p98vonq  d99_stage_q.sh
#   SRC=d27fresh DST=p98kq  EVENTS="039349_81 ..." d99_stage_q.sh     # clustering-drift control
#
# Why: run_clus_evt.sh:273-282 reads readout_window_ticks from protodune-sp-dnnroi-frames-anode*.tar.bz2 in the
# clustering input dir when one is there (6400 on these frames) and otherwise falls back to 10000.  d51vclus held no
# frames, so production ran every event at 10000; an arm dir that also holds the SP frames silently runs at 6400
# (84/120 events of p98voff) and the PR readout_edge_guard then rejects stops near tick 6400 (363 vs 264 firings).
# The imaging archives are linked (readlink -f, like stage_ql_tag.sh / d53_run_arms.sh), never copied; an existing
# DST dir is refused.
set -u
SRC=${SRC:?set SRC}; DST=${DST:?set DST}
W=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work
if [ -n "${EVENTS:-}" ]; then evs=$EVENTS; else
    evs=$(grep -v '^#' /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/stm/events.txt | awk 'NF>=2 {printf "%06d_%s\n", $1, $2}'); fi
n=0; bad=0
for e in $evs; do
    s=$W/${e}_$SRC; d=$W/${e}_$DST
    [ -e "$d" ] && { echo "REFUSE existing $d"; bad=1; continue; }
    na=$(ls "$s"/clusters-apa-anode*-ms-active.tar.gz "$s"/clusters-apa-anode*-ms-masked.tar.gz 2>/dev/null | wc -l)
    [ "$na" = 16 ] || { echo "SKIP $e: $na imaging archives in $s"; bad=1; continue; }
    mkdir -p "$d"
    for f in "$s"/clusters-apa-anode*-ms-active.tar.gz "$s"/clusters-apa-anode*-ms-masked.tar.gz; do
        ln -s "$(readlink -f "$f")" "$d/$(basename "$f")"
    done
    [ -e "$s/img-provenance.txt" ] && ln -s "$(readlink -f "$s/img-provenance.txt")" "$d/img-provenance.txt"
    n=$((n+1))
done
echo "staged $n event dirs $SRC -> $DST (imaging archives + img-provenance only); problems: $bad"
exit $bad
