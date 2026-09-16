#!/bin/bash
# doc pr/149 round 2: every secondary table of sec 13 from the finished arms (Stage 1 pr149r2*, Stage 2 pr149s2r2*, trace pr149r2t*).
# Usage: bash scripts/analysis/pr149/r2_secondary.sh > <log>
set -u
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
D=docs/pr/149_figs
python3 scripts/analysis/pr149/r2_zzi_sign.py --stage 2 --pairs pr149s2r2rscs:pr149s2r2s0 pr149s2r2rs:pr149s2r2s0 pr149s2r2rscs:pr149s2r2cs pr149s2r2cs:pr149s2r2s0 pr149s2r2rscs:pr149s2r2rs > $D/149_r2_primary_stage2.txt 2>&1; echo "primary2 rc=$?"
python3 scripts/analysis/pr149/r2_zzi_sign.py --stage 1 --pairs pr149r2rscs:pr149r2s0 pr149r2rs:pr149r2s0 pr149r2rscs:pr149r2cs pr149r2cs:pr149r2s0 pr149r2rscs:pr149r2rs > $D/149_r2_primary_stage1.txt 2>&1; echo "primary1 rc=$?"
for c in s2r2rs s2r2rscs s2r2cs; do python3 scripts/analysis/pr149/pr149_metrics.py compare --a pr149s2r2s0 --b pr149$c --samples mcp1k mcp2k --manifest-strata $D/149_stage2_selection.tsv --movers $D/149_r2_s2_movers_$c.tsv > $D/149_r2_s2_compare_$c.txt 2>&1; echo "compare $c rc=$?"; done
python3 scripts/analysis/pr149/vertex_tolerance.py --stage 2 --base pr149s2r2s0 --arms pr149s2r2cs pr149s2r2rs pr149s2r2rscs > $D/149_r2_s2_vertex_tolerance.txt 2>&1; echo "vt rc=$?"
python3 scripts/analysis/pr149/q1_verdict.py --base1 pr149r2s0 --base2 pr149s2r2s0 --cells r2cs:s2r2cs r2rs:s2r2rs r2rscs:s2r2rscs > $D/149_r2_q1_verdict.txt 2>&1; echo "q1 rc=$?"
python3 scripts/analysis/pr149/r2_topology.py --base pr149s2r2s0 --arms pr149s2r2cs pr149s2r2rs pr149s2r2rscs --samples mcp1k mcp2k > $D/149_r2_s2_topology.txt 2>&1; echo "topo rc=$?"
python3 scripts/analysis/pr149/r2_local_jitter.py --stage 2 --pairs pr149s2r2cs:pr149s2r2s0 pr149s2r2rs:pr149s2r2s0 pr149s2r2rscs:pr149s2r2s0 pr149s2kf:pr149s2s0 > $D/149_r2_local_jitter_stage2.txt 2>&1; echo "jit2 rc=$?"
python3 scripts/analysis/pr149/r2_local_jitter.py --stage 1 --pairs pr149r2cs:pr149r2s0 pr149r2rs:pr149r2s0 pr149r2rscs:pr149r2s0 pr149kf:pr149s0 > $D/149_r2_local_jitter_stage1.txt 2>&1; echo "jit1 rc=$?"
python3 scripts/analysis/pr149/r2_assoc_trace.py --arms pr149r2ts0 pr149r2tcs pr149r2trs pr149r2trscs --events-manifest $D/manifest_r2trace > $D/149_r2_assoc_trace.txt 2>&1; echo "trace rc=$?"
for c in r2s0 r2cs r2rs r2rscs; do python3 scripts/analysis/pr149/sentinels_tolerant.py --arms work-mcp1k-pr149s2$c work-mcp2k-pr149s2$c work-nuecc48-pr149$c work-ncpi0-pr149$c > $D/149_r2_sentinels_$c.txt 2>&1; echo "sent $c rc=$?"; done
echo SECONDARY_DONE
