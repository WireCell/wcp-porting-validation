#!/bin/bash
# doc pr/142 -- everything that runs AFTER the two arms land.
set -u
cd "$(dirname "$0")/.." || exit 1
O=${PR142_OUT:-/home/xqian/tmp/pr142}
mkdir -p "$O"
for t in empre0901 prod0901; do
    mkdir -p products/$t
    for s in nuecc48 ncpi0 mcp1k mcp2k; do
        # a table is only reusable if it has one row per pr_evt dir.  A
        # timed-out / killed pr_scores_table.py leaves a SHORT file behind and
        # `[ -f ]` alone would happily reuse it (seen 2026-09-01: 966 rows of
        # 1000, which then read out as "34 events only in the other arm").
        want=$(ls -d work-$s-$t/pr_evt* 2>/dev/null | wc -l)
        have=$(( $(wc -l < products/$t/$s-scores-$t.tsv 2>/dev/null || echo 1) - 1 ))
        [ "$want" -gt 0 ] && [ "$have" -eq "$want" ] && continue
        echo "  building products/$t/$s-scores-$t.tsv ($have of $want rows)"
        python3 pr_scores_table.py --root work-$s-$t --sample $s \
            --out products/$t/$s-scores-$t.tsv || exit 1
    done
done
echo "== systematic A/B =="
python3 scripts/pr142_campaign_ab.py \
    --a products/empre0901/*.tsv --b products/prod0901/*.tsv \
    --label-a empre0901 --label-b prod0901 --top 40 \
    --movers-tsv docs/pr/pr142-movers.tsv \
    --summary-tsv docs/pr/pr142-population.tsv > "$O/ab-empre-prod.txt" 2>&1
echo "  rc=$? -> $O/ab-empre-prod.txt"
echo "== Proof B: the restored arm vs the committed pre-campaign product =="
python3 scripts/pr142_campaign_ab.py \
    --a products/prod0825/*.tsv --b products/empre0901/*.tsv \
    --label-a prod0825 --label-b empre0901 --top 25 \
    --movers-tsv docs/pr/pr142-proofB-movers.tsv > "$O/ab-0825-empre.txt" 2>&1
echo "  rc=$? -> $O/ab-0825-empre.txt"
echo "== the campaign end to end, as production saw it =="
python3 scripts/pr142_campaign_ab.py \
    --a products/prod0825/*.tsv --b products/prod0901/*.tsv \
    --label-a prod0825 --label-b prod0901 --top 25 > "$O/ab-0825-prod.txt" 2>&1
echo "  rc=$? -> $O/ab-0825-prod.txt"
echo "== stage-resolved runtime + memory =="
python3 scripts/pr142_perf.py work-mcp1k-empre0901 work-mcp1k-prod0901 \
    --top 20 --jobs 8 --tsv docs/pr/pr142-perf-mcp1k.tsv > "$O/perf-mcp1k.txt" 2>&1
echo "  rc=$? -> $O/perf-mcp1k.txt"
python3 scripts/pr142_perf.py work-nuecc48-empre0901 work-nuecc48-prod0901 \
    --top 20 --jobs 8 --tsv docs/pr/pr142-perf-nuecc48.tsv > "$O/perf-nuecc48.txt" 2>&1
echo "  rc=$? -> $O/perf-nuecc48.txt"
echo "== Proof C: the new production IS the validated operating point =="
for s in nuecc48 ncpi0 mcp1k mcp2k; do
    python3 scripts/pr85_hash_gate.py work-pr141r1-off-$s work-$s-prod0901 \
        > "$O/gateC-$s.txt" 2>&1
    echo "  gateC $s rc=$? :: $(tail -2 "$O/gateC-$s.txt" | tr '\n' ' ')"
done
echo "== sentinels on the new production arm =="
python3 scripts/pr127_sentinels.py --arms 'work-*-prod0901' > "$O/sentinels.txt" 2>&1
echo "  rc=$? -> $O/sentinels.txt"
echo ANALYZE DONE
