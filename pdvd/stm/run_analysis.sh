#!/bin/bash
# doc pdvd/25 secs 13.6-13.7: the analysis step over a finished PR arm.
# Usage: ./stm/run_analysis.sh <tag>        (default stm1); writes stm/*.tsv, docs/pics/*.png
set -o pipefail
PDVD_DIR=$(cd "$(dirname "$0")/.." && pwd); cd "$PDVD_DIR" || exit 1
TAG=${1:-stm1}; mkdir -p docs/pics stm/logs/$TAG
echo "== census"; python3 stm/pr_census.py --tag $TAG --out stm/census_$TAG.tsv | tee stm/logs/$TAG/analysis.log
echo "== STM sample (tagger verdict required)"; python3 stm/collect_stm_sample.py --tag $TAG --plot docs/pics/pdvd_stm_sample_overlay.png | tee -a stm/logs/$TAG/analysis.log
echo "== STM sample, tagger-free (every recorded pass)"; python3 stm/collect_stm_sample.py --tag $TAG --no-tagger --suffix _notagger --plot docs/pics/pdvd_stm_sample_overlay_notagger.png | tee -a stm/logs/$TAG/analysis.log
echo "== dQ/dx vs rr + field check (tagged sample)"; python3 stm/plot_dqdx_rr.py -o docs/pics/pdvd_stm_dqdx_rr.png --tsv stm/dqdx_rr_field_check.tsv 2>&1 | tee -a stm/logs/$TAG/analysis.log
echo "== dQ/dx vs rr + field check (tagger-free sample)"; python3 stm/plot_dqdx_rr.py --points stm/sample_points_notagger.tsv --index stm/sample_index_notagger.tsv -o docs/pics/pdvd_stm_dqdx_rr_notagger.png --tsv stm/dqdx_rr_field_check_notagger.tsv 2>&1 | tee -a stm/logs/$TAG/analysis.log
echo "== contrast census (where the STM-tagged passes die)"; python3 stm/contrast_census.py --tag $TAG --out stm/contrast_census_$TAG.tsv | tee -a stm/logs/$TAG/analysis.log
echo "== dQ/dx sensitivity tiers"; for tier in "c20_loose:--min-contrast 2.0 --max-chi2 1e9 --max-shape-rms 1e9" "c15_loose:--min-contrast 1.5 --max-chi2 1e9 --max-shape-rms 1e9" "c20:--min-contrast 2.0 --max-chi2 1e9 --max-shape-rms 0.15"; do n=${tier%%:*}; f=${tier#*:}
  python3 stm/collect_stm_sample.py --tag $TAG $f --suffix _$n --plot docs/pics/pdvd_stm_sample_overlay_$n.png | grep -E 'kept' | tee -a stm/logs/$TAG/analysis.log
  python3 stm/plot_dqdx_rr.py --points stm/sample_points_$n.tsv --index stm/sample_index_$n.tsv -o docs/pics/pdvd_stm_dqdx_rr_$n.png --tsv stm/dqdx_rr_field_check_$n.tsv > /dev/null 2>&1; grep -E '^# muon|^# k_muon' stm/dqdx_rr_field_check_$n.tsv | tee -a stm/logs/$TAG/analysis.log; done
echo "== Michel, raw charge at the STM stop (with entry-end control)"; python3 stm/michel_stop_charge.py --tag $TAG --same-cluster --radius 12 -o docs/pics/pdvd_michel_stop_charge_${TAG}_sc12.png --tsv stm/michel_stop_charge_${TAG}_sc12.tsv | tee -a stm/logs/$TAG/analysis.log
echo "== Michel, tagger-independent raw stopper test"; python3 stm/raw_stopper_michel.py --tag $TAG -o docs/pics/pdvd_raw_stopper_michel_$TAG.png --tsv stm/raw_stopper_michel_$TAG.tsv 2>/dev/null | tee -a stm/logs/$TAG/analysis.log
echo "== Michel, showers at the STM stop (needs -nu dumps)"; python3 stm/michel_stop_end.py --tag $TAG -o docs/pics/pdvd_michel_stop_end_$TAG.png --tsv stm/michel_stop_end_$TAG.tsv | tee -a stm/logs/$TAG/analysis.log
echo "== Michel (flags 6-8 route, needs -nu dumps)"; python3 stm/michel_energy.py --tag $TAG -o docs/pics/pdvd_michel_energy.png --tsv stm/michel_candidates.tsv | tee -a stm/logs/$TAG/analysis.log
python3 stm/michel_energy.py --tag $TAG --loose -o docs/pics/pdvd_michel_energy_loose.png --tsv stm/michel_candidates_loose.tsv | tee -a stm/logs/$TAG/analysis.log
echo "== resources"; cat work/*_$TAG/pr_resource_*.txt | awk '{split($3,a,"=");split($4,b,"=");s+=a[2];if(b[2]>m)m=b[2];n++}END{printf "events %d mean wall %.0f s max rss %.1f GB\n",n,s/n,m}' | tee -a stm/logs/$TAG/analysis.log
