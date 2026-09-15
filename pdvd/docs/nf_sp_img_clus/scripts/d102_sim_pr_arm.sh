#!/usr/bin/env bash
# doc pdvd/102 P4 -- FORK BY DUPLICATION of d101_sim_pr_arm.sh (untouched) for the near-isochronous muon set:
# run numbers 900103 (PDVD) / 900104 (PDHD), source chain tag _d102 (d102_sim_chain.sh).  Settings identical.
#
# doc pdvd/101 sec 4 -- run the PR multi-track fit on the simulated single muons (d101_sim_chain.sh
# output work/<RUN6>_<k>_d101/) as one ARM: each event gets a FRESH dir work/<RUN6>_<k>_<ARM>/ holding
# symlinks to the d101 pctree (so every arm reads byte-identical input) and the PR outputs.
#
# Why these settings (each measured on the k=0 pilots, sec 4.1):
#   flag_mains_unmatched=true   no light in the sim => no Q/L match => nothing is a main without it
#   nu_per_bundle=false         per-bundle mode drops matched_flash_gid<0 (TaggerCheckNeutrino.cxx:2297)
#   nu_per_bundle_stm_only=false, protect_stm_only_bundles=false   a contained muon is never STM-tagged
#   -nu-legacy                  the full neutrino PR (multi-track fit); -nu is the STM+Michel chain,
#                               whose STM fit exits on a fully contained cluster (TaggerCheckSTM :3614)
#   time_offset (us)            the sim frame time imaging drops (d101-time-offset.txt, sim agent sec 0)
#   trackfitting_config         the sim-matched JSON (d101_make_sim_tf_json.py)
#   dl_weights=''               geometric vertex; the SCN net is not bit-stable (CLAUDE.md M4)
#
# Usage:
#   ARM=d101b DET=pdvd PIN=/home/xqian/tmp/d101/libpin_new JOBS=4 [TF=<json>] [EXTRA_TLA="-S ..."] d101_sim_pr_arm.sh
set -uo pipefail
IMG=/home/xqian/toolkit-dev/wcp-porting-img
S=$IMG/pdvd/docs/nf_sp_img_clus/scripts
ARM=${ARM:?set ARM}
DET=${DET:?set DET to pdvd or pdhd}
PIN=${PIN:?set PIN (a private copy of local/lib)}
JOBS=${JOBS:-4}
EXTRA_TLA=${EXTRA_TLA:-}
case "$DET" in
  pdvd) RUN6=900103; TOFF=-226.598; UP=PDVD ;;
  pdhd) RUN6=900104; TOFF=-251.234; UP=PDHD ;;
  *) echo "DET must be pdvd or pdhd" >&2; exit 2 ;;
esac
TF=${TF:-$IMG/pdvd/docs/nf_sp_img_clus/figs/101_tf_sim_${DET}.json}
[ -s "$PIN/libWireCellClus.so" ] || { echo "REFUSING: no pin at $PIN" >&2; exit 2; }
[ -s "$TF" ] || { echo "REFUSING: no track-fitting JSON $TF" >&2; exit 2; }
D=$IMG/$DET
if ls -d "$D"/work/${RUN6}_*_"$ARM" >/dev/null 2>&1; then
    echo "REFUSING: $D/work/${RUN6}_*_$ARM already exists (CLAUDE.md M13: new run, new tag)" >&2; exit 2
fi
n=0
for src in "$D"/work/${RUN6}_*_d102; do
    k=$(basename "$src" | sed -E "s/^${RUN6}_([0-9]+)_d102$/\1/")
    # guard: the recorded time offset must be the one this arm passes
    got=$(awk -F= '/time_offset_us|time_offset_pr_us|PR_TLA/{print}' "$src/d101-time-offset.txt" 2>/dev/null | grep -o -- "-[0-9.]*" | head -1)
    dst="$D/work/${RUN6}_${k}_${ARM}"; mkdir -p "$dst"
    for f in "$src"/pctree-evt*.tar.gz "$src"/pctree-evt*.tlas "$src"/d101-time-offset.txt "$src"/d101-sim-provenance.txt; do
        ln -s "$(readlink -f "$f")" "$dst/"
    done
    n=$((n+1))
done
echo "[$ARM] $DET staged $n events; tf=$TF extra=<${EXTRA_TLA:-none}> pin md5 $(md5sum "$PIN/libWireCellClus.so" | cut -c1-8)"
TLA="-S time_offset=$TOFF -S flag_mains_unmatched=true -S nu_per_bundle=false -S nu_per_bundle_stm_only=false -S protect_stm_only_bundles=false -A trackfitting_config=$TF -A dl_weights= $EXTRA_TLA"
( cd "$D" && env LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}" WCT_PYLIB=off \
      ${UP}_ALLOW_NO_QLMATCH=1 ${UP}_ALLOW_NO_ASSOC=1 ${UP}_MAX_JOBS="$JOBS" ${UP}_PR_TLA="$TLA" \
      ./run_pr_evt.sh -nu-legacy -stm-fit -s "$ARM" "$RUN6" all )
rc=$?
nout=$(ls "$D"/work/${RUN6}_*_"$ARM"/tracking-pr.root 2>/dev/null | wc -l)
echo "[$ARM] $DET rc=$rc events_with_tracking-pr.root=$nout of $n"
[ "$nout" -eq "$n" ] || echo "[$ARM] *** only $nout of $n events produced output -- rc=$rc is not a pass ***"
