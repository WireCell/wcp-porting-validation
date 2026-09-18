#!/bin/bash
# sbnd_xin/docs/109 rev 4: one arm of the nu_bundle_flash_group gate on the
# sbnd_xin data samples.
#
# Usage: scripts/d109r4_arms.sh <label> <libsnap> <manifest> [VAR=value ...]
#   label     arm tag; outputs go to work-<sample>-<label>/ (a FRESH dir, M13)
#   libsnap   pinned copy of local/lib (LD_LIBRARY_PATH), e.g. ~/tmp/d109r4-libsnap/new
#   manifest  m267      the doc 109 byte-gate manifest: nuecc48 (48) + ncpi0 (19)
#                       + the first 200 mcp1k events (docs/109_logs/events_mcp1k200.txt)
#             eligible  the events nu_bundle_flash_group can touch, all four samples
#                       (docs/109_logs/r4/events_eligible_<sample>.txt, from
#                       scripts/d109r4_eligible_census.py)
#             m267+eligible   the union of the two, one run per sample
#             all       every event of all four samples (3067)
#   VAR=val   extra env for run_pr_chain_batch.sh, e.g. SBND_NO_DL=1
#             SBND_NU_BUNDLE_FLASH_GROUP=1 PR_CFG_TREE=<cfg dir>
#
# Stage-A input: the doc 102 arms work-<sample>-d102m (read-only).  setarch -R
# for determinism (M4); PR_EXTRA_STAGES=pr_display as in production.
set -u
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
LABEL=${1:?usage: d109r4_arms.sh <label> <libsnap> <manifest> [VAR=value ...]}
PIN=${2:?usage: d109r4_arms.sh <label> <libsnap> <manifest> [VAR=value ...]}
MANIFEST=${3:?usage: d109r4_arms.sh <label> <libsnap> <manifest> [VAR=value ...]}
shift 3
cd "$SX" || exit 1
for kv in "$@"; do export "$kv"; done
export LD_LIBRARY_PATH=$PIN:${LD_LIBRARY_PATH:-}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES-pr_display}
export PR_JOBS=${PR_JOBS:-16}
LOGD=$HOME/tmp/d109r4
mkdir -p "$LOGD"
M267=$SX/docs/109_logs/events_mcp1k200.txt
ELIG=$SX/docs/109_logs/r4
echo "=== $LABEL start $(date +%F_%T) pin=$PIN manifest=$MANIFEST env: $*"
(cd "$PIN" && md5sum libWireCellClus.so libWireCellRoot.so)
for s in nuecc48 ncpi0 mcp1k mcp2k; do
    ids=()
    case $MANIFEST in
        m267)
            [ "$s" = mcp2k ] && continue
            [ "$s" = mcp1k ] && mapfile -t ids < "$M267" ;;
        eligible)
            mapfile -t ids < "$ELIG/events_eligible_$s.txt"
            [ ${#ids[@]} -gt 0 ] || continue ;;
        m267+eligible)
            case $s in
                nuecc48|ncpi0) ids=() ;;                      # the manifest already takes every event
                mcp1k) mapfile -t ids < <(sort -nu "$M267" "$ELIG/events_eligible_mcp1k.txt") ;;
                mcp2k) mapfile -t ids < "$ELIG/events_eligible_mcp2k.txt" ;;
            esac ;;
        all) ids=() ;;
        *) echo "unknown manifest $MANIFEST"; exit 2 ;;
    esac
    echo "=== $LABEL $s $(date +%T) ids=${#ids[@]} (0 = every event)"
    setarch x86_64 -R ./run_pr_chain_batch.sh "work-$s-d102m" "work-$s-$LABEL" data "${ids[@]}" \
        > "$LOGD/$LABEL-$s.log" 2>&1
    rc=$?
    echo "=== $LABEL $s rc=$rc dirs=$(ls -d work-$s-$LABEL/pr_evt* 2>/dev/null | wc -l)" \
         "not_rc0=$(grep -L 'rc=0' work-$s-$LABEL/pr_evt*/rc.txt 2>/dev/null | wc -l)"
done
(cd "$PIN" && md5sum libWireCellClus.so libWireCellRoot.so)
echo "=== $LABEL done $(date +%F_%T)"
