#!/bin/bash
# Post-run audit of a 1-step PR-chain log.  Exits non-zero if any SILENT
# failure mode fired -- the ones that leave rc=0 and a complete-looking output
# computed the wrong way.  Run it per event (or over a merged log) in any bulk
# harness; a run that is not audited is not validated.
#
#   ./check-pr-run.sh <lar.log> [...]
#
# Every check here corresponds to a bug this chain has actually shipped:
#   * the sp.jsonnet extVar that silently selected the wrong ROI thresholds
#   * the DL vertex that silently falls back to geometric with no torch
#   * RSE that silently comes out 0/0/N
# The pattern is always the same: plausible output, wrong provenance.
rc=0
for log in "$@"; do
    [ -r "$log" ] || { echo "CANNOT READ $log"; rc=1; continue; }
    bad=""

    # 1. DL (SCN) vertex fell back to geometric -- setup-dlvtx.sh not sourced,
    #    or the scn UPS product is unavailable.  WARN-level only in the job.
    n=$(grep -c 'DL vertex failed' "$log")
    [ "$n" -gt 0 ] && bad="$bad\n    DL vertex fell back to geometric ($n event(s)) -- source setup-dlvtx.sh"

    # 2. An RSE attacher was configured but never saw an art::Event, i.e. it is
    #    missing from the fcl 'inputers'.  Everything downstream then reports
    #    run/subrun 0.
    n=$(grep -c 'visit(art::Event&) never called' "$log")
    [ "$n" -gt 0 ] && bad="$bad\n    RSE attacher not registered as an inputer ($n)"

    # 3. MABC asked for metadata RSE and did not get it.
    n=$(grep -c 'rse_from_metadata set but input metadata has no' "$log")
    [ "$n" -gt 0 ] && bad="$bad\n    rse_from_metadata fell back ($n) -- attacher missing upstream?"

    # 4. The labeler could not find its point-cloud tree and passed the event
    #    through unlabeled.
    n=$(grep -c 'passing through unlabeled' "$log")
    [ "$n" -gt 0 ] && bad="$bad\n    labeler passed an event through UNLABELED ($n)"

    # 5. A split lost blobs.  The splitters assert this themselves; it is fatal
    #    to the clustering even though the job continues.
    n=$(grep -c 'BLOB LOSS' "$log")
    [ "$n" -gt 0 ] && bad="$bad\n    BLOB LOSS in separate() ($n)"

    # 6. The PR chain ran but produced no tracking-pr.root writer output.  Only
    #    meaningful when tracking_visitor is in pipeline_names.
    if grep -q 'SbndPrMagnifyTrackingVisitor' "$log" \
       && ! [ -f "$(dirname "$log")/tracking-pr.root" ]; then
        bad="$bad\n    tracking_visitor ran but tracking-pr.root is missing"
    fi

    if [ -n "$bad" ]; then
        echo "FAIL $log"; printf "$bad\n"; rc=1
    else
        echo "OK   $log"
    fi
done
exit $rc
