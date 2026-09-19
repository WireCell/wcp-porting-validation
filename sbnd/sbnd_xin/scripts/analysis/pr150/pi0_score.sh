#!/bin/bash
# doc sbnd_xin/pr/150 -- run the pi0 hand-scan census + the gamma ledger on one arm.
#
# FORK BY DUPLICATION (CLAUDE.md M10) of the `pi0` step of scripts/pr144_analyze.sh
# (doc pr/144's record; untouched).  What this fork changes:
#   * arm naming work-<sample>-pr150<cell>, and --tag for the reference arm;
#   * manifests come from scripts/analysis/pr150/pi0_manifests.py (absolute dump
#     paths, absent rows MARKED not dropped) instead of pr144_pi0_manifests.sh;
#   * it also runs pr132_gamma_ledger.py, which pr144_analyze.sh does not;
#   * everything lands in /home/xqian/tmp/pr150/scorers/pi0_<cell>.txt -- nothing
#     is written under docs/pr/ or em_display/ (M13).
#
# The census/ledger SCRIPTS are the committed ones and are NOT forked: both take
# --manifest98/--manifest141 overrides, so re-pointing an arm needs no edit.
#
# Label sets (doc pr/135 Repro): base em_labels/{emscan-0827,emscan-0828-agent5},
# overlay em_labels/pi0scan-0829-agent -> 66 hand pi0 / 132 gammas when every
# event is reachable.
#
# Env:
#   CENSUS=pr132   (default) scripts/pr132_pi0_census.py, base-wins + one overlay
#          pr141c2         scripts/pr141_pi0_census2.py, the newer per-field label
#                          precedence CHAIN "pi0mass-0904-owner,pi0scan-0829-agent"
#                          (doc pr/141 sec 13.1 -- this is what doc pr/144 ran, and
#                          it gives a DIFFERENT, owner-corrected denominator)
#   FUDGE / OFFSET  override the arm's own kine_shower_fudge_factor / pi0_mass_offset
#                   (by default both are read from the arm's compiled per-event
#                   config .wct-cfg-evt<ID>.json, the way pr144_analyze.sh does --
#                   NEVER hardcoded: an arm's scale is a property of the arm)
#
# Usage:  ./scripts/analysis/pr150/pi0_score.sh <cell>        # work-<s>-pr150<cell>
#         TAG=d102mpr ./scripts/analysis/pr150/pi0_score.sh   # the reference arm
set -u
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
OUT=/home/xqian/tmp/pr150/scorers
mkdir -p "$OUT" || exit 2
cd "$SX" || exit 2

CELL=${1:-}
TAG=${TAG:-}
if [ -z "$TAG" ]; then
  [ -n "$CELL" ] || { echo "usage: pi0_score.sh <cell>   |   TAG=<armtag> pi0_score.sh" >&2; exit 2; }
  TAG="pr150$CELL"
  MARG="--cell $CELL"
else
  MARG="--tag $TAG"
fi
REPORT=$OUT/pi0_${CELL:-$TAG}.txt
MAN=$OUT/manifests/$TAG
CENSUS=${CENSUS:-pr132}

{
echo "=== pr150 pi0 census + gamma ledger :: arm tag $TAG  ($(date +%F_%H:%M:%S))"
echo

echo "--- 1. manifests (denominator re-pointed at this arm) ---"
python3 scripts/analysis/pr150/pi0_manifests.py $MARG --out "$OUT/manifests" || exit 2
echo

# The arm's own energy scale, from the compiled config of the first event that
# has one.  A census run at the wrong --fudge silently mis-scales every hand
# mass in section B (pr132_pi0_census.py docstring).
read -r F O < <(python3 - "$TAG" <<'PY'
import glob, json, os, sys
SX = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
tag = sys.argv[1]
fud = off = None
for s in ("nuecc48", "ncpi0", "mcp1k", "mcp2k"):
    for f in sorted(glob.glob(os.path.join(SX, "work-%s-%s" % (s, tag),
                                           "pr_evt*", ".wct-cfg-evt*.json")))[:1]:
        got = []
        def walk(o):
            if isinstance(o, dict):
                if o.get("type") == "TaggerCheckNeutrino":
                    got.append(o.get("data", {}))
                for v in o.values(): walk(v)
            elif isinstance(o, list):
                for v in o: walk(v)
        walk(json.load(open(f)))
        if got:
            fud = got[0].get("kine_shower_fudge_factor")
            off = got[0].get("pi0_mass_offset")
    if fud is not None: break
# absent key => the C++ default (TaggerCheckNeutrino.h: fudge 0.80, offset 10)
print("%s %s" % (fud if fud is not None else 0.80, off if off is not None else 10))
PY
)
F=${FUDGE:-$F}; O=${OFFSET:-$O}
echo "--- 2. arm energy scale (read from .wct-cfg-evt*.json, NOT hardcoded) ---"
echo "    kine_shower_fudge_factor = $F     pi0_mass_offset = $O MeV"
echo

echo "--- 3. census ($CENSUS) ---"
case $CENSUS in
  pr132)
    python3 scripts/pr132_pi0_census.py \
        --manifest98 "$MAN/denom98.tsv" --manifest141 "$MAN/denom141.tsv" \
        --fudge "$F" --offset "$O" --overlay-tag pi0scan-0829-agent \
        --tsv "$OUT/pi0_${CELL:-$TAG}_census.tsv" ;;
  pr141c2)
    python3 scripts/pr141_pi0_census2.py \
        --manifest98 "$MAN/denom98.tsv" --manifest141 "$MAN/denom141.tsv" \
        --fudge "$F" --offset "$O" --chain "pi0mass-0904-owner,pi0scan-0829-agent" \
        --tsv "$OUT/pi0_${CELL:-$TAG}_census.tsv" ;;
  *) echo "unknown CENSUS=$CENSUS (pr132 | pr141c2)"; exit 2 ;;
esac
echo "    census rc=$?"
echo

echo "--- 4. gamma ledger (pr132_gamma_ledger.py) ---"
echo "    NOTE: the ledger has no --fudge.  Its OK band (0.80 <= kine_charge/e_lab"
echo "    <= 1.25) is fixed, and the labels carry SCAN-TIME energies at fudge 0.80,"
echo "    so on an arm at $F every ratio is scaled by $F/0.80.  Compare ledgers"
echo "    BETWEEN arms at the same fudge; do not compare to doc pr/135's 90.9 %."
python3 scripts/pr132_gamma_ledger.py \
    --manifest98 "$MAN/denom98.tsv" --manifest141 "$MAN/denom141.tsv" \
    --overlay-tag pi0scan-0829-agent \
    --tsv "$OUT/pi0_${CELL:-$TAG}_ledger.tsv"
echo "    ledger rc=$?"

echo
echo "--- 5. is this census READABLE on this arm? (pi0_id_drift.py vs d102mpr) ---"
echo "    The census resolves a hand gamma through showers[].id, a per-event"
echo "    reconstruction index.  A cell that changes clustering renumbers it and"
echo "    EVERY gamma reads 'absent-on-arm' -- a numbering artefact reported in the"
echo "    same column as a real missing shower (doc pr/141 sec 3).  Read this before"
echo "    reading section A."
if [ "$TAG" != d102mpr ]; then
  python3 scripts/analysis/pr150/pi0_id_drift.py --a "$TAG" --b d102mpr
  echo "    rc=$?"
else
  echo "    (this IS the reference arm; nothing to compare against)"
fi
echo
echo "=== done $(date +%F_%H:%M:%S)"
} 2>&1 | tee "$REPORT"
echo "wrote $REPORT"
