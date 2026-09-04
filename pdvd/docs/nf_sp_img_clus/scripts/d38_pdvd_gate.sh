#!/bin/bash
# doc pdvd/38: knob-OFF byte identity over the PDVD PR manifest.
# Compares the mabc-pr.zip member rollup and the calib-pr JSON (minus the
# vertex_scoreboard.dual_chain timer field, which is wall-clock) of two arms.
# Usage: ./d38_pdvd_gate.sh <ref-tag> <new-tag>
set -u
REF=${1:?ref tag}; NEW=${2:?new tag}
cd "$(dirname "$0")/../../.." || exit 9      # pdvd/
H=../abtest/hash_archive.py
same=0; diff=0; missing=0
for d in work/*_"$REF"; do
    b=${d%_$REF}
    n="${b}_${NEW}"
    [ -d "$n" ] || { echo "MISSING arm dir $n"; missing=$((missing+1)); continue; }
    for pat in mabc-pr.zip; do
        a=$(ls "$d"/$pat 2>/dev/null); c=$(ls "$n"/$pat 2>/dev/null)
        [ -n "$a" ] && [ -n "$c" ] || { echo "MISSING $pat $(basename "$b")"; missing=$((missing+1)); continue; }
        ha=$(python3 $H "$a" | awk '{print $1}'); hc=$(python3 $H "$c" | awk '{print $1}')
        if [ "$ha" = "$hc" ]; then same=$((same+1)); else diff=$((diff+1)); echo "DIFF zip $(basename "$b") $ha $hc"; fi
    done
    a=$(ls "$d"/calib-pr-evt*.json 2>/dev/null | head -1); c=$(ls "$n"/calib-pr-evt*.json 2>/dev/null | head -1)
    if [ -n "$a" ] && [ -n "$c" ]; then
        ha=$(python3 - "$a" <<'PY'
import hashlib,json,sys
d=json.load(open(sys.argv[1]))
# the key can be present-but-null, so `or {}` -- a bare .get(k, {}) returns None
(d.get("vertex_scoreboard") or {}).pop("dual_chain", None)
if isinstance(d.get("candidates"),list):
    for c in d["candidates"]: (c.get("vertex_scoreboard") or {}).pop("dual_chain", None)
print(hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest())
PY
)
        hc=$(python3 - "$c" <<'PY'
import hashlib,json,sys
d=json.load(open(sys.argv[1]))
# the key can be present-but-null, so `or {}` -- a bare .get(k, {}) returns None
(d.get("vertex_scoreboard") or {}).pop("dual_chain", None)
if isinstance(d.get("candidates"),list):
    for c in d["candidates"]: (c.get("vertex_scoreboard") or {}).pop("dual_chain", None)
print(hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest())
PY
)
        if [ "$ha" = "$hc" ]; then same=$((same+1)); else diff=$((diff+1)); echo "DIFF calib $(basename "$b")"; fi
    fi
done
echo "PDVD gate $NEW vs $REF: same=$same diff=$diff missing=$missing"
[ "$diff" -eq 0 ] && [ "$missing" -eq 0 ] && echo "PASS" || echo "FAIL"
