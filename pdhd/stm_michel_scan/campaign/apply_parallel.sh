#!/usr/bin/env bash
# doc pdhd/18 -- put the resolved verdicts onto the REAL viewer widgets, N processes.
#
# Ported from the doc pdvd/55 tranche-2 scratch applyall.sh / applyresume.sh.
# Lessons carried over:
#   * every process writes a PRIVATE labeldir (ROUND/lbl_<batch>_<i>/): the
#     viewer's save_labels() rewrites the whole dict, so two processes on one
#     labels.json clobber each other;
#   * <= 3 concurrent (5 were OOM-killed at 439/509 on PDVD);
#   * python3 -u (the first PDVD run's logs sat block-buffered for ~90 min);
#   * 25 s stagger (App takes free_port(5300); simultaneous starts bind-race).
# New here: BATCH, so the apply runs between scan waves instead of after all of
# them (mkspec.py writes spec_<BATCH>_<i>.json; merge.py unions every lbl_*).
# A process that dies can be resumed: RESUME=1 re-applies only the spec items not
# yet on disk in its private labeldir.
#
# Usage: apply_parallel.sh ROUND DET SHEET PREPDIR BATCH [N]
set -u
R=$1; DET=$2; SHEET=$3; PREP=$4; B=$5; N=${6:-3}
H=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$R/logs"
for i in $(seq 0 $((N-1))); do
  SPEC="$R/spec/spec_${B}_$i.json"
  [ -f "$SPEC" ] || { echo "no $SPEC"; continue; }
  L="$R/lbl_${B}_$i"; mkdir -p "$L"
  [ -f "$L/labels.json" ] || echo '{"labels": {}}' > "$L/labels.json"
  if [ "${RESUME:-0}" = 1 ]; then
    python3 - "$SPEC" "$L/labels.json" "$R/spec/resume_${B}_$i.json" <<'EOF'
import json, sys
spec, lab, out = sys.argv[1:4]
have = json.load(open(lab))["labels"]
todo = [it for it in json.load(open(spec)) if it["key"] not in have]
json.dump(todo, open(out, "w"), indent=1, ensure_ascii=False)
print("resume %s: %d of %d still to apply" % (spec, len(todo), len(json.load(open(spec)))))
EOF
    SPEC="$R/spec/resume_${B}_$i.json"
  fi
  nohup python3 -u "$H/scan_harness.py" apply --det "$DET" --tag apply18 \
      --labeldir "$L" --prepdir "$PREP" --manifest "$SHEET" --verdicts "$SPEC" \
      > "$R/logs/apply_${B}_$i.log" 2>&1 &
  echo "apply ${B}_$i pid $! spec $SPEC"
  sleep 25
done
wait
touch "$R/APPLY_DONE_$B"
echo "ALL APPLY DONE ($B)"
