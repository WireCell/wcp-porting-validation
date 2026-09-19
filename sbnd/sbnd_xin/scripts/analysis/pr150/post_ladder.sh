#!/bin/bash
# doc sbnd_xin/pr/150 sec 5 -- post-process the retune ladder: per level, the metrics / traj extracts on all four
# samples and the sentinel registry; the s0 sentinel printout restricted to the sentinel events present in the
# ladder arms (s0 has 0 FAIL on the full samples, so its FAIL on any subset is 0 by construction -- the file is
# still written so the reader sees the PASS/SKIP split on the same events); then pr150_ladder.py.
#   post_ladder.sh r1 r2 r3 r4
set -u
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin; cd $SX || exit 2
D=docs/pr/150_figs; P=scripts/analysis/pr150; L=/home/xqian/tmp/pr150/logs; M=/home/xqian/tmp/pr150/manifest_ladder
for lev in "$@"; do
  for s in nuecc48 ncpi0 mcp1k mcp2k; do
    [ -s $D/metrics/pr150$lev-$s.tsv ] || { python3 $P/pr150_metrics.py extract --arm pr150$lev --samples $s > $L/metrics_pr150${lev}_$s.log 2>&1; echo "metrics $lev $s rc=$?"; }
    [ -s $D/traj/pr150$lev-$s.tsv ] || { python3 $P/pr150_traj_eval.py extract --arm pr150$lev --samples $s --jobs 6 > $L/traj_pr150${lev}_$s.log 2>&1; echo "traj $lev $s rc=$?"; }
  done
  ARMS=""; for s in nuecc48 ncpi0 mcp1k mcp2k; do ARMS="$ARMS work-$s-pr150$lev"; done
  python3 scripts/analysis/pr149/sentinels_tolerant.py --arms $ARMS > $D/150_lad_sentinels_pr150$lev.txt 2>&1
  echo "sentinels $lev rc=$? PASS $(grep -c '^PASS' $D/150_lad_sentinels_pr150$lev.txt) FAIL $(grep -c '^FAIL' $D/150_lad_sentinels_pr150$lev.txt) SKIP $(grep -c '^SKIP' $D/150_lad_sentinels_pr150$lev.txt)"
done
# s0 on the same sentinel events: take the full-sample s0 printouts and keep the sentinels that are not SKIP in the
# first level's arms (present in the manifest); everything else becomes SKIP.
first=$1
python3 - "$D" "$first" <<'PY'
import re, sys
D, first = sys.argv[1:3]
present = {l.split()[1] for l in open(f'{D}/150_lad_sentinels_pr150{first}.txt') if re.match(r'^(PASS|FAIL)\s', l)}
out = []
for f in (f'{D}/150_s3_sentinels_s0.txt', f'{D}/150_s1_sentinels_s0.txt'):
    for l in open(f):
        m = re.match(r'^(PASS|FAIL|SKIP)\s+(\S+)', l)
        if not m:
            continue
        ev = m.group(2)
        if any(ev == o.split()[1] for o in out):
            continue
        out.append(l if ev in present else 'SKIP ' + l.split(None, 1)[1])
open(f'{D}/150_lad_sentinels_pr150s0.txt', 'w').write(f'# s0 sentinels restricted to the {len(present)} sentinel events present in the ladder arms (from the full-sample runs)\n' + ''.join(out))
print('s0 ladder sentinels: PASS', sum(l.startswith('PASS') for l in out), 'FAIL', sum(l.startswith('FAIL') for l in out), 'SKIP', sum(l.startswith('SKIP') for l in out))
PY
python3 $P/pr150_ladder.py --manifest $M --levels "$@" > $D/150_ladder_score.txt 2>&1; echo "ladder score rc=$?"; cat $D/150_ladder_score.txt
