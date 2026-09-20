#!/bin/bash
# doc sbnd_xin/115: stage-A completeness, straight off the PRODUCTS.
#
# Never off the runner's verdict: run_chain_group.sh writes its per-group log as
# "$OUTROOT/.g$K.log" by the INTERNAL group index and greps that log for its own success line,
# so concurrent invocations can read each other's verdict (doc 102, scripts/d102m_stageA_complete.sh).
# d115 runs 154/225 invocations per sample, which makes that hazard much worse than the 2-file
# case that script was written for -- and unlike it, this gate also checks that the events which
# arrived are the events the reco1 files actually hold, by joining on (run, subrun, event) from
# the census rather than trusting a count.
#
# Usage: stageA_complete.sh <cv|nuecc|off>
# Exit 0 only on "STAGE-A COMPLETE".
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
S=${1:?usage: stageA_complete.sh <cv|nuecc|off>}
case "$S" in
    cv)    OUT=$SX/work-r3cv-d115 ;;
    nuecc) OUT=$SX/work-r3nue-d115 ;;
    off)   OUT=$SX/work-r3off-d115 ;;
    *) echo "unknown sample: $S" >&2; exit 2 ;;
esac
CEN=$SX/products/d115/$S/file_rse.tsv
[ -s "$CEN" ] || { echo "ERROR: no census $CEN" >&2; exit 1; }

# beam-off is a single out_root (its 1000 event numbers are unique); the MC samples are one
# out_root per reco1 file.
python3 - "$S" "$OUT" "$CEN" <<'PY'
import os, sys, collections
S, OUT, CEN = sys.argv[1:4]
rows = [l.rstrip('\n').split('\t') for l in open(CEN)][1:]
want = collections.defaultdict(list)          # fileidx -> [(run, subrun, event)]
for fi, _f, _e, r, sr, ev in rows:
    want[fi].append((r, sr, ev))
nwant = sum(len(v) for v in want.values())

miss, empty, shortgrp, extra = [], [], [], []
seen_total = 0
for fi in sorted(want):
    root = OUT if S == 'off' else os.path.join(OUT, 'f' + fi)
    gev = os.path.join(root, 'g0', 'events.txt')
    if S == 'off':
        # one root, many groups: the union of every group's events.txt
        got = set()
        for g in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            p = os.path.join(root, g, 'events.txt')
            if g.startswith('g') and os.path.isfile(p):
                got |= {l.strip() for l in open(p) if l.strip()}
    else:
        if not (os.path.isfile(gev) and os.path.getsize(gev)):
            shortgrp.append(fi); got = set()
        else:
            got = {l.strip() for l in open(gev) if l.strip()}
    for (r, sr, ev) in want[fi]:
        seen_total += 1
        pct = os.path.join(root, 'ql_evt' + ev, 'pctree-evt%s.tar.gz' % ev)
        if ev not in got:
            miss.append((fi, r, sr, ev, 'not in events.txt'))
        elif not os.path.exists(pct):
            miss.append((fi, r, sr, ev, 'no pctree'))
        elif os.path.getsize(pct) == 0:
            empty.append((fi, r, sr, ev))
    if S != 'off':
        surplus = got - {e for (_r, _s, e) in want[fi]}
        if surplus:
            extra.append((fi, sorted(surplus)))
    if S == 'off':
        break                                   # one root covers the whole sample

nql = 0
for root in ([OUT] if S == 'off' else [os.path.join(OUT, d) for d in os.listdir(OUT)]) if os.path.isdir(OUT) else []:
    if os.path.isdir(root):
        nql += sum(1 for d in os.listdir(root) if d.startswith('ql_evt'))

print('%s: expected=%d ql_evt_dirs=%d short_groups=%d missing=%d empty=%d files_with_surplus=%d'
      % (S, nwant, nql, len(shortgrp), len(miss), len(empty), len(extra)))
for fi, r, sr, ev, why in miss[:40]:
    print('  MISSING f%s rse=(%s,%s,%s): %s' % (fi, r, sr, ev, why))
for fi, r, sr, ev in empty[:40]:
    print('  EMPTY   f%s rse=(%s,%s,%s)' % (fi, r, sr, ev))
for fi in shortgrp[:40]:
    print('  SHORT   f%s: no or empty g0/events.txt' % fi)
for fi, sur in extra[:10]:
    print('  SURPLUS f%s: events not in the census: %s' % (fi, sur))
ok = (not miss and not empty and not shortgrp and not extra and nql == nwant)
print('%s: STAGE-A %s' % (S, 'COMPLETE' if ok else 'INCOMPLETE'))
sys.exit(0 if ok else 1)
PY
