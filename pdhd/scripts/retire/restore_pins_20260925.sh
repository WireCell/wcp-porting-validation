#!/bin/bash
# Restore cold lib pins archived by archive_pins_20260925.py (doc sbnd_xin/125 sec 9).
#
#   ./restore_pins_20260925.sh                 # every archived pin, back at its original path
#   ./restore_pins_20260925.sh p82/libpin_p82  # just the named pin(s) (paths relative to ~/tmp)
#   RESTORE_TO=/some/dir ./restore_pins_20260925.sh p82/libpin_p82   # a test restore elsewhere
#
# The archive stores hardlinks between pins, so a single pin cannot be extracted on its own reliably:
# the whole archive is extracted into a staging dir (~35 GiB on disk), the wanted pins are moved into
# place, and the staging copy of the rest is removed.  Every restored file is checked against the
# frozen sha256 manifest.
set -u
T=/home/xqian/tmp
DST=${RESTORE_TO:-$T}
STAMP=20260925
ARCH=$T/pin-archive-$STAMP/cold-pins-$STAMP.tar.zst
MAN=/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/archive/records/cleanup-$STAMP/pins-cold/cold-pins.manifest.tsv
[ -s "$ARCH" ] || { echo "no archive $ARCH"; exit 2; }
if [ $# -eq 0 ]; then set -- $(cat "$T/pin-archive-$STAMP/cold-pins-$STAMP.list"); fi
for p in "$@"; do [ -e "$DST/$p" ] && { echo "REFUSING: $DST/$p exists"; exit 3; }; done
S=$(mktemp -d "$T/pin-restore-XXXX")
zstd -q -d --long=31 -c "$ARCH" | tar -C "$S" -xpf - || { echo "extract failed; staging $S"; exit 4; }
python3 - "$S" "$MAN" "$@" <<'PY' || exit 5
import sys, os, hashlib
S, MAN, pins = sys.argv[1], sys.argv[2], sys.argv[3:]
bad = n = 0
for line in open(MAN):
    if line.startswith("#"): continue
    rel, k, x, d, _ = line.rstrip("\n").split("\t")
    if not any(rel == p or rel.startswith(p + "/") for p in pins): continue
    q = os.path.join(S, rel); n += 1
    if k == "l": bad += (os.readlink(q) != x)
    else:
        h = hashlib.sha256(open(q, "rb").read()).hexdigest()
        bad += (h != d)
print(f"verified {n} rows, {bad} differ"); sys.exit(1 if bad else 0)
PY
for p in "$@"; do
  mkdir -p "$(dirname "$DST/$p")"; mv "$S/$p" "$DST/$p" && echo "restored $DST/$p"
  [ "$DST" = "$T" ] && rm -f "$T/$p.ARCHIVED"
done
python3 - "$S" <<'PY'
import os, sys
S = sys.argv[1]
for r, ds, fs in os.walk(S, topdown=False):
    for f in fs: os.unlink(os.path.join(r, f))
    for d in ds:
        q = os.path.join(r, d); os.unlink(q) if os.path.islink(q) else os.rmdir(q)
os.rmdir(S)
PY
echo done
