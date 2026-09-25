#!/bin/bash
# Undo compress_logs_20260925.py (doc sbnd_xin/125 sec 9): restore the original logs/configs.
#
#   ./uncompress_logs_20260925.sh                        # every file the pass compressed
#   ./uncompress_logs_20260925.sh 'work-*-d123lgop'      # only arms matching the glob(s)
#
# Each restored file is checked against the sha256 frozen in the manifest before its .zst goes.
set -u
R=/home/xqian/toolkit-dev/wcp-porting-img
MAN=$R/sbnd/sbnd_xin/archive/records/cleanup-20260925/logs-zst/logs-zst.manifest.tsv
[ -s "$MAN" ] || { echo "no manifest $MAN"; exit 2; }
python3 - "$R" "$MAN" "$@" <<'PY'
import sys, os, fnmatch, hashlib, subprocess
R, MAN, globs = sys.argv[1], sys.argv[2], sys.argv[3:]
n = bad = 0
for line in open(MAN):
    if line.startswith("#"): continue
    rel, size, digest, status = line.rstrip("\n").split("\t")
    if status != "ok": continue
    arm = rel.split("/")[2]
    if globs and not any(fnmatch.fnmatch(arm, g) for g in globs): continue
    p = os.path.join(R, rel); z = p + ".zst"
    if os.path.exists(p) or not os.path.exists(z): continue
    raw = subprocess.run(["zstd", "-q", "-d", "-c", z], capture_output=True).stdout
    if hashlib.sha256(raw).hexdigest() != digest: print("MISMATCH, kept .zst:", z); bad += 1; continue
    open(p, "wb").write(raw); st = os.stat(z); os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.unlink(z); n += 1
print(f"restored {n}, mismatches {bad}"); sys.exit(1 if bad else 0)
PY
