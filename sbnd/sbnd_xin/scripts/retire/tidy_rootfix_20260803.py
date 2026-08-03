#!/usr/bin/env python3
"""Tidy round 2026-08-03: repoint the self-locating root expression in every
script that moved into a subdirectory.

Both idioms resolved to sbnd_xin itself while the file sat at the top level:

    .sh   SBND_DIR=$(cd "$(dirname "$0")" && pwd)      # and the bare `cd` form
    .py   os.path.dirname(os.path.abspath(__file__))

After the move they resolve to the file's new subdirectory instead, silently
repointing $SBND_DIR/work, $SBND_DIR/input_files, HERE/"work-..." and the
HERE/"../../abtest" reaches.  This walks each one up by exactly the depth of its
new directory, so the expression means what it always meant.

Note the two abtest reaches (d52_ab_report, p54_ab_report, d60_ab_report) need
NO extra change: they are written relative to HERE, and HERE is restored to
sbnd_xin by this fix.

    ./tidy_rootfix_20260803.py            # dry run, prints a unified diff
    CONFIRM=yes ./tidy_rootfix_20260803.py
"""
import os, re, sys, csv, difflib

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
os.chdir(ROOT)
CONFIRM = os.environ.get("CONFIRM", "no") == "yes"

MAP = "scripts/retire/tidy_map_20260803.tsv"
moved = []
with open(MAP) as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        if row["action"] == "MOVE" and re.search(r'\.(sh|py)$', row["name"]):
            moved.append(row["newpath"])

# Which shell variables name the sbnd_xin ROOT and therefore need walking up.
#
# NOT a blanket rewrite of every `VAR=$(cd "$(dirname "$0")" && pwd)`.  Counter-
# example that a blanket fix would break: geom_ab_batch.sh sets SC and uses it
# only as "$SC/run_pr_geom_arm.sh" -- and run_pr_geom_arm.sh moved into the SAME
# directory, so SC must keep meaning "my own dir".  Add a variable here only
# after checking what it is dereferenced against.
ROOT_VARS = "SBND_DIR|SX"
SH_ASSIGN = re.compile(r'((?:' + ROOT_VARS + r')=\$\(cd\s+"\$\(dirname\s+"\$0"\)")'
                       r'(\s*&&\s*pwd(?:\s+-P)?\))')
SH_CD     = re.compile(r'^(\s*cd\s+"\$\(dirname\s+"\$0"\)")(\s*(?:\|\||$))', re.M)
PY_ROOT   = re.compile(r'os\.path\.dirname\(os\.path\.abspath\(__file__\)\)')

nfix = nfile = 0
for path in sorted(moved):
    if not os.path.isfile(path):
        continue
    depth = len(os.path.dirname(path).split(os.sep))      # scripts/runners -> 2
    ups_sh = "/".join([".."] * depth)
    ups_py = ", ".join(f'"{"."*2}"' for _ in range(depth))  # '"..", ".."'
    old = open(path, encoding="utf-8").read()
    new = old

    # IDEMPOTENCE GUARD.  The python substitution is NOT self-limiting: its own
    # output still contains the literal it matches on, so a second run nests the
    # expression inside itself and the root climbs two levels too far.  A first
    # pass of this script did exactly that to 22 files; they had to be restored
    # from the git index.  Bail out if the file already carries the fix.
    if "os.path.dirname(os.path.abspath(__file__)), \"..\"" in old \
       or re.search(r'dirname\s+"\$0"\)"/\.\.', old):
        continue

    if path.endswith(".sh"):
        new = SH_ASSIGN.sub(lambda m: f'{m.group(1)}/{ups_sh}{m.group(2)}', new)
        new = SH_CD.sub(lambda m: f'{m.group(1)}/{ups_sh}{m.group(2)}', new)
    else:
        new = PY_ROOT.sub(
            "os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "
            + ups_py + "))", new)

    if new == old:
        continue
    nfile += 1
    diff = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                     f"a/{path}", f"b/{path}"))
    nfix += sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    sys.stdout.writelines(diff)
    if CONFIRM:
        open(path, "w", encoding="utf-8").write(new)

print(f"\n{nfile} file(s), {nfix} line(s) rewritten   CONFIRM={CONFIRM}")
if not CONFIRM:
    print("dry run only -- re-run with CONFIRM=yes")
