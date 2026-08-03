#!/usr/bin/env python3
"""Tidy round 2026-08-03: repoint every script citation in docs/ (and the
subdirectory READMEs) at the script's new path.

Rewrites only forms that are unambiguously a PATH to the script, so prose that
merely names the file is left alone:

    python3 foo.py          -> python3 scripts/<sub>/foo.py
    ./foo.sh                -> ./scripts/<sub>/foo.sh
    bash foo.sh             -> bash scripts/<sub>/foo.sh
    `foo.py`                -> `scripts/<sub>/foo.py`      (backticked = a path)
    sbnd_xin/foo.py         -> sbnd_xin/scripts/<sub>/foo.py

Deliberately NOT rewritten: a bare `foo.py` in running prose with no backticks
and no command context.  Those read fine as a name and rewriting them would
churn the scientific record for no gain.

Idempotent: a citation that already contains the destination directory is
skipped, so the script can be re-run safely (the rootfix script in this same
round was NOT idempotent and had to be undone from the git index -- do not
repeat that mistake).

    ./tidy_docfix_20260803.py            # dry run, per-file counts
    CONFIRM=yes ./tidy_docfix_20260803.py
"""
import os, re, csv, sys, collections

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
os.chdir(ROOT)
CONFIRM = os.environ.get("CONFIRM", "no") == "yes"

MAP = "scripts/retire/tidy_map_20260803.tsv"
newpath = {}
with open(MAP) as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        if row["action"] == "MOVE" and re.search(r'\.(sh|py|pl|C)$', row["name"]):
            newpath[row["name"]] = row["newpath"]

TARGETS = []
for d in ("docs", "nusel_display", "ql_scan", "valfast", "stm_campaign"):
    for cur, sub, names in os.walk(d):
        for n in names:
            if n.endswith((".md", ".txt", ".sh", ".py")):
                TARGETS.append(os.path.join(cur, n))

def rules(name, dest):
    """(compiled pattern, replacement) pairs.  Each must not match text that
    already carries the destination prefix -- that is the idempotence guard."""
    e, d = re.escape(name), os.path.dirname(dest)
    guard = r'(?<!' + re.escape(d[-12:]) + r'/)' if len(d) >= 12 else ''
    return [
        (re.compile(r'(?<![\w/.-])(python3?\s+)' + e), r'\g<1>' + dest),
        (re.compile(r'(?<![\w/.-])(bash\s+)' + e),     r'\g<1>' + dest),
        (re.compile(r'(?<![\w/.-])\./' + e),           './' + dest),
        # backticked, either bare (`foo.py`) or with arguments
        # (`foo.py --detail 285185:21`).  The second form is why the closing
        # backtick cannot be part of the match -- requiring it missed 11 of the
        # 17 citations the gate caught on the first pass.
        (re.compile(r'`' + e + r'(?=[`\s])'),          '`' + dest),
        # `sbnd_xin/foo.py`, including when it is the tail of an ABSOLUTE path,
        # e.g. root -l -q '/nfs/.../toolkit/sbnd_xin/dump_stopping_dqdx.C(...)'
        # -- so no lookbehind here.
        (re.compile(r'sbnd_xin/' + e),                 'sbnd_xin/' + dest),
    ]

per = collections.Counter()
tot = 0
for path in sorted(set(TARGETS)):
    try:
        old = open(path, encoding="utf-8").read()
    except OSError:
        continue
    new = old
    for name, dest in newpath.items():
        if name not in new:
            continue
        for pat, rep in rules(name, dest):
            # Idempotence, done by checking the text BEFORE the match rather
            # than by re-substituting inside the matched span.  The earlier
            # nested-sub version silently did nothing for any rule with a
            # lookahead: m.group(0) excludes the lookahead character, so the
            # inner re-match never fired and 13 citations went unrewritten.
            def once(m, r=rep, dd=os.path.dirname(dest)):
                if new[max(0, m.start() - len(dd) - 1):m.start()].endswith(dd + "/"):
                    return m.group(0)            # already under its destination
                return m.expand(r) if "\\g" in r or "\\1" in r else r
            new = pat.sub(once, new)
    if new != old:
        n = sum(1 for a, b in zip(old.splitlines(), new.splitlines()) if a != b)
        per[path] = n
        tot += n
        if CONFIRM:
            open(path, "w", encoding="utf-8").write(new)

for p, n in sorted(per.items()):
    print(f"  {n:4d}  {p}")
print(f"\n{len(per)} file(s), {tot} line(s) changed   CONFIRM={CONFIRM}")
if not CONFIRM:
    print("dry run only -- re-run with CONFIRM=yes")
