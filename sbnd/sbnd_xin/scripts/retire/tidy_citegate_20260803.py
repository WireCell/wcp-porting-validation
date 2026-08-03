#!/usr/bin/env python3
"""Tidy round 2026-08-03 GATE: every path-shaped citation OF A SCRIPT THIS ROUND
MOVED must resolve on disk.

This is the check that makes the doc rewrite trustworthy.  It extracts citations
that are unambiguously a path -- `python3 X`, `./X`, `bash X`, `` `X` ``,
`sbnd_xin/X` -- and resolves each against sbnd_xin.

SCOPED TO THE MOVE SET on purpose.  An unscoped version of this gate reported
142 "failures", essentially all of them pre-existing and none of them caused by
this round: files that live outside sbnd_xin (`abtest/hash_archive.py`,
`qlport/scripts/ab_check.sh`), files inside a work-* arm (`run_pr3_evt.sh`),
tools that never existed here (`convert.C`, `magnify.sh`), and the placeholder
names in this round's own docstrings.  Reporting those as a gate failure would
have buried the one signal that matters -- did *I* break a citation.

Prose that merely names a file ("see gapjump_probe.py for the method") is not a
path claim and is not checked; tidy_docfix_20260803.py leaves those alone.

Exit 0 only when nothing in the move set is unresolved.
"""
import os, re, sys, csv, collections

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
os.chdir(ROOT)

MOVED = set()
with open("scripts/retire/tidy_map_20260803.tsv") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        if row["action"] == "MOVE" and re.search(r'\.(sh|py|pl|C)$', row["name"]):
            MOVED.add(row["name"])

TARGETS = []
for d in ("docs", "nusel_display", "ql_scan", "valfast", "stm_campaign", "scripts"):
    for cur, sub, names in os.walk(d):
        # this round's own tooling quotes placeholder names in its docstrings
        if cur.startswith("scripts/retire"):
            continue
        for n in names:
            if n.endswith((".md", ".txt", ".sh", ".py")):
                TARGETS.append(os.path.join(cur, n))

# a path-shaped citation of a script
CITE = re.compile(
    r'(?:python3?\s+|bash\s+|\./|`|sbnd_xin/)'
    r'((?:[\w.\-]+/)*[\w.\-]+\.(?:py|sh|pl|C))(?:`|\b)')

# things that are not ours to resolve
EXTERNAL = re.compile(r'^(/|~|\$|https?:|\.\./|wire-cell|wcsonnet|setup|lar)')

unresolved = collections.defaultdict(list)
nchecked = 0
for path in sorted(set(TARGETS)):
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    for m in CITE.finditer(txt):
        cite = m.group(1)
        if EXTERNAL.match(cite) or os.path.basename(cite) not in MOVED:
            continue
        nchecked += 1
        # Docs are written from two vantage points: from sbnd_xin itself, and
        # from the repo root (`sbnd_xin/foo.py`).  Accept both, plus relative to
        # the citing file's own directory (subdir READMEs cite local scripts).
        cands = [cite,
                 re.sub(r'^sbnd_xin/', '', cite),
                 os.path.join(os.path.dirname(path), cite)]
        if any(os.path.exists(c) for c in cands):
            continue
        unresolved[cite].append(path)

print(f"checked {nchecked} path-shaped citations of the {len(MOVED)} moved "
      f"scripts across {len(set(TARGETS))} files")
if not unresolved:
    print("\nGATE: PASS -- every cited script path resolves")
    sys.exit(0)

print(f"\nGATE: FAIL -- {len(unresolved)} distinct unresolved citation(s):")
for cite, where in sorted(unresolved.items(), key=lambda kv: -len(kv[1])):
    print(f"  {cite:52s} in {len(where)} file(s): {', '.join(sorted(set(where))[:3])}")
sys.exit(1)
