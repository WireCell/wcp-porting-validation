#!/usr/bin/env python3
"""doc pr/36 sec. 2.1 -- TaggerInfo field-coverage map.

Which TaggerInfo fields does the prototype's tagger stage WRITE, which does
the toolkit's tagger stage WRITE, and which are read as BDT inputs?

The intersection that matters:
    written by prototype  AND  never written by toolkit  AND  read as a BDT
    input  =>  a live defect (the BDT sees the init default forever).

Read-only.  Re-run after any tagger edit.
"""
import re, os, sys, collections

PROTO_DIR = "/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/pid/src"
PROTO_FILES = [
    "NeutrinoID_cosmic_tagger.h",
    "NeutrinoID_numu_tagger.h",
    "NeutrinoID_nue_tagger.h",
    "NeutrinoID_nue_functions.h",
    "NeutrinoID_ssm_tagger.h",
    "NeutrinoID_singlephoton_tagger.h",
]
# BDT-scoring files are listed separately: in the prototype they both READ
# features and WRITE a few (the sub-BDT score fields).
PROTO_BDT = ["NeutrinoID_nue_bdts.h", "NeutrinoID_numu_bdts.h"]

# Defaults to the live tree.  Set TK_DIR to a snapshot dir (e.g.
# /home/xqian/tmp/claude-25225/pr36, populated with `git show HEAD:<f>`) to
# audit a specific commit while the working tree carries other edits --
# that is how doc pr/36 read the toolkit at 23bd6783.
TK_DIR = os.environ.get("TK_DIR", "/nfs/data/1/xqian/toolkit-dev/toolkit/clus/src")
TK_FILES = [
    "NeutrinoTaggerCosmic.cxx",
    "NeutrinoTaggerNuMu.cxx",
    "NeutrinoTaggerNuE.cxx",
    "NeutrinoTaggerSSM.cxx",
    "NeutrinoTaggerSinglePhoton.cxx",
]
TK_BDT_DIR = "/nfs/data/1/xqian/toolkit-dev/toolkit/root/src"
TK_BDT = ["UbooneNueBDTScorer.cxx", "UbooneNumuBDTScorer.cxx"]

HDR = "/nfs/data/1/xqian/toolkit-dev/toolkit/clus/inc/WireCellClus/NeutrinoTaggerInfo.h"

# a WRITE is  <obj>.<name> <op>=   with op in {'', +, -, *, /} but not ==, !=,
# <=, >=.  Also count  <obj>.<name>++  /  --.
WRITE = r'(?:^|[^A-Za-z0-9_.])%s(?:\.|->)([A-Za-z0-9_]+)\s*(?:(?:[-+*/]?=(?!=))|\+\+|--)'
ANY   = r'(?:^|[^A-Za-z0-9_.])%s(?:\.|->)([A-Za-z0-9_]+)'


def strip_comments(txt):
    """Remove /* */ and // comments so commented-out code does not count."""
    txt = re.sub(r'/\*.*?\*/', '', txt, flags=re.S)
    txt = re.sub(r'//[^\n]*', '', txt)
    return txt


def scan(path, obj_names, pattern):
    try:
        raw = open(path, errors="replace").read()
    except OSError as e:
        sys.stderr.write("skip %s: %s\n" % (path, e))
        return {}
    txt = strip_comments(raw)
    out = collections.Counter()
    for obj in obj_names:
        for m in re.finditer(pattern % re.escape(obj), txt):
            out[m.group(1)] += 1
    return out


def union(paths, obj_names, pattern):
    tot = collections.Counter()
    per = {}
    for p in paths:
        c = scan(p, obj_names, pattern)
        per[os.path.basename(p)] = c
        tot.update(c)
    return tot, per


# ---- struct members (the universe) ------------------------------------
members = set()
th = open(HDR).read().split("\n")
s = None
for i, l in enumerate(th):
    if re.match(r'\s*struct TaggerInfo\s*\{', l):
        s = i
        break
depth, e = 0, None
for i in range(s, len(th)):
    depth += th[i].count("{") - th[i].count("}")
    if i > s and depth == 0:
        e = i
        break
for l in th[s:e + 1]:
    if l.lstrip().startswith("//"):
        continue
    m = re.match(r'\s*(?:float|int|bool|double)\s+([A-Za-z0-9_]+)\s*[\{=;]', l)
    if m:
        members.add(m.group(1))

# ---- writes -----------------------------------------------------------
pw, pw_per = union([os.path.join(PROTO_DIR, f) for f in PROTO_FILES],
                   ["tagger_info"], WRITE)
pwb, _ = union([os.path.join(PROTO_DIR, f) for f in PROTO_BDT],
               ["tagger_info"], WRITE)
tw, tw_per = union([os.path.join(TK_DIR, f) for f in TK_FILES],
                   ["ti", "tagger_info"], WRITE)
twb, _ = union([os.path.join(TK_BDT_DIR, f) for f in TK_BDT],
               ["ti", "tagger_info", "m_ti"], WRITE)

# ---- BDT inputs (any reference inside the scorer files) ----------------
p_bdt_ref, _ = union([os.path.join(PROTO_DIR, f) for f in PROTO_BDT],
                     ["tagger_info"], ANY)
t_bdt_ref, _ = union([os.path.join(TK_BDT_DIR, f) for f in TK_BDT],
                     ["ti", "tagger_info", "m_ti", "info"], ANY)

print("TaggerInfo struct members            : %d" % len(members))
print("prototype tagger-stage writes        : %d" % len(pw))
print("  (+ prototype BDT-file writes)      : %d" % len(pwb))
print("toolkit  tagger-stage writes         : %d" % len(tw))
print("  (+ toolkit BDT-scorer writes)      : %d" % len(twb))
print("prototype BDT files reference        : %d" % len(p_bdt_ref))
print("toolkit  BDT scorers reference       : %d" % len(t_bdt_ref))
print()

for name, c in sorted(pw_per.items()):
    print("  proto %-40s %4d fields" % (name, len(c)))
print()
for name, c in sorted(tw_per.items()):
    print("  tk    %-40s %4d fields" % (name, len(c)))
print()

P = set(pw) | set(pwb)
T = set(tw) | set(twb)
BDT = set(p_bdt_ref) | set(t_bdt_ref)

gap = sorted((P - T) & members)
print("=== WRITTEN by prototype, NEVER written by toolkit (%d, struct members only) ===" % len(gap))
for n in gap:
    tag = "  <-- BDT INPUT" if n in BDT else ""
    print("  %-45s proto_writes=%d%s" % (n, pw.get(n, 0) + pwb.get(n, 0), tag))
print()

live = [n for n in gap if n in BDT]
print("=== ...of which are referenced by a BDT scorer (%d) === *** LIVE ***" % len(live))
for n in live:
    print("  %s" % n)
print()

extra = sorted((T - P) & members)
print("=== WRITTEN by toolkit, NEVER written by prototype (%d) ===" % len(extra))
for n in extra:
    print("  %-45s tk_writes=%d" % (n, tw.get(n, 0) + twb.get(n, 0)))
print()

nowrite = sorted(members - P - T)
print("=== NEVER written by either side (%d) ===" % len(nowrite))
print("  " + " ".join(nowrite))
