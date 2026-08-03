#!/usr/bin/env python3
"""Tidy round 2026-08-03: who references whom, and is it an INVOCATION or a
passing MENTION.

Scans every text file under sbnd_xin (excluding work*/, archive/, bee*/, and the
tidy round's own map files) for each top-level script name that is about to move,
and classifies each hit:

  INVOCATION  the reference is executable and WILL break on the move:
                python3 foo.py / ./foo.sh / bash foo.sh / $SBND_DIR/foo.py /
                import foo / from foo import ...
  MENTION     prose or a comment naming the file; stale after the move but
                nothing breaks at runtime.

Then buckets by who is referencing:
  A) STAY  -> MOVE   the runner interface calls something that is moving.
  B) MOVE  -> MOVE   across destinations.
  D) other subdirs   valfast/, nusel_display/, ql_scan/, ...  ** valfast is the
                     tool for the next campaign; an INVOCATION here is critical **
  C) docs/           the doc-citation rewrite set.

Two earlier bugs in this file, both of which hid real edges -- do not
reintroduce them:
  1. the name pattern used a `(?<![\\w/.-])` lookbehind, which excluded a leading
     `/` and therefore missed every `$SBND_DIR/foo.py` call site -- the single
     most important class.
  2. the round's own tidy_map TSV lists every moved name, so it matched
     everything and buried the real edges in noise.

Read-only.  Run before AND after the move; after, INVOCATION count must be 0.
"""
import os, re, csv, sys, collections

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
os.chdir(ROOT)

MAP = "scripts/retire/tidy_map_20260803.tsv"
action, newpath = {}, {}
with open(MAP) as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        action[row["name"]] = row["action"]
        newpath[row["name"]] = row["newpath"]

SKIP_DIRS = re.compile(r'^(work|bee-|bee$|archive$|__pycache__$|input_files|pics$|'
                       r'sbnd_geometry$|scan-|showcase-|Woodpecker|Magnify-|products$)')
SKIP_FILES = re.compile(r'^tidy_(map|refscan|move)_\d+\.(py|tsv|sh)$')
TEXT = re.compile(r'\.(sh|py|md|jsonnet|txt|tsv|json|C|pl|cfg|yaml|yml)$')

files = []
for cur, sub, names in os.walk('.'):
    sub[:] = [s for s in sub
              if not SKIP_DIRS.match(s) and not os.path.islink(os.path.join(cur, s))]
    for n in names:
        p = os.path.join(cur, n)
        if os.path.islink(p) or not TEXT.search(n) or SKIP_FILES.match(n):
            continue
        files.append(p[2:] if p.startswith('./') else p)

moved = [n for n, a in action.items()
         if a == "MOVE" and re.search(r'\.(sh|py|pl|C)$', n)]

def invocation_res(name):
    """Regexes that mean 'this line executes or imports the file'."""
    esc = re.escape(name)
    stem = re.escape(os.path.splitext(name)[0])
    return [
        re.compile(r'python3?\s+[^\s;|&]*' + esc),      # python3 [path/]foo.py
        re.compile(r'(bash|sh|source|\.)\s+[^\s;|&]*' + esc),
        re.compile(r'\./' + esc),                        # ./foo.sh
        re.compile(r'\$\{?\w+\}?/' + esc),               # $SBND_DIR/foo.py
        re.compile(r'^\s*(import|from)\s+' + stem + r'\b', re.M),
        re.compile(r'\bperl\s+[^\s;|&]*' + esc),
    ]

# a bare-name occurrence at all (used to flag MENTIONs)
bare = {n: re.compile(r'(?<![\w.-])' + re.escape(n)) for n in moved}
inv = {n: invocation_res(n) for n in moved}

hits = []          # (reffile, name, kind)
for f in files:
    base = os.path.basename(f)
    try:
        txt = open(f, encoding='utf-8', errors='replace').read()
    except OSError:
        continue
    for n in moved:
        if base == n:
            continue
        if not bare[n].search(txt):
            continue
        kind = 'INVOCATION' if any(p.search(txt) for p in inv[n]) else 'MENTION'
        hits.append((f, n, kind))

def bucket(ref):
    d = os.path.dirname(ref)
    if d == '':
        return 'A_STAY' if action.get(os.path.basename(ref)) == 'STAY' else 'B_MOVE'
    if d.startswith('docs'):
        return 'C_DOCS'
    return 'D_OTHER'

report = collections.defaultdict(list)
for f, n, kind in hits:
    b = bucket(f)
    if b == 'B_MOVE':
        if os.path.dirname(newpath[os.path.basename(f)]) == os.path.dirname(newpath[n]):
            continue    # co-located, nothing to rewrite
    report[b].append((f, n, kind))

TITLES = {
    'A_STAY':  'A) STAY -> MOVE   (the runner interface)',
    'B_MOVE':  'B) MOVE -> MOVE   (across destinations)',
    'D_OTHER': 'D) subdirs -> MOVE   (valfast/ etc -- INVOCATIONs here are critical)',
    'C_DOCS':  'C) docs/ -> MOVE   (the doc-citation rewrite set)',
}

ninv = 0
for key in ('A_STAY', 'B_MOVE', 'D_OTHER', 'C_DOCS'):
    rows = sorted(set(report[key]))
    inv_rows = [r for r in rows if r[2] == 'INVOCATION']
    men_rows = [r for r in rows if r[2] == 'MENTION']
    ninv += len(inv_rows)
    print(f"\n=== {TITLES[key]} : {len(inv_rows)} INVOCATION, {len(men_rows)} MENTION ===")
    for f, n, _ in inv_rows:
        print(f"  !! {f:42s} -> {newpath[n]}")
    if key == 'C_DOCS':
        byname = collections.Counter(n for _, n, _ in men_rows)
        print(f"     {len(byname)} distinct scripts mentioned in "
              f"{len({f for f, _, _ in men_rows})} doc files (prose citations)")
    else:
        for f, n, _ in men_rows:
            print(f"     {f:42s} mentions {n}")

print(f"\nTOTAL INVOCATIONS to rewrite: {ninv}")
print(f"scanned {len(files)} text files for {len(moved)} moved names")
sys.exit(0)
