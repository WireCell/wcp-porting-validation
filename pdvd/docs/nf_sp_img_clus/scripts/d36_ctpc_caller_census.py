#!/usr/bin/env python3
"""doc pdvd/36 sec 11.1 -- who actually calls the ctpc query the metric changed?

Two unrelated functions in clus/ share the name `get_closest_points`:

  Grouping::get_closest_points(point, radius, apa, face, pind)  (>= 3 commas)
  Grouping::has_closest_point(...)                              (name is unique)
      the 2-D ctpc lattice query -- what doc 36 changed
  Cluster::get_closest_points(const Cluster&) and the Facade_Util.h template
      a 3-D cluster-to-cluster nearest-pair helper -- untouched

A name-only `grep -l` cannot separate them and also counts commented-out code
and prose, which is how two sessions produced counts of 19, 6 and 5 for a set
whose real size is 4.  This strips /* */ and // comments and string literals
first, then classifies each call by argument count.

Usage:  cd <toolkit>;  python3 <this> [clus]
"""
import glob, os, re, sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else 'clus'
NAME = re.compile(r'\b(get_closest_points|has_closest_point)\s*\(')

def strip(src):
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    src = re.sub(r'//[^\n]*', '', src)
    return re.sub(r'"(\\.|[^"\\])*"', '""', src)

rows = {}
for f in sorted(glob.glob(os.path.join(ROOT, 'src', '*.cxx'))
                + glob.glob(os.path.join(ROOT, 'inc', 'WireCellClus', '*.h'))):
    s = strip(open(f, errors='replace').read())
    for m in NAME.finditer(s):
        depth, args = 1, ''
        for ch in s[m.end():m.end()+400]:
            if ch == '(': depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0: break
            args += ch
        ncomma = args.count(',') - args.count('{')      # brace-init lists carry commas
        kind = 'ctpc' if (m.group(1) == 'has_closest_point' or ncomma >= 2) else 'cluster-pair'
        rows[(os.path.basename(f), kind)] = rows.get((os.path.basename(f), kind), 0) + 1

for kind in ('ctpc', 'cluster-pair'):
    sel = {f: n for (f, k), n in rows.items() if k == kind}
    print('\n%s form -- %d files, %d live calls' % (kind, len(sel), sum(sel.values())))
    for f, n in sorted(sel.items(), key=lambda x: -x[1]):
        print('   %-40s %d' % (f, n))
ext = {f: n for (f, k), n in rows.items()
       if k == 'ctpc' and not f.startswith('Facade_Grouping')}
print('\nEXTERNAL callers of the changed query: %d files, %d calls' % (len(ext), sum(ext.values())))
