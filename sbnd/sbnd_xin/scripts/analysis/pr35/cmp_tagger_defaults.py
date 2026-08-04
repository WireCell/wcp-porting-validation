#!/usr/bin/env python3
"""Compare prototype init_tagger_info() assignments against the toolkit's
TaggerInfo default-member-initializers.  Read-only; prints value mismatches
and name-set asymmetries.

doc pr/35 section 2.1.  Result at toolkit 23bd6783 / prototype_base pid 53ca938:
1023 prototype assignments, 1024 toolkit members, ZERO value mismatches; the
name asymmetry is two renames ({shw_sp_,}br3_7_shower_main_length ->
..._main_length, numu_cc_3_acc_track_length -> numu_cc_3_track_length) plus the
toolkit-only match_isFC.

Re-run after any edit to TaggerInfo or to the prototype's init_tagger_info.
Edit PROTO/TK below if the trees move."""
import re, sys

PROTO = "/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/pid/src/NeutrinoID.cxx"
TK    = "/nfs/data/1/xqian/toolkit-dev/toolkit/clus/inc/WireCellClus/NeutrinoTaggerInfo.h"

def norm(v):
    v = v.strip()
    if v in ("true",):  return "1"
    if v in ("false",): return "0"
    # numeric normalisation
    try:
        f = float(v)
        if f == int(f): return str(int(f))
        return repr(f)
    except ValueError:
        return v

# --- prototype ---------------------------------------------------------
proto = {}
order = []
lines = open(PROTO).read().split("\n")
# find init_tagger_info body
start = None
for i, l in enumerate(lines):
    if l.startswith("void WCPPID::NeutrinoID::init_tagger_info()"):
        start = i
        break
assert start is not None
depth = 0
end = None
for i in range(start, len(lines)):
    depth += lines[i].count("{") - lines[i].count("}")
    if i > start and depth == 0:
        end = i
        break
body = lines[start:end + 1]
sys.stderr.write("proto init_tagger_info lines %d..%d (%d)\n" % (start + 1, end + 1, len(body)))
pat = re.compile(r'^\s*tagger_info\.([A-Za-z0-9_]+)\s*=\s*([^;]+);')
for l in body:
    if l.lstrip().startswith("//"):
        continue
    m = pat.match(l)
    if m:
        name, val = m.group(1), norm(m.group(2))
        if name in proto and proto[name] != val:
            sys.stderr.write("  proto REASSIGN %s: %s -> %s\n" % (name, proto[name], val))
        if name not in proto:
            order.append(name)
        proto[name] = val

# --- toolkit -----------------------------------------------------------
tk = {}
th = open(TK).read().split("\n")
s = None
for i, l in enumerate(th):
    if re.match(r'\s*struct TaggerInfo\s*\{', l):
        s = i; break
assert s is not None
depth = 0
e = None
for i in range(s, len(th)):
    depth += th[i].count("{") - th[i].count("}")
    if i > s and depth == 0:
        e = i; break
tbody = th[s:e + 1]
sys.stderr.write("toolkit TaggerInfo lines %d..%d (%d)\n" % (s + 1, e + 1, len(tbody)))
p1 = re.compile(r'^\s*(?:float|int|bool|double)\s+([A-Za-z0-9_]+)\s*\{([^}]*)\}\s*;')
p2 = re.compile(r'^\s*(?:float|int|bool|double)\s+([A-Za-z0-9_]+)\s*=\s*([^;]+);')
p3 = re.compile(r'^\s*(?:float|int|bool|double)\s+([A-Za-z0-9_]+)\s*;')
for l in tbody:
    if l.lstrip().startswith("//"):
        continue
    m = p1.match(l) or p2.match(l)
    if m:
        tk[m.group(1)] = norm(m.group(2)) if m.group(2).strip() else "0"
        continue
    m = p3.match(l)
    if m:
        tk[m.group(1)] = "<uninit>"

print("proto assigned : %d" % len(proto))
print("toolkit members: %d" % len(tk))
print()
mism = [n for n in order if n in tk and tk[n] != proto[n]]
print("=== VALUE MISMATCH (%d) ===" % len(mism))
for n in mism:
    print("  %-45s proto=%-8s toolkit=%s" % (n, proto[n], tk[n]))
print()
only_p = [n for n in order if n not in tk]
print("=== in prototype init, ABSENT from toolkit struct (%d) ===" % len(only_p))
for n in only_p:
    print("  %-45s proto=%s" % (n, proto[n]))
print()
only_t = [n for n in tk if n not in proto]
print("=== in toolkit struct, NOT initialised by prototype init (%d) ===" % len(only_t))
for n in sorted(only_t):
    print("  %-45s toolkit=%s" % (n, tk[n]))
