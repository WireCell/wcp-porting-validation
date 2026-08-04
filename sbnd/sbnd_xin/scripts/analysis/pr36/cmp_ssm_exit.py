#!/usr/bin/env python3
"""doc pr/36 sec. 2.2 -- exit_ssm parity check.

The toolkit folds the prototype's 504-line WCPPID::NeutrinoID::exit_ssm_tagger()
into a 175-line lambda (NeutrinoTaggerSSM.cxx:712-887, self-described at :738 as
"a long repetitive block mirroring exit_ssm_tagger lines 2786-3081").  A 3x size
difference is worth checking rather than trusting.

Answer: the lambda assigns 146 ssm_ fields against the prototype's 356, and
every one of the 212 it omits already carries -999 as its
default-member-initializer in TaggerInfo -- which init_tagger_info (ti =
TaggerInfo{}) sets.  0 value mismatches.  The omission is exact, PROVIDED the
exit is unreachable after those fields are written; the toolkit's two exit
sites (:890, :911) both precede the first ParticleBlock write (:1355).

Read-only.  Re-run after any NeutrinoTaggerSSM.cxx or TaggerInfo edit.
"""
import re

PROTO = ("/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/pid/src/"
         "NeutrinoID_ssm_tagger.h")
TK    = "/nfs/data/1/xqian/toolkit-dev/toolkit/clus/src/NeutrinoTaggerSSM.cxx"
HDR   = ("/nfs/data/1/xqian/toolkit-dev/toolkit/clus/inc/WireCellClus/"
         "NeutrinoTaggerInfo.h")

# exit_ssm_tagger() body, 1-indexed lines 2706..3210
PROTO_LO, PROTO_HI = 2706, 3210
# the exit_ssm lambda, 1-indexed lines 712..887
TK_LO, TK_HI = 712, 887


def strip(txt):
    return re.sub(r'//[^\n]*', '', txt)


def norm(v):
    v = v.strip()
    if v == "true":
        return "1"
    if v == "false":
        return "0"
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return v


proto_lines = open(PROTO, errors="replace").read().split("\n")
pbody = strip("\n".join(proto_lines[PROTO_LO - 1:PROTO_HI]))
pv = dict(re.findall(r'tagger_info\.(ssm_[A-Za-z0-9_]+)\s*=\s*([^;]+);', pbody))

tk_lines = open(TK, errors="replace").read().split("\n")
tbody = strip("\n".join(tk_lines[TK_LO - 1:TK_HI]))
tf = set(re.findall(r'ti\.(ssm_[A-Za-z0-9_]+)\s*=', tbody))

hdr = open(HDR, errors="replace").read()
defaults = dict(re.findall(
    r'\b(?:float|int|bool|double)\s+(ssm_[A-Za-z0-9_]+)\s*\{([^}]*)\}\s*;', hdr))

print("prototype exit_ssm_tagger assigns : %d ssm_ fields" % len(pv))
print("toolkit  exit_ssm lambda  assigns : %d ssm_ fields" % len(tf))

missing = [n for n in pv if n not in tf]
print("missing from toolkit lambda       : %d" % len(missing))

bad = [(n, norm(pv[n]), norm(defaults.get(n, "<no default>")))
       for n in missing if norm(pv[n]) != norm(defaults.get(n, "<no default>"))]
print("of those, prototype exit value != toolkit struct DEFAULT : %d" % len(bad))
for n, p, d in sorted(bad):
    print("   %-45s proto_exit=%-10s tk_default=%s" % (n, p, d))

print("distinct proto exit values on the missing set : %s"
      % sorted(set(norm(pv[n]) for n in missing)))
print("distinct toolkit defaults on the missing set  : %s"
      % sorted(set(norm(defaults.get(n, "<no default>")) for n in missing)))

extra = sorted(tf - set(pv))
print("\nin toolkit lambda, NOT in prototype exit (%d): %s" % (len(extra), extra))
