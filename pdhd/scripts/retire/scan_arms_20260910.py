#!/usr/bin/env python3
"""Which ARMS do the hand scans resolve into?  Measured, never typed.

  usage:  python3 scan_arms_20260910.py [--json out.json]

WHY THIS EXISTS.  The owner asked on 2026-09-10 to keep the STM hand-scan
results for PDVD and PDHD.  Keeping the LABELS is not keeping the scan: a
label says "039252_7/98 is STM_MICHEL", and the only thing that turns that
back into physics is the dump the scanner was actually looking at.  A fresh
arm at a different operating point is never a substitute (the 08-31 prod0825
doctrine, restated in both PROTECTED.txt files).

WHY THE EXISTING MACHINERY WOULD HAVE MISSED IT.  The scan prep sets carry
their source arm in their own provenance key (`"arm": "d68a3"`), but
  * the label JSON never names an arm, so INTERLOCK 9's manifest test
    ("which arms does a live manifest resolve into") finds nothing; and
  * cit_20260908.py's ROOTS list has pdhd/stm_scan and pdhd/ql_scan but NOT
    pdhd/stm_michel_scan, so the citation census never read these files
    either.  d53v, d67v, d68d4, d68a3 and d53h scored zero on both tests
    while being the entire evidentiary basis of smx1/smx1a/smx3/smx4.
Both gaps are closed: this file is the positive derivation, cit_20260910.py
adds the missing roots, and INTERLOCK 13 refuses any round whose release
intersects what this prints.

OVER-EMITTING IS THE SAFE DIRECTION (the toks_* doctrine): an extra arm can
only pull a dir OUT of the release.  So every arm-shaped token found in any
prep set, viewer or scan harness is emitted, then intersected with the arms
that actually exist in each work tree.
"""
import os, re, sys, json, glob, collections

R = "/home/xqian/toolkit-dev/wcp-porting-img"
EVT = re.compile(r"^(\d{6})_(\d+)(?:_(.+))?$")
ARMTOK = re.compile(r"^[A-Za-z][A-Za-z0-9]{1,23}$")
# words that are arm-shaped but are not arms
JUNK = {"true","false","null","none","default","label","event","cluster","scan",
        "tag","det","pdvd","pdhd","sbnd","muon","michel","gamma","unassigned",
        "wire","fit","end","source","arm","json","data","work","docs","x","y","z"}

# Where a scan records what it was built from.  Prep sets carry a per-item
# provenance key; viewers and harnesses carry an ARM_DIR / arm= binding.
# Every scan/display tooling dir of both detectors, discovered rather than
# enumerated: on 2026-09-10 a hand-written list of three dirs missed
# pdvd/d08_scan (d08pv30on/off) and pdhd/d05_scan (d05mON) entirely.
PREP_GLOBS = [f"{R}/{d}/{s}/prep*/*.json" for d in ("pdhd", "pdvd")
              for s in ("stm_michel_scan", "stm_scan", "d08_scan", "d05_scan",
                        "ql_scan", "wf_scan")]
BIND_GLOBS = [f"{R}/{d}/{s}/*.{e}" for d in ("pdhd", "pdvd")
              for s in ("stm_michel_scan", "stm_scan", "d08_scan", "d05_scan",
                        "ql_scan", "wf_scan", "ql_display", "scripts")
              for e in ("py", "sh")]
BIND_RE = re.compile(r"""(?:ARM_DIR|ARM|arm|src_arm|src_tag)\s*[=:]\s*["']([A-Za-z0-9_.-]+)["']""")

def arms_on_disk(det):
    work = f"{R}/{det}/work"
    out = collections.Counter()
    if not os.path.isdir(work): return out
    for d in os.listdir(work):
        m = EVT.match(d)
        if m: out[m.group(3) or "(bare)"] += 1
    return out

def harvest():
    """token -> list of files that named it"""
    found = collections.defaultdict(set)
    for pat in PREP_GLOBS:
        for f in glob.glob(pat):
            try: j = json.load(open(f))
            except Exception: continue
            if not isinstance(j, dict): continue
            for k in ("arm", "src_arm", "src_tag"):
                v = j.get(k)
                if isinstance(v, str) and ARMTOK.match(v) and v.lower() not in JUNK:
                    found[v].add(os.path.dirname(f))
    for pat in BIND_GLOBS:
        for f in glob.glob(pat):
            try: t = open(f, errors="ignore").read()
            except OSError: continue
            for m in BIND_RE.finditer(t):
                v = m.group(1)
                if ARMTOK.match(v) and v.lower() not in JUNK:
                    found[v].add(f)
    return found

def label_tags():
    """every scan tag that actually holds labels -- what we are protecting FOR"""
    out = []
    for det in ("pdvd", "pdhd"):
        for d in glob.glob(f"{R}/{det}/work/*_labels/*"):
            if not os.path.isdir(d): continue
            n = len([x for x in glob.glob(d + "/*.json")])
            if n: out.append((det, os.path.relpath(d, f"{R}/{det}/work"), n))
    return out

if __name__ == "__main__":
    found = harvest()
    print("# scan tags carrying labels (the thing being protected FOR)")
    for det, tag, n in sorted(label_tags()):
        print(f"TAG   {det:5s} {tag:40s} {n} label file(s)")
    print("# arm tokens named by a prep set / viewer / harness")
    for tok in sorted(found):
        print(f"TOKEN {tok:24s} <- {', '.join(sorted(os.path.basename(x) for x in found[tok]))}")
    print("# resolved: token x arms that exist on disk")
    res = {}
    for det in ("pdvd", "pdhd"):
        disk = arms_on_disk(det)
        hit  = sorted(t for t in found if disk.get(t))
        res[det] = hit
        for t in hit:
            print(f"KEEP  {det:5s} {t:24s} {disk[t]:4d} dirs")
        miss = sorted(t for t in found if not disk.get(t))
        if miss: print(f"#     {det}: named but not on this tree: {' '.join(miss)}")
    for a in sys.argv[1:]:
        if a.startswith("--json="):
            json.dump(res, open(a.split("=", 1)[1], "w"), indent=1)
            print(f"# wrote {a.split('=',1)[1]}")
