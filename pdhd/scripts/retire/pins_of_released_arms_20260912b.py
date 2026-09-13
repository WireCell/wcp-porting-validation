#!/usr/bin/env python3
"""Round B (doc 105 sec 13): do the TOP-LEVEL pins of the arms round B released
still back anything?  Value-first, the same test as pins_of_dead_rounds_20260912.py.

  usage:  python3 pins_of_released_arms_20260912b.py [--list]

A PIN GOES WITH ITS ARMS.  Round B released work-*-d145np (sbnd) and the old pdvd
flip evidence (d41prod, d42fit, d43*, d44*, d45prod, d48nu3).  Their binaries are
top-level ~/tmp pins, which the nested-pin tool never looks at.  For each
candidate: every doc/script/~/tmp runner that names the pin is read for arm
tokens; the pin is KEPT if any of those arms is still on disk in ANY tree
(sbnd work-* dirs included, sample stripped).  Substrate names are dropped from
the survivors unless they carry the pin's own round (round A's measured defect:
runners name their src as well as their outputs).

Measured before writing this, and why each candidate is here:
  d145_libpin, _cont, _cont2   pr/145-148 binaries; d145np released, but d145prod
                               (pr/145's SHIPPED arm) survives -- the test decides.
  d41_libpin d42_libpin d44_libpin d45_libpin   docs pdvd/41-45; d41prov is kept
                               substrate -- the test decides.
NOT candidates: d146_libpin* back work-*-d146sv25, which survives; d47_libpin/new11
backs d48nu7 (kept substrate) and is in PROTECTED.txt.
"""
import os, re, sys, glob, subprocess

T  = "/home/xqian/tmp"
R  = "/home/xqian/toolkit-dev/wcp-porting-img"
TK = "/home/xqian/toolkit-dev/toolkit"
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = ["d145_libpin", "d145_libpin_cont", "d145_libpin_cont2",
              "d41_libpin", "d42_libpin", "d44_libpin", "d45_libpin"]
ROOTS = [f"{R}/pdvd/docs", f"{R}/pdhd/docs", f"{R}/sbnd/sbnd_xin/docs",
         f"{R}/pdvd/scripts", f"{R}/pdhd/scripts", f"{R}/sbnd/sbnd_xin/scripts", f"{TK}/clus/docs"]
ARM  = re.compile(r'\b(?:p|d|h)\d{1,3}[a-z][a-z0-9]{0,14}\b')
EVT  = re.compile(r'^\d{6}_\d+_(.+)$')
SAMP = re.compile(r"-(mcp1k|mcp2k|ncpi0|nuecc48|dbg25a)(?=-|$)|^(mcp1k|mcp2k|ncpi0|nuecc48|dbg25a)-")

def alive_arms():
    out = set()
    for tree in ("pdvd", "pdhd"):
        w = f"{R}/{tree}/work"
        for d in os.listdir(w):
            m = EVT.match(d)
            if m: out.add(m.group(1))
    for d in os.listdir(f"{R}/sbnd/sbnd_xin"):
        if d.startswith("work-") and os.path.isdir(f"{R}/sbnd/sbnd_xin/{d}"):
            arm = SAMP.sub("", d[5:]).strip("-")
            out.add(arm)
            out |= set(ARM.findall(arm))
    return out

def substrate():
    src = open(os.path.join(HERE, "plan_20260912b.py")).read()
    s = set()
    for m in re.finditer(r"substrate=\[(.*?)\]", src, re.S):
        s |= set(re.findall(r'"([^"]+)"', m.group(1)))
    return s - {"(bare)"}

def naming_files(pin):
    files = []
    r = subprocess.run(["grep", "-rlI", "--exclude-dir=retire", "--exclude-dir=archive",
                        f"tmp/{pin}\\b"] + ROOTS, capture_output=True, text=True)
    files += r.stdout.split()
    for f in glob.glob(f"{T}/*/*.sh") + glob.glob(f"{T}/*/*.md") + glob.glob(f"{T}/*.sh"):
        if "/libpin" in f or "/claude-" in f: continue
        try:
            if f"tmp/{pin}" in open(f, errors="ignore").read(): files.append(f)
        except OSError: pass
    return [f for f in files if "cleanup" not in os.path.basename(f)]

def main():
    alive, sub = alive_arms(), substrate()
    free, held = [], []
    for pin in CANDIDATES:
        p = f"{T}/{pin}"
        if not os.path.isdir(p): continue
        files = naming_files(pin)
        rnd = re.match(r"^((?:d|p|h)\d+)", pin).group(1)
        surv = set()
        for f in files:
            try: t = open(f, errors="ignore").read()
            except OSError: continue
            surv |= {a for a in ARM.findall(t) if a in alive}
        surv -= {a for a in sub if not re.match(rf"^{rnd}(?![0-9])", a)}
        (held if surv else free).append((p, sorted(surv)[:6], len(files)))
    if "--list" in sys.argv:
        for p, _s, _n in free: print(p)
        return 0
    for p, s, n in held: print(f"KEEP  {os.path.basename(p):<20} named by {n} file(s); survivors: {' '.join(s)}")
    for p, s, n in free: print(f"FREE  {os.path.basename(p):<20} named by {n} file(s)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
