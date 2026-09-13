#!/usr/bin/env python3
"""Which nested ~/tmp pins belong to a round that has NO surviving arm?

  usage:  python3 pins_of_dead_rounds_20260912.py [--list]

A PIN GOES WITH ITS ARMS.  A round writes its binary snapshot inside its own
scratch dir (~/tmp/p75/libpin_p75, ~/tmp/d64/libpin, ...).  Once every arm that
binary produced is released, the pin is regenerable from the commit its doc
records and nothing can read it back.  Until then it is load-bearing, and a
missing LD_LIBRARY_PATH entry is SILENTLY ignored (the M1 shape).

THE TEST IS VALUE-FIRST, NOT NAME-ADJACENT, and that distinction is the whole
file.  Matching a pin to its round by NAME says ~/tmp/p75/libpin_p75 belongs to
round p75, whose arms (p75hleg/p75hoff/p75vprod) are all released -- so release
it.  That is WRONG: d81_arms.sh names that pin, and it is the binary behind
p79vprod, the arm doc pdvd/81 sec 6 identifies as CURRENT PDVD PRODUCTION.  It
is the d08cap10 mistake exactly (a doc names the shipped VALUE in one sentence
and the ARM in another).  Same for ~/tmp/d66/libpin, which d68_arms.sh ties to
d68a3 and d68d4 -- the smx3/smx4 HAND-SCAN sources the owner asked to keep.

So: for each pin, find every doc/script that names it, pull the arm tokens out
of those files, and keep the pin if ANY of them is still on disk.

SELF-SEQUENCING: `alive` is recomputed from the tree at run time, so running
this before the pdhd release correctly reports nothing, and after it reports
the pins that release unlocked.  Nothing is hard-coded to a moment.
"""
import os, re, sys, subprocess

T  = "/home/xqian/tmp"
R  = "/home/xqian/toolkit-dev/wcp-porting-img"
TK = "/home/xqian/toolkit-dev/toolkit"
ROOTS = [f"{R}/pdvd/docs", f"{R}/pdhd/docs", f"{R}/sbnd/sbnd_xin/docs",
         f"{R}/pdvd/scripts", f"{R}/pdhd/scripts", f"{TK}/clus/docs"]

def tmp_text_files():
    """A round's OWN runner lives in ~/tmp beside its pin, and the doc/script
    roots above never see it.  Without this, a pin named only by
    ~/tmp/d62/run_arms.sh reads as "named by 0 files" and looks releasable on
    no evidence at all.  Text files only, depth <= 2, and never inside a pin
    directory (a pin holding a stray .txt must not vouch for itself)."""
    out, base = [], T.rstrip("/").count("/")
    for cur, subs, fs in os.walk(T):
        if cur.count("/") - base >= 2: subs[:] = []
        subs[:] = [d for d in subs
                   if not os.path.islink(os.path.join(cur, d))
                   and "libpin" not in d and "pin_lib" not in d
                   and "libsnap" not in d]
        for f in fs:
            if f.rsplit(".", 1)[-1] in ("sh", "py", "md", "txt", "log", "json"):
                out.append(os.path.join(cur, f))
    return out
TMPFILES = []
LIVE_TMP = ("h26", "h27", "h28", "p97", "mg10")   # live rounds, doc 105 sec 5 -- never touched
SUBSTRATE = set()
# {0,14}, not {1,14}: the old form needed TWO characters after the round letter
# and so could never match d53v / d53h / d67v -- the smx1 and smx3 SCAN SOURCES.
# Measured 2026-09-12: it released d53/libpin with 14 files naming it.
ARM  = re.compile(r'\b(?:p|d|h)\d{1,3}[a-z][a-z0-9]{0,14}\b')
EVT  = re.compile(r'^\d{6}_\d+_(.+)$')

def substrate_inputs():
    """The planner's substrate names -- a round's INPUTS, not what its pin made.

    2026-09-12 (doc 105), measured on the first dry run: every one of 30 nested
    pins was held by the same two "survivors", d16hnu and d16vnu.  Not a stray
    doc -- the rounds' own runners (d83_arms.sh, d96_arms.sh, h25/run_arm.sh)
    name them as the SRC the arms are staged from.  A survivor test that counts
    an input as an output can never release any pin of any round that staged
    from kept substrate, which is all of them.  So substrate tokens are dropped
    from a pin's survivors -- EXCEPT when the substrate carries the pin's own
    round number (d51/libpin made d51vclus; that pin still goes with it).
    Read from plan_<STAMP>.py's substrate=[...] lists, never retyped."""
    import glob as _g
    src = open(sorted(_g.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "plan_20260912.py")))[0]).read()
    out = set()
    for m in re.finditer(r"substrate=\[(.*?)\]", src, re.S):
        out |= set(re.findall(r'"([^"]+)"', m.group(1)))
    return out - {"(bare)"}

def alive_arms():
    out = set()
    for tree in ("pdvd", "pdhd"):
        w = f"{R}/{tree}/work"
        if not os.path.isdir(w): continue
        for d in os.listdir(w):
            m = EVT.match(d)
            if m: out.add(m.group(1))
    return out

def nested_pins():
    out = []
    for e in sorted(os.listdir(T)):
        p = os.path.join(T, e)
        if not os.path.isdir(p) or os.path.islink(p): continue
        if not re.match(r'^(d|p|h)\d+', e): continue
        if e.startswith(LIVE_TMP): continue      # a live round's pin is not a dead round's
        for x in sorted(os.listdir(p)):
            q = os.path.join(p, x)
            if os.path.isdir(q) and not os.path.islink(q) and ("libpin" in x or "pin_lib" in x):
                out.append(q)
    return out

def union_bytes(paths):
    seen, tot = set(), 0
    for root in paths:
        for cur, _s, fs in os.walk(root):
            for f in fs:
                try: st = os.lstat(os.path.join(cur, f))
                except OSError: continue
                if st.st_ino in seen: continue
                seen.add(st.st_ino); tot += st.st_size
    return tot

def grep_l(pat, targets, batch=1500):
    """grep -rl over a list too long for one argv (ARG_MAX; 8k+ files here)."""
    hits = []
    for i in range(0, len(targets), batch):
        r = subprocess.run(["grep", "-rlI", pat] + targets[i:i+batch],
                           capture_output=True, text=True)
        hits += r.stdout.split()
    return hits

def main():
    global TMPFILES
    TMPFILES = tmp_text_files()
    alive = alive_arms()
    global SUBSTRATE
    SUBSTRATE = substrate_inputs()
    safe, held = [], []
    for pin in nested_pins():
        rel = os.path.relpath(pin, T)
        files = grep_l(f"tmp/{rel}", ROOTS + TMPFILES)
        files = [f for f in files if "/retire/" not in f]
        surv = set()
        for f in files:
            try: t = open(f, errors="ignore").read()
            except OSError: continue
            surv |= {a for a in ARM.findall(t) if a in alive}
        rnd = re.match(r'^((?:d|p|h)\d+)', rel).group(1)
        surv -= {a for a in SUBSTRATE
                 if not re.match(rf'^{rnd}(?![0-9])', a)}
        (held if surv else safe).append((pin, sorted(surv)[:5], len(files)))
    if "--list" in sys.argv:
        for pin, _s, _n in safe: print(pin)
        return 0
    print(f"# {len(alive)} arm families alive; {len(safe)+len(held)} nested pins")
    print("# KEEP -- a file naming this pin also names a surviving arm")
    for pin, s, n in held:
        print(f"KEEP  {os.path.relpath(pin,T):<26} named by {n} file(s); survivors: {' '.join(s)}")
    print("# RELEASABLE -- no surviving arm in any file that names it")
    for pin, _s, n in safe:
        print(f"FREE  {os.path.relpath(pin,T):<26} named by {n} file(s)")
    print(f"# union of releasable: {union_bytes([p for p,_,_ in safe])/2**30:.2f} GiB "
          f"over {len(safe)} pin(s)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
