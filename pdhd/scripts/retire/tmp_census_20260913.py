#!/usr/bin/env python3
"""Cleanup ROUND C 2026-09-13 (doc 106): the ~/tmp census.  Retires nothing.

  usage:  python3 tmp_census_20260913.py            # report + tmp_tier_20260913.txt
          python3 tmp_census_20260913.py --list     # print the tier only

Run AFTER plan_20260913.py: "alive" is what survives the work release, read from
its tier1_*_20260913.txt files, not what is on disk now.

FOUR UNIT CLASSES, each with its own keep test:
  pin      a dir that directly holds libWireCellClus.so.  KEPT if permanent
           (the two production pins + a PROTECTED one), or VALUE-FIRST: some text
           file names the pin (by its <top>/<leaf> path, or by leaf name from
           inside its own round dir) AND names an arm that survives round C.
           Substrate names are dropped from the survivors unless they carry the
           pin's own round number (round A's measured defect: runners name their
           SRC as well as their outputs).
           CHANGE FROM ROUNDS A/B: their sweep SKIPPED any pin with >2 MiB of
           non-.so content as "a full local/lib snapshot".  Every modern pin is
           one (doc 81: "cp -a ../local/lib ... a partial pin is not a pin"), so
           that filter kept every dead round's pin.  A snapshot is tested like any
           other pin now; the non-.so size is printed, not obeyed.
  prep     <round>/prep_<arm>, the staged PR input of one arm.  KEPT iff <arm>
           survives round C (its outputs are the product; the prep is the
           intermediate).
  session  a Claude session scratchpad.  KEPT if it is this session or a live
           peer, its transcript was written in the last 12 h, any file in it was
           written in the last 6 h, a process has its cwd inside, or a committed
           doc/script names its path.
  scratch  ~/tmp/h28 minus libpin_h28 (the session that wrote it: launcher logs,
           private git indexes, superseded readouts, ~9 MB), and the harness's
           own leftovers at the claude-25225 root older than 7 days.
Everything else in ~/tmp is out of scope for this round and listed by size.

FREED BYTES ARE SET-RELATIVE.  Round A hardlinked 5663 files across pins, so a
pin's du is not what its removal frees.  An inode counts only if EVERY link to it
lies inside the release set.
"""
import os, re, sys, glob, time, json, collections

T     = "/home/xqian/tmp"
R     = "/home/xqian/toolkit-dev/wcp-porting-img"
TK    = "/home/xqian/toolkit-dev/toolkit"
HERE  = os.path.dirname(os.path.abspath(__file__))
STAMP = "20260913"
CL    = f"{T}/claude-25225"
NOW   = time.time()

ROOTS = [f"{R}/pdvd/docs", f"{R}/pdhd/docs", f"{R}/sbnd/sbnd_xin/docs", f"{R}/pdvd/scripts",
         f"{R}/pdhd/scripts", f"{R}/sbnd/sbnd_xin/scripts", f"{R}/pdhd/stm_michel_scan",
         f"{TK}/clus/docs"]
PERMANENT = {
    f"{T}/h28/libpin_h28": "PDHD production pin (doc pdhd/28 line 12; pdhd PROTECTED 2026-09-13)",
    f"{T}/p96/libpin_p96": "PDVD production pin (doc pdvd/96 Repro; pdvd PROTECTED 2026-09-13)",
    f"{T}/pdhdstm_libpin": "pdhd PROTECTED line 37 (stm-tagger-chain line 901, backs the stm0/stmw scan)",
}
# ~/tmp/h25, h25r: the blind re-judge session is open.  (h29 was held here from
# ~04:40 until that session pushed doc pdhd/29 as f0524f75; ~/tmp/h29 is not a unit.)
LIVE_TOP = ("h25",)
THIS     = "2279b083-d99b-4b7c-8de4-c2131283a379"
LIVE_SESSIONS = {THIS: "this session",
                 "1cab00a8-432a-4ff9-bce1-91f8710d5d7e": "peer pdhd25-lever-arms-blind-rejudge (open)"}
HARNESS_KEEP = {"bash-edit-diff", "bundled-skills", "auto-mode-classifier-errors"}

ARM  = re.compile(r"\b(?:p|d|h)\d{1,3}[a-z][a-z0-9]{0,14}\b")
EVT  = re.compile(r"^\d{6}_\d+_(.+)$")
SAMP = re.compile(r"-(mcp1k|mcp2k|ncpi0|nuecc48|dbg25a)(?=-|$)|^(mcp1k|mcp2k|ncpi0|nuecc48|dbg25a)-")

def families_on_disk():
    fam = collections.defaultdict(set)
    for tree in ("pdvd", "pdhd"):
        w = f"{R}/{tree}/work"
        for d in os.listdir(w):
            m = EVT.match(d)
            if m: fam[m.group(1)].add(f"{w}/{d}")
    s = f"{R}/sbnd/sbnd_xin"
    for d in os.listdir(s):
        if d.startswith("work-") and os.path.isdir(f"{s}/{d}"):
            arm = SAMP.sub("", d[5:]).strip("-")
            for a in {arm} | set(ARM.findall(arm)): fam[a].add(f"{s}/{d}")
    return fam

def alive_after():
    released = set()
    for tree in ("sbnd", "pdvd", "pdhd"):
        tf = f"{HERE}/tier1_{tree}_{STAMP}.txt"
        if not os.path.exists(tf): sys.exit(f"missing {tf} -- run plan_{STAMP}.py first")
        released |= {l.strip() for l in open(tf) if l.strip()}
    return {a for a, dirs in families_on_disk().items() if dirs - released}, len(released)

def substrate():
    src = open(f"{HERE}/plan_{STAMP}.py").read()
    s = set()
    for m in re.finditer(r"substrate=\[(.*?)\]", src, re.S):
        s |= set(re.findall(r'"([^"]+)"', m.group(1)))
    return s - {"(bare)"}

def newest(p):
    m = 0
    for cur, subs, fs in os.walk(p):
        subs[:] = [x for x in subs if not os.path.islink(os.path.join(cur, x))]
        for f in fs:
            try: m = max(m, os.lstat(os.path.join(cur, f)).st_mtime)
            except OSError: pass
    return m

def cwd_inside(p):
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        try: c = os.readlink(f"/proc/{pid}/cwd")
        except OSError: continue
        if (c + "/").startswith(p.rstrip("/") + "/"): return pid
    return None

def text_files():
    """Every text file a pin or prep could be named from: the committed docs and
    scripts, plus runners and notes inside ~/tmp (never inside a pin, a prep, or
    a scratchpad; never a cleanup record)."""
    out = []
    for root in ROOTS:
        for cur, subs, fs in os.walk(root):
            subs[:] = [x for x in subs if x not in ("retire", "archive", ".git")]
            out += [os.path.join(cur, f) for f in fs]
    for cur, subs, fs in os.walk(T):
        depth = cur[len(T):].count("/")
        subs[:] = [x for x in subs if depth < 2 and not x.startswith(("claude-", "prep_"))
                   and "libpin" not in x and "libsnap" not in x and x != "pin_lib"]
        out += [os.path.join(cur, f) for f in fs
                if f.endswith((".sh", ".md", ".py", ".txt", ".json", ".log", ".tsv"))]
    return [f for f in out if "cleanup" not in os.path.basename(f)]

def read_text(f):
    try:
        if os.path.getsize(f) > 8 << 20: return None
        with open(f, "rb") as fh: b = fh.read()
    except OSError:
        return None
    return None if b"\0" in b[:4096] else b.decode("utf-8", "ignore")

def main():
    alive, nrel = alive_after()
    sub = substrate()
    units = []          # (kind, path, verdict, why)

    # ---- pins ----------------------------------------------------------------
    pins = []
    for cur, subs, fs in os.walk(T):
        depth = cur[len(T):].count("/")
        subs[:] = [x for x in subs if depth < 3 and not x.startswith(("claude-", "prep_"))
                   and not os.path.islink(os.path.join(cur, x))]
        if "libWireCellClus.so" in fs and cur != T:
            pins.append(cur); subs[:] = []
    # ---- preps ---------------------------------------------------------------
    preps = sorted(p for p in glob.glob(f"{T}/*/prep_*") if os.path.isdir(p) and not os.path.islink(p))

    # one pass over the text: which files name which pin
    rel = {p: os.path.relpath(p, T) for p in pins}
    pat = {p: re.compile(r"(?<![A-Za-z0-9_.-])" + re.escape(rel[p]) + r"(?![A-Za-z0-9_])") for p in pins}
    leafpat = {p: re.compile(r"(?<![A-Za-z0-9_.-])" + re.escape(os.path.basename(p)) + r"(?![A-Za-z0-9_])")
               for p in pins if "/" in rel[p]}
    namers = collections.defaultdict(list)
    for f in text_files():
        t = read_text(f)
        if not t: continue
        for p in pins:
            if rel[p] in t and pat[p].search(t):
                namers[p].append((f, t)); continue
            if p in leafpat and f.startswith(os.path.dirname(p) + "/") and leafpat[p].search(t):
                namers[p].append((f, t))
    for p in sorted(pins):
        top = rel[p].split("/")[0]
        nonso = sum(os.lstat(os.path.join(c, f)).st_size for c, _s, fs in os.walk(p) for f in fs
                    if ".so" not in f and not os.path.islink(os.path.join(c, f))) >> 20
        if p in PERMANENT:
            units.append(("pin", p, "KEEP", PERMANENT[p])); continue
        if top.startswith(LIVE_TOP):
            units.append(("pin", p, "KEEP", "live prefix h25")); continue
        rm = re.match(r"^(?:pin_)?((?:d|p|h)\d+)", top)
        rnd = rm.group(1) if rm else None
        surv = set()
        for f, t in namers[p]:
            surv |= {a for a in ARM.findall(t) if a in alive}
        surv -= {a for a in sub if not (rnd and re.match(rf"^{rnd}(?![0-9])", a))}
        n = len(namers[p])
        if surv:
            units.append(("pin", p, "KEEP", f"named by {n} file(s); survivors {' '.join(sorted(surv)[:6])}; non-.so {nonso} MiB"))
        else:
            units.append(("pin", p, "FREE", f"named by {n} file(s), no surviving arm; non-.so {nonso} MiB"))
    # A prep a scan record names was SERVED to the scanners (pdvd smx3..smx9 sheets
    # name tmp/p90/prep_p90vprod, tmp/d68/prep_d68a3, ...).  Measured 2026-09-13:
    # every such prep's arm also survives, so this rule changes nothing today; it
    # is here so a future release of the arm cannot take the served prep with it.
    scanrec = set()
    for root in (f"{R}/pdvd/docs/scan", f"{R}/pdhd/docs/scan", f"{R}/pdhd/stm_michel_scan",
                 f"{R}/pdvd/work/stm_michel_labels", f"{R}/pdhd/work/stm_michel_labels"):
        for cur, _s, fs in os.walk(root):
            for f in fs:
                t = read_text(os.path.join(cur, f))
                if t: scanrec |= set(re.findall(r"tmp/([A-Za-z0-9_]+/prep_[A-Za-z0-9_]+)", t))
    for p in preps:
        top, arm = p[len(T) + 1:].split("/")[0], os.path.basename(p)[5:]
        if top.startswith(LIVE_TOP) or arm.startswith(LIVE_TOP):
            units.append(("prep", p, "KEEP", "live prefix"))
        elif os.path.relpath(p, T) in scanrec:
            units.append(("prep", p, "KEEP", "named by a scan record (served to the scanners)"))
        elif arm in alive:
            units.append(("prep", p, "KEEP", f"arm {arm} survives round C"))
        else:
            units.append(("prep", p, "FREE", f"arm {arm} not on disk after round C"))

    # ---- session scratchpads ----------------------------------------------------
    named_docs = collections.defaultdict(list)
    for root in ROOTS:
        for cur, subs, fs in os.walk(root):
            subs[:] = [x for x in subs if x not in ("retire", "archive", ".git")]
            for f in fs:
                t = read_text(os.path.join(cur, f))
                if t and "claude-25225" in t:
                    for m in re.finditer(r"claude-25225/([^/\s`'\")]+)(?:/([0-9a-f-]{36}))?", t):
                        named_docs[m.group(2) or m.group(1)].append(os.path.join(cur, f))
    transcripts = {}
    for j in glob.glob("/home/xqian/.claude/projects/*/*.jsonl"):
        transcripts[os.path.basename(j)[:-6]] = max(transcripts.get(os.path.basename(j)[:-6], 0), os.path.getmtime(j))
    for proj in sorted(glob.glob(f"{CL}/-*")):
        for s in sorted(glob.glob(f"{proj}/*")):
            if not os.path.isdir(s) or os.path.islink(s): continue
            u = os.path.basename(s)
            why = None
            if u in LIVE_SESSIONS: why = LIVE_SESSIONS[u]
            elif NOW - transcripts.get(u, 0) < 12 * 3600: why = "transcript written in the last 12 h"
            elif NOW - newest(s) < 6 * 3600: why = "a file written in the last 6 h"
            elif cwd_inside(s): why = f"process {cwd_inside(s)} has its cwd inside"
            elif named_docs.get(u): why = f"named by {named_docs[u][0]}"
            units.append(("session", s, "KEEP" if why else "FREE",
                          why or f"transcript {int((NOW - transcripts.get(u, 0)) / 3600)} h old, idle"))
    # ---- scratch -------------------------------------------------------------------
    for e in sorted(os.listdir(f"{T}/h28")):
        p = f"{T}/h28/{e}"
        if p in PERMANENT: continue
        units.append(("scratch", p, "FREE", "~/tmp/h28 scratch (its author, 2026-09-13)"))
    for e in sorted(os.listdir(CL)):
        p = f"{CL}/{e}"
        if e.startswith("-") or e in HARNESS_KEEP: continue
        age = NOW - (newest(p) if os.path.isdir(p) and not os.path.islink(p) else os.lstat(p).st_mtime)
        if named_docs.get(e):
            units.append(("scratch", p, "KEEP", f"named by {named_docs[e][0]}"))
        elif age > 7 * 86400:
            units.append(("scratch", p, "FREE", f"harness leftover, {int(age / 86400)} d old"))

    # ---- guards on the unit list itself --------------------------------------------
    for kind, p, v, _w in units:
        assert p.startswith(T + "/") and p != T, p
        assert not (v == "FREE" and os.path.islink(p) and os.path.isdir(p)), f"symlinked dir unit {p}"
        assert not (v == "FREE" and any(p == k or k.startswith(p + "/") for k in PERMANENT)), f"{p} contains a permanent pin"
        assert not (v == "FREE" and (p + "/").startswith(f"{CL}/-home-xqian-toolkit-dev-toolkit/{THIS}/")), p

    free = [p for _k, p, v, _w in units if v == "FREE"]
    # ---- set-relative bytes ------------------------------------------------------
    inodes, nominal = {}, 0
    for p in free:
        walker = [(os.path.dirname(p), [], [os.path.basename(p)])] if not os.path.isdir(p) or os.path.islink(p) \
                 else os.walk(p)
        for cur, subs, fs in walker:
            if subs: subs[:] = [x for x in subs if not os.path.islink(os.path.join(cur, x))]
            for f in fs:
                try: st = os.lstat(os.path.join(cur, f))
                except OSError: continue
                if not os.path.isfile(os.path.join(cur, f)) or os.path.islink(os.path.join(cur, f)): continue
                nominal += st.st_blocks * 512
                k = (st.st_dev, st.st_ino)
                c, n, b = inodes.get(k, (0, st.st_nlink, st.st_blocks * 512))
                inodes[k] = (c + 1, n, b)
    freed = sum(b for c, n, b in inodes.values() if c >= n)
    shared = sum(b for c, n, b in inodes.values() if c < n)

    if "--list" in sys.argv:
        print("\n".join(free)); return 0
    by = collections.defaultdict(lambda: collections.Counter())
    for kind, _p, v, _w in units: by[kind][v] += 1
    print(f"alive families after round C: {len(alive)} (work dirs released by the plan: {nrel})")
    for kind in ("pin", "prep", "session", "scratch"):
        print(f"\n== {kind}: KEEP {by[kind]['KEEP']}  FREE {by[kind]['FREE']}")
        for k, p, v, w in units:
            if k == kind and (v == "KEEP" or kind != "prep"):
                print(f"  {v:<4}  {os.path.relpath(p, T):<60} {w}")
        if kind == "prep":
            fr = collections.Counter(p[len(T) + 1:].split("/")[0] for k, p, v, _w in units if k == "prep" and v == "FREE")
            print("  FREE by round: " + " ".join(f"{r}:{n}" for r, n in sorted(fr.items())))
    print(f"\nrelease units {len(free)} | nominal {nominal / 2**30:.2f} GiB | "
          f"FREED (every link inside the set) {freed / 2**30:.2f} GiB | "
          f"shared with a kept file, frees nothing {shared / 2**30:.2f} GiB")
    open(f"{HERE}/tmp_tier_{STAMP}.txt", "w").write("\n".join(free) + "\n")
    json.dump([dict(kind=k, path=p, verdict=v, why=w) for k, p, v, w in units],
              open(f"{HERE}/tmp_census_{STAMP}.json", "w"), indent=1)
    print(f"tier file: {HERE}/tmp_tier_{STAMP}.txt ({len(free)} lines).  This script removed nothing.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
