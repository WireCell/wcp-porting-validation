#!/usr/bin/env python3
"""Cleanup ROUND D 2026-09-16 (doc sbnd_xin/111): the ~/tmp census.  Retires nothing.

  usage:  python3 tmp_census_20260921.py            # report + tmp_tier_20260921.txt
          python3 tmp_census_20260921.py --list     # print the tier only

Run AFTER plan_20260921.py: "alive" is what survives the work release, read from
its tier1_*_20260921.txt files, not what is on disk now.

Fork of tmp_census_20260913.py (round C).  What THIS round changes:

  1. PERMANENT PINS FOLLOW PRODUCTION.  ~/tmp/d102/libpin_d102 is the pin both
     current production PR arms ran on (PDHD d108hflip, doc pdvd/108 sec 3 "clus md5
     091e142b9481 unchanged before and after"; PDVD d103vflip, doc pdvd/103 sec 10.5).
     ~/tmp/d102m-libsnap backs sbnd production d102m/d102mpr.  libpin_h28 and
     libpin_p96 lose their permanent status (their production arms are superseded)
     and are judged value-first like every other pin.
  2. HELD BY THE LIVE PEER (message 2026-09-16): ~/tmp/d111, d112, d113 (doc pdvd/113
     will write there), d103 (own103v/own103v2 owner sets, blind-label transcripts) and
     any d108*.  h25/h29 are no longer live (idle since 2026-09-14 / pushed f0524f75).
  3. A FIFTH UNIT CLASS, round scratch.  Round C's scratch class was only ~/tmp/h28 and
     old harness leftovers, and it left ~10 GiB of closed-round scratch "for the owner"
     (doc 106 sec 7).  The owner's words this round cover it: "retire the intermediate
     test files".  A round dir is a top-level ~/tmp dir whose name carries a round
     number (doc|pr|d|p|h|q<N>) that maps to a committed doc, or one listed in
     EXTRA_ROUNDS.  Each IMMEDIATE CHILD of it is a unit unless it
       * is or contains a pin (the pin class judges it, value-first),
       * is a prep_* / prep* dir (the prep class judges it),
       * is a registered git worktree (removed with `git worktree remove` by the sweep,
         never rm -rf, so the repo is not left with a stale registration),
       * is scan-facing (scan*, own*, shots*, prep*): a set that was served to a scanner,
       * names a SURVIVING arm as a [-_.]-separated token (arm_p100flip, chain_p98von,
         d102m-A-mcp1k.log):
         the run logs of production / scan-source arms stay with the arm,
       * is named in a PROTECTED.txt, or
       * was written in the last 2 h.
     A round dir with no children (a loose file) is itself the unit.
  4. REGISTERED WORKTREES are listed for the sweep, not put in the rm tier.

FREED BYTES ARE SET-RELATIVE (round A hardlinked pins): an inode counts only if
every link to it lies inside the release set.
"""
import os, re, sys, glob, time, json, subprocess, collections

T     = "/home/xqian/tmp"
R     = "/home/xqian/toolkit-dev/wcp-porting-img"
TK    = "/home/xqian/toolkit-dev/toolkit"
HERE  = os.path.dirname(os.path.abspath(__file__))
STAMP = "20260921"
# CENSUS_SUFFIX: a re-census writes <name>.<suffix>.{txt,json} and never touches the committed
# plan-time records (the doc-104 defect the planner's PLAN_SUFFIX fixed; round D review).
SUFFIX = ('.' + os.environ["CENSUS_SUFFIX"]) if os.environ.get("CENSUS_SUFFIX") else ""
CL    = f"{T}/claude-25225"
NOW   = time.time()

ROOTS = [f"{R}/pdvd/docs", f"{R}/pdhd/docs", f"{R}/sbnd/sbnd_xin/docs", f"{R}/pdvd/scripts",
         f"{R}/pdhd/scripts", f"{R}/sbnd/sbnd_xin/scripts", f"{R}/pdhd/stm_michel_scan",
         f"{TK}/clus/docs"]
PERMANENT = {
    f"{T}/d102/libpin_d102": "the pin of BOTH production PR arms: PDHD d108hflip (doc pdvd/108 sec 3) and PDVD d103vflip (doc pdvd/103 sec 10.5)",
    f"{T}/d102m-libsnap":    "sbnd production pin: work-*-d102m / d102mpr (doc 102; sbnd PROTECTED)",
    f"{T}/pdhdstm_libpin":   "pdhd PROTECTED line 37 (stm-tagger-chain line 901, backs the stm0/stmw scan)",
}
# ROUND I: this is the SECOND copy of the held-prefix list -- sweep_tmp has its own at its
# lines 73/133, and narrowing only that one would have left every unit here KEEP, so the
# sweep would have had nothing to refuse and the hold would have looked like a pass.
# Round D set these for the pdvd doc-113 PEER, which is finished; docs pdvd/113-116 are all
# committed and pushed and ListAgents shows no live peer.  Narrowed to this round's own open
# rounds, matching plan_20260921.py's sbnd open_prefix and sweep_tmp_20260921.sh.
LIVE_TOP = ("d119", "d120")
THIS     = "87534ba0-ada7-4811-b7af-02fbd8996b32"
LIVE_SESSIONS = {THIS: "this session",
                 "b41f3f5a-bc3d-48f8-87a3-9359dfd245bf": "peer steiner-path-3d-deviation (doc pdvd/113, live)"}
HARNESS_KEEP = {"bash-edit-diff", "bundled-skills", "auto-mode-classifier-errors"}
# round dirs whose name carries no round number
EXTRA_ROUNDS = {"xtrack": "doc pdvd/47 (transverse smearing in simulation), committed; its summary is figs/47_*",
 # ROUND I: `mg18` does not match RNUM -- "mg" is not a round prefix -- so the round-scratch
 # loop hit `continue` and the directory was SILENTLY SKIPPED: never classified, never in the
 # tier, and absent from the census report altogether, so its 17.5 GiB looked like it had
 # simply been judged a keep.  Same shape as round D's lesson 2 (the grammar refused pr149).
 # EXTRA_ROUNDS is the designed escape hatch for a round whose name carries no number the
 # grammar can map, and the owner released this one by name on 2026-09-21 ("Release all of
 # it"), exactly as the mg10 precedent (doc 105 sec 13) requires.
 "mg18": "the undocumented 2026-09-18 master-merge validation round (toolkit b93673ef/d2646110), released BY NAME by the owner on 2026-09-21; 17.5 GiB, of which 17.4 is four CMake build trees (9 CMakeCache.txt)"}
# never round scratch, whatever the name says
NOT_ROUNDS = {"claude-25225", "cleanup-20260921"}
SCANFACING = re.compile(r"^(scan|own|shots|prep)")

ARM  = re.compile(r"\b(?:pr|p|d|h|q)\d{1,3}[a-z][a-z0-9]{0,14}\b")
EVT  = re.compile(r"^\d{6}_\d+_(.+)$")
SAMP = re.compile(r"-(mcp1k|mcp2k|ncpi0|nuecc48|dbg25a)(?=-|$)|^(mcp1k|mcp2k|ncpi0|nuecc48|dbg25a)-")
RNUM = re.compile(r"^(?:doc|pr|d|p|h|q)(\d{1,3})(?![0-9])")

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
    """Substrate AND production names of the plan.  Round D: a doc that names production
    as its BASELINE does not make its round's pin a production pin -- measured on the first
    run, pr149/libpin and pr149r2/libpin were "kept" only because doc pr/149 names d102m."""
    src = open(f"{HERE}/plan_{STAMP}.py").read()
    s = set()
    for m in re.finditer(r"(?:substrate|production)=\[(.*?)\]", src, re.S):
        s |= set(re.findall(r'"([^"]+)"', m.group(1)))
    s |= {SAMP.sub("", x[5:]).strip("-") for x in s if x.startswith("work-")}
    return s - {"(bare)"}

def committed_doc_numbers():
    out = set()
    for droot in (f"{R}/pdvd/docs", f"{R}/pdhd/docs", f"{R}/sbnd/sbnd_xin/docs"):
        for cur, _s, fs in os.walk(droot):
            for f in fs:
                m = re.match(r"^(\d{1,3})[_-]", f)
                if m and f.endswith(".md"): out.add(int(m.group(1)))
    return out

def protected_paths():
    out = set()
    for p in (f"{R}/sbnd/sbnd_xin/scripts/retire/PROTECTED.txt", f"{R}/pdhd/scripts/retire/PROTECTED.txt",
              f"{R}/pdvd/scripts/retire/PROTECTED.txt"):
        for line in open(p):
            if line.startswith("#"): continue
            for tok in line.split("\t")[0].split():
                if tok.startswith(T + "/"): out.add(tok.rstrip("/"))
    return out

def registered_worktrees():
    r = subprocess.run(["git", "-C", R, "worktree", "list", "--porcelain"], capture_output=True, text=True).stdout
    return {l.split(" ", 1)[1] for l in r.splitlines() if l.startswith("worktree ")} - {R}

def newest(p):
    if not os.path.isdir(p) or os.path.islink(p):
        try: return os.lstat(p).st_mtime
        except OSError: return 0
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
    docs = committed_doc_numbers()
    prot = protected_paths()
    wts = registered_worktrees()
    units = []          # (kind, path, verdict, why)

    # ---- pins ----------------------------------------------------------------
    pins = []
    for cur, subs, fs in os.walk(T):
        depth = cur[len(T):].count("/")
        subs[:] = [x for x in subs if depth < 3 and not x.startswith(("claude-", "prep_"))
                   and not os.path.islink(os.path.join(cur, x))]
        if "libWireCellClus.so" in fs and cur != T:
            pins.append(cur); subs[:] = []
    preps = sorted(p for p in glob.glob(f"{T}/*/prep_*") if os.path.isdir(p) and not os.path.islink(p))

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
            units.append(("pin", p, "KEEP", f"live/held prefix {top}")); continue
        if p in prot:
            units.append(("pin", p, "KEEP", "named in a PROTECTED.txt")); continue
        rm = re.match(r"^(?:pin_)?((?:pr|d|p|h|q)\d+)", top)
        rnd = rm.group(1) if rm else None
        surv = set()
        for f, t in namers[p]:
            surv |= {a for a in ARM.findall(t) if a in alive}
        surv -= {a for a in sub if not (rnd and re.match(rf"^{rnd}(?![0-9])", a))}   # substrate + production
        n = len(namers[p])
        if surv:
            units.append(("pin", p, "KEEP", f"named by {n} file(s); survivors {' '.join(sorted(surv)[:6])}; non-.so {nonso} MiB"))
        else:
            units.append(("pin", p, "FREE", f"named by {n} file(s), no surviving arm; non-.so {nonso} MiB"))

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
            units.append(("prep", p, "KEEP", "live/held prefix"))
        elif os.path.relpath(p, T) in scanrec:
            units.append(("prep", p, "KEEP", "named by a scan record (served to the scanners)"))
        elif arm in alive:
            units.append(("prep", p, "KEEP", f"arm {arm} survives round D"))
        else:
            units.append(("prep", p, "FREE", f"arm {arm} not on disk after round D"))

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
    # ---- harness leftovers at the claude-25225 root ---------------------------------
    for e in sorted(os.listdir(CL)):
        p = f"{CL}/{e}"
        if e.startswith("-") or e in HARNESS_KEEP: continue
        age = NOW - newest(p)
        if named_docs.get(e):
            units.append(("scratch", p, "KEEP", f"named by {named_docs[e][0]}"))
        elif age > 7 * 86400:
            units.append(("scratch", p, "FREE", f"harness leftover, {int(age / 86400)} d old"))

    # ---- round scratch (header item 3) ----------------------------------------------
    pin_set, prep_set = set(pins), set(preps)
    wt_list = []
    for top in sorted(os.listdir(T)):
        tp = f"{T}/{top}"
        if top in NOT_ROUNDS or top.startswith(("claude-", "code-", "mcp-")) or os.path.islink(tp):
            continue
        if top in EXTRA_ROUNDS:
            ground = EXTRA_ROUNDS[top]
        else:
            m = RNUM.match(top)
            if not m or int(m.group(1)) not in docs: continue
            ground = f"round {m.group(0)} -> committed doc number {int(m.group(1))}"
        if top.startswith(LIVE_TOP):
            units.append(("round", tp, "KEEP", f"live/held prefix ({ground})")); continue
        if tp in PERMANENT or tp in pin_set or tp in prot:
            continue
        children = [tp] if not os.path.isdir(tp) else [f"{tp}/{c}" for c in sorted(os.listdir(tp))]
        for c in children:
            leaf = os.path.basename(c)
            if os.path.islink(c):
                # measured on the first run: d145flipproof/fhicl is a symlink to a dir; the
                # unit guard refuses symlinked dirs, so a link is never a unit.
                units.append(("round", c, "KEEP", "a symlink, never a unit")); continue
            if c in wts:
                wt_list.append(c); units.append(("round", c, "KEEP", "registered worktree -> git worktree remove")); continue
            if any(q == c or q.startswith(c + "/") for q in pin_set | set(PERMANENT)):
                units.append(("round", c, "KEEP", "is/contains a pin (judged in the pin class)")); continue
            if c in prep_set or any(q.startswith(c + "/") for q in prep_set) or leaf.startswith("prep"):
                units.append(("round", c, "KEEP", "is/contains a prep (judged in the prep class)")); continue
            if any(w.startswith(c + "/") for w in wts):
                units.append(("round", c, "KEEP", "contains a registered worktree")); continue
            if SCANFACING.match(leaf):
                units.append(("round", c, "KEEP", "scan-facing (served to a scanner)")); continue
            # arm_<arm> / chain_<arm> / sp_<arm>: the run logs of an arm that SURVIVES (production
            # and scan-source provenance) stay with it.
            # [-_.]: measured on the second run, ~/tmp/d102m-A-mcp1k.log (sbnd PRODUCTION's own run
            # log) split on [_.] alone gives "d102m-A-mcp1k" and was released as doc-102 scratch.
            hit = sorted(t for t in re.split(r"[-_.]", leaf) if t in alive)
            if hit:
                units.append(("round", c, "KEEP", f"logs/products of surviving arm {hit[0]}")); continue
            if any(q == c or q.startswith(c + "/") or c.startswith(q + "/") for q in prot):
                units.append(("round", c, "KEEP", "named in a PROTECTED.txt")); continue
            if NOW - newest(c) < 2 * 3600:
                units.append(("round", c, "KEEP", "written in the last 2 h")); continue
            if cwd_inside(c):
                units.append(("round", c, "KEEP", "a process has its cwd inside")); continue
            units.append(("round", c, "FREE", ground))

    # ---- collapse: a round dir whose every child is FREE is ONE unit ------------------
    # (measured on the first run: 21437 per-file units, mostly loose logs of d38_arms2-like dirs)
    kids = collections.defaultdict(list)
    for i, (k, p, v, w) in enumerate(units):
        if k == "round" and p != T + "/" + p[len(T) + 1:].split("/")[0]:
            kids[T + "/" + p[len(T) + 1:].split("/")[0]].append(i)
    drop = set()
    for top, idx in kids.items():
        if os.listdir(top) and len(idx) == len(os.listdir(top)) and all(units[i][2] == "FREE" for i in idx):
            if any(q.startswith(top + "/") for q in pins + preps) or top in prot:
                continue
            drop |= set(idx)
            units.append(("round", top, "FREE", units[idx[0]][3] + f" (whole dir, {len(idx)} entries)"))
    units = [u for i, u in enumerate(units) if i not in drop]

    # ---- guards on the unit list itself --------------------------------------------
    for kind, p, v, _w in units:
        assert p.startswith(T + "/") and p != T, p
        assert not (v == "FREE" and os.path.islink(p) and os.path.isdir(p)), f"symlinked dir unit {p}"
        assert not (v == "FREE" and any(p == k or k.startswith(p + "/") for k in PERMANENT)), f"{p} contains a permanent pin"
        assert not (v == "FREE" and (p + "/").startswith(f"{CL}/-home-xqian-toolkit-dev-toolkit/{THIS}/")), p
        assert not (v == "FREE" and p[len(T) + 1:].startswith(LIVE_TOP)), f"held unit {p}"
        assert not (v == "FREE" and any(w == p or w.startswith(p + "/") for w in wts)), f"worktree inside {p}"

    free = [p for _k, p, v, _w in units if v == "FREE"]
    # nested FREE units (a FREE pin inside a FREE round child cannot happen by construction; assert it)
    fs_ = sorted(free)
    for a, b in zip(fs_, fs_[1:]):
        assert not b.startswith(a + "/"), f"nested units {a} / {b}"
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
    print(f"alive families after round D: {len(alive)} (work dirs released by the plan: {nrel})")
    for kind in ("pin", "prep", "session", "scratch", "round"):
        print(f"\n== {kind}: KEEP {by[kind]['KEEP']}  FREE {by[kind]['FREE']}")
        for k, p, v, w in units:
            if k == kind and (v == "KEEP" or kind not in ("prep", "session", "scratch")):
                print(f"  {v:<4}  {os.path.relpath(p, T):<60} {w}")
        if kind in ("prep", "session", "scratch"):
            fr = collections.Counter(p[len(T) + 1:].split("/")[0] for k, p, v, _w in units if k == kind and v == "FREE")
            print("  FREE by top dir: " + " ".join(f"{r}:{n}" for r, n in sorted(fr.items())))
    print(f"\nregistered worktrees for `git worktree remove` (not in the rm tier): {len(wt_list)}")
    for w in wt_list: print(f"  WT    {w}")
    print(f"\nrelease units {len(free)} | nominal {nominal / 2**30:.2f} GiB | "
          f"FREED (every link inside the set) {freed / 2**30:.2f} GiB | "
          f"shared with a kept file, frees nothing {shared / 2**30:.2f} GiB")
    open(f"{HERE}/tmp_tier_{STAMP}{SUFFIX}.txt", "w").write("\n".join(free) + "\n")
    open(f"{HERE}/tmp_worktrees_{STAMP}{SUFFIX}.txt", "w").write("\n".join(wt_list) + "\n")
    json.dump([dict(kind=k, path=p, verdict=v, why=w) for k, p, v, w in units],
              open(f"{HERE}/tmp_census_{STAMP}{SUFFIX}.json", "w"), indent=1)
    print(f"tier file: {HERE}/tmp_tier_{STAMP}{SUFFIX}.txt ({len(free)} lines).  This script removed nothing.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
