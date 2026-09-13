#!/usr/bin/env python3
"""
Cleanup round 2026-09-12 -- planner for sbnd_xin + pdvd + pdhd.  Doc 105.

OWNER INSTRUCTION, verbatim:
  "it is time to clean up a bit the pdhd, pdvd, sbnd_xin and ~/tmp directories
   to save some disk space.  We have done this multiple times, please do it
   like what we did before."
So the KEEP TEST IS DOC 104's, unchanged (the owner's 09-10 words still govern:
"keep the latest production, but for work* directory that are intermediate
results, we can retire them"; "save the hand scan results for STM ... PDVD and
PDHD").  An arm is kept only if it is
    substrate | latest production | a hand-scan source | live | OPEN.
Read plan_20260910.py's header for why each of those exists.  Fork of
plan_20260910.py; what THIS round changes:

  1. PRODUCTION MOVED, read from primary source, not the `prod` substring.
     PDVD = p96vprod (doc pdvd/96: "the flipped file with NO TLA", confirmed
     against its measurement arm p96vscope -- the pair is one claim).
     PDHD = h26q2dprod (doc pdhd/26 sec 1.2 line 121: "PDHD production =
     h26q2dprod (identical to h26conf on every verdict)"; h26conf is the
     additive-gate baseline, the other half of that claim).
     sbnd = work-*-d102m / d102mpr, unchanged since 09-08.
     p79vprod (09-10's PDVD production) is SUPERSEDED and its PROTECTED.txt
     line is moved to RETIRED, deliberately.

  2. THE ROUND GRAMMAR GREW AN `h`.  PDHD rounds since doc pdhd/18 name arms
     h18b, h22conf, h25kr ...  Every arm-token regex here said (?:p|d), so
     live_tokens() could not see an h-round as live and INTERLOCK 14 could not
     map h22conf to doc pdhd/22.  Both now accept h.  (INTERLOCK 14 failing
     closed would have refused the round; live_tokens failing open would have
     released a live h-round -- the dangerous half.)

  3. THE HAND-SCAN SET GREW, and half of it has no prep set.  smx5..smx9
     (pdvd) and smx18..smx27 + own19..own26 (pdhd) were all scanned after
     09-10.  scan_arms_20260912.py now also reads docs/scan records, because
     smx25/smx26 put h25k/h25kr/h25r movers in front of the scanners through
     prep-pdhd-smx19 re-use rather than a prep dir of their own.

  4. LIVE: the peer round that wrote doc pdhd/26 at 18:24 today is still
     running (~/tmp/h27 written 18:24, h27cfg* work dirs, its transcript
     written 18:31).  h26*/h27* are protected by PREFIX before the first plan
     run -- live_tokens() cannot see ~/tmp, and doc pdhd/26 is committed.

  5. A CONFIRM-TIME RE-PLAN MUST NOT OVERWRITE THE PLAN-TIME RECORD.  On
     2026-09-11 05:00 the 09-10 driver's INTERLOCK A re-ran the planner for
     `1 pdhd`, which rewrote keep_pdvd_20260910.txt (a committed record, +120
     p82vprod lines) and every prebroken_*_20260910.txt baseline.  With
     PLAN_SUFFIX set (the driver sets it) every output of this script goes to
     <name>.<suffix>.txt and the plan-time files are never touched.

This script RETIRES NOTHING.  It prints a plan and writes tier files.
"""
import os, re, sys, json, time, subprocess, collections, glob

R     = "/home/xqian/toolkit-dev/wcp-porting-img"
STAMP = "20260912"
SUFFIX = ('.' + os.environ["PLAN_SUFFIX"]) if os.environ.get("PLAN_SUFFIX") else ""
HERE  = os.path.dirname(os.path.abspath(__file__))
EVT   = re.compile(r"^(\d{6})_(\d+)(?:_(.+))?$")
SAMP  = re.compile(r"-(mcp1k|mcp2k|ncpi0|nuecc48|dbg25a)(?=-|$)|^(mcp1k|mcp2k|ncpi0|nuecc48|dbg25a)-")

def scan_arms(det):
    """The hand-scan sources, re-derived here rather than pasted."""
    p = os.path.join(HERE, f"scan_arms_{STAMP}.json")
    if not os.path.exists(p):
        sys.exit(f"missing {p} -- run: python3 scan_arms_{STAMP}.py --json={p}")
    return list(json.load(open(p)).get(det, []))

TREES = {
 "sbnd": dict(
   root=f"{R}/sbnd/sbnd_xin", work=f"{R}/sbnd/sbnd_xin", unit="siblingdir",
   substrate=["work-dbg25a-ql"],
   # doc 102, ref/prod-2026-09-08.  Unchanged from 09-08: this IS the latest
   # production and the owner's "keep the latest production" names it.
   production=["work-mcp1k-d102m","work-mcp2k-d102m","work-ncpi0-d102m","work-nuecc48-d102m",
               "work-mcp1k-d102mpr","work-mcp2k-d102mpr",
               "work-ncpi0-d102mpr","work-nuecc48-d102mpr"],
   scan_arms=[],
   flip_evidence=[],
   # mg10 = the live master-merge validation round (arms in all three trees).
   open_prefix=("work-mg10","work-nuecc48-mg10","work-mcp1k-mg10","work-mcp2k-mg10",
                "work-ncpi0-mg10","work-d103","work-d104","work-d149","work-d150"),
   keep_arms=[
     # doc pr/148 sec 16 names d145np as the input of a session that has not
     # happened.  An OPEN round's named input stays, campaign or no.
     "work-mcp1k-d145np","work-mcp2k-d145np","work-ncpi0-d145np","work-nuecc48-d145np",
     # the sentinel suite's own negative-control layer and the 31/31 witness.
     "work-s144pos-mcp1k","work-s144pos-mcp2k","work-s144pos-nuecc48",
     "work-s144posleg-mcp1k","work-s144posleg-mcp2k","work-s144posleg-nuecc48",
     "work-s144neg-dvtx","work-s144neg-prox","work-s144neg-gf","work-s144neg-sccc",
     "work-s144neg-memgeo","work-s144neg-backg","work-s144neg-cone","work-s144neg-bfill",
     # INTERLOCK 9 caught these: of the 420 arms em_display's manifests name,
     # these are among the handful that still EXIST, so the display resolves
     # into them (doc 100 round C -- "which arms do the live manifests still
     # RESOLVE into" is the test, not "is it cited").
     "work-pr134-f086-mcp1k","work-pr134-f086-mcp2k",
     "work-pr134-f086-ncpi0","work-pr134-f086-nuecc48",
   ],
   uncited_ok={},
   tier2={}, tier3={}, tier4={},
 ),

 "pdvd": dict(
   root=f"{R}/pdvd", work=f"{R}/pdvd/work", unit="armsuffix",
   # corrected symlink census 2026-09-10, every arm with inbound > 0, plus the
   # bare <run6>_<idx> dirs (pdhd's are 847-inbound substrate; pdvd's are few
   # and cheap and a live round writes into them).
   substrate=["d51vclus","keep","d27fresh","d41prov","d39r2prov","d143pnew",
              "d28dlfp","d34base","d48nu7","d11vtrace","d16vnu","d42fit",
              "d31r6e2e","(bare)"],
   # doc pdvd/96 sec 0 + line 266: p96vprod runs the flipped file with no TLA
   # and d96_confirm.py reads it against p96vscope -- one claim, two arms.
   production=["p96vprod","p96vscope"],
   scan_arms=scan_arms("pdvd"),
   flip_evidence=[],
   # mg10: still no committed doc.  h26/h27: the live PDHD peer round also
   # writes PDVD cfg dirs (h25vcfg 14:26 today).  p97: the next PDVD round's
   # name, nothing on disk yet -- a prefix for a round that has not started
   # costs nothing and is what held on 09-04 and 09-10.
   # h28: a peer session (doc pdhd/27-28) announced a new live round at 19:3x on
   # 2026-09-12 -- work tags h28off/h28wl/h28prod (pdhd), p97voff/p97vwl (pdvd),
   # pin ~/tmp/h28/libpin_h28, record pdhd/docs/scan/d28/ (uncommitted).
   open_prefix=("mg10","h26","h27","h28","p97"),
   keep_arms=["magnify","ql_scores","ql_labels","stm_michel_labels",
              "d08_scan_labels","d08pv_scan_labels"],
   uncited_ok={},
   tier2={}, tier3={}, tier4={},
 ),

 "pdhd": dict(
   root=f"{R}/pdhd", work=f"{R}/pdhd/work", unit="armsuffix",
   substrate=["d51hclus","d09","d09ctl","d09ctl2","stm0","(bare)","d11seponA",
              "d11sepoffA","stmwc","d02prod","d16hnu","wcc","d11prod"],
   # doc pdhd/26 line 121: "PDHD production = h26q2dprod (identical to h26conf
   # on every verdict)"; h26conf is the additive gate's baseline (line 104).
   production=["h26q2dprod","h26conf"],
   scan_arms=scan_arms("pdhd"),
   flip_evidence=[],
   # h26/h27: LIVE peer round (header item 4).  p97: see pdvd.
   # h28: a peer session (doc pdhd/27-28) announced a new live round at 19:3x on
   # 2026-09-12 -- work tags h28off/h28wl/h28prod (pdhd), p97voff/p97vwl (pdvd),
   # pin ~/tmp/h28/libpin_h28, record pdhd/docs/scan/d28/ (uncommitted).
   open_prefix=("mg10","h26","h27","h28","p97"),
   keep_arms=["stmw","allpd","ql_labels","stm_scan_labels","stm_michel_labels",
              "d08_scan_labels","d05_scan_labels"],
   uncited_ok={},
   tier2={}, tier3={}, tier4={},
 ),
}

def load_citations():
    p = os.path.join(HERE, f"cit_{STAMP}.json")
    t = os.path.join(HERE, f"toks_{STAMP}.json")
    for f in (p, t):
        if not os.path.exists(f):
            sys.exit(f"missing {f} -- run:  python3 toks_{STAMP}.py && "
                     f"python3 cit_{STAMP}.py toks_{STAMP}.txt cit_{STAMP}.json")
    d = json.load(open(p))
    toks = {k.split("\t", 1)[1]: v["toks"] for k, v in json.load(open(t)).items()}
    return d["count"], d.get("who", {}), toks

def protected_lines():
    """Union of every PROTECTED.txt.  Returns (exact names, prefixes).

    A line ending in '*' is a PREFIX.  The 09-08 matcher compared
    `arm == line.strip('*')`, which matched nothing, so every `d51v*`-shaped
    line in these files protected exactly zero dirs and the round was relying
    on its own inline open_prefix instead.  Honouring them is strictly the
    safe direction; the lines that should no longer protect anything were
    moved to each file's RETIRED section rather than left to fail silently."""
    names, prefixes = set(), set()
    for p in (f"{R}/sbnd/sbnd_xin/scripts/retire/PROTECTED.txt",
              f"{R}/pdhd/scripts/retire/PROTECTED.txt",
              f"{R}/pdvd/scripts/retire/PROTECTED.txt"):
        if not os.path.exists(p): continue
        for line in open(p):
            line = line.strip()
            if not line or line.startswith("#"): continue
            for tok in line.split("\t")[0].split():
                (prefixes if tok.endswith("*") else names).add(tok.rstrip("*"))
    return names, prefixes

DOCROOTS = ("pdvd/docs", "pdhd/docs", "sbnd/sbnd_xin/docs",
            "pdvd/scripts", "pdhd/scripts", "sbnd/sbnd_xin/scripts")

def live_tokens():
    """Arm families named by a record that is NOT YET COMMITTED.

    THE RULE THIS REPLACES was a hand-typed open_prefix, and on 2026-09-10 it
    was already wrong: doc pdvd/82 was written at 21:13 and its p82v* arms at
    21:26, while the list said p79/p80/p81.  A live round is exactly a round
    whose record is still uncommitted, so derive it instead of typing it --
    that is self-maintaining and cannot go stale between rounds.

    Untracked file  -> every arm token in it is live.
    Modified file   -> only tokens on ADDED lines (a tracked doc names its
                       whole campaign's history; the new lines are the live part).
    Binary files are skipped: pdhd/docs/scripts/ currently holds a stray ELF
    executable, which `strings` would happily mine for false arm tokens."""
    toks, files = set(), []
    g = subprocess.run(["git","-C",R,"status","--porcelain","--"] + list(DOCROOTS),
                       capture_output=True, text=True).stdout.splitlines()
    # {0,14}: {1,14} could not match a 4-character arm such as d53v (doc 105).
    ARMRE = re.compile(r"\b(?:p|d|h)[0-9]{1,3}[a-z][a-z0-9]{0,14}\b")
    for line in g:
        st, _, path = line[:2], None, line[3:].strip()
        # THE ROUND MUST NOT READ ITS OWN OUTPUT.  tier1_*.txt, keep_*.txt and
        # toks.txt all live under scripts/retire and name EVERY arm in the
        # tree; on the first run they made 154 pdvd families look "live" and
        # cut the release from 119.93 GiB to 11.40.  This is doc 91's
        # "protected because protected" and the 09-05/09-06 recurrence of it --
        # cit_*.py carries --exclude-dir=retire for exactly this reason.
        if "/retire/" in f"/{path}" or path.rstrip("/").endswith("/retire"):
            continue
        # ...AND NOT ITS OWN DOCS.  2026-09-12, measured: the only thing marking
        # p82vprod "live" was an uncommitted edit to doc 104 (its sec 15 names the
        # arms the 09-11 re-plan wrote), and doc 105 will name every arm it
        # releases.  A cleanup doc names arms in order to RELEASE them; reading it
        # as liveness is doc 91's "protected because protected" on the doc layer.
        if re.search(r"(^|/)[0-9]*_?[^/]*(cleanup|retire)[^/]*$", path):
            continue
        full = os.path.join(R, path)
        paths = []
        if os.path.isdir(full):
            for cur, _s, fs in os.walk(full): paths += [os.path.join(cur, f) for f in fs]
        else:
            paths = [full]
        for fp in paths:
            try:
                with open(fp, "rb") as fh: head = fh.read(4096)
                if b"\0" in head: continue            # binary: skip
                if st.strip() == "??":
                    text = open(fp, errors="ignore").read()
                else:
                    text = subprocess.run(["git","-C",R,"diff","-U0","--",
                                           os.path.relpath(fp, R)],
                                          capture_output=True, text=True).stdout
                    text = "\n".join(l[1:] for l in text.splitlines()
                                     if l.startswith("+") and not l.startswith("+++"))
            except OSError:
                continue
            hits = set(ARMRE.findall(text))
            if hits: files.append((path, sorted(hits)[:6]))
            toks |= hits
    return toks, files

LIVE, LIVE_FILES = live_tokens()
CIT, WHO, TOKS = load_citations()
PROT, PROTPFX  = protected_lines()
fails          = []

def prot_hit(d, a):
    if d in PROT or (a and a in PROT): return True
    for p in PROTPFX:
        if d.startswith(p) or (a and a.startswith(p)): return True
    return False

def check(tree, n, ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  INTERLOCK {n}: {msg}")
    if not ok: fails.append((tree, n))

def du_kb(paths):
    out = {}
    for i in range(0, len(paths), 400):
        r = subprocess.run(["du","-sk"]+paths[i:i+400], capture_output=True, text=True).stdout
        for l in r.splitlines():
            kb, p = l.split("\t", 1); out[os.path.basename(p)] = int(kb)
    return out

def arm_token(d, unit):
    if unit == "armsuffix":
        m = EVT.match(d)
        return None if not m else (m.group(3) or "(bare)")
    return SAMP.sub("", d[5:]).strip("-") if d.startswith("work-") else None

def owner_of(linkpath):
    """Owning arm dir of a symlink TARGET, by grammar match on the normalised
    path parts.  Never relpath against a ROOT constant -- that was vacuous for
    absolutely-spelled links because /nfs/data/1/... is itself a symlink."""
    full = os.path.normpath(os.path.join(os.path.dirname(linkpath),
                                         os.readlink(linkpath)))
    own = None
    for part in full.split(os.sep):
        if EVT.match(part) or part.startswith("work-"): own = part
    return own

def plan_tree(tree, cfg):
    print(f"\n{'='*78}\n== {tree}   ({cfg['work']})\n{'='*78}")
    WORK, unit = cfg["work"], cfg["unit"]
    entries = sorted(d for d in os.listdir(WORK)
                     if os.path.isdir(os.path.join(WORK, d))
                     and not os.path.islink(os.path.join(WORK, d)))
    parsed    = {d: arm_token(d, unit) for d in entries}
    universe  = {d: a for d, a in parsed.items() if a is not None}
    out_scope = sorted(d for d, a in parsed.items() if a is None)

    SCAN = set(cfg["scan_arms"])
    keep_names = (set(cfg["substrate"]) | set(cfg["production"]) | SCAN
                  | set(cfg["flip_evidence"]) | set(cfg["keep_arms"]))
    def is_open(d, a):
        if d.startswith(cfg["open_prefix"]) or (a or "").startswith(cfg["open_prefix"]):
            return True
        return bool(a) and a in LIVE        # derived: named by an uncommitted record
    keep_dirs = {d for d, a in universe.items()
                 if d in keep_names or a in keep_names or is_open(d, a)}
    for d, a in universe.items():
        if prot_hit(d, a): keep_dirs.add(d)

    cand  = {d for d in universe} - keep_dirs
    rest  = sorted(cand)

    def cited(d):
        return max([CIT.get(d, 0), CIT.get(universe[d] or "", 0)]
                   + [CIT.get(t, 0) for t in TOKS.get(d, ())])
    # REPORTED, NOT OBEYED.  See header item 1.
    tier1 = list(rest)
    tier2 = tier3 = tier4 = []

    # --- transitive closure: keeping a dir means keeping its substrate -----
    relset = set(tier1)
    for _ in range(12):
        pull = set()
        for d in keep_dirs:
            for cur, sub, files in os.walk(os.path.join(WORK, d)):
                for e in files + sub:
                    fp = os.path.join(cur, e)
                    if os.path.islink(fp):
                        o = owner_of(fp)
                        if o in relset: pull.add(o)
        if not pull: break
        keep_dirs |= pull; relset -= pull
    closure = sorted(set(tier1) - relset)
    tier1 = [d for d in tier1 if d in relset]
    if closure:
        print(f"        closure: +{len(closure)} dirs pulled back as substrate of a kept dir")
    KEEP = sorted(keep_dirs)

    def ndirs(name):
        return sum(1 for d, a in universe.items() if d == name or a == name)

    # ---- INTERLOCK 1: substrate present
    short = {a: ndirs(a) for a in cfg["substrate"] if ndirs(a) == 0}
    check(tree, 1, not short, f"substrate present ({short or 'all present'})")

    # ---- INTERLOCK 2: no KEPT symlink may resolve into a releasing dir
    dangle = []
    for d in KEEP:
        for cur, sub, files in os.walk(os.path.join(WORK, d)):
            for e in files + sub:
                fp = os.path.join(cur, e)
                if os.path.islink(fp) and owner_of(fp) in relset:
                    dangle.append(f"{d}: {os.path.relpath(fp, WORK)} -> {owner_of(fp)}")
            if len(dangle) > 5: break
    check(tree, 2, not dangle, f"no kept symlink resolves into a releasing dir "
          f"({len(dangle)} would dangle{'; e.g. ' + dangle[0] if dangle else ''})")

    # ---- INTERLOCK 3: live-WRITER guard (mtime double-sample + scoped ps)
    def snap(dirs):
        out = {}
        for d in dirs:
            acc = []
            for cur, sub, files in os.walk(os.path.join(WORK, d)):
                for f in files:
                    try: acc.append(os.path.getmtime(os.path.join(cur, f)))
                    except OSError: pass
                if len(acc) > 3000: break
            out[d] = (len(acc), max(acc) if acc else 0)
        return out
    sample = tier1[::max(1, len(tier1)//120)] if tier1 else []
    before = snap(sample)
    ps  = subprocess.run(["ps","-eo","cmd"], capture_output=True, text=True).stdout
    busy = [l for l in ps.splitlines()
            if re.search(r"wire-cell|run_pr_evt|run_clus_evt|run_img_evt|wcsonnet", l)
            and cfg["root"] in l and f"plan_{STAMP}" not in l and "grep" not in l]
    time.sleep(12)
    moved = [d for d in sample if before[d] != snap([d])[d]]
    check(tree, 3, not moved and not busy,
          f"no live writer ({len(moved)} of {len(sample)} sampled dirs moved, "
          f"{len(busy)} tree-scoped wire-cell procs)")

    # ---- INTERLOCK 4: pre-existing broken symlinks recorded BEFORE the round
    pre = subprocess.run(["find", cfg["root"], "-xtype", "l"],
                         capture_output=True, text=True).stdout.split()
    open(os.path.join(HERE, f"prebroken_{tree}_{STAMP}{SUFFIX}.txt"), "w").write("\n".join(pre))
    check(tree, 4, True, f"pre-existing broken symlinks recorded: {len(pre)} "
                         f"(the post-state number only means something against this)")

    # ---- INTERLOCK 5: PROTECTED.txt names nothing in the release
    hit = sorted({d for d in tier1 if prot_hit(d, universe[d])})
    check(tree, 5, not hit, f"PROTECTED.txt clear of the release ({hit[:4] or 'clear'})")

    # ---- INTERLOCK 7: nothing being released is a record/label dir
    RECORD = ("labels","decisions","ql_labels","stm_scan_labels","stm_michel_labels",
              "snap","sweep","em_labels","vertex_labels","scan_labels","state-","archive")
    bad = [d for d in tier1 if any(k in d for k in RECORD)]
    check(tree, 7, not bad, f"no record/label dir in the release ({bad or 'clear'})")

    # ---- INTERLOCK 8: never delete THROUGH a symlink
    thru = [d for d in tier1 if os.path.islink(os.path.join(WORK, d))]
    check(tree, 8, not thru, f"no target is itself a symlink ({thru or 'clear'})")

    # ---- INTERLOCK 9: no arm a LIVE manifest still RESOLVES into
    man = set()
    for pat in ("em_display/*.tsv","em_display/*.txt","*/manifest*.tsv","docs/scan/*.tsv",
                "*_scan/*.tsv","*_labels/*.json","*_labels/*/*.json",
                "*_scan/prep*/*.json","*_scan/*.py"):
        for f in glob.glob(os.path.join(cfg["root"], pat)):
            try: t = open(f, errors="ignore").read()
            except OSError: continue
            for mm in re.finditer(r"work-[A-Za-z0-9_.-]+|[0-9]{6}_[0-9]+_[A-Za-z0-9-]+", t):
                man.add(mm.group(0))
    resolved = {a for a in man if os.path.isdir(os.path.join(WORK, a))}
    inrel    = sorted(resolved & relset)
    check(tree, 9, not inrel, f"manifests name {len(man)} arms, {len(resolved)} exist, "
          f"{len(inrel)} in the release ({inrel[:4] or 'none'})")

    # ---- INTERLOCK 11: the production and substrate sets must RESOLVE
    unres = sorted(n for n in cfg["substrate"] + cfg["production"] + cfg["scan_arms"]
                   if not any(d == n or universe[d] == n for d in universe))
    check(tree, 11, not unres,
          f"every substrate/production/scan name resolves ({unres or 'all resolve'})")

    # ---- INTERLOCK 13 (NEW): the hand scans keep their SOURCE arm.
    # Re-derived here from the scan tooling, not read from the tier config, so
    # that a config typo cannot quietly drop a scan.  The owner asked for the
    # STM scans by name; this is that request made mechanical.
    sa = set(cfg["scan_arms"])
    lost = sorted(a for a in sa
                  if any(universe[d] == a for d in tier1))
    missing = sorted(a for a in sa if ndirs(a) == 0)
    check(tree, 13, not lost and not missing,
          f"hand-scan sources kept: {sorted(sa) or 'none for this tree'}"
          f"{'; IN RELEASE: ' + str(lost) if lost else ''}"
          f"{'; DOES NOT RESOLVE: ' + str(missing) if missing else ''}")

    # ---- INTERLOCK 14 (NEW): an UNCITED family may not be released silently.
    # Citation no longer keeps an arm (header item 1), which removes the only
    # thing that used to make an undocumented family visible.  So the burden
    # flips: a cited family releases on its doc; an UNCITED one must be shown
    # to belong to a round whose doc IS committed.  Derived from the family's
    # own round number, not from a hand-written excuse -- d11fvoff releases
    # because docs/11_*.md exists and is committed, and if no such doc existed
    # the family would be an undocumented round and this would refuse.
    RNUM = re.compile(r"^(?:d|p|h)(\d{1,3})")
    committed_docs = set()
    for droot in (f"{R}/pdvd/docs", f"{R}/pdhd/docs", f"{R}/sbnd/sbnd_xin/docs"):
        for cur, _s, fs in os.walk(droot):
            for f in fs:
                m = re.match(r"^(\d{1,3})[_-]", f)
                if m and f.endswith(".md"): committed_docs.add(int(m.group(1)))
    unc = collections.Counter()
    for d in tier1:
        if not cited(d): unc[universe[d]] += 1
    ungrounded = []
    for a in unc:
        m = RNUM.match(a or "")
        if not (m and int(m.group(1)) in committed_docs):
            ungrounded.append(a)
    ungrounded = sorted(ungrounded)
    check(tree, 14, not ungrounded,
          f"{len(unc)} uncited families in the release; each maps to a committed "
          f"doc by round number ({len(ungrounded)} do not: {ungrounded[:6] or 'none'})")

    # ---- INTERLOCK 15 (NEW): the RECORD must outlive the BYTES.
    # The whole licence for this round is that a committed doc replaces the
    # per-event products.  So no released family may be named by a record that
    # is still uncommitted -- that arm's doc could still change, or be a live
    # round's.  This is the negative half of INTERLOCK 16.
    leak = sorted({universe[d] for d in tier1 if universe[d] in LIVE})
    check(tree, 15, not leak,
          f"nothing released is named by an uncommitted record ({leak or 'clear'})")

    # ---- INTERLOCK 16 (NEW): show what the live derivation actually found,
    # so a PASS above is readable rather than vacuous (doc 91's "protected
    # because protected" and the 09-05 zero-fires shape).
    mine = sorted({a for a in LIVE if any(universe[d] == a for d in universe)})
    check(tree, 16, True,
          f"live-by-uncommitted-record: {len(LIVE)} tokens from "
          f"{len(LIVE_FILES)} file(s); {len(mine)} resolve in this tree "
          f"({mine[:8] or 'none'})")

    # ------------------------------------------------------------- report --
    sz = du_kb([os.path.join(WORK, d) for d in tier1 + KEEP])
    t1kb = sum(sz.get(d, 0) for d in tier1)
    keepkb = sum(sz.get(d, 0) for d in KEEP)
    print(f"\n  universe {len(universe)} dirs | KEEP {len(KEEP)} = {keepkb/1048576:.2f} GiB"
          f" | RELEASE {len(tier1)} = {t1kb/1048576:.2f} GiB"
          f" | out-of-scope (untouched) {len(out_scope)}")
    byarm = collections.Counter()
    for d in tier1: byarm[universe[d]] += sz.get(d, 0)
    print(f"  --- RELEASE ---   {'arm':<24}{'dirs':>6}{'GiB':>9}{'cited':>7}")
    for a, kb in byarm.most_common(25):
        n = sum(1 for d in tier1 if universe[d] == a)
        print(f"                    {a:<24}{n:>6}{kb/1048576:>9.2f}{CIT.get(a,0):>7}")
    if len(byarm) > 25: print(f"                    ... and {len(byarm)-25} more families")
    tf = os.path.join(HERE, f"tier1_{tree}_{STAMP}{SUFFIX}.txt")
    with open(tf, "w") as fh:
        for d in tier1: fh.write(os.path.join(WORK, d) + "\n")
    print(f"  tier 1 file: {tf}  ({len(tier1)} lines, {t1kb/1048576:.2f} GiB)")
    kf = os.path.join(HERE, f"keep_{tree}_{STAMP}{SUFFIX}.txt")
    with open(kf, "w") as fh:
        for d in KEEP: fh.write(os.path.join(WORK, d) + "\n")
    return t1kb

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in TREES] or list(TREES)
    g1 = 0
    for t in want: g1 += plan_tree(t, TREES[t])
    print(f"\n{'='*78}\nGRAND TOTAL release {g1/1048576:.2f} GiB")
    print(f"interlock failures: {fails or 'NONE'}")
    print("\nThis script retired nothing.  Review the tier files, then run the "
          "retire driver (CONFIRM=yes).")
    sys.exit(1 if fails else 0)
