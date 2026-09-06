#!/usr/bin/env python3
"""
Cleanup round 2026-09-05 -- planner for pdhd + pdvd + sbnd_xin.

Fork of pdvd/scripts/retire/plan_20260904.py.  Two things carried deliberately:
  * INTERLOCK 2 resolves a link's owning arm by matching the arm GRAMMAR against
    the normalised path parts.  It does NOT relpath against a ROOT constant --
    that formulation was vacuous for absolutely-spelled links because
    /nfs/data/1 is itself a symlink for /home/xqian, and it cost work-dbg25a-ql
    in the 09-04 round (memory: feedback_symlink_check_path_alias).
  * INTERLOCK 3 is a live-WRITER guard, not an age threshold.  Age is not
    liveness: most targets here were written hours ago by closed rounds, while
    an untouched dir can belong to a running session.

NEW in this round -- INTERLOCK 6, the cross-repo citation resolution.  In the
09-04 round pdvd docs 28/31 cited an SBND arm that the SBND-only blast-radius
check never saw, and it was released.  Every tree is now resolved against the
docs and scripts of ALL THREE trees.

This script RETIRES NOTHING.  It prints a plan and a tier file.
"""
import os, re, sys, time, json, subprocess, collections

R = "/home/xqian/toolkit-dev/wcp-porting-img"
EVT = re.compile(r"^(\d{6})_(\d+)(?:_(.+))?$")     # <run6>_<idx>[_<arm>]
STAMP = "20260905"

# ---------------------------------------------------------------- per tree --
# `unit` says what a releasable object IS in this tree.  pdvd/pdhd group by
# arm suffix under a single work/; sbnd_xin's arms are sibling work-* dirs.
# Forking one grammar onto the other tree gives dirs=0 or one rm -rf work/.
TREES = {
 "pdhd": dict(
   root=f"{R}/pdhd", work=f"{R}/pdhd/work", unit="armsuffix",
   # Primary source: docs/stm-tagger-chain.md:95 (repro) and :901 ("Not
   # committed:").  The 224 hand labels resolve 30/30 into stm0 AND stmw.
   substrate=["(bare)"],                      # the 38 bare dirs stm0 borrows from
   keep_arms=["stm0","stmw","stmc4000","stmc2000","stmc1000","stmc250",
              "phdump","phdumpw","wcc","wccdump","wccdumpw",
              # pulled by INTERLOCK 6: cited outside pdhd (pdvd qlmatch docs)
              "qlt", "perfslide"],
   live_prefix=(),
 ),
 "pdvd": dict(
   root=f"{R}/pdvd", work=f"{R}/pdvd/work", unit="armsuffix",
   # d27fresh = stage_{pr,ql}_tag.sh documented default src_tag (primary
   # source, not a name-read) and 9215 inbound links.  d41prov is the
   # production PR chain doc 43 sec 6 staged its 99 events from.  keep is
   # protected by doc 24:143,234 ("_keep is never removed") despite 0 inbound.
   substrate=["keep","d27fresh","d41prov","d39r2prov"],
   # Arms backing constants that shipped to PDVD production on 2026-09-05:
   # doc 43 sec 6 -> d43p90c5 (+ its A/B set); doc 44 repro -> d42fit, d44sig,
   # d44ref.  An arm that backs a just-made decision is not superseded.
   keep_arms=["d43p90c5","d43prod","d43fvoff","d43fvd50","d43p90c3","d43p80c3",
              "d42fit","d44sig","d44sigb","d44sign6","d44ref","d41prod",
              # d44prod/d38qnewprod surfaced via INTERLOCK 6, not via the name scan
              "d44prod","d38qnewprod"],
   live_prefix=("d45",),                      # PID 1440435 is reading these now
 ),
 "sbnd": dict(
   root=f"{R}/sbnd/sbnd_xin", work=f"{R}/sbnd/sbnd_xin", unit="siblingdir",
   substrate=["work-mcp2k-grp0825","work-mcp1k-grp0825","work-nuecc48-grp0825",
              "work-ncpi0-grp0825","work-dbg25a-ql"],
   keep_arms=["work-mcp2k-d97fv","work-mcp1k-d97fv","work-nuecc48-d97fv","work-ncpi0-d97fv",
              "work-mcp2k-d97fvpr2","work-mcp1k-d97fvpr2","work-nuecc48-d97fvpr2","work-ncpi0-d97fvpr2",
              "work-vtx105-base-mcp2k","work-vtx105-base-mcp1k",
              "work-sent97-mcp2k","work-sent97-mcp1k","work-sent97-nuecc48","work-sent97-ncpi0",
              "work-probe178410a","work-dbg25a-d97prodchk","work-tfix388-r9",
              # INTERLOCK 9: the 10 arms em_display's manifests still RESOLVE
              # into.  Doc 100: the manifests name 420 arms and 10 exist -- the
              # decisive question is resolution, never "is it cited".
              "work-pr130r1-probe141-mcp1k","work-pr130r1-probe141-mcp2k",
              "work-pr130r1-probe98-mcp1k","work-pr130r1-probe98-mcp2k",
              "work-pr130r1-probe98-ncpi0","work-pr130r1-probe98-nuecc48",
              "work-pr134-f086-mcp1k","work-pr134-f086-mcp2k",
              "work-pr134-f086-ncpi0","work-pr134-f086-nuecc48"],
   live_prefix=("work-d45sbnd","work-stmcamp-d45"),
 ),
}

def arm_of(name):
    m = EVT.match(name)
    return None if not m else (m.group(3) or "(bare)")

def du_kb(paths):
    if not paths: return {}
    out = {}
    for i in range(0, len(paths), 500):
        r = subprocess.run(["du","-sk"]+paths[i:i+500], capture_output=True, text=True).stdout
        for l in r.splitlines():
            kb, p = l.split("\t", 1); out[os.path.basename(p)] = int(kb)
    return out

# ------------------------------------------------------- shared inventories --
def citations():
    """Every work-token named in ANY tree's docs/scripts (INTERLOCK 6)."""
    dirs = []
    for t in ("pdhd","pdvd","sbnd/sbnd_xin"):
        for sub in ("docs","scripts"):
            p = f"{R}/{t}/{sub}"
            if os.path.isdir(p): dirs.append(p)
    for extra in (f"{R}/pdhd/stm_scan", f"{R}/pdhd/ql_scan", f"{R}/qlport/scripts"):
        if os.path.isdir(extra): dirs.append(extra)
    # EXCLUDE the retire machinery's own products.  A planner that reads its
    # own tier file scores every candidate as "cited" and protects it because
    # it is protected -- the doc 91 defect, reproduced here on the first run
    # (11 of 12 pdhd dirs came back "cited" and none of them were).
    txt = subprocess.run(["grep","-rhoE","--exclude-dir=retire",
        "--exclude=tier_*.txt","--exclude=*.log",
        r"work[A-Za-z0-9_.-]*|[0-9]{6}_[0-9]+_[A-Za-z0-9-]+"]+dirs,
        capture_output=True, text=True).stdout.split()
    c = collections.Counter(txt)
    # A bare "work" is not an arm name; it matched 2006 times and means nothing.
    for junk in ("work","work-dir","works","working"):
        c.pop(junk, None)
    return c

def protected_lines():
    """Union of every PROTECTED.txt in the tree (field 1, whitespace-split)."""
    names = set()
    for p in (f"{R}/sbnd/sbnd_xin/scripts/retire/PROTECTED.txt",
              f"{R}/pdhd/scripts/retire/PROTECTED.txt",
              f"{R}/pdvd/scripts/retire/PROTECTED.txt"):
        if not os.path.exists(p): continue
        for line in open(p):
            line = line.strip()
            if not line or line.startswith("#"): continue
            names.update(line.split("\t")[0].split())
    return names

CIT  = citations()
PROT = protected_lines()

fails = []
def check(tree, n, ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  INTERLOCK {n}: {msg}")
    if not ok: fails.append((tree, n))

def plan_tree(tree, cfg):
    print(f"\n{'='*74}\n== {tree}   ({cfg['work']})\n{'='*74}")
    WORK = cfg["work"]
    entries = sorted(d for d in os.listdir(WORK)
                     if os.path.isdir(os.path.join(WORK, d))
                     and not os.path.islink(os.path.join(WORK, d)))
    if cfg["unit"] == "armsuffix":
        parsed = {d: arm_of(d) for d in entries}
    else:
        parsed = {d: (d if d.startswith("work-") else None) for d in entries}
    universe   = {d: a for d, a in parsed.items() if a is not None}
    out_scope  = sorted(d for d, a in parsed.items() if a is None)

    keep_arms = set(cfg["substrate"]) | set(cfg["keep_arms"]) | {
        a for a in set(universe.values()) if a.startswith(cfg["live_prefix"])}
    # PROTECTED.txt union, matched on the arm token or the dir name
    for d, a in universe.items():
        if d in PROT or a in PROT or any(a == p.strip("*") for p in PROT):
            keep_arms.add(a)

    RETIRE = sorted(d for d, a in universe.items() if a not in keep_arms)
    # INTERLOCK 6 is LOAD-BEARING, not advisory: any dir carrying an exact
    # citation in ANY tree's docs is pulled out of the release.  M13 protects
    # the record, not the summary of it -- "the doc already has the numbers" is
    # explicitly not a reason (sbnd PROTECTED.txt header).
    CITED  = sorted(d for d in RETIRE if CIT.get(d, 0) or CIT.get(universe[d], 0))
    RETIRE = [d for d in RETIRE if d not in set(CITED)]
    KEEP   = sorted([d for d, a in universe.items() if a in keep_arms] + CITED)

    # TRANSITIVE CLOSURE.  Keeping a cited dir is not enough -- it must keep
    # whatever it borrows from, or INTERLOCK 2 fails.  It did fail here: the
    # cited 039252_5_d37off1 holds img-provenance.txt -> 039252_5_d34base and
    # d34base was in the release.  Iterate to a fixed point.
    def link_owner(linkpath):
        full = os.path.normpath(os.path.join(os.path.dirname(linkpath),
                                             os.readlink(linkpath)))
        own = None
        for part in full.split(os.sep):
            if EVT.match(part) or part.startswith("work-"): own = part
        return own
    keepset, retset = set(KEEP), set(RETIRE)
    for _ in range(12):
        pull = set()
        for d in keepset:
            p_ = os.path.join(WORK, d)
            for cur, sub, files in os.walk(p_):
                for e in files + sub:
                    fp = os.path.join(cur, e)
                    if os.path.islink(fp):
                        o = link_owner(fp)
                        if o in retset: pull.add(o)
        if not pull: break
        keepset |= pull; retset -= pull
    CLOSURE = sorted(keepset - set(KEEP))
    KEEP, RETIRE = sorted(keepset), sorted(retset)
    if CLOSURE:
        print(f"        closure: +{len(CLOSURE)} dirs pulled in as substrate of a kept dir")

    # ---- INTERLOCK 1: substrate present at full coverage
    cnt = collections.Counter(universe.values())
    exp = max(cnt.values()) if cnt else 0
    short = {a: cnt.get(a, 0) for a in cfg["substrate"] if cnt.get(a, 0) == 0}
    check(tree, 1, not short, f"substrate present ({short or 'all present'})")

    # ---- INTERLOCK 2: no KEPT symlink may resolve into a RETIRING dir.
    # Owning dir resolved by grammar match on normalised path parts -- never by
    # relpath against a ROOT constant (the alias defect).
    retset = set(RETIRE)
    def owner_of(linkpath):
        full = os.path.normpath(os.path.join(os.path.dirname(linkpath),
                                             os.readlink(linkpath)))
        own = None
        for part in full.split(os.sep):
            if EVT.match(part) or part.startswith("work-"): own = part
        return own
    dangle = []
    for d in KEEP:
        p = os.path.join(WORK, d)
        for cur, sub, files in os.walk(p):
            for e in files + sub:
                fp = os.path.join(cur, e)
                if os.path.islink(fp) and owner_of(fp) in retset:
                    dangle.append(f"{d}: {os.path.relpath(fp, WORK)} -> {owner_of(fp)}")
            if len(dangle) > 5: break
    check(tree, 2, not dangle,
          f"no kept symlink resolves into a retiring dir "
          f"({len(dangle)} would dangle{'; e.g. ' + dangle[0] if dangle else ''})")

    # ---- INTERLOCK 3: live-writer guard (mtime double-sample + scoped ps)
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
    sample = RETIRE[::max(1, len(RETIRE)//150)] if RETIRE else []
    before = snap(sample)
    ps = subprocess.run(["ps","-eo","cmd"], capture_output=True, text=True).stdout
    busy = [l for l in ps.splitlines()
            if re.search(r"wire-cell|run_pr_evt|run_clus_evt|run_img_evt|wcsonnet", l)
            and cfg["root"] in l and "plan_20260905" not in l and "grep" not in l]
    time.sleep(12)
    moved = [d for d in sample if before[d] != snap([d])[d]]
    live_in_retire = [d for d in RETIRE
                      if (universe[d] or "").startswith(cfg["live_prefix"])] if cfg["live_prefix"] else []
    check(tree, 3, not moved and not busy and not live_in_retire,
          f"no live writer ({len(moved)} dirs moved, {len(busy)} tree-scoped procs, "
          f"{len(live_in_retire)} live-prefix dirs in retire set)")

    # ---- INTERLOCK 4: pre-existing broken symlinks recorded BEFORE the round
    pre = subprocess.run(["find", cfg["root"], "-xtype", "l"],
                         capture_output=True, text=True).stdout.split()
    check(tree, 4, True, f"pre-existing broken symlinks recorded: {len(pre)} "
                         f"(post-state must not exceed this)")

    # ---- INTERLOCK 5: PROTECTED.txt names nothing in the retire set
    hit = sorted({universe[d] for d in RETIRE if d in PROT or universe[d] in PROT})
    check(tree, 5, not hit, f"PROTECTED.txt clear of the retire set ({hit or 'clear'})")

    # ---- INTERLOCK 6 (NEW): cross-repo exact-name citation resolution.
    # Substring matching is what made three inherited protections false in doc
    # 91; this is exact-token.
    still = [d for d in RETIRE if CIT.get(d, 0) or CIT.get(universe[d], 0)]
    check(tree, 6, not still,
          f"{len(CITED)} cited dirs PULLED from the release and kept; "
          f"{len(still)} cited dirs remain in it (must be 0)")
    if CITED:
        print(f"        pulled: {', '.join(CITED[:8])}{' ...' if len(CITED)>8 else ''}")

    # ---- INTERLOCK 7: nothing in the retire set is a record/label dir
    RECORD = ("labels","decisions","ql_labels","stm_scan_labels","snap","sweep","em_labels","vertex_labels")
    bad = [d for d in RETIRE if any(k in d for k in RECORD)]
    check(tree, 7, not bad, f"no record/label dir in the retire set ({bad or 'clear'})")

    # ---- INTERLOCK 8: never delete THROUGH a symlink
    thru = [d for d in RETIRE if os.path.islink(os.path.join(WORK, d))]
    check(tree, 8, not thru, f"no retire target is itself a symlink ({thru or 'clear'})")

    # ---- INTERLOCK 9: no arm a LIVE manifest resolves into may be released.
    man = set()
    for pat in ("em_display/*.tsv","em_display/*.txt","*/manifest*.tsv","docs/scan/*.tsv"):
        for f in __import__("glob").glob(os.path.join(cfg["root"], pat)):
            try: t = open(f, errors="ignore").read()
            except OSError: continue
            for mm in re.finditer(r"work-[A-Za-z0-9_.-]+|[0-9]{6}_[0-9]+_[A-Za-z0-9-]+", t):
                man.add(mm.group(0))
    resolved = {a for a in man if os.path.isdir(os.path.join(WORK, a))}
    inrel = sorted(resolved & set(RETIRE))
    check(tree, 9, not inrel,
          f"manifests name {len(man)} arms, {len(resolved)} exist, "
          f"{len(inrel)} in the release ({inrel or 'none'})")

    sz = du_kb([os.path.join(WORK, d) for d in RETIRE])
    tot = sum(sz.values())
    byarm = collections.Counter()
    for d in RETIRE: byarm[universe[d]] += sz.get(d, 0)
    print(f"\n  universe {len(universe)} dirs | KEEP {len(KEEP)} | RETIRE {len(RETIRE)}"
          f" = {tot/1048576:.2f} GiB | out-of-scope (untouched) {len(out_scope)}")
    print(f"  {'arm':<16}{'dirs':>6}{'GiB':>9}   cited?")
    for a, kb in byarm.most_common(40):
        n = sum(1 for d in RETIRE if universe[d] == a)
        print(f"  {a:<16}{n:>6}{kb/1048576:>9.2f}   {'yes' if CIT.get(a,0) else '-'}")
    tf = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tier_{tree}_{STAMP}.txt")
    with open(tf, "w") as fh:
        for d in RETIRE: fh.write(os.path.join(WORK, d) + "\n")
    print(f"  tier file: {tf}  ({len(RETIRE)} lines, {tot/1048576:.2f} GiB)")
    return tot

if __name__ == "__main__":
    want = sys.argv[1:] or list(TREES)
    grand = 0
    for t in want:
        grand += plan_tree(t, TREES[t])
    print(f"\n{'='*74}\nGRAND TOTAL staged: {grand/1048576:.2f} GiB")
    print(f"interlock failures: {fails or 'NONE'}")
    print("\nThis script retired nothing.  Review the tier files, then the owner runs the retire driver.")
    sys.exit(1 if fails else 0)
