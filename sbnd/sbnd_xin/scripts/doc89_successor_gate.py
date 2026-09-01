#!/usr/bin/env python3
"""doc 89 Phase 2 -- prove work-<s>-prod0901b supersedes work-<s>-prod0901.

The retire round releases prod0901 (10.2 G) ONLY on this evidence.  The claim
is deliberately narrow and cannot pass by accident:

    on every one of 3067 events, every product the two arms SHARE is
    identical, and the ONLY difference is the T_cluster tree that
    save_in_scope adds.

Why this is not a formality.  prod0901 ran at toolkit ddce7430 on a different
binary (pr142-libsnap: libWireCellClus.so 430ffa3e); prod0901b ran at d52d818c
(31b7e2ed).  Between them lie FIVE changes -- doc 77 r3, doc 77 r4, the master
merge, doc 87's knobs at defaults, and the save_in_scope flip -- each gated
byte-identical, but each only on the 308-event manifest.  This is the first
end-to-end check of the whole chain at full sample scale.

Per-product method, and why each is what it is:
  archives  member-CONTENT sha256, never the file bytes.  tar.gz and zip embed
            mtimes, so `cmp` on the container reports a regression that does
            not exist (M2).  mabc-pr.zip and pctree-pr-evt<N>.tar.gz.
  root      every tree present in BOTH files, branch by branch, equal_nan=True
            (T_rec_charge.reduced_chi2 carries NaNs and a naive != flags every
            NaN row).  Extra trees in B are reported, not failed -- T_cluster
            is expected and anything ELSE extra is a failure.
  calib     JSON compare EXCLUDING vertex_scoreboard.dual_chain.off_ms, a
            wall-clock timer that makes two identical dumps read as DIFFER.
  nusel     raw bytes; it is a plain TSV with no embedded time.

Usage:
    scripts/doc89_successor_gate.py [--jobs N] [--out <tsv>]
Exit 0 = every event OK.  1 = any mismatch (the first few are named).
"""
import argparse, hashlib, json, os, sys, tarfile, zipfile
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = [("nuecc48", 48), ("ncpi0", 19), ("mcp1k", 1000), ("mcp2k", 2000)]
TIMER_KEYS = {"off_ms", "on_ms", "elapsed_ms", "ms"}


def members(path):
    """(name -> sha256) of an archive's members, content only."""
    out = {}
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for n in sorted(z.namelist()):
                out[n] = hashlib.sha256(z.read(n)).hexdigest()
    else:
        with tarfile.open(path, "r:*") as t:
            for m in sorted(t.getmembers(), key=lambda x: x.name):
                if not m.isfile():
                    continue
                f = t.extractfile(m)
                h = hashlib.sha256()
                for c in iter(lambda: f.read(1 << 20), b""):
                    h.update(c)
                out[m.name] = h.hexdigest()
    return out


def strip_timers(o):
    """Drop wall-clock fields wherever they appear; they are not physics."""
    if isinstance(o, dict):
        return {k: strip_timers(v) for k, v in o.items() if k not in TIMER_KEYS}
    if isinstance(o, list):
        return [strip_timers(v) for v in o]
    return o


def cmp_root(pa, pb):
    import numpy as np, uproot
    def eq(x, y):
        a, b = np.asarray(x), np.asarray(y)
        if a.dtype == object or b.dtype == object:
            return len(a) == len(b) and all(eq(p, q) for p, q in zip(a, b))
        if a.shape != b.shape:
            return False
        if a.dtype.kind == "f" and b.dtype.kind == "f":
            return bool(np.array_equal(a, b, equal_nan=True))
        return bool(np.array_equal(a, b))
    with uproot.open(pa) as A, uproot.open(pb) as B:
        ka = {k.split(";")[0] for k in A.keys()}
        kb = {k.split(";")[0] for k in B.keys()}
        extra = sorted(kb - ka)
        if extra != ["T_cluster"]:
            return f"extra trees {extra}"
        if sorted(ka - kb):
            return f"trees LOST {sorted(ka - kb)}"
        for t in sorted(ka & kb):
            ta, tb = A[t], B[t]
            if set(ta.keys()) != set(tb.keys()):
                return f"{t}: branch list differs"
            aa, bb = ta.arrays(library="np"), tb.arrays(library="np")
            for br in sorted(aa):
                if not eq(aa[br], bb[br]):
                    return f"{t}.{br} differs"
    return ""


def one(job):
    sample, evt = job
    a = os.path.join(ROOT, f"work-{sample}-prod0901",  evt)
    b = os.path.join(ROOT, f"work-{sample}-prod0901b", evt)
    eid = evt.replace("pr_evt", "")
    res = {"root": "", "arch": "", "nusel": "", "calib": ""}
    try:
        for name in ("mabc-pr.zip", f"pctree-pr-evt{eid}.tar.gz"):
            fa, fb = os.path.join(a, name), os.path.join(b, name)
            if not (os.path.exists(fa) and os.path.exists(fb)):
                res["arch"] = f"{name} missing on one side"; break
            if members(fa) != members(fb):
                res["arch"] = f"{name} member content differs"; break
        na, nb = os.path.join(a, f"nusel-evt{eid}.tsv"), os.path.join(b, f"nusel-evt{eid}.tsv")
        if os.path.exists(na) != os.path.exists(nb):
            res["nusel"] = "present on one side only"
        elif os.path.exists(na) and open(na, "rb").read() != open(nb, "rb").read():
            res["nusel"] = "bytes differ"
        ca, cb = os.path.join(a, f"calib-pr-evt{eid}.json"), os.path.join(b, f"calib-pr-evt{eid}.json")
        if os.path.exists(ca) != os.path.exists(cb):
            res["calib"] = "present on one side only"
        elif os.path.exists(ca):
            if strip_timers(json.load(open(ca))) != strip_timers(json.load(open(cb))):
                res["calib"] = "differs (timers excluded)"
        ra, rb = os.path.join(a, "tracking-pr.root"), os.path.join(b, "tracking-pr.root")
        if not (os.path.exists(ra) and os.path.exists(rb)):
            res["root"] = "missing on one side"
        else:
            res["root"] = cmp_root(ra, rb)
    except Exception as e:                       # a crash is a FAIL, never a skip
        res["root"] = res["root"] or f"EXCEPTION {type(e).__name__}: {e}"
    bad = [f"{k}:{v}" for k, v in res.items() if v]
    return (sample, eid, res["root"] or "-", res["arch"] or "-",
            res["nusel"] or "-", res["calib"] or "-",
            "OK" if not bad else "; ".join(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "retire",
                                                  "state-20260901", "successor-gate.tsv"))
    a = ap.parse_args()
    jobs, want = [], 0
    for s, n in PAIRS:
        old = os.path.join(ROOT, f"work-{s}-prod0901")
        new = os.path.join(ROOT, f"work-{s}-prod0901b")
        for d in (old, new):
            if not os.path.isdir(d):
                sys.exit(f"REFUSING: {d} does not exist")
        evts = sorted(x for x in os.listdir(new) if x.startswith("pr_evt"))
        if len(evts) != n:
            sys.exit(f"REFUSING: {new} has {len(evts)} pr_evt dirs, expected {n}")
        want += n
        jobs += [(s, e) for e in evts]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=4), 1):
            rows.append(r)
            if i % 250 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    ok = sum(1 for r in rows if r[-1] == "OK")
    with open(a.out, "w") as fh:
        fh.write("# doc 89 Phase 2 -- prod0901 -> prod0901b successor gate\n")
        fh.write("# shared products identical; T_cluster the only added tree\n")
        fh.write("sample\tevent\troot\tarchives\tnusel\tcalib\tverdict\n")
        for r in sorted(rows):
            fh.write("\t".join(r) + "\n")
    print(f"\n=== SUCCESSOR GATE ({want} events) ===")
    if ok == want:
        print(f"PASS -- {ok}/{want} events: every shared product identical, "
              f"T_cluster the only addition")
    else:
        print(f"FAIL -- {ok}/{want} OK; first mismatches:")
        for r in sorted(rows):
            if r[-1] != "OK":
                print(f"  {r[0]} evt{r[1]}: {r[-1]}")
    print(f"written: {a.out}")
    return 0 if ok == want else 1


if __name__ == "__main__":
    sys.exit(main())
