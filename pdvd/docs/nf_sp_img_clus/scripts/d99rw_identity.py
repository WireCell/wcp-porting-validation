#!/usr/bin/env python3
"""doc pdvd/99 sec 4.4 -- is a readout-window arm identical to its source where the window did not change?

    python3 d99rw_identity.py --arm p99rgon --base p98vonq [--nt 10000|6400|all] [--clus-base d51vclus] [--jobs 24]

Per event of pdvd/stm/events.txt whose SP frame length (d99/frame_nticks_120evt.txt) matches --nt:
  pctree  sha256 of every member of pctree-evt*.tar.gz (member content, not the tarball: M2)   [--clus-base for the
          production arm, whose p96vprod dir links d51vclus's pctree; default = --base]
  tlas    line-by-line, the opflash_input path spelling set aside and the readout_window_ticks line reported apart
  pr      every TTree branch of tracking-pr.root and tracking-stm.root, NaN-aware
  mabc    every member of mabc-pr.zip, content
An event reads IDENTICAL only if all four are.
"""
import argparse, glob, hashlib, math, os, sys, tarfile, zipfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import uproot
import d99_control_compare as C

PDVD = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd"
NT = PDVD + "/docs/nf_sp_img_clus/d99/frame_nticks_120evt.txt"


def one_file(d, pat):
    f = sorted(glob.glob(os.path.join(d, pat)))
    return f[0] if f else None


def tar_members(path):
    out = {}
    with tarfile.open(path) as tf:
        for m in tf:
            if m.isfile():
                out[m.name] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
    return out


def zip_members(path):
    with zipfile.ZipFile(path) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}


def same_value(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype == object or b.dtype == object:
            return all(same_value(x, y) for x, y in zip(a.ravel(), b.ravel()))
        if a.dtype.kind in "fc":
            return bool(np.array_equal(a, b, equal_nan=True))
        return bool(np.array_equal(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def root_diff(pa, pb):
    """-> (n branches compared, list of differing tree/branch names)"""
    fa, fb = uproot.open(pa), uproot.open(pb)
    ta = {k.split(";")[0] for k, c in fa.classnames().items() if c == "TTree"}
    tb = {k.split(";")[0] for k, c in fb.classnames().items() if c == "TTree"}
    bad = [f"tree-set {sorted(ta ^ tb)}"] if ta != tb else []
    n = 0
    for t in sorted(ta & tb):
        A, B = fa[t], fb[t]
        if set(A.keys()) != set(B.keys()):
            bad.append(f"{t}: branch-set"); continue
        for br in A.keys():
            n += 1
            try:   # awkward + d99_control_compare.arr_equal: nested vectors, NaN pattern equal, finite values bit-equal
                if not C.arr_equal(A[br].array(library="ak"), B[br].array(library="ak")):
                    bad.append(f"{t}/{br}")
            except Exception as ex:          # an unreadable branch is reported, not skipped
                bad.append(f"{t}/{br} unreadable {type(ex).__name__}")
    return n, bad


def check(job):
    evt, arm, base, clus_base = job
    da, db, dc = (f"{PDVD}/work/{evt}_{x}" for x in (arm, base, clus_base))
    res = {}
    pa, pc = one_file(da, "pctree-evt*.tar.gz"), one_file(dc, "pctree-evt*.tar.gz")
    res["pctree"] = "missing" if not (pa and pc) else ("identical" if tar_members(pa) == tar_members(pc) else "DIFFERS")
    ta, tc = one_file(da, "pctree-evt*.tlas"), one_file(dc, "pctree-evt*.tlas")
    if ta and tc:
        la = [l for l in open(ta).read().splitlines() if not l.startswith("opflash_input=")]
        lc = [l for l in open(tc).read().splitlines() if not l.startswith("opflash_input=")]
        win = [l for l in la if l.startswith("readout_window_ticks=")]
        rest = [l for l in la if not l.startswith("readout_window_ticks=")] == [l for l in lc if not l.startswith("readout_window_ticks=")]
        res["tlas"] = ("identical" if la == lc else ("window only" if rest else "DIFFERS")) + f" ({win[0] if win else 'no window'})"
    else:
        res["tlas"] = "missing"
    nbr, bad = 0, []
    for fn in ("tracking-pr.root", "tracking-stm.root"):
        a, b = os.path.join(da, fn), os.path.join(db, fn)
        if not (os.path.exists(a) and os.path.exists(b)):
            bad.append(f"{fn} missing"); continue
        n, bb = root_diff(a, b); nbr += n; bad += [f"{fn}:{x}" for x in bb]
    res["pr"] = f"identical ({nbr} branches)" if not bad else f"DIFFERS {len(bad)}/{nbr} ({', '.join(bad[:4])}{' ...' if len(bad) > 4 else ''})"
    za, zb = os.path.join(da, "mabc-pr.zip"), os.path.join(db, "mabc-pr.zip")
    if os.path.exists(za) and os.path.exists(zb):
        ma, mb = zip_members(za), zip_members(zb)
        same = sum(ma.get(k) == v for k, v in mb.items())
        res["mabc"] = ("identical" if ma == mb else "DIFFERS") + f" {same}/{len(mb)} vs {len(ma)}"
    else:
        res["mabc"] = "missing"
    ok = all(v.startswith("identical") for k, v in res.items() if k != "tlas") and res["tlas"].startswith(("identical", "window only"))
    return evt, ok, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--clus-base", default=None)
    ap.add_argument("--nt", default="10000")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--events", nargs="*", default=None, help="restrict to these events (e.g. the knob-unset control)")
    a = ap.parse_args()
    nt = {l.split()[0]: l.split()[1] for l in open(NT) if l.strip() and not l.startswith("#")}
    evs = [e for e in sorted(nt) if (a.nt == "all" or nt[e] == a.nt) and (a.events is None or e in a.events)]
    cb = a.clus_base or a.base
    print(f"# doc pdvd/99 sec 4.4 -- {a.arm} against {a.base} (pctree/tlas against {cb}) on the {len(evs)} events with "
          f"frame length {a.nt}; d99rw_identity.py")
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(check, [(e, a.arm, a.base, cb) for e in evs]))
    for evt, ok, r in rows:
        print(f"{evt}: {'IDENTICAL' if ok else 'differs  '} | pctree {r['pctree']} | tlas {r['tlas']} | pr {r['pr']} | mabc-pr {r['mabc']}")
    n_ok = sum(ok for _, ok, _ in rows)
    print(f"SUMMARY {a.arm} vs {a.base}, frame length {a.nt}: identical {n_ok}/{len(rows)}")


if __name__ == "__main__":
    sys.exit(main())
