#!/usr/bin/env python3
"""doc pdvd/99 -- reproduction controls: compare two arms event by event, content only.

    python3 d99_control_compare.py --a p98kq --b-clus d51vclus --b-pr p96vprod --events 039349_81 039252_0 ...

Per event:
  pctree   pctree-evt*.tar.gz of A vs B-clus, abtest/hash_archive.py member-content hash (never raw bytes, M2)
  tlas     pctree-evt*.tlas of A vs B-clus, line by line (opflash path spelling and `img_wires` excluded from the verdict,
           reported)
  pr       every branch of every tree in tracking-pr.root and tracking-stm.root of A vs B-pr, NaN-aware
           (NaN pattern equal and finite values bit-equal; an ak.to_list compare calls NaN != NaN a difference)
  mabc     mabc-pr.zip members, sha256 of payloads
"""
import argparse, glob, hashlib, os, sys, zipfile
import numpy as np
import awkward as ak
import uproot

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
W = IMG + "/pdvd/work"
sys.path.insert(0, IMG + "/abtest")
import hash_archive as HA


def rollup(p):
    h = hashlib.sha256()
    for name, data in HA.members(p):
        h.update(name.encode()); h.update(data)
    return h.hexdigest()


def one(path):
    g = sorted(glob.glob(path))
    return g[0] if g else None


def arr_equal(x, y):
    try:
        fx = ak.to_numpy(ak.flatten(x, axis=None)); fy = ak.to_numpy(ak.flatten(y, axis=None))
    except Exception:
        return ak.to_list(x) == ak.to_list(y)
    if ak.to_list(ak.num(x, axis=0)) != ak.to_list(ak.num(y, axis=0)) or fx.shape != fy.shape:
        return False
    if x.ndim > 1 and ak.to_list(ak.num(x, axis=1)) != ak.to_list(ak.num(y, axis=1)):
        return False
    if fx.dtype.kind == "f":
        nx, ny = np.isnan(fx), np.isnan(fy)
        return bool(np.array_equal(nx, ny) and np.array_equal(fx[~nx], fy[~ny]))
    return bool(np.array_equal(fx, fy))


def trees(a, b):
    out, nb, nd = [], 0, 0
    if not (a and b and os.path.exists(a) and os.path.exists(b)):
        return f"missing ({bool(a and os.path.exists(a))}/{bool(b and os.path.exists(b))})", 0, 0
    A, B = uproot.open(a), uproot.open(b)
    ka = sorted(k.split(";")[0] for k in A.keys()); kb = sorted(k.split(";")[0] for k in B.keys())
    if ka != kb:
        return f"tree sets differ {sorted(set(ka) ^ set(kb))}", 0, 1
    for t in ka:
        if not hasattr(A[t], "keys"):
            continue
        if sorted(A[t].keys()) != sorted(B[t].keys()):
            out.append(f"{t}: branch sets differ"); nd += 1; continue
        for br in A[t].keys():
            nb += 1
            if not arr_equal(A[t][br].array(library="ak"), B[t][br].array(library="ak")):
                nd += 1; out.append(f"{t}.{br}")
    return ("identical" if nd == 0 else "DIFF: " + ", ".join(out[:8])), nb, nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b-clus", default="d51vclus")
    ap.add_argument("--b-pr", default="p96vprod")
    ap.add_argument("--events", nargs="+", required=True)
    a = ap.parse_args()
    tot = {"pctree": 0, "tlas": 0, "pr": 0, "mabc": 0}
    for e in a.events:
        A, BC, BP = f"{W}/{e}_{a.a}", f"{W}/{e}_{a.b_clus}", f"{W}/{e}_{a.b_pr}"
        pa, pb = one(A + "/pctree-evt*.tar.gz"), one(BC + "/pctree-evt*.tar.gz")
        pct = (pa and pb and rollup(pa) == rollup(pb))
        ta, tb = one(A + "/pctree-evt*.tlas"), one(BC + "/pctree-evt*.tlas")
        la = open(ta).read().split() if ta else []; lb = open(tb).read().split() if tb else []
        dl = sorted(set(la) ^ set(lb))
        tl_ok = all(x.startswith(("opflash_input=", "img_wires=")) for x in dl)
        pr_txt, nb, nd = trees(one(A + "/tracking-pr.root"), one(BP + "/tracking-pr.root"))
        st_txt, nb2, nd2 = trees(one(A + "/tracking-stm.root"), one(BP + "/tracking-stm.root"))
        za, zb = one(A + "/mabc-pr.zip"), one(BP + "/mabc-pr.zip")
        if za and zb:
            ma = {n: hashlib.sha256(zipfile.ZipFile(za).read(n)).hexdigest() for n in zipfile.ZipFile(za).namelist()}
            mb = {n: hashlib.sha256(zipfile.ZipFile(zb).read(n)).hexdigest() for n in zipfile.ZipFile(zb).namelist()}
            mz = f"{sum(1 for k in ma if mb.get(k) == ma[k])}/{len(ma)} vs {len(mb)}"
            mz_ok = ma == mb
        else:
            mz, mz_ok = "missing", False
        tot["pctree"] += bool(pct); tot["tlas"] += tl_ok; tot["pr"] += (nd == 0 and nd2 == 0 and nb > 0); tot["mabc"] += mz_ok
        print(f"{e}: pctree {'identical' if pct else 'DIFF'} | tlas {'identical' if not dl else ('identical except ' if tl_ok else 'DIFF ') + str(dl)} | "
              f"tracking-pr {pr_txt} ({nb} branches) | tracking-stm {st_txt} ({nb2}) | mabc-pr {mz}")
    n = len(a.events)
    print(f"SUMMARY {a.a} vs {a.b_clus}/{a.b_pr}: pctree {tot['pctree']}/{n}, tlas {tot['tlas']}/{n}, PR trees {tot['pr']}/{n}, mabc-pr {tot['mabc']}/{n}")


if __name__ == "__main__":
    main()
