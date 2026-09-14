#!/usr/bin/env python3
"""doc pdvd/100 round 2, gate X3 -- is a PDHD PR arm identical to another on every event they share?

    python3 d100_pdhd_identity.py --arm h100b --base h100a [--jobs 24]

The same PR-stage comparison as d99rw_identity.py (imported, not copied): every TTree branch of tracking-pr.root and
tracking-stm.root (awkward, NaN pattern equal, finite values bit-equal) and every member of mabc-pr.zip by content.  The
pctree is a symlink to the same source file on both arms (d53_run_arms.sh stages it that way), so it is checked by real
path.  An event reads IDENTICAL only if all of them are; an event on one side only is reported, never skipped.
"""
import argparse, glob, os, sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d99rw_identity as I

PDHD = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd"


def check(job):
    evt, arm, base = job
    da, db = f"{PDHD}/work/{evt}_{arm}", f"{PDHD}/work/{evt}_{base}"
    res = {}
    pa, pb = I.one_file(da, "pctree-evt*.tar.gz"), I.one_file(db, "pctree-evt*.tar.gz")
    res["pctree"] = "missing" if not (pa and pb) else ("identical (same file)" if os.path.realpath(pa) == os.path.realpath(pb)
                                                         else ("identical" if I.tar_members(pa) == I.tar_members(pb) else "DIFFERS"))
    nbr, bad = 0, []
    for fn in ("tracking-pr.root", "tracking-stm.root"):
        a, b = os.path.join(da, fn), os.path.join(db, fn)
        if not os.path.exists(a) and not os.path.exists(b):
            continue
        if not (os.path.exists(a) and os.path.exists(b)):
            bad.append(f"{fn} on one side only"); continue
        n, bb = I.root_diff(a, b); nbr += n; bad += [f"{fn}:{x}" for x in bb]
    if nbr == 0 and not bad:
        bad.append("no PR output on either side")
    res["pr"] = f"identical ({nbr} branches)" if not bad else f"DIFFERS {len(bad)}/{nbr} ({', '.join(bad[:4])}{' ...' if len(bad) > 4 else ''})"
    za, zb = os.path.join(da, "mabc-pr.zip"), os.path.join(db, "mabc-pr.zip")
    if os.path.exists(za) and os.path.exists(zb):
        ma, mb = I.zip_members(za), I.zip_members(zb)
        res["mabc"] = ("identical" if ma == mb else "DIFFERS") + f" {sum(ma.get(k) == v for k, v in mb.items())}/{len(mb)} vs {len(ma)}"
    else:
        res["mabc"] = "missing"
    return evt, all(v.startswith("identical") for v in res.values()), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    ev = lambda arm: {os.path.basename(d)[: -len(arm) - 1] for d in glob.glob(f"{PDHD}/work/*_{arm}")}
    ea, eb = ev(a.arm), ev(a.base)
    print(f"# doc pdvd/100 round 2 gate X3 -- PDHD {a.arm} ({len(ea)} dirs) against {a.base} ({len(eb)} dirs); d100_pdhd_identity.py")
    for e in sorted(ea ^ eb):
        print(f"{e}: ONLY IN {a.arm if e in ea else a.base}")
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(check, [(e, a.arm, a.base) for e in sorted(ea & eb)]))
    for evt, ok, r in rows:
        print(f"{evt}: {'IDENTICAL' if ok else 'differs  '} | pctree {r['pctree']} | pr {r['pr']} | mabc-pr {r['mabc']}")
    n_ok = sum(ok for _, ok, _ in rows)
    print(f"SUMMARY PDHD {a.arm} vs {a.base}: identical {n_ok}/{len(rows)}; events on one side only {len(ea ^ eb)}")
    return 0 if (n_ok == len(rows) and not (ea ^ eb) and rows) else 1


if __name__ == "__main__":
    sys.exit(main())
