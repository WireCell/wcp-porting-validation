#!/usr/bin/env python3
"""doc pdvd/99 -- content-hash manifest of the SP frame archives of both arms, and gate G3.

    python3 d99_manifest.py --out-dir pdvd/docs/nf_sp_img_clus/d99

Writes  frames_manifest_p98voff.txt / frames_manifest_p98von.txt  (one line per archive:
        event anode sha256-rollup n-members bytes, rollup = abtest/hash_archive.py's member-content hash)
        g3_bottom_identity.txt  (per event: bottom anodes 0-3 identical OFF vs ON?  top 4-7 differ?)
The manifests are the record kept when the OFF frames are trimmed (owner 2026-09-13: trim after
imaging) and the proof that the kept ON frames are the ones every downstream number used.
G3 PASS = every event of pdvd/stm/events.txt has 8 archives in both arms, bottom identical on all
of them, top different on all of them.
"""
import argparse, collections, hashlib, os, sys, tarfile
from concurrent.futures import ProcessPoolExecutor

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
WORK = IMG + "/pdvd/work"
sys.path.insert(0, IMG + "/abtest")
import hash_archive as HA


def events():
    out = []
    for l in open(IMG + "/pdvd/stm/events.txt"):
        if l.startswith("#") or len(l.split()) < 2:
            continue
        run, idx = l.split()[:2]
        out.append("%06d_%s" % (int(run), idx))
    return out


def one(job):
    arm, evt, an = job
    p = f"{WORK}/{evt}_{arm}/protodune-sp-dnnroi-frames-anode{an}.tar.bz2"
    if not os.path.exists(p):
        return arm, evt, an, None, 0, 0
    h = hashlib.sha256(); n = 0
    for name, data in HA.members(p):
        h.update(name.encode()); h.update(data); n += 1
    return arm, evt, an, h.hexdigest(), n, os.path.getsize(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", nargs=2, default=["p98voff", "p98von"])
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    E = events()
    jobs = [(arm, e, an) for arm in a.arms for e in E for an in range(8)]
    R = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for arm, e, an, h, n, b in ex.map(one, jobs, chunksize=4):
            R[(arm, e, an)] = (h, n, b)
    for arm in a.arms:
        with open(f"{a.out_dir}/frames_manifest_{arm}.txt", "w") as f:
            f.write(f"# doc pdvd/99 SP frame archives of arm {arm}: event anode sha256(member names+payloads, sorted) n_members bytes\n")
            for e in E:
                for an in range(8):
                    h, n, b = R[(arm, e, an)]
                    f.write(f"{e} {an} {h or 'MISSING'} {n} {b}\n")
    off, on = a.arms
    cnt = collections.Counter(); bad = []
    with open(f"{a.out_dir}/g3_bottom_identity.txt", "w") as f:
        f.write(f"# doc pdvd/99 gate G3: {off} vs {on}; bottom = anodes 0-3 must be identical, top = 4-7 must differ\n")
        for e in E:
            miss = [an for an in range(8) for arm in a.arms if R[(arm, e, an)][0] is None]
            bot = all(R[(off, e, an)][0] == R[(on, e, an)][0] for an in range(4))
            top = all(R[(off, e, an)][0] != R[(on, e, an)][0] for an in range(4, 8))
            ok = not miss and bot and top
            cnt["pass" if ok else "fail"] += 1
            if not ok:
                bad.append(e)
            f.write(f"{e} complete={'no' if miss else 'yes'} bottom_identical={bot} top_differ={top} {'PASS' if ok else 'FAIL'}\n")
        f.write(f"# G3 {'PASS' if not bad else 'FAIL'}: {cnt['pass']}/{len(E)} events; failing {bad}\n")
    tot = {arm: sum(R[(arm, e, an)][2] for e in E for an in range(8)) for arm in a.arms}
    print(f"G3 {'PASS' if not bad else 'FAIL'}: {cnt['pass']}/{len(E)} events pass; failing {bad[:10]}")
    print("bytes " + "  ".join(f"{arm} {tot[arm] / 2**30:.2f} GiB" for arm in a.arms))


if __name__ == "__main__":
    main()
