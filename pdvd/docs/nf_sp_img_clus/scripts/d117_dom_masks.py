#!/usr/bin/env python3
"""doc pdvd/117 round 5 (the Michel-domain test), step 1 -- each data Michel's keep and muon masks in crop coordinates.

    python3 d117_dom_masks.py -j 8 > ../../scan/d117/dom/mask_closure.txt   # -> /home/xqian/tmp/d117/dom/masks_pdvd.npz

d98_michel_crops.build_one() is run unchanged on the latest arm (p98vonq labels/masks, d117sp frames through
d117_crops.sp_dir, the sw99 record), with C.dilate wrapped so the three masks it builds (keep, muon, keep_w, in that
order) are captured inside the worker; keep and muon are cut with make_crops.cut at the crop's own centre.  C.main() is
NOT called (it writes candidates.tsv / crops.tsv).  Gate: frame * (keep & ~muon), cut from the captured masks, equals
the stored Z of /home/xqian/tmp/d117/latest/crops_pdvd.npz exactly, for every crop.
"""
import argparse, multiprocessing as mp, os, sys
import numpy as np
import d117_crops                                  # noqa: F401  sp_dir -> work/<evt>_d117sp
import d98_michel_crops as C

C.ARM["pdvd"] = "p98vonq"
C.m.VD_REC = C.IMG + "/pdvd/docs/scan/pdvd_stm_michel_p98vonq_carried_sw99_verdicts.json"
OUT = "/home/xqian/tmp/d117/dom"
REF = "/home/xqian/tmp/d117/latest/crops_pdvd.npz"
_dilate = C.dilate
_stash = []


def _capture(mask, dch, dtk):
    out = _dilate(mask, dch, dtk)
    _stash.append(out)
    return out


C.dilate = _capture


def worker(args):
    _stash.clear()
    res = C.build_one(args)
    if res is None:
        return None
    info, crops = res
    assert len(_stash) == 3, len(_stash)
    keep, muon = _stash[0], _stash[1]
    evt = info["key"].split("/")[0]
    chans, frame, tbin = C.load_gauss(os.path.join(C.sp_dir("pdvd", evt), C.FRAME_PFX["pdvd"] % info["apa"]))
    ck = C.make_crops.cut(chans, keep.astype(np.float32), tbin, info["c_center"], info["t_center"])[0] > 0.5
    cm = C.make_crops.cut(chans, muon.astype(np.float32), tbin, info["c_center"], info["t_center"])[0] > 0.5
    return info["key"], ck, cm, crops["Z"], crops["M"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=8)
    a = ap.parse_args()
    todo = [("pdvd", r) for r in C.candidates("pdvd") if r["selected"]]
    with mp.Pool(a.j) as pool:
        res = [r for r in pool.map(worker, todo, chunksize=1) if r is not None]
    ref = np.load(REF)
    rk = {str(k): i for i, k in enumerate(ref["keys"])}
    print(f"# doc pdvd/117 round 5 mask capture (d117_dom_masks.py): {len(todo)} selected, {len(res)} built; "
          f"reference {REF} ({len(rk)} crops)")
    nbad, keys, K, Mu = 0, [], [], []
    for key, ck, cm, Z, M in res:
        i = rk.get(key)
        if i is None:
            print(f"  {key}: not in the reference"); nbad += 1; continue
        # M = frame * keep; the gate rebuilds Z from the stored M and the captured masks
        zr = ref["M"][i] * (ck & ~cm)
        ok = np.array_equal(zr, ref["Z"][i]) and np.array_equal(Z, ref["Z"][i]) and np.array_equal(M * ck, ref["M"][i])
        nbad += not ok
        if not ok:
            print(f"  {key}: MISMATCH max|dZ| {np.abs(zr - ref['Z'][i]).max():.4g}")
        keys.append(key); K.append(ck); Mu.append(cm)
    print(f"gate: M * (keep & ~muon) == stored Z on {len(keys) - nbad}/{len(rk)} crops: {'PASS' if nbad == 0 and len(keys) == len(rk) else 'FAIL'}")
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/masks_pdvd.npz", keys=np.array(keys), keep=np.stack(K), muon=np.stack(Mu))
    print(f"wrote {OUT}/masks_pdvd.npz")
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
