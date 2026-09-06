#!/usr/bin/env python3
"""doc pdhd/02 -- PDHD fork (BY DUPLICATION) of pdvd/docs/nf_sp_img_clus/scripts/d42_proj2d_selfcheck.py;
the PDVD script is untouched.  One change: a "pdhd" entry in PLANES with the CUMULATIVE per-plane
channel counts (3200 U, 3200 V, 3840 W) = the PdvdMagnifyTrackingVisitor ChanScheme thresholds
(see d44_sigma_fit.py header).

Original header follows.

doc pdvd/42 -- positive control for the STM T_proj_data dump.

For every (cluster, pass) block of one tracking-stm.root, count the T_proj_data
cells that lie on the block's own trajectory footprint (within 1 channel x 1
slice of a T_rec_charge point) and report the fraction of those cells that carry
charge_pred > 0.  Before the doc-42 fix only the LAST fitted block of an event
had predictions (PDVD 039252/2: 6923 of 177857 cells); after it every block
whose fit converged must predict on its own footprint.

Usage: d42_proj2d_selfcheck.py <tracking-stm.root> [--det pdvd|sbnd] [--min-frac 0.9]
Exit 0 when every block with >= 10 footprint cells and status in {0,2,3,4,5,7}
passes; 1 otherwise.  Also prints the total predicted-cell count and a CONTENT
sha256 of T_proj_data (sorted (block, channel, slice, charge, err, pred)
tuples) -- qlport/scripts/hash_root_trees.py cannot see vector<vector<int>>
branches and reports two different T_proj_data trees as SAME (verified on
039252/2 d42fitold vs d42fitnew: 155 vs 16 blocks, both "SAME").
"""
import argparse, hashlib, sys
import numpy as np
import uproot

PLANES = {"pdvd": (3808, 7616), "sbnd": (3968, 7936), "pdhd": (3200, 6400)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--det", default="pdvd", choices=PLANES)
    ap.add_argument("--min-frac", type=float, default=0.5)
    a = ap.parse_args()
    f = uproot.open(a.root)
    if "T_proj_data" not in {k.split(";")[0] for k in f.keys()}:
        print("%s: no T_proj_data (no STM fit in this event) -- nothing to check" % a.root); return 0
    r = f["T_rec_charge"].arrays(["pu", "pv", "pw", "pt", "ndf", "status"], library="np")
    d = f["T_proj_data"].arrays(library="np")
    blocks = [int(c) for c in d["cluster_id"][0]]
    bounds = PLANES[a.det]
    ncell = npred = 0
    bad = []
    rows = []
    hsh = hashlib.sha256()
    for i, blk in enumerate(blocks):
        ch = np.asarray(list(d["channel"][0][i]), dtype=np.int64)
        ts = np.asarray(list(d["time_slice"][0][i]), dtype=np.int64)
        qp = np.asarray(list(d["charge_pred"][0][i]), dtype=float)
        q_ = np.asarray(list(d["charge"][0][i]), dtype=np.int64)
        qe = np.asarray(list(d["charge_err"][0][i]), dtype=np.int64)
        order = np.lexsort((ts, ch))
        hsh.update(np.column_stack([np.full(len(ch), blk), ch[order], ts[order], q_[order], qe[order], qp[order].astype(np.int64)]).tobytes())
        ncell += len(ch); npred += int((qp > 0).sum())
        m = r["ndf"] == blk
        if m.sum() == 0:
            rows.append((blk, -1, 0, 0, float("nan"))); continue
        st = int(r["status"][m][0])
        pl = np.digitize(ch, bounds)
        near = np.zeros(len(ch), bool)
        for P, key in enumerate(("pu", "pv", "pw")):
            pc = np.round(r[key][m]).astype(np.int64); pt = np.round(r["pt"][m]).astype(np.int64)
            idx = np.where(pl == P)[0]
            if len(idx) == 0: continue
            # vectorised 1x1 neighbourhood test
            dc = np.abs(ch[idx][:, None] - pc[None, :]) <= 1
            dt = np.abs(ts[idx][:, None] - pt[None, :]) <= 1
            near[idx] = (dc & dt).any(axis=1)
        nfoot = int(near.sum()); frac = float((qp[near] > 0).mean()) if nfoot else float("nan")
        rows.append((blk, st, len(ch), nfoot, frac))
        if st in (0, 2, 3, 4, 5, 7) and nfoot >= 10 and not (frac >= a.min_frac):
            bad.append((blk, st, nfoot, frac))
    print("%s: blocks=%d cells=%d cells_pred>0=%d (%.1f%%) T_proj_data_sha256=%s" % (a.root, len(blocks), ncell, npred, 100.0 * npred / max(ncell, 1), hsh.hexdigest()[:16]))
    print("block status ncells nfoot pred_frac_on_footprint")
    for blk, st, n, nf, fr in rows:
        print("%6d %3d %7d %6d %s" % (blk, st, n, nf, "nan" if np.isnan(fr) else "%.3f" % fr))
    if bad:
        print("FAIL: %d block(s) below %.2f on their own footprint: %s" % (len(bad), a.min_frac, bad))
        return 1
    print("PASS: every fitted block predicts on its own footprint (>= %.2f)" % a.min_frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
