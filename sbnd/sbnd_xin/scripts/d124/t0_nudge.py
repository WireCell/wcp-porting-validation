#!/usr/bin/env python3
"""doc sbnd_xin/124 -- the flash-time nudge: is a PR verdict sensitive to the ~10 ns by which the hit flash's
time differs from reco1's, with every cluster and point held fixed?

Copies one Q/L event dir (ql_evt<E>) of arm A into a scratch Q/L root, symlinking every file except the
pctree, which is rewritten member by member (same names, same order) with one array changed: the live
cluster_scalar 'cluster_t0' (ns).  Every cluster whose t0 equals (|d| < --tol ns) one of arm A's beam-window
flash times of a TPC is moved to arm B's beam-window flash time of that TPC.  The PR job rebuilds x_t0cor from
cluster_t0 (clustering_switch_scope -> add_corrected_points), so this moves the beam clusters rigidly in x by
dt * v_drift and changes nothing else (flash objects, cluster membership, ids and points are arm A's).

  --mode null   rewrite the pctree with no change (the repack null)
  --mode t0     move the beam clusters' t0 to arm B's beam flash time

The flash times come from the PR outputs of the two arms (T_flash in_window rows of tracking-pr.root).

usage: t0_nudge.py --ql <arm_A_ql_root> --pra <arm_A_pr_root> --prb <arm_B_pr_root> --out <scratch_ql_root>
                   --mode null|t0 [--tol 1.0] <evt> ...
"""
import argparse, io, json, os, re, sys, tarfile
import numpy as np, uproot


def beam_times(pr_dir):
    f = uproot.open(os.path.join(pr_dir, "tracking-pr.root"))
    t = f["T_flash"].arrays(["tpc", "time_us", "in_window"], library="np")
    out = {}
    for tpc, tu, w in zip(t["tpc"], t["time_us"], t["in_window"]):
        if int(w) == 1:
            out.setdefault(int(tpc), []).append(float(tu) * 1000.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ql", required=True); ap.add_argument("--pra", required=True); ap.add_argument("--prb", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--mode", choices=("null", "t0"), required=True)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("events", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rl = os.path.join(a.ql, ".lineage_reality")
    if os.path.exists(rl):
        open(os.path.join(a.out, ".lineage_reality"), "w").write(open(rl).read())
    for e in a.events:
        src = os.path.join(a.ql, "ql_evt%s" % e); dst = os.path.join(a.out, "ql_evt%s" % e)
        os.makedirs(dst, exist_ok=True)
        for fn in os.listdir(src):
            if fn == "pctree-evt%s.tar.gz" % e:
                continue
            p = os.path.join(dst, fn)
            if not os.path.lexists(p):
                os.symlink(os.path.realpath(os.path.join(src, fn)), p)
        ta = beam_times(os.path.join(a.pra, "pr_evt%s" % e)); tb = beam_times(os.path.join(a.prb, "pr_evt%s" % e))
        mapping = []
        for tpc, lst in ta.items():
            if tpc in tb and len(lst) == 1 and len(tb[tpc]) == 1:
                mapping.append((lst[0], tb[tpc][0], tpc))
        # the cluster_t0 member: its metadata datapath names it
        tin = tarfile.open(os.path.join(src, "pctree-evt%s.tar.gz" % e))
        members = tin.getmembers()
        target = None
        for m in members:
            if m.name.endswith("_metadata.json"):
                md = json.loads(tin.extractfile(m).read())
                if md.get("datapath", "").endswith("/live/pointclouds/namedpcs/cluster_scalar/arrays/cluster_t0"):
                    target = m.name.replace("_metadata.json", "_array.npy")
        if target is None:
            print("evt %s: no cluster_t0 array" % e, file=sys.stderr); sys.exit(1)
        out = tarfile.open(os.path.join(dst, "pctree-evt%s.tar.gz" % e), "w:gz")
        nmoved = 0
        for m in members:
            data = tin.extractfile(m).read() if m.isfile() else None
            if m.name == target and a.mode == "t0":
                arr = np.load(io.BytesIO(data))
                new = arr.copy()
                for fa, fb, tpc in mapping:
                    sel = np.abs(arr - fa) < a.tol
                    new[sel] = fb; nmoved += int(sel.sum())
                # keep the writer's own npy header (WCT's reader wants descr 'f8'; numpy writes '<f8')
                hl = 10 + int.from_bytes(data[8:10], "little") if data[6] == 1 else 12 + int.from_bytes(data[8:12], "little")
                assert len(data) - hl == new.nbytes
                data = data[:hl] + new.astype(arr.dtype).tobytes()
            ti = tarfile.TarInfo(m.name); ti.size = len(data) if data is not None else 0
            ti.mtime = m.mtime; ti.mode = m.mode; ti.type = m.type
            out.addfile(ti, io.BytesIO(data) if data is not None else None)
        out.close()
        print("evt %s mode=%s beam A->B %s  clusters moved %d" % (
            e, a.mode, " ".join("tpc%d %.1f->%.1f ns" % (t, x, y) for x, y, t in mapping), nmoved))


if __name__ == "__main__":
    main()
