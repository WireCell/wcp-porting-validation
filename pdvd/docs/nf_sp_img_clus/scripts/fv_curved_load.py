#!/usr/bin/env python3
"""Doc pdvd/41 -- stage 1: cache the t0-corrected imaged points of the 120-event
PDVD cosmic arm, with the per-point RAW readout x (for the readout-window mask)
and the per-cluster flash t0, into one npz.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_curved_load.py work/*_d28dlfp \
      --out /home/xqian/tmp/doc41/points_d28dlfp.npz --jobs 16

Readers are duplicated from stm/end_reach.py (bee_points) and
docs/nf_sp_img_clus/scripts/goodpoint_pitch_census.py (load_pct), per the local
"duplicate, don't import" convention.

Frames (feedback: pctree-dump offline traps):
  * Bee clustering-global x IS x_t0cor (cm); clusters with no flash t0 carry the
    sentinel x = +-1.48e8 cm -> cut |x| < 1e4 AND join T_cluster.cluster_t0_us > -1e5.
  * pctree 3d/x is the RAW readout x (mm), same point order as the Bee layer
    (asserted here: |bee.x - x_t0cor/10| == 0 on every physical point).
"""
import argparse, io, json, os, re, sys, tarfile, zipfile
from multiprocessing import Pool
import numpy as np
import uproot
from scipy.spatial import cKDTree

SENT = 1e4


def bee_points(workdir):
    z = zipfile.ZipFile(os.path.join(workdir, "mabc-pr.zip"))
    for n in z.namelist():
        if n.endswith("-clustering-global.json"):
            d = json.loads(z.read(n))
            return (np.column_stack([d["x"], d["y"], d["z"]]).astype(float),
                    np.asarray(d["q"], float), np.asarray(d["cluster_id"], int),
                    int(d["runNo"]), int(d["eventNo"]))
    raise RuntimeError("no clustering-global layer in " + workdir)


def load_pct(tgz, want):
    out = {}
    with tarfile.open(tgz, "r:gz") as tf:
        members = {m.name: m for m in tf.getmembers()}
        for name, m in members.items():
            if not name.endswith("_metadata.json"):
                continue
            md = json.load(io.BytesIO(tf.extractfile(m).read()))
            if md.get("datatype") != "pcarray":
                continue
            dp = md.get("datapath", "")
            if not want(dp):
                continue
            an = name.replace("_metadata.json", "_array.npy")
            if an in members:
                out[dp] = np.load(io.BytesIO(tf.extractfile(members[an]).read()))
    return out


def read_tlas(path):
    kv = {}
    for line in open(path):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            kv[k] = v
    return kv


def one(workdir):
    P, Q, C, run, evt = bee_points(workdir)
    T = uproot.open(os.path.join(workdir, "tracking-pr.root"))["T_cluster"].arrays(library="np")
    t0 = dict(zip(T["cluster_id"].astype(int), T["cluster_t0_us"].astype(float)))
    npts = dict(zip(T["cluster_id"].astype(int), T["npoints"].astype(int)))
    tgz = [f for f in os.listdir(workdir) if re.match(r"pctree-evt\d+\.tar\.gz$", f)]
    assert len(tgz) == 1, workdir
    tgz = os.path.join(workdir, tgz[0])
    d = load_pct(tgz, lambda p: "/namedpcs/3d/arrays/" in p
                 and p.rsplit("/", 1)[-1] in ("x", "x_t0cor", "y", "z", "wpid"))
    g = lambda n: [v for k, v in d.items() if k.endswith("/arrays/" + n)][0]
    xraw, xcor, py, pz = g("x") / 10.0, g("x_t0cor") / 10.0, g("y") / 10.0, g("z") / 10.0
    apa = g("wpid") >> 4                       # iface/src/WirePlaneId.cxx:36
    pside = np.where(apa < 4, -1, 1).astype(np.int8)   # anodes 0-3 bottom drift (x<0), 4-7 top
    assert len(xraw) == len(P), (workdir, len(xraw), len(P))
    phys = np.abs(P[:, 0]) < SENT
    # The Bee layer of an older arm (d28dlfp) is a re-partition of the same points in a
    # different ORDER (29254 of 65532 differ on 039252/0), so match geometrically:
    # same (x_t0cor, y, z) to 0.01 cm, one-to-one.
    pp = np.abs(xcor) < SENT
    tree = cKDTree(np.column_stack([xcor[pp], py[pp], pz[pp]]))
    dist, idx = tree.query(P[phys], k=1)
    # A cluster the PR stage re-timed (039252/4 cluster 126: 558 points, all 0.117 cm off
    # in x) has no partner at 0.01 cm; accept the same (y, z) within half a drift step in x.
    miss = dist > 0.01
    nret0 = 0
    if miss.any():
        Pm = P[phys][miss]
        dyz = np.hypot(py[pp][idx[miss]] - Pm[:, 1], pz[pp][idx[miss]] - Pm[:, 2])
        dx = np.abs(xcor[pp][idx[miss]] - Pm[:, 0])
        assert ((dyz < 0.01) & (dx < 0.15)).all(), (workdir, int(miss.sum()), float(dyz.max()), float(dx.max()))
        nret0 = int(miss.sum())
    ndup = int(phys.sum()) - len(np.unique(idx))   # exact coordinate duplicates (18 on 039252/0)
    xr = np.full(len(P), np.nan); xr[phys] = xraw[pp][idx]
    side = np.zeros(len(P), np.int8); side[phys] = pside[pp][idx]
    xraw = xr
    pt0 = np.array([t0.get(int(c), -1e9) for c in C])
    matched = pt0 > -1e5
    assert np.array_equal(matched, phys), workdir                     # the t0 join is exact
    # one t0 per cluster => (xraw - x) is constant within a cluster AND drift side (a
    # cathode crosser gets opposite-sign corrections in its two halves); gates the match
    off = xraw[phys] - P[phys, 0]
    for c in np.unique(C[phys]):
        for sg in (-1, 1):
            o = off[(C[phys] == c) & (side[phys] == sg)]
            if len(o):
                assert o.max() - o.min() < 0.02, (workdir, int(c), sg, o.min(), o.max())
    tl = read_tlas(tgz.replace(".tar.gz", ".tlas"))
    return dict(run=run, evt=evt, workdir=workdir,
                x=P[:, 0].astype(np.float32), y=P[:, 1].astype(np.float32), z=P[:, 2].astype(np.float32),
                xraw=xraw.astype(np.float32), side=side, q=Q.astype(np.float32), cid=C.astype(np.int32),
                t0=pt0.astype(np.float64), phys=phys,
                ndup=ndup, nret0=nret0, nclus=len(t0), nmatched=int(sum(v > -1e5 for v in t0.values())),
                off_bot=float(tl["trigger_offset_bot_us"]), off_top=float(tl["trigger_offset_top_us"]),
                v_bot=float(tl["drift_speed_bot_mmus"]), v_top=float(tl["drift_speed_top_mmus"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    dirs = sorted(a.dirs)
    with Pool(a.jobs) as pool:
        res = pool.map(one, dirs, chunksize=1)
    cols = {}
    ev_run, ev_evt, ev_nclus, ev_nmat, ev_off = [], [], [], [], []
    for i, r in enumerate(res):
        n = len(r["x"])
        for k in ("x", "y", "z", "xraw", "side", "q", "cid", "t0", "phys"):
            cols.setdefault(k, []).append(r[k])
        cols.setdefault("ev", []).append(np.full(n, i, np.int16))
        ev_run.append(r["run"]); ev_evt.append(r["evt"]); ev_nclus.append(r["nclus"]); ev_nmat.append(r["nmatched"])
        ev_off.append((r["off_bot"], r["off_top"], r["v_bot"], r["v_top"]))
        print(f"{i:3d} {r['run']} {r['evt']:7d} pts {n:6d} phys {int(r['phys'].sum()):6d} "
              f"clusters {r['nclus']:3d} matched {r['nmatched']:3d} dup-match {r['ndup']:3d} re-t0 {r['nret0']:4d}", flush=True)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["ev_run"] = np.array(ev_run); out["ev_evt"] = np.array(ev_evt)
    out["ev_nclus"] = np.array(ev_nclus); out["ev_nmatched"] = np.array(ev_nmat)
    out["ev_tlas"] = np.array(ev_off); out["ev_nret0"] = np.array([r["nret0"] for r in res]); out["ev_ndup"] = np.array([r["ndup"] for r in res]); out["ev_dir"] = np.array([r["workdir"] for r in res])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print("wrote", a.out, "points", len(out["x"]), "physical", int(out["phys"].sum()),
          "events", len(res), "clusters", sum(ev_nclus), "matched", sum(ev_nmat))


if __name__ == "__main__":
    main()
