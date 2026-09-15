#!/usr/bin/env python3
"""doc pdvd/101 Phase 2 -- how far the CLUSTERED points of a simulated muon lie from the true
straight segment (the consistency check of the sim -> SP -> imaging -> clustering chain).

Reads the pctree the clustering job saved (work/<run6>_<k>_<tag>/pctree-evt<N>.tar.gz; the
TensorDM pctree: per-node parent index 'pointtrees/<N>/live', per-node point counts
'lpcmaps/arrays/3d', points concatenated in node order in 'namedpcs/3d/arrays/{x,y,z}',
root -> cluster -> blob), assigns every point to its cluster, and against the truth segment
(d101_make_muons.py truth JSON, tail/head in cm) reports, per event:
  nclus, npts, the largest cluster's share of the points;
  d     3-D distance of a point to the truth SEGMENT (cm): median / p90 for the largest
        cluster and for all points;
  dxT   x residual at fixed TRANSVERSE position: t = (p_yz - a_yz).u_T / (L sin th),
        x_line(t) = a_x + t L cos th, dxT = x - x_line(t) -- a time-base or drift-speed
        error shows here as a constant / a slope in t, which the perpendicular distance of a
        steep track hides;
  dyz   the transverse (y,z) distance to the line (cm);
  t range of the largest cluster (coverage: 0..1 = tail..head) and its x extent vs truth.
  slope of dxT vs t (cm per unit t) -- a drift-speed mismatch.
Optionally cross-checks the pctree against the clustering Bee JSON in the mabc zips: the same
number of points and the same x mean (a pctree/Bee disagreement is reported, not fixed).

Usage:
  d101_residual.py --truth /home/xqian/tmp/d101/sim/truth_pdvd.json --tag d101 [--ks 0,15] \
      --out /home/xqian/tmp/d101/sim/residual_pdvd_d101.tsv
"""
import argparse, glob, io, json, os, sys, tarfile, zipfile
import numpy as np

WCP = "/home/xqian/toolkit-dev/wcp-porting-img"


def load_pctree(tgz):
    arrs, tree = {}, None
    with tarfile.open(tgz, "r:gz") as tf:
        mem = {m.name: m for m in tf.getmembers()}
        for name, m in mem.items():
            if not name.endswith("_metadata.json"):
                continue
            md = json.load(io.BytesIO(tf.extractfile(m).read()))
            an = name.replace("_metadata.json", "_array.npy")
            if an not in mem:
                continue
            dp = md.get("datapath", "")
            if md.get("datatype") == "pctree" and dp.endswith("/live"):
                tree = np.load(io.BytesIO(tf.extractfile(mem[an]).read()))
            elif md.get("datatype") == "pcarray" and "/live/" in dp and (
                    dp.endswith("/lpcmaps/arrays/3d") or "/namedpcs/3d/arrays/" in dp
                    or "/namedpcs/cluster_scalar/arrays/" in dp):
                arrs[dp.split("/live/", 1)[1]] = np.load(io.BytesIO(tf.extractfile(mem[an]).read()))
    if tree is None:
        raise RuntimeError("no live pctree in " + tgz)
    return tree, arrs


def cluster_of_points(tree, counts):
    """tree[i] = parent node index (root = 0 with parent 0).  Returns per-point cluster ordinal."""
    n = len(tree)
    depth = np.zeros(n, int)
    for i in range(1, n):
        depth[i] = depth[tree[i]] + 1
    clus_nodes = [i for i in range(1, n) if depth[i] == 1]
    ordinal = {c: j for j, c in enumerate(clus_nodes)}
    top = np.arange(n)
    for i in range(1, n):
        top[i] = i if depth[i] == 1 else top[tree[i]]
    lab = np.repeat([ordinal.get(int(top[i]), -1) for i in range(n)], counts)
    return lab, len(clus_nodes)


def bee_clustering(workdir):
    out = []
    for z in sorted(glob.glob(os.path.join(workdir, "mabc-*.zip"))):
        with zipfile.ZipFile(z) as zf:
            for nm in zf.namelist():
                if nm.endswith(".json") and "clustering" in nm and "img" not in nm:
                    d = json.loads(zf.read(nm))
                    if "x" in d and len(d["x"]):
                        out.append((os.path.basename(z), nm, len(d["x"]), float(np.mean(d["x"])),
                                    len(set(d.get("cluster_id", [])))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ks", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = json.load(open(a.truth))
    ks = [int(x) for x in a.ks.split(",")] if a.ks else [e["k"] for e in T["events"]]
    cols = ["det", "k", "theta", "phi", "status", "nclus", "npts", "main_npts", "main_frac",
            "main_d_med", "main_d_p90", "all_d_med", "all_d_p90", "main_dxT_med", "main_dxT_p16", "main_dxT_p84",
            "main_dxT_slope", "main_dyz_med", "main_dyz_p90", "main_frac_d_lt2", "main_t_min", "main_t_max",
            "main_x_min", "main_x_max", "true_x_min", "true_x_max", "n_clusters_ge100pts", "bee_check", "pctree"]
    rows = []
    for k in ks:
        e = next(x for x in T["events"] if x["k"] == k)
        wd = os.path.join(WCP, "%s/work/%06d_%d_%s" % (T["det"], T["run"], k, a.tag))
        tg = sorted(glob.glob(os.path.join(wd, "pctree-evt*.tar.gz")))
        row = dict(det=T["det"], k=k, theta=e["theta_deg"], phi=e["phi_deg"])
        if len(tg) != 1:
            row.update(status="no_pctree(%d)" % len(tg)); rows.append(row); continue
        tree, arrs = load_pctree(tg[0])
        cnt = arrs["lpcmaps/arrays/3d"]
        x, y, z = (arrs["pointclouds/namedpcs/3d/arrays/" + c] / 10.0 for c in "xyz")
        if int(cnt.sum()) != len(x):
            row.update(status="count_mismatch %d vs %d" % (cnt.sum(), len(x))); rows.append(row); continue
        lab, nclus = cluster_of_points(tree, cnt)
        P = np.column_stack([x, y, z])
        A = np.array(e["tail_cm"]); B = np.array(e["head_cm"]); L = np.linalg.norm(B - A); u = (B - A) / L
        s = np.clip((P - A) @ u / L, 0, 1)
        d = np.linalg.norm(P - (A + np.outer(s * L, u)), axis=1)
        uyz = u[1:] / np.linalg.norm(u[1:]); Lt = L * np.linalg.norm(u[1:])
        t = ((P[:, 1:] - A[1:]) @ uyz) / Lt
        xline = A[0] + t * (B[0] - A[0])
        dxT = P[:, 0] - xline
        yzline = A[1:] + np.outer(t * Lt, uyz)
        dyz = np.linalg.norm(P[:, 1:] - yzline, axis=1)
        sizes = np.bincount(lab[lab >= 0], minlength=nclus)
        main = int(np.argmax(sizes)) if nclus else -1
        mm = lab == main
        q = lambda v, p: float(np.percentile(v, p)) if len(v) else float("nan")
        slope = float(np.polyfit(t[mm], dxT[mm], 1)[0]) if mm.sum() > 10 else float("nan")
        bee = bee_clustering(wd)
        bee_s = ";".join("%s:%s n=%d xmean=%.2f ncl=%d" % b for b in bee) if bee else "none"
        row.update(status="ok", nclus=nclus, npts=len(x), main_npts=int(mm.sum()), main_frac=mm.mean(),
                   main_d_med=q(d[mm], 50), main_d_p90=q(d[mm], 90), all_d_med=q(d, 50), all_d_p90=q(d, 90),
                   main_dxT_med=q(dxT[mm], 50), main_dxT_p16=q(dxT[mm], 16), main_dxT_p84=q(dxT[mm], 84),
                   main_dxT_slope=slope, main_dyz_med=q(dyz[mm], 50), main_dyz_p90=q(dyz[mm], 90),
                   main_frac_d_lt2=float((d[mm] < 2).mean()), main_t_min=q(t[mm], 0), main_t_max=q(t[mm], 100),
                   main_x_min=q(P[mm, 0], 0), main_x_max=q(P[mm, 0], 100),
                   true_x_min=min(A[0], B[0]), true_x_max=max(A[0], B[0]),
                   n_clusters_ge100pts=int((sizes >= 100).sum()), bee_check=bee_s, pctree=tg[0])
        rows.append(row)
    with open(a.out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(("%.3f" % r[c]) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + "\n")
    for r in rows:
        if r.get("status") != "ok":
            print("%s k=%d %s" % (r["det"], r["k"], r["status"])); continue
        print("%s k=%2d th=%2g ph=%2g nclus=%d(>=100pts %d) npts=%d main=%.3f | d main med %.2f p90 %.2f | all med %.2f p90 %.2f | "
              "dxT med %+.2f [%+.2f,%+.2f] slope %+.2f | dyz med %.2f | t %.2f..%.2f | x %.1f..%.1f (true %.1f..%.1f)"
              % (r["det"], r["k"], r["theta"], r["phi"], r["nclus"], r["n_clusters_ge100pts"], r["npts"], r["main_frac"],
                 r["main_d_med"], r["main_d_p90"], r["all_d_med"], r["all_d_p90"], r["main_dxT_med"], r["main_dxT_p16"],
                 r["main_dxT_p84"], r["main_dxT_slope"], r["main_dyz_med"], r["main_t_min"], r["main_t_max"],
                 r["main_x_min"], r["main_x_max"], r["true_x_min"], r["true_x_max"]))
        print("      bee: %s" % r["bee_check"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
