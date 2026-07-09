#!/usr/bin/env python3
"""Sample the ProtoDUNE-VD PDFastSimANN computable-graph photon models
(TFLoaderMLP SavedModels on StashCache cvmfs) onto grids:

  protodune_vd_v4_{128,175}nm_tf2.6  -- the dunesw v10_05_00d00 nominal models
                                        (v4 geometry: Arapucas y-mirrored vs as-built!)
  protodune_vd_v5_{128,175}nm_tf2.6  -- v5 = as-built geometry, matches the real
                                        detector (raw-data opdet positions exactly)

Inputs are pos_x/pos_y/pos_z in cm, GDML world frame (== WCT frame: cathode at
x=0); output 'vis_full' is [N, 40] visibilities.

Stages (all run by default):
  order    coarse-grid hot-centroid per channel -> one-to-one assignment against
           the GDML opdet positions of the model's geometry version (gate), and
           for v5 also against the raw-data opdet positions -> wct_opdet map.
  checkv4  v4 128nm ANN evaluated at the 25^3 PVS voxel centers vs the Ar voxel
           library (needs work/photlib_vis_Ar.npy from extract_photlib.py).
  sample   fine grid over the active volume for v5 128nm (+ 175nm), written to
           work/ann_vis_v5_{128,175}nm.npz (vis [nx,ny,nz,40] f32 + grid meta).

Run inside a TF venv:  /home/xqian/tmp/tfvenv/bin/python sample_ann.py
"""
import argparse
import json
import os

import numpy as np

STASH = "/cvmfs/dune.osgstorage.org/pnfs/fnal.gov/usr/dune/persistent/stash/PhotonPropagation/ComputableGraph"
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLKIT_CFG = os.environ.get(
    "PDVD_CFG_DIR",
    "/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/protodunevd")
NCH = 40

# Fine sampling grid, cm, world/WCT frame: active volume + one step margin.
FINE_STEP = 10.0
FINE_LO = np.array([-350.0, -340.0, -10.0])
FINE_N = (71, 69, 33)   # -> +350 / +340 / +310

# Coarse grid for channel-order centroids (covers the PDs outside the active
# volume: membrane walls |y|~418, PMT/cathode planes).
COARSE_STEP = 20.0
COARSE_LO = np.array([-360.0, -440.0, -170.0])
COARSE_N = (37, 45, 32)


def load_model(name):
    import tensorflow as tf
    m = tf.saved_model.load(os.path.join(STASH, name))
    sig = m.signatures["serving_default"]

    def run(pts_cm):
        out = np.empty((len(pts_cm), NCH), dtype=np.float32)
        for i in range(0, len(pts_cm), 65536):
            c = pts_cm[i:i + 65536].astype(np.float32)
            r = sig(pos_x=tf.constant(c[:, 0:1]), pos_y=tf.constant(c[:, 1:2]),
                    pos_z=tf.constant(c[:, 2:3]))
            out[i:i + len(c)] = list(r.values())[0].numpy()
        return out

    return run


def grid_points(lo, step, n):
    ax = [lo[i] + step * np.arange(n[i]) for i in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)
    return g.reshape(-1, 3), ax


def hot_centroid(vis, pts, ntop=12):
    out = np.empty((NCH, 3))
    for ch in range(NCH):
        top = np.argsort(vis[:, ch])[-ntop:]
        w = vis[top, ch]
        out[ch] = (pts[top] * w[:, None]).sum(axis=0) / w.sum()
    return out


def assign(cents_mm, pos_mm, label, gate_mm):
    from scipy.optimize import linear_sum_assignment
    dist = np.linalg.norm(cents_mm[:, None, :] - pos_mm[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(dist)
    dmax = dist[rows, cols].max()
    print(f"  {label}: bijective assignment, max centroid-opdet distance {dmax:.0f} mm "
          f"(gate {gate_mm:.0f} mm)")
    if dmax >= gate_mm:
        raise SystemExit(f"channel-order gate FAILED for {label}")
    return cols


def stage_order(run, version):
    pts, _ = grid_points(COARSE_LO, COARSE_STEP, COARSE_N)
    vis = run(pts)
    cents = hot_centroid(vis, pts) * 10.0   # cm -> mm

    with open(os.path.join(HERE, f"pdvd-{version}-gdml-opdets.json")) as f:
        gdml = json.load(f)["opdets"]
    gpos = np.array([[o["x"], o["y"], o["z"]] for o in gdml])
    # centroid-vs-PD offsets: inward pull from one-sided illumination + 20cm grid
    cols = assign(cents, gpos, f"{version} ANN vs {version} GDML", 600.0)
    chan2node = [gdml[c]["node"] for c in cols]
    chan2pos = gpos[cols]

    wct = None
    if version == "v5":
        with open(os.path.join(TOOLKIT_CFG, "pdvd-opdet-geom.json")) as f:
            jjo = json.load(f)["opdets"]
        jpos = np.array([[o["x"], o["y"], o["z"]] for o in jjo])
        jdet = [o["opdet"] for o in jjo]
        wct = []
        for ch in range(NCH):
            d = np.linalg.norm(jpos - chan2pos[ch], axis=1)
            i = int(np.argmin(d))
            wct.append(jdet[i] if d[i] < 1.0 else None)   # data == v5 exactly
        nlive = sum(w is not None for w in wct)
        print(f"  v5 ANN channel -> wct_opdet: {nlive}/40 live (dead: "
              f"{[ch for ch in range(NCH) if wct[ch] is None]})")
    return chan2node, chan2pos, wct


def stage_checkv4(run):
    gj = json.load(open(os.path.join(HERE, "work", "photlib_grid.json")))
    n, org, stp = gj["n"], np.array(gj["origin_mm"]), np.array(gj["step_mm"])
    ids = np.arange(n[0] * n[1] * n[2])
    ix, iy, iz = ids % n[0], (ids // n[0]) % n[1], ids // (n[0] * n[1])
    centers = (org + (np.stack([ix, iy, iz], 1) + 0.5) * stp) / 10.0  # mm->cm
    lib = np.load(os.path.join(HERE, "work", "photlib_vis_Ar.npy"))
    ann = run(centers)
    both = (lib > 1e-8) & (ann > 1e-8)
    r = np.corrcoef(np.log(lib[both]), np.log(ann[both]))[0, 1]
    rat = ann[both] / lib[both]
    q = np.percentile(rat, [16, 50, 84])
    print(f"  v4 ANN vs Ar voxel library at {both.sum()} (voxel,ch) entries: "
          f"log-corr {r:.3f}, ANN/lib ratio 16/50/84% = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}")
    np.save(os.path.join(HERE, "work", "ann_v4_at_voxels.npy"), ann)


def stage_sample(run, model, tag, chan2node, chan2pos, wct):
    pts, ax = grid_points(FINE_LO, FINE_STEP, FINE_N)
    vis = run(pts).reshape(*FINE_N, NCH)
    out = os.path.join(HERE, "work", f"ann_vis_{tag}.npz")
    np.savez_compressed(
        out, vis=vis,
        origin_cm=FINE_LO, step_cm=np.array([FINE_STEP] * 3), n=np.array(FINE_N),
        chan_node=np.array(chan2node), chan_pos_mm=chan2pos,
        wct_opdet=np.array([-1 if w is None else w for w in wct]),
        model=model)
    print(f"  wrote {out}: vis {vis.shape} f32, "
          f"{os.path.getsize(out)/1e6:.1f} MB compressed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="order,checkv4,sample")
    args = ap.parse_args()
    stages = args.stages.split(",")

    if "order" in stages or "checkv4" in stages:
        print("== v4 128nm ANN")
        run4 = load_model("protodune_vd_v4_128nm_tf2.6")
        if "order" in stages:
            stage_order(run4, "v4")
        if "checkv4" in stages:
            stage_checkv4(run4)

    print("== v5 ANN")
    run5 = load_model("protodune_vd_v5_128nm_tf2.6")
    chan2node, chan2pos, wct = stage_order(run5, "v5")
    if "sample" in stages:
        stage_sample(run5, "protodune_vd_v5_128nm_tf2.6", "v5_128nm",
                     chan2node, chan2pos, wct)
        run5x = load_model("protodune_vd_v5_175nm_tf2.6")
        cn, cp, wx = stage_order(run5x, "v5")
        stage_sample(run5x, "protodune_vd_v5_175nm_tf2.6", "v5_175nm", cn, cp, wx)


if __name__ == "__main__":
    main()
