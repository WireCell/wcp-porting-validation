#!/usr/bin/env python3
"""Convert the 3D sp (blob/space-point) nodes of a nugraph HDF5 file into a
Bee point-cloud zip for visual validation.

Per event, one Bee point set "nugraph_sp":
  - x,y,z  : sp/pos, converted mm -> cm (Bee wants cm),
  - q      : 1 for nu (y_semantic==0), 0 for cosmic (y_semantic==1),
             -1 for ghost/unlabeled (y_semantic==-1),
  - cluster_id : sp/y_instance (the truth trackid; -1 for ghost).

Usage:  python3 h5_sp_to_bee.py nugraph.h5 [out.zip]
Then upload:  BROWSER=echo bash ../sbnd_xin/upload-to-bee.sh TensorSetLabeler/out.zip
"""
import sys, os, json, zipfile
import numpy as np
import h5py


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    h5path = sys.argv[1]
    outzip = sys.argv[2] if len(sys.argv) > 2 else "nugraph_sp_bee.zip"

    f = h5py.File(h5path, "r")
    keys = list(f["dataset"].keys())
    print("events in file:", len(keys))

    tmp = outzip + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    zf = zipfile.ZipFile(outzip, "w", zipfile.ZIP_DEFLATED)

    for idx, key in enumerate(keys):
        r = f["dataset"][key][()]
        run = int(np.asarray(r["metadata/run"]))
        sub = int(np.asarray(r["metadata/subrun"]))
        evt = int(np.asarray(r["metadata/event"]))
        pos = np.asarray(r["sp/pos"], dtype=float)      # mm
        sem = np.asarray(r["sp/y_semantic"]).astype(int)
        inst = np.asarray(r["sp/y_instance"]).astype(int)

        feat = np.asarray(r["sp/features"], dtype=float)  # [charge, reco_cluster_id, ...]
        x = (pos[:, 0] / 10.0).tolist()   # mm -> cm
        y = (pos[:, 1] / 10.0).tolist()
        z = (pos[:, 2] / 10.0).tolist()

        # Truth present (sim) -> colour by trackid, q = nu(1)/cosmic(0)/ghost(-1).
        # Truth absent (data, all y_instance == -1) -> input-only graph: colour
        # by reco_cluster_id and use charge for q, for reco hand-scan.
        has_truth = not bool(np.all(inst == -1))
        if has_truth:
            q = np.where(sem == 0, 1.0, np.where(sem == 1, 0.0, -1.0))
            cid = inst
            mode = "truth (nu=%d cosmic=%d ghost=%d)" % (
                int((sem == 0).sum()), int((sem == 1).sum()), int((sem == -1).sum()))
        else:
            q = feat[:, 0]                 # charge
            cid = feat[:, 1].astype(int)   # reco_cluster_id
            mode = "data/input-only (colour = reco_cluster_id; %d reco clusters)" % (
                len(set(cid.tolist())))
        print("  [%d] rse=(%d,%d,%d): %d sp  %s" % (idx, run, sub, evt, len(x), mode))

        obj = {
            "runNo": str(run), "subRunNo": str(sub), "eventNo": str(evt),
            "geom": "sbnd", "type": "nugraph_sp",
            "x": x, "y": y, "z": z,
            "q": q.tolist(),
            "cluster_id": cid.tolist(),
            "real_cluster_id": cid.tolist(),
        }
        zf.writestr("data/%d/%d-nugraph_sp.json" % (idx, idx), json.dumps(obj))

    zf.close()
    print("wrote", outzip)


if __name__ == "__main__":
    main()
