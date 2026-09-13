#!/usr/bin/env python3
"""doc pdvd/98 -- run the DUNE-VD diffusion drift regressor (m3-200k-w, W plane only) on the real-data Michel crops
of d98_michel_crops.py, and first prove that this import path reproduces the model's published predictions.

    CUDA_VISIBLE_DEVICES=1 python3 d98_predict.py --closure     # scan/d98/closure.txt
    CUDA_VISIBLE_DEVICES=1 python3 d98_predict.py               # scan/d98/scores.tsv

The model is used exactly as its own validation does (validation/score_val.py): load_model, normalize = log1p(x)/5,
predict in bf16 autocast on CUDA (fp32 is a different function by 26 cm, diffusion_t0_ml_validation.md sec 7).
Nothing in DNN_ROI_SP is modified; it is imported by path.

Closure gate: the stored test split (ml_prod200k/ml_dataset, sparse store) is scored for the first N_CLOSURE rows of
runs/m3-200k-w/predictions_test.csv and compared row by row; PASS = max |d mu| < 1 cm.
"""
import argparse, csv, json, os, sys
import numpy as np

ML = "/home/xqian/toolkit-dev/DNN_ROI_SP/simulation/dunevd_singlep/diffusion_t0/ml"
RUN = ML + "/runs/m3-200k-w"
DATASET = "/home/xqian/work/data/dunevd_singlep/ml_prod200k/ml_dataset"
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
SCAN = IMG + "/pdvd/docs/scan/d98"
OUT_TMP = "/home/xqian/tmp/d98"
N_CLOSURE = 64
sys.path.insert(0, ML); sys.path.insert(0, ML + "/../validation")
import score_val                                   # noqa: E402
import sparse_store                                # noqa: E402

VARIANTS = ("Z", "S", "M", "Zw")
SCORE_COLS = ["det", "key", "tier", "inrange", "drift_tick", "drift_fit"] + \
             [f"{s}_{v}" for v in VARIANTS for s in ("mu", "sig")] + \
             ["q_keep", "q_overlap", "frac_overlap", "frac_lost", "n_pix", "ke_best", "ke_region", "ratio", "q2d_ovl",
              "n_stop_gammas", "kind", "source", "conn_type", "michel_len", "kink_deg", "apa", "face", "d97"]


def closure(net, dev):
    rows = list(csv.DictReader(open(RUN + "/predictions_test.csv")))[:N_CLOSURE]
    store = sparse_store.SparseCrops(DATASET)
    imgs = np.stack([store.row(int(r["row"]), view=2) for r in rows])      # W view, dense (256,1024)
    mu, sig = score_val.predict(net, imgs, dev)
    dmu = mu - np.array([float(r["mu_cm"]) for r in rows])
    dsg = sig - np.array([float(r["sigma_cm"]) for r in rows])
    ok = np.abs(dmu).max() < 1.0
    with open(SCAN + "/closure.txt", "w") as f:
        f.write("# doc pdvd/98 -- closure of this import path against runs/m3-200k-w/predictions_test.csv (d98_predict.py --closure)\n")
        f.write(f"rows compared: {len(rows)} (first rows of the file; W view of the sparse store, bf16 on {dev})\n")
        f.write(f"max |d mu| = {np.abs(dmu).max():.3f} cm, mean d mu = {dmu.mean():+.3f}; max |d sigma| = {np.abs(dsg).max():.3f} cm\n")
        f.write(f"gate max|d mu| < 1 cm: {'PASS' if ok else 'FAIL'}\n")
        for r, a, b in list(zip(rows, mu, sig))[:5]:
            f.write(f"   row {r['row']}: published mu {r['mu_cm']} sigma {r['sigma_cm']} -> here {a:.2f} {b:.2f}\n")
    print(open(SCAN + "/closure.txt").read())
    return 0 if ok else 1


def score(net, dev, dets):
    cand = {(r["det"], r["key"]): r for r in csv.DictReader(
        (l for l in open(SCAN + "/candidates.tsv") if not l.startswith("#")), delimiter="\t")}
    out = []
    for det in dets:
        fn = f"{OUT_TMP}/crops_{det}.npz"
        if not os.path.exists(fn):
            print("no crops for", det); continue
        z = np.load(fn)
        keys = [str(k) for k in z["keys"]]; meta = json.loads(str(z["meta"]))
        pred = {}
        for v in VARIANTS:
            mu, sig = score_val.predict(net, z[v], dev)
            pred[v] = (mu, sig)
        for i, k in enumerate(keys):
            c = cand[(det, k)]; mt = meta[i]
            ratio = float(c["ratio"]); q2d_ovl = float(c["q2d_ovl"]); nsg = int(c["n_stop_gammas"])
            tierA = ratio >= 0.8 and 0 <= q2d_ovl < 0.2 and nsg == 0 and 0 <= mt["frac_lost"] < 0.2
            r = dict(det=det, key=k, tier="A" if tierA else "B", inrange=mt["inrange"],
                     drift_tick=mt["drift_tick"], drift_fit=mt["drift_fit"], q_keep=mt["q_keep"],
                     q_overlap=mt["q_overlap"], frac_overlap=mt["frac_overlap"], frac_lost=mt["frac_lost"], n_pix=mt["n_pix"],
                     ke_best=float(c["ke_best"]), ke_region=float(c["ke_region"]), ratio=ratio, q2d_ovl=q2d_ovl,
                     n_stop_gammas=nsg, kind=c["kind"], source=c["source"], conn_type=int(c["conn_type"]),
                     michel_len=float(c["michel_len"]), kink_deg=float(c["kink_deg"]), apa=mt["apa"],
                     face=mt["face"], d97=c["d97"])
            for v in VARIANTS:
                r[f"mu_{v}"] = float(pred[v][0][i]); r[f"sig_{v}"] = float(pred[v][1][i])
            out.append(r)
    with open(SCAN + "/scores.tsv", "w") as f:
        f.write("# doc pdvd/98 -- m3-200k-w predictions on the real-data Michel crops (d98_predict.py; bf16, CUDA)\n"
                "# variants: Z = Michel only, muon zeroed; S = overlap cells scaled by (charge-pred_mu)/charge; "
                "M = muon left in; Zw = Z with the wider mask.  mu/sig in cm from the response plane.\n"
                "# tier A = Bragg ratio>=0.8 & q2d_ovl<0.2 & n_stop_gammas==0 & frac_lost<0.2; inrange = 80<=drift_tick<=340\n")
        f.write("\t".join(SCORE_COLS) + "\n")
        for r in out:
            f.write("\t".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in SCORE_COLS) + "\n")
    print("wrote", SCAN + "/scores.tsv", len(out), "rows")
    for det in dets:
        R = [r for r in out if r["det"] == det]
        print(f"  {det}: {len(R)} scored, tier A {sum(r['tier'] == 'A' for r in R)}, in-range {sum(r['inrange'] for r in R)}")
    return 0


def main():
    global SCAN, OUT_TMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--closure", action="store_true")
    ap.add_argument("--det", default="pdhd,pdvd")
    ap.add_argument("--run", default="", help="doc pdvd/98 sec 11: read/write scan/d98/<run>/ and /home/xqian/tmp/d98/<run>/")
    a = ap.parse_args()
    if a.run:
        SCAN, OUT_TMP = f"{SCAN}/{a.run}", f"{OUT_TMP}/{a.run}"
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "the published function is the bf16 CUDA one; refuse to score on CPU"
    net, ep = score_val.load_model(RUN, dev)
    print("model", RUN, "epoch", ep, "device", dev)
    os.makedirs(SCAN, exist_ok=True)
    if a.closure:
        return closure(net, dev)
    return score(net, dev, a.det.split(","))


if __name__ == "__main__":
    sys.exit(main())
