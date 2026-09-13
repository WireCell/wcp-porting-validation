#!/usr/bin/env python3
"""doc pdvd/98 sec 11 -- does the top gain or the wire order own the slope drop on the same top Michels?

    CUDA_VISIBLE_DEVICES=1 python3 d98_rescale_check.py [--run latest_sw99] > ../../scan/d98/latest_sw99/rescale_check.txt

The latest arm differs from production on the top volume by two things at once: the v7 wire order in SP and the top gain
(top SP frames x1.1249 per channel, doc pdvd/99 sec 0).  The knob-off rerun's frames were trimmed to 6 events, and the
July production frames are retired, so the two are separated on the crops themselves (round-1 crops kept in scratch):
  A  pixels: on each common object with the same crop origin, latest / 1.1249 (top) against the round-1 crop --
     the charge ratio and the L1 residual sum|latest/s - round1| / sum round1 (0 = the gain is the whole change);
  B  model: latest top crops divided by 1.1249, and round-1 top crops multiplied by 1.1249, re-scored with the same bf16
     CUDA path as d98_predict.py; the readouts on the common in-range objects for all four image sets.
  - latest / 1.1249 back at round 1's slope and round 1 x 1.1249 at the latest slope -> the gain owns the drop;
  - neither moves                                                                   -> the wire order owns it.
Bottom crops are never scaled (bottom frames carry no gain).
"""
import argparse, collections, csv, json, sys
import numpy as np
import d98_plots as P
import d98_predict as PR

S_TOP = 1.1249
CARRIED = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/scan/pdvd_stm_michel_smx9_carried_p98vonq.json"


def rows(path):
    return {r["key"]: r for r in csv.DictReader((l for l in open(path) if not l.startswith("#")), delimiter="\t")
            if r["det"] == "pdvd"}


def crops(path):
    z = np.load(path)
    keys = [str(k) for k in z["keys"]]
    return z["Z"], {k: i for i, k in enumerate(keys)}, json.loads(str(z["meta"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="latest_sw99")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "the published function is the bf16 CUDA one"
    net, ep = PR.score_val.load_model(PR.RUN, dev)
    ZL, iL, mL = crops(f"{PR.OUT_TMP}/{a.run}/crops_pdvd.npz")
    Z1, i1, m1 = crops(f"{PR.OUT_TMP}/crops_pdvd.npz")
    topL = np.array([m["apa"] >= 4 for m in mL]); top1 = np.array([m["apa"] >= 4 for m in m1])
    ZLd = ZL.copy(); ZLd[topL] /= S_TOP
    Z1u = Z1.copy(); Z1u[top1] *= S_TOP
    mu = {n: PR.score_val.predict(net, X, dev) for n, X in (("L", ZL), ("Ld", ZLd), ("1", Z1), ("1u", Z1u))}
    lat = rows(f"{P.SCAN}/{a.run}/scores.tsv")
    r1 = rows(P.SCAN + "/scores.tsv")
    print(f"# doc pdvd/98 sec 11 -- the top gain divided out of the latest crops / put into the round-1 crops (s {S_TOP}); "
          f"d98_rescale_check.py; model epoch {ep}")
    for n, rr, ii in (("L", lat, iL), ("1", r1, i1)):
        d = max(abs(float(rr[k]["mu_Z"]) - float(mu[n][0][i])) for k, i in ii.items())
        print(f"closure {'latest ' if n == 'L' else 'round 1'}: max |mu_Z re-scored - scores.tsv| = {d:.4f} cm over {len(ii)} crops "
              f"(gate < 0.1: {'PASS' if d < 0.1 else 'FAIL'})")
    cf = {r["key"]: r["carried_from"] for r in json.load(open(CARRIED))}
    pairs = [(k, cf[k]) for k in iL if cf.get(k) in i1]

    print(f"\n== A. pixels on the {len(pairs)} common objects (latest top / {S_TOP}; bottom as is)")
    for vol in ("top", "bottom"):
        same, qr, l1, l1raw = 0, [], [], []
        for kl, k1 in pairs:
            ml, mo = mL[iL[kl]], m1[i1[k1]]
            if (ml["apa"] >= 4) != (vol == "top"):
                continue
            if (ml["c0"], ml["t0"]) != (mo["c0"], mo["t0"]):
                continue
            same += 1
            xl, x1 = ZL[iL[kl]].astype(np.float64), Z1[i1[k1]].astype(np.float64)
            s = S_TOP if vol == "top" else 1.0
            qr.append(xl.sum() / x1.sum())
            l1.append(np.abs(xl / s - x1).sum() / x1.sum()); l1raw.append(np.abs(xl - x1).sum() / x1.sum())
        n_vol = sum((mL[iL[kl]]["apa"] >= 4) == (vol == "top") for kl, _ in pairs)
        if same:
            print(f"  {vol:6s} common {n_vol}, same crop origin {same}: charge latest/round1 median {np.median(qr):.4f} "
                  f"[p16 {np.quantile(qr, .16):.4f}, p84 {np.quantile(qr, .84):.4f}]; L1 residual as is median {np.median(l1raw):.4f}, "
                  f"after /s median {np.median(l1):.4f} [p84 {np.quantile(l1, .84):.4f}, max {max(l1):.4f}]")

    print("\n== B. readouts on the common in-range objects, per image set")
    for vol in ("all", "top", "bottom"):
        L = []
        for kl, k1 in pairs:
            if not (int(lat[kl]["inrange"]) == 1 and int(r1[k1]["inrange"]) == 1):
                continue
            if vol != "all" and (mL[iL[kl]]["apa"] >= 4) != (vol == "top"):
                continue
            L.append((float(lat[kl]["drift_tick"]), float(r1[k1]["drift_tick"]),
                      mu["L"][0][iL[kl]], mu["L"][1][iL[kl]], mu["Ld"][0][iL[kl]], mu["Ld"][1][iL[kl]],
                      mu["1"][0][i1[k1]], mu["1"][1][i1[k1]], mu["1u"][0][i1[k1]], mu["1u"][1][i1[k1]]))
        L = np.array(L)
        print(f"  {vol}: n {len(L)}; median |mu(latest/s) - mu(round 1)| {np.median(np.abs(L[:, 4] - L[:, 6])):.2f} cm, "
              f"median |mu(round 1 x s) - mu(latest)| {np.median(np.abs(L[:, 8] - L[:, 2])):.2f} cm, "
              f"median |mu(latest) - mu(round 1)| {np.median(np.abs(L[:, 2] - L[:, 6])):.2f} cm")
        for name, x, y, sg in (("round 1 (production)", 1, 6, 7), ("round 1 x s (gain put in)", 1, 8, 9),
                               ("latest", 0, 2, 3), ("latest / s (gain taken out)", 0, 4, 5)):
            rng = np.random.default_rng(P.SEED)
            print(f"    {name:28s} Z: {P.fmt(P.readout(L[:, x], L[:, y], L[:, sg], rng))}")

    print("\n== C. where the rest sits: per object, split by crop origin (same c0,t0 as round 1 or moved) and anode")
    print("  (anodes 2,3,6,7 exchange face channel sets between the v5 and v7 wire order, doc pdvd/99 sec 4.3; 4-7 carry the gain)")
    for vol in ("top", "bottom"):
        for same_o in (True, False):
            sel = [(kl, k1) for kl, k1 in pairs if (mL[iL[kl]]["apa"] >= 4) == (vol == "top")
                   and ((mL[iL[kl]]["c0"], mL[iL[kl]]["t0"]) == (m1[i1[k1]]["c0"], m1[i1[k1]]["t0"])) == same_o]
            if not sel:
                continue
            dmu_s = np.array([abs(mu["Ld"][0][iL[kl]] - mu["1"][0][i1[k1]]) for kl, k1 in sel])
            dmu = np.array([abs(mu["L"][0][iL[kl]] - mu["1"][0][i1[k1]]) for kl, k1 in sel])
            dsc = np.array([abs(mu["1u"][0][i1[k1]] - mu["1"][0][i1[k1]]) for kl, k1 in sel])
            apas = sorted(collections.Counter(mL[iL[kl]]["apa"] for kl, _ in sel).items())
            dor = [(mL[iL[kl]]["c0"] - m1[i1[k1]]["c0"], mL[iL[kl]]["t0"] - m1[i1[k1]]["t0"]) for kl, k1 in sel] if not same_o else []
            print(f"  {vol:6s} {'same origin ' if same_o else 'origin moved'} n {len(sel):2d} anodes {dict(apas)}: median |mu latest - round1| "
                  f"{np.median(dmu):.2f} cm, after the gain is taken out {np.median(dmu_s):.2f} cm; the scale alone on round 1 "
                  f"{np.median(dsc):.2f} cm" + (f"; origin shift (dc, dt) median ({np.median([d[0] for d in dor]):+.0f}, "
                                                  f"{np.median([d[1] for d in dor]):+.0f}), max |dc| {max(abs(d[0]) for d in dor)}, "
                                                  f"max |dt| {max(abs(d[1]) for d in dor)}" if dor else ""))
    print("\n== D. leave one out: OLS slope on the common in-range objects without each object (only those moving the "
          "latest - round 1 slope difference by > 0.03 are listed)")
    for vol in ("all", "top", "bottom"):
        T = [(kl, k1) for kl, k1 in pairs if int(lat[kl]["inrange"]) == 1 and int(r1[k1]["inrange"]) == 1
             and (vol == "all" or (mL[iL[kl]]["apa"] >= 4) == (vol == "top"))]

        def slopes(sel):
            x1 = np.array([float(r1[k1]["drift_tick"]) for _, k1 in sel]); xl = np.array([float(lat[kl]["drift_tick"]) for kl, _ in sel])
            f = lambda x, y: np.polyfit(x, np.array(y), 1)[0]
            return (f(x1, [mu["1"][0][i1[k1]] for _, k1 in sel]), f(x1, [mu["1u"][0][i1[k1]] for _, k1 in sel]),
                    f(xl, [mu["L"][0][iL[kl]] for kl, _ in sel]), f(xl, [mu["Ld"][0][iL[kl]] for kl, _ in sel]))
        s = slopes(T)
        print(f"  {vol:6s} n {len(T)}: slope round 1 {s[0]:.3f}, round 1 x s {s[1]:.3f}, latest {s[2]:.3f}, latest / s {s[3]:.3f}; "
              f"drop {s[0] - s[2]:+.3f} of which the scale alone {s[0] - s[1]:+.3f} (round 1 side) / {s[3] - s[2]:+.3f} (latest side)")
        for j, (kl, k1) in enumerate(T):
            u = slopes(T[:j] + T[j + 1:])
            if abs((u[0] - u[2]) - (s[0] - s[2])) > 0.03:
                print(f"    without {kl:14s} (npix {m1[i1[k1]]['n_pix']}->{mL[iL[kl]]['n_pix']}, mu {mu['1'][0][i1[k1]]:.0f}->"
                      f"{mu['L'][0][iL[kl]]:.0f} cm at drift {float(r1[k1]['drift_tick']):.0f}->{float(lat[kl]['drift_tick']):.0f}): "
                      f"round 1 {u[0]:.3f}, latest {u[2]:.3f}, drop {u[0] - u[2]:+.3f}, scale alone {u[0] - u[1]:+.3f}")

    print("\n== E. every common object")
    for kl, k1 in pairs:
        ml, mo = mL[iL[kl]], m1[i1[k1]]
        print(f"    {kl:14s} <- {k1:14s} apa {ml['apa']} inrange {int(lat[kl]['inrange'])}{int(r1[k1]['inrange'])} "
              f"origin {'same ' if (ml['c0'], ml['t0']) == (mo['c0'], mo['t0']) else 'moved'} "
              f"q latest/r1 {ml['q_keep'] / mo['q_keep']:.3f} npix {mo['n_pix']}->{ml['n_pix']} drift {r1[k1]['drift_tick']}->{lat[kl]['drift_tick']} "
              f"mu r1 {mu['1'][0][i1[k1]]:6.1f} r1xs {mu['1u'][0][i1[k1]]:6.1f} latest {mu['L'][0][iL[kl]]:6.1f} latest/s {mu['Ld'][0][iL[kl]]:6.1f}")


if __name__ == "__main__":
    sys.exit(main())
