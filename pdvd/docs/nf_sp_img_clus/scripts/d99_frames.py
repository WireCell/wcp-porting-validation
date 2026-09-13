#!/usr/bin/env python3
"""doc pdvd/99 -- per-anode, per-plane charge sums of SP frame archives, and the L1SP share.

    python3 d99_frames.py --events 039252_0 --arms p98voff p98von [--nol1 p98g2nol1] [--json J]
    python3 d99_frames.py --all --arms p98voff p98von --json sums.json     # before the OFF trim

Plane of a channel: pgrapher params wires file (protodunevd-wires-larsoft-v7-uvwfit), anode ->
faces -> planes (ident 0/1/2 = U/V/W) -> wires -> channel.  Sums are over frame_gauss (the archive
tag gauss<N>, i.e. DNN-ROI + L1SP output), positive samples only.

Reported per plane:
  sum(ON)/sum(OFF)                         the charge-scale change (prediction: top 1/0.889 = 1.125, bottom 1)
  median per-channel ratio                 over channels with sum > 3e4 e in both arms
  L1SP share (with --nol1 = the same OFF config run with -L off):
     channels where the OFF gauss differs from the no-L1SP gauss, their OFF charge / plane OFF charge
"""
import argparse, bz2, collections, glob, io, json, os, tarfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
WORK = IMG + "/pdvd/work"
WIRES = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v7-uvwfit.json.bz2"
PL = "UVW"


def plane_map():
    S = json.load(bz2.open(WIRES))["Store"]
    out = {}
    for a in S["anodes"]:
        A = a["Anode"]
        for fi in A["faces"]:
            for pi in S["faces"][fi]["Face"]["planes"]:
                P = S["planes"][pi]["Plane"]
                for wi in P["wires"]:
                    out[(A["ident"], S["wires"][wi]["Wire"]["channel"])] = P["ident"]
    return out


def load_gauss(path):
    arr = {}
    with tarfile.open(path, "r:bz2") as tf:
        for m in tf:
            for pre in ("frame_gauss", "channels_gauss"):
                if m.name.startswith(pre):
                    arr[pre] = np.load(io.BytesIO(tf.extractfile(m).read()))
    return arr["channels_gauss"].astype(int), arr["frame_gauss"].astype(float)


def event_sums(job):
    evt, arms, nol1 = job
    PM = plane_map()
    out = {}
    for an in range(8):
        per = {}
        for arm in arms + ([nol1] if nol1 else []):
            f = f"{WORK}/{evt}_{arm}/protodune-sp-dnnroi-frames-anode{an}.tar.bz2"
            if not os.path.exists(f):
                per[arm] = None
                continue
            ch, G = load_gauss(f)
            o = np.argsort(ch)
            per[arm] = (ch[o], G[o])
        base = per.get(arms[0])
        if base is None:
            continue
        ch = base[0]
        planes = np.array([PM[(an, int(c))] for c in ch])
        for p in range(3):
            m = planes == p
            row = {}
            for arm in arms:
                if per.get(arm) is None:
                    continue
                assert np.array_equal(per[arm][0], ch)
                Gp = np.clip(per[arm][1][m], 0, None)
                row[arm] = dict(sum=float(Gp.sum()), chan=Gp.sum(axis=1))
            if len(arms) > 1 and all(a in row for a in arms[:2]):
                c0, c1 = row[arms[0]]["chan"], row[arms[1]]["chan"]
                sel = (c0 > 3e4) & (c1 > 3e4)
                row["ratio_sum"] = row[arms[1]]["sum"] / max(row[arms[0]]["sum"], 1e-9)
                row["ratio_med"] = float(np.median(c1[sel] / c0[sel])) if sel.any() else None
                row["n_sel"] = int(sel.sum())
            if nol1 and per.get(nol1) is not None:
                G0 = np.clip(base[1][m], 0, None); Gn = np.clip(per[nol1][1][m], 0, None)
                diff = np.any(base[1][m] != per[nol1][1][m], axis=1)
                row["l1sp_nchan"] = int(diff.sum())
                row["l1sp_share"] = float(G0[diff].sum() / max(G0.sum(), 1e-9))
                row["l1sp_absdiff_frac"] = float(np.abs(G0 - Gn).sum() / max(G0.sum(), 1e-9))
            for arm in arms:
                if arm in row:
                    row[arm] = {"sum": row[arm]["sum"]}
            out[f"{an}{PL[p]}"] = row
    return evt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--nol1", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    evts = a.events
    if a.all:
        evts = sorted({os.path.basename(d).rsplit("_" + a.arms[0], 1)[0]
                       for d in glob.glob(f"{WORK}/*_{a.arms[0]}") if glob.glob(d + "/protodune-sp-dnnroi-frames-anode*.tar.bz2")})
    res = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for evt, out in ex.map(event_sums, [(e, a.arms, a.nol1) for e in evts]):
            res[evt] = out
    print(f"events {len(res)}  arms {a.arms}  nol1 {a.nol1}")
    # pooled per anode/plane over events
    keys = sorted({k for o in res.values() for k in o}, key=lambda k: (int(k[:-1]), k[-1]))
    for k in keys:
        rows = [o[k] for o in res.values() if k in o]
        line = f"  anode {k[:-1]} {k[-1]}  vol {'top   ' if int(k[:-1]) >= 4 else 'bottom'}"
        if len(a.arms) > 1:
            s0 = sum(r[a.arms[0]]["sum"] for r in rows if a.arms[0] in r and a.arms[1] in r)
            s1 = sum(r[a.arms[1]]["sum"] for r in rows if a.arms[0] in r and a.arms[1] in r)
            meds = [r["ratio_med"] for r in rows if r.get("ratio_med") is not None]
            line += f"  sum {a.arms[1]}/{a.arms[0]} {s1 / max(s0, 1e-9):.4f}  per-event median-channel ratio p50 " \
                    f"{np.median(meds) if meds else float('nan'):.4f}"
        if a.nol1:
            sh = [r["l1sp_share"] for r in rows if "l1sp_share" in r]
            nc = sum(r["l1sp_nchan"] for r in rows if "l1sp_nchan" in r)
            ad = [r["l1sp_absdiff_frac"] for r in rows if "l1sp_absdiff_frac" in r]
            line += f"  L1SP-touched chans {nc}  share of OFF charge {np.mean(sh):.4f}  |OFF-noL1|/OFF {np.mean(ad):.4f}"
        print(line)
    for vol, an in (("bottom", range(4)), ("top", range(4, 8))):
        for p in PL:
            rows = [o[f"{x}{p}"] for o in res.values() for x in an if f"{x}{p}" in o]
            if len(a.arms) > 1:
                rr = [r for r in rows if a.arms[0] in r and a.arms[1] in r]
                s0 = sum(r[a.arms[0]]["sum"] for r in rr); s1 = sum(r[a.arms[1]]["sum"] for r in rr)
                print(f"  {vol:6s} {p}: sum {a.arms[1]}/{a.arms[0]} {s1 / max(s0, 1e-9):.4f}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
