#!/usr/bin/env python3
"""doc pdvd/117 S6 (simulation half) -- ISOCHRONOUS straight tracks across the collection wires at a ladder of drift
distances, for calibrating the time-width estimator of d117_s6_data.py against a known D_L.

    python3 d117_s6_tracks.py --cfg <compiled xtrack driver json> --anode 0 --outdir /home/xqian/tmp/d117/s6sim

Each track runs along the W pitch direction (perpendicular to the collection wires) for --nwires wires at a fixed
drift distance d from the W plane, so every W wire sees a short pulse whose time width is response + SP + 2 D_L t.
A second track at each d is tilted in x so it crosses the W wires at --tpw ticks per wire (the data sample's box term,
d117_s6_data.py).  Tracks at different d are >= 20 cm apart in drift (>= 270 ticks), so a shared channel is resolved
by time.  Geometry helpers (wires file, plane sorting, Drifter xregions) are d47_make_xtracks.py's, imported.
Truth: per track the W channels it crosses and the x at each crossing -> drift = |x - x_W|.
"""
import argparse, json, os, sys
import numpy as np
import d47_make_xtracks as X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--anode", type=int, default=0)
    ap.add_argument("--face", type=int, default=None)
    ap.add_argument("--drifts", default="25,40,60,85,110,140,170,200,230,260,290,320")
    ap.add_argument("--nwires", type=int, default=24)
    ap.add_argument("--tpw", type=float, default=1.5)
    ap.add_argument("--charge", type=float, default=-500.0)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    anodes, faces, planes, wires, P = X.load_geom(os.path.join(X.DATA, X.WIRES["pdvd"]))
    cfg = json.load(open(a.cfg))
    drifter = next(n for n in cfg if n.get("type") == "Drifter")["data"]
    tick = next(n for n in cfg if n.get("type") in ("DepoTransform", "DepoFluxSplat"))["data"]["tick"]
    an = next(x for x in anodes if x["ident"] == a.anode)
    cand = []
    for fi, fidx in enumerate(an["faces"]):
        pg = [X.plane_geometry(planes[pi], wires, P) for pi in faces[fidx]["planes"][:3]]
        for xr in drifter["xregions"]:
            if abs(xr["anode"] - pg[2]["x"]) < 20:
                cand.append((fi, fidx, pg, xr))
    if a.face is not None:
        cand = [c for c in cand if c[0] == a.face]
    if len(cand) != 1:
        print("face selection ambiguous/empty:", [(c[0], c[3]) for c in cand], file=sys.stderr); return 1
    fi, fidx, pg, xr = cand[0]
    W = pg[2]
    sgn = 1.0 if xr["cathode"] < xr["anode"] else -1.0
    v_mm_ns = drifter["drift_speed"]
    xW = W["x"]
    nW = len(W["idents"])
    mid = 0.5 * (W["bbox_min"] + W["bbox_max"])
    # a point on the W plane centre line, at wire coordinate k: c0 + pdir*pitch*k, shifted along the wire to the middle
    s_mid = float(np.dot(W["wdir"], mid - W["c0"]))
    drifts = [float(d) for d in a.drifts.split(",")]
    tracks, truth = [], []
    starts = [8 + (i % 4) * (a.nwires + 8) for i in range(2 * len(drifts))]
    if max(starts) + a.nwires + 8 > nW:
        print("too many wires per track for", nW, file=sys.stderr); return 1
    dx_per_wire = a.tpw * tick * v_mm_ns          # mm of x per wire at the tilt
    for i, d in enumerate(drifts):
        for kind, tilt in (("iso", 0.0), ("tilt", dx_per_wire)):
            k0 = starts[len(tracks)]; k1 = k0 + a.nwires
            x0 = xW - sgn * d * 10.0                      # mm, d cm from the W plane toward the cathode
            x1 = x0 - sgn * tilt * a.nwires               # the tilted track goes further from the anode
            p0 = W["c0"] + W["pdir"] * W["pitch"] * (k0 - 0.5) + W["wdir"] * s_mid
            p1 = W["c0"] + W["pdir"] * W["pitch"] * (k1 + 0.5) + W["wdir"] * s_mid
            p0 = np.array([x0, p0[1], p0[2]]); p1 = np.array([x1, p1[1], p1[2]])
            if not (min(xr["cathode"], xr["response"]) < min(x0, x1) and max(x0, x1) < max(xr["cathode"], xr["response"])):
                print(f"drift {d} cm outside response..cathode {xr}", file=sys.stderr); return 1
            tracks.append({"tail": (p0 / 10.0).tolist(), "head": (p1 / 10.0).tolist(), "charge": a.charge})
            chans = []
            for k in range(k0, k1 + 1):
                f = (k - (k0 - 0.5)) / (a.nwires + 1.0)
                xk = x0 + f * (x1 - x0)
                chans.append({"k": k, "channel": int(W["channels"][k]), "x_mm": float(xk),
                              "drift_cm": float(abs(xk - xW) / 10.0)})
            truth.append({"id": len(tracks) - 1, "kind": kind, "drift_cm": d, "tpw": 0.0 if tilt == 0 else a.tpw,
                          "w_channels": chans})
    hdr = {"anode": a.anode, "face_index": fi, "xregion_mm": xr, "x_W_mm": xW, "pitch_W_mm": W["pitch"],
           "lar": {"DL_cm2s": drifter["DL"] * 1e7, "DT_cm2s": drifter["DT"] * 1e7,
                   "drift_speed_mm_us": v_mm_ns * 1e3}, "tick_ns": tick, "tracks": truth}
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(tracks, open(f"{a.outdir}/tracks_s6_a{a.anode}.json", "w"))
    json.dump(hdr, open(f"{a.outdir}/truth_s6_a{a.anode}.json", "w"), indent=1)
    print(f"anode {a.anode} face {fi}: x_W {xW:.1f} mm, xregion {xr}, {len(tracks)} tracks, "
          f"drift speed {v_mm_ns * 1e3:.5f} mm/us, DL {drifter['DL'] * 1e7:.4f} cm2/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
