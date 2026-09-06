#!/usr/bin/env python3
"""doc pdvd/47 -- generate controlled straight tracks at a fixed angle to the DRIFT direction
for the sim -> NF -> SP transverse-width study, plus the TRUTH file the estimator reads.

Every track starts 1 cm past the response plane and ends near the cathode, moving in the
transverse direction --dir (y or z, default z) by tan(theta) * |dx| (--tan-theta, default
0.30 = 16.7 deg from the drift axis).  Each time tick therefore samples a different drift
distance, and in every plane the track advances a FIXED, small number of wires per time
slice (the "prolonged" regime doc 44 selects in data: |dwire/dslice| < 0.25-0.44), so each
wire sees an ordinary ~20-tick pulse.  A track exactly along x (--tan-theta 0) is NOT
usable: every wire then carries a ~2 ms DC signal that the induction planes and the SP
ROI high-pass filters cannot preserve (measured 2026-09-05: PDHD collection gauss empty).
The sub-pitch phase sweeps continuously along each track, so all phases are sampled.

Placement: seeded rejection sampling of (y, z_start) inside the region, with the whole
track kept inside it, accepting only sets whose pairwise separation is >= --min-sep wires
in EVERY plane (separations are constant along x since all tracks share one direction).

Wire coordinate convention: the wires of a plane are sorted along the pitch direction
(oriented so the sorted order has increasing projection) and indexed 0..N-1 in that order
(the wire index the estimator's +-3 window is built on); w = (pdir.(p - c0))/pitch with c0
the centre of wire 0.  Per plane the truth line is w(x) = w0 + slope * |x - x_start|
(slope in wires/mm of drift; advance per slice = |slope| * v * tick * ticks_per_slice).
The channel of each wire comes from the wires file -- for PDHD's wrapped induction planes
several wires of the same face share a channel, which is why windows are built in WIRE
order and mapped to channels, never in channel-rank order.

The drift volume (anode cut / response plane / cathode x) is read from the Drifter
'xregions' of a COMPILED driver config (--cfg), picking the region whose anode x is within
20 mm of the chosen face's W-plane x (pass --face when both faces of an anode sit at the
same x, as on PDVD).

Usage:
  d47_make_xtracks.py --det pdhd --cfg S1.json --anode 1 --face 1 --n 10 --seed 47 \
      --y0 150 --y1 450 --z0 5 --z1 230 --outdir /home/xqian/tmp/xtrack/pdhd
  -> tracks_pdhd_a1.json  (the 'tracks' TLA)   truth_pdhd_a1.json  (per track per plane)
"""
import argparse, bz2, json, os, sys
import numpy as np

WIRES = {"pdhd": "protodunehd-wires-larsoft-v1.json.bz2",
         "pdvd": "protodunevd-wires-larsoft-v7-uvwfit.json.bz2",
         "sbnd": "sbnd-wires-geometry-v0206.json.bz2"}
DATA = "/home/xqian/toolkit-dev/wire-cell-data"


def load_geom(path):
    s = json.load(bz2.open(path))["Store"]
    anodes = [a["Anode"] for a in s["anodes"]]; faces = [f["Face"] for f in s["faces"]]
    planes = [p["Plane"] for p in s["planes"]]; wires = [w["Wire"] for w in s["wires"]]
    points = [p["Point"] for p in s["points"]]
    P = np.array([[p["x"], p["y"], p["z"]] for p in points], float)
    return anodes, faces, planes, wires, P


def plane_geometry(plane, wires, P):
    """Wires of one plane sorted along the pitch direction. Returns dict."""
    ws = [wires[i] for i in plane["wires"]]
    tails = P[[w["tail"] for w in ws]]; heads = P[[w["head"] for w in ws]]
    d = heads - tails; d /= np.linalg.norm(d, axis=1)[:, None]
    wdir = d.mean(axis=0); wdir /= np.linalg.norm(wdir)
    xhat = np.array([1.0, 0, 0]); pdir = np.cross(wdir, xhat); pdir /= np.linalg.norm(pdir)
    cen = 0.5 * (tails + heads)
    proj = cen @ pdir
    if np.argsort(proj)[0] != 0 and proj[-1] < proj[0]:
        pdir = -pdir; proj = -proj          # orient along the file's wire order if it is monotone
    order = np.argsort(proj)
    proj_s = proj[order]; pitch = np.median(np.diff(proj_s))
    return dict(wdir=wdir, pdir=pdir, pitch=float(pitch), order=order, proj=proj_s,
                c0=cen[order[0]], x=float(cen[:, 0].mean()),
                idents=[ws[i]["ident"] for i in order], channels=[ws[i]["channel"] for i in order],
                segments=[ws[i]["segment"] for i in order],
                bbox_min=np.vstack([tails, heads]).min(axis=0), bbox_max=np.vstack([tails, heads]).max(axis=0))


def wire_coord(pg, p):
    return float(np.dot(pg["pdir"], p - pg["c0"]) / pg["pitch"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=WIRES)
    ap.add_argument("--wires", default=None)
    ap.add_argument("--cfg", required=True, help="compiled driver JSON (Drifter xregions, lar, tick)")
    ap.add_argument("--anode", type=int, required=True, help="anode ident")
    ap.add_argument("--face", type=int, default=None, help="face index within the anode (default: the one with a drift volume)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--y0", type=float, required=True); ap.add_argument("--y1", type=float, required=True)
    ap.add_argument("--z0", type=float, required=True); ap.add_argument("--z1", type=float, required=True)
    ap.add_argument("--min-sep", type=float, default=8.0, help="wires, every plane")
    ap.add_argument("--max-phase-gap", type=float, default=0.2)
    ap.add_argument("--start-past-response", type=float, default=1.0, help="cm past the response plane")
    ap.add_argument("--end-before-cathode", type=float, default=5.0, help="cm before the cathode cut")
    ap.add_argument("--charge", type=float, default=-500.0, help="electrons per 0.1 mm step (5000 e/mm)")
    ap.add_argument("--tan-theta", type=float, default=0.30, help="transverse displacement per unit drift")
    ap.add_argument("--dir", default="z", choices=("y", "z"), help="transverse direction of motion")
    ap.add_argument("--ticks-per-slice", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    wires_path = a.wires or os.path.join(DATA, WIRES[a.det])
    anodes, faces, planes, wires, P = load_geom(wires_path)
    cfg = json.load(open(a.cfg))
    drifter = next(n for n in cfg if n.get("type") == "Drifter")["data"]
    tick = next(n for n in cfg if n.get("type") in ("DepoTransform", "DepoFluxSplat"))["data"]["tick"]
    an = next(x for x in anodes if x["ident"] == a.anode)
    # pick the face with a drift volume: the xregion whose anode x is within 20 mm of the W plane x
    cand = []
    for fi, fidx in enumerate(an["faces"]):
        pg = [plane_geometry(planes[pi], wires, P) for pi in faces[fidx]["planes"][:3]]
        for xr in drifter["xregions"]:
            if abs(xr["anode"] - pg[2]["x"]) < 20:
                cand.append((fi, fidx, pg, xr))
    if a.face is not None:
        cand = [c for c in cand if c[0] == a.face]
    if len(cand) != 1:
        print("face selection ambiguous/empty:", [(c[0], c[3]) for c in cand], file=sys.stderr); return 1
    fi, fidx, pg, xr = cand[0]
    sgn = 1.0 if xr["cathode"] < xr["anode"] else -1.0          # drift direction sign: cathode -> anode
    x_start = xr["response"] - sgn * a.start_past_response * 10.0   # mm; 1 cm past the response plane toward the cathode
    x_end = xr["cathode"] + sgn * a.end_before_cathode * 10.0
    for p in pg:
        for w in ("y", "z"):
            pass
    lo = np.array([a.y0, a.z0]) * 10.0; hi = np.array([a.y1, a.z1]) * 10.0
    rng = np.random.default_rng(a.seed)
    L = abs(x_end - x_start)
    tdir = np.array([1.0, 0.0]) if a.dir == "y" else np.array([0.0, 1.0])      # (y, z)
    disp = a.tan_theta * L                                                      # transverse displacement over the track
    # the whole track must fit: shrink the start window along tdir
    hi_start = hi - tdir * disp
    if (hi_start <= lo).any():
        print("region too small for tan_theta %.2f: displacement %.0f mm" % (a.tan_theta, disp), file=sys.stderr); return 1
    slope = [float(np.dot(g["pdir"][1:], tdir)) * a.tan_theta / g["pitch"] for g in pg]   # wires per mm of drift

    def coords(yz):
        p = np.array([x_start, yz[0], yz[1]])
        return [wire_coord(g, p) for g in pg]

    def inside(yz):
        cs0 = coords(yz); cs1 = coords(yz + tdir * disp)
        return all(5 < c < len(g["idents"]) - 6 for c, g in zip(cs0, pg)) and all(5 < c < len(g["idents"]) - 6 for c, g in zip(cs1, pg))

    def ok_sep(cs, acc):
        return all(all(abs(cs[i] - o[i]) >= a.min_sep for i in range(3)) for o in acc)

    pts = []
    for _ in range(400000):
        yz = lo + rng.random(2) * (hi_start - lo)
        if not inside(yz):
            continue
        if ok_sep(coords(yz), [coords(q) for q in pts]):
            pts.append(yz)
        if len(pts) == a.n:
            break
    if len(pts) < a.n:
        print("could not place %d tracks (got %d); widen the region or lower --min-sep" % (a.n, len(pts)), file=sys.stderr); return 1
    hw = 3
    tracks, truth = [], []
    v_tick = drifter["drift_speed"] * tick     # mm per tick
    for tid, yz in enumerate(pts):
        yz1 = yz + tdir * disp
        tracks.append({"tail": [x_start / 10.0, yz[0] / 10.0, yz[1] / 10.0],
                       "head": [x_end / 10.0, yz1[0] / 10.0, yz1[1] / 10.0], "charge": a.charge})
        tp = []
        for pi, g in enumerate(pg):
            w0 = coords(yz)[pi]; w1 = wire_coord(g, np.array([x_end, yz1[0], yz1[1]]))
            k0, k1 = int(round(w0)), int(round(w1))
            tp.append({"plane": pi, "w0": w0, "w1": w1, "slope_wire_per_mm": slope[pi],
                       "advance_wire_per_slice": abs(slope[pi]) * v_tick * a.ticks_per_slice,
                       "k_range": [min(k0, k1), max(k0, k1)],
                       "idents": g["idents"][max(0, min(k0, k1) - hw): max(k0, k1) + hw + 1],
                       "channels": g["channels"][max(0, min(k0, k1) - hw): max(k0, k1) + hw + 1],
                       "segments": g["segments"][max(0, min(k0, k1) - hw): max(k0, k1) + hw + 1],
                       "k_first": max(0, min(k0, k1) - hw)})
        truth.append({"id": tid, "y0_mm": float(yz[0]), "z0_mm": float(yz[1]), "y1_mm": float(yz1[0]), "z1_mm": float(yz1[1]),
                      "x_start_mm": float(x_start), "x_end_mm": float(x_end), "planes": tp})
    hdr = {"det": a.det, "wires": os.path.basename(wires_path), "cfg": os.path.abspath(a.cfg), "anode": a.anode,
           "face_index": fi, "face_ident": faces[fidx]["ident"], "seed": a.seed, "n": len(pts),
           "pitch_mm": [g["pitch"] for g in pg], "x_plane_mm": [g["x"] for g in pg],
           "nwires": [len(g["idents"]) for g in pg],
           "pdir": [g["pdir"].tolist() for g in pg], "wdir": [g["wdir"].tolist() for g in pg],
           "xregion_mm": xr, "drift_sign": sgn, "x_start_mm": float(x_start), "x_end_mm": float(x_end),
           "tan_theta": a.tan_theta, "tdir_yz": tdir.tolist(), "slope_wire_per_mm": slope,
           "advance_wire_per_slice": [abs(sl) * v_tick * a.ticks_per_slice for sl in slope], "ticks_per_slice": a.ticks_per_slice,
           "min_sep_wires": a.min_sep, "hw": hw,
           "lar": {"DL_mm2_ns": drifter["DL"], "DT_mm2_ns": drifter["DT"], "drift_speed_mm_ns": drifter["drift_speed"],
                   "lifetime_ns": drifter["lifetime"], "DL_cm2s": drifter["DL"] * 1e7, "DT_cm2s": drifter["DT"] * 1e7,
                   "drift_speed_mm_us": drifter["drift_speed"] * 1e3},
           "tick_ns": tick, "ticks_per_mm": 1.0 / (drifter["drift_speed"] * tick),
           "charge_e_per_mm": -a.charge * 10.0 if a.charge <= 0 else None,
           "tracks": truth}
    os.makedirs(a.outdir, exist_ok=True)
    tag = a.tag
    base = "%s_a%d%s" % (a.det, a.anode, ("_" + tag) if tag else "")
    json.dump(tracks, open(os.path.join(a.outdir, "tracks_%s.json" % base), "w"))
    json.dump(hdr, open(os.path.join(a.outdir, "truth_%s.json" % base), "w"), indent=1)
    print("%s: anode %d face %d (ident %d) xregion %s -> x %.1f..%.1f mm; pitch %s; %d tracks" %
          (a.det, a.anode, fi, faces[fidx]["ident"], xr, x_start, x_end, ["%.4f" % g["pitch"] for g in pg], len(pts)))
    print("  tan_theta %.2f dir %s: advance per slice U/V/W = %s wires" % (a.tan_theta, a.dir, ["%.3f" % x for x in hdr["advance_wire_per_slice"]]))
    for t in truth:
        print("  t%02d y0=%7.1f z0=%7.1f -> z1/y1 %7.1f  " % (t["id"], t["y0_mm"], t["z0_mm"], t["z1_mm"] if a.dir == "z" else t["y1_mm"]) +
              "  ".join("%s w %7.2f->%7.2f ch%d..%d" % ("UVW"[p["plane"]], p["w0"], p["w1"], p["channels"][0], p["channels"][-1]) for p in t["planes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
