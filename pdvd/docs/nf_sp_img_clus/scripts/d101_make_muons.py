#!/usr/bin/env python3
"""doc pdvd/101 Phase 2 -- known-truth single straight muons for the Steiner-graph +
track-fit trajectory / dQ/dx study at coarse wire pitch (PDVD bottom CRP, PDHD APA1).

One muon per event, L = --length cm (default 100), fully inside ONE face of ONE anode:
  * >= --margin-yz cm (15) from the edges of the region covered by all three wire planes
    of that face (intersection of the U/V/W wire bounding boxes, read from the wires file);
  * >= --margin-x cm (20) from the W wire plane AND from the cathode cut, and in addition
    >= --resp-margin cm (5) past the Drifter response plane (depos inside the response
    plane go through Drifter.cxx's best-effort back-up fudge; d47 started 1 cm past it).
Angle grid (event k = 4*i_theta + i_phi):
  theta = angle to the drift (x) axis in --theta (80,60,40,20 deg),
  phi   = azimuth in the y-z plane measured from the collection (W) wire direction (+y)
          toward +z, in --phi (0,30,60,90 deg).
  direction d = (cos theta, sin theta cos phi, sin theta sin phi); tail = mid - L/2 d,
  head = mid + L/2 d (head has the larger global x).
Midpoint: the (y,z) centre of the face's three-plane region and the drift distance
--x-frac (0.5) of the W-plane -> cathode distance, plus a deterministic offset per event
(numpy default_rng(--seed + k); uniform +-(--off-yz) cm in y,z and +-(--off-x) cm in drift
distance; re-drawn until every margin holds).  Repeats (--repeat "16:0,17:5"): event 16 =
the SAME track as event 0 with a different simulation seed, event 17 = event 5 likewise.

Geometry (wires file loading, per-plane pitch/direction, wire coordinate) is IMPORTED from
d47_make_xtracks.py, not copied.  The drift volume is read from the Drifter xregions of a
COMPILED sim driver config (--cfg), the tick / drift speed / transport from the same file.

Per-plane truth: w = pdir.(p - c0)/pitch at tail and head (wire index along the sorted
pitch direction, d47 convention), wires spanned, ticks spanned L|cos theta|/(v tick), and
advance_wire_per_slice = |dw/ds| * v * tick * 4 (wires per 4-tick slice; inf-like when the
track is perpendicular to the drift) -- so a thin or empty event is attributable.

Usage:
  d101_make_muons.py --det pdvd --cfg probe_pdvd_a1.json --anode 1 --face 0 --run 900101 \
      --out /home/xqian/tmp/d101/sim/truth_pdvd.json
  d101_make_muons.py --det pdhd --cfg probe_pdhd_a1.json --anode 1 --face 1 --run 900102 \
      --out /home/xqian/tmp/d101/sim/truth_pdhd.json
Writes the truth JSON (header + one record per event, each carrying its own 'tracks' TLA
list) and, next to it, tracks_<det>_evt<k>.json (the driver's `tracks` TLA).
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d47_make_xtracks as m  # noqa: E402  (load_geom, plane_geometry, wire_coord, WIRES, DATA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=("pdvd", "pdhd"))
    ap.add_argument("--wires", default=None)
    ap.add_argument("--cfg", required=True, help="compiled sim driver JSON (Drifter xregions, lar, tick)")
    ap.add_argument("--anode", type=int, required=True)
    ap.add_argument("--face", type=int, required=True, help="face index within the anode")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--tag", default="d101", help="work-dir tag: work/<run6>_<k>_<tag>")
    ap.add_argument("--length", type=float, default=100.0, help="cm")
    ap.add_argument("--theta", default="80,60,40,20")
    ap.add_argument("--phi", default="0,30,60,90")
    ap.add_argument("--repeat", default="16:0,17:5", help="k_new:k_src pairs (same track, new sim seed)")
    ap.add_argument("--margin-yz", type=float, default=15.0)
    ap.add_argument("--margin-x", type=float, default=20.0)
    ap.add_argument("--resp-margin", type=float, default=5.0)
    ap.add_argument("--x-frac", type=float, default=0.5)
    ap.add_argument("--off-yz", type=float, default=5.0)
    ap.add_argument("--off-x", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=101, help="placement seed")
    ap.add_argument("--sim-seed-base", type=int, default=None, help="default run*100")
    ap.add_argument("--charge", type=float, default=-500.0, help="electrons per 0.1 mm step")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    wires_path = a.wires or os.path.join(m.DATA, m.WIRES[a.det])
    anodes, faces, planes, wires, P = m.load_geom(wires_path)
    cfg = json.load(open(a.cfg))
    drifter = next(n for n in cfg if n.get("type") == "Drifter")["data"]
    dt_node = next(n for n in cfg if n.get("type") == "DepoTransform")["data"]
    tick = dt_node["tick"]
    v = drifter["drift_speed"]  # mm/ns
    an = next(x for x in anodes if x["ident"] == a.anode)
    fidx = an["faces"][a.face]
    pg = [m.plane_geometry(planes[pi], wires, P) for pi in faces[fidx]["planes"][:3]]
    xw = pg[2]["x"]
    wd = pg[2]["wdir"]
    if abs(wd[1]) < 0.99:
        print("W wires are not along y: wdir=%s" % wd, file=sys.stderr); return 1
    cand = [xr for xr in drifter["xregions"] if abs(xr["anode"] - xw) < 20]
    if len(cand) != 1:
        print("xregion selection ambiguous/empty for W x=%.2f: %s" % (xw, cand), file=sys.stderr); return 1
    xr = cand[0]
    ux = 1.0 if xr["cathode"] > xw else -1.0          # unit x step from the anode toward the cathode
    s_cath = abs(xr["cathode"] - xw)                   # mm, W plane -> cathode cut
    s_resp = abs(xr["response"] - xw)                  # mm, W plane -> response plane
    s_lo = max(a.margin_x * 10, s_resp + a.resp_margin * 10)
    s_hi = s_cath - a.margin_x * 10
    ylo = max(g["bbox_min"][1] for g in pg); yhi = min(g["bbox_max"][1] for g in pg)
    zlo = max(g["bbox_min"][2] for g in pg); zhi = min(g["bbox_max"][2] for g in pg)
    Y0, Y1 = ylo + a.margin_yz * 10, yhi - a.margin_yz * 10
    Z0, Z1 = zlo + a.margin_yz * 10, zhi - a.margin_yz * 10
    yc, zc = 0.5 * (ylo + yhi), 0.5 * (zlo + zhi)
    L = a.length * 10.0
    thetas = [float(t) for t in a.theta.split(",")]
    phis = [float(p) for p in a.phi.split(",")]
    sim_seed_base = a.sim_seed_base if a.sim_seed_base is not None else a.run * 100

    def x_of_s(s):
        return xw + ux * s

    def plane_rows(tail, head, d):
        rows = []
        for pi, g in enumerate(pg):
            w0 = m.wire_coord(g, tail); w1 = m.wire_coord(g, head)
            # wires per mm of drift distance along the track
            dws = abs(float(np.dot(g["pdir"][1:], d[1:]))) / g["pitch"] / max(abs(d[0]), 1e-9)
            rows.append({"plane": "UVW"[pi], "w_tail": w0, "w_head": w1, "wires_spanned": abs(w1 - w0),
                         "advance_wire_per_slice": dws * v * tick * 4})
        return rows

    events = []
    base = {}
    for it, th in enumerate(thetas):
        for ip, ph in enumerate(phis):
            k = it * len(phis) + ip
            t, p = math.radians(th), math.radians(ph)
            d = np.array([math.cos(t), math.sin(t) * math.cos(p), math.sin(t) * math.sin(p)])
            rng = np.random.default_rng(a.seed + k)
            ok = False
            for attempt in range(2000):
                dy, dz = (rng.random(2) * 2 - 1) * a.off_yz * 10
                ds = (rng.random() * 2 - 1) * a.off_x * 10
                mid = np.array([x_of_s(a.x_frac * s_cath + ds), yc + dy, zc + dz])
                tail = mid - 0.5 * L * d; head = mid + 0.5 * L * d
                s_t, s_h = abs(tail[0] - xw), abs(head[0] - xw)
                ys = (tail[1], head[1]); zs = (tail[2], head[2])
                if (min(s_t, s_h) >= s_lo and max(s_t, s_h) <= s_hi and min(ys) >= Y0 and max(ys) <= Y1
                        and min(zs) >= Z0 and max(zs) <= Z1 and ux * (tail[0] - xw) > 0 and ux * (head[0] - xw) > 0):
                    ok = True
                    break
            if not ok:
                print("event %d (theta %g phi %g): no placement satisfies the margins" % (k, th, ph), file=sys.stderr)
                return 1
            base[k] = (tail, head, d, mid, th, ph, attempt)
    nbase = len(thetas) * len(phis)
    order = [(k, None) for k in range(nbase)]
    for pair in [x for x in a.repeat.split(",") if x]:
        kn, ks = [int(x) for x in pair.split(":")]
        order.append((kn, ks))
    for k, src in order:
        tail, head, d, mid, th, ph, attempt = base[src if src is not None else k]
        seed = sim_seed_base + k + (1000 if src is not None else 0)
        s_t, s_h = abs(tail[0] - xw), abs(head[0] - xw)
        tr = [{"tail": [float(x) / 10 for x in tail], "head": [float(x) / 10 for x in head], "charge": a.charge}]
        events.append({
            "k": k, "repeat_of": src, "theta_deg": th, "phi_deg": ph, "sim_seed": seed,
            "placement_attempt": attempt,
            "tail_cm": tr[0]["tail"], "head_cm": tr[0]["head"], "mid_cm": [float(x) / 10 for x in mid],
            "dir": d.tolist(), "length_cm": a.length, "charge_e_per_0p1mm": a.charge,
            "margins_cm": {"y_lo": (min(tail[1], head[1]) - ylo) / 10, "y_hi": (yhi - max(tail[1], head[1])) / 10,
                           "z_lo": (min(tail[2], head[2]) - zlo) / 10, "z_hi": (zhi - max(tail[2], head[2])) / 10,
                           "x_to_Wplane_min": min(s_t, s_h) / 10, "x_past_response_min": (min(s_t, s_h) - s_resp) / 10,
                           "x_to_cathode_min": (s_cath - max(s_t, s_h)) / 10},
            "drift_distance_cm": [s_t / 10, s_h / 10],
            "ticks_spanned": abs(s_h - s_t) / (v * tick),
            "planes": plane_rows(tail, head, d),
            "tracks": tr,
            "workdir": "%s/work/%06d_%d_%s" % (a.det, a.run, k, a.tag),
        })
    hdr = {
        "doc": "pdvd/101 phase 2", "det": a.det, "run": a.run, "tag": a.tag,
        "wires": os.path.basename(wires_path), "cfg": os.path.abspath(a.cfg),
        "anode": a.anode, "face_index": a.face, "face_ident": faces[fidx]["ident"],
        "xregion_mm": xr, "W_plane_x_mm": xw, "plane_x_mm": [g["x"] for g in pg],
        "pitch_mm": [g["pitch"] for g in pg], "wdir": [g["wdir"].tolist() for g in pg],
        "pdir": [g["pdir"].tolist() for g in pg], "nwires": [len(g["idents"]) for g in pg],
        "face_region_mm": {"y": [ylo, yhi], "z": [zlo, zhi]},
        "allowed_region_mm": {"y": [Y0, Y1], "z": [Z0, Z1], "drift_distance_from_W": [s_lo, s_hi]},
        "ux_anode_to_cathode": ux, "W_to_response_mm": s_resp, "W_to_cathode_mm": s_cath,
        "angle_convention": "d=(cos th, sin th cos ph, sin th sin ph); theta to +x (drift axis), "
                            "phi from +y (W wire direction) toward +z; head = mid + L/2 d",
        "placement": {"seed": a.seed, "rng": "numpy default_rng(seed+k)", "x_frac": a.x_frac,
                      "off_yz_cm": a.off_yz, "off_x_cm": a.off_x, "margin_yz_cm": a.margin_yz,
                      "margin_x_cm": a.margin_x, "resp_margin_cm": a.resp_margin},
        "sim_seed_rule": "run*100 + k (+1000 for a repeat)",
        "lar_from_cfg": {"DL_cm2_s": drifter["DL"] * 1e7, "DT_cm2_s": drifter["DT"] * 1e7,
                         "drift_speed_mm_us": v * 1e3, "lifetime_ms": drifter["lifetime"] / 1e6,
                         "fluctuate": drifter.get("fluctuate")},
        "tick_ns": tick,
        "events": events,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(hdr, open(a.out, "w"), indent=1)
    for e in events:
        json.dump(e["tracks"], open(os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                                 "tracks_%s_evt%d.json" % (a.det, e["k"])), "w"))
    print("%s anode %d face %d (ident %d): W x %.1f mm, response %.1f mm, cathode %.1f mm; y %.0f..%.0f z %.0f..%.0f mm"
          % (a.det, a.anode, a.face, faces[fidx]["ident"], xw, xr["response"], xr["cathode"], ylo, yhi, zlo, zhi))
    for e in events:
        mg = e["margins_cm"]
        print("  k=%2d th=%2g ph=%2g seed=%d tail=(%.1f,%.1f,%.1f) head=(%.1f,%.1f,%.1f) "
              "min margins y%.1f/%.1f z%.1f/%.1f Wx%.1f cath%.1f  adv/slice U%.2f V%.2f W%.2f%s"
              % (e["k"], e["theta_deg"], e["phi_deg"], e["sim_seed"], *e["tail_cm"], *e["head_cm"],
                 mg["y_lo"], mg["y_hi"], mg["z_lo"], mg["z_hi"], mg["x_to_Wplane_min"], mg["x_to_cathode_min"],
                 *[p["advance_wire_per_slice"] for p in e["planes"]],
                 (" (repeat of %d)" % e["repeat_of"]) if e["repeat_of"] is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
