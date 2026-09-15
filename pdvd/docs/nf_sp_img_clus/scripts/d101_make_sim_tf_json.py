#!/usr/bin/env python3
"""doc pdvd/101 -- sim-matched TrackFitting JSON for the light-less single-muon simulation.

The production PDHD/PDVD track-fitting JSONs carry EFFECTIVE transverse widths measured on
DATA (doc pdvd/44, pdhd/02) and physical DL/DT at the data field.  The simulation drifts with
the transport in its own compiled config and its SP leaves a smaller effective width (doc
pdvd/47 sec 12, S1 production sim: PDVD 1.29/1.27/0.54 mm, PDHD 2.69/2.79/0.39 mm).  Fitting
simulated muons with the data JSON would make the truth residual measure a model mismatch, not
the sampling / Steiner / fit chain, so the sim arm uses a copy with only these keys replaced:

  DL, DT                         <- the COMPILED sim config's lar.DL / lar.DT (pass them in)
  ind_sigma_u_T, ind_sigma_v_T, col_sigma_w_T  <- doc pdvd/47 sim c (mm)
  add_sigma_L                    <- 1/(2 pi Gaus_wide) * the sim drift speed (Gaus_wide 0.12 MHz)

Every other key is copied verbatim.  The production files are only read.

Usage:
  d101_make_sim_tf_json.py --det pdvd --DL 4.1307 --DT 7.9135 --drift 1.568 --out FILE
(DL/DT in cm^2/s, drift in mm/us, exactly as read from the compiled sim config.)
"""
import argparse, json, math

TK = "/home/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment"
SRC = {"pdvd": TK + "/protodunevd/pdvd_track_fitting.json",
       "pdhd": TK + "/pdhd/pdhd_track_fitting.json"}
SIM_C = {"pdvd": (1.29, 1.27, 0.54), "pdhd": (2.69, 2.79, 0.39)}   # mm, doc pdvd/47 sec 12 (S1)
GAUS_WIDE_MHZ = 0.12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=SRC)
    ap.add_argument("--DL", type=float, required=True, help="cm^2/s from the compiled sim config")
    ap.add_argument("--DT", type=float, required=True, help="cm^2/s from the compiled sim config")
    ap.add_argument("--drift", type=float, required=True, help="mm/us from the compiled sim config")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(SRC[a.det]))
    c = SIM_C[a.det]
    add_sigma_L = 1.0 / (2 * math.pi * GAUS_WIDE_MHZ) * a.drift       # us * mm/us = mm
    new = dict(d)
    new["_comment_d101_sim"] = (
        "doc pdvd/101 SIM-MATCHED copy of %s: DL=%.4f DT=%.4f cm^2/s and add_sigma_L=%.4f mm from the "
        "compiled sim config (drift %.5f mm/us), ind_sigma_u_T/ind_sigma_v_T/col_sigma_w_T = %s mm = doc "
        "pdvd/47 sec 12 sim S1 effective width.  All other keys verbatim.  NOT for data."
        % (SRC[a.det], a.DL, a.DT, add_sigma_L, a.drift, c))
    new["DL"] = a.DL * 1e-7
    new["DT"] = a.DT * 1e-7
    new["ind_sigma_u_T"], new["ind_sigma_v_T"], new["col_sigma_w_T"] = c
    new["add_sigma_L"] = add_sigma_L
    json.dump(new, open(a.out, "w"), indent=4)
    changed = [k for k in new if not k.startswith("_") and d.get(k) != new[k]]
    print("wrote", a.out, "changed keys:", changed)


if __name__ == "__main__":
    main()
