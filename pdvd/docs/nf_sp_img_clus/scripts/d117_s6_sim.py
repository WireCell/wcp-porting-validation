#!/usr/bin/env python3
"""doc pdvd/117 S6 (simulation half) -- the SAME time-width estimator as d117_s6_data.py on simulated isochronous tracks
at a known D_L; the response of the estimator's slope to D_L converts the data's slope into a D_L.

    python3 d117_s6_sim.py [--arms DL0_s1,DL0_s2,DL4_s1,DL4_s2,DL8_s1,DL8_s2,DL4_nn] > ../../scan/d117/s6/s6_sim.txt

Inputs: /home/xqian/tmp/d117/s6sim/<arm>-anode0-sp.tar.bz2 (frame_gauss, run_s6sim.sh) and truth_s6_a0.json
(d117_s6_tracks.py).  The frame tick of a pulse = drift_time/0.5 us + T_off; T_off is found per arm as the mode of
(peak tick - drift tick) over every run on every track channel, then each (track, channel) is measured with
d117_s6_data.measure() at that expected tick -- same window, same isolation rule, same variance -- and the summary is
d117_s6_data.summarize() (the same binned-median line) per arm.
"""
import argparse, json, sys
import numpy as np
import d98_michel_crops as C
import d117_s6_data as D

O = "/home/xqian/tmp/d117/s6sim"


def runs_peaks(row):
    nz = np.nonzero(row > 0)[0]
    if not len(nz):
        return []
    br = np.nonzero(np.diff(nz) > 1)[0]
    segs = np.split(nz, br + 1)
    return [int(s[np.argmax(row[s])]) for s in segs]


def arm_rows(arm, truth, d=O, anode=0):
    chans, fr, tbin = C.load_gauss(f"{d}/{arm}-anode{anode}-sp.tar.bz2")
    idx = {int(c): i for i, c in enumerate(chans)}
    v = truth["lar"]["drift_speed_mm_us"]
    dts = []
    for tr in truth["tracks"]:
        for wc in tr["w_channels"]:
            i = idx.get(wc["channel"])
            if i is None:
                continue
            te = wc["drift_cm"] / (v * 0.1) / 0.5
            dts += [p + tbin - te for p in runs_peaks(fr[i])]
    h, e = np.histogram(dts, bins=np.arange(-2000, 2000, 2))
    toff = float(e[np.argmax(h)] + 1)
    out = []
    for tr in truth["tracks"]:
        for wc in tr["w_channels"][1:-1]:                          # the end wires see a partial track
            i = idx.get(wc["channel"])
            if i is None:
                continue
            te = wc["drift_cm"] / (v * 0.1) / 0.5 + toff - tbin
            m = D.measure(fr[i], int(round(te)) - 5)
            if m is None:
                continue
            out.append(dict(evt=f"{arm}:{tr['id']}", t_us=wc["drift_cm"] / (v * 0.1), drift=wc["drift_cm"],
                            tpw=tr["tpw"], kind=tr["kind"], **m))
    return out, toff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="DL0_s1,DL0_s2,DL4_s1,DL4_s2,DL8_s1,DL8_s2,DL4_nn")
    ap.add_argument("--dir", default=O, help="sim output dir (top CRP: /home/xqian/tmp/d117/s6sim/top)")
    ap.add_argument("--anode", type=int, default=0)
    a = ap.parse_args()
    truth = json.load(open(f"{a.dir}/truth_s6_a{a.anode}.json"))
    print(f"# doc pdvd/117 S6 simulation (d117_s6_sim.py): anode {a.anode} face {truth['face_index']}, {len(truth['tracks'])} tracks, "
          f"v {truth['lar']['drift_speed_mm_us']} mm/us; the data estimator applied unchanged")
    groups = {}
    for arm in a.arms.split(","):
        rows, toff = arm_rows(arm, truth, a.dir, a.anode)
        base = arm.split("_")[0] + ("_nn" if arm.endswith("_nn") else "")
        groups.setdefault(base, []).extend(rows)
        iso = sum(r["iso"] for r in rows)
        print(f"  {arm}: T_off {toff:+.0f} ticks, measured {len(rows)}, isolated {iso}")
    for base, rows in groups.items():
        print(f"\n== {base}")
        D.summarize(rows, label=base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
