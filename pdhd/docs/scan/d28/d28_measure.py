#!/usr/bin/env python3
"""doc pdhd/28 sec 3 -- what the wire lookup does, against preregistered.txt K3 / K4 / P1.

    STM_SCAN_RECORD=<smx27 record> python3 d28_measure.py [--hd-off h28off --hd-on h28wl --vd-off p97voff --vd-on p97vwl]

K3  per-cell: U/V role-1 rows of T_stm_michel_2d classed by the channel's wires on its APA's active face (the class
    d27_wrap_face.py sec 5 uses), recorded d_stop_cm beyond 100 cm, knob off vs on; michel_q2d_n_rewired by APA side;
    candidates whose control has no U and no V cell.
K4  PDHD hand Michel (APA0 strict, smx27) pooled medians of michel_ke_q2d_region / _ctl against doc 27's offline bounds,
    and per is_stm & michel_found candidate against [fixlo - 2, fix + 2] MeV (d27/ghost_twin.json).  The knob-off arm's
    values must equal ghost_twin.json's rec_* (the offline twin of production) -- a consistency check of the join.
P1  PDVD: candidates whose region or control moved, and hand Michel medians off vs on.
"""
import argparse, bz2, collections, glob, json, os, sys
import numpy as np, uproot
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM
ap = argparse.ArgumentParser()
ap.add_argument("--hd-off", default="h28off"); ap.add_argument("--hd-on", default="h28wl")
ap.add_argument("--vd-off", default="p97voff"); ap.add_argument("--vd-on", default="p97vwl")
a = ap.parse_args()
GT = {x["key"]: x for x in json.load(open(IMG + "/pdhd/docs/scan/d27/ghost_twin.json"))}

g = json.load(bz2.open("/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunehd-wires-larsoft-v1.json.bz2"))["Store"]
c2w = collections.defaultdict(list); NW = {}
for an in g["anodes"]:
    an = an["Anode"]
    for fi in an["faces"]:
        f = g["faces"][fi]["Face"]
        for pi in f["planes"]:
            pl = g["planes"][pi]["Plane"]; NW[(an["ident"], f["ident"], pl["ident"])] = len(pl["wires"])
            for idx, wi in enumerate(pl["wires"]):
                c2w[(an["ident"], g["wires"][wi]["Wire"]["channel"])].append((f["ident"], pl["ident"], idx))
ACT = {0: 0, 1: 1, 2: 0, 3: 1}      # d27_wrap_face.py sec 5, from the W cells


def arm_rows(det, arm):
    T = {}; C = {}
    for fn in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/tracking-pr.root")):
        ev = os.path.basename(os.path.dirname(fn))[: -len(arm) - 1]
        u = uproot.open(fn)
        if "T_stm_michel" not in u: continue
        t = u["T_stm_michel"].arrays(library="np")
        for i in range(len(t["cluster_id"])):
            T[f"{ev}/{int(t['cluster_id'][i])}"] = {k: t[k][i] for k in t}
        if "T_stm_michel_2d" in u:
            C[ev] = u["T_stm_michel_2d"].arrays(["cluster_id", "apa", "plane", "channel", "face", "wire", "role", "d_stop_cm"], library="np")
    return T, C


def exists(det, arm):
    return bool(glob.glob(f"{IMG}/{det}/work/*_{arm}/tracking-pr.root"))


print("=== K3 (PDHD): U/V role-1 cells beyond 100 cm, by the channel's active-face wires")
for arm in (a.hd_off, a.hd_on):
    if not exists("pdhd", arm): print(f"  {arm}: not on disk"); continue
    T, C = arm_rows("pdhd", arm)
    cls = collections.defaultdict(list)
    for ev, c in C.items():
        s = (c["plane"] < 2) & (c["role"] == 1)
        for apa, pl, ch, fa, wi, d in zip(*(c[k][s].tolist() for k in ("apa", "plane", "channel", "face", "wire", "d_stop_cm"))):
            act = [x for x in c2w[(apa, ch)] if x[0] == ACT[apa] and x[1] == pl]
            k = "channel has 1 active-face wire" if len(act) == 1 else "channel has 2 active-face wires"
            k += ", recorded on the %s face" % ("active" if fa == ACT[apa] else "OTHER")
            cls[k].append(d)
    print(f"  [{arm}]")
    for k in sorted(cls):
        v = np.array(cls[k]); print(f"    {k:62s} n {len(v):6d}  p50 {np.median(v):6.1f}  beyond 100 cm {(v > 100).mean():.3f}")
    sm = [r for r in T.values() if int(r["is_stm"]) == 1 and int(r["michel_found"]) == 1]
    nouv = sum(1 for r in T.values() if int(r.get("michel_q2d_ctl_n_u", 0)) == 0 and int(r.get("michel_q2d_ctl_n_v", 0)) == 0)
    print(f"    candidates {len(T)}; control with no U and no V cell {nouv}")
    if "michel_q2d_n_rewired" in next(iter(T.values())):
        side = collections.defaultdict(list)
        for key, r in T.items():
            ev = key.rsplit("/", 1)[0]; c = C.get(ev)
            if c is None: continue
            m = (c["cluster_id"] == int(key.rsplit("/", 1)[1])) & (c["plane"] == 2) & (c["role"] == 1)
            if not m.any(): continue
            apa = collections.Counter(c["apa"][m].tolist()).most_common(1)[0][0]
            side["APA%d" % apa].append(int(r["michel_q2d_n_rewired"]))
        for k in sorted(side):
            v = np.array(side[k]); print(f"    michel_q2d_n_rewired {k}: candidates {len(v)}, > 0 on {(v > 0).sum()}, p50 {np.median(v):.0f}")

print("\n=== K4 (PDHD): against doc 27's offline bounds")
if exists("pdhd", a.hd_off) and exists("pdhd", a.hd_on):
    Toff, _ = arm_rows("pdhd", a.hd_off); Ton, _ = arm_rows("pdhd", a.hd_on)
    # the join: the knob-off arm is production, so it must equal the offline twin's recorded values
    dev = max(max(abs(float(Toff[k]["michel_ke_q2d_region"]) - x["rec_stop"]), abs(float(Toff[k]["michel_ke_q2d_ctl"]) - x["rec_ctl"]))
              for k, x in GT.items() if k in Toff)
    print(f"  join check: {sum(k in Toff for k in GT)} / {len(GT)} ghost_twin items found in {a.hd_off}; max |chain - rec| {dev:.2e} MeV")
    its, _ = BM.items("pdhd", a.hd_on, "strict"); J = {x[0]: x for x in its}
    hm = [k for k in Ton if k in J and J[k][1] in BM.STOP and J[k][2] in BM.MICHEL_KINDS
          and int(Ton[k]["is_stm"]) == 1 and int(Ton[k]["michel_found"]) == 1]
    for est, lo, hi, nm in (("michel_ke_q2d_region", 38.9, 44.0, "region"), ("michel_ke_q2d_ctl", 7.8, 10.2, "control")):
        off = np.median([float(Toff[k][est]) for k in hm]); on = np.median([float(Ton[k][est]) for k in hm])
        ok = lo - 1 <= on <= hi + 1
        print(f"  hand Michel (n {len(hm)}) {nm}: off {off:.1f} -> on {on:.1f} MeV | registered bound [{lo}, {hi}] +-1 -> {'HELD' if ok else 'MISSED'}")
    inb = tot = 0; outs = []
    for k, x in GT.items():
        if k not in Ton or not x["corrected_ok"]: continue
        for est, fx, fl in (("michel_ke_q2d_region", "fix_stop", "fixlo_stop"), ("michel_ke_q2d_ctl", "fix_ctl", "fixlo_ctl")):
            v = float(Ton[k][est]); lo_, hi_ = min(x[fx], x[fl]) - 2, max(x[fx], x[fl]) + 2
            tot += 1
            if lo_ <= v <= hi_: inb += 1
            else: outs.append((k, est.replace("michel_ke_q2d_", ""), round(v, 1), round(x[fl], 1), round(x[fx], 1)))
    print(f"  per candidate (both estimators): inside [fixlo-2, fix+2] on {inb} / {tot} = {inb / max(1, tot):.3f} "
          f"(registered >= 0.80 -> {'HELD' if inb / max(1, tot) >= 0.8 else 'MISSED'})")
    for o in outs[:15]: print("    outside: %s %s C++ %.1f bounds %.1f..%.1f" % o)
    ch = [(k, float(Ton[k]["michel_ke_q2d_region"]) - float(Toff[k]["michel_ke_q2d_region"])) for k in hm]
    d = np.array([x[1] for x in ch])
    print(f"  hand Michel region shift on - off: p10/p50/p90 {np.percentile(d, 10):+.1f}/{np.median(d):+.1f}/{np.percentile(d, 90):+.1f}; above 52.8 MeV off {sum(float(Toff[k]['michel_ke_q2d_region']) > 52.83 for k in hm)} on {sum(float(Ton[k]['michel_ke_q2d_region']) > 52.83 for k in hm)}")
else:
    print("  arms not on disk")

print("\n=== P1 (PDVD): how much the lookup moves")
if exists("pdvd", a.vd_off) and exists("pdvd", a.vd_on):
    Toff, _ = arm_rows("pdvd", a.vd_off); Ton, _ = arm_rows("pdvd", a.vd_on)
    mv = [k for k in Toff if k in Ton and (abs(float(Ton[k]["michel_ke_q2d_region"]) - float(Toff[k]["michel_ke_q2d_region"])) > 1e-9
                                         or abs(float(Ton[k]["michel_ke_q2d_ctl"]) - float(Toff[k]["michel_ke_q2d_ctl"])) > 1e-9)]
    print(f"  candidates {len(Toff)}; region or control moved on {len(mv)} ({len(mv) / max(1, len(Toff)):.3f}); registered < 0.05")
    nr = np.array([int(r["michel_q2d_n_rewired"]) for r in Ton.values()])
    print(f"  michel_q2d_n_rewired > 0 on {(nr > 0).sum()}, max {nr.max()}")
    its, _ = BM.items("pdvd", a.vd_on, "all"); J = {x[0]: x for x in its}
    hm = [k for k in Ton if k in J and J[k][1] in BM.STOP and J[k][2] in BM.MICHEL_KINDS and int(Ton[k]["is_stm"]) == 1 and int(Ton[k]["michel_found"]) == 1]
    for est in ("michel_ke_q2d_region", "michel_ke_q2d_ctl"):
        print(f"  hand Michel (n {len(hm)}) {est}: off {np.median([float(Toff[k][est]) for k in hm]):.2f} -> on {np.median([float(Ton[k][est]) for k in hm]):.2f}")
    for k in mv[:10]:
        print(f"    moved {k}: region {float(Toff[k]['michel_ke_q2d_region']):.2f} -> {float(Ton[k]['michel_ke_q2d_region']):.2f}, "
              f"control {float(Toff[k]['michel_ke_q2d_ctl']):.2f} -> {float(Ton[k]['michel_ke_q2d_ctl']):.2f}, rewired {int(Ton[k]['michel_q2d_n_rewired'])}")
else:
    print("  arms not on disk")
