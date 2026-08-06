#!/usr/bin/env python3
"""doc pr/40 G2/G4: per-segment pdg/flag_shower comparison, off vs on.

Usage: pr40_seg_pid.py <off_arm_dir> <on_arm_dir>
"""
import json, math, sys, os, glob

CASES = [(388, 23020), (74544, 12018), (174637, 9050), (256587, 11079),
         (267597, 5001), (269774, 13034), (423981, 12013), (433451, 4078),
         (489330, 4018)]

MIP = 56000.0  # e/cm, wct-pr-perevt.jsonnet mip_dqdx (production SBND scale)


def load_seg(arm, evt, sid):
    f = os.path.join(arm, f"pr_evt{evt}", f"calib-pr-evt{evt}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    for s in d["segments"]:
        if s["id"] == sid:
            return s
    return None


def med_dqdx_mip(seg):
    pts = seg["points"]
    dq = sorted(p["dQ"] / p["dx"] for p in pts if p["dx"] > 0)
    if not dq:
        return 0.0
    return dq[len(dq) // 2] / MIP


def main():
    off, on = sys.argv[1], sys.argv[2]
    print(f"{'evt':>7} {'seg':>6} {'off_pdg':>8} {'on_pdg':>7} {'off_fs':>7} {'on_fs':>6} {'medMIP':>7}  moved?")
    for ev, sid in CASES:
        so = load_seg(off, ev, sid)
        sn = load_seg(on, ev, sid)
        if so is None or sn is None:
            print(f"{ev:>7} {sid:>6}  MISSING (off={so is not None} on={sn is not None})")
            continue
        moved = (so["particle_id"] != sn["particle_id"]) or (so["flag_shower"] != sn["flag_shower"])
        print(f"{ev:>7} {sid:>6} {so['particle_id']:>8} {sn['particle_id']:>7} "
              f"{str(so['flag_shower']):>7} {str(sn['flag_shower']):>6} {med_dqdx_mip(so):>7.2f}  {'YES' if moved else 'no'}")

    # G4 census: all pdg==11 && !flag_shower && median>1.75 MIP && L>2cm segments
    # in the OFF arm, and how they moved.
    print()
    print("=== G4 census (off-arm: pdg11 & !flag_shower & med>1.75xMIP & L>2cm) ===")
    n_total = 0
    n_moved = 0
    moved_to = {}
    for f in sorted(glob.glob(os.path.join(off, "pr_evt*", "calib-pr-evt*.json"))):
        evt = int(os.path.basename(f).split("evt")[-1].split(".")[0])
        d = json.load(open(f))
        for s in d["segments"]:
            if not s["is_main_cluster"]:
                continue
            if s["particle_id"] != 11 or s["flag_shower"]:
                continue
            pts = s["points"]
            if len(pts) < 2:
                continue
            L = sum(math.dist((pts[i]["x"], pts[i]["y"], pts[i]["z"]),
                               (pts[i + 1]["x"], pts[i + 1]["y"], pts[i + 1]["z"]))
                    for i in range(len(pts) - 1))
            if L <= 2.0:
                continue
            if med_dqdx_mip(s) <= 1.75:
                continue
            n_total += 1
            sn = load_seg(on, evt, s["id"])
            if sn is None:
                continue
            if sn["particle_id"] != s["particle_id"] or sn["flag_shower"] != s["flag_shower"]:
                n_moved += 1
                key = (sn["particle_id"], sn["flag_shower"])
                moved_to[key] = moved_to.get(key, 0) + 1
    print(f"total in class: {n_total}   moved: {n_moved}")
    for k, v in sorted(moved_to.items(), key=lambda kv: -kv[1]):
        print(f"  -> pdg={k[0]} flag_shower={k[1]}: {v}")


if __name__ == "__main__":
    main()
