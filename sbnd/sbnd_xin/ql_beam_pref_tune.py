#!/usr/bin/env python3
"""Grade the beam-window flash match on the reco1 48-evt nueCC sample (beam_pref tuning).

Every event in extracted-2025fall-48evt-fsprod carries a neutrino, so the
in-beam flash (0.2 < t_corrected < 2.2 us) should end up matched to substantial,
light-consistent charge.  For each work root (one per beam_pref operating
point) this reads the calib dumps and reports, per event and aggregated:

  matched    beam flash has >=1 auto_selected cluster
  npair      number of clusters on the beam flash
  pred/meas  sum of matched bundles' total_pred_light / flash total_PE
             (over-collection shows as >>1, orphaning as ~0)
  best ks / chi2ndf   of the beam flash's selected bundles
  moved      pairs that differ from the first (baseline) root

Grade per event: GOOD  matched and 0.33 <= pred/meas <= 3
                 WEAK  matched but budget outside that band
                 ORPHAN no cluster matched to the beam flash

Usage:
  ./ql_beam_pref_tune.py work-fsprod-bpv-off work-fsprod-bpv-w050 [...]
Repro (doc 22): roots produced by
  SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod \
  SBND_WORK_ROOT=$PWD/work-fsprod-bpv-<cfg> [BEAMPREF_WEIGHT=<w>] \
  SBND_MAX_JOBS=6 ./run_ql_evt.sh data all -calib [-beam-pref]
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TLOW_US, THIGH_US = 0.2, 2.2


def read_event(cpath):
    c = json.load(open(cpath))
    beam = [f for f in c["flashes"] if TLOW_US < f["time"] < THIGH_US]
    out = []
    sel_all = {(b["flash_gid"], b["main_cluster"])
               for b in c["bundles"] if b["auto_selected"]}
    for f in beam:
        bs = [b for b in c["bundles"]
              if b["auto_selected"] and b["flash_gid"] == f["gid"]]
        pred = sum(b["total_pred_light"] for b in bs)
        out.append(dict(
            gid=f["gid"], time=f["time"], meas=f["total_PE"],
            npair=len(bs), pred=pred,
            ratio=(pred / f["total_PE"] if f["total_PE"] > 0 else 0.0),
            best_ks=min((b["ks_dis"] for b in bs), default=None),
            best_c2n=min((b["chi2"] / max(b["ndf"], 1) for b in bs), default=None),
            clusters=sorted(b["main_cluster"] for b in bs)))
    return out, sel_all


def collect(root):
    evs = {}
    for p in sorted(glob.glob(os.path.join(root, "ql_evt*", "calib-evt*.json"))):
        ev = int(re.search(r"calib-evt(\d+)\.json", p).group(1))
        evs[ev] = read_event(p)
    return evs


def grade(flashes):
    """Grade the event by its dominant (max-PE) beam flash."""
    if not flashes:
        return "NOBEAM", None
    f = max(flashes, key=lambda x: x["meas"])
    if f["npair"] == 0:
        return "ORPHAN", f
    if 0.33 <= f["ratio"] <= 3.0:
        return "GOOD", f
    return "WEAK", f


def main():
    roots = sys.argv[1:]
    if not roots:
        print(__doc__)
        sys.exit(1)
    data = {r: collect(os.path.join(HERE, r)) for r in roots}
    base = roots[0]
    print("%-24s %6s %6s %6s %6s | med ratio | moved pairs vs %s"
          % ("root", "GOOD", "WEAK", "ORPHAN", "NOBEAM", base))
    for r in roots:
        counts = dict(GOOD=0, WEAK=0, ORPHAN=0, NOBEAM=0)
        ratios = []
        moved = 0
        for ev, (flashes, sel) in sorted(data[r].items()):
            g, f = grade(flashes)
            counts[g] += 1
            if f and f["npair"]:
                ratios.append(f["ratio"])
            if r != base and ev in data[base]:
                moved += len(sel ^ data[base][ev][1])
        ratios.sort()
        med = ratios[len(ratios) // 2] if ratios else float("nan")
        print("%-24s %6d %6d %6d %6d |   %.2f    | %s"
              % (r, counts["GOOD"], counts["WEAK"], counts["ORPHAN"],
                 counts["NOBEAM"], med, "-" if r == base else moved))
    if os.environ.get("BPTUNE_DETAIL"):
        evs = sorted(data[base])
        hdr = "  ".join("%-18s" % r.replace("work-fsprod-bpv-", "") for r in roots)
        print("\nevt        " + hdr)
        for ev in evs:
            row = []
            for r in roots:
                fl = data[r].get(ev, ([], set()))[0]
                g, f = grade(fl)
                row.append("%-18s" % (
                    "%s n=%d r=%.2f" % (g[0], f["npair"], f["ratio"]) if f else g))
            print("evt%-8d %s" % (ev, "  ".join(row)))


if __name__ == "__main__":
    main()
