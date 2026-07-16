#!/usr/bin/env python3
"""18-event before/after census for the anode containment slack (anode_ext1_margin).

Read-only.  Compares each event's canonical `_spcov` Q-L calib dump (anode floor
anode_ext1 - 1.0 = -3.0 cm, the pre-2026-07-16 hard-coded slack) against the `_am2`
variant (anode_ext1_margin = 2.0 cm => floor -4.0 cm, the new PDVD production
default).  Everything else -- the 2 cm anode pull, cathode_ext1 = 2.0 cm, the
saturation/coverage flags, the drift speeds, the `_spcov` light archives -- is held
identical, so the only moving part is the anode floor.

For every cluster it reads QLMatching's own final selection (each bundle's
auto_selected flag, the field the ql_scan viewer renders) and reports per event and
in aggregate:

  n_sel        auto_selected bundles (matched cluster<->flash pairs)
  n_moved      clusters whose auto_selected FLASH changed off->on
  n_bright     of those, how many moved to a BRIGHTER flash (higher total_PE)
  n_gained     clusters matched only in `_am2` (unmatched before)
  n_lost       clusters matched only in `_spcov` (unmatched after) -- watch this
  n_cont       candidate (flash,cluster) bundles marked contained; the widened
               floor can only ADD, so cont_on >= cont_off is an invariant
  orphan_PE    brightest flash carrying zero auto_selected clusters (the evt298567
               symptom: a bright flash left with nothing)

Flashes are keyed by TIME, not gid: gid can renumber cosmetically between runs
(see project_ql_matching_pointer_nondeterminism), which would fake "moved" counts.

This prices the change: the floor moves for EVERY PDVD cluster, so a single-event
case-check cannot see an over-admission regression.  n_lost > 0 or a large n_moved
that is NOT toward brighter flashes is the signal to stop and look.

Repro:
    cd wcp-porting-img/pdvd
    python3 docs/qlport/scripts/compare_am2.py
"""
import glob
import json

OFF_GLOB = "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd/work/039252_%d_spcov/calib-evt*.json"
ON_GLOB = "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd/work/039252_%d_am2/calib-evt*.json"
NIDX = 18
TKEY = 2  # decimals for the time key (ns-level jitter must not break matching)


def load(path):
    d = json.load(open(path))
    fl = {f["gid"]: f for f in d["flashes"]}
    sel = {}       # cluster uid -> (time_key, total_PE)
    ncont = 0
    for b in d["bundles"]:
        if b.get("contained"):
            ncont += 1
        if not b.get("auto_selected"):
            continue
        f = fl[b["flash_gid"]]
        key = round(f["time"], TKEY)
        for c in [b["main_cluster"]] + b.get("other_clusters", []):
            sel[c] = (key, f["total_PE"])
    # brightest flash with no auto_selected cluster
    used = {v[0] for v in sel.values()}
    orphans = [f["total_PE"] for f in d["flashes"] if round(f["time"], TKEY) not in used]
    return sel, ncont, (max(orphans) if orphans else 0.0), len(d["flashes"])


def main():
    hdr = ("%-3s %-8s  %5s %5s  %5s %6s  %5s %5s  %7s %7s  %9s %9s" %
           ("idx", "event", "sel0", "sel1", "moved", "bright", "gain", "lost",
            "cont0", "cont1", "orphan0", "orphan1"))
    print(hdr)
    print("-" * len(hdr))
    T = dict(sel0=0, sel1=0, moved=0, bright=0, gain=0, lost=0, cont0=0, cont1=0)
    rows = 0
    for i in range(NIDX):
        offp = sorted(glob.glob(OFF_GLOB % i))
        onp = sorted(glob.glob(ON_GLOB % i))
        if not offp or not onp:
            print("%-3d %-8s  (missing dump: off=%d on=%d)" % (i, "?", len(offp), len(onp)))
            continue
        evt = offp[0].split("calib-evt")[1].split(".json")[0]
        s0, c0, o0, _ = load(offp[0])
        s1, c1, o1, _ = load(onp[0])
        moved = [u for u in set(s0) & set(s1) if s0[u][0] != s1[u][0]]
        bright = [u for u in moved if s1[u][1] > s0[u][1]]
        gain = set(s1) - set(s0)
        lost = set(s0) - set(s1)
        print("%-3d %-8s  %5d %5d  %5d %6d  %5d %5d  %7d %7d  %9.0f %9.0f" %
              (i, evt, len(s0), len(s1), len(moved), len(bright),
               len(gain), len(lost), c0, c1, o0, o1))
        T["sel0"] += len(s0); T["sel1"] += len(s1)
        T["moved"] += len(moved); T["bright"] += len(bright)
        T["gain"] += len(gain); T["lost"] += len(lost)
        T["cont0"] += c0; T["cont1"] += c1
        rows += 1
    print("-" * len(hdr))
    print("%-3s %-8s  %5d %5d  %5d %6d  %5d %5d  %7d %7d" %
          ("ALL", "%d evts" % rows, T["sel0"], T["sel1"], T["moved"], T["bright"],
           T["gain"], T["lost"], T["cont0"], T["cont1"]))
    print()
    print("contained bundles: %+d (%+.1f%%)  -- widening the floor can only ADD; a "
          "negative value means something else moved"
          % (T["cont1"] - T["cont0"],
             100.0 * (T["cont1"] - T["cont0"]) / T["cont0"] if T["cont0"] else 0.0))
    print("matched clusters : %+d   moved: %d (of which %d to a brighter flash)"
          % (T["sel1"] - T["sel0"], T["moved"], T["bright"]))
    if T["lost"]:
        print("WARNING: %d cluster(s) matched before and unmatched after -- inspect."
              % T["lost"])


if __name__ == "__main__":
    main()
