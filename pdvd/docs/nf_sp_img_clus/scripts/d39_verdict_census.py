#!/usr/bin/env python3
"""doc pdvd/39 -- compare the cosmic-tagger verdict SETS between two PR arms.

Counts are not enough: TGM +0 on the count is compatible with an arbitrary
number of clusters swapping in and out (feedback_count_vs_set_census).  This
compares the tagged cluster-id SETS per tagger and reports the symmetric
difference, which is what the doc pdvd/39 sec.5 gate is stated in.

Usage:
    d39_verdict_census.py <arm_a_dir> <arm_b_dir> [more dirs ...]

Each dir is a work/<run6>_<evt>[_<tag>]/ holding wct_pr_<run6>_<evt>.log.
The first dir is the reference; every later one is compared against it.

Repro (doc pdvd/39 sec.5):
    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
    docs/nf_sp_img_clus/scripts/d39_verdict_census.py \
        work/039252_2_d39stm2 work/039252_2_d39lean
"""
import re
import sys
import glob
import os

# "visit: TaggerCheckTGM: cluster 1 -> TGM=false"  (arrow is U+2192)
PAT = re.compile(
    r"visit: TaggerCheck(TGM|STM|FC): cluster (\d+) \S+ "
    r"(?:TGM|STM|FC)=(true|false|1|0)")
# TaggerCheckSTM logs both verdicts on one line: "-> STM=1 TGM=0"
PAT_STM = re.compile(r"visit: TaggerCheckSTM: cluster (\d+) \S+ STM=([01]) TGM=([01])")
# CreateSteinerGraph's skip accounting, when skip_flags is on.
PAT_SKIP = re.compile(
    r"CreateSteinerGraph: skip_flags \[([^\]]*)\]: skipped (\d+) cluster\(s\), (\d+) remain")
PAT_MAINS = re.compile(r"visit: TaggerCheck(?:TGM|FC): beam_window_only "
                       r"\[([-0-9.]+), ([-0-9.]+)\) us: (\d+) main\(s\) evaluated, "
                       r"(\d+) out of window")


def read_arm(d):
    """Return {tagger: set(cluster_id)} plus bookkeeping for one arm dir."""
    logs = glob.glob(os.path.join(d, "wct_pr_*.log"))
    if not logs:
        raise SystemExit("no wct_pr_*.log in %s" % d)
    sets = {"TGM": set(), "STM": set(), "FC": set()}
    skips = []
    mains = None
    window = None
    for line in open(logs[0], errors="replace"):
        m = PAT_STM.search(line)
        if m:
            cid = int(m.group(1))
            if m.group(2) == "1":
                sets["STM"].add(cid)
            # STM also reports the TGM flag it saw; the authoritative TGM set
            # comes from TaggerCheckTGM's own line, so do not add it here.
            continue
        m = PAT.search(line)
        if m:
            tag, cid, val = m.group(1), int(m.group(2)), m.group(3)
            if val in ("true", "1"):
                sets[tag].add(cid)
            continue
        m = PAT_SKIP.search(line)
        if m:
            skips.append((m.group(1), int(m.group(2)), int(m.group(3))))
            continue
        m = PAT_MAINS.search(line)
        if m and mains is None:
            window = (float(m.group(1)), float(m.group(2)))
            mains = (int(m.group(3)), int(m.group(4)))
    # wall time, if the runner wrote its resource file
    wall = None
    for rf in glob.glob(os.path.join(d, "pr_resource_*.txt")):
        for tok in open(rf).read().split():
            if tok.startswith("wall_s="):
                wall = int(tok.split("=")[1])
    return sets, skips, mains, window, wall


def main(argv):
    dirs = argv[1:]
    if len(dirs) < 2:
        raise SystemExit(__doc__)
    arms = [(d, read_arm(d)) for d in dirs]

    ref_name, (ref_sets, _, ref_mains, ref_window, _) = arms[0]
    print("reference arm: %s" % os.path.basename(ref_name))
    if ref_mains:
        print("  beam window [%g, %g) us: %d main(s) evaluated, %d out of window"
              % (ref_window[0], ref_window[1], ref_mains[0], ref_mains[1]))
        if ref_mains[1] == 0:
            print("  NOTE: 0 out of window -- on PDVD the beam gate selects "
                  "everything (readout-wide window, doc pdvd/25 sec 2.1)")
    for tag in ("TGM", "STM", "FC"):
        print("  %-3s tagged: %3d" % (tag, len(ref_sets[tag])))

    ok = True
    for name, (sets, skips, mains, _window, wall) in arms:
        print()
        print("== %s ==" % os.path.basename(name))
        if wall is not None:
            print("   wall_s=%d" % wall)
        for flags, nskip, nkeep in skips:
            print("   CreateSteinerGraph skip_flags [%s]: skipped %d, kept %d"
                  % (flags, nskip, nkeep))
        if name == ref_name:
            print("   (reference)")
            continue
        for tag in ("TGM", "STM", "FC"):
            a, b = ref_sets[tag], sets[tag]
            only_a = sorted(a - b)
            only_b = sorted(b - a)
            sym = len(only_a) + len(only_b)
            verdict = "IDENTICAL" if sym == 0 else "DIFFERS"
            print("   %-3s %-9s |ref|=%3d |arm|=%3d  sym_diff=%d"
                  % (tag, verdict, len(a), len(b), sym))
            if sym:
                if only_a:
                    print("        only in ref: %s" % only_a)
                if only_b:
                    print("        only in arm: %s" % only_b)
                # FC is allowed to move on TGM clusters: with the Steiner build
                # skipped, cluster_fc_check has no steiner_pc and returns the
                # conservative is_fc=false.  Everything else fails the gate.
                if tag == "FC":
                    moved = set(only_a) | set(only_b)
                    outside = sorted(moved - ref_sets["TGM"])
                    if outside:
                        print("        FC moved on NON-TGM clusters: %s" % outside)
                        ok = False
                    else:
                        print("        all FC moves are on TGM-tagged clusters "
                              "(expected: no steiner_pc => is_fc=false)")
                else:
                    ok = False
    print()
    print("GATE: %s" % ("PASS -- TGM and STM sets identical" if ok else
                        "FAIL -- a set moved where it must not"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
