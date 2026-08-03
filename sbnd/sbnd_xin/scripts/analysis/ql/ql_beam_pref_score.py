#!/usr/bin/env python3
"""Score a Q/L run against the hand-scan labels, per work root (beam_pref tuning).

Truth: work/ql_labels/<mode>/.scan_state-evt<ID>.json  "selected" = list of
[flash_gid, main_cluster] true-match pairs (the doc-16 convention: end-to-end
true-match agreement was 93/100 data, 92/113 MC at ladder adoption).

Prediction: the matcher's auto_selected bundles in the regenerated calib dumps
<work_root>/ql_evt<ID>/calib-evt<ID>.json, keyed the same way.

Reports, per mode and per work root:
  agree   = true pairs the matcher also selected
  miss    = true pairs the matcher did not select
  extra   = matcher-selected pairs the hand-scan did not tag
  beam±   = of the disagreements, how many involve a beam-window flash
            (0.2 < t < 2.2 us) -- the population the beam_pref knob may move.

Usage:
  ./ql_beam_pref_score.py work-bpval-off work-bpval-w050 [...]
Repro (doc 22): roots produced by
  SBND_WORK_ROOT=$PWD/work-bpval-<cfg> ./run_ql_evt.sh <mode> all -calib [-beam-pref]
  (BEAMPREF_WEIGHT=<w> for the weight scan; evt<ID> symlinked from work/)
"""
import glob
import gzip
import json
import os
import re
import sys

HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
TLOW_US, THIGH_US = 0.2, 2.2


def calib_open(path):
    """Open a calib dump, transparently accepting a gzipped `<path>.gz` sibling.

    The 2026-07-24 work-tree consolidation gzipped the archived dumps (~6x);
    fresh -calib runs still write plain .json, so both names must resolve.
    """
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt")
    return open(path)


def load_truth(mode):
    sd = os.path.join(HERE, "work", "ql_labels", mode)
    truth = {}
    for p in sorted(glob.glob(os.path.join(sd, ".scan_state-evt*.json"))):
        ev = int(re.search(r"evt(\d+)", os.path.basename(p)).group(1))
        truth[ev] = {tuple(k) for k in json.load(open(p))["selected"]}
    return truth


def score_root(root, truth):
    n_true = n_agree = n_extra = 0
    beam_miss = beam_extra = 0
    detail = []
    for ev, want in sorted(truth.items()):
        cpath = os.path.join(root, "ql_evt%d" % ev, "calib-evt%d.json" % ev)
        if not (os.path.exists(cpath) or os.path.exists(cpath + ".gz")):
            detail.append("evt%d: MISSING calib dump" % ev)
            continue
        c = json.load(calib_open(cpath))
        ftime = {f["gid"]: f["time"] for f in c["flashes"]}  # us
        got = {(b["flash_gid"], b["main_cluster"])
               for b in c["bundles"] if b["auto_selected"]}
        agree = want & got
        miss = want - got
        extra = got - want
        n_true += len(want)
        n_agree += len(agree)
        n_extra += len(extra)

        def in_beam(pair):
            t = ftime.get(pair[0])
            return t is not None and TLOW_US < t < THIGH_US

        beam_miss += sum(in_beam(p) for p in miss)
        beam_extra += sum(in_beam(p) for p in extra)
        if miss or extra:
            detail.append("evt%d: miss=%s extra=%s"
                          % (ev, sorted(miss), sorted(extra)))
    return dict(n_true=n_true, n_agree=n_agree, n_extra=n_extra,
                beam_miss=beam_miss, beam_extra=beam_extra, detail=detail)


def main():
    roots = sys.argv[1:]
    if not roots:
        print(__doc__)
        sys.exit(1)
    for mode in ("data", "mc"):
        truth = load_truth(mode)
        print("== %s (%d events, %d true matches) ==" %
              (mode, len(truth), sum(len(v) for v in truth.values())))
        for root in roots:
            r = score_root(os.path.join(HERE, root), truth)
            print("%-22s agree %3d/%3d  extra %2d  (beam-window: miss %d, extra %d)"
                  % (os.path.basename(root), r["n_agree"], r["n_true"],
                     r["n_extra"], r["beam_miss"], r["beam_extra"]))
        if os.environ.get("BPSCORE_DETAIL"):
            for root in roots:
                r = score_root(os.path.join(HERE, root), truth)
                print("-- %s %s --" % (mode, os.path.basename(root)))
                for d in r["detail"]:
                    print("   " + d)


if __name__ == "__main__":
    main()
