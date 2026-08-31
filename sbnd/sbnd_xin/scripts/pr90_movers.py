#!/usr/bin/env python3
"""pr/90 mover adjudication: pr86_movers.py with an explicit label epoch.

Fork of scripts/pr86_movers.py (doc pr/86 sec 10; that script is left
byte-identical -- it hardcodes the prod0813-era default tag family via
vtx_io.load_labels() and has no switch).  This round's arms are
harv3-lineage, so scoring them against the prod0813 labels would hit the
vtx_io.py pooling trap ("quietly wrong denominators").  Identical
classification rules:

For each hand-labelled event present in both arms:
    moved   = |main_vertex_A - main_vertex_B|          (arm-vs-arm, cm)
    dA, dB  = |click - main_vertex_{A,B}|              (click->main, cm)
Movers (moved > --min-move, default 0.05 cm) are classified:
    ADVERSE  dB > dA + 1.0 cm     -- moved OFF the click past the pr/78/79
                                     1 cm "correct" tolerance bar
    toward   dB < dA - 0.01
    on       dB <= 1.0 cm         (and not ADVERSE)
    away     everything else (small drift off, within the 1 cm bar)
Caveat carried from pr/85: labels with b1 = 0.00 are reco-anchored, so a
small `away` needs the corner-position cross-check before being called a
regression; ADVERSE is the stop-the-line class.

Exit 1 if any ADVERSE, else 0.

Usage: pr90_movers.py <arm_A> <arm_B> [--tsv out.tsv] [--min-move 0.05]
                      [--tags harv3|prod0813|mcp2k]
"""
import argparse
import json
import math  # noqa: F401  (parity with pr86_movers.py)
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)                       # sbnd_xin/
sys.path.insert(0, SX)
from vtx_rules import vtx_io                     # noqa: E402

TAGSETS = {
    "harv3": vtx_io.TAGS_HARV3,
    "prod0813": vtx_io.TAGS,
    "mcp2k": vtx_io.TAGS_MCP2K,
    "vtx100": vtx_io.TAGS_VTX100,   # doc pr/103: 2026-08-20 carried epoch (all samples)
    "vtx105": vtx_io.TAGS_VTX105,   # doc pr/105: 2026-08-21 carried epoch, all 7 tags incl. mcp2k auto/ragree
}


def load_dump(arm, ev):
    p = os.path.join(SX, arm, "pr_evt%d" % ev, "calib-pr-evt%d.json" % ev)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    t = d.get("tagger", {}) or {}
    return dict(
        vtx=vtx_io.xyz(d.get("main_vertex")),
        nue=t.get("nue_score"), numu=t.get("numu_score"),
        cosmict=t.get("cosmict_flag"),
    )


def fmt(v):
    return "%.2f" % v if v is not None else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--min-move", type=float, default=0.05, help="cm")
    ap.add_argument("--tags", default="harv3", choices=sorted(TAGSETS),
                    help="label epoch (default: harv3, the current-production "
                         "carry + delta scans)")
    args = ap.parse_args()

    truth = {}
    for doc in vtx_io.load_labels(tags=TAGSETS[args.tags]):
        ev = doc.get("eventNo")
        if doc["truth"] is None:
            continue
        if ev in truth and truth[ev][0] != doc["truth"]:
            print("# WARN conflicting labels for evt %d (tags differ); "
                  "keeping the first" % ev, file=sys.stderr)
            continue
        truth.setdefault(ev, (doc["truth"], doc.get("b1")))

    rows = []
    n_lab = n_cmp = 0
    for ev in sorted(truth):
        n_lab += 1
        a = load_dump(args.arm_a, ev)
        b = load_dump(args.arm_b, ev)
        if a is None or b is None:
            continue
        n_cmp += 1
        click, b1 = truth[ev]
        moved = vtx_io.dist(a["vtx"], b["vtx"])
        if moved is None or moved <= args.min_move:
            continue
        dA = vtx_io.dist(click, a["vtx"])
        dB = vtx_io.dist(click, b["vtx"])
        if dA is None or dB is None:
            verdict = "unmeasurable"
        elif dB > dA + 1.0:
            verdict = "ADVERSE"
        elif dB < dA - 0.01:
            verdict = "toward"
        elif dB <= 1.0:
            verdict = "on"
        else:
            verdict = "away"
        dn = (a["nue"] is not None and b["nue"] is not None
              and abs(a["nue"] - b["nue"]) > 1e-6)
        dm = (a["numu"] is not None and b["numu"] is not None
              and abs(a["numu"] - b["numu"]) > 1e-6)
        rows.append(dict(
            ev=ev, moved=moved, dA=dA, dB=dB, verdict=verdict, b1=b1,
            nue=("%.2f->%.2f" % (a["nue"], b["nue"])) if dn else "=",
            numu=("%.2f->%.2f" % (a["numu"], b["numu"])) if dm else "=",
            cosmict=("%s->%s" % (a["cosmict"], b["cosmict"]))
                    if a["cosmict"] != b["cosmict"] else "=",
        ))

    rows.sort(key=lambda r: -r["moved"])
    for r in rows:
        print("evt %-7d moved %6.2f cm  click->main %s -> %s  b1 %s  %-8s "
              "nue %s  numu %s  cosmict %s"
              % (r["ev"], r["moved"], fmt(r["dA"]), fmt(r["dB"]),
                 fmt(r["b1"]), r["verdict"], r["nue"], r["numu"],
                 r["cosmict"]))
    n_adv = sum(1 for r in rows if r["verdict"] == "ADVERSE")
    print("# labels %d, compared %d, movers > %.2f cm: %d (ADVERSE %d, "
          "toward %d, on %d, away %d) [tags=%s]"
          % (n_lab, n_cmp, args.min_move, len(rows), n_adv,
             sum(1 for r in rows if r["verdict"] == "toward"),
             sum(1 for r in rows if r["verdict"] == "on"),
             sum(1 for r in rows if r["verdict"] == "away"),
             args.tags))
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("event\tmoved_cm\tdA_cm\tdB_cm\tverdict\tb1_cm\t"
                    "nue\tnumu\tcosmict\n")
            for r in rows:
                f.write("%d\t%.3f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                        % (r["ev"], r["moved"], fmt(r["dA"]), fmt(r["dB"]),
                           r["verdict"], fmt(r["b1"]), r["nue"], r["numu"],
                           r["cosmict"]))
        print("# tsv -> %s" % args.tsv)
    return 1 if n_adv else 0


if __name__ == "__main__":
    sys.exit(main())
