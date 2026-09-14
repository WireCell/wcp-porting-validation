#!/usr/bin/env python3
"""doc 108 sec 3.4: are the on-disk MC reco1 files enough for truth-in-ROOT?

Usage:
  python3 d108_reco1_probe.py [--reco1 /nfs/data/1/yuhw/2025-fall-prod-sample/round2-patrec/mc_paths-v10_14_02_03-100files]
      [--run /nfs/data/1/xqian/sbnd_data/run] [--d107 products/d107]
  root -l -b -q 'd108_reco1_root_probe.C("<one reco1 file>")'   # class layouts + emulated read

1. coverage: which production input files (summary-merged.csv "file") are on disk,
   and what the covered events hold (doc 107 candidates.tsv / truth.tsv);
2. the truth data products present in one file (Events branch names);
3. whether uproot can deserialize them (it cannot: memberwise-split vectors).
"""
import argparse
import collections
import csv
import os

import uproot

TRUTH_PRODUCTS = [
    "simb::MCTruths_generator__GenieGen.", "simb::GTruths_generator__GenieGen.",
    "simb::MCFluxs_generator__GenieGen.", "simb::MCParticles_largeant__GenieGen.",
    "simb::MCParticlesimb::MCTruthsim::GeneratedParticleInfoart::Assns_largeant__GenieGen.",
    "sim::ParticleAncestryMaps_largeant__GenieGen.", "sim::SimEnergyDeposits_ionandscint_priorSCE_G4.",
    "sim::SimChannels_simtpc2d_simpleSC_DetSim.", "simb::MCTruths_corsika__GenieGen.",
    "recob::Wires_simtpc2d_dnnsp_DetSim.",
]


def products_only(paths):
    """Entries and truth/wire product branches of arbitrary art files (no coverage join)."""
    keys = ("MCTruth", "GTruth", "MCParticle", "SimEnergyDeposit", "SimChannel", "recob::Wire", "Assns_largeant")
    for p in paths:
        E = uproot.open(p)["Events"]
        print("==", p, "entries", E.num_entries, "size_MB", os.path.getsize(p) // 2**20)
        for n in sorted(b.name for b in E.branches):
            if any(k in n for k in keys):
                print("   ", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reco1", default="/nfs/data/1/yuhw/2025-fall-prod-sample/round2-patrec/"
                    "mc_paths-v10_14_02_03-100files")
    ap.add_argument("--run", default="/nfs/data/1/xqian/sbnd_data/run")
    ap.add_argument("--d107", default="products/d107")
    ap.add_argument("--products-only", nargs="+", metavar="ART_FILE",
                    help="only list entries + truth products of these files (sec 3.5 second sample)")
    a = ap.parse_args()
    if a.products_only:
        products_only(a.products_only)
        return

    have = set(f for f in os.listdir(a.reco1) if f.endswith(".root"))
    R = list(csv.DictReader(open(os.path.join(a.run, "summary-merged.csv"))))
    per_file = collections.Counter(os.path.basename(r["file"]) for r in R)
    print("== 1 coverage")
    print("production events", len(R), "input files", len(per_file),
          "events/file min/max", min(per_file.values()), max(per_file.values()))
    print("reco1 files on disk", len(have), "of which production inputs", len(have & set(per_file)))
    ev = {(r["run"], r["subrun"], r["event"]) for r in R if os.path.basename(r["file"]) in have}
    print("production events covered", len(ev), "runs", dict(collections.Counter(e[0] for e in ev)))
    C = list(csv.DictReader(open(os.path.join(a.d107, "candidates.tsv")), delimiter="\t"))
    T = list(csv.DictReader(open(os.path.join(a.d107, "truth.tsv")), delimiter="\t"))
    c = [x for x in C if (x["run"], x["subrun"], x["event"]) in ev]
    t = [x for x in T if (x["run"], x["subrun"], x["event"]) in ev]

    def fv(x):
        X, Y, Z = float(x["vx"]), float(x["vy"]), float(x["vz"])
        return 5 < abs(X) < 190 and abs(Y) < 190 and 10 < Z < 450
    print("candidate rows", len(c), "truth interactions", len(t),
          dict(collections.Counter((x["flav"], x["ccnc"]) for x in t)))
    print("truth in FV", sum(fv(x) for x in t), dict(collections.Counter((x["flav"], x["ccnc"]) for x in t if fv(x))))

    p = os.path.join(a.reco1, sorted(have & set(per_file))[0])
    E = uproot.open(p)["Events"]
    names = {b.name for b in E.branches}
    print("== 2 products in", os.path.basename(p), "entries", E.num_entries)
    for n in TRUTH_PRODUCTS:
        print("  ", "present" if n in names else "ABSENT ", n)
    print("== 3 uproot deserialization")
    for n in ("EventAuxiliary", "simb::MCTruths_generator__GenieGen.", "simb::MCParticles_largeant__GenieGen.",
              "sim::SimChannels_simtpc2d_simpleSC_DetSim."):
        b = E[n] if n == "EventAuxiliary" else E[n + "/" + n + "obj"]
        try:
            b.array(entry_start=0, entry_stop=1, library="np")
            print("   OK ", n)
        except Exception as e:
            print("   FAIL", n, type(e).__name__, str(e).splitlines()[0][:120])


if __name__ == "__main__":
    main()
