#!/usr/bin/env python3
"""doc sbnd_xin/115: build the per-interaction truth table for a round-3 MC sample.

Two independent sources are merged, and the overlap between them is checked row by row:

  1. the event-level GENIE truth TSV shipped with the sample
     (xin-round3-samples/<dir>/truth/<sample>-truth.tsv, one row per (event, true neutrino):
     run subrun event n_nu inu nu_pdg ccnc mode interaction_type E_nu_MeV vtx_x/y/z_cm t_ns
     lepton_pdg target_pdg n_genie_particles in_active_tpc);
  2. the true deposited energy per interaction, read straight out of the reco1 files by
     scripts/d115/truth_edep.sh -> d108_reco1_truth_vectors.C, which the TSV does not carry
     and which doc 107 sec 5.6's signal definition needs.

Source 2 also re-derives pdg / ccnc / mode / vertex / time, so this script does not merely
join -- it asserts the two agree and fails loudly if they do not.  That is a free provenance
check in both directions: on the TSV (made at FNAL with PyROOT + the nusimdata dictionaries)
and on the bare-ROOT macro (made here with neither).

Output: products/d115/<sample>/truth_base.tsv, the first 13 of doc 107's TRUTH_COLS

    run subrun event idx flav mode ccnc Etot Edep T vx vy vz

The remaining three (event_has_candidate, n_candidates, min_cand_dist_cm) describe the
reconstruction, not the truth, and are filled by d115_truth_join.py, which writes the final
truth.tsv that d107_selection.py reads.

Conventions, both taken from doc 107 sec 5.5 so the two records stay comparable:
  * flavour names are SIGN-BLIND (+-14 -> numu, +-12 -> nue), matching TensorSetLabeler;
  * Edep is the SimEnergyDeposit sum over the descendants of that generator MCTruth, the same
    quantity Bee's truth nodes carry (doc 108 sec 3.3 validated the two agree to +-0.11 MeV).

Usage:
  python3 scripts/d115/build_truth.py <cv|nuecc> [--pilot] [--edep PATH] [--out PATH]

  --pilot  keep only the events edep_raw.txt covers, instead of treating a TSV event with no
           macro row as a defect.  For a partial extraction (a pilot arm), never a full run.
"""
import collections
import os
import re
import sys
import zlib

SX = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))

SAMPLE = {
    "cv": ("mc-cv", "mc-cv-truth.tsv"),
    "nuecc": ("mc-nuecc", "mc-nuecc-truth.tsv"),
}

# simb::int_type_ mode codes -> the names doc 107's Bee truth text used.
MODE = {0: "QE", 1: "RES", 2: "DIS", 3: "COH", 4: "CohElastic", 5: "ElectronScattering",
        6: "IMDAnnihilation", 7: "InverseBetaDecay", 8: "GlashowResonance", 9: "AMNuGamma",
        10: "MEC", 11: "Diffractive", 12: "EM", 13: "WeakMix"}

# art ProductID = crc32 of the branch name.  Computed, not hardcoded: d108_truth_vs_bee.py
# identified the generator product by a heuristic on the Assns keys, which is unnecessary
# when the branch name is known.
GEN_ID = zlib.crc32(b"simb::MCTruths_generator__GenieGen.") & 0xFFFFFFFF
COR_ID = zlib.crc32(b"simb::MCTruths_corsika__GenieGen.") & 0xFFFFFFFF

HDR_RE = re.compile(r"^r(\d+)_s(\d+)_e(\d+) entry (\d+):")
NU_RE = re.compile(r"^\s+NU idx=(\d+) pdg=(\S+) ccnc=(\S+) mode=(\S+) int_type=(\S+) "
                   r"E_GeV=(\S+) vtx_cm=\((\S+),(\S+),(\S+)\) t_ns=(\S+)")
ED_RE = re.compile(r"^\s+EDEP_MeV by \(MCTruth product id, key\):(.*)\s+unmatched=(\S+)")
PAIR_RE = re.compile(r"\((\d+),(\d+)\)=([-\d.eE+]+)")

# Tolerances, from d108_truth_vs_bee.py (which validated this macro against Bee).
TOL_E_MEV, TOL_VTX_CM, TOL_T_US = 0.11, 0.02, 0.001


def flavour(pdg):
    """Sign-blind flavour name, as TensorSetLabeler's pdg_name does (doc 107 sec 5.5)."""
    return {12: "nue", 14: "numu", 16: "nutau"}.get(abs(int(pdg)), "pdg%d" % int(pdg))


def parse_edep(path):
    """-> {(run, subrun, event): {inu: edep_MeV}}, plus the per-event macro truth for the check."""
    edep, macro = {}, {}
    rse = None
    for line in open(path):
        if line.startswith("#"):
            continue
        m = HDR_RE.match(line)
        if m:
            rse = tuple(int(x) for x in m.group(1, 2, 3))
            edep[rse], macro[rse] = {}, {}
            continue
        if rse is None:
            continue
        m = NU_RE.match(line)
        if m:
            macro[rse][int(m.group(1))] = dict(
                pdg=int(float(m.group(2))), ccnc=int(float(m.group(3))),
                mode=int(float(m.group(4))), E_MeV=float(m.group(6)) * 1000.0,
                vx=float(m.group(7)), vy=float(m.group(8)), vz=float(m.group(9)),
                T_us=float(m.group(10)) / 1000.0)
            continue
        m = ED_RE.match(line)
        if m:
            for pid, key, val in PAIR_RE.findall(m.group(1)):
                if int(pid) == GEN_ID:
                    edep[rse][int(key)] = float(val)
    return edep, macro


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample", choices=sorted(SAMPLE))
    ap.add_argument("--pilot", action="store_true",
                    help="keep only the events edep_raw covers (a partial extraction)")
    ap.add_argument("--edep", default="", help="override the edep_raw.txt path")
    ap.add_argument("--out", default="", help="override the truth_base.tsv path")
    args = ap.parse_args()
    s = args.sample
    sdir, tname = SAMPLE[s]
    tsv = os.path.join(SX, "xin-round3-samples", sdir, "truth", tname)
    raw = args.edep or os.path.join(SX, "products", "d115", s, "edep_raw.txt")
    out = args.out or os.path.join(SX, "products", "d115", s, "truth_base.tsv")
    for p in (tsv, raw):
        if not os.path.exists(p):
            sys.exit("ERROR: missing input %s" % p)

    edep, macro = parse_edep(raw)
    print("edep_raw: %d events, %d generator interactions with a non-zero Edep"
          % (len(edep), sum(len(v) for v in edep.values())))

    rows, bad, nomacro = [], collections.Counter(), 0
    seen = set()
    with open(tsv) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        col = {k: i for i, k in enumerate(hdr)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                continue
            run, subrun, event = (int(f[col[k]]) for k in ("run", "subrun", "event"))
            n_nu = int(f[col["n_nu"]])
            inu = int(f[col["inu"]])
            if n_nu == 0:
                continue                         # the TSV's "no true neutrino" placeholder row
            rse = (run, subrun, event)
            if args.pilot and rse not in macro:
                continue
            seen.add(rse)
            pdg = int(f[col["nu_pdg"]])
            ccnc = int(f[col["ccnc"]])
            mode = int(f[col["mode"]])
            etot = float(f[col["E_nu_MeV"]])
            vx, vy, vz = (float(f[col["vtx_%s_cm" % a]]) for a in "xyz")
            t_us = float(f[col["t_ns"]]) / 1000.0

            # --- cross-check against the bare-ROOT macro --------------------------------
            mm = macro.get(rse, {}).get(inu)
            if mm is None:
                nomacro += 1
            else:
                if mm["pdg"] != pdg:
                    bad["pdg"] += 1
                if mm["ccnc"] != ccnc:
                    bad["ccnc"] += 1
                if mm["mode"] != mode:
                    bad["mode"] += 1
                if abs(mm["E_MeV"] - etot) > TOL_E_MEV:
                    bad["Etot"] += 1
                if max(abs(mm["vx"] - vx), abs(mm["vy"] - vy), abs(mm["vz"] - vz)) > TOL_VTX_CM:
                    bad["vtx"] += 1
                if abs(mm["T_us"] - t_us) > TOL_T_US:
                    bad["T"] += 1

            rows.append([run, subrun, event, inu, flavour(pdg), MODE.get(mode, "mode%d" % mode),
                         "CC" if ccnc == 0 else "NC", "%.1f" % etot,
                         "%.1f" % edep.get(rse, {}).get(inu, 0.0), "%.3f" % t_us,
                         "%.4f" % vx, "%.4f" % vy, "%.4f" % vz])

    missing_events = sorted(set(edep) - seen)
    print("truth TSV: %d interactions over %d events" % (len(rows), len(seen)))
    print("cross-check vs d108_reco1_truth_vectors.C: "
          + ("ALL AGREE" if not bad else "MISMATCHES " + dict(bad).__repr__())
          + ("" if not nomacro else "  (+%d interactions with no macro row)" % nomacro))
    if missing_events:
        print("  NOTE: %d event(s) in edep_raw but not in the TSV: %s"
              % (len(missing_events), missing_events[:5]))

    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("run\tsubrun\tevent\tidx\tflav\tmode\tccnc\tEtot\tEdep\tT\tvx\tvy\tvz\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print("-> %s" % out)
    return 1 if (bad or nomacro) else 0


if __name__ == "__main__":
    sys.exit(main())
