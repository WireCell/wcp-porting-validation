#!/usr/bin/env python3
"""doc sbnd_xin/115 sec 13: the POT / beam-gate arithmetic behind the addendum.

Every input here is a number measured elsewhere in doc 115 and quoted at the top, so this
script is the arithmetic and nothing else -- it reads no arm and cannot go stale against one.
Re-derive the inputs with:

  POT            root -l -b -q 'scripts/d115/pot_sum.C("<list>")'
  flash windows  root -l -b -q 'scripts/d115/flash_window.C("<arm>/pr_evt*/tracking-pr.root")'
  selections     docs/115_sel/{cv,nuecc}/d107_selection.txt, docs/115_off/d115_beamoff_rate.txt

Why it exists: sec 9 of the doc recorded "no POT or trigger normalisation available in the
staged sample".  That is wrong -- sumdata::POTSummary_generator__GenieGen. is in the SubRuns
tree of every MC reco1 file -- and the two PURITIES in sec 6/7 are in-sample numbers that mean
something different once the samples are put on a common exposure.  See doc 115 sec 13.

Usage: python3 scripts/d115/normalisation.py
"""

# ---- measured inputs ---------------------------------------------------------------------
POT_CV,  N_CV  = 1.116251e17, 2017          # pot_sum.C over mc-cv/reco1/*.root
POT_NUE, N_NUE = 4.050539e19, 2001          # pot_sum.C over mc-nuecc/reco1/*.root
N_CV_ACTIVE    = 954                        # mc-cv events with a true nu in the active TPC

POT_PER_SPILL  = 5.0e12                     # BNB nominal.  Every gate number scales linearly.

# mc-cv, numu_score > 0.9, reco vertex in FV  (docs/115_sel/cv/d107_selection.txt)
CV_NUMU_SEL, CV_NUMU_SIG = 448, 387
CV_NUMU_COSMIC_FAKE      = 17               # unmatched, nearest true vertex > 50 cm

# mc-nuecc, nue_score > cut, reco vertex in FV  (docs/115_sel/nuecc/d107_selection.txt)
NUE_SEL = {7.0: (618, 632), 4.0: (734, 781)}          # (signal, all selected)
# mc-cv at the same cuts (docs/115_sel/cv, docs/115_sel_edep100/cv): (all selected, of which
# true nueCC in FV).  The difference is the non-nue background -- the only part mc-cv
# contributes, since its nueCC signal would double-count the intrinsic-nue sample's.
CV_NUE  = {7.0: (5, 4), 4.0: (7, 4)}
NUE_FV_CV, NUE_FV_NUECC = 5, 1511           # true nueCC in FV, each sample

# beam-off, 1000 gates  (docs/115_off/d115_beamoff_rate.txt)
OFF_GATES = 1000
OFF_NUMU, OFF_NUE7, OFF_NUE4 = 5, 0, 0
# T_flash in-window group multiplicity, flash_window.C
OFF_FLASH_HIST = {0: 485, 1: 502, 2: 13}


def main():
    w = POT_CV / POT_NUE
    gates = POT_CV / POT_PER_SPILL
    out = []
    A = out.append

    A("doc 115 sec 13 -- POT normalisation")
    A("")
    A("## 13.1 exposure")
    A("  mc-cv    %.6e POT / %d events = %.4e POT/event" % (POT_CV, N_CV, POT_CV / N_CV))
    A("  mc-nuecc %.6e POT / %d events = %.4e POT/event" % (POT_NUE, N_NUE, POT_NUE / N_NUE))
    A("  weight nuecc -> cv exposure   w = %.6e  (1/%.1f)" % (w, 1 / w))
    A("  cross-check true nueCC in FV: nuecc x w = %.2f   mc-cv observed = %d"
      % (NUE_FV_NUECC * w, NUE_FV_CV))
    A("  beam gates at %.1e POT/spill: %.0f" % (POT_PER_SPILL, gates))
    A("  generated nu per spill (rockbox) %.4f ; with nu in the active TPC %.4f"
      % (N_CV / gates, N_CV_ACTIVE / gates))

    A("")
    A("## 13.2 nueCC purity on a common exposure (mc-cv POT)")
    A("  %-6s %9s %9s %9s %9s" % ("cut", "signal", "nue-bkg", "cv-bkg", "purity"))
    for cut in sorted(NUE_SEL, reverse=True):
        sig, sel = NUE_SEL[cut]
        cv_sel, cv_sig = CV_NUE[cut]
        S, Bnue, Bcv = sig * w, (sel - sig) * w, cv_sel - cv_sig
        tot = S + Bnue + Bcv
        A("  >%-5g %9.3f %9.3f %9d %8.1f %%" % (cut, S, Bnue, Bcv, 100 * S / tot))
        # the cv background is a handful of events; a sqrt(n) band on it alone already spans
        # most of [0,1], which is the point of sec 13.2 -- this round does not measure it
        lo, hi = max(0.0, Bcv - Bcv ** 0.5), Bcv + Bcv ** 0.5 + 0.5
        A("         cv background %d event(s), ~68 %% [%.2f, %.2f] -> purity [%.0f, %.0f] %%"
          % (Bcv, lo, hi, 100 * S / (S + Bnue + hi), 100 * S / (S + Bnue + lo)))

    A("")
    A("## 13.3 beam-off scaled to the MC exposure (numu_score > 0.9)")
    nz = OFF_FLASH_HIST.get(0, 0)
    A("  off-beam events with NO in-window flash group: %d/%d = %.1f %%"
      % (nz, OFF_GATES, 100.0 * nz / OFF_GATES))
    A("    consistent with a zero-bias stream (f ~ 1) but NOT proof: in_window counts flashes")
    A("    WCT reconstructed above flash_minPE=50 in ITS window, not the SBND hardware trigger.")
    A("    f is the missing input; the ladder below is the honest reading.")
    A("  off-beam selected rate  %d/%d = %.2f %% per gate" % (OFF_NUMU, OFF_GATES,
                                                              100.0 * OFF_NUMU / OFF_GATES))
    A("  %-5s %12s %10s %14s %9s" % ("f", "cosmic-only", "total", "contamination", "purity"))
    for f in (1.0, 0.3, 0.1, 0.03):
        c = OFF_NUMU * gates / OFF_GATES * f
        A("  %-5.2f %12.1f %10.1f %13.2f %% %8.1f %%"
          % (f, c, CV_NUMU_SEL + c, 100 * c / (CV_NUMU_SEL + c),
             100 * CV_NUMU_SIG / (CV_NUMU_SEL + c)))
    c1 = OFF_NUMU * gates / OFF_GATES
    tot = CV_NUMU_SEL + c1
    A("  decomposition at f = 1 (total %.1f selected):" % tot)
    A("    signal (true numuCC in FV, matched)        %6.1f  %5.1f %%"
      % (CV_NUMU_SIG, 100 * CV_NUMU_SIG / tot))
    A("    nu-induced background                      %6.1f  %5.1f %%"
      % (CV_NUMU_SEL - CV_NUMU_SIG, 100 * (CV_NUMU_SEL - CV_NUMU_SIG) / tot))
    A("      of which CORSIKA fakes inside nu gates   %6.1f  %5.1f %%"
      % (CV_NUMU_COSMIC_FAKE, 100 * CV_NUMU_COSMIC_FAKE / tot))
    A("    cosmic-only gates (beam-off)               %6.1f  %5.1f %%" % (c1, 100 * c1 / tot))
    A("  in-sample purity %.1f %% -> exposure-weighted %.1f %%"
      % (100 * CV_NUMU_SIG / CV_NUMU_SEL, 100 * CV_NUMU_SIG / tot))

    A("")
    A("## 13.2 cosmic limit for the nue selection")
    S = NUE_SEL[7.0][0] * w
    ul = 1.14 * gates / OFF_GATES                      # per unit f
    A("  beam-off %d/%d gates pass either nue cut; 68 %% Poisson upper limit ~1.1 per %d gates"
      % (OFF_NUE7 + OFF_NUE4, OFF_GATES, OFF_GATES))
    A("  = <= %.0f*f events at %.0f gates, against a %.2f-event signal: the cosmic term exceeds"
      % (ul, gates, S))
    A("  the whole signal for any f above ~%.2f, and f is unknown (sec 13.3)." % (S / ul))

    print("\n".join(out))


if __name__ == "__main__":
    main()
