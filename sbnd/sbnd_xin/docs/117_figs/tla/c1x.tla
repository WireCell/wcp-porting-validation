# doc 117 cell c1x -- CURRENT trajectory, ONE-step vertex chain, fit exclusion OFF.
# = doc pr/112's `nofitx` arm (SBND_FIT_EXCLUSION=false), the exclusion-free chain ALONE, which
# scored best of the five strategies on the data hand-scan metric (812 of 1011 against the
# production dual chain's 805 and the single exclusion-ON chain's 777, pr/112 sec 12.2).
# fit_exclusion=true is itself an SBND production flip (owner 2026-08-20, doc pr/98 sec 7: fits
# equal-or-better in 11/12 top movers, cost ~1.15x median); false restores the pre-flip fit path.
# So this cell trades fit quality for vertex choice -- which is the trade this round measures.
dl_vtx_dual_chain=false
fit_exclusion=false
