// Truth muon census for an SBND MC reco1 art file -- bare ROOT, no LArSoft.
//
//   root -l -b -q 'mc_truth_muons.C("<reco1-mc.root>", ent0, ent1, t0lo, t0hi, emin)'
//
// Prints, per art entry, every sim::MCTrack in the [t0lo,t0hi] ns window above
// emin MeV, with its origin (1 = beam neutrino, 2 = cosmic ray), start/end
// points and an [AV] marker on each end that lies inside the SBND active
// volume (|x| < 200, |y| < 200, 0 < z < 500 cm, world coords).  Grep the output
// for 'pdg=   13|pdg=  -13' plus a trailing '[AV]' to get the muons that STOP
// inside the TPC -- the truth-level definition of a stopping muon (STM).
//
// Why bare ROOT with TTree::Draw: the art file carries StreamerInfo for
// sim::MCTrack / sim::MCStep, so cling reads them through emulated classes with
// no LArSoft dictionaries (same trick as dump_truth_sed.C).  As in
// toolkit/root/src/SBNDReco1Reader.h, NEVER call Events->GetEntry(): the tree
// has ~2500 branches and deserializing all of them segfaults.  TTree::Draw
// touches only the branches named in the expression, which is safe.
//
// Gotcha: run it from a directory where the toolkit's own rootmap does not
// hijack the autoloader, or with
//   ROOT_INCLUDE_PATH=<toolkit>/root/src root -l -b -q ...
// (the WireCellRoot dictionary payload uses a relative include path).
//
// Used by docs/67_round2-patrec-10evt.md.
bool mctm_inav(double x, double y, double z)
{
    return fabs(x) < 200 && fabs(y) < 200 && z > 0 && z < 500;
}

void mc_truth_muons(const char* fn, Long64_t ent0 = 0, Long64_t ent1 = 9,
                    double t0lo = -2000000, double t0hi = 2000000, double emin = 100.)
{
    TFile f(fn);
    TTree* t = (TTree*) f.Get("Events");
    if (!t) { printf("no Events tree in %s\n", fn); return; }
    const char* P = "sim::MCTracks_mcreco__G4.obj";
    for (Long64_t e = ent0; e <= ent1; ++e) {
        t->Draw("EventAuxiliary.id_.event_", "", "goff", 1, e);
        int evt = (int) t->GetV1()[0];
        Long64_t n = t->Draw(Form("%s.fPDGCode:%s.fOrigin:%s.fStart._position.fP.fX:"
                                  "%s.fStart._position.fP.fY:%s.fStart._position.fP.fZ",
                                  P, P, P, P, P), "", "goff", 1, e);
        std::vector<double> pdg(n), org(n), sx(n), sy(n), sz(n);
        for (Long64_t i = 0; i < n; ++i) {
            pdg[i] = t->GetV1()[i]; org[i] = t->GetV2()[i];
            sx[i] = t->GetV3()[i]; sy[i] = t->GetV4()[i]; sz[i] = t->GetVal(4)[i];
        }
        t->Draw(Form("%s.fEnd._position.fP.fX:%s.fEnd._position.fP.fY:%s.fEnd._position.fP.fZ:"
                     "%s.fStart._position.fE:%s.fStart._momentum.fE",
                     P, P, P, P, P), "", "goff", 1, e);
        std::vector<double> ex(n), ey(n), ez(n), st(n), en(n);
        for (Long64_t i = 0; i < n; ++i) {
            ex[i] = t->GetV1()[i]; ey[i] = t->GetV2()[i]; ez[i] = t->GetV3()[i];
            st[i] = t->GetV4()[i]; en[i] = t->GetVal(4)[i];
        }
        printf("EVENT entry=%lld evt=%d ntracks=%lld\n", e, evt, n);
        for (Long64_t i = 0; i < n; ++i) {
            if (st[i] < t0lo || st[i] > t0hi) continue;
            if (en[i] < emin) continue;
            printf("  pdg=%5.0f org=%.0f E=%8.1fMeV t0=%9.0fns start=(%7.1f,%7.1f,%7.1f)%s "
                   "end=(%7.1f,%7.1f,%7.1f)%s\n",
                   pdg[i], org[i], en[i], st[i], sx[i], sy[i], sz[i],
                   mctm_inav(sx[i], sy[i], sz[i]) ? " [AV]" : "     ",
                   ex[i], ey[i], ez[i],
                   mctm_inav(ex[i], ey[i], ez[i]) ? " [AV]" : "     ");
        }
    }
}
