// Dump the beam-neutrino (fOrigin==1) sim::MCTrack start vertices of an SBND MC
// reco1 art file -- bare ROOT, no LArSoft.  Used as a *label-independent
// fingerprint* of an event: two reconstructions of the same art entry must
// agree on these coordinates to <0.01 cm, while different events of the same
// production sit 13-360 cm apart (docs/67_round2-patrec-10evt.md sec 9.2/10.1).
//
//   root -l -b -q 'mc_nu_vertices.C("<reco1-mc.root>", "tag", ent0, ent1, emin)'
//
// One line per vertex:
//   <tag> ent<N> run=<r> sr=<s> evt=<e>  <x> <y> <z>
//
// Same bare-ROOT rules as mc_truth_muons.C: the art file carries StreamerInfo
// for sim::MCTrack, so cling reads it through emulated classes; NEVER call
// Events->GetEntry() (~2500 branches, segfaults) -- TTree::Draw touches only the
// branches named in the expression.  Run with
//   ROOT_INCLUDE_PATH=<toolkit>/root/src
// so the toolkit rootmap does not hijack the autoloader.
void mc_nu_vertices(const char* fn, const char* tag = "file", Long64_t ent0 = 0,
                    Long64_t ent1 = -1, double emin = 50.)
{
    TFile f(fn);
    TTree* t = (TTree*) f.Get("Events");
    if (!t) { printf("no Events tree in %s\n", fn); return; }
    if (ent1 < 0) ent1 = t->GetEntries() - 1;
    const char* P = "sim::MCTracks_mcreco__G4.obj";
    for (Long64_t e = ent0; e <= ent1; ++e) {
        t->Draw("EventAuxiliary.id_.subRun_.run_.run_:"
                "EventAuxiliary.id_.subRun_.subRun_:"
                "EventAuxiliary.id_.event_", "", "goff", 1, e);
        int run = (int) t->GetV1()[0], sr = (int) t->GetV2()[0], evt = (int) t->GetV3()[0];
        Long64_t n = t->Draw(Form("%s.fOrigin:%s.fStart._position.fP.fX:"
                                  "%s.fStart._position.fP.fY:%s.fStart._position.fP.fZ:"
                                  "%s.fStart._momentum.fE",
                                  P, P, P, P, P), "", "goff", 1, e);
        for (Long64_t i = 0; i < n; ++i) {
            if ((int) t->GetV1()[i] != 1) continue;         // 1 = beam neutrino
            if (t->GetVal(4)[i] < emin) continue;
            printf("%s ent%lld run=%d sr=%d evt=%d  %.2f %.2f %.2f\n",
                   tag, e, run, sr, evt,
                   t->GetV2()[i], t->GetV3()[i], t->GetV4()[i]);
            fflush(stdout);   // ROOT tears these art files down with a
                              // 'free(): invalid pointer' abort; unflushed
                              // block-buffered output would be lost.
        }
    }
}
