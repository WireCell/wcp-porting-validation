// doc sbnd_xin/115 sec 13.1: total generated POT of an MC sample.
//
// Every SBND MC reco1 file carries sumdata::POTSummary_generator__GenieGen. in its SubRuns
// tree, one row per subrun.  No LArSoft and no dictionary is needed: the members are read
// through the file's own StreamerInfo, the same way d108_reco1_truth_vectors.C reads
// simb::MCTruth.  Prints one ROW line per file (events, POT) so the caller can check that
// POT/file is flat while events/file is Poisson -- the evidence that no filter ran between
// gen and reco1, which is the condition for total-POT / total-events to be the exposure.
//
// Usage: root -l -b -q 'scripts/d115/pot_sum.C("<file-list.lst>")'
void pot_sum(const char* lst) {
    std::ifstream in(lst);
    std::string p;
    double tot = 0;
    long nf = 0, nev = 0;
    while (std::getline(in, p)) {
        if (p.empty()) continue;
        TFile* f = TFile::Open(p.c_str());
        if (!f || f->IsZombie()) { printf("BAD %s\n", p.c_str()); continue; }
        TTree* s = (TTree*)f->Get("SubRuns");
        TTree* e = (TTree*)f->Get("Events");
        double pot = 0;
        if (s) {
            s->Draw("sumdata::POTSummary_generator__GenieGen.obj.totpot", "", "goff");
            for (Long64_t i = 0; i < s->GetSelectedRows(); ++i) pot += s->GetV1()[i];
        }
        Long64_t ne = e ? e->GetEntries() : 0;
        printf("ROW %lld %.6e %s\n", ne, pot, gSystem->BaseName(p.c_str()));
        tot += pot; nev += ne; nf++;
        f->Close();
    }
    printf("FILES %ld  EVENTS %ld  TOTPOT %.6e  POT/EVT %.4e\n", nf, nev, tot, nev ? tot/nev : 0.0);
}
