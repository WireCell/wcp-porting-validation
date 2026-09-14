// doc 108 sec 3.4 -- reco1 truth classes seen by bare ROOT (no LArSoft dictionaries).
//   root -l -b -q 'd108_reco1_root_probe.C("/path/reco1-....root")'
// Part 1 prints the StreamerInfo layout the file embeds for the truth classes
// (what a mirror-class reader would have to reproduce).  Part 2 tries an
// emulated-class TTree::Scan of MCTruth fields; on these files it fails with
// TBufferFile::CheckByteCount errors and prints no rows.
void d108_reco1_root_probe(const char* path)
{
    TFile* f = TFile::Open(path);
    std::set<std::string> want = {"simb::MCTruth", "simb::MCNeutrino", "simb::MCParticle", "simb::GTruth",
                                  "sim::SimChannel", "sim::IDE", "sim::SimEnergyDeposit",
                                  "sim::GeneratedParticleInfo", "sim::ParticleAncestryMap",
                                  "art::EventAuxiliary", "art::EventID"};
    TIter it(f->GetStreamerInfoList());
    TObject* o;
    while ((o = it())) {
        auto* s = dynamic_cast<TStreamerInfo*>(o);
        if (!s || !want.count(s->GetName())) continue;
        printf("== %s v%d\n", s->GetName(), s->GetClassVersion());
        TIter e(s->GetElements());
        TStreamerElement* el;
        while ((el = (TStreamerElement*) e())) printf("   %-45s %s\n", el->GetTypeName(), el->GetName());
    }
    TTree* t = (TTree*) f->Get("Events");
    t->SetBranchStatus("*", 0);
    t->SetBranchStatus("simb::MCTruths_generator__GenieGen.*", 1);
    t->SetBranchStatus("EventAuxiliary*", 1);
    Long64_t n = t->Scan("EventAuxiliary.id_.event_:simb::MCTruths_generator__GenieGen.obj.fMCNeutrino.fCCNC:"
                         "simb::MCTruths_generator__GenieGen.obj.fMCNeutrino.fNu.fpdgCode", "", "colsize=12", 6);
    printf("emulated scan rows %lld\n", n);
}
