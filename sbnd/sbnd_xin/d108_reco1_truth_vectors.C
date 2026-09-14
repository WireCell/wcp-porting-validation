// doc 108 sec 3.6 -- per-event VECTOR truth from an SBND MC reco1 art file, bare ROOT.
//
// DEFAULT USAGE: ONE ENTRY PER ROOT PROCESS (the emulated read aborts with
// "free(): invalid pointer/size" after a few entries in one process; each entry
// in its own process reads cleanly -- docs/108_logs/reco1_truth_vectors.txt):
//
//   for e in $(seq 0 $((N-1))); do
//     ROOT_INCLUDE_PATH=<toolkit>/root/src root -l -b -q 'd108_reco1_truth_vectors.C("<reco1.root>", '$e', '$e')'
//   done
//
// ROOT_INCLUDE_PATH keeps the toolkit rootmap from hijacking the autoloader (same
// recipe as scripts/root/mc_truth_muons.C).  One TTree::Draw per expression;
// never Events->GetEntry().  TBufferFile::CheckByteCount errors are noise.
//
// Prints per entry: the generator MCTruth count (one per beam-nu interaction),
// each interaction's signed nu pdg, CCNC, mode, interaction type, nu E (GeV),
// vertex (cm) and time (ns) from the neutrino's first trajectory point; and the
// deposited energy Edep (MeV) grouped by the MCTruth each deposit descends from:
//   SimEnergyDeposit |trackID|  (negative = a dropped secondary of |trackID|;
//                                NOT origTrackID, which loses the delta rays)
//   -> MCParticle ftrackId      (largeant vector index i)
//   -> Assns row with ptr_data_1_.second == i
//   -> (ptr_data_2_ product id, ptr_data_2_.second key)
// One product id is the generator MCTruths (key = nu index), the other the corsika
// MCTruths.  Deposits whose |trackID| has no MCParticle are counted as "unmatched".
Long64_t d1(TTree* t, const TString& ex, Long64_t e) { return t->Draw(ex, "", "goff", 1, e); }

void d108_reco1_truth_vectors(const char* fn, Long64_t ent0 = 0, Long64_t ent1 = -1)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    TFile f(fn);
    TTree* t = (TTree*) f.Get("Events");
    t->SetEstimate(5000000);
    const TString M = "simb::MCTruths_generator__GenieGen.obj";
    const TString C = "simb::MCTruths_corsika__GenieGen.obj";
    const TString P = "simb::MCParticles_largeant__GenieGen.obj";
    const TString A = "simb::MCParticlesimb::MCTruthsim::GeneratedParticleInfoart::Assns_largeant__GenieGen.obj";
    const TString S = "sim::SimEnergyDeposits_ionandscint_priorSCE_G4.obj";
    if (ent1 < 0) ent1 = t->GetEntries() - 1;
    for (Long64_t e = ent0; e <= ent1; ++e) {
        auto grab = [&](const TString& ex) {
            Long64_t n = d1(t, ex, e);
            return std::vector<double>(t->GetV1(), t->GetV1() + std::max<Long64_t>(n, 0));
        };
        int run = (int) grab("EventAuxiliary.id_.subRun_.run_.run_")[0];
        int sub = (int) grab("EventAuxiliary.id_.subRun_.subRun_")[0];
        int evt = (int) grab("EventAuxiliary.id_.event_")[0];
        auto pdg = grab(M + ".fMCNeutrino.fNu.fpdgCode");
        auto ccnc = grab(M + ".fMCNeutrino.fCCNC");
        auto mode = grab(M + ".fMCNeutrino.fMode");
        auto itype = grab(M + ".fMCNeutrino.fInteractionType");
        auto E = grab(M + ".fMCNeutrino.fNu.ftrajectory.ftrajectory.second.fE");
        auto x = grab(M + ".fMCNeutrino.fNu.ftrajectory.ftrajectory.first.fP.fX");
        auto y = grab(M + ".fMCNeutrino.fNu.ftrajectory.ftrajectory.first.fP.fY");
        auto z = grab(M + ".fMCNeutrino.fNu.ftrajectory.ftrajectory.first.fP.fZ");
        auto tt = grab(M + ".fMCNeutrino.fNu.ftrajectory.ftrajectory.first.fE");
        const size_t ncors = grab(C + ".fOrigin").size();

        // particle index -> (product id, key)
        auto tid = grab(P + ".ftrackId");
        auto p1 = grab(A + ".ptr_data_1_.second");
        auto pid = grab(A + ".ptr_data_2_.first.id_.value_");
        auto key = grab(A + ".ptr_data_2_.second");
        std::vector<std::pair<double, double>> owner(tid.size(), {-1, -1});
        for (size_t r = 0; r < p1.size() && r < pid.size() && r < key.size(); ++r) {
            const size_t i = (size_t) p1[r];
            if (i < owner.size()) owner[i] = {pid[r], key[r]};
        }
        std::unordered_map<long long, size_t> tid2idx;
        for (size_t i = 0; i < tid.size(); ++i) tid2idx[(long long) tid[i]] = i;

        auto str = grab(S + ".trackID");
        auto sed = grab(S + ".edep");
        std::map<std::pair<double, double>, double> edep;  // (pid,key) -> MeV
        double unmatched = 0;
        for (size_t j = 0; j < str.size() && j < sed.size(); ++j) {
            auto it = tid2idx.find(std::llabs((long long) str[j]));
            if (it == tid2idx.end() || owner[it->second].first < 0) { unmatched += sed[j]; continue; }
            edep[owner[it->second]] += sed[j];
        }

        printf("r%d_s%d_e%d entry %lld: n_nu=%zu n_corsika_MCTruth=%zu n_mcparticle=%zu n_sed=%zu\n",
               run, sub, evt, e, pdg.size(), ncors, tid.size(), str.size());
        for (size_t i = 0; i < pdg.size(); ++i) {
            auto at = [&](const std::vector<double>& v) { return i < v.size() ? v[i] : NAN; };
            printf("   NU idx=%zu pdg=%g ccnc=%g mode=%g int_type=%g E_GeV=%.4f vtx_cm=(%.2f,%.2f,%.2f) t_ns=%.1f\n",
                   i, pdg[i], at(ccnc), at(mode), at(itype), at(E), at(x), at(y), at(z), at(tt));
        }
        printf("   EDEP_MeV by (MCTruth product id, key):");
        for (auto& kv : edep) printf(" (%.0f,%.0f)=%.1f", kv.first.first, kv.first.second, kv.second);
        printf("  unmatched=%.1f\n", unmatched);
    }
}
