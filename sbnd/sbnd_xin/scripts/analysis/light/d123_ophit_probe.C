// doc 123 -- dump reco1 OpHits + OpFlashes for selected events, bare ROOT + the
// wire-cell-sbnd-reco1 mirror dictionary (no LArSoft).
//   root -l -b -q 'd123_ophit_probe.C("reco1.root","events.txt","out.tsv",tlo_us,thi_us)'
// out rows:  H event entry ch t_us pe width_us        (hits in [tlo,thi], t = start+rise, SBND flash-finder input time)
//            F event entry tpc t_us pe                (all reco1 opflashtpc{0,1})
//            N event entry nhits_total
R__LOAD_LIBRARY(/home/xqian/toolkit-dev/wire-cell-sbnd-reco1/install/lib/libWireCellSBNDReco1.so)
#include <fstream>
#include <set>
void d123_ophit_probe(const char* path, const char* evlist, const char* out, double tlo, double thi)
{
    std::set<unsigned> want;
    { std::ifstream in(evlist); unsigned e; while (in >> e) want.insert(e); }
    TFile* f = TFile::Open(path);
    TTree* t = (TTree*) f->Get("Events");
    std::ofstream o(out);
    const Long64_t n = t->GetEntries();
    TBranch* baux = t->GetBranch("EventAuxiliary");
    art::EventAuxiliary aux; art::EventAuxiliary* paux = &aux;
    baux->SetAddress(&paux);
    for (Long64_t i = 0; i < n; ++i) {
        baux->GetEntry(i);
        unsigned ev = aux.id_.event_;
        if (!want.empty() && !want.count(ev)) continue;
        art::Wrapper<std::vector<recob::OpHit>> wh; auto* ph = &wh;
        TBranch* bh = t->GetBranch("recob::OpHits_ophitpmt__Reco1.");
        bh->SetAddress(&ph); bh->GetEntry(i); bh->ResetAddress();
        o << "N\t" << ev << "\t" << i << "\t" << wh.obj.size() << "\t" << aux.id_.subRun_.run_.run_ << "\t" << aux.id_.subRun_.subRun_ << "\n";
        for (auto& h : wh.obj) {
            double tt = h.fStartTime + h.fRiseTime;
            if (tt < tlo || tt > thi) continue;
            o << "H\t" << ev << "\t" << i << "\t" << h.fOpChannel << "\t" << tt << "\t" << h.fPE << "\t" << h.fWidth << "\n";
        }
        for (int tpc = 0; tpc < 2; ++tpc) {
            art::Wrapper<std::vector<recob::OpFlash>> wf; auto* pf = &wf;
            TBranch* bf = t->GetBranch(Form("recob::OpFlashs_opflashtpc%d__Reco1.", tpc));
            bf->SetAddress(&pf); bf->GetEntry(i); bf->ResetAddress();
            for (auto& fl : wf.obj) {
                double pe = 0; for (double x : fl.fPEperOpDet) pe += x;
                o << "F\t" << ev << "\t" << i << "\t" << tpc << "\t" << fl.fTime << "\t" << pe << "\n";
            }
        }
    }
    baux->ResetAddress();
    printf("done %s\n", path);
}
