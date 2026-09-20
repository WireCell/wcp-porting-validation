// doc sbnd_xin/115 -- list (entry, run, subrun, event) of an SBND reco1 art file with bare ROOT.
//
// One TTree::Draw over EventAuxiliary, no LArSoft and no PyROOT (this environment has neither
// the LArSoft dictionaries nor the ROOT python module).  Used for two things:
//   * the event-ID uniqueness pre-flight -- ql_evt<ID>/pr_evt<ID> are keyed by the art event
//     NUMBER alone, and SBND MC event numbers repeat across files (mc-cv: 50 distinct numbers
//     over 2017 events), so the campaign runs one out_root per reco1 file and must prove each
//     file is internally unique;
//   * the file -> event map every later d115 stage joins on.
//
// Usage: root -l -b -q 'rse_list.C("<reco1.root>")'   -> "RSE <entry> <run> <subrun> <event>"
void rse_list(const char* fn)
{
    TFile f(fn);
    TTree* t = (TTree*) f.Get("Events");
    if (!t) { printf("NOTREE\n"); return; }
    t->SetEstimate(1000000);
    Long64_t n = t->Draw("EventAuxiliary.id_.subRun_.run_.run_"
                         ":EventAuxiliary.id_.subRun_.subRun_"
                         ":EventAuxiliary.id_.event_", "", "goff");
    if (n < 0) { printf("DRAWFAIL\n"); return; }
    double *r = t->GetV1(), *s = t->GetV2(), *e = t->GetV3();
    for (Long64_t i = 0; i < n; ++i)
        printf("RSE %lld %d %d %d\n", i, (int) r[i], (int) s[i], (int) e[i]);
}
