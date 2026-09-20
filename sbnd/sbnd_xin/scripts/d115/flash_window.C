// doc sbnd_xin/115 sec 13.3: in-beam-window flash multiplicity per event.
//
// Counts DISTINCT in-window flash GROUPS per (run, subrun, event) from the doc-109 T_flash
// tree -- groups, not rows, because doc 109 rev 4 writes one row per (flash, TPC) and a single
// physical flash appears under both anodes.
//
// What it is for: the beam-off sample's normalisation.  If the off-beam stream were
// light-triggered, essentially every event would carry an in-window flash.  Measuring how many
// carry none is what says whether 1 off-beam event == 1 beam gate (sec 13.3's `f`).
//
// Usage: root -l -b -q 'scripts/d115/flash_window.C("<arm>/pr_evt*/tracking-pr.root")'
void flash_window(const char* glob) {
    TChain c("T_flash");
    int nf = c.Add(glob);
    Int_t run, subrun, event, inw, fg;
    c.SetBranchAddress("run", &run);
    c.SetBranchAddress("subrun", &subrun);
    c.SetBranchAddress("event", &event);
    c.SetBranchAddress("in_window", &inw);
    c.SetBranchAddress("flash_group", &fg);
    std::map<std::tuple<int,int,int>, std::set<int>> grp;
    Long64_t N = c.GetEntries();
    for (Long64_t i = 0; i < N; ++i) {
        c.GetEntry(i);
        auto k = std::make_tuple(run, subrun, event);
        grp[k];                       // every event exists, even with zero in-window flashes
        if (inw) grp[k].insert(fg);
    }
    std::map<int,int> hist;
    for (auto& kv : grp) hist[kv.second.size()]++;
    long tot = 0, evs = 0;
    printf("files %d  T_flash rows %lld\n", nf, N);
    for (auto& h : hist) {
        printf("  in-window flash groups = %d : %d events\n", h.first, h.second);
        tot += (long)h.first * h.second; evs += h.second;
    }
    printf("  events %ld   mean in-window flash groups %.4f\n", evs, evs ? (double)tot/evs : 0.0);
}
