// Find which art file (and entry) of a file list holds a given run/subrun --
// bare ROOT, no LArSoft, works on /pnfs or xroot:// paths (dCache) as well as
// local copies.
//
//   root -l -b -q 'mc_find_rse.C("<list.lst>", 32, 10)'      // run 32, subrun 10
//   root -l -b -q 'mc_find_rse.C("<list.lst>", -1, -1)'      // dump every file's r/s/e
//
// Prints one line per matching entry:
//   MATCH <file-index> <entry> run <r> subrun <s> event <e>  <path>
// and, in dump mode, one line per file with its run and the subruns/events it
// holds.  Reading is per-branch on EventAuxiliary only (never
// Events->GetEntry(): the tree has ~2500 branches and deserializing all of them
// segfaults -- same rule as toolkit/root/src/SBNDReco1Reader.h), so it costs a
// few seconds per file no matter how big the file is.
//
// Written for docs/67_round2-patrec-10evt.md sec 9: yuhw's Bee set carries
// run 32/subrun 10 and run 31/subrun 88, which are NOT in the ten files staged
// on wcgpu1 (all run 713).  Run this over the full 100-file gpvm list on a node
// that can see /pnfs to identify the two files, then stage them and rerun the
// chain with the Repro block of doc 67.
void mc_find_rse(const char* listfile, int want_run = -1, int want_subrun = -1)
{
    std::ifstream in(listfile);
    if (!in) { printf("cannot open list %s\n", listfile); return; }
    std::string path;
    int ifile = 0;
    while (std::getline(in, path)) {
        if (path.empty()) continue;
        ++ifile;
        TFile* f = TFile::Open(path.c_str());
        if (!f || f->IsZombie()) { printf("FILE %d UNREADABLE %s\n", ifile, path.c_str()); continue; }
        TTree* t = (TTree*) f->Get("Events");
        if (!t) { printf("FILE %d NO-EVENTS-TREE %s\n", ifile, path.c_str()); delete f; continue; }
        Long64_t n = t->Draw("EventAuxiliary.id_.subRun_.run_.run_:"
                             "EventAuxiliary.id_.subRun_.subRun_:"
                             "EventAuxiliary.id_.event_", "", "goff");
        std::string evts;
        int run = -1, subrun = -1;
        for (Long64_t i = 0; i < n; ++i) {
            int r = (int) t->GetV1()[i], s = (int) t->GetV2()[i], e = (int) t->GetV3()[i];
            run = r; subrun = s;
            evts += TString::Format(" %d", e).Data();
            if (want_run >= 0 && (r != want_run || s != want_subrun)) continue;
            if (want_run >= 0)
                printf("MATCH file %d entry %lld run %d subrun %d event %d  %s\n",
                       ifile, i, r, s, e, path.c_str());
        }
        if (want_run < 0)
            printf("FILE %3d run %d subrun %d nevt %lld events:%s  %s\n",
                   ifile, run, subrun, n, evts.c_str(), gSystem->BaseName(path.c_str()));
        delete f;
    }
}
