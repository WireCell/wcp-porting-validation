// Dump LArSoft sim::SimEnergyDeposit truth into the (N, x, y, z, Q) tree that
// wire-cell-sbnd-magnify-tracking-convert reads in MC mode (-f1 -a <this> -n T).
//
// The converter pairs every fitted point with the nearest truth point and
// accumulates each truth point's Q onto its nearest fitted point; the
// Magnify-tracking GUI then draws true_dQ/1000/rec_dx as the truth dQ/dx curve
// next to the fitted one (Data.cc DrawDQDX).  So Q must be in ELECTRONS and the
// truth set must contain only the particle whose track was fitted -- the
// converter applies NO distance cut, so any stray deposit dumps its charge onto
// whichever fitted point happens to be nearest (typically an endpoint, which
// then reads as a fake Bragg peak).  Hence the -R cut here.
//
// Why bare ROOT: the art file's product branch is unsplit
// (sim::SimEnergyDeposits_ionandscint_priorSCE_G4.obj, splitlevel 1) but the
// file carries the StreamerInfo for sim::SimEnergyDeposit v20, so TTree::Draw
// reads it through an emulated class with no LArSoft dictionaries.  As in
// SBNDReco1Reader.h, never call Events->GetEntry(): only per-branch reads are
// safe on these files, and TTree::Draw touches only the branches named.
//
// Coordinates: SimEnergyDeposit positions are LArSoft world cm and are written
// through unchanged.  numElectrons is the post-recombination ionization at the
// deposit -- before drift attenuation, so the reco charge is expected LOWER.
// The instance is 'priorSCE': if space-charge was simulated, truth sits at the
// undistorted position and the pairing distance carries that offset.  Both are
// physics offsets to report, never to scale away.
//
// Usage:
//   root -l -b -q 'dump_truth_sed.C("<art.root>", <run>, <event>,
//                                   "<tracking-stm.root>", <block|-1>,
//                                   <R_cm>, "<out.root>")'
// e.g.
//   root -l -b -q 'dump_truth_sed.C("input_files/input-10evt-mc/2025f-mc.root",
//                                   228, 11,
//                                   "work-mcsim-stmon/nusel_evt11/tracking-stm.root",
//                                   80, 5.0, "truth-evt11-blk80.root")'
//
// block = ndf = cluster_id*10 + pass (SbndMagnifyTrackingVisitor); -1 uses every
// fitted point in the file.

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <vector>

#include "TFile.h"
#include "TTree.h"

namespace {

    // One deposit, reduced to what the truth tree needs.
    struct Dep {
        double x, y, z;     // mid-point of the G4 step, cm
        double q;           // ionization electrons
        double t;           // start time, ns (orders the trajectory)
        int    orig;        // origTrackID: secondaries attributed to the parent
        int    pdg;
    };

    double min_dist2(const std::vector<double>& fx, const std::vector<double>& fy,
                     const std::vector<double>& fz, double x, double y, double z)
    {
        double best = 1e30;
        for (size_t i = 0; i < fx.size(); ++i) {
            const double d = (fx[i] - x) * (fx[i] - x) + (fy[i] - y) * (fy[i] - y) +
                             (fz[i] - z) * (fz[i] - z);
            if (d < best) best = d;
        }
        return best;
    }

}  // namespace

int dump_truth_sed(const char* art_file, int run, int event, const char* track_file,
                   int block = -1, double radius_cm = 5.0,
                   const char* out_file = "truth.root",
                   const char* product = "sim::SimEnergyDeposits_ionandscint_priorSCE_G4.obj")
{
    // ---------------------------------------------------------------- fitted track
    TFile* ftrk = TFile::Open(track_file);
    if (!ftrk || ftrk->IsZombie()) { printf("ERROR: cannot open %s\n", track_file); return 1; }
    TTree* trec = (TTree*) ftrk->Get("T_rec_charge");
    if (!trec) { printf("ERROR: no T_rec_charge in %s\n", track_file); return 1; }

    Double_t tx, ty, tz, tndf, trr;
    trec->SetBranchAddress("x", &tx);
    trec->SetBranchAddress("y", &ty);
    trec->SetBranchAddress("z", &tz);
    trec->SetBranchAddress("ndf", &tndf);
    const bool has_rr = trec->GetBranch("rr") != 0;
    if (has_rr) trec->SetBranchAddress("rr", &trr);

    std::vector<double> fx, fy, fz;
    double rr_min = 1e30, rr_end[3] = {0, 0, 0};
    double rr_max = -1e30, rr_far[3] = {0, 0, 0};
    for (Long64_t i = 0; i < trec->GetEntries(); ++i) {
        trec->GetEntry(i);
        if (block >= 0 && (int) std::lround(tndf) != block) continue;
        fx.push_back(tx); fy.push_back(ty); fz.push_back(tz);
        if (has_rr) {
            if (trr < rr_min) { rr_min = trr; rr_end[0] = tx; rr_end[1] = ty; rr_end[2] = tz; }
            if (trr > rr_max) { rr_max = trr; rr_far[0] = tx; rr_far[1] = ty; rr_far[2] = tz; }
        }
    }
    if (fx.empty()) { printf("ERROR: no fitted points for block %d\n", block); return 1; }
    printf("fitted points: %zu (block %d) from %s\n", fx.size(), block, track_file);

    // ---------------------------------------------------------------- art entry
    TFile* fart = TFile::Open(art_file);
    if (!fart || fart->IsZombie()) { printf("ERROR: cannot open %s\n", art_file); return 1; }
    TTree* events = (TTree*) fart->Get("Events");
    if (!events) { printf("ERROR: no Events tree in %s\n", art_file); return 1; }
    // NOT SetEstimate(-1): that sets the buffer to the tree's ENTRY count (13
    // here), and TTree::Draw then returns the right count while GetVal() holds
    // uninitialized memory past the 13th value -- coordinates of order 1e300.
    events->SetEstimate(1000000);

    Long64_t entry = -1;
    {
        const Long64_t n = events->Draw("EventAuxiliary.id_.event_:"
                                        "EventAuxiliary.id_.subRun_.run_.run_",
                                        "", "goff");
        for (Long64_t i = 0; i < n; ++i) {
            if ((int) events->GetVal(1)[i] == run && (int) events->GetVal(0)[i] == event) {
                entry = i;
                break;
            }
        }
    }
    if (entry < 0) { printf("ERROR: run %d event %d not in %s\n", run, event, art_file); return 1; }
    printf("art entry %lld = run %d event %d\n", entry, run, event);

    // ---------------------------------------------------------------- deposits
    TString e;
    const char* v[] = {"startPos.fCoordinates.fX", "startPos.fCoordinates.fY",
                       "startPos.fCoordinates.fZ", "endPos.fCoordinates.fX",
                       "endPos.fCoordinates.fY",   "endPos.fCoordinates.fZ",
                       "numElectrons", "origTrackID", "pdgCode", "startTime"};
    for (int i = 0; i < 10; ++i) {
        if (i) e += ":";
        e += TString::Format("%s.%s", product, v[i]);
    }
    Long64_t nd = events->Draw(e, "", "goff", 1, entry);
    if (nd <= 0) { printf("ERROR: no deposits read for %s\n", product); return 1; }
    if (nd > events->GetEstimate()) {   // buffer was too small: grow and redo
        events->SetEstimate(nd + 1);
        nd = events->Draw(e, "", "goff", 1, entry);
    }

    std::vector<Dep> deps;
    deps.reserve(nd);
    for (Long64_t i = 0; i < nd; ++i) {
        Dep d;
        d.x = 0.5 * (events->GetVal(0)[i] + events->GetVal(3)[i]);
        d.y = 0.5 * (events->GetVal(1)[i] + events->GetVal(4)[i]);
        d.z = 0.5 * (events->GetVal(2)[i] + events->GetVal(5)[i]);
        d.q = events->GetVal(6)[i];
        d.orig = (int) events->GetVal(7)[i];
        d.pdg = (int) events->GetVal(8)[i];
        d.t = events->GetVal(9)[i];
        deps.push_back(d);
    }
    printf("deposits in event: %lld\n", nd);

    // ------------------------------------------- dominant particle near the track
    const double r2 = radius_cm * radius_cm;
    std::map<int, double> q_near;
    std::map<int, int> pdg_of;
    std::vector<char> near(deps.size(), 0);
    for (size_t i = 0; i < deps.size(); ++i) {
        if (deps[i].q <= 0) continue;
        pdg_of[deps[i].orig] = deps[i].pdg;
        if (min_dist2(fx, fy, fz, deps[i].x, deps[i].y, deps[i].z) > r2) continue;
        near[i] = 1;
        q_near[deps[i].orig] += deps[i].q;
    }
    if (q_near.empty()) {
        printf("ERROR: no deposit within %.1f cm of the fitted track --"
               " truth and reco are not in the same frame\n", radius_cm);
        return 1;
    }
    double q_near_tot = 0;
    for (std::map<int, double>::const_iterator it = q_near.begin(); it != q_near.end(); ++it)
        q_near_tot += it->second;

    // Elect by COVERAGE, not by charge: each fitted point votes for the particle
    // owning the deposit nearest to it.  Charge would elect a dense blob -- a
    // 1 cm vertex proton outvoted an 85 cm muon on the first event tried.
    std::map<int, int> votes;
    std::vector<double> pair_dist(fx.size(), 0);
    for (size_t k = 0; k < fx.size(); ++k) {
        double best = 1e30;
        int owner = -1;
        for (size_t i = 0; i < deps.size(); ++i) {
            if (deps[i].q <= 0) continue;
            const double d = (deps[i].x - fx[k]) * (deps[i].x - fx[k]) +
                             (deps[i].y - fy[k]) * (deps[i].y - fy[k]) +
                             (deps[i].z - fz[k]) * (deps[i].z - fz[k]);
            if (d < best) { best = d; owner = deps[i].orig; }
        }
        pair_dist[k] = std::sqrt(best);
        if (owner >= 0) votes[owner]++;
    }
    std::vector<std::pair<int, int> > rank;   // (votes, origTrackID)
    for (std::map<int, int>::const_iterator it = votes.begin(); it != votes.end(); ++it)
        rank.push_back(std::make_pair(it->second, it->first));
    std::sort(rank.rbegin(), rank.rend());
    const int dom = rank[0].second;

    {   // pairing quality: if this is not ~cm, nothing downstream means anything
        std::vector<double> s(pair_dist);
        std::sort(s.begin(), s.end());
        printf("nearest-deposit distance over the %zu fitted points:"
               " median %.2f cm, p90 %.2f cm, max %.2f cm\n",
               s.size(), s[s.size() / 2], s[(size_t)(0.9 * s.size())], s.back());
    }
    printf("particles owning the nearest deposit (top 3 of %zu):\n", rank.size());
    for (size_t i = 0; i < rank.size() && i < 3; ++i) {
        const int id = rank[i].second;
        printf("   id %9d  pdg %11d  %4d/%zu points (%5.1f%%),  %10.3e e- within %.1f cm\n",
               id, pdg_of.count(id) ? pdg_of[id] : 0, rank[i].first, fx.size(),
               100. * rank[i].first / fx.size(), q_near.count(id) ? q_near[id] : 0., radius_cm);
    }

    // -------------------------------------------------------------- selection
    std::vector<Dep> keep;
    double q_dom_all = 0, q_keep = 0;
    for (size_t i = 0; i < deps.size(); ++i) {
        if (deps[i].orig != dom || deps[i].q <= 0) continue;
        q_dom_all += deps[i].q;
        if (!near[i]) continue;
        q_keep += deps[i].q;
        keep.push_back(deps[i]);
    }
    std::sort(keep.begin(), keep.end(),
              [](const Dep& a, const Dep& b) { return a.t < b.t; });

    printf("selected particle: origTrackID %d, pdg %d\n", dom, pdg_of[dom]);
    printf("   deposits kept:  %zu\n", keep.size());
    printf("   charge kept:    %.4e e- of %.4e e- for this particle"
           " (%.2f%%; %.2f%% dropped by the %.1f cm cut)\n",
           q_keep, q_dom_all, 100. * q_keep / q_dom_all,
           100. * (1. - q_keep / q_dom_all), radius_cm);
    printf("   charge purity:  %.2f%% of the charge within %.1f cm of the track"
           " belongs to it (the rest is other particles the fit also collects)\n",
           q_near_tot > 0 ? 100. * q_near[dom] / q_near_tot : 0., radius_cm);

    {   // full extent of the particle, cut or not: does it stop in the detector?
        std::vector<Dep> all;
        for (size_t i = 0; i < deps.size(); ++i)
            if (deps[i].orig == dom && deps[i].q > 0) all.push_back(deps[i]);
        std::sort(all.begin(), all.end(),
                  [](const Dep& a, const Dep& b) { return a.t < b.t; });
        if (!all.empty())
            printf("   full true path (no cut): (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f) cm,"
                   " %zu deposits\n", all.front().x, all.front().y, all.front().z,
                   all.back().x, all.back().y, all.back().z, all.size());
    }

    if (!keep.empty()) {
        const Dep& a = keep.front();
        const Dep& b = keep.back();
        printf("   kept path:      first (%.2f, %.2f, %.2f) -> last (%.2f, %.2f, %.2f) cm,"
               " t %.1f -> %.1f ns\n", a.x, a.y, a.z, b.x, b.y, b.z, a.t, b.t);
        if (has_rr) {
            const double d_end = std::sqrt((b.x - rr_end[0]) * (b.x - rr_end[0]) +
                                           (b.y - rr_end[1]) * (b.y - rr_end[1]) +
                                           (b.z - rr_end[2]) * (b.z - rr_end[2]));
            const double d_far = std::sqrt((b.x - rr_far[0]) * (b.x - rr_far[0]) +
                                           (b.y - rr_far[1]) * (b.y - rr_far[1]) +
                                           (b.z - rr_far[2]) * (b.z - rr_far[2]));
            printf("   truth end vs fitted rr=%.2f end: %.2f cm; vs rr=%.2f end: %.2f cm\n",
                   rr_min, d_end, rr_max, d_far);
        }
    }

    // ------------------------------------------------------------------ write
    TFile* fout = TFile::Open(out_file, "RECREATE");
    TTree* t = new TTree("T", "truth deposits for one MC particle");
    Int_t N = (Int_t) keep.size();
    std::vector<double> ox, oy, oz, oq;
    for (size_t i = 0; i < keep.size(); ++i) {
        ox.push_back(keep[i].x); oy.push_back(keep[i].y);
        oz.push_back(keep[i].z); oq.push_back(keep[i].q);
    }
    std::vector<double>* pox = &ox;
    std::vector<double>* poy = &oy;
    std::vector<double>* poz = &oz;
    std::vector<double>* poq = &oq;
    t->Branch("N", &N, "N/I");
    t->Branch("x", &pox);
    t->Branch("y", &poy);
    t->Branch("z", &poz);
    t->Branch("Q", &poq);
    t->Fill();
    fout->Write();
    fout->Close();
    printf("wrote %s: %d truth points, %.4e e-\n", out_file, N, q_keep);
    return 0;
}
