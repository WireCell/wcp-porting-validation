// mcs_probe.cxx -- doc 83 (MCS cost + high-side outliers) diagnostic tool.
//
// Unlike mcs_dump.cxx (which links ROOT and the RAW upstream mcs.cxx for the
// doc-80 round-0 golden reference), this tool links the ALREADY-INSTALLED
// ROOT-free toolkit port, WireCellMcs (local/lib/libWireCellMcs.so) -- it is
// the estimator that actually ran in work-mcp1k-mcs80on, and every symbol it
// needs (WireCell::Mcs::detail::*) is a plain exported free function, so no
// ROOT and no ubreco source tree are required.  Build: `make` here (see the
// Makefile `mcs_probe` target).
//
// Modes:
//   replay   CLOUD.txt OUT.json [--maskmax]
//       Run MuonMCS::run() once (the official result), plus the same
//       trim_trajectory+form_segs replicated here to expose the per-segment
//       angle/distance/vx arrays and a dense lnL(KE) curve (1..4001 MeV,
//       2000 pts, same convention as mcs_dump.cxx).  --maskmax additionally
//       re-runs estimate_energy with the single largest-|angle| index masked
//       via the library's own angle_keep mechanism (the cathode-excision
//       path, doc 80 sec 7.5) -- the decisive over-clustering/kink test.
//   bench    CLOUD.txt N OUT.json
//       Repeat MuonMCS::run() N times on the same cloud; report wall-clock
//       per call for run() as a whole and for trim_trajectory/form_segs/
//       estimate_energy individually (called back-to-back on the same
//       trimmed points so the split is exact, not resampled).
//   synthetic NSEGS T_MEV NTRIALS SEED OUT.json [--ivx N]
//       Toy null test (doc 83 Part 2 step 2): for NSEGS 14-cm segments along
//       a track whose START-of-track KE is T_MEV, draw per-segment angles
//       from the SAME tuned double-Gaussian the estimator scores against
//       (pred_theta_{xz,yz}_pars, exported), at the locally-degraded energy
//       ke_from_rr(rr_from_ke(T_MEV) - distance_i) -- i.e. the tune's own
//       Bragg slowing, not a separate model.  vx is fixed (--ivx selects
//       which of the 5 slices to sample from; default ivx=2, a "typical"
//       intermediate-angle track) because the toy is testing angle-count
//       starvation, not vx dependence.  Runs NTRIALS reps of
//       WireCell::Mcs::detail::estimate_energy on synthetic angle vectors
//       (the shipped estimator, not a re-implementation) and reports the
//       keguess/T_MEV ratio distribution.
//
// JSON numbers use the same inf/nan-as-string convention as mcs_dump.cxx
// (jnum) so the output is valid JSON even where the estimator legitimately
// produces a non-finite value (e.g. a railed 4 GeV scan).
#include "WireCellMcs/MuonMCS.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace WireCell::Mcs;
using namespace WireCell::Mcs::detail;
using Clock = std::chrono::steady_clock;

namespace {

void jnum(FILE* f, double v)
{
    if (std::isnan(v)) { fprintf(f, "\"nan\""); }
    else if (std::isinf(v)) { fprintf(f, v > 0 ? "\"inf\"" : "\"-inf\""); }
    else { fprintf(f, "%.17g", v); }
}

void jvec(FILE* f, const std::vector<double>& v)
{
    fprintf(f, "[");
    for (size_t i = 0; i < v.size(); i++) { if (i) fprintf(f, ","); jnum(f, v[i]); }
    fprintf(f, "]");
}

bool read_cloud(const std::string& path, std::vector<double>& start, std::vector<double>& end,
                std::vector<std::vector<double>>& points)
{
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        double x, y, z;
        if (!(iss >> x >> y >> z)) continue;
        lineno++;
        if (lineno == 1) { start = {x, y, z}; }
        else if (lineno == 2) { end = {x, y, z}; }
        else { points.push_back({x, y, z}); }
    }
    return lineno >= 2;
}

// The upstream aggregate likelihood (mcs.cxx:541-566 / MuonMCS.cxx:1024-1053),
// rebuilt HERE from the exported single-angle terms so it is guaranteed to be
// the same function the estimator minimises -- not a re-derivation.
double lnlikelihood_track(double KE, const std::vector<double>& segs_distance,
                          const std::vector<double>& segs_angle_x,
                          const std::vector<double>& segs_angle_y,
                          const std::vector<double>& vx_comps,
                          const McsOptions& opt, McsCounters& ctr,
                          const std::vector<char>& angle_keep)
{
    double lnl = 0;
    int n = (int)segs_distance.size();
    double rrtot_guess = rr_from_ke(KE);
    for (int i = 1; i < n; i++) {
        if (!angle_keep.empty() && !angle_keep[i]) continue;
        double distance1 = segs_distance[i - 1];
        double distance2 = segs_distance[i];
        double rrguess1 = std::max(rrtot_guess - distance1, 1.);
        double rrguess2 = std::max(rrtot_guess - distance2, 1.);
        double keguess1 = ke_from_rr(rrguess1);
        double keguess2 = ke_from_rr(rrguess2);
        double keguess = (keguess1 + keguess2) / 2;
        double vx = (i - 1 < (int)vx_comps.size()) ? vx_comps[i - 1] : 0.0;
        lnl += lnlikelihood_theta_xz(segs_angle_x[i], keguess, opt, ctr);
        lnl += lnlikelihood_theta_yz(segs_angle_y[i], keguess, vx, opt, ctr);
    }
    return lnl;
}

int mode_replay(int argc, char** argv)
{
    if (argc < 4) { fprintf(stderr, "usage: mcs_probe replay CLOUD.txt OUT.json [--maskmax]\n"); return 1; }
    std::string cloud_path = argv[2], out_path = argv[3];
    bool maskmax = (argc > 4 && std::string(argv[4]) == "--maskmax");

    std::vector<double> start, end;
    std::vector<std::vector<double>> points;
    if (!read_cloud(cloud_path, start, end, points)) {
        fprintf(stderr, "cannot read cloud %s\n", cloud_path.c_str());
        return 1;
    }

    McsOptions opt;  // defaults: all fixes on, cathode excision off
    MuonMCS mcs(opt);
    McsResult res = mcs.run(start, end, points);

    // Replicate run()'s internal trim+form_segs to expose the segment arrays.
    TrimResult trim = trim_trajectory(points, start, end);
    McsCounters ctr2;
    SegResult segs = form_segs(trim.points_final, start, end, /*seg_length=*/14.0, opt, ctr2);
    std::vector<double> vx_components;
    for (const auto& a : segs.aAxes) { if (!a.empty()) vx_components.push_back(a[0]); }

    FILE* jf = fopen(out_path.c_str(), "w");
    if (!jf) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
    fprintf(jf, "{\"cloud\":\"%s\",\"npoints_in\":%d,\n", cloud_path.c_str(), (int)points.size());
    fprintf(jf, "\"result\":{\"mu_tracklen\":"); jnum(jf, res.mu_tracklen);
    fprintf(jf, ",\"ke_MCS\":"); jnum(jf, res.ke_MCS);
    fprintf(jf, ",\"ke_tracklen\":"); jnum(jf, res.ke_tracklen);
    fprintf(jf, ",\"ambiguity_MCS\":"); jnum(jf, res.ambiguity_MCS);
    fprintf(jf, ",\"nsegs\":%d,\"bad_path\":%s},\n", res.nsegs, res.bad_path ? "true" : "false");

    fprintf(jf, "\"segments\":{\"distance\":"); jvec(jf, segs.distance);
    fprintf(jf, ",\"angle_projB\":"); jvec(jf, segs.angle_projB);
    fprintf(jf, ",\"angle_projC\":"); jvec(jf, segs.angle_projC);
    fprintf(jf, ",\"vx\":"); jvec(jf, vx_components);
    fprintf(jf, "},\n");

    // Dense lnL(KE) curve, no mask -- same grid as mcs_dump.cxx.
    McsCounters ctr_curve;
    fprintf(jf, "\"likelihood_curve\":[");
    for (int i = 0; i < 2000; i++) {
        double ke = 1.0 + i * 2.0;
        double lnl = lnlikelihood_track(ke, segs.distance, segs.angle_projB, segs.angle_projC,
                                        vx_components, opt, ctr_curve, {});
        if (i) fprintf(jf, ",");
        fprintf(jf, "["); jnum(jf, ke); fprintf(jf, ","); jnum(jf, lnl); fprintf(jf, "]");
    }
    fprintf(jf, "]");

    if (maskmax) {
        // Mask the single largest-|angle| index (angle_projB, angle_projC
        // combined in quadrature) and re-minimise via the shipped
        // angle_keep mechanism -- the exact cathode-excision code path, doc
        // 80 sec 7.5, repurposed here as the kink/over-clustering probe.
        int n = (int)segs.distance.size();
        int worst = -1;
        double worst_mag = -1;
        for (int i = 1; i < n; i++) {
            double mag = std::hypot(segs.angle_projB[i], segs.angle_projC[i]);
            if (mag > worst_mag) { worst_mag = mag; worst = i; }
        }
        std::vector<char> keep(n, 1);
        if (worst >= 0) keep[worst] = 0;
        McsCounters ctr3;
        EnergyResult er = estimate_energy(segs.distance, segs.angle_projB, segs.angle_projC,
                                          vx_components, opt, ctr3, keep);
        fprintf(jf, ",\n\"maskmax\":{\"masked_index\":%d,\"masked_angle_mag\":", worst);
        jnum(jf, worst_mag);
        fprintf(jf, ",\"ke_MCS\":"); jnum(jf, er.keguess);
        fprintf(jf, ",\"ambiguity_MCS\":"); jnum(jf, er.ambiguity);
        fprintf(jf, "}");
    }
    fprintf(jf, "\n}\n");
    fclose(jf);
    return 0;
}

int mode_bench(int argc, char** argv)
{
    if (argc < 5) { fprintf(stderr, "usage: mcs_probe bench CLOUD.txt N OUT.json\n"); return 1; }
    std::string cloud_path = argv[2];
    int n_reps = std::atoi(argv[3]);
    std::string out_path = argv[4];

    std::vector<double> start, end;
    std::vector<std::vector<double>> points;
    if (!read_cloud(cloud_path, start, end, points)) {
        fprintf(stderr, "cannot read cloud %s\n", cloud_path.c_str());
        return 1;
    }
    McsOptions opt;
    MuonMCS mcs(opt);

    std::vector<double> t_run_us, t_trim_us, t_form_us, t_energy_us;
    t_run_us.reserve(n_reps);
    for (int i = 0; i < n_reps; i++) {
        auto t0 = Clock::now();
        McsResult res = mcs.run(start, end, points);
        auto t1 = Clock::now();
        t_run_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        (void)res;

        auto s0 = Clock::now();
        TrimResult trim = trim_trajectory(points, start, end);
        auto s1 = Clock::now();
        McsCounters ctr;
        SegResult segs = form_segs(trim.points_final, start, end, 14.0, opt, ctr);
        auto s2 = Clock::now();
        std::vector<double> vx;
        for (const auto& a : segs.aAxes) { if (!a.empty()) vx.push_back(a[0]); }
        McsCounters ctr2;
        EnergyResult er = estimate_energy(segs.distance, segs.angle_projB, segs.angle_projC, vx, opt, ctr2);
        auto s3 = Clock::now();
        (void)er;
        t_trim_us.push_back(std::chrono::duration<double, std::micro>(s1 - s0).count());
        t_form_us.push_back(std::chrono::duration<double, std::micro>(s2 - s1).count());
        t_energy_us.push_back(std::chrono::duration<double, std::micro>(s3 - s2).count());
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
    };
    FILE* jf = fopen(out_path.c_str(), "w");
    if (!jf) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
    fprintf(jf, "{\"cloud\":\"%s\",\"npoints_in\":%d,\"nreps\":%d,\n", cloud_path.c_str(), (int)points.size(), n_reps);
    fprintf(jf, "\"median_us\":{\"run\":%.3f,\"trim_trajectory\":%.3f,\"form_segs\":%.3f,\"estimate_energy\":%.3f},\n",
            median(t_run_us), median(t_trim_us), median(t_form_us), median(t_energy_us));
    fprintf(jf, "\"run_us\":"); jvec(jf, t_run_us);
    fprintf(jf, "\n}\n");
    fclose(jf);
    return 0;
}

int mode_synthetic(int argc, char** argv)
{
    if (argc < 7) {
        fprintf(stderr, "usage: mcs_probe synthetic NSEGS T_MEV NTRIALS SEED OUT.json [--ivx N]\n");
        return 1;
    }
    int nsegs = std::atoi(argv[2]);
    double T_mev = std::atof(argv[3]);
    int ntrials = std::atoi(argv[4]);
    long seed = std::atol(argv[5]);
    std::string out_path = argv[6];
    int ivx = 2;
    // doc 83 correction: the toy's angles are drawn from the SHIPPED tune,
    // but doc 80's own pull test (sec 9.3) found the tune's sigma is too
    // NARROW at T>~400 MeV (pull core width 1.34 at 200-400, 2.15 at
    // 400-800) -- exactly where several outlier buckets sit.  A toy drawn
    // from the nominal tune therefore UNDERSTATES the true outlier rate in
    // those buckets.  --sigma-scale inflates BOTH double-Gaussian widths by
    // a constant factor so the toy can be re-run at the MEASURED pull width
    // for an honest bracket instead of a single (biased-low) number.
    double sigma_scale = 1.0;
    for (int i = 7; i + 1 < argc; i++) {
        if (std::string(argv[i]) == "--ivx") ivx = std::atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--sigma-scale") sigma_scale = std::atof(argv[i + 1]);
    }
    ivx = std::max(0, std::min(4, ivx));
    const double vx_edges[6] = {0, 0.1, 0.2, 0.35, 0.75, 1};
    double vx_val = 0.5 * (vx_edges[ivx] + vx_edges[ivx + 1]);

    std::mt19937 rng(seed);
    std::normal_distribution<double> nrm(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    double rrtot_true = rr_from_ke(T_mev);
    std::vector<double> distances(nsegs + 1);
    distances[0] = 0;  // sentinel, never scored
    for (int i = 1; i <= nsegs; i++) { distances[i] = (i - 0.5) * 14.0; }

    McsOptions opt;
    std::vector<double> ratios;
    ratios.reserve(ntrials);
    for (int t = 0; t < ntrials; t++) {
        std::vector<double> ax(nsegs + 1, 0.0), ay(nsegs + 1, 0.0);
        std::vector<double> vx_comps(nsegs, vx_val);
        for (int i = 1; i <= nsegs; i++) {
            double rr_local = std::max(rrtot_true - distances[i], 1.0);
            double T_local = ke_from_rr(rr_local);
            VD pxz = pred_theta_xz_pars(T_local);
            VD pyz = pred_theta_yz_pars(T_local, ivx);
            auto draw_mixture = [&](const VD& pars) {
                double sigma = (uni(rng) < pars[2]) ? pars[0] : pars[1];
                return sigma_scale * sigma * nrm(rng);
            };
            ax[i] = draw_mixture(pxz);
            ay[i] = draw_mixture(pyz);
        }
        McsCounters ctr;
        EnergyResult er = estimate_energy(distances, ax, ay, vx_comps, opt, ctr, {});
        ratios.push_back(er.keguess / T_mev);
    }

    FILE* jf = fopen(out_path.c_str(), "w");
    if (!jf) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
    fprintf(jf, "{\"nsegs\":%d,\"T_MeV\":%.3f,\"ntrials\":%d,\"seed\":%ld,\"ivx\":%d,\"vx\":%.4f,"
                "\"sigma_scale\":%.4f,\n",
            nsegs, T_mev, ntrials, seed, ivx, vx_val, sigma_scale);
    fprintf(jf, "\"keguess_over_T\":"); jvec(jf, ratios);
    fprintf(jf, "\n}\n");
    fclose(jf);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: mcs_probe {replay|bench|synthetic} ...\n");
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "replay") return mode_replay(argc, argv);
    if (mode == "bench") return mode_bench(argc, argv);
    if (mode == "synthetic") return mode_synthetic(argc, argv);
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
