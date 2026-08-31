// mcs_dump.cxx -- doc 80 round 0 step 0c/0d instrumented reference dumper.
//
// Runs the UNMODIFIED upstream MCS (../mcs_standalone/src/mcs.{h,cxx} @ 6aa0b9c)
// on an input cloud and dumps every intermediate the ROOT-free port must match
// (doc 80 sec 6.2 gates #1-#7) to a JSON fixture.
//
// The Minimize replica below transcribes ROOT v6-32-02
// math/mathcore/src/BrentMethods.cxx + BrentMinimizer1D.cxx (LGPL-2.1-or-later)
// for CROSS-CHECK ONLY: its results are compared against ROOT's actual
// TF1::GetMinimumX in the same run.  This tool never enters the toolkit.
//
// Usage:
//   ./mcs_dump --root ../mcs_standalone/data/simulated_event.root out.json
//   ./mcs_dump --txt cloud.txt out.json        (line1: start xyz, line2: end xyz, then x y z per line)
//   add --shuffle SEED to permute the input point order (std::mt19937) first.

#include "TROOT.h"
#include "TFile.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TGraph.h"
#include "TF1.h"
#include "mcs.h"
#include "mcs.cxx"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using VD = std::vector<double>;
using VVD = std::vector<std::vector<double>>;

static void jnum(FILE* f, double v)
{
    // upstream -log(0)/acos(>1) produce inf/nan (doc 80 sec 6.4 bugs #9/#10);
    // map to strings so the fixture stays valid JSON (pr94_root_gate convention)
    if (std::isnan(v)) fprintf(f, "\"nan\"");
    else if (std::isinf(v)) fprintf(f, v > 0 ? "\"inf\"" : "\"-inf\"");
    else fprintf(f, "%.17g", v);
}
static void jvec(FILE* f, const VD& v) {
    fprintf(f, "[");
    for (size_t i = 0; i < v.size(); i++) { if (i) fprintf(f, ","); jnum(f, v[i]); }
    fprintf(f, "]");
}
static void jvecvec(FILE* f, const VVD& v) {
    fprintf(f, "[");
    for (size_t i = 0; i < v.size(); i++) { if (i) fprintf(f, ","); jvec(f, v[i]); }
    fprintf(f, "]");
}

// ---- replica of ROOT v6-32-02 BrentMethods::MinimStep, type 0, no log ----
struct StepResult {
    double xmin0, xmax0;      // input range
    VD grid_x, grid_y;        // the npx-point scan
    int argmin;               // index of grid minimum (strict <, earlier wins)
    double xmin1, xmax1, xmiddle;  // bracket out
};
template <class F>
static StepResult minim_step(F&& func, double xmin, double xmax, int npx)
{
    StepResult r;
    r.xmin0 = xmin; r.xmax0 = xmax;
    double dx = (xmax - xmin) / (npx - 1);
    double xxmin = xmin;
    double yymin = func(xxmin);
    r.grid_x.push_back(xxmin); r.grid_y.push_back(yymin);
    r.argmin = 0;
    for (int i = 1; i <= npx - 1; i++) {
        double x = xmin + i * dx;
        double y = func(x);
        r.grid_x.push_back(x); r.grid_y.push_back(y);
        if (y < yymin) { xxmin = x; yymin = y; r.argmin = i; }
    }
    r.xmin1 = std::max(xmin, xxmin - dx);
    r.xmax1 = std::min(xmax, xxmin + dx);
    r.xmiddle = std::min(xxmin, r.xmax1);
    return r;
}

// ---- replica of ROOT v6-32-02 BrentMethods::MinimBrent, type 0 ----
template <class F>
static double minim_brent(F&& func, double& xmin, double& xmax, double xmiddle,
                          bool& ok, int& niter, double epsabs, double epsrel, int itermax)
{
    const double c = 3.81966011250105097e-01;
    double u, v, w, x, fv, fu, fw, fx, e, p, q, r, t2, d = 0, m, tol;
    v = w = x = xmiddle;
    e = 0;
    double a = xmin;
    double b = xmax;
    fv = fw = fx = func(x);
    for (int i = 0; i < itermax; i++) {
        m = 0.5 * (a + b);
        tol = epsrel * (std::fabs(x)) + epsabs;
        t2 = 2 * tol;
        if (std::fabs(x - m) <= (t2 - 0.5 * (b - a))) { ok = true; niter = i - 1; return x; }
        if (std::fabs(e) > tol) {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2 * (q - r);
            if (q > 0) p = -p; else q = -q;
            r = e;
            e = d;
            if (std::fabs(p) >= std::fabs(0.5 * q * r) || p <= q * (a - x) || p >= q * (b - x)) {
                e = (x >= m ? a - x : b - x);
                d = c * e;
            }
            else {
                d = p / q;
                u = x + d;
                if (u - a < t2 || b - u < t2) d = (m - x >= 0) ? std::fabs(tol) : -std::fabs(tol);
            }
        }
        else {
            e = (x >= m ? a - x : b - x);
            d = c * e;
        }
        u = (std::fabs(d) >= tol ? x + d : x + ((d >= 0) ? std::fabs(tol) : -std::fabs(tol)));
        fu = func(u);
        if (fu <= fx) {
            if (u < x) b = x; else a = x;
            v = w; fv = fw; w = x; fw = fx; x = u; fx = fu;
        }
        else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) { v = w; fv = fw; w = u; fw = fu; }
            else if (fu <= fv || v == x || v == w) { v = u; fv = fu; }
        }
    }
    ok = false;
    xmin = a; xmax = b; niter = itermax;
    return x;
}

// ---- replica of BrentMinimizer1D::Minimize as used by TF1::GetMinimumX ----
struct MinimizeResult {
    StepResult first_step;    // the FIRST MinimStep of the search loop (gate #3)
    double x;                 // converged minimum
    int nsearch;              // outer iterations used (1 = single bracket+brent)
    bool ok;
};
template <class F>
static MinimizeResult replica_get_minimum_x(F&& func, double xlow, double xup,
                                            int npx = 100, double eps = 1e-10, int maxiter = 100)
{
    MinimizeResult mr;
    double xmin = xlow, xmax = xup;
    int maxIter1 = 10;
    int niter1 = 0, niter2 = 0;
    bool ok = false;
    double x = 0;
    while (!ok) {
        if (niter1 > maxIter1) break;
        StepResult sr = minim_step(func, xmin, xmax, npx);
        if (niter1 == 0) mr.first_step = sr;
        xmin = sr.xmin1; xmax = sr.xmax1;
        x = minim_brent(func, xmin, xmax, sr.xmiddle, ok, niter2, eps, eps, maxiter);
        niter1++;
    }
    mr.x = x; mr.nsearch = niter1; mr.ok = ok;
    return mr;
}

// ---- instrumented MCS: protected members reachable from a subclass ----
class InstrumentedMCS : public MCS {
public:
    void dump_run(const VD& vtx_start, const VD& vtx_end, const VVD& points_in, FILE* jf)
    {
        fprintf(jf, "{\n\"meta\":{\"upstream\":\"uboone/Multiple_Coulomb_Scattering @ 6aa0b9c\","
                    "\"root_version\":\"6.32.02\",\"doc\":\"80_mcs-integration round 0\"},\n");
        fprintf(jf, "\"input\":{\"start\":"); jvec(jf, vtx_start);
        fprintf(jf, ",\"end\":"); jvec(jf, vtx_end);
        fprintf(jf, ",\"npoints\":%d,\"points\":", (int)points_in.size()); jvecvec(jf, points_in);
        fprintf(jf, "},\n");

        // ---- interp side probes (gate #1) ----
        {
            const int un = 20;
            double uCSDA[un] = { .9833, 1.786, 3.321, 6.598, 10.58, 30.84, 42.50, 67.32, 106.3, 172.5,
                                 238.5, 493.4, 616.3, 855.2, 1202., 1758., 2297., 4359., 5354., 7298. };
            double rr_lo = uCSDA[0] / rho, rr3 = uCSDA[3] / rho, rr_hi = uCSDA[un - 1] / rho;
            VD ke_probes = { -5.0, 0.0, 0.5, rr_lo, 0.75, 1.0, 2.0, 5.0, rr3, 14.3, 50.0, 123.456,
                             1000.0, rr_hi, 6000.0, 10000.0 };
            VD rr_probes = { 0.001, 1.0, 9.999, 10.0, 11.0, 123.456, 500.0, 1300.0, 13999.0, 14000.0, 20000.0 };
            fprintf(jf, "\"interp_probes\":{\"uKEfromRR\":[");
            for (size_t i = 0; i < ke_probes.size(); i++) {
                if (i) fprintf(jf, ",");
                fprintf(jf, "["); jnum(jf, ke_probes[i]); fprintf(jf, ","); jnum(jf, uKEfromRR->Eval(ke_probes[i])); fprintf(jf, "]");
            }
            fprintf(jf, "],\"uRRfromKE\":[");
            for (size_t i = 0; i < rr_probes.size(); i++) {
                if (i) fprintf(jf, ",");
                fprintf(jf, "["); jnum(jf, rr_probes[i]); fprintf(jf, ","); jnum(jf, uRRfromKE->Eval(rr_probes[i])); fprintf(jf, "]");
            }
            fprintf(jf, "]},\n");
        }

        // ---- stage 1: trim (mcs.cxx:296) ----
        int npoints_trajectory = points_in.size();
        auto trajectory_tuple = trim_trajectory(npoints_trajectory, points_in, vtx_start, vtx_end);
        bool bad_path = std::get<0>(trajectory_tuple);
        VVD tp_final = std::get<1>(trajectory_tuple);
        int npf = tp_final.size();
        fprintf(jf, "\"trim\":{\"bad_path\":%s,\"npoints_final\":%d,\"points_final\":",
                bad_path ? "true" : "false", npf);
        jvecvec(jf, tp_final);
        fprintf(jf, "},\n");

        // ---- rr / tracklen (mcs.cxx:82-95) ----
        double rr_path = 0;
        for (int i = 1; i < npf; i++) rr_path += MCSHelper::norm(MCSHelper::diff(tp_final[i], tp_final[i - 1]));
        double KE_rr_path = uKEfromRR->Eval(rr_path);
        bool short_path = bad_path || npf < 20 ||
            MCSHelper::norm(MCSHelper::diff(tp_final.back(), vtx_end)) < 2 * seg_length;
        fprintf(jf, "\"rr\":{\"rr_path\":"); jnum(jf, rr_path);
        fprintf(jf, ",\"KE_rr_path\":"); jnum(jf, KE_rr_path);
        fprintf(jf, ",\"emu_tracklen\":"); jnum(jf, (KE_rr_path + Mmu) / 1000.);
        fprintf(jf, ",\"early_return\":%s},\n", short_path ? "true" : "false");
        if (short_path) {
            fprintf(jf, "\"outputs\":{\"mu_tracklen\":"); jnum(jf, rr_path);
            fprintf(jf, ",\"emu_tracklen\":"); jnum(jf, (KE_rr_path + Mmu) / 1000.);
            fprintf(jf, ",\"emu_MCS\":-1,\"ambiguity_MCS\":-1}\n}\n");
            return;
        }

        // ---- stage 2+3: segments and angles (mcs.cxx:365) ----
        auto segs_tuple = form_segs(tp_final, vtx_start, vtx_end, seg_length);
        std::vector<Track> segs = std::get<0>(segs_tuple);
        VVD axes = std::get<1>(segs_tuple);
        VVD segs_aAxes = std::get<2>(segs_tuple);
        VVD segs_COM = std::get<3>(segs_tuple);
        VD segs_distance = std::get<4>(segs_tuple);
        VD segs_angle = std::get<5>(segs_tuple);
        VD segs_angle_projB = std::get<6>(segs_tuple);
        VD segs_angle_projC = std::get<7>(segs_tuple);

        fprintf(jf, "\"segments\":{\"nsegs_container\":%d,\"ndist\":%d,\"track_axes\":",
                (int)segs.size(), (int)segs_distance.size());
        jvecvec(jf, axes);
        fprintf(jf, ",\"per_seg\":[");
        for (size_t k = 0; k < segs_distance.size(); k++) {
            if (k) fprintf(jf, ",");
            double vx = segs_aAxes[k].empty() ? 0.0 : segs_aAxes[k][0];
            double vx_abs = std::abs(vx);
            VD vx_edges = { 0, 0.1, 0.2, 0.35, 0.75, 1 };
            int ivx = 0 * (vx_abs >= vx_edges[0] && vx_abs < vx_edges[1]) + 1 * (vx_abs >= vx_edges[1] && vx_abs < vx_edges[2])
                    + 2 * (vx_abs >= vx_edges[2] && vx_abs < vx_edges[3]) + 3 * (vx_abs >= vx_edges[3] && vx_abs < vx_edges[4])
                    + 4 * (vx_abs >= vx_edges[4] && vx_abs < vx_edges[5]);
            fprintf(jf, "{\"npoints\":%d,\"distance\":", segs[k].N); jnum(jf, segs_distance[k]);
            fprintf(jf, ",\"angle\":"); jnum(jf, segs_angle[k]);
            fprintf(jf, ",\"angle_projB\":"); jnum(jf, segs_angle_projB[k]);
            fprintf(jf, ",\"angle_projC\":"); jnum(jf, segs_angle_projC[k]);
            fprintf(jf, ",\"vx\":"); jnum(jf, vx);
            fprintf(jf, ",\"ivx\":%d,\"com\":", ivx); jvec(jf, segs_COM[k]);
            fprintf(jf, ",\"aAxis\":"); jvec(jf, segs_aAxes[k]);
            if (segs[k].N > 0) {
                fprintf(jf, ",\"first_point\":"); jvec(jf, segs[k].points.front());
                fprintf(jf, ",\"last_point\":"); jvec(jf, segs[k].points.back());
            }
            fprintf(jf, "}");
        }
        fprintf(jf, "]},\n");

        if (segs.size() < 2) {
            fprintf(jf, "\"outputs\":{\"mu_tracklen\":"); jnum(jf, rr_path);
            fprintf(jf, ",\"emu_tracklen\":"); jnum(jf, (KE_rr_path + Mmu) / 1000.);
            fprintf(jf, ",\"emu_MCS\":-1,\"ambiguity_MCS\":-1,\"early_return_nsegs\":true}\n}\n");
            return;
        }

        VD vx_components;
        for (const auto& seg_dir : segs_aAxes) if (!seg_dir.empty()) vx_components.push_back(seg_dir[0]);

        // ---- stage 4: energy (mcs.cxx:569), instrumented replica ----
        VD par = segs_distance;   // pack exactly as estimate_energy does
        par.insert(par.begin(), (double)segs_distance.size());
        par.insert(par.end(), segs_angle_projB.begin(), segs_angle_projB.end());
        par.insert(par.end(), segs_angle_projC.begin(), segs_angle_projC.end());
        par.insert(par.end(), vx_components.begin(), vx_components.end());

        auto lnl = [this, &par](double ke) {
            double KE = ke;
            return lnlikelihood_track(&KE, &par[0]);
        };

        double emin = 0, emax = 4e3;
        // ROOT ground truth
        TF1* f2 = new TF1("lnl", [this](double* KE, double* p) { return lnlikelihood_track(KE, p); },
                          emin, emax, (int)par.size());
        f2->SetParameters(&par[0]);
        double keguess = f2->GetMinimumX(emin + 1e-3, emax - 1e-3);
        double keguess_lower = f2->GetMinimumX(emin + 1e-3, keguess * 0.8);
        double keguess_higher = f2->GetMinimumX(std::min(keguess * 1.2, emax - 2e-3), emax - 1e-3);
        double l_keguess = std::exp(-lnl(keguess));
        double l_keguess_lower = std::exp(-lnl(keguess_lower));
        double l_keguess_higher = std::exp(-lnl(keguess_higher));
        double ambiguity = std::max(l_keguess_lower / l_keguess, l_keguess_higher / l_keguess);
        delete f2;

        // replica on the SAME ranges (validates the recipe the port re-implements)
        MinimizeResult m1 = replica_get_minimum_x(lnl, emin + 1e-3, emax - 1e-3);
        MinimizeResult m2 = replica_get_minimum_x(lnl, emin + 1e-3, keguess * 0.8);
        MinimizeResult m3 = replica_get_minimum_x(lnl, std::min(keguess * 1.2, emax - 2e-3), emax - 1e-3);

        fprintf(jf, "\"minimize\":{\"calls\":[");
        const MinimizeResult* mrs[3] = { &m1, &m2, &m3 };
        double root_x[3] = { keguess, keguess_lower, keguess_higher };
        for (int c = 0; c < 3; c++) {
            const StepResult& sr = mrs[c]->first_step;
            if (c) fprintf(jf, ",");
            fprintf(jf, "{\"xmin0\":"); jnum(jf, sr.xmin0);
            fprintf(jf, ",\"xmax0\":"); jnum(jf, sr.xmax0);
            fprintf(jf, ",\"npx\":100,\"argmin\":%d,\"bracket\":[", sr.argmin);
            jnum(jf, sr.xmin1); fprintf(jf, ","); jnum(jf, sr.xmax1); fprintf(jf, ","); jnum(jf, sr.xmiddle);
            fprintf(jf, "],\"grid_y\":"); jvec(jf, sr.grid_y);
            fprintf(jf, ",\"replica_x\":"); jnum(jf, mrs[c]->x);
            fprintf(jf, ",\"replica_nsearch\":%d,\"root_x\":", mrs[c]->nsearch); jnum(jf, root_x[c]);
            fprintf(jf, ",\"replica_minus_root\":"); jnum(jf, mrs[c]->x - root_x[c]);
            fprintf(jf, "}");
        }
        fprintf(jf, "],\n\"keguess\":"); jnum(jf, keguess);
        fprintf(jf, ",\"keguess_lower\":"); jnum(jf, keguess_lower);
        fprintf(jf, ",\"keguess_higher\":"); jnum(jf, keguess_higher);
        fprintf(jf, ",\"l_keguess\":"); jnum(jf, l_keguess);
        fprintf(jf, ",\"l_keguess_lower\":"); jnum(jf, l_keguess_lower);
        fprintf(jf, ",\"l_keguess_higher\":"); jnum(jf, l_keguess_higher);
        fprintf(jf, ",\"ambiguity\":"); jnum(jf, ambiguity);
        fprintf(jf, "},\n");

        // dense likelihood curve for plots / basin inspection
        fprintf(jf, "\"likelihood_curve\":[");
        for (int i = 0; i < 2000; i++) {
            double ke = 1.0 + i * 2.0;
            if (i) fprintf(jf, ",");
            fprintf(jf, "["); jnum(jf, ke); fprintf(jf, ","); jnum(jf, lnl(ke)); fprintf(jf, "]");
        }
        fprintf(jf, "],\n");

        // cross-check: the real estimate_energy end-to-end
        VD ee = estimate_energy(segs_distance, segs_angle_projB, segs_angle_projC, vx_components);
        fprintf(jf, "\"outputs\":{\"mu_tracklen\":"); jnum(jf, rr_path);
        fprintf(jf, ",\"emu_tracklen\":"); jnum(jf, (KE_rr_path + Mmu) / 1000.);
        fprintf(jf, ",\"emu_MCS\":"); jnum(jf, (ee[0] + Mmu) / 1000.);
        fprintf(jf, ",\"ambiguity_MCS\":"); jnum(jf, ee[1]);
        fprintf(jf, ",\"estimate_energy_ke\":"); jnum(jf, ee[0]);
        fprintf(jf, ",\"replica_consistent\":%s}\n}\n",
                (ee[0] == keguess && ee[1] == ambiguity) ? "true" : "false");
    }
};

int main(int argc, char* argv[])
{
    std::string mode, infile, outfile;
    long shuffle_seed = -1;
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "--root" || args[i] == "--txt") { mode = args[i]; infile = args[++i]; }
        else if (args[i] == "--shuffle") { shuffle_seed = atol(args[++i].c_str()); }
        else outfile = args[i];
    }
    if (mode.empty() || outfile.empty()) {
        fprintf(stderr, "usage: mcs_dump --root|--txt input [--shuffle SEED] out.json\n");
        return 2;
    }

    VD vstart, vend;
    VVD points;
    if (mode == "--root") {
        TFile* file = TFile::Open(infile.c_str());
        TMatrixD* mat = (TMatrixD*)file->Get("spacepoints_evt0");
        TVectorD* sv = (TVectorD*)file->Get("reco_start_evt0");
        TVectorD* ev = (TVectorD*)file->Get("reco_end_evt0");
        vstart = { (*sv)[0], (*sv)[1], (*sv)[2] };
        vend = { (*ev)[0], (*ev)[1], (*ev)[2] };
        for (int i = 0; i < mat->GetNrows(); i++) points.push_back({ (*mat)(i, 0), (*mat)(i, 1), (*mat)(i, 2) });
    }
    else {
        std::ifstream in(infile);
        if (!in) { fprintf(stderr, "cannot open %s\n", infile.c_str()); return 2; }
        double x, y, z;
        in >> x >> y >> z; vstart = { x, y, z };
        in >> x >> y >> z; vend = { x, y, z };
        while (in >> x >> y >> z) points.push_back({ x, y, z });
    }

    if (shuffle_seed >= 0) {
        std::mt19937 rng((unsigned)shuffle_seed);
        std::shuffle(points.begin(), points.end(), rng);
    }

    FILE* jf = fopen(outfile.c_str(), "w");
    if (!jf) { fprintf(stderr, "cannot write %s\n", outfile.c_str()); return 2; }
    InstrumentedMCS mcs;
    mcs.dump_run(vstart, vend, points, jf);
    fclose(jf);
    mcs.cleanUp();
    return 0;
}
