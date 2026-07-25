// Dump the prototype's stopping-particle dQ/dx reference TGraphs.
//
// These five graphs are the origin of the dE/dx LinterpFunction tables in
// wcp-porting-img/sbnd/particle_dataset.jsonnet, which the STM tagger's
// Bragg-peak test compares against (clus/src/TaggerCheckSTM.cxx eval_stm_core).
// The prototype loads the same file at pid/src/ToyFiducial.cxx:53-58.
//
// Usage (from prototype_base/input_data_files):
//   root -l -b -q '/path/to/toolkit/sbnd_xin/dump_stopping_dqdx.C("stopping_ave_dQ_dx.root")'
//   root -l -b -q '/path/to/toolkit/sbnd_xin/dump_stopping_dqdx.C("stopping_ave_dQ_dx_v2.root")'
//
// See sbnd_xin/docs/47_stm-bragg-reference-sbnd-retune.md section 2.

void dump_stopping_dqdx(const char* fn = "stopping_ave_dQ_dx.root")
{
    TFile* f = new TFile(fn);
    if (f->IsZombie()) { printf("cannot open %s\n", fn); return; }
    printf("== %s ==\n", fn);
    f->ls();
    const char* names[5] = {"muon", "pion", "kaon", "proton", "electron"};
    for (int k = 0; k < 5; k++) {
        TGraph* g = (TGraph*) f->Get(names[k]);
        if (!g) { printf("%-9s MISSING\n", names[k]); continue; }
        int n = g->GetN();
        double x0, y0, x1, y1, xl, yl;
        g->GetPoint(0, x0, y0);
        g->GetPoint(1, x1, y1);
        g->GetPoint(n - 1, xl, yl);
        printf("%-9s N=%4d  first=(%.4g,%.6g) second=(%.4g,%.6g) last=(%.4g,%.6g)  title='%s'\n",
               names[k], n, x0, y0, x1, y1, xl, yl, g->GetTitle());
    }
}
