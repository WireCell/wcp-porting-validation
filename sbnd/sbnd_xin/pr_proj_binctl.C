// Negative control for the Magnify-tracking projection binning.
//
// Reproduces exactly the three lines of Magnify-tracking-SBND/event/Data.cc that
// decide where a charge cell and the fitted track land on the pad:
//   TH2F(n, lo, hi, nTime, 0, nTime)      <- old, bin centre = index + 0.5
//   hc->SetBinContent(x+1, y+1, z)        <- fill by index
//   g->SetPoint(i, us[i], t)              <- polyline at the raw index
// and draws the same pads with the new edges (lo-0.5 .. hi-0.5).  Left column is
// the OLD binning: it must reproduce the half-channel offset seen in the GUI.

void binctl(const char* fname, int cid = 5, int t0 = 564, int t1 = 604,
            int u0 = 836, int u1 = 876, int v0 = 4307, int v1 = 4347,
            int w0 = 7983, int w1 = 8023)
{
    const int nChannel_u = 3968, nChannel_v = 3968, nChannel_w = 3340, nTime = 857;
    const int lo[3] = {0, nChannel_u, nChannel_u + nChannel_v};
    const int nch[3] = {nChannel_u, nChannel_v, nChannel_w};
    const int x0[3] = {u0, v0, w0}, x1[3] = {u1, v1, w1};
    const char* pname[3] = {"U", "V", "W"};

    TFile* f = TFile::Open(fname);
    TTree* T_rec = (TTree*)f->Get("T_rec");
    TTree* T_proj = (TTree*)f->Get("T_proj_data");

    std::vector<int>* rec_cluster_id = 0;
    std::vector<std::vector<double> >*ru = 0, *rv = 0, *rw = 0, *rt = 0;
    T_rec->SetBranchAddress("rec_cluster_id", &rec_cluster_id);
    T_rec->SetBranchAddress("rec_u", &ru);
    T_rec->SetBranchAddress("rec_v", &rv);
    T_rec->SetBranchAddress("rec_w", &rw);
    T_rec->SetBranchAddress("rec_t", &rt);
    T_rec->GetEntry(0);

    std::vector<int>* pcid = 0;
    std::vector<std::vector<int> >*pch = 0, *pts = 0, *pq = 0;
    T_proj->SetBranchAddress("cluster_id", &pcid);
    T_proj->SetBranchAddress("channel", &pch);
    T_proj->SetBranchAddress("time_slice", &pts);
    T_proj->SetBranchAddress("charge", &pq);
    T_proj->GetEntry(0);

    int ib = -1, ip = -1;
    for (size_t i = 0; i < rec_cluster_id->size(); ++i) if (rec_cluster_id->at(i) == cid) ib = i;
    for (size_t i = 0; i < pcid->size(); ++i) if (pcid->at(i) == cid) ip = i;
    if (ib < 0 || ip < 0) { printf("cluster %d not found\n", cid); return; }
    printf("block %d (%zu points), proj entry %d (%zu cells)\n",
           ib, ru->at(ib).size(), ip, pch->at(ip).size());

    gStyle->SetOptStat(0);
    TCanvas* c = new TCanvas("c", "", 1400, 1800);
    c->Divide(2, 3);

    std::vector<std::vector<double> >* rec[3] = {ru, rv, rw};

    for (int p = 0; p < 3; ++p) {
        for (int variant = 0; variant < 2; ++variant) {   // 0 = old edges, 1 = new
            const double shift = variant ? 0.5 : 0.0;
            TH2F* h = new TH2F(Form("h%d_%d", p, variant), "",
                               nch[p], lo[p] - shift, lo[p] + nch[p] - shift,
                               nTime, -shift, nTime - shift);
            for (size_t i = 0; i < pch->at(ip).size(); ++i) {
                int x = pch->at(ip).at(i);
                if (x < lo[p] || x >= lo[p] + nch[p]) continue;
                h->SetBinContent(x - lo[p] + 1, pts->at(ip).at(i) + 1, pq->at(ip).at(i));
            }
            h->GetZaxis()->SetRangeUser(500, 20000);
            h->GetXaxis()->SetRangeUser(x0[p], x1[p]);
            h->GetYaxis()->SetRangeUser(t0, t1);
            h->GetXaxis()->SetTitle("Channel");
            h->GetYaxis()->SetTitle("Time Slice");
            h->SetTitle(Form("%s plane -- %s binning (%s)", pname[p],
                             variant ? "NEW" : "OLD",
                             variant ? "edges lo-0.5 .. hi-0.5" : "edges lo .. hi"));

            const std::vector<double>& xs = rec[p]->at(ib);
            const std::vector<double>& ts = rt->at(ib);
            TGraph* g = new TGraph();
            int n = 0;
            for (size_t i = 0; i < xs.size(); ++i) {
                if (xs[i] < x0[p] - 5 || xs[i] > x1[p] + 5) continue;
                if (ts[i] < t0 - 5 || ts[i] > t1 + 5) continue;
                g->SetPoint(n++, xs[i], ts[i]);
            }
            g->SetMarkerStyle(20);
            g->SetMarkerSize(0.6);
            g->SetMarkerColor(6);
            g->SetLineColor(6);
            g->SetLineWidth(2);

            c->cd(2 * p + variant + 1);
            gPad->SetRightMargin(0.13);
            h->Draw("colz");
            if (n) g->Draw("Psame");
        }
    }
    c->SaveAs("binctl.png");
    printf("wrote binctl.png\n");
}
