// Headless exercise of the REAL Magnify-tracking-SBND viewer code.
//
// Mirrors GuiController's init sequence (GuiController.cc:36-52) -- 3x3 canvas,
// Data(filename, sign), data->c1 = canvas, SetCurrentCluster, DrawNewCluster --
// but without a TGWindow, then drives the same public entry points the GUI calls
// and writes a PNG.  This runs Data.cc itself (DisplayL, the per-sub-cluster
// dQ/dx graphs, the projection binning, DrawSubclusters, DrawBadCh, ZoomProj),
// which the pr_proj_binctl.C control deliberately does not.
//
// Usage (from Magnify-tracking-SBND/scripts, after loadClasses.C):
//   root -l -b -q loadClasses.C '../../toolkit/sbnd_xin/pr_proj_guishot.C("f.root", 0, 79, 20, "out.png")'
//
// See docs/pr/7_magnify-tracking-projection-alignment.md.

void pr_proj_guishot(const char* filename, int cluster = 0, int pointIndex = 79,
                     int zoomBin = 20, const char* png = "guishot.png", int sign = 0)
{
    gStyle->SetOptStat(0);
    TCanvas* can = new TCanvas("can", "", 1600, 900);
    can->Divide(3, 3, 0.005, 0.005);

    Data* data = new Data(filename, sign);
    data->c1 = can;
    data->currentCluster = cluster;
    data->DrawNewCluster();

    const int n = data->rec_u->at(cluster).size();
    printf("cluster index %d -> id %d, %d points\n",
           cluster, data->rec_cluster_id->at(cluster), n);
    if (pointIndex >= 0 && pointIndex < n) {
        data->DrawPoint(pointIndex);
        data->ZoomProj(pointIndex, zoomBin);
    }

    can->SaveAs(png);
    printf("wrote %s\n", png);
}
