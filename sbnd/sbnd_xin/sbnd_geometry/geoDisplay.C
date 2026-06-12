// geoDisplay.C -- view a GDML geometry in ROOT's OpenGL viewer.
//
// Usage (interactive; the OGL window needs a display):
//
//   // whole detector
//   root -l 'geoDisplay.C("sbnd_v02_06_nowires.gdml")'
//
//   // cathode plane (CPA): the 4x4 grid of mesh + foil panels and the
//   // steel frame, with the enclosing TPC-active box hidden so the grid
//   // is not swallowed by the 2x4x5 m LAr box it lives in.
//   root -l 'geoDisplay.C("sbnd_v02_06_nowires.gdml","volTPCActive")'
//
//   // a single CPA panel assembly (LAr box + VM2000 foil + TPB)
//   root -l 'geoDisplay.C("sbnd_v02_06_nowires.gdml","volCPA_FoilTPB")'
//
// List the volume names matching a substring (e.g. to find CPA parts):
//   root -l 'geoDisplay.C+("sbnd_v02_06_nowires.gdml")'  // then geoFind("CPA")
//   or:  root -l -q -e 'gROOT->ProcessLine(".x geoDisplay.C(\"sbnd_v02_06_nowires.gdml\",\"?CPA\")")'
//
// Notes on the SBND cathode (CPA):
//   * The CPA panels are placed directly inside volTPCActive -- there is no
//     wrapper volume that bounds only the cathode, hence hideMother below.
//   * volCPAMesh is a solid box of matSteelMesh, NOT modeled as wires, so it
//     renders as a flat panel.
//   * The VM2000 foil (volCPAFoil, ~0.05 mm) and TPB coating (volCPATPB,
//     ~2e-5 mm) are far too thin to be visually distinguishable in 3D.
//   The structure you CAN see is the 4x4 array of mesh/foil panels + the
//   steel frame (volCPA_East / volCPA_West).

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TList.h"
#include "TGLViewer.h"
#include "TVirtualPad.h"
#include "TCanvas.h"
#include "TView.h"
#include "TString.h"
#include "TPRegexp.h"
#include <set>

// List logical-volume names containing `pat` (case-sensitive substring).
void geoFind(const char *pat = "CPA")
{
   if (!gGeoManager) { printf("No geometry loaded.\n"); return; }
   TIter next(gGeoManager->GetListOfVolumes());
   TGeoVolume *v = nullptr;
   int n = 0;
   while ((v = (TGeoVolume *)next())) {
      TString name = v->GetName();
      if (name.Contains(pat)) {
         printf("  %-28s  ndaughters=%d\n", v->GetName(), v->GetNdaughters());
         ++n;
      }
   }
   printf("[geoFind] %d volume(s) matching \"%s\"\n", n, pat);
}

void geoDisplay(TString filename,
                TString volname = "",
                Int_t   VisLevel = 6,
                Bool_t  hideMother = kTRUE,
                Bool_t  checkOverlaps = kFALSE)
{
   TGeoManager *geo = TGeoManager::Import(filename);
   if (!geo) { printf("[geoDisplay] failed to import %s\n", filename.Data()); return; }
   geo->DefaultColors();

   // A leading '?' means: just list matching volume names and return.
   if (volname.BeginsWith("?")) {
      geoFind(volname(1, volname.Length() - 1).Data());
      return;
   }

   if (checkOverlaps) {
      geo->CheckOverlaps(1e-5, "d");
      geo->PrintOverlaps();
   }

   // Pick the volume to draw: named sub-volume, or the top volume.
   TGeoVolume *vol = nullptr;
   if (volname.Length()) {
      vol = geo->GetVolume(volname);
      if (!vol) {
         printf("[geoDisplay] volume \"%s\" not found. Candidates:\n", volname.Data());
         geoFind(volname.Data());
         return;
      }
   } else {
      vol = geo->GetTopVolume();
   }

   geo->SetVisOption(1);     // draw all volumes, not only leaves
   geo->SetVisLevel(VisLevel);

   // Hide the drawn volume's own shape so its daughters (e.g. the CPA grid)
   // are not enclosed/swallowed by the mother box. Only matters when a
   // specific sub-volume is requested.
   if (hideMother && volname.Length())
      vol->SetVisibility(kFALSE);

   vol->Print();
   printf("[geoDisplay] drawing \"%s\" (%d daughters, VisLevel=%d, hideMother=%d)\n",
          vol->GetName(), vol->GetNdaughters(), VisLevel, hideMother);
   vol->Draw("ogl");

   if (gPad) {
      TGLViewer *v = (TGLViewer *)gPad->GetViewer3D();
      if (v) {
         v->SetStyle(TGLRnrCtx::kOutline);
         v->SetSmoothPoints(kTRUE);
         v->SetLineScale(0.5);
         // v->UseDarkColorSet();
         v->UpdateScene();
      }
   }
}

// ---------------------------------------------------------------------------
// geoAnode -- batch (headless) drawing of the SBND anode region.
//
// Renders the two TPCs side by side so the drift "gap" between them is
// visible: West anode | drift | shared cathode (x=0) | drift | East anode.
// The wire planes are drawn as their bare LAr boxes (NO wires); the dense
// cathode mesh and all non-TPC clutter (PDS, field cage, APA frames, ullage)
// are hidden. Works with `root -l -b -q`.
//
//   root -l -b -q 'geoDisplay.C("sbnd_v02_06_nowires.gdml","","top")'  // ignored args
//   root -l -b -q geoAnode.C   // see wrapper, or call directly:
//   root -l -b -q -e 'gROOT->ProcessLine(".x geoDisplay.C"); geoAnode();'
//
// view: "top"  -> look down +y, project x-z (shows the drift gap)   [default]
//       "front"-> look down the drift x, project z-y (anode face-on)
//       "side" -> look down z, project x-y
//       "3d"   -> oblique perspective
// ---------------------------------------------------------------------------
void geoAnode(TString filename = "sbnd_v02_06_nowires.gdml",
              TString out      = "anode_topview.png",
              TString view     = "top",
              Int_t   VisLevel = 4)
{
   TGeoManager *geo = TGeoManager::Import(filename);
   if (!geo) { printf("[geoAnode] failed to import %s\n", filename.Data()); return; }
   geo->DefaultColors();

   // Whitelist: making a mother volume invisible does NOT hide its daughters
   // (the traversal still descends), so to strip the PDS/field-cage/cathode-mesh
   // clutter we instead hide EVERY volume's own box, then re-enable only the
   // boxes we want: the two TPCs, their active LAr volumes, and the three bare
   // wire-plane boxes per TPC (no wires). The cathode is shown implicitly where
   // the two active boxes meet at x=0.
   // volOneAPAFrame is the steel support frame that HOLDS the wire planes: a
   // picture-frame of hollow tubes (two 4150 mm vertical rails + horizontal
   // cross-bars). Each anode wall is built from TWO such frames stacked along
   // z, so their rails form a seam near the detector centre (z_J ~ 2505 mm) --
   // this is the real geometric "structure along the W plane".
   const std::set<TString> keep = {
      "volTPC_East", "volTPC_West", "volTPCActive",
      "volTPCPlaneVert", "volTPCPlane_U", "volTPCPlane_V",
      "volOneAPAFrame"};
   TIter nextv(geo->GetListOfVolumes());
   TGeoVolume *vv = nullptr;
   while ((vv = (TGeoVolume *)nextv()))
      vv->SetVisibility(keep.count(vv->GetName()) ? kTRUE : kFALSE);

   TGeoVolume *cryo = geo->GetVolume("volCryostat");
   if (!cryo) { printf("[geoAnode] volCryostat not found\n"); return; }

   geo->SetVisOption(1);
   geo->SetVisLevel(VisLevel);

   TCanvas *c = new TCanvas("cAnode", "SBND anode region", 1200, 900);
   cryo->Draw();

   if (gPad) {
      TView *v = gPad->GetView();
      if (v) {
         if      (view == "top")   v->Top();    // down +y -> x-z plane
         else if (view == "front") v->Front();  // down drift x -> z-y plane
         else if (view == "side")  v->Side();   // down z -> x-y plane
         else                      v->SetView(30, 60, 0, *(new Int_t));  // 3d
         gPad->Modified();
         gPad->Update();
      }
   }
   c->SaveAs(out);
   printf("[geoAnode] wrote %s (view=%s, VisLevel=%d)\n",
          out.Data(), view.Data(), VisLevel);
}
