// SBNDOpDetDumper
//
// Small one-off analyzer module that prints SBND optical-detector geometry to
// stdout in CSV format, prefixed with "OPDET:" so it can be grep'd out of the
// larsoft log. Used to seed the WCT-side semi-analytical-sbnd.json data file
// for the wire-cell-toolkit/match port.
//
// Build alongside an existing larwirecell module (e.g. by dropping the .cc
// in larwirecell/qlmatch/) and add the module name to that subdir's
// CMakeLists.txt. Or build with cetbuildtools as an art module in
// sbndcode/Utilities/.
//
// One-time use. Output the CSV, hand it to build_semi_analytical_sbnd_json.py
// together with semimodel_sbnd-dump.fcl, and check the resulting JSON in.

#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art_root_io/TFileService.h"

#include "larcore/Geometry/Geometry.h"
#include "larcore/CoreUtils/ServiceUtil.h"
#include "larcorealg/Geometry/OpDetGeo.h"

#include <iostream>

namespace sbnd {

  class SBNDOpDetDumper : public art::EDAnalyzer {
  public:
    explicit SBNDOpDetDumper(fhicl::ParameterSet const& pset) : EDAnalyzer(pset) {}
    void analyze(art::Event const&) override
    {
      auto const& geom = *(lar::providerFrom<geo::Geometry>());
      const std::size_t n = geom.NOpDets();
      std::cout << "# OPDET CSV: idx,x_cm,y_cm,z_cm,h_cm,w_cm,type,orientation\n";
      for (std::size_t i = 0; i < n; ++i) {
        geo::OpDetGeo const& opDet = geom.OpDetGeoFromOpDet(i);
        auto const c = opDet.GetCenter();
        double h = -1, w = -1;
        int type = 1, orient = 0;
        if (opDet.isSphere()) {
          type = 1; orient = 0; h = -1; w = -1;
        }
        else if (opDet.isBar()) {
          type = 0;
          h = opDet.Height();
          w = opDet.Length();
          if (opDet.Width() > opDet.Length()) { orient = 2; w = opDet.Width(); }
          else if (opDet.Width() > opDet.Height()) { orient = 1; h = opDet.Width(); }
          else { orient = 0; }
        }
        else {
          type = 2; orient = 0; h = -1; w = -1;
        }
        std::cout << "OPDET:" << i << "," << c.X() << "," << c.Y() << "," << c.Z()
                  << "," << h << "," << w << "," << type << "," << orient << "\n";
      }
    }
  };

} // namespace sbnd

DEFINE_ART_MODULE(sbnd::SBNDOpDetDumper)
