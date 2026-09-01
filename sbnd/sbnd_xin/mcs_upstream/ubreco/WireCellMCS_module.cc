#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "art/Persistency/Common/PtrMaker.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include "nusimdata/SimulationBase/MCParticle.h"
#include "lardata/Utilities/AssociationUtil.h"
#include "lardataobj/RecoBase/SpacePoint.h"
#include "ubobj/WcpPort/SpacePointStructs.h"
#include "ubreco/WcpPortedReco/ProducePort/WCMCSTrajectory.h"

#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include <memory>
#include <string>
#include <dirent.h>
//#include <iostream>
#include <numeric>
#include "TGraph2D.h"
#include "TGraph.h"
#include "TF1.h"

class WireCellMCS;

class WireCellMCS : public art::EDProducer {
public:
  explicit WireCellMCS(fhicl::ParameterSet const & p);
  WireCellMCS(WireCellMCS const &) = delete;
  WireCellMCS(WireCellMCS &&) = delete;
  WireCellMCS & operator = (WireCellMCS const &) = delete;
  WireCellMCS & operator = (WireCellMCS &&) = delete;

  void reconfigure(fhicl::ParameterSet const& pset);
  void writeOutput(art::Event &e);

private:
  void produce(art::Event &e) override;
  void cleanUp();

  //helper functions
  double beta(double gamma);
  double gamma(double KE, double mass);
  double sigmoid(double x);
  void increment_energy (double &e);
  void decrement_dist (double &x, double xmax);
  void setUKEfromRR();
  void setUKEfromEX (TGraph *uKEfromRR, TGraph *uRRfromKE);
  double sigmaH (double T);
  double sigmoid(double x, std::vector<double> par);
  double quartic_decay(double x, std::vector<double> par);
  std::vector<double> pred_theta_xz_pars(double T);
  std::vector<double> pred_theta_yz_pars(double T, int vx_index);
  double double_gaussian(double angle, std::vector<double> pars);
  double lnlikelihood_theta_xz(double angle, double T);
  double lnlikelihood_theta_yz(double angle, double T, double vx);
  double lnlikelihood_track(double* KE, double* par);
  std::vector<double> estimate_energy(std::vector<double> segs_distance, std::vector<double> segs_angle_x, std::vector<double> segs_angle_y,  std::vector<double> vx_comps);

  //constants
  //const int MAX_TRACKS = 1e4;
  const double seg_length = 14; //cm
  //const int Z = 18;					//Atomic number of argon
  //const double A = 39.948;				//Atomic weight of Argon in amu
  const double I = 188.0*pow(10,-6); 			//eV
  //const double K = 0.307;				// MeV * cm^2 / mol
  //const double Mp = 938.28; 				// MeV for proton
  const double Mmu = 105.658;				// MeV for muon
  //const double Me  = 0.51;				// MeV for electron
  const double rho = 1.396;				// LAr density [g/cm3]
  //const double x = 0.5;					//setp size in cm

  //variables set in fhicl file
  bool f_wirecellPF;
  std::string fPFInputTag;
  std::string fportedWCSpacePointsTrecchargeblobLabel;
  double res_sigma1_xz, res_sigma2_xz;
  std::vector<double> res_sigma1_yz, res_sigma2_yz, par_sigma1_xz, par_sigma2_xz, par_ratio_xz;
  std::vector<std::vector<double>> par_sigma1_yz, par_sigma2_yz, par_ratio_yz;

  //internal variables
  TGraph* uKEfromRR;
  TGraph* uRRfromKE;
  //TGraph2D* uKEfromEX;
  double mu_tracklen;
  double emu_tracklen;
  double emu_MCS;
  double ambiguity_MCS;
};

//Constructor
WireCellMCS::WireCellMCS(fhicl::ParameterSet const & p) : EDProducer{p}
{
  std::cout << "Begin WireCellMCS constructor" << std::endl;
  produces<std::vector<double>>();

  reconfigure(p);
  std::cout << "Done WireCellMCS constructor" << std::endl;
}

//initialization using fhicl parameters
void WireCellMCS::reconfigure(fhicl::ParameterSet const& pset)
{
  f_wirecellPF = pset.get<bool>("wirecellPF", false);
  fPFInputTag  = pset.get<std::string>("PF_inputtag");
  //TODO: include this info in a fhicl file
  //fportedWCSpacePointsTrecchargeblobLabel = pset.get<std::string>("portedWCSpacePointsTrecchargeblob");
  fportedWCSpacePointsTrecchargeblobLabel = "portedWCSpacePointsTrecchargeblob";

  //Double Gaussian tune parameters
  res_sigma1_xz = pset.get<double>("MCS_res_sigma1_xz");
  res_sigma2_xz = pset.get<double>("MCS_res_sigma2_xz");
  par_sigma1_xz = pset.get<std::vector<double>>("MCS_par_sigma1_xz");
  par_sigma2_xz = pset.get<std::vector<double>>("MCS_par_sigma2_xz");
  par_ratio_xz  = pset.get<std::vector<double>>("MCS_par_ratio_xz");
  res_sigma1_yz = pset.get<std::vector<double>>("MCS_res_sigma1_yz");
  res_sigma2_yz = pset.get<std::vector<double>>("MCS_res_sigma2_yz");
  par_sigma1_yz = pset.get<std::vector<std::vector<double>>>("MCS_par_sigma1_yz");
  par_sigma2_yz = pset.get<std::vector<std::vector<double>>>("MCS_par_sigma2_yz");
  par_ratio_yz  = pset.get<std::vector<std::vector<double>>>("MCS_par_ratio_yz");
  /*
  res_sigma1_xz = 0.005776;
  res_sigma2_xz = 0.01821;
  par_sigma1_xz = { -0.449144931, 0.793132642, -1.291292240, 0.536765147, -0.084910516, 0.146304242 };
  par_sigma2_xz = {  0.562850793, 0.118940108,  0.000625509, 0,           -0.000000100, 1.217639251 };
  par_ratio_xz  = {  0.839684805, 0.839684805,  0, 0 };
  res_sigma1_yz = { 0.0449, 0.0206,  0.01403, 0.0131,  0.0114  };
  res_sigma2_yz = { 0.1506, 0.06154, 0.03965, 0.04179, 0.07347 };
  par_sigma1_yz = {{ -1.0,         -0.084325217,  0.487240052,  0.395496655, -0.187184874, 0.166128734 },
                   {  0.0,         -0.575280374,  0.070151974,  0.187260875, -0.099717108, 0.160128002 },
                   { -0.153367057, -0.583042532,  0.983374136, -0.712652874,  0.134743902, 0.465439107 },
                   { -0.268993212,  0.103899779, -0.588953942,  0.282356661, -0.067930741, 0.176282668 },
                   { -0.2,         -0.724028910,  0.660065851, -0.327141529,  0.038426745, 0.094357571 }};
  par_sigma2_yz = {{  0.193250363,  3.046073125,  15.0,        -7.519451962, -1.0,         0.5         },
                   {  2.226214634, -5.407245785,  7.227026554, -2.411882646, -0.000001517, 0.224728907 },
                   {  0.25,         1.670060693, -2.086639043,  1.122640876,  0.000000440, 0.268340031 },
                   {  0.430965414, -0.079204912,  0.220949488, -0.000010000, -0.000010000, 0.3         },
                   {  1.635001641, -2.414781711,  1.669221724, -0.320484259,  0.0,         0.157018001 }};
  par_ratio_yz  = {{  0.519230936,  1.0,          1.663982350,  0.084218766 },
                   {  0.7,          0.788705752,  5.0,          2.045055158 },
                   {  0.822433461,  0.701381849,  5.0,          0.0         },
                   {  0.897408009,  0.7,          2.182533745,  0.297834161 },
                   {  0.9,          0.9,          0.0,          0.0         }};
  */
}

//save output variables
void WireCellMCS::writeOutput(art::Event &e) {
  //std::cout << "E_RR, E_MCS = " << emu_tracklen << ",  " << emu_MCS << std::endl;
  auto outputs = std::make_unique<std::vector<double>>();
  outputs->push_back( mu_tracklen);
  outputs->push_back(emu_tracklen);
  outputs->push_back(emu_MCS);
  outputs->push_back(ambiguity_MCS);
  e.put(std::move(outputs));
}

//clean up allocated memory
void WireCellMCS::cleanUp()
{
  uKEfromRR = nullptr;
  uRRfromKE = nullptr;
  //uKEfromEX = nullptr;
}

void WireCellMCS::produce(art::Event &e){

  std::cout << "Begin WireCellMCS::produce" << std::endl;

  //Create graphs mapping between residual range and energy
  setUKEfromRR();
  setUKEfromEX(uKEfromRR, uRRfromKE);

  //prepare outputs
  mu_tracklen   = -1;
  emu_tracklen  = -1;
  emu_MCS       = -1;
  ambiguity_MCS = -1;

  //get muon start and end position
  double reco_emu = -1;
  std::vector<double> vtx_muon_start_reco, vtx_muon_end_reco;
  if( f_wirecellPF ){
    auto particleHandle = e.getHandle<std::vector<simb::MCParticle>>(fPFInputTag);
    if (!particleHandle) {
      std::cout << "Couldn't find PF particles" << std::endl;
      writeOutput(e);
      cleanUp();
      return;
    }
    for (auto const& particle: *particleHandle){
      auto start_mom = particle.Momentum();
      if (particle.PdgCode()==13 && start_mom.E()>reco_emu) {
        reco_emu = start_mom.E();
        auto start_pos = particle.Position();
        auto end_pos   = particle.EndPosition();
	vtx_muon_start_reco = { start_pos.X(), start_pos.Y(), start_pos.Z() };
	vtx_muon_end_reco   = {   end_pos.X(),   end_pos.Y(),   end_pos.Z() };
      }
    }
  }

  //skip events without a reco muon
  if (reco_emu==-1) {
    std::cout << "No reco muon" << std::endl;
    writeOutput(e);
    cleanUp();
    return;
  }

  //Read in trajectory points and trim to only include muon path
  auto v_TrecchargeblobSpacePoint = e.getProduct<std::vector<TrecchargeblobSpacePoint>>(fportedWCSpacePointsTrecchargeblobLabel);
  int npoints_trajectory = v_TrecchargeblobSpacePoint.size();
  std::vector<std::vector<double>> trajectory_points_initial;
  for (int i=0;i<npoints_trajectory;i++) { trajectory_points_initial.push_back( {v_TrecchargeblobSpacePoint[i].x, v_TrecchargeblobSpacePoint[i].y, v_TrecchargeblobSpacePoint[i].z} ); }
  std::tuple<bool,std::vector<std::vector<double>>> trajectory_tuple = trim_trajectory(npoints_trajectory, trajectory_points_initial, vtx_muon_start_reco, vtx_muon_end_reco);
  bool bad_path                                            = std::get<0>(trajectory_tuple);
  std::vector<std::vector<double>> trajectory_points_final = std::get<1>(trajectory_tuple);
  int npoints_trajectory_final = trajectory_points_final.size();

  //compute residual range and emu_rr
  double rr_path = 0;
  for (int i=1;i<npoints_trajectory_final;i++) { rr_path += norm(diff(trajectory_points_final[i],trajectory_points_final[i-1])); }
  double KE_rr_path = uKEfromRR->Eval(rr_path);
  mu_tracklen = rr_path;
  emu_tracklen = (KE_rr_path+Mmu)/1000.;   //convert from KE to E and from MeV to GeV

  //skip events where a path cannot be traversed from muon start to end
  //skip events with very short tracks (remember the track is in reverse order)
  if (bad_path || npoints_trajectory_final<20 || norm(diff(trajectory_points_final.back(),vtx_muon_end_reco)) < 2*seg_length) {
    std::cout << "Bad/short path" << std::endl;
    writeOutput(e);
    return;
  }

  //segment path
  std::cout << "Segmenting muon path" << std::endl;
  std::tuple< std::vector<Track>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double> > segs_tuple = form_segs(trajectory_points_final, vtx_muon_start_reco, vtx_muon_end_reco, seg_length);             
  std::vector<Track> segs                     = std::get<0>(segs_tuple);
  std::vector<std::vector<double>> axes       = std::get<1>(segs_tuple);
  std::vector<std::vector<double>> segs_aAxes = std::get<2>(segs_tuple);
  std::vector<std::vector<double>> segs_COM   = std::get<3>(segs_tuple);
  std::vector<double> segs_distance           = std::get<4>(segs_tuple);  //distance from muon start to seg midpoint
  std::vector<double> segs_angle              = std::get<5>(segs_tuple);
  std::vector<double> segs_angle_projB        = std::get<6>(segs_tuple);
  std::vector<double> segs_angle_projC        = std::get<7>(segs_tuple);
  //std::vector<double> segs_displacement       = std::get<8>(segs_tuple);

  //skip events without any angle measurements
  if (segs.size()<2) {
    std::cout << "Short path" << std::endl;
    writeOutput(e);
    cleanUp();
    return;
  }

  //Get the x-component of each segment fit vector
  std::vector<double> vx_components;
  for (const auto& seg_dir : segs_aAxes) {
    if (!seg_dir.empty()) { vx_components.push_back(seg_dir[0]); }
  }

  //estimate energy
  std::cout << "Estimating energy" << std::endl;
  std::vector<double> kemu_MCS_tuple = estimate_energy(segs_distance, segs_angle_projB, segs_angle_projC, vx_components);
  emu_MCS       = (kemu_MCS_tuple[0]+Mmu)/1000.;   //convert from KE to E and from MeV to GeV
  ambiguity_MCS =  kemu_MCS_tuple[1];

  //write outputs
  writeOutput(e);
  cleanUp();
  std::cout << "Done WireCellMCS" << std::endl;
}

double WireCellMCS::beta(double gamma) {return sqrt(1.-(1./pow(gamma,2)));}

double WireCellMCS::gamma(double KE, double mass) {return KE/mass+1;}	//KE, mass in MeV

double WireCellMCS::sigmoid(double x) { return 1. / (1. + std::exp(-x)); }

//This graph gives a function from muon residual range values to total kinetic energy values.
//The data is taken from Atomic Data and Nuclear Data Tables 78: MUON STOPPING POWER AND RANGE TABLES 10 MeV–100 TeV
// http://pdg.lbl.gov/2018/AtomicNuclearProperties/adndt.pdf
// http://pdg.lbl.gov/2017/AtomicNuclearProperties/HTML/liquid_argon.html
// http://pdg.lbl.gov/2017/AtomicNuclearProperties/MUE/muE_liquid_argon.pdf
void WireCellMCS::setUKEfromRR() {
  const int un = 20;																	//number of entries in muon KE vs RR table
  //double uCSDA[un]   = { .9937, 1.795, 3.329, 6.605, 10.58, 30.84, 42.50, 67.32, 106.3, 172.5, 238.4, 493.4, 616.3, 855.2, 1202., 1758., 2297., 4359.,  5354.,  7298. };	
  double uCSDA[un]     = { .9833, 1.786, 3.321, 6.598, 10.58, 30.84, 42.50, 67.32, 106.3, 172.5, 238.5, 493.4, 616.3, 855.2, 1202., 1758., 2297., 4359.,  5354.,  7298. };	
  double uEnergy[un]   = {  10.0,  14.0,  20.0,  30.0,  40.0,  80.0, 100.0, 140.0, 200.0, 300.0, 400.0, 800.0, 1000., 1400., 2000., 3000., 4000., 8000., 10000., 14000. };							//Total particle energy in MeV
  double uResRange[un];
  for (int i=0;i<un;i++) { uResRange[i] = uCSDA[i]/rho; }	//cm
  uKEfromRR = new TGraph(un,uResRange,uEnergy);
  uRRfromKE = new TGraph(un,uEnergy,uResRange);
}

//helper function that increases the energy in steps from 10 MeV to 14 GeV 
void WireCellMCS::increment_energy (double &e) {
  e *= 1.2;
  e += 2;
}

//helper function that decreases the remaining distance of a muon track
void WireCellMCS::decrement_dist (double &x, double xmax) {
  double x_offset = xmax-x;
  x_offset *= 1.1;
  x_offset += 0.5;
  x = xmax - x_offset;
}

//Helper function that computes a 2D graph mapping {starting energy, distance} -> remaining energy
void WireCellMCS::setUKEfromEX (TGraph *uKEfromRR, TGraph *uRRfromKE) {

  /*
  double emin = 10.;
  double emax = 14000;
  double min_dist = uRRfromKE->Eval(emin);
  
  uKEfromEX = new TGraph2D();  
  int i=0;
  for (double e=emin;e<emax;increment_energy(e)) {
    double rr_start = uRRfromKE->Eval(e);
    for (double x=rr_start;x>min_dist;decrement_dist(x,rr_start)) {
      double e_final = uKEfromRR->Eval(rr_start-x);
      i++;
      uKEfromEX->SetPoint(i,e,x,e_final);
    }
    i++;
    uKEfromEX->SetPoint(i,e,0,e);
  }
  */
}


//highland formula
double WireCellMCS::sigmaH (double T){ return 13.6*(T+Mmu)/T/(T+2*Mmu); }

//helper function to weight the area of each Gaussian using a sigmoid distribution
double WireCellMCS::sigmoid(double x, std::vector<double> par) { return par[0] + (par[1]-par[0])*(1 - 1./(1+std::exp(-par[2]*(x/1000.-par[3])))); }

//helper function to modify highland formula based on parameter values
//first 5 parameters define a quartic with the last acting as a scale parameter
double WireCellMCS::quartic_decay(double x, std::vector<double> par) {
  double u   = x/par.back()/1000.;
  double val = 0;
  for (int i=0,n=par.size();i<n-1;i++) { val += par[i]*std::pow(u,i); }
  return 1 + val*std::exp(-u);
}

//function to calculate theta_xz PDF parameters
std::vector<double> WireCellMCS::pred_theta_xz_pars(double T) {
  double sigma1     = std::sqrt(std::pow(sigmaH(T) * quartic_decay(T,par_sigma1_xz), 2) + std::pow(res_sigma1_xz, 2));
  double sigma2     = std::sqrt(std::pow(sigmaH(T) * quartic_decay(T,par_sigma2_xz), 2) + std::pow(res_sigma2_xz, 2));
  double area_ratio = sigmoid(T, par_ratio_xz);
  return {sigma1, sigma2, area_ratio};
}

//function to calculate theta_yz PDF parameters
std::vector<double> WireCellMCS::pred_theta_yz_pars(double T, int vx_index) {
  double sigma1 = std::sqrt(std::pow(sigmaH(T) * quartic_decay(T,par_sigma1_yz[vx_index]), 2) + std::pow(res_sigma1_yz[vx_index], 2));
  double sigma2 = std::sqrt(std::pow(sigmaH(T) * quartic_decay(T,par_sigma2_yz[vx_index]), 2) + std::pow(res_sigma2_yz[vx_index], 2));
  double area_ratio = sigmoid(T, par_ratio_yz[vx_index]);
  return {sigma1, sigma2, area_ratio};
}

//Two-Gaussian PDF that computes the likelihood of an angle given input parameter values
double WireCellMCS::double_gaussian(double angle, std::vector<double> pars) {
  double sigma1 = pars[0];
  double sigma2 = pars[1];
  double ratio  = pars[2];
  double gaussian1 = (1./(std::sqrt(2*M_PI)*sigma1)) * std::exp(-0.5*std::pow(angle/sigma1,2));
  double gaussian2 = (1./(std::sqrt(2*M_PI)*sigma2)) * std::exp(-0.5*std::pow(angle/sigma2,2));
  return ratio*gaussian1 + (1-ratio)*gaussian2;
}

//function to computethe likelihood of a given theta_xz angle measurement given a kinetic energy estimate T
double WireCellMCS::lnlikelihood_theta_xz(double angle, double T) {
  std::vector<double> pars = pred_theta_xz_pars(T);
  return -std::log(double_gaussian(angle, pars));
}

/*
//function to computethe likelihood of a given theta_yz angle measurement given a kinetic energy estimate T
double WireCellMCS::lnlikelihood_theta_yz(double angle, double T, double vx) {

  // Define vx edges and emu edges
  std::vector<double> vx_edges  = {0, 0.1, 0.2, 0.35, 0.75, 1};
  std::vector<double> emu_edges = {0, 550, 600, 1250, 1300};

  // Determine the vx_index based on vx
  double vx_abs = std::abs(vx);
  int ivx = 0*(vx_abs>=vx_edges[0] && vx_abs<vx_edges[1]) + 1*(vx_abs>=vx_edges[1] && vx_abs<vx_edges[2]) + 2*(vx_abs>=vx_edges[2] && vx_abs<vx_edges[3]) + 3*(vx_abs>=vx_edges[3] && vx_abs<vx_edges[4]) + 4*(vx_abs>=vx_edges[4] && vx_abs<vx_edges[5]);

  //probability using PDFs from each vx slice
  std::vector<double> pvx = { double_gaussian(angle,pred_theta_yz_pars(T,0)), double_gaussian(angle,pred_theta_yz_pars(T,1)), double_gaussian(angle,pred_theta_yz_pars(T,2)), double_gaussian(angle,pred_theta_yz_pars(T,3)), double_gaussian(angle,pred_theta_yz_pars(T,4)) };

  //apply a different PDF based on vx slice. For highest vx slices, drop down to lower vx slices at large energies
  double probability = 0.0;
  double scale1 = (T-emu_edges[1]) / (emu_edges[2]-emu_edges[1]);
  double scale2 = (T-emu_edges[3]) / (emu_edges[4]-emu_edges[3]);
  if      (ivx == 4) {
    if                           (T < emu_edges[1]) { probability = pvx[ivx]; }
    else if (T >= emu_edges[1] && T < emu_edges[2]) { probability = (1-scale1)*pvx[ivx] + scale1*pvx[ivx-1]; }
    else if (T >= emu_edges[2] && T < emu_edges[3]) { probability = pvx[ivx-1]; }
    else if (T >= emu_edges[3] && T < emu_edges[4]) { probability = (1-scale2)*pvx[ivx-1] + scale2*pvx[ivx-2]; }
    else if (T >= emu_edges[4])                     { probability = pvx[ivx-2]; }
  } else if (ivx == 3) {
    if                           (T < emu_edges[3]) { probability = pvx[ivx]; }
    else if (T >= emu_edges[3] && T < emu_edges[4]) { probability = (1-scale2)*pvx[ivx] + scale2*pvx[ivx-1]; }
    else if (T >= emu_edges[4])                     { probability = pvx[ivx-1]; }
  } else if (ivx <= 2)                              { probability = pvx[ivx]; }

  probability = pvx[ivx]; //debugging
  if (probability==0) { std::cout << "error, probability = 0" << std::endl; }
  return -std::log(probability);
}
*/

//function to computethe likelihood of a given theta_yz angle measurement given a kinetic energy estimate T
double WireCellMCS::lnlikelihood_theta_yz(double angle, double T, double vx) {

  // Define vx edges and emu edges
  std::vector<double> vx_edges  = {0, 0.1, 0.2, 0.35, 0.75, 1};
  std::vector<double> emu_edges = {600, 950, 1300};

  // Determine the vx_index based on vx
  double vx_abs = std::abs(vx);
  int ivx = 0*(vx_abs>=vx_edges[0] && vx_abs<vx_edges[1]) + 1*(vx_abs>=vx_edges[1] && vx_abs<vx_edges[2]) + 2*(vx_abs>=vx_edges[2] && vx_abs<vx_edges[3]) + 3*(vx_abs>=vx_edges[3] && vx_abs<vx_edges[4]) + 4*(vx_abs>=vx_edges[4] && vx_abs<vx_edges[5]);

  //probability using PDFs from each vx slice
  std::vector<double> pvx = { double_gaussian(angle,pred_theta_yz_pars(T,0)), double_gaussian(angle,pred_theta_yz_pars(T,1)), double_gaussian(angle,pred_theta_yz_pars(T,2)), double_gaussian(angle,pred_theta_yz_pars(T,3)), double_gaussian(angle,pred_theta_yz_pars(T,4)) };

  //apply a different PDF based on vx slice. For highest vx slices, drop down to lower vx slices at large energies
  double probability = 0.0;
  double width = 50;
  double scale1 = sigmoid((T-emu_edges[0])/width);
  double scale2 = sigmoid((T-emu_edges[2])/width);
  if      (ivx == 4) {
    if      (T <  emu_edges[1]) { probability = (1-scale1)*pvx[ivx]   + scale1*pvx[ivx-1]; }
    else if (T >= emu_edges[1]) { probability = (1-scale2)*pvx[ivx-1] + scale2*pvx[ivx-2]; }
  } else if (ivx == 3) {
                                { probability = (1-scale2)*pvx[ivx]   + scale2*pvx[ivx-1]; }
  } else if (ivx <= 2)          { probability = pvx[ivx]; }

  return -std::log(probability);
}

//function to computethe likelihood of a given series of theta_xz and theta_yz angle measurement given an energy estimate E
double WireCellMCS::lnlikelihood_track(double* KE, double* par){
  int nsegs = par[0];
  double lnlikelihood = 0;
  for (int i=2; i<nsegs+1; i++) {
    double theta_xz = par[i + nsegs];     // angle in x-projection
    double theta_yz = par[i + 2*nsegs];   // angle in y-projection
    double vx       = par[i + 3*nsegs];   // vx slice
    
    double distance1 = par[i-1];
    double distance2 = par[i];
    double rrtot_guess = uRRfromKE->Eval(KE[0]);
    double rrguess1 = std::max(rrtot_guess - distance1, 1.); //Ensure at least 1 cm of distance
    double rrguess2 = std::max(rrtot_guess - distance2, 1.);
    double keguess1 = uKEfromRR->Eval(rrguess1);
    double keguess2 = uKEfromRR->Eval(rrguess2);
    //double keguess1  = std::max(1.0, uKEfromEX->Interpolate(KE[0], distance1)); // Ensure energy stays above 1 MeV
    //double keguess2  = std::max(1.0, uKEfromEX->Interpolate(KE[0], distance2));
    double keguess   = (keguess1+keguess2)/2 ; // angles is matched to avg energy of the two segments 
	
    //compute likelihood
    double lnl_xz = lnlikelihood_theta_xz(theta_xz, keguess);
    double lnl_yz = lnlikelihood_theta_yz(theta_yz, keguess, vx);
    lnlikelihood += lnl_xz + lnl_yz;
  }
    return lnlikelihood;
}

//Estimate the most likely energy estimate by combining the likelihood predictions from theta_xz and theta_yz
std::vector<double> WireCellMCS::estimate_energy(std::vector<double> segs_distance, std::vector<double> segs_angle_x, std::vector<double> segs_angle_y,  std::vector<double> vx_comps){
  // Combine angle and distance vectors for intput into TF1
  segs_distance.insert(segs_distance.begin(), segs_distance.size());  // Add counter as the first entry
  segs_distance.insert(segs_distance.end(), segs_angle_x.begin(), segs_angle_x.end());  // Combine angle_x
  segs_distance.insert(segs_distance.end(), segs_angle_y.begin(), segs_angle_y.end());  // Combine angle_y
  segs_distance.insert(segs_distance.end(), vx_comps.begin(), vx_comps.end());  // Combine vx components

  double emin = 0;
  double emax = 4e3; //4 GeV max estimate

  // Create a TF1 object to minimize the likelihood
  //TF1* f_lnlikelihood2 = new TF1("Negative LnLikelihood of Given Track using Angle-Based MCS", lnlikelihood_track, emin, emax, segs_distance.size());
  TF1* f_lnlikelihood2 = new TF1("Negative LnLikelihood of Given Track using Angle-Based MCS", [this](double* KE, double* par){ return lnlikelihood_track(KE,par); }, emin, emax, segs_distance.size());
  f_lnlikelihood2->SetParameters(&segs_distance[0]);

  // Find the energy that minimizes the likelihood
  double keguess = f_lnlikelihood2->GetMinimumX(emin + 1e-3, emax - 1e-3);

  //define lower and upper bounds
  double keguess_lower  = f_lnlikelihood2->GetMinimumX(                      emin + 1e-3,  keguess*0.8);
  double keguess_higher = f_lnlikelihood2->GetMinimumX(std::min(keguess*1.2, emax - 2e-3), emax - 1e-3);

  //get likelihood at e_guess, e_guess_lower, e_guess_higher
  double l_keguess        = std::exp(-lnlikelihood_track(&keguess,       &segs_distance[0]));
  double l_keguess_lower  = std::exp(-lnlikelihood_track(&keguess_lower, &segs_distance[0]));
  double l_keguess_higher = std::exp(-lnlikelihood_track(&keguess_higher,&segs_distance[0]));

  //copmute the ambiguity score as the highest ratio
  double ambiguity_score = std::max(l_keguess_lower/l_keguess ,l_keguess_higher/l_keguess );

  // Clean up
  delete f_lnlikelihood2;
  return {keguess, ambiguity_score};
} 


DEFINE_ART_MODULE(WireCellMCS)
