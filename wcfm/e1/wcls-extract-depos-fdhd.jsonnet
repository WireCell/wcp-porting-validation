// wcfm E1 (doc 07): depo extraction from the WC_FM_Sim pilot G4 files.
//
// A two-node Wire-Cell graph run inside a `lar` job (art module WireCellToolkit,
// stock cvmfs dunesw): read the IonAndScint SimEnergyDeposit product of the FD-HD
// 1x2x6 pilot events and write the UN-DRIFTED depos to a WCT depo file, one
// depo_data_N / depo_info_N pair per art event (N = 0-based event counter).
//
//     wclsSimDepoSetSource --> DepoFileSink
//
// Forked by duplication from DNN_ROI_SP/simulation/stageA/wcls-extract-depos.jsonnet
// (PD-HD cosmics); differences: id_is_track set explicitly (the pilot's own sim
// jsonnet used false = SED vector index), outname from the fcl `params`.
//
// Conversions done by wclsSimDepoSetSource (larwirecell SimDepoSetSource.cxx):
// position MidPoint() cm -> mm, time Time() ns, charge = scale * NumElectrons()
// (scale -1: electrons are negative in WCT), id = G4 TrackID (id_is_track),
// pdg = PdgCode(). Energy is NOT written by DepoFileSink (7 data columns
// t,q,x,y,z,dL,dT; 4 info columns id,pdg,gen,child).

local g = import 'pgraph.jsonnet';

local outname = std.extVar('outname');

local deposet = g.pnode({
    type: 'wclsSimDepoSetSource',
    name: '',
    data: {
        model: '',            // ignored by the converter (ElectronsAdapter always)
        scale: -1,
        art_tag: 'IonAndScint',
        assn_art_tag: '',
        id_is_track: true,
    },
}, nin=0, nout=1);

local depofilesink = g.pnode({
    type: 'DepoFileSink',
    name: 'e1depos',
    data: {
        outname: outname,
    },
}, nin=1, nout=0);

local graph = g.pipeline([deposet, depofilesink]);

local app = {
    type: 'Pgrapher',
    data: {
        edges: g.edges(graph),
    },
};

g.uses(graph) + [app]
