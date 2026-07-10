// Per-event SBND pattern-recognition (PR) job — standalone, self-contained.
//
// Loads the persisted post-QL point-cloud tree written by the Q/L job
// (run_ql_evt.sh -save-pctree -> work/ql_evt<ID>/pctree-evt<ID>.tar.gz) and
// runs the PR-tail visitors on it inside a MultiAlgBlobClustering node.  With
// the default empty pipeline this is the round-trip identity gate: the
// 'clustering' Bee layer of the output mabc-pr.zip must be content-identical
// to the Q/L job's mabc-all-apa.zip clustering layer.
// See ../docs/sbnd-pattern-recognition.md.  Driven by run_pr_evt.sh.
//
// Usage (called from run_pr_evt.sh):
//   wire-cell \
//     --tla-str  input=work/ql_evt2/pctree-evt2.tar.gz \
//     --tla-code anode_indices='[0,1]' \
//     --tla-str  output_dir=work/pr_evt2 \
//     --tla-code run=0 --tla-code subrun=0 --tla-code event=2 \
//     --tla-str  reality=sim \
//     --tla-code 'pipeline_names=[]' \
//     -c wct-pr-perevt.jsonnet

local g = import 'pgraph.jsonnet';
local wc = import 'wirecell.jsonnet';
local tools_maker = import 'pgrapher/common/tools.jsonnet';

function(
    input          = '.',
    anode_indices  = [0, 1],
    output_dir     = '.',
    run            = 0,
    subrun         = 0,
    event          = 0,
    reality        = 'sim',
    // Same LAr TLAs as the Q/L job so the anode/params objects are identical.
    DL             = 6.2,
    DT             = 9.8,
    lifetime       = 6.0,
    driftSpeed     = 1.563,
    // PR visitors to run, by name (see clus_pr's cm_by_name in
    // cfg/pgrapher/experiment/sbnd/clus.jsonnet). [] = pass-through (identity gate).
    pipeline_names = [],
    // When set, re-save the (post-PR) tree to this TensorDM tarball.  Used by the
    // round-trip gate (re-save with pipeline_names=[] and compare member hashes
    // against the input tarball).
    save_tensors   = '',
)
    local base = import 'pgrapher/experiment/sbnd/simparams.jsonnet';
    local params = base {
        lar: super.lar {
            DL:          DL * wc.cm2 / wc.s,
            DT:          DT * wc.cm2 / wc.s,
            lifetime:    lifetime * wc.ms,
            drift_speed: driftSpeed * wc.mm / wc.us,
        },
    };

    local tools_all = tools_maker(params);
    local anodes  = [tools_all.anodes[i] for i in anode_indices];

    local source = g.pnode({
        type: 'TensorFileSource',
        name: 'pr_pctree',
        data: { inname: input, prefix: 'clustering_' },
    }, nin=0, nout=1);

    local clus_mod = import 'clus.jsonnet';
    local clus_maker = clus_mod(
        output_dir=output_dir,
        runNo=run,
        subRunNo=subrun,
        eventNo=event,
        reality=reality,
    );
    local pr = clus_maker.pr(anodes, dump=true,
                             pipeline_names=pipeline_names,
                             tensor_outname=save_tensors);

    local graph = g.intern(
        innodes=[source],
        outnodes=[pr],
        edges=[g.edge(source, pr, 0, 0)],
    );

    local app = {
        type: 'Pgrapher',
        data: { edges: g.edges(graph) },
    };

    local cmdline = {
        type: 'wire-cell',
        data: {
            plugins: ['WireCellGen', 'WireCellPgraph', 'WireCellAux', 'WireCellSio',
                      'WireCellSigProc', 'WireCellImg', 'WireCellClus'],
            apps: ['Pgrapher'],
        },
    };

    [cmdline] + g.uses(graph) + [app]
