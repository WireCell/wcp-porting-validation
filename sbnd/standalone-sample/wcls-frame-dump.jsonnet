// Dump recob::Wire (DNN signal ROIs) from an artROOT file into WCT frame files.
//
// Reads recob::Wire back into a WCT frame via wclsCookedFrameSource, then
// writes float32 frames with FrameFileSink (viewable with `wirecell-plot frame`).
//
// All product tags and the output name are parameterized via extVars so the
// same config serves both MC (simtpc2d:dnnsp) and data (sptpc2d:dnnsp); the
// driving fcl sets them through `structs`.

local g = import "pgraph.jsonnet";
local wc = import "wirecell.jsonnet";

local params = import 'pgrapher/experiment/sbnd/simparams.jsonnet';

local wcls_input = g.pnode({
    type: 'wclsCookedFrameSource',
    name: 'sigs',
    data: {
        nticks: params.daq.nticks,
        frame_scale: 50,
        summary_scale: 50,
        frame_tags: ["orig"],
        recobwire_tags:   std.extVar('recobwire_tags'),
        trace_tags:       std.extVar('trace_tags'),
        summary_tags:     std.extVar('summary_tags'),
        input_mask_tags:  std.extVar('input_mask_tags'),
        output_mask_tags: std.extVar('output_mask_tags'),
    },
}, nin=0, nout=1);

local sink = g.pnode({
    type: "FrameFileSink",
    data: {
        outname: std.extVar('outname'),
        tags: std.extVar('trace_tags'),
        digitize: false, // float32 output
        masks: true,     // include chanmask_<name>_<ident>.npy arrays
    },
}, nin=1, nout=0);

local graph = g.pipeline([wcls_input, sink], "main");

local app = {
    type: 'Pgrapher',
    data: { edges: g.edges(graph) },
};

local cmdline = {
    type: "wire-cell",
    data: {
        plugins: ["WireCellGen", "WireCellPgraph", "WireCellSio", "WireCellRoot", "WireCellLarsoft"],
        apps: ["Pgrapher"],
    }
};

[cmdline] + g.uses(graph) + [app]
