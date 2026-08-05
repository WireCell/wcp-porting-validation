// SBND reco1 art-file dump job — sbnd_xin's own copy, and now the single
// source of truth for OUR chain.
//
// History.  This job used to live in-tree at
// cfg/pgrapher/experiment/sbnd/wct-reco1-dump.jsonnet, driving the
// SBNDReco1FrameSource / SBNDReco1OpFlashSource factories compiled into
// libWireCellRoot.so.  WCT master commit 5f684887 moved those sources out of
// the toolkit into the standalone plugin
//   https://github.com/WireCell/wire-cell-sbnd-reco1
// (rationale: WCT issue #494 — the mirror ROOT classes here carry the REAL
// LArSoft names with a trimmed layout, so when libWireCellRoot.so was loaded
// inside a lar/LArSoft process ROOT could pick the trimmed dictionary for a
// full-layout product and corrupt memory).  The in-tree cfg was deleted with
// them.
//
// The standalone repo ships its own cfg/wct-reco1-dump.jsonnet, but it was
// forked BEFORE our commit 9cb6c860 and therefore has no MC product-name
// TLAs — it can only read SBND *data* reco1 files.  Rather than diverge that
// clone (it is upstream's, and we want `git pull` there to stay clean), this
// file keeps our full-featured version and simply targets the new plugin.
// The only functional difference from the deleted in-tree file is the plugin
// name: WireCellRoot -> WireCellSBNDReco1.
//
// Verified 2026-08-03: dumping the 48-event 2025fall data file through the
// old in-tree reader and through the standalone plugin gives byte-identical
// frames-dnn.tar.bz2 and opflash_apa{0,1}.tar.gz by hash_archive.py member
// content.  See sbnd_xin/docs/69_master-merge-reco1-externalization.md.
//
// Requires (run_reco1_dump.sh sets both):
//   WIRECELL_PATH    .../wire-cell-sbnd-reco1/install/share/wirecell  (not
//                    strictly needed for this file, but keeps the upstream
//                    cfg reachable)
//   LD_LIBRARY_PATH  .../wire-cell-sbnd-reco1/install/lib
// The installed .so carries an RPATH to ROOT + the WCT prefix (we configure
// with -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON), so no ROOT env is needed.
//
// Dump an SBND reco1 art/LArSoft ROOT file into the standalone-chain
// interchange formats -- directly with the toolkit, no LArSoft.
//
// Toolkit-only counterpart of yuhw's LArSoft dumps
// (input_files/wcls-frame-dump.jsonnet + wcls-flash-dump.jsonnet):
//   SBNDReco1FrameSource   -> FrameFileSink   frames-dnn.tar.bz2
//   SBNDReco1OpFlashSource -> TensorFileSink  opflash_apa{0,1}.tar.gz
//
// By default ALL events in the art file are streamed into the combined
// archives (yuhw sample-dir layout); pass entry >= 0 to dump one event.
//
// caf_offset_mode ('none'|'product'|'auto'|'override'): whether the
// opflash tensor-set metadata carries the per-event frame_apply_at_caf
// (ns) flash-time re-reference.  'product' reads the authoritative
// FrameShiftInfo::fFrameApplyAtCaf written by the sbndcode FrameShift
// producer (requires a *_frameshift.root input); 'auto' approximates it
// from the ported FrameShift derivation (pre-FrameShift files; low by
// ~0.26 us on the dev sample); 'none' omits the key =>
// FlashTensorToOpticalPCs no-op, byte-identical semantics to the older
// 10-event dumps.  See wire-cell-sbnd-reco1/docs/DESIGN.md.
//
// Run (from sbnd_xin/): see run_reco1_dump.sh, or directly:
//   wire-cell -l stderr -L info \
//     --tla-str input=<reco1.root> --tla-str output_dir=<dir> \
//     -c wct-reco1-dump.jsonnet

function(input, output_dir='.', entry='-1', caf_offset_mode='none', caf_offset_override='0',
         wire_product='', badmask_product='', summary_product='', flash_process='Reco1',
         frameshift_product='')
// caf_offset_mode: none | product | auto | override (validated in C++)
//
// wire_product / badmask_product / summary_product: art branch names of the
// three TPC products.  Empty ('') => key omitted => the C++ defaults, which are
// the SBND *data* reco1 names (sptpc2d/dnnsp, process Reco1).  SBND *MC* reco1
// files (detsim-g4-gen) carry the same products under the DetSim process and
// the simtpc2d module label, so an MC extraction passes
//   --tla-str wire_product='recob::Wires_simtpc2d_dnnsp_DetSim.'
//   --tla-str badmask_product='ints_simtpc2d_badmasks_DetSim.'
//   --tla-str summary_product='doubles_simtpc2d_wienersummary_DetSim.'
// (see sbnd_xin/docs/67).  flash_process is the process name of the two
// opflashtpc<N> products ('Reco1' in both data and MC to date).
//
// frameshift_product: art InputTag of the sbnd::timing::FrameShiftInfo
// product consumed in caf_offset_mode=product.  Empty ('') => key omitted =>
// the C++ default 'sbnd::timing::FrameShiftInfo_frameshift__FRAMESHIFT.'
// (the 2025fall data reco1 files' process name).  Added 2026-08-05 (doc 71
// campaign): the NCpi0 sideband file carries the SAME product under a
// DIFFERENT process instance, '...__FILTERFRAMESHIFT.' (an extra filter
// stage ran in that production), so
//   --tla-str frameshift_product='sbnd::timing::FrameShiftInfo_frameshift__FILTERFRAMESHIFT.'
// is required for that one file. Confirm the exact instance name per file
// with a branch-name grep before assuming FRAMESHIFT -- do not guess.

local g = import 'pgraph.jsonnet';

local entry_num = std.parseInt(entry);
local caf_override_ns = std.parseJson(caf_offset_override);

local frame_src = g.pnode({
    type: 'SBNDReco1FrameSource',
    name: 'reco1frames',
    data: {
        filename: input,
        entry: entry_num,
        // wire/badmask/summary products, tag, scales, nticks, tick:
        // C++ defaults match the SBND data reco1 files (sptpc2d, dnnsp).
        // Keys omitted when the TLA is '' => compiled config byte-identical
        // to a pre-knob data extraction.
    } + (if wire_product != '' then { wire_product: wire_product } else {})
      + (if badmask_product != '' then { badmask_product: badmask_product } else {})
      + (if summary_product != '' then { summary_product: summary_product } else {}),
}, nin=0, nout=1);

local frame_sink = g.pnode({
    type: 'FrameFileSink',
    name: 'reco1frames',
    data: {
        outname: output_dir + '/frames-dnn.tar.bz2',
        tags: ['dnnsp'],
        digitize: false,  // float32 output
        masks: true,      // include chanmask_bad_<ident>.npy
    },
}, nin=1, nout=0);

local flash_srcs = [
    g.pnode({
        type: 'SBNDReco1OpFlashSource',
        name: 'tpc%d' % n,
        data: {
            filename: input,
            entry: entry_num,
            flash_product: 'recob::OpFlashs_opflashtpc%d__%s.' % [n, flash_process],
            caf_offset_mode: caf_offset_mode,
            // Only meaningful in override mode; C++ default 0.  Key omitted
            // otherwise => compiled config byte-identical to pre-override runs.
            [if caf_offset_mode == 'override' then 'caf_offset_override']: caf_override_ns,
        } + (if frameshift_product != '' then { frameshift_product: frameshift_product } else {}),
    }, nin=0, nout=1)
    for n in [0, 1]
];

local flash_sinks = [
    g.pnode({
        type: 'TensorFileSink',
        name: 'opflash_sink_apa%d' % n,
        data: {
            outname: output_dir + '/opflash_apa%d.tar.gz' % n,
            prefix: 'opflash_',
        },
    }, nin=1, nout=0)
    for n in [0, 1]
];

local pipes = [g.pipeline([frame_src, frame_sink], 'frames')] +
              [g.pipeline([flash_srcs[n], flash_sinks[n]], 'flash%d' % n) for n in [0, 1]];

local app = {
    type: 'Pgrapher',
    data: {
        edges: std.foldl(function(acc, p) acc + g.edges(p), pipes, []),
    },
};

local cmdline = {
    type: 'wire-cell',
    data: {
        // WireCellSBNDReco1 (standalone plugin) replaces WireCellRoot here --
        // the ONLY functional change from the deleted in-tree version.
        plugins: ['WireCellSBNDReco1', 'WireCellSio', 'WireCellPgraph', 'WireCellAux'],
        apps: ['Pgrapher'],
    },
};

[cmdline] + std.foldl(function(acc, p) acc + g.uses(p), pipes, []) + [app]
