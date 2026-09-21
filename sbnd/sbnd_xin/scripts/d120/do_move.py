#!/usr/bin/env python3
"""doc sbnd_xin/120 round A -- relocate the PDHD/PDVD standalone jobs in-tree.

Copies each file into cfg/pgrapher/experiment/<det>/ and replaces the work-dir
copy with a 1-line re-export shim carrying a header in the doc-64 shape.  The
runners `cd` into the work dir and name the file bare, so the shim is what keeps
run_pr_evt.sh / run_clus_evt.sh working with no edit at all.
"""
import os
import sys

TK = "/nfs/data/1/xqian/toolkit-dev"
WD = f"{TK}/wcp-porting-img"
CFG = f"{TK}/toolkit/cfg/pgrapher/experiment"
DEST = {"pdhd": "pdhd", "pdvd": "protodunevd"}

# stem -> one-line description used in the shim header
MOVERS = {
    "pdhd": {
        "wct-pr-perevt": "per-event pattern-recognition job (539 TLA defaults = the PDHD operating point)",
        "wct-clustering": "standalone per-APA + all-APA clustering job",
        "wct-img-all": "standalone imaging job over SP frames",
        "wct-sp-to-magnify": "SP frames -> Magnify ROOT converter",
        "wct-light-reco": "WCT-native light reconstruction (stage 2)",
        "wct-light-allpd-reco": "all-160-photodetector light reconstruction",
        "wct-light-fullstream-reco": "full-stream photodetector light reconstruction",
        "wct-light-convert": "light-data ROOT -> WCT format converter",
    },
    "pdvd": {
        "wct-pr-perevt": "per-event pattern-recognition job (545 TLA defaults = the PDVD operating point)",
        "wct-clustering": "standalone clustering + Q/L job",
        "wct-nf-sp": "pure-WCT noise filtering + signal processing",
        "wct-nf-sp-dnnroi": "noise filtering + signal processing with DNN-ROI",
        "wct-sp-to-magnify": "SP frames -> Magnify ROOT converter",
        "wct-light-reco": "all-40-OpDet light reconstruction",
    },
}

SHIM = '''// PD{DETU} {desc} -- THIN RE-EXPORT.
//
// The implementation now lives in the canonical, in-tree job
//   cfg/pgrapher/experiment/{dest}/{stem}.jsonnet
// which is the single source of truth for it, including every TLA default that
// carries a production value.  Promoted out of the working repo by doc
// sbnd_xin/120 on the owner's ask 2026-09-21 -- the move SBND made for its own
// standalone jobs in doc sbnd_xin/64 on 2026-07-27, applied to PDHD and PDVD.
//
// This file stays as a thin re-export so the runners keep working with no edit:
// run_pr_evt.sh / run_clus_evt.sh `cd` into {det}/ and name this file bare, and
// top-level arguments (--tla-str / --tla-code) bind through the import
// unchanged.  The compiled config is byte-identical either way; the gate is doc
// 120 sec 3 -- every relocated file compiled bare AND at its runner's own TLA
// set, from this same path, before and after the move.
//
// EDIT THE IN-TREE FILE, NOT THIS ONE.

import 'pgrapher/experiment/{dest}/{stem}.jsonnet'
'''


def main():
    dry = "--apply" not in sys.argv
    for det, files in MOVERS.items():
        dest = DEST[det]
        for stem, desc in files.items():
            src = f"{WD}/{det}/{stem}.jsonnet"
            dst = f"{CFG}/{dest}/{stem}.jsonnet"
            if not os.path.exists(src):
                sys.exit(f"missing source: {src}")
            if os.path.exists(dst):
                sys.exit(f"destination exists, refusing: {dst}")
            body = open(src).read()
            shim = SHIM.format(DETU=det[2:].upper(), desc=desc, dest=dest,
                               stem=stem, det=det)
            print(f"{'DRY  ' if dry else 'MOVE '}{src}  ->  {dst}"
                  f"   ({len(body)} B body, {len(shim)} B shim)")
            if dry:
                continue
            with open(dst, "w") as fh:
                fh.write(body)
            with open(src, "w") as fh:
                fh.write(shim)
    print("\ndone" if not dry else "\n(dry run -- pass --apply)")


if __name__ == "__main__":
    main()
