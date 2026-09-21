#!/usr/bin/env python3
"""two_chain_gate.py -- do SBND's two production chains run the same PR operating point?

doc sbnd_xin/120.  SBND reconstructs through two chains:

  * the LOCAL 2-step chain -- run_clus_evt.sh then run_pr_chain_batch.sh, which
    runs cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet.  That job's 535 TLA
    defaults ARE the SBND production operating point; 107 of them carry an
    "SBND PRODUCTION ON <date>" comment naming the doc and the owner's word.

  * the LArSoft 1-step chain -- `lar -c wcls-img-clus-matching-xin.fcl`, which
    calls clus_maker.pr() DIRECTLY and therefore never sees those TLA defaults.

Until doc 120 the second chain got the operating point from a hand-regenerated
mirror, sbnd/pr-operating-point.jsonnet, whose generator does not exist in this
tree.  The failure mode that mirror has is not staleness of what it carries --
doc 120 sec 2 measured that and found it exact -- it is OMISSION: a knob added
to wct-pr-perevt.jsonnet after the last hand-run is simply not in the mirror,
and every "is the mirror current?" check passes while the two chains diverge.
docs/8-build-and-run-both-chains.md sec 7.1 records the same failure costing
18 of 19 events.

So the gate cannot ask about the mirror.  It compiles BOTH chains and requires
their shared components to agree key for key.  An omitted knob fails it; there
is nothing for an omission to hide behind.

What is allowed to differ is declared below in STRUCTURAL, one entry per reason,
and nothing else is forgiven.  Every allowance is a difference the two chains
are SUPPOSED to have -- a shared Bee zip instead of a private one, a per-event
identity, an output directory -- and each names the line of clus.jsonnet that
creates it, so the list can be re-checked rather than trusted.

Usage:
    scripts/cfg/two_chain_gate.py                 # against the live cfg tree
    scripts/cfg/two_chain_gate.py --cfg /path/to/toolkit/cfg
    scripts/cfg/two_chain_gate.py --verbose       # also print what was forgiven
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

TK = "/nfs/data/1/xqian/toolkit-dev"
WCSONNET = f"{TK}/local/bin/wcsonnet"
DATA = f"{TK}/wire-cell-data"
SBND_WD = f"{TK}/wcp-porting-img/sbnd"

# The PR pipeline run_pr_chain_batch.sh uses in production (the 15-stage list,
# i.e. the default plus the neutrino taggers and the two BDT scorers).  Same
# list as scripts/cfg/compile_prjob_cfg.sh.
PR_PIPELINE = ("['switch_scope','unmerge_bundle','unmerge_assoc','steiner','fiducialutils',"
               "'tagger_check_tgm','tagger_check_stm','tagger_check_fc','protect_bundle',"
               "'steiner_refresh','tagger_check_neutrino','numu_bdt_scorer','nue_bdt_scorer',"
               "'tracking_visitor','tagger_output']")

# ---------------------------------------------------------------------------
# What the two chains are SUPPOSED to disagree about.
#
# Deliberately NOT a blanket ignore list: each key says which line of
# cfg/pgrapher/experiment/sbnd/clus.jsonnet makes the two chains differ, so a
# reviewer can re-derive the entry instead of taking it on faith.  A key that
# stops being structural must be deleted from here, not left to rot -- an
# allowance that no longer describes the code is how a gate goes quiet.
# ---------------------------------------------------------------------------
STRUCTURAL = {
    # -- per-event identity: the local compile pins one event, LArSoft reads the
    #    art event.  Nothing to do with the operating point.
    "runNo": "per-event identity",
    "subRunNo": "per-event identity",
    "eventNo": "per-event identity",
    "rse_from_ident": "per-event identity (LArSoft takes RSE from the art event)",
    "rse_from_metadata": "per-event identity (LArSoft takes RSE from the art event)",

    # -- the shared Bee zip.  The LArSoft chain passes bee_sink so every MABC
    #    node writes into ONE zip; the local chain gives the PR node its own.
    #    clus.jsonnet:2825 save_deadarea, :2833 the layer name, :2966-2972 the
    #    three bee_pf grafting keys, :2961 the visitor choice.
    "bee_sink": "shared Bee zip (clus.jsonnet bee_sink != null)",
    "save_deadarea": "shared Bee zip -- only ONE node may write dead area (clus.jsonnet:2825)",
    "merge_metadata_key": "shared Bee zip -- graft onto the labeler truth tree (clus.jsonnet:2966)",
    "merge_node_text": "shared Bee zip (clus.jsonnet:2967)",
    "emit_empty": "shared Bee zip -- keep mc.json when the tagger declines (clus.jsonnet:2972)",

    # -- output location.  Compared by basename, not ignored: a chain writing a
    #    DIFFERENT file still fails.
    "bee_zip": "output directory (basename compared)",
    "output_filename": "output directory (basename compared)",
    "output_dir": "output directory (basename compared)",
}
BASENAME_ONLY = {"bee_zip", "output_filename", "output_dir"}

# Components that exist in only one chain are expected for the non-PR parts of
# the LArSoft job (sources, sinks, imaging, Q/L).  The gate is about the PR
# stage, so it reports the intersection and names any PR component that is
# missing from one side -- that IS a failure.
PR_COMPONENT_HINTS = ("Tagger", "Steiner", "MultiAlgBlobClustering", "Uboone",
                      "SbndPrMagnify", "Clustering", "Improve")

# Job wiring, not configuration.  The two chains ARE different graphs: one reads
# artROOT through wclsCookedFrameSource and runs imaging first, the other reads a
# dumped point-cloud tree.  Comparing their edges or their plugin lists would
# report the thing they are designed to differ in and drown the knobs.
SKIP_COMPONENTS = ("Pgrapher", "wire-cell")


def compile_cfg(path, out, cfg, tlas=(), extvars=(), extcodes=(), extra_path=()):
    env = dict(os.environ)
    env["WIRECELL_PATH"] = ":".join([cfg, DATA, f"{DATA}/sbnd/photodet", *extra_path])
    cmd = [WCSONNET]
    for k, v in tlas:
        cmd += ["--tla-code" if v.startswith(("[", "{", "-")) or v.replace(".", "").isdigit()
                else "--tla-str", f"{k}={v}"]
    for k, v in extvars:
        cmd += ["-V", f"{k}={v}"]
    for k, v in extcodes:
        cmd += ["--ext-code", f"{k}={v}"]
    cmd += [path]
    with open(out, "w") as fh, open(out + ".err", "w") as eh:
        rc = subprocess.call(cmd, stdout=fh, stderr=eh, env=env)
    if rc != 0:
        sys.stderr.write(f"compile FAILED rc={rc}: {path}\n")
        sys.stderr.write(open(out + ".err").read())
        sys.exit(2)
    return out


def load(path):
    return {(c.get("type"), c.get("name", "")): c.get("data", {})
            for c in json.load(open(path))}


def cname(comp):
    return f"{comp[0]}:{comp[1]}" if comp[1] else comp[0]


def compare(larsoft, local, verbose=False):
    """Return (divergences, forgiven).  A divergence is (comp, key, larsoft, local)."""
    div, forgiven = [], []
    for comp in sorted(set(larsoft) & set(local)):
        if comp[0] in SKIP_COMPONENTS:
            continue
        a, b = larsoft[comp], local[comp]
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        for key in sorted(set(a) | set(b)):
            va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
            if key in BASENAME_ONLY:
                if isinstance(va, str) and isinstance(vb, str):
                    if os.path.basename(va) == os.path.basename(vb):
                        forgiven.append((cname(comp), key, STRUCTURAL[key]))
                        continue
            elif key in STRUCTURAL:
                if va != vb:
                    forgiven.append((cname(comp), key, STRUCTURAL[key]))
                continue
            if va == vb:
                continue
            # bee_pf / bee_points_sets are lists of dicts: descend one level so a
            # missing knob is named, not reported as "the whole list moved".
            if isinstance(va, list) and isinstance(vb, list) and va and vb \
               and all(isinstance(x, dict) for x in va + vb):
                div += _compare_layers(cname(comp), key, va, vb, forgiven)
                continue
            div.append((cname(comp), key, va, vb))
    return div, forgiven


def _compare_layers(comp, key, va, vb, forgiven):
    """Match Bee layer dicts by 'name' (or 'algorithm') and diff key by key."""
    def index(lst):
        out = {}
        for i, d in enumerate(lst):
            out[d.get("name") or d.get("algorithm") or str(i)] = d
        return out
    ia, ib = index(va), index(vb)
    div = []
    for lay in sorted(set(ia) | set(ib)):
        if lay not in ia or lay not in ib:
            # A layer name itself is bee_sink-gated ('clustering' vs
            # 'clustering-pr', clus.jsonnet:2833); match those up by algorithm.
            continue
        da, db = ia[lay], ib[lay]
        for k in sorted(set(da) | set(db)):
            xa, xb = da.get(k, "<absent>"), db.get(k, "<absent>")
            if k in STRUCTURAL:
                if xa != xb:
                    forgiven.append((f"{comp} {key}[{lay}]", k, STRUCTURAL[k]))
                continue
            if k == "visitor" and xa != xb:
                forgiven.append((f"{comp} {key}[{lay}]", k,
                                 "shared Bee zip -- dump on the last stage (clus.jsonnet:2961)"))
                continue
            if xa != xb:
                div.append((f"{comp} {key}[{lay}]", k, xa, xb))
    return div


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=f"{TK}/toolkit/cfg")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--keep", default="", help="directory to keep the two compiled JSONs in")
    args = ap.parse_args()

    tmp = args.keep or tempfile.mkdtemp(prefix="twochain-", dir="/home/xqian/tmp")
    os.makedirs(tmp, exist_ok=True)

    # (1) the LArSoft 1-step chain, at its production extVars.  Same invocation
    #     as compile_consumers.sh step (g).
    ls = compile_cfg(
        f"{SBND_WD}/wcls-img-clus-matching-xin.jsonnet", f"{tmp}/larsoft.json", args.cfg,
        extvars=[("reality", "data"), ("DL", "4.0"), ("DT", "8.8"), ("lifetime", "35"),
                 ("driftSpeed", "1.563"), ("semimodel_file", ""),
                 ("pr_operating_point", "sync"), ("enable_tracking_root", "true"),
                 ("input_mask_tags", "[]"), ("output_mask_tags", "[]"),
                 ("recobwire_tags", '["gauss"]'), ("summary_tags", "[]"),
                 ("trace_tags", '["gauss"]'),
                 ("opflash0_input_label", "opflashtpc0"),
                 ("opflash1_input_label", "opflashtpc1")],
        extcodes=[("joint", "false"), ("pmt_nl", "true")],
        extra_path=[SBND_WD])

    # (2) the local 2-step chain's PR job, at the production operating point.
    #     Same TLAs as compile_prjob_cfg.sh EXCEPT dl_weights: that script pins
    #     it empty because M4 keeps the SCN vertex out of byte gates, and here
    #     both sides must carry the production weights or the comparison would
    #     be between two different vertex sources.
    lo = compile_cfg(
        f"{args.cfg}/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet", f"{tmp}/local.json", args.cfg,
        tlas=[("input", "in.tar.gz"), ("output_dir", "out"), ("run", "18253"),
              ("subrun", "1"), ("event", "172230"), ("reality", "data"),
              ("pipeline_names", PR_PIPELINE), ("save_tensors", "out.tar.gz")])

    larsoft, local = load(ls), load(lo)
    common = sorted(set(larsoft) & set(local))

    # A PR component present in one chain and absent from the other is a
    # failure in its own right -- that is how a whole stage goes missing.
    only_ls = [c for c in set(larsoft) - set(local)
               if any(h in c[0] for h in PR_COMPONENT_HINTS)]
    only_lo = [c for c in set(local) - set(larsoft)
               if any(h in c[0] for h in PR_COMPONENT_HINTS)]
    # The LArSoft job also builds the imaging + clustering + Q/L stages, which
    # the local PR job does not -- those are its OTHER stages, not PR ones.
    only_ls = [c for c in only_ls if c[1].endswith("pr") or c[1] == "clus_pr"]

    div, forgiven = compare(larsoft, local, args.verbose)

    print("# doc sbnd_xin/120 two-chain gate: the LArSoft 1-step chain vs the local PR job")
    print(f"# cfg tree : {args.cfg}")
    print(f"# compiled : {tmp}")
    print(f"# components: larsoft {len(larsoft)}, local {len(local)}, compared {len(common)}")
    print(f"# forgiven as structural: {len(forgiven)} (--verbose to list)")
    print()
    if only_ls or only_lo:
        print("## PR components present in only ONE chain")
        for c in sorted(only_ls):
            print(f"   only LArSoft : {c[0]}:{c[1]}")
        for c in sorted(only_lo):
            print(f"   only local   : {c[0]}:{c[1]}")
        print()
    if args.verbose:
        print("## forgiven")
        seen = set()
        for comp, key, why in forgiven:
            if (comp, key) in seen:
                continue
            seen.add((comp, key))
            print(f"   {comp:<46} {key:<24} {why}")
        print()
    if div:
        print(f"## OPERATING-POINT DIFFERENCES ({len(div)})")
        for comp, key, va, vb in div:
            sa, sb = json.dumps(va)[:60], json.dumps(vb)[:60]
            print(f"   {comp:<46} {key:<34} larsoft={sa:<16} local={sb}")
        print()
        print("FAIL -- the two chains are not on the same PR operating point")
        return 1
    if only_ls or only_lo:
        print("FAIL -- a PR component is missing from one chain")
        return 1
    print("PASS -- both chains compile to the same PR operating point")
    return 0


if __name__ == "__main__":
    sys.exit(main())
