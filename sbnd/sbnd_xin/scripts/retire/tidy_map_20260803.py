#!/usr/bin/env python3
"""Tidy round 2026-08-03: build the name -> new-path map for every top-level
script, and emit it as a TSV plus a scripts/README.md index.

Read-only.  Nothing is moved here -- tidy_move_20260803.sh consumes the TSV.

Design notes:
  - STAY is deliberately small: the runner interface, the jsonnet job configs,
    the four symlinks, and the three cross-campaign python tools with the
    heaviest doc citation (nusel_extract.py has 28).  Everything else moves.
  - Importer/importee pairs are pinned to the SAME destination rather than
    patched with sys.path: ql_prefilter_{tune,parity}, fc_speck_audit +
    inspect_pctree.
  - The order of RULES matters -- first match wins.
"""
import os, re, csv, sys, collections

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
os.chdir(ROOT)

# ---------------------------------------------------------------- STAY
STAY = {
    # the runner interface
    "_runlib.sh",
    "run_img_evt.sh", "run_clus_evt.sh", "run_clust_QL_evt.sh", "run_ql_evt.sh",
    "run_pr_evt.sh", "run_nusel_evt.sh", "run_select_evt.sh", "run_bee_img_evt.sh",
    "run_sp_to_magnify_evt.sh", "run_reco1_dump.sh", "run_pr_chain_batch.sh",
    "run_full1k_nusel.sh",
    # cross-campaign python tools (nusel_extract: 28 doc cites and imported by
    # stm_fv_census; relink_tags: called as $BASE/relink_tags.py by every
    # retire_*.sh; pr_scores_table: the scores table every PR doc quotes)
    "nusel_extract.py", "pr_scores_table.py", "relink_tags.py",

    # --- pinned by tidy_refscan_20260803.py bucket A ---
    # These three are INVOKED by runners that stay:
    #   run_bee_img_evt.sh   -> upload-to-bee.sh, wct-img-2-bee.py
    #   run_clust_QL_evt.sh  -> upload-to-bee.sh
    #   run_select_evt.sh    -> merge_sel_archives.py
    # They belong in scripts/bee/ by topic, but moving them means editing three
    # production runners immediately before the owner's PR-tuning campaign.
    # Keeping them here costs 3 of ~80 top-level entries and keeps every runner
    # BYTE-IDENTICAL, which is the property that actually matters right now.
    # Revisit once the campaign has landed.
    "upload-to-bee.sh", "wct-img-2-bee.py", "merge_sel_archives.py",
}

# ---------------------------------------------------------------- rules
# (destination, regex).  First match wins.
RULES = [
    # --- bee tooling (builders + upload) ---
    ("scripts/bee", r"^(make_scan_bee\.sh|upload-to-bee\.sh|zip-upload\.sh|unzip\.pl"
                    r"|make_pr_bee\.py|make_stmfit_bee\.py|merge_mabc_bee\.py"
                    r"|wct-img-2-bee\.py|bee_frame_probe\.py|merge_sel_archives\.py)$"),
    # --- config compilation ---
    ("scripts/cfg", r"^compile_.*\.sh$"),
    # --- profiling / perf ---
    ("scripts/perf", r"^(profile_pr\d+\.sh|clustering_timing_analysis\.py"
                     r"|pr11_config_cost\.py|p54_ab_report\.py)$"),
    # --- ROOT macros ---
    ("scripts/root", r"^.*\.C$"),
    # --- shell drivers that are not the daily interface ---
    ("scripts/runners", r"^(run_d52_campaign\.sh|run_perf54_nusel\.sh|run_mcbase\.sh"
                        r"|run_pr_geom_arm(_dl)?\.sh|geom_ab_(batch|summary)\.sh"
                        r"|s4_(chain|nuecc48|off)\.sh|d66_flip_plots\.sh"
                        r"|tmp_run_pr_chain_\w+\.sh)$"),
    # --- doc-keyed campaign analysis ---
    ("scripts/analysis/pr11", r"^pr11_.*\.py$"),
    ("scripts/analysis/pr20", r"^pr20_.*\.py$"),
    ("scripts/analysis/pr23", r"^pr23_.*\.py$"),
    ("scripts/analysis/pr24", r"^pr24_.*\.py$"),
    ("scripts/analysis/pr25", r"^pr25_.*\.py$"),
    # --- topical analysis ---
    # ql_prefilter_parity imports ql_prefilter_tune -> same dir, both here
    ("scripts/analysis/ql", r"^(ql_.*\.py|lm_tune\.py|automask_prototype\.py"
                            r"|unmatched_census\.py|check_dead5\.py)$"),
    ("scripts/analysis/light", r"^(pmt_.*\.py|saturation_pe\.py|flash_.*\.py)$"),
    ("scripts/analysis/stm", r"^(stm_.*\.py|stmfit_.*\.py|stmon_stats\.py"
                             r"|d66_.*\.py|bwgate_report\.py|bw_label_census\.py"
                             r"|d52_ab_report\.py|d60_ab_report\.py)$"),
    ("scripts/analysis/cathode", r"^(cathode_.*\.py|kink_probe\.py|cbr_sweep_compare\.py)$"),
    ("scripts/analysis/geom", r"^(pr_proj_align\.py|yz_coverage\.py|tgm_readout_cut\.py)$"),
    # --- everything else analysis ---
    ("scripts/analysis/misc", r"^.*\.py$"),
    # --- loose data products ---
    ("products", r"^(.*\.csv|pr20(on|off)\.txt|upload-.*stmid-map\.txt)$"),
]

def dest(name):
    if name in STAY:
        return None
    for d, pat in RULES:
        if re.match(pat, name):
            return d
    return "UNCLASSIFIED"

# ---------------------------------------------------------------- collect
entries = sorted(e for e in os.listdir('.')
                 if os.path.isfile(e) or os.path.islink(e))

# Files that must never move: the jsonnet job configs the runners load as
# $SBND_DIR/<name>, and the symlinks (input_files -> an external sample tree,
# sbnd_track_fitting.json -> the toolkit cfg tree, referenced by 9 scripts).
PINNED = {e for e in entries if e.endswith('.jsonnet')}
PINNED |= {e for e in entries if os.path.islink(e) and e != 'upload-to-bee.sh'}
PINNED |= {'trash-all-apa.tar.gz'}     # cited by docs 5 and 8 as a path

rows, unclassified = [], []
for e in entries:
    if e in PINNED or e in STAY:
        rows.append((e, "", "STAY"))
        continue
    d = dest(e)
    if d is None:
        rows.append((e, "", "STAY"))
    elif d == "UNCLASSIFIED":
        unclassified.append(e)
    else:
        rows.append((e, f"{d}/{e}", "MOVE"))

if unclassified:
    print("!! UNCLASSIFIED -- add a rule before moving anything:")
    for e in unclassified:
        print(f"     {e}")
    sys.exit(2)

with open("scripts/retire/tidy_map_20260803.tsv", "w", newline="") as fh:
    # lineterminator="\n" is load-bearing: csv.writer defaults to \r\n, and the
    # stray \r made `awk -F'\t' '$3=="MOVE"'` in tidy_move_20260803.sh match
    # nothing at all -- a silent zero-move "success".
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    w.writerow(["name", "newpath", "action"])
    w.writerows(rows)

bydest = collections.Counter(r[1].rsplit("/", 1)[0] for r in rows if r[2] == "MOVE")
nstay = sum(1 for r in rows if r[2] == "STAY")
nmove = sum(1 for r in rows if r[2] == "MOVE")
print(f"STAY {nstay}   MOVE {nmove}   (of {len(entries)} top-level files)")
for d, n in sorted(bydest.items()):
    print(f"  {d:28s} {n}")
print("\nSTAY set:")
for r in rows:
    if r[2] == "STAY":
        print(f"  {r[0]}")
print("\nmap -> scripts/retire/tidy_map_20260803.tsv")
