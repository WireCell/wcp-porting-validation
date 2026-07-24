# Pctree flash-merge provenance + exact TGM main-cluster mode (doc 38)

Two coupled changes, both default-OFF:

1. **`save_real_cluster_id`** (MABC config; QL TLA `save_rcid`; runner
   `run_ql_evt.sh -save-rcid`): persist the flash-merge per-blob provenance
   through the pctree tarball, so a tree saved to disk and loaded back
   carries the full in-memory information.
2. **`main_component_mode="real"`** (TaggerCheckTGM; TLA
   `tgm_main_pair_mode`; runner `run_nusel_evt.sh -main-pair-real`): the
   doc-36 main-pair guard reads that provenance and identifies the main
   cluster EXACTLY instead of via the largest-path-component proxy.

## Symptom / motivation

The QL flash merge (`examine_bundles use_flash_t0`) records, per blob, the
pre-merge ident of each merged member (`real_cluster_id` in the "perblob"
PC) — but the array never survived `-save-pctree`:

- The tensor serializer (`TensorDM as_tensors`) concatenates same-named
  local PCs across tree nodes and `Dataset::append` **silently drops** any
  array whose key is absent from the first-seen node's same-named PC.
  Ordinary clusters carry `perblob{isolated}`, merged clusters
  `perblob{isolated, real_cluster_id}` → `real_cluster_id` vanished.
- Even when persisted, the PR job's first pipeline stage (`switch_scope`)
  rebuilds every cluster via `separate()`/`from()`, which copies
  scalars/flags/scopes but NOT node-local PCs.

Consequently the PR job (where the taggers run) had no per-blob main-cluster
information — doc 36 had to use the largest-component proxy, and the PR Bee
dumps lost the per-fragment colors the QL dumps have.

## Fix

Toolkit:

- `clus/src/ClusteringFuncs.cxx` `merge_clusters()`: new optional
  `orig_main_aname` — save a per-blob marker (1/0) of the merged cluster's
  **representative** member (the flash/flags donor, i.e. the one main the
  merged cluster reports).  NB the `main_cluster` FLAG cannot serve here:
  with `flag_matched_mains` every bundle's main carries it, and an 80 ns
  flash group can merge several bundles — all mains of their own bundle
  (evt289343 grp 5: all six members were flagged main; the first "exact"
  attempt using the flag rejected nothing).
- `clus/src/clustering_examine_bundles.cxx`: passes
  `real_cluster_main` alongside `real_cluster_id`.
- `clus/src/MultiAlgBlobClustering.cxx`: `save_real_cluster_id` knob
  (default false).  At tensor-save time, give every perblob-carrying
  cluster the two arrays (fill-in: own ident / all-1) so the perblob key
  set is homogeneous and the serializer keeps everything.  The fill-in
  values reproduce exactly what a reader assumes for an unmerged cluster,
  so the QL Bee output is unchanged (verified byte-identical).
- `aux/src/TensorDMpointtree.cxx`: `as_tensors` now WARNS when it drops a
  heterogeneous key (the silent drop cost this debugging session).
- `clus/src/clustering_switch_scope.cxx`: carry + row-partition the two
  arrays across the `separate()` rebuild, parallel to the blob partition.
  Absent arrays (legacy tarballs) => no-op.
- `clus/src/TaggerCheckTGM.cxx`: `main_component_mode` ("path" default =
  doc-36 proxy; "real" = a pair end is "in the main" iff its blob's
  `real_cluster_main` is 1).  Falls back to the proxy when the array is
  absent.  Fails open on any lookup/size mismatch.

Config threading: `cfg/pgrapher/common/clus.jsonnet`
(`main_component_mode`), `cfg/pgrapher/experiment/sbnd/clus.jsonnet`
(`save_real_cluster_id` on `clus_all_apa`; `tgm_main_pair_mode` on
`clus_pr`/`pr()`), `wct-clus-matching-perevt.jsonnet` (`save_rcid` TLA),
`wct-pr-perevt.jsonnet` (`tgm_main_pair_mode` TLA), `run_ql_evt.sh`
(`-save-rcid`), `run_nusel_evt.sh` (`-main-pair-real`, which also passes
`-save-rcid` to a Q/L step it launches).

## Verification

- `wcdoctest-clus` 518/518, `wcdoctest-aux` 88473/88473,
  `wcdoctest-util` 13697/13697.
- Compiled-config proofs (old HEAD trees vs new, production TLAs):
  QL knob off `cmp`-identical, on adds `save_real_cluster_id: true` once;
  PR mode "path" `cmp`-identical at the doc-36 op point, "real" adds
  `main_component_mode: "real"` once.
- Smoke (evt289343, root `work-mcp1000b-smoke38c`): tarball carries
  int32 `real_cluster_id`/`real_cluster_main` (representative = member 8,
  the 495-blob main track; corner fragment 17 = 0); QL `mabc-all-apa.zip`
  byte-identical to lm2; PR Bee dump regains the QL per-fragment colors;
  sentinel `check_tgm: cluster 9 CASE-A pair (2,3) rejected: neither end
  in the pre-merge main cluster (26.1 cm chord)`; TGM=false.
- Gate A — QL knob-off (new binary, `-save-pctree -lm -calib`, no rcid):
  `work-mcp10-rcidoff` vs `work-mcp10-lm2` — all QL products (3 Bee zips,
  pctree tarball, calib json) — **10/10 byte-identical PASS**.
- Gate B — full 30-event knob-on reprocess `work-mcp{10,1000,1000b}-mainreal`
  (fresh QL `-save-rcid` + PR `-main-pair-real` at the doc-35 op point):
  QL `mabc-all-apa.zip` vs lm2 — **30/30 byte-identical PASS** (the knob
  only adds tarball content); verdict flips vs `*-fvzi` — exactly ONE:
  **evt289343 main 9 TGM 1→0, label TGM → nu-candidate** (all other TSV
  diffs are the usual torn-log `0↔-1` stm/fc parse artifacts, both
  directions); agreement with the doc-36 proxy roots `*-mainpair` —
  verdicts identical on all 30 events (exact == proxy on this sample,
  artifacts aside).
- Gate C1 — PR production-default off-gate (no main-pair flags) on the
  fvzi trees: `work-mcp*-prodoff` vs `work-mcp*-fvzi` —
  **30/30 byte-identical PASS**.
- Gate C2 — PR doc-36 path-mode off-gate (`-main-pair`) on the fvzi trees:
  `work-mcp*-mp36off` vs `work-mcp*-mainpair` —
  **30/30 byte-identical PASS**.
- Viewer: `:5010` tag `mcp10-mainreal` over the three mainreal roots,
  `--prev` mainpair → fvzi (×3) → lm2 → ctpcfix.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild        # + M1 freshness proof
cd sbnd_xin
# fresh roots; per-event IMAGING symlinked from the lm2 roots (QL reruns fresh)
for pair in "work-mcp10-lm2 work-mcp10-mainreal" \
            "work-mcp1000-lm2 work-mcp1000-mainreal" \
            "work-mcp1000b-lm2 work-mcp1000b-mainreal"; do set -- $pair
  mkdir -p $2; for d in $1/evt*; do ln -sfn $PWD/$d $2/$(basename $d); done
done
F="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real"
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-mainreal ./run_nusel_evt.sh data all $F
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
for e in 10 11 12 13 14 15 16 17 18 19; do
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-mainreal ./run_nusel_evt.sh data 1 $F
done
for e in 20 21 22 23 24 25 26 27 28 29; do
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000b-mainreal ./run_nusel_evt.sh data 1 $F
done
for r in work-mcp1000-mainreal work-mcp1000b-mainreal; do
  python3 nusel_extract.py --merge $r/nusel_evt*/nusel-evt*.tsv \
    --out $r/nusel-table.tsv --events-out $r/nusel-events.tsv
done
```
