# 30 — Every matched bundle main carries flag_main_cluster (`flag_matched_mains`)

**Symptom.** SBND evt286021: Bee's op display shows the beam-window flash
`#12: (1.15751 us, 2565.56 pe) TPC1  matching: 19`, but the nusel scan row for
that physical flash reads `bundle grp 7 t=1.158 us main None`, label
`no-bundle` — and the event summary called the whole event `no-bundle`, i.e.
the only in-beam matched cluster was invisible to the selection.

**Root cause.** `flag_main_cluster` was set in exactly one place,
`QLMatching::decompose_cluster_groups` (`match/src/QLMatching.cxx:1243`), and
only on clusters that function actually **split**: a cluster is flagged only if
it carries an `isolated`/`perblob` array holding both a main (`-1`) and
sub-components. A compact, single-component matched cluster is never split, so
it keeps `flag_main_cluster = 0` even though it *is* the main cluster of a real
bundle. Every downstream consumer iterates main-flagged clusters
(`TaggerCheckTGM/STM/FC`, `nusel_extract`'s qualifying-bundle filter), so such a
bundle gets no verdict and no table row.

evt286021 evidence:

| source | says |
|---|---|
| QL log | `flash_bundles_map: flash id 7 time 1157.5 ns, cluster gidx 4 total_pred_light 437.17` |
| pctree `cluster_scalar` | ident 8: `matched_flash_gid=1000007`, **`flag_main_cluster=0`**, `flag_associated_cluster=0` |
| PR log | taggers evaluated clusters 1, 10, 12, 13, 14, 15, 16 only — never 8 |

**Not a display bug.** The `19` vs `8` id mismatch is a separate, expected
effect: `fill_bee_flashes` writes the op layer *before* MABC's clustering
pipeline (`clus/src/MultiAlgBlobClustering.cxx:2216-2233`), so `matching: N`
uses the pre-pipeline **img** enumeration, while the clustering layer and the
pctree use the post-pipeline one. img-19 and clustering-8 are the same object
(141 pts, 8.3 cm, y 189.7–197.2, z 150.0–154.7). Bee's flash→cluster link is
consistent with its *img* view, not its *clustering* view.

**Fix.** New QLMatching knob **`flag_matched_mains`** (C++ default **false**).
When on, `apply_matched_t0s` stamps `main_cluster` on every bundle main that
`decompose_cluster_groups` left unflagged — the point where "this cluster is a
matched main" is known. Off, the legacy split-mains-only flag set is untouched.

| piece | file |
|---|---|
| member + rationale | `match/inc/WireCellMatch/QLMatching.h` (`m_flag_matched_mains`) |
| configure + round-trip | `match/src/QLMatching.cxx` (`flag_matched_mains`) |
| stamp + sentinel log | `match/src/QLMatching.cxx::apply_matched_t0s` |
| jsonnet overlay | `sbnd_xin/qlmatching.jsonnet` (`mainflag_on`, key-suppressed) |
| entry-point TLA | `sbnd_xin/wct-clus-matching-perevt.jsonnet` (`main_flag`, default false) |
| runner | `sbnd_xin/run_ql_evt.sh` — **ON by default for this chain**; `-no-main-flag` / `SBND_QL_MAIN_FLAG=0` restores legacy |

The C++ and jsonnet defaults stay false, so every other config and detector
(PDHD, PDVD, uBooNE) is unaffected; only this runner opts in.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild        # then M1 freshness proof
cd sbnd_xin
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-mainflag ./run_nusel_evt.sh data all -chord
# -> work-mcp10-mainflag/nusel-{table,events}.tsv
# viewer (both batches in one 20-event list):
nusel_display/serve_nusel_scan.sh 5010 --tag mcp10-mainflag \
  --prev ../work-mcp10-merge2:mcp10-merge --prev ../work-mcp10-chord:mcp10-chord \
  --prev ../work-mcp10:mcp10 ../work-mcp10-mainflag ../work-mcp1000-mainflag
```

## Verification

**Compiled config.** knob off ⇒ compiled JSON byte-identical to the pre-change
jsonnet (`diff` empty, `/home/xqian/tmp/mainflag-cfg1/{base_off,new_off}.json`);
knob on ⇒ `"flag_matched_mains" : true` appears once (joint QLMatching node).

**Knob-off runtime gate — PASS, byte-identical.** 10-event mcp10 manifest,
knob-off QL rerun with the new binary into `/home/xqian/tmp/mfgate`, compared
against the pre-change baseline `sbnd_xin/work-mcp10/ql_evt*/`:
20/20 archives identical by `abtest/hash_archive.py` member-content hash
(10 × `mabc-all-apa.zip` + 10 × `pctree-evt*.tar.gz`).
Freshness: `libWireCellMatch.so` 18:24 > sources 18:20/18:21.
`./build/match/wcdoctest-match`: 4 cases / 36 assertions passed.

**Knob-on effect (`work-mcp10-mainflag`, NOT bit-identical — that is the fix).**
Sentinel: `flag_matched_mains: stamped main_cluster on 10 matched main(s) that
decompose_cluster_groups left unflagged`.

- bundles 74 → **117** (+43); **no bundle lost**, all 74 previous bundles keep
  their identity.
- evt286021: the 1.158 us bundle appears as `main_id 8, npts 141, len 8.3 cm,
  in_beam 1, tgm 0 stm 0 fc 0 → nu-candidate`; event label **no-bundle →
  nu-candidate**. No other event label changed (286197 stays `no-bundle` — its
  in-beam flash has no matched cluster at all).
- Of the 43 new bundles, 27 are substantial (>50 pts or >5 cm), including
  several 200–500 cm tracks that no tagger had ever evaluated (e.g. evt285185
  cl 16: 4196 pts / 480 cm; evt285999 cl 5: 6103 pts / 406 cm → TGM). 1 is
  in-beam (evt286021).
- Reconciliation with the 58 matched-but-unflagged clusters counted in the
  knob-off pctrees: 43 of them become tabulated bundles, the other 15 fail the
  `require_in_scope` filter and are reported as out-of-scope shards instead
  (total drops 47 → 62 = +15). 43 + 15 = 58, exactly.
- 4 of the 74 common bundles show an `stm`/`fc` cell flipping between `-1` and
  `0`. These are **torn log lines**, not verdict changes: with more clusters
  evaluated the interleaved-write tear points move, and `nusel_extract` reports
  an unparsed verdict as `-1` (see the `VERDICT` comment block there). No
  `tgm` verdict changed anywhere.

**Gates NOT run, and why.** No PDHD/PDVD (`abtest/ab_compare.sh`) or uBooNE
(`qlport/ab_check.sh`) gate was run. `QLMatching` is shared with them, but the
new code is a single `if (m_flag_matched_mains)` guard inside
`apply_matched_t0s` and no config outside `sbnd_xin/` sets the key (the
compiled-config proof shows it absent when off), so those chains are unchanged
by construction. The SBND knob-off gate above exercises the same guard.

## Second batch — 10 more events (`work-mcp1000-mainflag`)

Entries 10–19 of `staged-mcp2025c-1000evt` (run 18255/1, events 286681, 287099,
287517, 287759, 287825, 288067, 288287, 288397, 288639, 288727), imaging reused
from `work-mcp1000/evt<ID>` by symlink, QL rerun with the knob on:

```bash
cd sbnd_xin
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
for e in 10 11 12 13 14 15 16 17 18 19; do   # 5-way parallel
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-mainflag \
    ./run_nusel_evt.sh data 1 -chord
done
python3 nusel_extract.py --merge work-mcp1000-mainflag/nusel_evt*/nusel-evt*.tsv \
  --out work-mcp1000-mainflag/nusel-table.tsv \
  --events-out work-mcp1000-mainflag/nusel-events.tsv
```

10/10 OK, 96 bundles. Event labels: 7 nu-candidate, 1 cosmic-tagged (287099),
2 no-beam-flash (287759, 288727 — no in-window flash at all). Both work roots
are served together by the viewer, giving one 20-event scan list.

## Open

The knob widens the tagger population by ~60% on this sample, which is the
intended fix but has not been hand-scanned. The new in-beam bundle
(evt286021 cl 8) is untagged and therefore promotes its event to
nu-candidate — scan tag `mcp10-mainflag` carries the `mcp10-merge` baseline
labels forward for exactly that review.
