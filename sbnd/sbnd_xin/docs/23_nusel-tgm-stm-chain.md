# 23 — Neutrino-selection chain: Q/L bundles → TGM/STM taggers → per-bundle label table

**Goal.** Establish the downstream chain that consumes the Q/L matching result
and labels every beam-coincident bundle, as the first stage of a neutrino
selection: run the cosmic taggers (TGM = through-going muon, then STM = stopped
muon) on every matched bundle and extract a per-bundle table — flash time/PE,
main-cluster id, bundle size, tagger flags, label.  The beam light flash sits
in **(0.2, 2.2) µs**; several flashes can fall in that window, so labels are
per *bundle* (one row per matched main cluster), not per event.

First exercised on the MCP2025C reco1 data sample (`work-mcp10`, 10 events,
run 18255; doc 21/22 lineage).  Scripts: `run_nusel_evt.sh` (driver) +
`nusel_extract.py` (table), both in `sbnd_xin/`.  Chain machinery: the PR
milestones M1–M8 of `sbnd/docs/sbnd-pattern-recognition.md` (all pre-existing;
**no toolkit C++ or cfg change was needed for this chain**).

## 1. The chain

```
work/evt<ID>/icluster-*.npz ──run_ql_evt.sh -save-pctree──▶ work/ql_evt<ID>/
    (imaging, prerequisite)        (Q/L matching, prod. defaults)   ├─ mabc-all-apa.zip
                                                                    └─ pctree-evt<ID>.tar.gz
work/ql_evt<ID>/pctree-evt<ID>.tar.gz
    ──wct-pr-perevt.jsonnet──▶  switch_scope → steiner → fiducialutils
                                → tagger_check_tgm → tagger_check_stm
                                ──▶ work/nusel_evt<ID>/{wct_nusel_evt<ID>.log, mabc-pr.zip}
{pctree, PR log} ──nusel_extract.py──▶ work/nusel_evt<ID>/nusel-evt<ID>.tsv
    'all' merge ──▶ work/nusel-table.tsv (per bundle) + work/nusel-events.tsv (per event)
```

- Step 1 is **skipped when the pctree tarball already exists**; when missing
  (e.g. a work tree produced without `-save-pctree`, like the original
  `work-mcp10`/`work-mcp1000` runs) the driver reruns `run_ql_evt.sh
  <mode> -save-pctree <idx>` — the Q/L job is deterministic, so this
  reproduces the same matching and only adds the tarball.
- Pipeline order is load-bearing: `fiducialutils` must precede the taggers
  (they silently no-op without it) and TGM must precede STM
  (`TaggerCheckSTM` skips TGM-flagged mains; a through-going muon is never a
  stopped muon).  The neutrino-PR stage (`tagger_check_neutrino`) is
  deliberately NOT in this pipeline — this chain is the tagger/label stage.

### Usage

```
# one event / all events of the default sample tree (work/):
./run_nusel_evt.sh data 2
./run_nusel_evt.sh data all

# the MCP2025C reco1 10-event tree used here:
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
SBND_WORK_ROOT=$PWD/work-mcp10 ./run_nusel_evt.sh data all

# knobs: -bw l,h   beam window in µs (default 0.2,2.2)
#        -save-pr-tree   also persist the post-PR tree
```

For the 1000-event tree (`work-mcp1000`, staged per-entry under
`input_files_reco1/staged-mcp2025c-1000evt/e<i>/`), run per entry with
`SBND_INPUT_DIR=…/e<i>` the same way the imaging/QL fan-out did (doc 21);
each single-event input dir has exactly one idx (1).

## 2. Table definition

One row per **matched bundle** (= main cluster; QLMatching sets
`flag_main_cluster` per matched flash bundle) plus one row per **beam-window
flash that matched no bundle** (`label=no-bundle` — beam light with no
associated charge is itself selection-relevant).  Columns:

| column | meaning |
|---|---|
| `run subrun event` | RSE (run/subrun read from the reco1 opflash tensor metadata) |
| `main_id` | main-cluster ident (= `cluster N` in the tagger log lines); −1 for no-bundle rows |
| `flash_gid` | matched flash gid (APA1 gids offset +1000000, per-APA matching) |
| `flash_apa` | APA of the matched flash |
| `flash_time_us` | `cluster_t0` in µs = matched flash time, trigger-referenced (CAF frame-shift applied at extraction for MCP2025C; doc 21) |
| `flash_pe` | summed PE of the matched flash (root opflash PC) |
| `in_beam` | 1 if `flash_time_us` ∈ [bw_low, bw_high) |
| `n_bundle` | clusters sharing this `matched_flash_gid` (incl. the main) |
| `n_assoc` | of those, clusters carrying `flag_associated_cluster` |
| `npts_main`, `npts_bundle` | 3d points of the main / of the whole bundle |
| `len_main_cm` | main-cluster bounding-box diagonal (corrected coords) |
| `tgm`, `stm` | tagger verdicts (−1 = no verdict found in the log) |
| `label` | `TGM` \| `STM` \| `nu-candidate` (in-window, untagged) \| `not-tagged` (out-of-window, untagged) \| `no-bundle` |

Per-event summary (`nusel-events.tsv`): `nu-candidate` if any in-window bundle
is untagged; else `cosmic-tagged` if every in-window bundle is TGM/STM; else
`no-bundle` if the window has only unmatched flashes; else `no-beam-flash`.

**Verdicts come from the PR log, not from a re-saved tree** — `set_flag`
writes the scalar PC only on *tagged* clusters, and non-uniform per-cluster
arrays do not survive TensorDM serialization
(`sbnd-pattern-recognition.md` §2.2/§7).  The log lines are stable:
`TaggerCheckTGM: cluster N → TGM=…`, `TaggerCheckSTM: cluster N → STM=… TGM=…`,
`TaggerCheckSTM: cluster N already TGM; skipping`.

Bundle structure (idents, `cluster_t0`, `matched_flash_gid`, main/associated
flags, per-cluster point slices via the `lpcmaps` node→row-count arrays, and
the root opflash PC) comes from the **input** (post-QL) pctree — QLMatching
materializes these on every cluster precisely so they persist.

## 3. First results — MCP2025C reco1, 10 events (run 18255)

All 10 events completed (PR job 2–5 s/event on top of the Q/L rerun;
zero crashes, steiner gracefully skips <2-terminal mains).  116 rows:
108 bundles + 8 unmatched beam-window flashes.

| label | bundles | of which in-window |
|---|---|---|
| TGM | 67 | 6 |
| not-tagged (out-of-window) | 35 | — |
| nu-candidate | 6 | 6 |
| STM | 0 | 0 |

Per-event: **6 nu-candidate** (284657, 285185, 286065, 286241, 286329,
286527), **2 cosmic-tagged** (284349, 285999), **2 no-bundle** (286021,
286197).

The six nu-candidates are all substantial in-time clusters —
124.7/381.4/269.4/310.4/145.1/139.5 cm with 0.4–4.4 k points, flash times
0.58–2.00 µs.  (Whether they are neutrinos or in-time cosmics is the next
stage's question — this chain only establishes the cosmic-tagger labels;
`check_neutrino_candidate`, FC/LM-type flags and the neutrino PR come later.)

### Observations (feed the follow-ups)

1. **In-window TGM tags are dominated by sparse boundary fragments.**  All six:
   3–28 points, 0.5–100 cm, e.g. 284349 main 6 = a 4-point, 0.8 cm sliver at
   the TPC0 anode face (x = −201.5 cm).  They are tagged through the
   prototype-faithful CASE-A branch `ngroups==2 && both-ends-out &&
   no-interior-point`, which is **not** beam-protected — in the prototype
   (`Cosmic_tagger.h:1441`) exactly as in the port.  All other TGM branches
   are conservative for in-window bundles (never tag) until
   `check_neutrino_candidate` is ported.
2. **The beam flash frequently carries two mains: a fragment and the real
   cluster.**  In 285185, 286065, 286329, 286527 the same in-window flash has
   a tiny TGM-tagged main *and* a large nu-candidate main.  The per-bundle
   table (rather than a per-flash or per-event flag) is what keeps these
   separable; a per-flash selection would take the surviving main.  The
   fragment mains are the doc-22 residual class (boundary-fragment
   acquisition) seen from the other side.
3. **A matched cluster can carry neither flag.**  284349's in-window flash
   (gid 4) has main 6 (4 pts) plus a 2176-point cluster with the same gid but
   `flag_main_cluster=0`, `flag_associated_cluster=0` — it therefore gets no
   tagger verdict at all.  It is counted in `n_bundle`/`npts_bundle` (2180
   pts), which is how such rows remain visible.  Check QLMatching's flag
   semantics for multi-cluster bundles if this matters for the selection.
4. **No STM tag in 108 mains** — consistent with the M7 observation; a
   genuine SBND stopped-muon example still awaits a larger sample (the
   1000-event tree is the natural place to look).
5. **Out-of-window bundles labeled `not-tagged` (35) are mostly large
   cosmics whose TGM evaluation failed the geometric tests** (single-end
   exiters etc.) — STM=0, TGM=0.  They are irrelevant to the beam selection
   but kept in the table for tagger-efficiency studies.

## 4. Gotchas

- **Invoke from anywhere, but the real path matters**: `sbnd_xin` is also
  reachable through the `toolkit/sbnd_xin` symlink; the PR jsonnet's relative
  `import '../particle_dataset.jsonnet'` only resolves from the real location
  (`wcp-porting-img/sbnd/`).  `run_nusel_evt.sh` canonicalizes with `pwd -P`
  (note: `run_pr_evt.sh` has the same latent issue — it uses the logical
  path and fails when invoked via the symlink).
- The beam window feeds BOTH the TGM beam protection (in the C++ config, via
  the `beam_window_us` TLA) and the table's `in_beam`/label columns — the
  driver passes the same `-bw` value to both, keep it that way.
- The `-save-pr-tree` option exists for display/round-trip work, but do not
  read tagger flags from that tarball (see §2 on flag serialization).
- `work/nusel-table.tsv` is regenerated by every `all` run from the
  *current* sample's per-event TSVs only (stale `nusel_evt*` dirs from other
  samples in the same work root are not merged).

## 5. Follow-ups

1. Port `check_neutrino_candidate` (2dtoy `ToyFiducial.cxx:1284`) — unlocks
   real TGM decisions on in-window bundles and closes the unprotected
   fragment branch (§3.1).
2. Run the chain over `work-mcp1000` (1000 events): beam-window calibration
   with statistics, STM search, and the in-window TGM fragment rate.
3. Main-cluster selection for beam flashes: the fragment-vs-real-cluster
   split (§3.2) and the unflagged-companion case (§3.3) live in QLMatching's
   bundle/flag assignment — revisit together with doc 22's residual class.
4. Add the LM (light-mismatch) style flags and fully-contained check to the
   label set when their ports land.
