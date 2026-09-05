# 39 — A cosmic-taggers-only PDVD chain, and an STM-scoped Bee display

**Status.** Shipped: the reduced chain is the PDVD runner default, and the Bee
set now carries three STM-scoped layers. **Not shipped, gate FAILED:** building
the Steiner graph only for STM candidates. The knob for it exists and is
verdict-neutral; its *ordering prerequisite* is not affordable (§5).

## 0. Repro block

```
# Pinned library used for every arm below (a peer rebuilds local/lib mid-campaign):
#   /home/xqian/tmp/d39/lib_d39/libWireCellClus.so   2026-09-04 17:47  md5 742f9b2df5293e83
# toolkit HEAD at the time: 20773e0b   wcp-porting-img: 128ec5dc
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd

# The deliverable: event 298595 = run 39252, event index 2, on the reduced chain.
# M13 -- fresh tag, pctree symlinked read-only from the source of truth.
mkdir -p work/039252_2_d39stm2
for f in pctree-evt298595.tar.gz pctree-evt298595.tlas img-provenance.txt; do
  ln -sf "$PWD/work/039252_2_d27fresh/$f" "work/039252_2_d39stm2/$f"; done
LD_LIBRARY_PATH=/home/xqian/tmp/d39/lib_d39 ./run_pr_evt.sh -s d39stm2 -stm-fit 39252 2

# The full-PR control, same binary, same event (shows exactly what -stm removes):
LD_LIBRARY_PATH=/home/xqian/tmp/d39/lib_d39 ./run_pr_evt.sh -s d39nu -nu -stm-fit 39252 2

# The sec.5 gate, 20 events x 3 arms (16-way parallel):
#   A  -stm                                            (production order)
#   B  -stmlean  -S steiner_skip_flags=['TGM']         (reorder + skip)
#   C  -stmlean                                        (reorder only)
docs/nf_sp_img_clus/scripts/d39_verdict_census.py work/<ev>_d39gA work/<ev>_d39gB
# full record: stm/gates/d39_tgmskip_gate.txt

# Compiled-config gate (knobs absent => byte-identical):
CFGROOT=<cfg tree at HEAD> abtest/compile_all_cfg.sh <outdir>   # 16 job configs
# plus pdvd_pr (both pipelines) and uboone_mabc compiled by hand, base vs new.
```

## 1. What prompted this

PR validation was blocked by output volume, not by a bug: every event's Bee set
carried `track_fit`, `shower_track`, `vertices` and the `mc` particle flow on
top of `clustering` and `stm_fit`. The ask was to stop the chain at the cosmic
taggers, scope the display to the STM result, and build Steiner only for
clusters that can reach the STM tagger.

## 2. Two things that were already true

Both were asked for; neither needed any work. Recording them because in each
case the **source comment says the opposite**.

- **STM already skips TGM-flagged mains.** `TaggerCheckSTM.cxx:566`,
  `if (main_cluster->get_flag(Flags::TGM)) { ...; continue; }`. The comment at
  `:564-565` — *"No existing pipeline pre-sets the flag, so this is inert unless
  tagger_check_tgm runs earlier"* — is **stale**: both the PDVD and the SBND
  production chains run `tagger_check_tgm` first, so the skip is live. Event
  298595 exercises it 31 times (`"cluster N already TGM; skipping"`).
- **STM already skips fully-contained clusters.** `TaggerCheckSTM.cxx:3421`
  returns on `fc_result.is_fc`, computed by the same `Facade::cluster_fc_check`
  (`Clustering_Util.cxx:75`) that `TaggerCheckFC.cxx:207` uses, and under PDVD
  defaults (`stm_consistent_fv=true`) with the same fiducial. STM computes it
  itself because `tagger_check_fc` runs *after* STM.

## 3. PDVD has no effective beam-coincident bundle

Worth stating because it sizes everything else. `wct-pr-perevt.jsonnet:1030`
sets `beam_window_us = [-10000, 10000]` — readout-wide, "every matched bundle is
in window" (doc 25 §2.1). The `beam_window_only` gate is *on* and selects
*everything*; the log says so directly:

```
visit: TaggerCheckTGM: beam_window_only [-10000.000, 10000.000) us: 92 main(s) evaluated, 0 out of window
```

Combined with `flag_mains` making every flash-matched cluster a main (PDVD's Q/L
matching flags none), that is 92 mains evaluated on one event, 121 clusters
built. This is the source of the volume, and it is why "build Steiner only for
STM candidates" looked worth real time.

## 4. What shipped

### 4.1 The reduced chain is the runner default

`run_pr_evt.sh:76`, `MODE=nu` → `MODE=stm`. `-nu` restores the full PR tail in
one token. No config edit was needed for the layer removal: the four PR layers
are *self-gating* on `visitor: 'TaggerCheckNeutrino:pr'`
(`protodunevd/pr.jsonnet:1826,1854,1887`), that visitor never fires without the
stage, and an unfilled Bee set is never written
(`MultiAlgBlobClustering.cxx:637`). The `mc` set bails the same way (`:3741`).

Measured on event 298595, same binary, `-nu` vs `-stm`:

| layer | `-nu` | `-stm` |
|---|---:|---:|
| `clustering` | 7 532 107 B | 7 532 107 B |
| `stm_fit` | 352 188 | 352 188 |
| `stm` | 219 532 | 219 532 |
| `steiner_graph` | 209 891 | 209 891 |
| `steiner_terminals` | 48 481 | 48 481 |
| `track_fit` | 39 574 | **absent** |
| `shower_track` | 208 392 | **absent** |
| `vertices` | 2 779 | **absent** |
| `mc` | 6 407 | **absent** |

Exactly the four named layers disappear; every other layer is byte-identical
(`hash_archive.py --members`), so the PR tail does not feed anything the cosmic
stage produces. Wall 60 s → 52 s, peak RSS 2.84 → 1.75 GB.

### 4.2 Three STM-scoped Bee layers

Two default-OFF fields on `BeePointsConfig`:

- **`require_flag`** (string, default `""`) — restricts a set to clusters
  carrying a tagger flag, tested in `fill_bee_points`'s three cluster loops.
- **`steiner_terminals_only`** (bool, default `false`) — inside the *existing*
  `pcname == "steiner_pc"` branch, keeps only `flag_steiner_terminal` points.

and three entries in `protodunevd/pr.jsonnet`'s `bee_points_sets`, all bound to
`visitor: 'TaggerCheckSTM:pr'` so they capture the grouping as the tagger saw it
(before `protect_bundle` can split a cluster out from under its flag), and all
wrapped in `if std.member(pipeline_names, 'tagger_check_stm')`.

Event 298595:

| layer | points | clusters | cluster ids |
|---|---:|---:|---|
| `clustering` | 196 745 | 121 | all |
| `stm` | 5 660 | 9 | 39, 40, 55, 86, 87, 97, 100, 109, 111 |
| `steiner_graph` | 5 569 | 9 | same 9 |
| `steiner_terminals` | 1 282 | 9 | same 9 |
| `stm_fit` | 8 869 | **25** | every cluster STM *evaluated* |

Invariants checked: the 9 ids are exactly the 9 `STM=1` log lines;
`steiner_terminals` (1 282) equals the count of `real_cluster_id == 1` inside
`steiner_graph` and every one of its points carries the terminal flag.

The `stm_fit`/`stm` contrast is the useful one and is deliberate (owner
decision): `persist_stm_fit` is called for **every** evaluated main
(`TaggerCheckSTM.cxx:614`, unconditional on the verdict), so `stm_fit` shows 25
candidates and `stm` tells you which 9 were actually tagged.

**Bee set for event 298595 (cosmic-only chain), uploaded at the owner's request:**
https://www.phy.bnl.gov/twister/bee/set/6d8cb2c4-abbc-4f3d-97fc-e430583344e3/event/list/

### 4.3 A crash landmine, removed

The first arm died with SIGSEGV at `MultiAlgBlobClustering.cxx:2923`.
`Dataset::get()` returns a null pointer for an array the cloud does not carry
and the existing code dereferenced it immediately. The steiner cloud uses the
**default-scope array names** (`x_t0cor,y,z`), not plain `x,y,z` — the uBooNE
steiner set spells it correctly (`clus/test/uboone-mabc.jsonnet:387`), my first
config did not. Fixed on both sides: the config uses `t0cor_coords`, and the C++
now names the missing array in a WARN and skips the cluster instead of crashing.

## 5. The gate that FAILED: Steiner only for STM candidates

### 5.1 The half that is impossible

**Fully-contained cannot be excluded from the Steiner build.**
`cluster_fc_check` requires a non-empty `steiner_pc` and returns the
conservative `is_fc=false` without one (`Clustering_Util.cxx:85-90`). FC-ness is
*computed from* the Steiner boundary, so "skip FC clusters" is circular. Only
the TGM half is even expressible.

### 5.2 The knob, and the ordering it needs

`CreateSteinerGraph` gained `skip_flags` (list of strings, default empty),
applied in the cluster filter with an INFO line reporting the saving. Because
`steiner` runs *third* in production — before every tagger — nothing is flagged
yet at that point, so using it requires moving `tagger_check_tgm` ahead of
`steiner`. That is runner mode `-stmlean`. `steiner_refresh` carries the same
list: it runs `replace=false`, i.e. it builds exactly the clusters with no graph
yet, so without it the refresh rebuilds everything the first pass skipped.

### 5.3 Three arms factorize the change exactly

20 PDVD events, same pinned binary, verdict **sets** compared per tagger — not
counts, which hide swaps.

| comparison | isolates | TGM | STM | FC |
|---|---|---|---|---|
| B vs A | reorder **+** skip | 18/20 events differ, 48 clusters | 14/20, 25 | 2/20, 2 |
| C vs A | **reorder alone** | 18/20, **48** | 14/20, **25** | 0/20, 0 |
| B vs C | **skip alone** | **0/20, 0** | **0/20, 0** | 2/20, 2 |

The reorder and the skip separate cleanly:

- **`skip_flags` is verdict-neutral.** Holding the order fixed, TGM and STM are
  identical on all 20 events. Its only effect is FC on 2 events — and both moves
  are on TGM-tagged clusters, which is the predicted and physically right answer
  (a through-going muon is not fully contained).
- **The reorder is the whole problem.** Moving `tagger_check_tgm` ahead of
  `steiner` changes the TGM verdict on 18 of 20 events and pulls clusters out of
  STM into TGM on 14 of 20.

Mechanism, on event 298595 clusters 97 and 118: in the production order TGM
*rejects* them on its charge-support test —

```
check_tgm: cluster 97 CASE-A pair (0,1) rejected: no 30.0 cm-step charge path between the ends (302.8 cm chord)
check_tgm: cluster 118 CASE-B pair (0,5) rejected: rescued end, straight chord 173.0 cm has an unsupported run > 30.0 cm
```

— and in the reordered chain it accepts both (`TGM=true`). TGM's chord-support
test queries the ctpc; running it before `CreateSteinerGraph` gives it a
different view of the charge than production does. **Not a relabeling artefact:**
`clustering-global` is byte-identical across all three arms
(`d13aede9…`), so the cluster ids being compared are the same objects.

### 5.4 Verdict

**Do not flip.** The reorder buys a 23 % wall saving (mean 39 s → 30 s over the
20 events; 33 of 121 clusters skipped on event 298595, both Steiner passes) and
costs a cosmic-tagger verdict change on nearly every event. For a chain whose
entire purpose right now is to validate those taggers, that is the wrong trade.

Both pieces ship **available and OFF**: `skip_flags` defaults to `[]`
(`steiner_skip_flags=[]` in the driver), and `-stmlean` is a runner mode nobody
gets by accident. Re-running the arm is one flag plus one TLA.

## 6. Gates

- **Compiled config, knobs absent ⇒ byte-identical.** All 16 configs in
  `abtest/compile_all_cfg.sh` identical, including `sbnd_pr`; `uboone_mabc`
  identical (0 diff lines). `pdvd_pr` differs on **exactly** the three new
  `bee_points_sets` entries and nothing else — no `skip_flags` key appears, so
  the Steiner stage is unchanged. Checked on both the `-nu` and `-stm` pipelines.
- **`./build/clus/wcdoctest-clus`**: 295 cases / 22 628 assertions pass,
  including a new case pinning `skip_flags` empty.
- **Freshness (M1)**: `libWireCellClus.so` 2026-09-04 17:47, newer than every
  source edit; the new code verified present in the pinned copy before the arms.

## 7. Not established

- The gate ran on **20 events, one detector, data only**. The 18/20 TGM rate is
  a rate on that sample, not a bound.
- The *reason* TGM's chord-support test sees different charge before vs after
  `CreateSteinerGraph` is characterized empirically (which clusters, which
  rejection lines) but not root-caused to a specific cache or population step.
  That is the open question if anyone wants the Steiner saving back.
- `stm_fit` was deliberately **left unrestricted**, so it still shows rejected
  candidates. If the display is still too busy, restricting it is a one-line
  `require_flag: 'STM'` on that entry.
- Pre-existing, reported not fixed: the `steiner_pc` Bee branch hard-codes a
  4000 e threshold in `calc_charge_wcp` (`MultiAlgBlobClustering.cxx:2961`) — a
  uBooNE value applied to a PDVD dump. It affects only the `q` shading of the
  two new Steiner layers.

## 8. Next

1. **Hand-scan the 298595 set** (§4.2 link): are the 9 STM clusters the right 9,
   and do their Steiner terminals sit on the track skeleton?
2. Turn the PR tail back on with `-nu` when the cosmic stage is validated.
3. If the Steiner build cost matters later, §5.3's factorization says the knob
   is sound — what needs solving is giving TGM its production view of the charge
   without running `CreateSteinerGraph` first.

## 9. Related

- doc 25 §2.1 — PDVD's readout-wide beam window
- doc 30 — the same event 298595, `stm_fit` vs `track_fit` (why they disagree)
- doc 37 — Steiner terminals, the 0.5 cm thinning now in production
- doc 38 — the gap-aware end trim, also in this binary
