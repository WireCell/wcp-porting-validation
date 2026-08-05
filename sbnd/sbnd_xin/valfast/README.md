# valfast — the 629-event fast PR-validation sample

Purpose: PR-chain development A/B gates used to run the full population
(1000 MCP2025C data + 48 Lynn nueCC + 23 MC = 1071 events) even though only
the events that actually yield PR results can move. This directory pins that
subset — the **629 events with `nu_evaluated=1`** in the doc pr/11 census —
plus a runner and a comparator, so a validation arm takes minutes, not hours.

## Repro (derivation of the manifests)

```bash
cd sbnd_xin
T=docs/pr/11_scores-table.tsv          # doc pr/11 census, 1071 rows
for s in mcp1k nuecc48 r1qlmc r2mc; do
  awk -F'\t' -v s=$s 'NR>1 && $1==s && $15==1 {print $4}' $T | sort -n \
    > valfast/events-$s.txt
done
M=input_files_reco1/staged-mcp2025c-1000evt/entry_event_map.tsv
awk -F'\t' 'NR==FNR {want[$1]=1; next} FNR>1 && ($4 in want) {print $1}' \
  valfast/events-mcp1k.txt $M | sort -n > valfast/entries-mcp1k.txt
wc -l valfast/events-*.txt             # 572 + 47 + 5 + 5 = 629
```

Column 15 `nu_evaluated` is set iff the PR log carries
`TaggerCheckNeutrino: selected main cluster` — i.e. TaggerInfo/KineInfo were
filled with real features. It is the authoritative "yields PR results" flag:
`nu_evaluated=1` ⟺ non-empty `numu_score` ⟺ non-empty `kine_reco_Enu_MeV`
(629 each way, 0 disagreements). NOTE the Bee `track_fit` layer is **not** a
discriminator — `mabc-pr.zip` contains it for evaluated and unevaluated
events alike. `event_label` (nu-candidate/…) is not a proxy either: 83
nu-candidates are unscored and 141 cosmic-tagged events are scored.

| sample  | events | of population |
|---|---|---|
| mcp1k   | 572 | 1000 |
| nuecc48 | 47  | 48  |
| r1qlmc  | 5   | 10  |
| r2mc    | 5   | 13  |

## Files

- `events-<sample>.txt` — event IDs, one per line, per sample (IDs collide
  across samples — evt 12 is in both MC sets — so per-sample files, never one
  merged list).
- `entries-mcp1k.txt` — the same 572 mcp1k events as art ENTRIES for
  `run_full1k_nusel.sh` (-full mode).
- `run_valfast.sh <tag> [-full] [-j N] [sample ...]` — build one arm.
- `valfast_compare.sh <tagA> <tagB> [sample ...]` — A/B report + PASS/DIFF.
- `vf_tree_compare.py`, `nusel_hash_compare.py` — helpers (see below).

## Usage

```bash
# PR-tail A/B of a PR-stage knob (default OFF vs ON):
./valfast/run_valfast.sh myknoboff
SBND_MY_KNOB=1 ./valfast/run_valfast.sh myknobon
./valfast/valfast_compare.sh myknoboff myknobon

# change can touch clustering / Q-L products => -full (regenerates the nusel
# stage on the subset first; ~+15 min/arm):
./valfast/run_valfast.sh myknoboff -full
```

- **PR-tail mode** (default): the PR chain (`run_pr_chain_batch.sh`, the full
  13-stage pipeline incl. `tagger_check_neutrino`) runs from pinned KEEP
  ql_roots: `work-mcp1kall-d59k` / `work-nuecc48-nuf` / `work-r1ql-first10` /
  `work-r2patrec-f1`. Both arms share those inputs byte-for-byte, so any
  archive diff is attributable to the change. **Caveat**: those pctrees
  predate the pr/14–pr/20 clustering defaults — fine for A/B, but they are
  not today's production clustering.
- **-full mode**: regenerates the nusel stage per sample first (mcp1k via
  `TAG=vf<tag> ENTRIES=… run_full1k_nusel.sh`; nuecc48 via the
  `scripts/runners/s4_nuecc48.sh` pattern with imaging symlinked from `work/`; MC via the doc
  67 recipes), then the PR chain from those fresh roots. Use whenever the
  change can move clustering/Q-L products.

Knob env vars (`SBND_*`) pass through untouched. Existing output roots are
refused (M13): new run ⇒ new tag.

## Gates in `valfast_compare.sh`

1. **PR archives — HARD**: `hash_archive.py` member hashes of `mabc-pr.zip` +
   `pctree-pr-evt*.tar.gz` per event.
2. **Physics score columns — HARD**: `pr_scores_table.py` output diffed by
   `vf_scores_diff.py` (fork of doc pr/20 gate PI-6's `scripts/analysis/pr20/pr20_scores_diff.py`;
   timing/RSS columns excluded by name; `kine_reco_Enu_MeV` compared with a
   1e-5 RELATIVE tolerance because the chain has a documented run-to-run
   noise floor in that one column — last-digit flutter surviving
   `setarch -R`, measured 7 cells / 629 events / max 2e-7 relative on the
   vfaa A/A′ arms. Every other column is exact).
3. **nusel-side archives — HARD, -full arms only**: `mabc-all-apa.zip` +
   `pctree-evt*.tar.gz` + `nusel_evt*/mabc-pr.zip` member hashes
   (`nusel_hash_compare.py`; `tracking-stm.root` is never hashed — ROOT
   embeds timestamps).
4. **All seven trees of `tracking-pr.root`, EXACT — HARD since 2026-08-05**
   (`vf_tree_compare_all.py`): `T_bad_ch Trun T_proj_data T_rec_charge T_proj
   T_tagger T_kine`, branch by branch, through `pr36_cmp.py`'s own
   `_to_py`/`branch_equal`.
5. **`calib-pr-evt<ID>.json` — HARD**, split into tagger / kine / other keys.
   Only present when the arms were built with `PR_EXTRA_STAGES=pr_display`;
   otherwise reported as an absent-side skip. It is the **only** outlet for
   pr/36 F1's `match_isFC` (doc pr/36 §10.9 — not booked in `T_tagger`).

   > **What changed, and why.** Until 2026-08-05 gate 4 covered only
   > `T_tagger`/`T_kine`, compared vector branches as **sorted multisets**, and
   > was **INFORMATIONAL** — because the 2026-08-02 A/A′ below measured
   > fill-order permutations and occasional value flips (`shw_sp_lol_1_v_angle`
   > 32.5→131.8 on r1qlmc evt 16) even under `setarch -R`. Doc pr/37 §2.5
   > re-measured that floor at toolkit `2457320d` and found **0 of 629 events
   > differ, exact, on all seven trees, on both address layouts**. The
   > instability is gone, so the gate is now wide and hard. `VF_CMP_LEGACY=1`
   > restores the old comparator and the old INFORMATIONAL semantics exactly,
   > so every number recorded below stays reproducible.
   >
   > Two trees the old gate never opened carry real signal: doc pr/31 §12.4's
   > "all five fixes are NULL" was overturned by a single `T_rec_charge`
   > branch, and doc pr/32's arms moved `mabc-pr.zip` on four events. Neither
   > was visible to the comparator its round used.

## Validation (A/A′, 2026-08-05 — toolkit `2457320d`, doc pr/37 §2.5)

**Three** PR-tail arms at HEAD, one binary throughout (`libWireCellClus.so`
mtime `2026-08-04 20:55:43`, verified before, between and after every arm), all
four samples, `PR_EXTRA_STAGES=pr_display`, `-j 8`:

| arm | address layout |
|---|---|
| `vf37a` | `setarch x86_64 -R` |
| `vf37b` | `setarch x86_64 -R` — **matched** layout |
| `vf37c` | ASLR **on** — **cross** layout, the property a production campaign needs |

629/629 `rc=0` in all three. Both pairs, compared with the wide comparator at
`VF_CMP_JOBS=24`:

| pair | events | mabc | pctree | trees (ALL, EXACT) | calib | verdict |
|---|---|---|---|---|---|---|
| `vf37a` vs `vf37b` (matched) | 629 | 629/629 | 629/629 | **629/629** | 497/497 | **VALFAST PASS** |
| `vf37a` vs `vf37c` (cross) | 629 | 629/629 | 629/629 | **629/629** | 497/497 | **VALFAST PASS** |

Both also PASS under `VF_CMP_LEGACY=1`. The score columns are exact on all four
samples — **the 7 tolerated `kine_reco_Enu_MeV` noise-floor cells of the
2026-08-02 record did not recur**, so the 1e-5 relative tolerance in
`vf_scores_diff.py` is currently unexercised. It is kept rather than removed:
one clean pair does not retire a documented noise floor.

`calib 497/497` is the count of events that *have* a calib dump — i.e. that
yield a PR result. See the manifest-drift caveat below.

**Retained**, not deleted, contrary to the disposal rule below: these arms are
the measurement that licenses the hard gate, and they are listed in
`scripts/retire/PROTECTED.txt` until the population campaign has run against
the promoted gate.

## Validation (A/A′, 2026-08-02 — superseded, kept for provenance)

Two identical-config PR-tail arms at HEAD (`libWireCellClus.so` md5
`a0594031…`), tags `vfaa1`/`vfaa2`, all four samples, `-j 8`:

- build: **~14 min and 1.9 GB per arm** (572+47+5+5 events; mcp1k arm 1.7 GB);
  629/629 `rc=0` in both arms.
- compare (~35 min): archives **629/629 identical** on both `mabc-pr.zip` and
  `pctree-pr-evt*.tar.gz`; score columns PASS on all four samples with
  exactly the 7 known kine-noise-floor cells tolerated; tree feature
  branches: 536/572 + 9/47 + 5/5 + 5/5 multiset-identical — the remainder is
  the M4 residual described above (heaviest on nueCC48, which has the most
  shower candidates). Overall: **VALFAST PASS**.

The A/A′ arms were deleted after this record, per the disposal rule.

## Caveats

- **The manifest is frozen** (derived once from the pr/11 census at toolkit
  HEAD `289d78e4`-era outputs). An event *outside* the 629 that starts
  evaluating is INVISIBLE by construction — it is never run. (An event *inside*
  the 629 that stops evaluating IS caught: `T_tagger`, `T_kine`, `T_rec_charge`
  and `T_proj_data` stop being written, which both comparators report.)
- **MEASURED 2026-08-05 — the manifest is 21 % stale.** Only **497 of the 629**
  events still yield a PR result at `2457320d`: mcp1k **442/572**, nuecc48
  46/47, r1qlmc 4/5, r2mc 5/5. All `rc=0`; the events reach
  `TaggerCheckNeutrino` and it selects no main cluster. **A large share is input
  vintage, not HEAD behaviour** — the nueCC48 loss is evt 271851, which the
  pinned hub loses (a 253.9 cm isochronous band still glued to the candidate,
  then TGM-tagged) and which the production root `work-nuecc48-prod0803`
  reconstructs fine at the same binary: doc pr/18's `protect_iso_band_xext` is a
  **clustering**-stage fix the pinned hub predates. **Consequence: gate a knob
  campaign in `-full` mode** (or one `-full` regeneration shared by both arms
  via `VF_QLROOT_TAG`), never PR-tail on the pinned hubs. Full analysis: doc
  pr/37 §1.4. Before shipping a default flip, still run the full-population
  drivers (`run_full1k_nusel.sh` + `run_pr_chain_batch.sh` over everything)
  at least once.
- **valfast arms are transient.** Record the `valfast_compare.sh` summary
  (with tags) in the round doc, then DELETE the `work-vf*-<tag>` and
  `work-*-vf<tag>` roots. This is how `sbnd_xin` stays under control — the
  2026-08-02 retirement round removed 135 GB of exactly such arms.
- Event dirs inside -full nusel roots are symlinks into the BASE imaging
  hubs (M11/M13) — imaging is never regenerated and never written to.
