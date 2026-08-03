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
4. **T_tagger/T_kine trees — INFORMATIONAL** (`vf_tree_compare.py`): scalars
   exact, vector branches as multisets. Not a hard gate because the
   per-candidate feature vectors are NOT run-to-run stable even under
   `setarch -R` with one binary and one input (M4 residual): fill-order
   permutations are routine, and occasional single-value flips occur (e.g.
   `shw_sp_lol_1_v_angle` 32.5→131.8 on r1qlmc evt 16 in the A/A′ smoke)
   while archives and every score column stay bit-identical. A `rows=≠` line
   with `mabc==`, `pctree==` and a clean scores-diff is this known
   instability, not a knob effect.

## Validation (A/A′, 2026-08-02)

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
  HEAD `289d78e4`-era outputs). A change that flips a previously-unevaluated
  event into evaluation (or vice versa) is INVISIBLE to this gate by
  construction. Before shipping a default flip, still run the full-population
  drivers (`run_full1k_nusel.sh` + `run_pr_chain_batch.sh` over everything)
  at least once.
- **valfast arms are transient.** Record the `valfast_compare.sh` summary
  (with tags) in the round doc, then DELETE the `work-vf*-<tag>` and
  `work-*-vf<tag>` roots. This is how `sbnd_xin` stays under control — the
  2026-08-02 retirement round removed 135 GB of exactly such arms.
- Event dirs inside -full nusel roots are symlinks into the BASE imaging
  hubs (M11/M13) — imaging is never regenerated and never written to.
