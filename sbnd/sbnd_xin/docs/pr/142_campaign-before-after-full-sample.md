# 142 — The EM-clustering + π⁰ campaign, before vs after, on the full 3067-event production

**Status: COMPLETE** (2026-09-01).  Two full-sample arms (3067 events each), three config proofs, Proof C PASS 478/478, and the campaign measured end to end: pi0 exact 27 -> 36 of 66, q_extra 14.9% -> 6.7%, zero nusel or nu_evaluated change, 3 vertex movers, and zero runtime or memory cost.  One unattributed effect is recorded in sec 6.2.

Owner's request: *"a summary of the
pi0 clustering and EM clustering improvements rounds since the beginning … I want
to confirm that we do not have degradation, but improvements … a comparison with
the before and after the entire campaign state … can you also record the memory
and running time distribution"*, and, on the axes: *"1. nueCC, numuCC BDT
2. nusel 3. nu vertex 4. neutrino energy etc. … If there are large discrepancies,
investigation should be done to ensure things are understood."*

**Scope, per the owner's answer to the one open fork**: the "before" arm restores
the **EM/π⁰ knobs only** (docs pr/117–141). Doc 84 rounds 1–4 (MCS + long-muon)
land inside the same commit window and stay **ON in both arms**.

**No code, no knob, no flip.** The toolkit is untouched at `ddce7430`. The only
edit outside this document is two single-line, no-op-by-default hooks in
`run_pr_chain_batch.sh` (`PR_CFG_TREE`, `PR_EXTRA_TLA`), both placed on existing
lines so **no line number in that file moves**.

## Repro block

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# 0. pin the binary (a peer wcbuild has swapped local/lib mid-campaign before)
mkdir -p /home/xqian/tmp/pr142-libsnap
cp -a /nfs/data/1/xqian/toolkit-dev/local/lib/*.so* /home/xqian/tmp/pr142-libsnap/

# 1. the pre-campaign cfg overlay (Proof A's reference side)
mkdir -p $SP/precfg/pgrapher/experiment/sbnd
for f in clus.jsonnet wct-pr-perevt.jsonnet; do
  git -C ../../../toolkit show 8d93260d:cfg/pgrapher/experiment/sbnd/$f \
      > $SP/precfg/pgrapher/experiment/sbnd/$f
done
for f in ../../../toolkit/cfg/pgrapher/experiment/sbnd/*; do
  b=$(basename $f); [ -e $SP/precfg/pgrapher/experiment/sbnd/$b ] || \
      ln -s $(readlink -f $f) $SP/precfg/pgrapher/experiment/sbnd/$b
done

# 2. the restore list, DERIVED (not hand-listed) -- see sec 2.2
#    docs/pr/pr142-restore-empre.tla, 39 entries

# 3. Proof A: the pilot triple on ncpi0 (19 events x 3 arms, ~6 min)
PR_JOBS=16 PR_EXTRA_STAGES=pr_display PR_CFG_TREE=$SP/precfg \
  ./run_pr_chain_batch.sh work-ncpi0-grp0825 work-pr142pilot-precfg-ncpi0 data
PR_JOBS=16 PR_EXTRA_STAGES=pr_display PR_EXTRA_TLA=$PWD/docs/pr/pr142-restore-empre.tla \
  ./run_pr_chain_batch.sh work-ncpi0-grp0825 work-pr142pilot-empre-ncpi0 data
PR_JOBS=16 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-ncpi0-grp0825 work-pr142pilot-head-ncpi0 data

# 4. the two full-sample arms (3067 events each, per-event mode)
./scripts/pr142_arms.sh

# 5. the score tables, then the systematic comparison
for t in empre0901 prod0901; do mkdir -p products/$t
  for s in nuecc48 ncpi0 mcp1k mcp2k; do
    python3 pr_scores_table.py --root work-$s-$t --sample $s \
        --out products/$t/$s-scores-$t.tsv
  done
done
python3 scripts/pr142_campaign_ab.py \
    --a products/empre0901/*.tsv --b products/prod0901/*.tsv \
    --label-a empre0901 --label-b prod0901 \
    --movers-tsv docs/pr/pr142-movers.tsv --summary-tsv docs/pr/pr142-population.tsv
```

---

## 1. The campaign, round by round

Twenty-five documents, 2026-08-28 → 09-01. Sources are each doc's own Status line
and gate ledger; production state is read from
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` at `ddce7430`.

| doc | date | knob(s) | production | headline number |
|---|---|---|---|---|
| **117** | 08-28 | `shower_pass4_best_owner`, `shower_merge_relax` / `shower_flank_absorb` | **ON**, **ON** / OFF | §8: `168596` qF1 **0.852 → 1.000**; under-clustered bucket median 0.887 → **0.899** |
| **118** | 08-28 | `shower_merge_relax_continuity` / `shower_ex1_conn3_body_dis` | **ON** / OFF | §6: 4 merges in 98 events; `423981` **0.750 → 1.000**, no negative deltas |
| **119** | 08-28 | — probe + census only | n/a | §4: 24 OUT + 3 WRONGOWNER of 4248 member rows — the expel predicate is measured dead |
| **120** | 08-28 | `stem_backfill_back_guard` / `shower_ex1_walk_em_track_guard` | **ON** / OFF | §5: exactly **2 events** change; `47212` qF1 0.965 → **1.000** |
| **121** | 08-28 | `shower_ex1_dedup_rehome` | **ON** | §5: fires on **1 event in 239**; `348471` qF1 **0.205 → 0.895** |
| **122** | 08-28 | — probes + census only | n/a | §1: BAD dQ/dx 1.34–1.67 MIP vs GOOD 1.09–1.57 — overlapping; all three guards die |
| **123** | 08-28 | `shower_pass4_prune_detached` (G=40), `shower_pass4_track_guard_len=50`; r2 `kine_count_guard_freed` | **ON** ×3 | §8: **Σ q_extra 4.861e7 → 2.690e7 (−45 %)**, median qF1 0.887 → 0.935 |
| **124** | 08-28 | A `shower_pass4_prune_gap2=25` / C `shower_pass3_cone_guard_len` / B — | **ON** / deferred / — | §A.4: median qF1 **0.935 → 0.949**, Σ q_extra → 2.41e7, **Σ q_miss unchanged** |
| **125** | 08-29 | `shower_pass3_cone_guard_len=15`, `shower_samevtx_track_absorb`, `shower_satellite_absorb` | **ON** ×3 | §5: four fake electrons fixed; 37112 becomes one **797.3 MeV** shower |
| **126** | 08-29 | — audit; recommends fudge → 0.84 | n/a | §4g: π⁰ peak **140.6 MeV** [136.3, 144.3] ⇒ implied fudge **0.829** (a floor) |
| **127** | 08-29 | K5 flip + `sccc_max_gap` 6 → 10 | **ON** | §3.2: 137238's 89 MeV e⁻ reconnects; nusel **239/239** byte-identical |
| **128** | 08-29 | `pf_orphan_near_cross_cluster`, `kine_count_near_cross_cluster`, `pf_conn4_near_candidate`, `kine_count_conn4_near` | **ON** ×4 | §7: **478/478 archives byte-identical** |
| **129** | 08-29 | `kine_guard_freed_impact=20`, `kine_guard_freed_miss_deg=30` | **ON** | §6: blast radius **exactly one event in 239** — 393505 Enu 940.4 → **566.1** |
| **130** | 08-29 | satellites ship `stem_backfill_back_dvtx=45`, `shower_pass4_prox_guard_len=50` | **ON** ×2 | guard-freed pool = **3 objects in 3 events, 710.66 MeV**; 141-set q_miss 43.7 % / q_extra 56.3 % |
| **131** | 08-29 | — census probe only | n/a | §2: **1529.3 of 197457.8 MeV = 0.77 %** reaches no PF output ⇒ *stop population-scale completeness rounds* |
| **132** | 08-30 | fudge **0.80 → 0.84**, K7 `pi0_readmit_retyped`, K8 `pi0_admit_type3`, K3 = 28 | **ON** ×4 | §10.5: **31 exact / 2 fakes** of 66 hand π⁰; γ ledger **90.9 % OK** |
| **133** | 08-30 | K20 `pi0_admit_muon_showers`, K3 28 → **29**, K16 @120 | **ON** | §8: **32 exact / 15 partial / 1 none / 18 no-group** |
| **134** | 08-30/31 | K22 `pi0_nc_frag_merge` + the NC chain, K24 `pi0_prefer_main_vertex` / K23 | **ON** ×6 / OFF | §7: **census 32 → 33 of 66 — exact reaches 50.0 %** |
| **135** | 08-30/31 | — review; §10 records fudge 0.84 → **0.86** | **0.86 ON** | §9: **33 of 66 exact**, γ ledger 90.9 % OK, nueCC-fake counter 0 |
| **136** | 08-30/31 | `shower_pass4_prefilter_v1_escape`, `_max_v2=90` / `_max_dis` | **ON** ×2 / OFF | §5: `q_miss` **14.8 %** / `q_extra` **8.9 %** of target charge on 90 hand-marked showers |
| **137** | 08-31 | — design + feasibility | n/a | §4c: *"best purity across all three families: **27–36 %**"* — the trigger is the hard half |
| **138** | 08-31 | `shower_split` (+7 numerics) | **ON** | §0: `onV1c90 + splitter` → π⁰ exact **32 → 35** of 66, `q_extra` 12.0 % → **6.7 %** |
| **139** | 08-31→09-03 | `shower_split_em_start` / five more split knobs | **ON** / all **OFF, do not flip** | §1: μ-typed daughters **11 → 2**, **461 MeV** of EM energy restored, 0 ADVERSE |
| *(140)* | — | *deliberately unused* — the `pr140_*` scripts belong to doc 139 | — | — |
| **141** | 08-31 | `pi0_mu_shower_hypothesis` (M1), `_max_len` (M2), `_hyp_min_len` (M3) | **all OFF — "flip nothing"** | §16: census **36 of 66 exact**; §22: **≥ 29 % of μ-typed objects > 50 MeV are EM showers**, ≈ 1045 MeV of `kine_reco_Enu` missing |

**Denominator warning.** 26 exact is of **50** hand π⁰ (pr/126); 31 / 32 / 33 / 35 /
36 are of **66** (pr/132 onward). They are not one series. The campaign's π⁰
trajectory on the 66-object denominator is **31 → 32 → 33 → 35 → 36**.

### 1.1 What the campaign changed, in one paragraph

`8d93260d` (2026-08-25 16:35) → `ddce7430` (2026-08-31 20:04): **82 commits,
zero merges**, only two config files touched (`wct-pr-perevt.jsonnet`,
`clus.jsonnet`). The TLA population goes **380 → 504**: **+124 knobs added, 67
genuinely turned ON via 27 owner flips, 2 pre-existing values re-tuned
(`sccc_max_gap` 6→10, `mcs_muon_source`), and nothing turned OFF.**
`sbnd_track_fitting.json`, `qlmatching.jsonnet`, the SP filters and the sim
params are bit-identical on both sides.

Of the 67, **55 are EM/π⁰ (docs pr/117–141)** and **12 are doc 84's MCS /
long-muon family**. This round's "before" arm restores the first group and keeps
the second.

### 1.2 What this comparison can and cannot certify

Doc 85 §1 is explicit that these four samples carry **no truth label anywhere in
this chain** — "NCπ⁰" and "ν<sub>e</sub>CC" name the *sample selection*, not the
per-event interaction. So at population scale a *"no degradation"* claim is
bounded to: no crash/`rc` regression, no `nu_evaluated` selection-rate
regression, no unexplained mover, and distributions that move in the direction
the rounds intended. The campaign's actual quality claims — 36/66 exact π⁰, the
`q_extra` floor — live on the **239-event labeled subsets** and are re-run there
in §7 as a *"did the measured gains survive"* check. They are subset metrics and
are never extended to 3067 events.

---

## 2. The two arms, and the proofs that they are what they claim to be

| arm | out_root | config |
|---|---|---|
| **before** | `work-<s>-empre0901` | HEAD binary + HEAD cfg, 39 EM/π⁰ knobs restored to pre-campaign values |
| **after** | `work-<s>-prod0901` | HEAD binary + HEAD cfg as shipped — **this is the new production product** |

Same binary (pinned), same per-event mode (`PR_GROUP_SIZE` unset), same
`PR_JOBS=16`, samples **interleaved** (`empre` then `prod`, per sample) so each
A/B pair sees the same box conditions. Q/L inputs are the existing
`work-<s>-grp0825` roots: **1000 + 2000 + 19 + 48 = 3067** events.

### 2.1 Why per-event mode

`pr_scores_table.py` has always emitted `wall_s / core_s / timecmd_wall_s /
maxrss_kb`. Those columns are **blank in every committed product table**
(`prod0813`, `prod0819`, `prod0825`, `prod0830`) for one reason: those
productions ran in **group mode**, which writes a single `.time.meta` per
16-event group and slices the per-event logs out of a shared group log. Per-event
mode restores all four columns — verified on the pilot, where every one is
populated. This is what makes the owner's memory/runtime request answerable at
all.

### 2.2 The restore list is derived, not hand-listed

`docs/pr/pr142-restore-empre.tla`, **39 entries**. Built by extracting the
top-level TLA defaults of `wct-pr-perevt.jsonnet` at both commits (380 vs 504
params), taking the added-and-ON plus re-tuned set, subtracting the 12 doc-84
knobs, and choosing each entry's value from that key's own **emission guard** so
the compiled JSON drops the key exactly as the pre-campaign config did
(`[if X then 'X']` ⇒ `false`; `[if X != null …]` ⇒ `null`; `[if X != 0 …]` ⇒ `0`).

Two zero-sentinel traps were checked directly rather than assumed:

- `kine_shower_fudge_factor` is **`null`** pre-campaign (`8d93260d:701`, key
  suppressed ⇒ the C++ 0.80), so restoring to `null` is right. Had it been a
  non-null value, the restore would have been wrong by ~14 % on **every** EM
  energy.
- `kine_guard_freed_miss_deg` needs **no** entry: its key is emitted only when
  `kine_guard_freed_impact != 0`, which the file already sets to 0. Twenty other
  child knobs are likewise already suppressed at HEAD by their parents and are
  correctly absent from the file.

`PR_EXTRA_TLA` is appended **last** to the runner's TLA array, so a restore entry
wins over any `SBND_*` env var; both arms are launched from an environment with
zero `SBND_*` set.

### 2.3 Proof A — the restore list is complete (machine-checked)

The stated risk of any hand-built restore list is a missed knob. Killed by
compiling both sides from **different sources** and requiring the residual to
match a list written down in advance (a `wcsonnet` proof that compares a config
to itself is vacuous — pr/129's lesson):

- reference side: the **pre-campaign cfg tree** at `8d93260d`, via `PR_CFG_TREE`,
  empty environment → `work-pr142pilot-precfg-ncpi0`
- test side: HEAD cfg + the restore file → `work-pr142pilot-empre-ncpi0`

Diffing the two arms' compiled `.wct-cfg-evt<ID>.json` over all 19 ncpi0 events:

> **16 differing keys, on 19 of 19 events, and every single one is doc-84:**
> `mcs_enable`, `mcs_muon_source`, `mcs_range_comparator_chain`,
> `mcs_bridged_members`, `mcs_cathode_x`, `mcs_cathode_xcut`,
> `mcs_point_source`, `mcs_output`, `long_muon_range_empty_chain_fallback`,
> `long_muon_members_geometry`, `long_muon_cathode_bridge`, `…_lever`,
> `…_short_gap`, `…_track_partner`, `long_muon_stub_bridge_len`,
> `long_muon_angle_relax_long`.

The 12 named doc-84 flips plus the 4 children that ride along behind
`mcs_enable`. **Nothing from the EM/π⁰ campaign leaks — PASS.**

### 2.4 Proof A′ — the restored arm differs from production by exactly the campaign

The same diff between `work-pr142pilot-empre-ncpi0` and
`work-pr142pilot-head-ncpi0` returns **40 keys: the 39 restored entries plus
`kine_guard_freed_miss_deg`**, its gated child. No more, no fewer — the "before"
arm is production minus the EM/π⁰ campaign, and nothing else.

### 2.5 Proof C — the "after" arm is the validated operating point (pilot)

`work-pr141r1-off-*` is the natural gate reference: doc pr/141's off arm is
production (its three M knobs ship OFF). Its provenance was checked **before**
the gate rather than after, because reading a FAIL as a physics finding when the
reference is wrong is the easy mistake here:

- its `.batch_pr_evt*.log` header shows the same 16-stage pipeline including
  `pr_display`, `reality=data`, `dl=on`;
- its compiled `.wct-cfg-evt<ID>.json` is **identical on all 19 ncpi0 events, 0
  differing keys**, to this round's HEAD arm.

Gate: `pr85_hash_gate.py work-pr142pilot-head-ncpi0 work-pr141r1-off-ncpi0`
→ **PASS, all 38 archives byte-identical, 0 unpaired, rc=0.** Repeated on the
full manifest in §6.

### 2.6 Runtime instrumentation restored

Per-event mode brings back all four columns, verified on the pilot
(`ncpi0` evt 18625: `wall_s 17.034  core_s 12.828  timecmd_wall_s 18
maxrss_kb 1267972`). `wall_s`/`core_s` come from the job's own
`Timer: Total N wall-sec, M core-sec` line; `maxrss_kb` from `timecmd.py`'s
`RUSAGE_CHILDREN`. **`core_s` is the number to compare** — wall under
`PR_JOBS`-way concurrency is contention-dominated (doc pr/11 §90, doc 76 §1.1) —
and `scripts/pr142_perf.py` goes one level down into the `MABC timing:` and
`MEM: … res=` lines that every PR log has always carried and that nothing has
read at population scale before.

---

## 3. The systematic comparison — 3067 events, `empre0901` → `prod0901`

`scripts/pr142_campaign_ab.py`, full output in the Repro block's command. The 49
degenerate rows (doc 85: `kine_reco_Enu` 0.0 *and* vertex (0,0,0) exactly — an
unmerge shard the tagger selected, `KineInfo` never filled, nothing blanks the
row) are cut before every distribution. `cosmict_flag` is used, never
`cosmic_flag`.

### 3.0 Completeness and failures

| | `empre0901` | `prod0901` |
|---|---|---|
| rows | **3067** | **3067** |
| joined / unpaired | 3067 / **0** | |
| `rc` census | **rc=0 on all 3067** | **rc=0 on all 3067** |

**Zero crashes, zero timeouts, zero unpaired events on either arm.**

### 3.1 nusel — the selection is untouched

| migration | count |
|---|---|
| `nu-candidate` → `nu-candidate` | 1567 |
| `cosmic-tagged` → `cosmic-tagged` | 932 |
| `no-beam-flash` → `no-beam-flash` | 343 |
| `no-bundle` → `no-bundle` | 225 |
| **any label change** | **0** |
| **`nu_evaluated` flips** | **0** |

Per-sample `nu_evaluated` / degenerate / clean counts are identical on both arms
(mcp1k 461/16/445, mcp2k 905/33/872, nuecc48 48/0/48, ncpi0 19/0/19). This is the
expected result — nusel is upstream of everything the campaign touched — and it
is worth having as a null: **the campaign did not gain or lose a single
neutrino candidate.**

### 3.2 Neutrino energy

Median `kine_reco_Enu` on each arm's own clean evaluated population:

| sample | n | `empre0901` | `prod0901` | Δ |
|---|---:|---:|---:|---:|
| nuecc48 | 48 | 1578.2 MeV | **1501.9** | −4.8 % |
| ncpi0 | 19 | 1386.5 | **1322.7** | −4.6 % |
| mcp1k | 445 | 574.8 | 575.1 | +0.1 % |
| mcp2k | 872 | 532.2 | 533.6 | +0.3 % |

Exactly the intended shape: the two **EM-rich** samples lose ~5 %, the two νμ
samples do not move. That is the `kine_shower_fudge_factor` `0.80 → 0.86` scale
(−7 % on EM charge) partly offset by the charge the clustering rounds *recover*.
Across the 120 events whose energy moves past threshold the net is
**−6835 MeV (95 down, 25 up, mean −57.0 MeV)** — a downward re-scaling of EM
energy, not a loss of reconstruction.

### 3.3 νe / νμ BDT

Score medians are flat on the νμ samples (`numu_score` 1.815 → 1.786 mcp1k,
1.608 → 1.605 mcp2k). What matters is the working points (doc 85 §7):

| point | nuecc48 | ncpi0 | mcp1k | mcp2k |
|---|---|---|---|---|
| `numu_score > 0.9` | net **−1** (4 pass both) | net **−1** (7) | net **−2** of 269 | net **−2** of 507 |
| `nue_score > 7.0` (uB) | net **+1** (30 → 31 of 48) | 0 | 0 | 0 |
| `nue_score > 4.30103` | net **+1** | 0 | 0 | net −1 |
| `nue_score > 0.7` | 0 | 0 | 0 | net +1 |

Read as physics: on the **νe sample the νe selection gains an event** and on the
**two νμ samples it gains none** — the campaign moves νe efficiency without
buying νe background. The νμ point loses 1–2 events per sample (0.4–0.7 % of
those passing), and two of the six losses are on the νe and NCπ⁰ samples, where
losing a νμ pass is the desired direction.

Both arms are post-clamp-removal, so this pair is directly comparable. A
`nue_score` comparison against `prod0825` is **not** — see §6.

### 3.4 ν vertex — 3 movers in 3067 events

| sample | event | move | attribution |
|---|---|---:|---|
| mcp2k | 76346 | **60.2 cm** | the π⁰ NC back-projection vertex — the event doc pr/133 §K21 was built for ("to 1.3 cm" of the owner's label) |
| mcp1k | 169626 | **59.6 cm** | π⁰ NC vertex; named in the doc pr/132 r4/r6/r7 commits |
| mcp2k | 499423 | 4.3 cm | π⁰ chain; same main cluster, different reported vertex |

All three keep the **same selected main cluster** (identical
`TaggerCheckNeutrino: selected main cluster` line on both arms); what moves is
the reported neutrino vertex, which is precisely what `pi0_prefer_main_vertex` /
`pi0_bp_vertex_miss_cm` were shipped to do and what the owner approved in
pr/134 §14. **No unexplained vertex motion anywhere in 3067 events.**

### 3.5 Mover census

**312 movers of 3067 = 10.2 %** — mcp2k 168, mcp1k 81, nuecc48 45, ncpi0 18.

| class | n |
|---|---:|
| `numu_score` | 223 |
| `nue_score` | 124 |
| `kine_reco_Enu` | 120 |
| `nue_fill` (νe BDT fill status) | 25 |
| **ν vertex > 1 cm** | **3** |
| `event_label` / `nu_evaluated` / `rc` | **0 / 0 / 0** |

The shape is what 25 rounds of measured-local knobs predict: the campaign moves
*energies and scores*, not *selections*. Every mover class above threshold is
either the EM charge scale, the π⁰ chain, or the shower-clustering knobs — the
three families §1 lists — and the two classes that would signal something
unintended (a label change, a lost candidate) are empty.

---

## 4. Runtime and memory

Per-event, both arms, per-event mode at `PR_JOBS=16`, run sequentially and
interleaved per sample on the same 64-core box (`loadavg` 15–25 throughout, well
inside M5's bar).

| sample | n | arm | wall med | core med | core Σ | peak RSS med / p90 / max |
|---|---:|---|---:|---:|---:|---|
| nuecc48 | 48 | empre / prod | 17.2 / 16.9 s | 16.0 / 16.0 s | 0.31 / 0.31 h | 1.21 / 1.31 / 1.45 GiB ; 1.21 / 1.30 / 1.44 |
| ncpi0 | 19 | empre / prod | 17.6 / 16.3 s | 13.5 / 12.5 s | 0.08 / 0.08 h | 1.18 / 1.27 / 1.28 ; 1.20 / 1.27 / 1.28 |
| mcp1k | 1000 | empre / prod | 6.6 / 6.3 s | 1.5 / 1.6 s | 1.07 / 1.07 h | 0.45 / 1.20 / 1.48 ; 0.45 / 1.20 / 1.48 |
| mcp2k | 2000 | empre / prod | 6.3 / 6.4 s | 1.5 / 1.6 s | 2.15 / 2.16 h | 0.45 / 1.19 / 1.39 ; 0.45 / 1.20 / 1.39 |
| **ALL** | **3067** | **empre / prod** | **6.5 / 6.5 s** | **1.7 / 1.7 s** | **3.61 / 3.61 h** | **0.46 / 1.20 / 1.48 ; 0.46 / 1.20 / 1.48** |

> **The campaign is cost-neutral. 67 new production knobs cost 3.61 core-hours
> against 3.61, and not one megabyte of peak RSS** (median 0.46 GiB, p90 1.20,
> max 1.48 on both arms, to three digits). Total wall 6.76 h vs 6.78 h (+0.3 %,
> inside the contention noise this measurement admits).

**Read `core`, not `wall`.** Under 16-way concurrency wall is contention
dominated (doc pr/11 §90, doc 76 §1.1); `core_s` comes from the job's own
`Timer: Total` line and `maxrss_kb` from `RUSAGE_CHILDREN`, which is per process
and concurrency-insensitive.

**Sizing rules that fall out of this** (the numbers a future production planner
wants):

- **~1.5 GiB per concurrent job** is the ceiling to plan against, on every
  sample. It does not grow with sample size and it did not grow across the
  campaign.
- The distribution is strongly **bimodal by sample**: the νμ beam samples median
  1.5 core-sec / 0.45 GiB (most events have no neutrino candidate and exit
  early); the νe and NCπ⁰ selections median 12–16 core-sec / 1.2 GiB. A mixed
  production is dominated by the few percent of events that reconstruct.
- Tail: max 159 s wall / 1.48 GiB in 3067 events. `PR_TIMEOUT=3600` has ~23×
  headroom.

### 4.1 Where the time and the memory go (`scripts/pr142_perf.py`)

Stage-resolved from the `MABC timing:` and `MEM: … res=` lines every PR log has
always carried — read here at population scale for the first time.

- On the **reconstructing** samples `TaggerCheckNeutrino` is the whole cost:
  **7.9 s of a 13 s event** (ncpi0), with `UbooneNueBDTScorer` second at 1.9 s
  and `CreateSteinerGraph` third at 0.47 s.
- On **mcp1k** (mostly non-reconstructing) no step reaches 130 ms: `done`
  121 ms, `loaded live` 78 ms, `CreateSteinerGraph` 72 ms,
  `SbndPrMagnifyTrackingVisitor` 54 ms.
- Peak resident set is reached at the **end** of the chain and is dominated by
  the accumulated point clouds, not by any single step: the high-water mark sits
  at `done` / `PrDisplayDump` (1.03–1.04 GiB on ncpi0), with
  `UbooneNueBDTScorer` adding ~140 MB when it loads its forest.
- Both arms agree step-for-step to ≲ 3 %, which is the same statement as the
  totals above, made per stage.

**Reference point for how production actually runs.** `prod0830` in group mode
(`PR_GROUP_SIZE=16`, 193 groups) cost **8.1 core-hours** for the same 3067
events at ≤ 1.5 GiB per process. Per-event mode is the price of per-event
numbers (doc 76 §10.7 prices it at ~+41 % summed process wall); both are quoted
so neither is mistaken for the other.

> **Do not** derive per-event timing from log timestamp spans in a group-mode
> arm: they overlap, and on prod0830 they imply 399-way concurrency on a
> 64-core box. That is an artifact, not a measurement.

---

## 5. Did the measured gains survive? (labeled subsets, on these arms)

The 239-event hand-scan manifest is a subset of the 3067, so every campaign
metric runs directly on the two new arms. **These are subset metrics and are not
extended to the full sample** (§1.2).

### 5.1 The π⁰ census — 27 → 36 of 66 exact

`scripts/pr141_pi0_census2.py`, precedence chain `pi0mass-0904-owner,
pi0scan-0829-agent`, `--fudge` matched to each arm's own scale (0.80 for
`empre0901`, whose `kine_shower_fudge_factor` is suppressed to the C++ default;
0.86 for `prod0901`).

| | `empre0901` | `prod0901` |
|---|---:|---:|
| **exact** | **27 (40.9 %)** | **36 (54.5 %)** |
| partial | 15 (22.7 %) | 15 (22.7 %) |
| none | 2 | 3 |
| no-group | 22 (33.3 %) | **12 (18.2 %)** |
| rescan coverage | 28 of 109 | 35 of 109 |

**+9 exact π⁰, and 10 hand-labeled π⁰ move out of "no group found at all".**
`prod0901` reproduces doc pr/141 §16's **36 of 66** exactly — an independent
confirmation that the new full-sample production is the operating point pr/141
closed on. The **27** is new: the campaign's own baseline series started at 31
(pr/132, already at fudge 0.84 with two flips in), so the true pre-campaign
number had never been measured. The campaign's π⁰ gain is **+9, not +5**.

### 5.2 Charge attribution on the 90 hand-marked showers

`em_display/em117_score.py --cross-run` over both hand-mark sets, against the
probe pair (§5.4).

| | `empre0901` | `prod0901` |
|---|---:|---:|
| median `q_f1` (90 showers) | 0.887 | **0.922** |
| `q_miss` / target | 15.9 % | 16.7 % |
| **`q_extra` / target** | **14.9 %** | **6.7 %** |

**Over-clustered charge is cut by more than half (14.9 % → 6.7 %) for +0.8 pt of
`q_miss`**, and the median shower improves in *every* bucket:
under-clustered 0.887 → 0.908, both 0.740 → **0.994**, over-clustered
0.740 → **0.894** (98-set); 0.879 → **0.921** (141-set, where `q_extra` alone
falls 5.77e7 → 2.15e7, −63 %).

`prod0901` reproduces doc pr/139 §3bis's published production point —
**`q_miss` 16.7 %, `q_extra` 6.7 %** — to the decimal. Second independent
confirmation of the operating point.

Caveat carried forward from doc pr/138 §3.3: a raw `q_miss` rise measured
cross-run against scan-time labels is partly a measurement artefact (93–94 % of
one was), because the labels were taken on an earlier arm. The `q_extra` fall is
not subject to that caveat and is the larger effect here.

### 5.3 Sentinels — 6 FAIL, and **all six are pre-existing**

`scripts/pr127_sentinels.py` on `work-*-prod0901`: **27 PASS / 6 FAIL / 0 SKIP.**

The same suite on the campaign's own reference arm `work-pr141r1-off-*`:
**16 PASS / 6 FAIL / 11 SKIP** — and the FAIL sets are **character-for-character
identical**: `137238, 173819, 292643, 393505, 406125, 47212`.

> **This round introduces zero new sentinel failures.** The six are the standing
> state of production and each was already attributed: `393505` = the 0.86 scale
> (Enu 559.9 vs a window `[560, 572]` that needs rebasing by 0.1 MeV),
> `47212` = the K24 re-pairing the owner approved, `137238` = pre-existing since
> the pr/93 r4 Q/L-era drift, plus `173819` (pr/125 guard, e⁻ 283 MeV against a
> `< 200` assertion), `292643` (pr/130 B, the π⁰ node is gone) and `406125`
> (pr/124 A: the `pr124 pass4_prune2:` log line **is absent from the arm
> entirely**, i.e. the shipped prune no longer fires on its own event).

Two of those deserve the owner's eye — `406125` is exactly the failure mode
pr/127 built this suite to catch (a shipped fix that silently stopped firing),
and `292643`/`173819` assert on PF content that has moved. **Reported, not
tuned** (CLAUDE.md §5.7). None of them is caused by anything in this round.

**A byproduct worth keeping:** the full-sample arm evaluates **33 sentinels
where the standard 239-event arm can only reach 22** (11 SKIP). Those 11 all
PASS. The sentinel suite has been running at two-thirds coverage for the whole
campaign because the round arms are the hand-scan manifest; a full-sample
production is a strictly better substrate for it.

### 5.4 The probe pair

`em117_score.py` needs per-shower membership sidecars, which `prep_pr117.py`
parses out of `SHOWER_CONTENT` probe lines — emitted only under
`WCT_SHOWER_CONTENT_DEBUG`, which the two production arms deliberately do **not**
set. So the metric got its own pair on the 239-event manifest with identical
knobs: `work-pr142probe-{empre,prod}-<sample>`, 239 events each, 239/239
sidecars written for both.

---

## 6. Proof B — the "before" arm against the committed pre-campaign product, and the one thing the campaign shipped without a knob

`products/prod0825` is the last full production taken **before** the campaign
(2026-08-25, its arms since retired; the score table is committed). Comparing it
to `empre0901` isolates everything that is *not* the EM/π⁰ configuration.

**303 movers of 3067 (9.9 %)**, and the classes are the ones written down in
advance — except one:

| class | n | attribution |
|---|---:|---|
| `numu_score` | 130 | the BDT score clamp removal (doc 85 §9, unknobbed) + doc 84 |
| `nue_score` | 103 | same |
| `kine_reco_Enu` | 28 | doc 84's MCS / long-muon energies |
| `nue_fill` | 6 | same |
| **`nu_evaluated`** | **70** | **none of the above — see below** |
| **ν vertex > 1 cm** | **0** | — |

The clamp removal is visible exactly where predicted: `prod0825`'s `numu_score`
range is clipped at **±4.30103** and every later arm reaches **6.48**. This is
why §3.3's BDT comparison is run `empre0901` vs `prod0901` (both post-removal)
and never against `prod0825`.

### 6.1 The ν vertex is bit-identical across the whole campaign window

On the **1377 events evaluated by both** `prod0825` and `empre0901`, the
reconstructed neutrino vertex moved **more than 1 cm on 0 events, with a maximum
displacement of 0.000 cm**. The DL/SCN vertex is reproducible across two
different binaries and two config eras. Together with §3.4's three π⁰-chain
movers, that is the whole story of the vertex in this campaign.

### 6.2 The 70 `nu_evaluated` flips are NOT configuration

This is what the audit was for, so it gets its own arm.

1. The same **70 events, identically**, flip between `prod0825`→`prod0830` and
   `prod0825`→`empre0901`. So the cause is common to both later arms and is not
   the EM/π⁰ knobs (which are OFF in `empre0901`).
2. Re-running exactly those 70 events with the **entire pre-campaign cfg tree**
   at `8d93260d` (`PR_CFG_TREE`, doc 84 turned OFF as well —
   `work-pr142ef-precfg-*`, 70/70 rc=0) gives **70 of 70 matching `empre0901`
   and 0 of 70 matching `prod0825`.**

⇒ **The flips survive every configuration change being reverted. They come from
the binary — an unknobbed code change inside the campaign window — not from any
knob.** Direction: **56 events gain a reconstruction, 14 lose one, net +42**
(1433 evaluated vs 1390, +3.0 %).

It is **not** DL-vertex nondeterminism: §6.1 shows the vertex is bit-identical
on every event evaluated by both arms, and three independent current-binary arms
agree on all 70.

The candidates are the campaign's two deliberately-unknobbed changes — the BDT
clamp removal (`59f75bb8`) and the excluded-energy census (`546f52a8` +
`bd99f443`), both doc 85 round 2, both owner-requested — or another unconditional
change among the 53 `clus/` commits in the window. Separating them needs the
pre-campaign **binary** rebuilt into its own prefix, which is outside this
round's scope. **Recorded here as the one effect the campaign shipped that no
round measured**, and it is a net gain, not a regression.

---

## 7. Verdict

**The campaign is an improvement, and nothing is degraded that is not explained.**

What the 3067-event comparison establishes:

- **Nothing broke.** `rc=0` on all 3067 events of both arms; zero unpaired
  events; zero crashes, zero timeouts.
- **Nothing was lost from the selection.** The nusel `event_label` migration
  matrix is diagonal, and `nu_evaluated` flips **0** between the two arms. The
  campaign neither gained nor lost a neutrino candidate.
- **The vertex is untouched** except where it was meant to move: 3 movers > 1 cm
  in 3067, all three the π⁰ NC-vertex chain the owner approved in pr/134 §14;
  and across the *whole* campaign window the vertex is bit-identical (max
  0.000 cm) on every event both arms evaluate.
- **Energy moves the intended way**: −4.8 % / −4.6 % on the two EM-rich samples,
  +0.1 % / +0.3 % on the two νμ samples. That is the EM charge scale, by
  construction.
- **The νe selection gains and buys no background**: `nue_score > 7.0` net **+1**
  on the νeCC sample, net **0** on both νμ samples. The νμ point loses 1–2 per
  sample (0.4–0.7 %), two of the six losses on samples where losing a νμ pass is
  the right direction.
- **It costs nothing to run**: 3.61 core-hours against 3.61, peak RSS identical
  to three digits.
- **The measured gains survive at the operating point**: π⁰ exact **27 → 36 of
  66**, `q_extra` **14.9 % → 6.7 %**, median shower `q_f1` **0.887 → 0.922**.
- **No new sentinel failure.** Six FAIL, character-for-character the same six the
  campaign's own reference arm produces.

Three things the owner should see, none of them a regression:

1. **§6.2 — 70 events (2.3 %) change their `nu_evaluated` verdict from an
   unknobbed code change**, net +42 gained. Every configuration revert leaves it
   in place. Nothing in the campaign measured this; it is a net gain, and naming
   the commit needs the pre-campaign binary rebuilt.
2. **§5.3 — `406125`'s shipped pr/124 prune no longer fires at all** (its log
   line is absent from the arm). Pre-existing, but it is precisely the failure
   mode pr/127's sentinel suite exists to catch, and `292643` / `173819` assert
   on PF content that has moved.
3. **§5.3 — the sentinel suite has been running at 22 of 33** because the round
   arms are the 239-event manifest. On a full-sample arm the other 11 run, and
   all 11 pass. Worth pointing the suite at a production arm from now on.

### 7.1 What this comparison does NOT certify

These four samples carry **no truth label anywhere in this chain** (doc 85 §1) —
"NCπ⁰" and "νeCC" name the *sample selection*, not the interaction. So at
population scale "no degradation" means exactly what §7's first six bullets say:
no failure, no lost candidate, no unexplained mover, distributions moving in the
direction the rounds intended. It does **not** mean 3067 events certified a
physics improvement, and no number here should be quoted as if it did. The
campaign's physics claims — 36/66 exact π⁰, the `q_extra` floor, the median
`q_f1` — are **239-event, hand-labeled subset metrics**, are reported as such in
§5, and their statistical reach is the size of those label sets, not 3067.

The one axis this round could not close is the νμ/νe **efficiency and purity**
against truth. That needs an MC-truth sample, which this chain does not carry.

### 7.2 Recommended next step

**Adopt `prod0901` as the production baseline** (`products/prod0901/`, committed
here) and retire `prod0830` from that role. Then, in order:

1. **Point the sentinel suite at a full-sample production arm** and adjudicate
   `406125` first — a shipped fix that no longer fires is the cheapest real
   defect on the board, and pr/127 exists because that class went unnoticed for
   ten days once already.
2. **Settle §6.2** if the owner wants it named: build `8d93260d` into its own
   prefix and re-run the 70 events. One afternoon, and it closes the last
   unattributed effect of the campaign.
3. The **PID front** doc pr/141 §22 opened (≥ 29 % of μ-typed objects above
   50 MeV are EM showers, ≈ 1045 MeV of `kine_reco_Enu` missing across 239
   events) remains the largest measured defect and is untouched by this round.

---

## 8. Artifacts

| what | where |
|---|---|
| the two arms | `work-<sample>-{empre0901,prod0901}`, 3067 events each |
| score tables | `products/{empre0901,prod0901}/<sample>-scores-<tag>.tsv` (committed) |
| the restore list | `docs/pr/pr142-restore-empre.tla` (39 entries, derived) |
| movers | `docs/pr/pr142-movers.tsv` (312), `pr142-proofB-movers.tsv` (303) |
| population | `docs/pr/pr142-population.tsv` |
| π⁰ census | `docs/pr/pr142-pi0-census-{empre,prod}.tsv` |
| stage perf | `docs/pr/pr142-perf-{mcp1k,nuecc48}.tsv` |
| probe pair | `work-pr142probe-{empre,prod}-<sample>` (239 events each) |
| Proof A pilot | `work-pr142pilot-{precfg,empre,head}-ncpi0` |
| §6.2 attribution arm | `work-pr142ef-precfg-<sample>` (70 events) |
| scripts | `scripts/pr142_{arms,probe_arms,analyze}.sh`, `scripts/pr142_{campaign_ab,perf}.py` |

Binary pinned for every arm: `LD_LIBRARY_PATH=/home/xqian/tmp/pr142-libsnap`,
snapshot of `local/lib` at toolkit `ddce7430`, `libWireCellClus.so`
mtime 2026-08-31 19:55, md5 `430ffa3eeffb3253b5bc40b6a36f8531`, verified newer
than every `clus/` source (M1).
