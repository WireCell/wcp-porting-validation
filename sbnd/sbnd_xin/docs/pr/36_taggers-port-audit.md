# doc pr/36 — the taggers: a prototype↔toolkit fidelity audit

**Why.** Doc pr/27 §0 lists eight PR stages. This audits **step 8, the
taggers** — the last one, and by a wide margin the largest. It is the ninth and
final entry in the series that began with pr/28 (vertex fit + trajectory
dQ/dx) and ran through pr/29 (Steiner graph), pr/30 (proto-vertex + segment),
pr/31 (topology/PID/direction), pr/32 (neutrino vertex ID), pr/33 (EM shower
clustering), pr/34 (particle flow) and pr/35 (energy reconstruction).

**Status. §11 SHIPPED (toolkit `2457320d`): F1/F4/F5/F6/F7 SBND
PRODUCTION ON, F3 plumbed but OFF (owner), F2 measured dead-by-construction.**
§§0–10 remain the audit record; §11 is the implementation round (owner
instruction 2026-08-04: bug fixes + improvements default ON in SBND, validated
on nueCC48).

> **§10 added — the owner filter, 13 → 7**, re-verified at toolkit
> **`407c5ba9`** (§3 was written at `23bd6783`, twelve commits earlier). §10
> keeps the bugs and the port gaps, drops the three findings where the toolkit
> improves on the prototype, resolves two outright, and proposes a fix and an
> exact edit site for each survivor. **Seven findings, five knobs** — the shapes
> differ and §10.1a says how.  **§11 implements the round**; its measurements
> corrected several §10 counts (18 call sites not 22; four F4 sums not three;
> the prototype's singlephoton neutrino_type write and its :222 SCE site are
> commented out) — see §11.2.
>
> Three things in §10 change §3 rather than extend it, and are flagged there:
> **P12's stated mechanism is retracted** and replaced by a verified one with a
> much larger plausible population (§10.3); **P11 is resolved as a
> non-defect, and the "fix" §3 gestures at would be a regression** (§10.10a);
> and **§2.9's parity claim is narrower than it reads** (§10.13). §6's
> unexamined `muon_segs` was examined and is a third live site (§10.5), and P6
> is 22 call sites, not one (§10.6).

**Headline.**

1. **`match_isFC` is not the same quantity on the two sides.** The prototype
   *reads* it from the input `T_eval` ROOT branch — the upstream light-matching
   containment verdict. The toolkit *recomputes* it with
   `Facade::cluster_fc_check`, and unlike the only other caller
   (`TaggerCheckSTM`) it passes no fiducial volume and no tolerance, so it
   lands on the historical `FiducialUtils` fallback that `TaggerCheckSTM`'s own
   comment describes as disagreeing with TGM/FC. `match_isFC` is numu XGBoost
   variable **70** and an nue XGBoost input. **(P1)**
2. **`singlephoton_tagger` drops the space-charge position correction at every
   site.** The prototype applies `func_pos_SCE_correction` to the neutrino
   vertex and to every track/shower start point it records
   (`NeutrinoID_singlephoton_tagger.h:13, :103, :132, :222, :317`); the toolkit
   records raw positions and says so in a comment. Materially bounded: no
   `shw_sp_*` field is a BDT input on **either** side, so the exposure is the
   ntuple and any downstream analysis, not the scores. **(P2)**
3. **Two float accumulations run in pointer-address order.** `broken_muon_id`
   sums `acc_length`/`acc_direct_length` over a `std::set<SegmentPtr>`, and
   `mip_quality` iterates a `std::set<ShowerPtr>`. Both are faithful to the
   prototype's `std::set<ProtoSegment*>`, so this is a *reproduced* hazard, not
   a new one — but `SegmentIndexCmp` exists and is already used elsewhere in
   the same stage (`NeutrinoTaggerSSM.cxx:608`). **(P3, P5)**

Everything else is smaller, and several findings are the toolkit being
**better** than the prototype (P9, P10).

**A note on severity.** This stage is where the pipeline's verdict is formed,
so a divergence here is as live as it gets — but only for fields the scorers
actually read. §2.1 establishes mechanically that the two sides write **the
same 966 `TaggerInfo` fields**, which bounds the whole problem: no BDT input
lacks a writer on one side and has one on the other. That is a statement about
*code*, not about any particular event — §2.1 records what it does and does not
prove. Findings below are about the *values* in those fields, not about which
fields exist.

**A note on provenance.** pr/34 GOTCHA 1 established that `prototype_base/pid`
is **not** pristine upstream (+5833/−989 over 26 files vs the merge-base
`a5fc0b9`). For this stage the exposure is **nil**, and unusually it is nil on
both halves:

* none of `NeutrinoID_cosmic_tagger.h`, `_numu_tagger.h`, `_nue_tagger.h`,
  `_nue_functions.h`, `_nue_bdts.h`, `_numu_bdts.h`, `_ssm_tagger.h`,
  `_singlephoton_tagger.h` appears in `git -C pid diff --stat a5fc0b9..HEAD`
  at all;
* `NeutrinoID.cxx` *is* in that diff (204 lines), but the tagger driver block
  is byte-identical across the two revisions — verified directly, not inferred
  from hunk headers:
  `diff <(sed -n '251,292p' src/NeutrinoID.cxx) <(git show a5fc0b9:src/NeutrinoID.cxx | sed -n '234,275p')`
  is empty. The block is shifted +17 lines and otherwise unchanged.

This does **not** relieve pr/28–pr/33 (see §7.9).

---

## Repro

```bash
# Toolkit read at this commit.  NOTE: a concurrent session held 5 modified
# files in the working tree, so every toolkit source below was read from a
# HEAD snapshot, not from the working tree.
cd /nfs/data/1/xqian/toolkit-dev/toolkit
git rev-parse --short HEAD            # 23bd6783
mkdir -p /home/xqian/tmp/claude-25225/pr36
for f in clus/src/TaggerCheckNeutrino.cxx clus/src/NeutrinoTaggerCosmic.cxx \
         clus/src/NeutrinoTaggerNuMu.cxx clus/src/NeutrinoTaggerSSM.cxx \
         clus/src/NeutrinoTaggerNuE.cxx clus/src/NeutrinoTaggerSinglePhoton.cxx \
         clus/src/NeutrinoKinematics.cxx \
         clus/inc/WireCellClus/TaggerCheckNeutrino.h \
         clus/inc/WireCellClus/NeutrinoTaggerInfo.h ; do
  git show HEAD:$f > /home/xqian/tmp/claude-25225/pr36/$(basename $f)
done

# Prototype read at this commit (submodule, branch 'port').
cd prototype_base/pid && git rev-parse --short HEAD      # 53ca938
git merge-base HEAD origin/master                        # a5fc0b9...

# The two mechanical checks this doc's positive results rest on.
python3 sbnd_xin/scripts/analysis/pr36/cmp_tagger_fields.py   # section 2.1
python3 sbnd_xin/scripts/analysis/pr36/cmp_ssm_exit.py        # section 2.2
```

No event was run. See §9.

---

## §0 Scope

### What was audited

| # | toolkit | lines | prototype counterpart | lines | depth |
|---|---|---|---|---|---|
| 0 | `TaggerCheckNeutrino::visit()` tagger block `:735-:860` | 125 | `NeutrinoID.cxx:251-292` | 42 | **exhaustive** |
| 1 | `PatternAlgorithms::cosmic_tagger` `NeutrinoTaggerCosmic.cxx:471` | 882 | `NeutrinoID_cosmic_tagger.h:1` | 865 | entry + flags 1-5 |
| 2 | `PatternAlgorithms::numu_tagger` `NeutrinoTaggerNuMu.cxx:161` | 224 | `NeutrinoID_numu_tagger.h:1` | 263 | **exhaustive** |
| 3 | `count_daughters` ×2 `NeutrinoTaggerNuMu.cxx:82, :118` | 79 | `NeutrinoID_numu_tagger.h:264, :294` | 57 | signature + call sites |
| 4 | `PatternAlgorithms::ssm_tagger` `NeutrinoTaggerSSM.cxx:573` | 976 | `NeutrinoID_ssm_tagger.h:1` | 2291 | entry + exit path |
| 5 | `exit_ssm` lambda `NeutrinoTaggerSSM.cxx:712-887` | 175 | `exit_ssm_tagger()` `:2706-3210` | 504 | **exhaustive (script)** |
| 6 | `PatternAlgorithms::nue_tagger` `NeutrinoTaggerNuE.cxx:4236` | 178 | `NeutrinoID_nue_tagger.h:2` | 264 | **exhaustive** |
| 7 | the ~20 nue sub-taggers | ~4100 | `_nue_tagger.h` + `_nue_functions.h` | ~4400 | **dispatch + 3 deep-dives** |
| 8 | `PatternAlgorithms::singlephoton_tagger` `NeutrinoTaggerSinglePhoton.cxx:2184` | 349 | `NeutrinoID_singlephoton_tagger.h:2` | 543 | entry + SCE sweep |
| 9 | the 10 `*_sp` sub-taggers | ~2050 | `_singlephoton_tagger.h` | ~2200 | field coverage only |
| 10 | `UbooneNueBDTScorer` / `UbooneNumuBDTScorer` | 3200 | `_nue_bdts.h` + `_numu_bdts.h` | 1829 | **dispatch + input list** |
| 11 | `Facade::cluster_fc_check` call `TaggerCheckNeutrino.cxx:852` | 3 | ctor arg `NeutrinoID.cxx:62` | — | **exhaustive** |
| 12 | `TaggerInfo` `NeutrinoTaggerInfo.h:68-1400` | 1024 fields | `init_tagger_info` `NeutrinoID.cxx:2224` | 1193 | **exhaustive (pr/35)** |

Totals: ~11,700 toolkit lines against ~14,500 prototype lines. That is roughly
three times the largest previous round, and **exhaustive line-by-line coverage
was not achievable in one pass.** The audit is therefore structured as:

* **mechanical, complete** — the `TaggerInfo` field-coverage map (§2.1), the
  `exit_ssm` parity check (§2.2), the determinism sweep (§6), and the
  provenance check;
* **read in full** — every tagger's entry point and dispatch chain, plus the
  driver;
* **sampled** — three nue sub-taggers read line-by-line
  (`broken_muon_id`, `mip_quality`, `compare_muon_energy`), chosen because the
  coverage map and the determinism sweep both pointed at them.

### What was *not* audited — state this plainly

* **Per-sub-tagger cut-value fidelity for the ~30 sub-taggers not deep-dived.**
  Their *field writes* are proven complete (§2.1) and their *dispatch order* is
  proven identical (§2.3), but the numeric thresholds inside them were not
  compared line by line. A wrong constant in, say, `track_overclustering` would
  not be caught by anything in this doc.
* **The BDT weight files themselves** — the XML under `uboone/weights/`. Not
  read, not compared, not validated on SBND (§4).
* The ~30 sub-BDT reader variable *orderings* inside the scorers beyond the
  spot checks in §2.4.
* `TaggerCheckTGM` / `TaggerCheckSTM` / `TaggerCheckFC` / `QLMatching`'s LM —
  pr/27 §10 calls these "upstream verdicts, not computed here". Out of scope,
  except where `match_isFC` reaches across (P1).
* `ClusteringTaggerFlagTransfer.cxx`.
* Anything requiring an event to be run (§9).

---

## §1 Trust tiers

Carried from pr/28 §3b through pr/35, unchanged.

**Tier A — checked directly in this round.** Both sides read at the commits in
the Repro block; every file:line anchor below was verified by printing that
line. Mechanical results (§2.1, §2.2, §6) are Tier A by construction.

**Tier B — inherited.** Statements that rest on pr/27's stage map, on the
in-tree `clus/docs/tagger/*_review.md` set, or on earlier rounds' conclusions.
pr/33 found an in-tree review doc whose *proposed fix created* a divergence, so
Tier B claims are flagged where they matter and none of the findings below rest
on one alone.

---

## §2 What matches

The positives are load-bearing this round: they are what makes a stage this
large auditable at all, and they are established mechanically rather than
asserted.

### 2.1 The two sides write the same 966 `TaggerInfo` fields

This is the single most important result in the doc.

`sbnd_xin/scripts/analysis/pr36/cmp_tagger_fields.py` parses both sides for
*writes* (`X = `, `X += `, `X++`, through `.` or `->`, comments stripped) and
intersects with the `TaggerInfo` member list and with the scorers' reference
sets:

```
TaggerInfo struct members            : 1024
prototype tagger-stage writes        : 966
toolkit  tagger-stage writes         : 966

  proto NeutrinoID_cosmic_tagger.h                 67 fields
  proto NeutrinoID_numu_tagger.h                    9 fields
  proto NeutrinoID_nue_tagger.h                   262 fields
  proto NeutrinoID_nue_functions.h                 64 fields
  proto NeutrinoID_ssm_tagger.h                   362 fields
  proto NeutrinoID_singlephoton_tagger.h          202 fields

  tk    NeutrinoTaggerCosmic.cxx                   94 fields
  tk    NeutrinoTaggerNuMu.cxx                      9 fields
  tk    NeutrinoTaggerNuE.cxx                     299 fields
  tk    NeutrinoTaggerSSM.cxx                     362 fields
  tk    NeutrinoTaggerSinglePhoton.cxx            202 fields
```

966 = 966, and the per-file asymmetry between `Cosmic` (94 vs 67) and `NuE`
(299 vs 326) is entirely accounted for by **file placement**: the toolkit put
`bad_reconstruction` — the nue "br1" sub-tagger — in `NeutrinoTaggerCosmic.cxx`
(`:97`), because `cosmic_tagger` also calls it (`:610`, `:613`, `:849`), which
mirrors the prototype where `cosmic_tagger.h:101` calls the function defined in
`nue_tagger.h:3450`. The 27 `br1_*` fields move between columns; the union is
unchanged.

**Do not read this as "no field is missing" without the caveat that the
regex missed one idiom on the first pass.** The initial run reported 47
prototype-only fields including all 26 `br1_*`; that was a false alarm from a
write pattern that only matched `ti.` and not `ti->`. `bad_reconstruction`
takes `TaggerInfo* ti` and writes `ti->br1_1_energy`
(`NeutrinoTaggerCosmic.cxx:144`). The script now covers both; the number to
trust is the 966 above. **Any future use of this script must sanity-check a
handful of hits by hand before publishing counts.**

**What this does and does not prove.** It proves every field has a *writer*,
on both sides, in the same place. It does **not** prove the writer is *reached*
on any given event, and two coarse gates make that a real distinction:
`nue_tagger` returns `false` at `NeutrinoTaggerNuE.cxx:4308` when no good
shower is found, skipping all 20 sub-taggers and ~300 fields; and
`TaggerCheckNeutrino.cxx:751` gates the entire tagger block on
`if (final_main_vertex)`, so a vertexless event leaves all 966 at their
defaults. Both gates are symmetric — the prototype's nesting
(`NeutrinoID_nue_tagger.h:71`) and its `if (flag_tagger)` block do the same —
so this is not a divergence. It does mean **per-event reachability was not
tested**, and no claim here rests on it.

The only fields the prototype writes and the toolkit does not are 20
`*_score` values. They come from the **BDT-file** row of the counts above
(`+50` prototype writes against `+25` toolkit), not from the 966 — the two
numbers are not in tension. They are TMVA-path-only; see §5.1.

Five fields go the other way (toolkit writes, prototype does not):
`nue_score`, `numu_score` (both written by the scorer stage rather than by the
tagger, symmetric with the prototype's `NeutrinoID.cxx:277-286`),
`br3_7_main_length` / `shw_sp_br3_7_main_length` / `numu_cc_3_track_length` —
the three renames pr/35 GOTCHA 3 already identified.

Seventeen fields are written by neither side: `cosmict_score`, `match_isFC`,
`photon_flag`, and the 14 `ssm_kine_*`. The first three are written by the
driver (`TaggerCheckNeutrino.cxx:852`, `:835`) or by a downstream consumer, and
are covered by P1/P4. The `ssm_kine_*` block is dead on **both** sides.

### 2.2 `exit_ssm` is exactly equivalent despite being 1/3 the length

`NeutrinoTaggerSSM.cxx:738` carries the comment
`// (long repetitive block mirroring exit_ssm_tagger lines 2786-3081)`, and the
lambda is 175 lines against the prototype function's 504. That asymmetry is
worth checking rather than trusting, so
`sbnd_xin/scripts/analysis/pr36/cmp_ssm_exit.py` does:

```
prototype exit_ssm_tagger assigns : 356 ssm_ fields
toolkit  exit_ssm lambda  assigns : 146 ssm_ fields
missing from toolkit lambda       : 212
of those, prototype exit value != toolkit struct DEFAULT : 0
distinct proto exit values on the missing set : ['-999']
distinct toolkit defaults on the missing set  : ['-999']
```

All 212 are the eight `ParticleBlock` groups (`ssm_prim_track1/2`,
`ssm_prim_shw1/2`, `ssm_daught_track1/2`, `ssm_daught_shw1/2` × ~27 fields),
every one of which already carries `-999` as its default-member-initializer and
is therefore set by `init_tagger_info` (`ti = TaggerInfo{}`).

The mirror question — why the lambda writes 146 fields at all, if
`init_tagger_info` already ran — has the expected answer: they are the ones
whose exit value is *not* the struct default, so `init_tagger_info` alone would
not produce them. Two of the 146 have no counterpart in the prototype's exit at
all — `ssm_con_nu_angle_z` and `ssm_flag_st_kdar` — which is the only asymmetry
the script reports in that direction.

The equivalence has a precondition — the exit must not be reachable *after*
those fields were written — and it holds: the toolkit's two exit sites are
`:890` (`Nsm == 0`) and `:911` (`!ssm_sg`), and the first `ParticleBlock` write
is `:1355`. Verified, not assumed.

### 2.3 The nue veto chain runs in exactly the prototype's order

Read side by side, `NeutrinoID_nue_tagger.h:2-266` against
`NeutrinoTaggerNuE.cxx:4236-4414`. All 20 sub-taggers, in order:

`gap_identification` → `mip_quality` → `mip_identification` →
`pi0_identification` → `single_shower_pio_tagger` → `multiple_showers` →
`other_showers` → `shower_to_wall` → `single_shower` → `stem_length` →
`low_energy_michel` → `broken_muon_id` → `compare_muon_energy` →
`angular_cut` → `stem_direction` → `vertex_inside_shower` →
{`bad_reconstruction`, `bad_reconstruction_1`, `low_energy_overlapping`} →
{`bad_reconstruction_2`, `bad_reconstruction_3`, `high_energy_overlapping`} →
`track_overclustering`.

The two three-way groups are grouped the same way on both sides (all three run,
then a single `||` decides), which matters because each has a side effect on
`TaggerInfo` — short-circuiting would change the ntuple. Neither side
short-circuits. The `flag_nue = false` assignments are one-for-one, and
`mip_identification`'s `flag_strong_check` argument carries the same
`flag_single_shower && max_energy < 400 MeV` condition
(prototype `:106-110`, toolkit `:4344`).

The candidate-selection preamble matches too: the 80 MeV / (60 MeV ∧ `n_segs≥3`
∧ `n_3seg>0`) admission test, the `max_energy` argmax, the `good_showers`
membership check, and `num_valid_tracks`'s
`length > 8 cm || (!dir_weak && length > 5 cm)` rule.

### 2.4 The BDT dispatch is faithful for the path production uses

The prototype gates on `flag_bdt` (`NeutrinoID.cxx:277-287`):
`>0` → `cal_numu_bdts_xgboost()`, `==1` → `cal_bdts_xgboost()`,
`==2` → `cal_bdts()` (TMVA). The toolkit has no gate — it runs
`numu_bdt_scorer` and `nue_bdt_scorer` as unconditional pipeline stages, and
ports only the XGBoost variant of each, saying so in-file
(`UbooneNueBDTScorer.cxx:37`, `UbooneNumuBDTScorer.cxx:11`).

That is correct if and only if production runs `flag_bdt == 1`. Established the
same four-leg way as pr/35's P2:

1. `int flag_bdt = 1; // default 1 xgboost, 2 for TMVA` —
   `wire-cell-prod-nue.cxx:34` and `wire-cell-prod-nue-port.cxx:34`.
2. The only override is `case 'b':` (`-port:60-61`).
3. The production run scripts pass `-d0 -o1 -g...` and **no `-b`**
   (established in pr/35 §2 for the same scripts).
4. The scripts invoke the **`-port`** variant, which carries the identical
   default and switch.

So `flag_bdt == 1`, the TMVA combination is dead in production, and its absence
from the toolkit is a correct port. §5.1 records what that costs in ntuple
terms.

### 2.5 `numu_tagger` is a line-faithful port

The smallest tagger and the only one read end to end on both sides. Flag-1's
composite muon test — `|pdg|==13 ∧ length>5cm ∧ medium_dQ_dx < cut·43e3 ∧
(length>40cm ∨ direct_length>0.925·length) ∧ ¬(n_daughter_tracks>1 ∨
n_daughter_all−n_daughter_tracks>2)` — is identical term for term
(prototype `:46`, toolkit `:203-207`), including the `0.8866+0.9533·
(18cm/length)^0.4234` dQ/dx cut, which the toolkit factors into
`muon_dqdx_cut()`. All eight `numu_cc_1_*` array pushes happen on every
iteration on both sides, including the non-firing one — so the arrays stay
index-aligned. Flag-2's 18 cm long-muon-shower threshold and the
`max_muon = nullptr` reset when a shower wins both match.

### 2.6 `init_tagger_info` — unchanged from pr/35

`ti = TaggerInfo{}` (`NeutrinoKinematics.cxx:18-21`) reproduces the
prototype's 1193-line assignment list exactly: 1023 assignments, 1024 struct
members, **0 value mismatches**. Re-verified this round with the committed
`pr35/cmp_tagger_defaults.py`. The four name asymmetries are two renames plus
toolkit-only `match_isFC`.

### 2.7 The apa/face derivation is a deliberate, documented improvement

`TaggerCheckNeutrino.cxx:788-804` derives `nue_apa`/`nue_face` from the main
vertex through `m_dv->contained_by()` instead of assuming uBooNE's single
drift volume `(0,0)`, guarding `apa() >= 0` so an uncontained vertex keeps the
legacy `(0,0)` rather than letting `wire_angles().at(-1)` throw. The same
derivation and the same guard appear at `NeutrinoTaggerSinglePhoton.cxx:2202-2216`
and `NeutrinoTaggerNuE.cxx:2762`. This is required for SBND and correctly has
no prototype counterpart.

### 2.8 The cosmic-verdict drop is symmetric

`cosmic_tagger`'s return value is discarded by the toolkit. It is discarded by
the prototype too — `NeutrinoID.cxx:259` is
`// if (flag_cosmic) tagger_info.cosmic_flag = false;`, commented out. So
`cosmic_flag` is permanently `1` on both sides, and it is a numu XGBoost input
on both sides. Faithful. (Contrast `photon_flag`, P14.)

### 2.9 The vertex-outside-FV test keeps the tightened boundary

`cosmic_tagger`'s `flag_cosmic_1` is easy to get wrong: the prototype calls
`fid->inside_fiducial_volume(test_p, offset_x, &stm_tol_vec)` with all
tolerances `-1.5 cm` — "Xiangpan's boundary" — and leaves the untightened call
commented out one line below. The toolkit keeps the tightened version
(`NeutrinoTaggerCosmic.cxx:536-537`), including the
`|fit_pt − wcpt| > 5 cm ⇒ use wcpt` fallback for the test point. See P11 for
the one detail that differs.

> **Narrower than it reads — see §10.13.** The toolkit keeps the *tightening*;
> it does not keep the *boundary switch*. In the prototype, passing a tolerance
> vector moves the test onto the `boundary_SCB_*` polygon family, not just an
> inset of the same surface.

### 2.10 `neutrino_type`'s only live read is genuinely a print

`NeutrinoTaggerNuE.cxx:45` and `:360` claim the prototype's `neutrino_type`
bitmask "was used only in a debug print". That claim is checkable and it
survives: the only non-assignment read anywhere in the prototype is
`NeutrinoID_nue_functions.h:671`, `bool flag_numuCC = (neutrino_type >> 2) & 1U;`
inside `compare_muon_energy` — and `flag_numuCC` appears exactly once more, in
the `std::cout` at `:679`. It is in no condition. Dropping the bitmask does not
change any tagger's logic. (It does change an output branch — P4.)

---

## §3 Divergences

Thirteen, ordered by the severity assigned in §8.

### P1 — `match_isFC` is read from upstream in the prototype and recomputed here, on a fiducial volume that is known to disagree

**Severity: high. Class: port defect (definition change on a live BDT input).**

Prototype. `match_isFC` is a **constructor argument**
(`NeutrinoID.cxx:62`, `, match_isFC(match_isFC)`), and the app fills it by
reading the input file:

```cpp
// wire-cell-prod-nue-port.cxx:236-250
bool match_isFC = false;
T_eval->SetBranchAddress("match_isFC", &match_isFC);
...
    match_isFC = false;                    // reset per entry
```

`T_eval` is the *upstream* Wire-Cell matching tree. So in the prototype
`match_isFC` is the light-matching stage's containment verdict, carried
forward unmodified.

Toolkit. `TaggerCheckNeutrino.cxx:851-853` recomputes it:

```cpp
if (main_cluster) {
    auto fc_result = Facade::cluster_fc_check(*main_cluster, m_dv);
    tagger_info.match_isFC = fc_result.is_fc ? 1.0f : 0.0f;
}
```

Two things are different, and they compound.

**(a) It is a different quantity.** A recomputed cluster-boundary check is not
the upstream matching verdict. Whether they agree is an empirical question that
nothing in the tree answers.

**(b) It is computed on a fiducial volume the tree already flags as wrong.**
`cluster_fc_check`'s full signature is

```cpp
// Clustering_Util.cxx:75
FCCheckResult cluster_fc_check(Cluster& cluster, IDetectorVolumes::pointer dv,
                               IFiducial::pointer fiducial,
                               const std::vector<double>& fv_tolerance)
```

and the only other caller passes both:

```cpp
// TaggerCheckSTM.cxx:2938
auto fc_result = Facade::cluster_fc_check(cluster, m_dv,
                                          m_use_fiducial ? m_fiducial : nullptr,
                                          m_fv_tolerance);
```

whose knob is documented at `TaggerCheckSTM.cxx:692-697`:

> Containment fiducial volume for the `cluster_fc_check` gate. Unset by
> default ⇒ the historical FiducialUtils / sensitive-volume-union fallback…
> **See `configure()` for why that fallback disagrees with TGM/FC.**

`TaggerCheckNeutrino` passes neither argument, so it is permanently on the
fallback that comment warns about — the same class of problem
`project_stm_containment_fv_inconsistency` recorded for STM and that was fixed
there by adding the knob. Here there is no knob.

**Why it is live.** `match_isFC` is numu XGBoost variable **70**
(`UbooneNumuBDTScorer.cxx:235`, `:508`) and an nue XGBoost input
(`UbooneNueBDTScorer.cxx:331`, `:1659`). It is a binary feature, so a
disagreement is not a small perturbation — it moves the input across its full
range.

**Two readings (M15).** (i) The toolkit deliberately made the stage
self-contained rather than depending on an upstream ROOT branch, and the FV
choice is an oversight to be closed by mirroring `TaggerCheckSTM`'s knob.
(ii) The recomputation is itself the divergence and `match_isFC` should be
threaded from the matching stage. **I have no basis to choose, and this is
exactly the case §5 rule 4 says to surface rather than pick.** The cheap first
step is (i)'s diagnostic half: run both FV definitions on one event and see
whether `match_isFC` even differs.

### P2 — `singlephoton_tagger` records raw positions where the prototype applies the space-charge correction

**Severity: high (fidelity), bounded (not a BDT input). Class: port gap.**

The prototype corrects every position it records:

```
NeutrinoID_singlephoton_tagger.h:13   Point corr_nu_vtx  = mp.func_pos_SCE_correction(nu_vtx);
                             :103,132 Point corr_trk_vtx = mp.func_pos_SCE_correction(trk_vtx);
                             :222,317 Point corr_shw_vtx = mp.func_pos_SCE_correction(shw_vtx);
```

The toolkit records them raw and says so:

```cpp
// NeutrinoTaggerSinglePhoton.cxx:2243
// Positions of proton/MIP track start points (raw, no SCE correction).
std::vector<float> trk_x, trk_y, trk_z;
```

The `SpContext` is constructed with `nullptr` in its geometry-helper slot
(`:2219-2222`), so there is no correction available to apply even if a call site
wanted one.

This propagates further than the coordinates themselves: `max_shw_dis`
(prototype `:324`, `:327-328`) is a **distance between two corrected points**,
so an uncorrected pair changes a scalar feature, not just three coordinates.

**Bound on the damage.** `shw_sp_*` is not a BDT input on either side —
`grep -c 'shw_sp_'` returns **0** in both `NeutrinoID_nue_bdts.h` and
`UbooneNueBDTScorer.cxx`. So this stage's output is the tagger ntuple and any
downstream single-photon analysis, not `nue_score` / `numu_score`. That makes
P2 the same shape as pr/34's findings — real, and display-side.

**Related but distinct.** pr/35 GOTCHA 10 recorded that
`clus_geom_helper` defaults to `""` and SBND never sets it, so
`kine_nu_{x,y,z}_corr` are raw despite the `_corr` name. That one is a *config*
gap with a wired-up path; this one is *unconditional* — there is no config that
would turn it on.

### P3 — `broken_muon_id` accumulates float lengths in pointer-address order

**Severity: high. Class: prototype determinism hazard reproduced (M4).**

```cpp
// NeutrinoTaggerNuE.cxx:1394, :1518-1522
std::set<SegmentPtr> muon_segments;
...
for (SegmentPtr mseg : muon_segments) {
    acc_length        += segment_track_length(mseg);
    acc_direct_length += segment_track_direct_length(mseg);
    tmp_clusters.insert(mseg->cluster());
}
```

`SegmentPtr` is a smart pointer, so `std::set<SegmentPtr>` orders by **address**.
Two floating-point sums accumulate in that order, and both feed hard
thresholds a few lines later (`:1562-1574`):
`acc_length > 0.65 · total_length`, `acc_direct_length > 0.94 · acc_length`,
`acc_length > connected_length · 0.9`, `acc_length > 0.8 · total_length`. The
result is `flag_bad`, which vetoes the nue candidate outright. `acc_length`
also reaches the BDT through `Ep` (`brm_Ep`, via the muon range function).

This is **faithful**: the prototype does the same thing with
`std::set<WCPPID::ProtoSegment*>` (`NeutrinoID_nue_tagger.h:1036`, iterated at
`:1139`). So the class is "reproduced", not "introduced". Two reasons it is
still worth the owner's attention:

* the toolkit already has the fix and uses it in this very stage —
  `NeutrinoTaggerSSM.cxx:608` declares
  `std::map<SegmentPtr, std::tuple<bool,bool,double>, SegmentIndexCmp>`;
* the rest of the stage is visibly determinism-conscious. Three files carry
  comments of the form *"ordered_edges (not `boost::edges`): FP `+=` is
  order-sensitive"* (`NeutrinoTaggerNuE.cxx:631`,
  `NeutrinoTaggerSinglePhoton.cxx:2051`, `NeutrinoTaggerSSM.cxx:1195`). The
  `std::set<SegmentPtr>` case was simply not swept with the same eye.

**Reconciling with the two in-tree determinism nulls** — say this explicitly or
the owner will raise it, as in pr/35 P3. doc 60 §7 found the SBND PR chain
deterministic, but its remeasurement scope stops at
`switch_scope → steiner → fiducialutils → TGM → STM → FC`, which is the cosmic
side and upstream of the nue chain. And `c05bc5f7` swept output-vector
*ordering* in the tagger files while **explicitly ruling out float
accumulation**. Neither reaches `NeutrinoTaggerNuE.cxx:1518`.

### P4 — the `neutrino_type` bitmask is dropped, and it is an output branch

**Severity: medium. Class: port gap (output only).**

The prototype maintains a bitmask (`NeutrinoID.cxx:58`, `neutrino_type(0)`) with
bits set by three taggers:

| bit | set by | line |
|---|---|---|
| 1 | `cosmic_tagger` | `NeutrinoID_cosmic_tagger.h:861` |
| 2 | `numu_tagger`, when `flag_numu_cc` | `NeutrinoID_numu_tagger.h:252` |
| 3 | `numu_tagger`, when not | `:254` |
| 5 | `nue_tagger`, when `flag_nue` | `NeutrinoID_nue_tagger.h:259` |
| 5 | `singlephoton_tagger` | `NeutrinoID_singlephoton_tagger.h:538` |

The toolkit tracks none of them and says so
(`NeutrinoTaggerNuMu.cxx:25`, `:159`, `:376`; `NeutrinoTaggerNuE.cxx:45`).

§2.10 establishes the toolkit's stated justification is correct as far as
*logic* goes. What the comment does not cover is that the prototype app
**persists** it:

```cpp
// wire-cell-prod-nue-port.cxx:1485-1505
int neutrino_type = 0;
T_match1->Branch("neutrino_type", &neutrino_type, "neutrino_type/I");
...
    neutrino_type = it->second->get_neutrino_type();
```

So the prototype's output carries a per-candidate classification summary the
toolkit's does not. Whether anything downstream reads that branch is unknown to
me (§7.5). Worth noting that bit 5 is set by *both* `nue_tagger` and
`singlephoton_tagger`, so it is not simply `nue_score > cut`.

### P5 — `mip_quality` iterates a `std::set<ShowerPtr>`

**Severity: medium. Class: prototype determinism hazard reproduced (M4).**

```cpp
// NeutrinoTaggerNuE.cxx:1685-1686, :1730
std::set<ShowerPtr> connected_showers;
std::set<ShowerPtr> tmp_pi0_showers;
...
for (ShowerPtr shower1 : connected_showers) { ... }
```

Same address-ordering mechanism as P3. The loop body (`:1730-1748`) is
first-wins / boolean rather than accumulating, so the exposure is narrower —
it decides `flag_inside_pi0` and `flag_split`, which set
`mip_quality_flag_inside_pi0` (a BDT input) and gate the veto. Order matters
only when two showers would give different answers; that this is possible is
clear, that it happens is not measured.

`tmp_pi0_showers` is used only through `.count()` and `.size()` and is safe.
`good_showers` (`:4264`) is `.count()`-only and safe. `cluster_acc_length`
(`:634`) is a `std::map<Facade::Cluster*, double>` but is never iterated —
lookup only, and its per-key `+=` runs over `ordered_edges` — so it is safe
too. Enumerated so a future reader does not re-chase them.

### P6 — the stem endpoint is chosen by a different rule

**Severity: medium. Class: port defect.**

Prototype (`NeutrinoID_nue_tagger.h:71-75`) picks the end of the stem segment
by **wcpt index identity** with the main vertex:

```cpp
if (main_vertex->get_wcpt().index == sg->get_wcpt_vec().front().index){
  test_p = sg->get_point_vec().front();
}else{
  test_p = sg->get_point_vec().back();
}
```

Toolkit (`NeutrinoTaggerNuE.cxx:4316`) picks it by **geometric proximity to
the fitted vertex point**:

```cpp
Point test_p = seg_endpoint_near(sg, vtx_fit_pt(main_vertex));
// :85-91  returns front if |ref−front| <= |ref−back|, else back
```

These agree whenever the segment's vertex-side endpoint is also its nearest
endpoint to the fitted vertex — normally true, and not guaranteed. They can
differ when the fit has pulled the vertex away from the wcpt (the same
`|fit − wcpt| > 5 cm` situation `cosmic_tagger` explicitly guards against,
§2.9), or on a short segment whose two ends are nearly equidistant. The index
comparison is exact; the distance comparison has a tie at `<=`.

`test_p` feeds `dir = shower_cal_dir_3vector(max_shower, test_p, 15 cm)`,
hence `angle_beam`, hence `angular_cut` and `ti.cme_angle_beam`. Picking the
wrong end reverses the direction — a ~180° error, not a small one.

The same helper is used inside `compare_muon_energy` (`:369`) where the
prototype (`NeutrinoID_nue_functions.h`) uses its own endpoint rule; that call
site was not compared in detail.

### P7 — π is `3.1415926` in the prototype and `M_PI` in the toolkit

**Severity: low. Class: cosmetic (numerically real, physically negligible).**

The prototype writes degree conversions as `x/3.1415926*180.` throughout
(`nue_tagger.h:192`, `nue_functions.h:679`, `:686`, and ~40 more sites); the
toolkit writes `x / M_PI * 180.0`. The literal is short by
1.8×10⁻⁸ relative, so the prototype's degree values are **larger** by that
factor: at 135° the difference is 2.4×10⁻⁶ degrees.

Recorded because `angular_cut`'s thresholds are hard (135°, 90°) and because a
byte-comparison of the two ntuples will show it on every angle field. It cannot
plausibly flip a cut except on an exact-tie event. Not a defect; do not "fix"
it in the direction of the prototype.

### P8 — `tmp_clusters` counts pointers where the prototype counts ids

**Severity: low. Class: port defect (latent).**

`broken_muon_id`'s multi-cluster requirement is `tmp_ids.size() > 1` in the
prototype (`NeutrinoID_nue_tagger.h:1138` declared, tested at `:1183`, a `std::set<int>` of cluster ids)
and `tmp_clusters.size() > 1` in the toolkit
(`NeutrinoTaggerNuE.cxx:1516` declared, tested at `:1566`, a `std::set<Facade::Cluster*>`).

These agree iff cluster↔id is injective within the event. It should be, and
doc 53 (`project_real_cluster_id_epochs`) is a reminder that cluster identity
and cluster id have had a non-trivial relationship in this tree before. Flagged
as latent, not as a live defect.

### P9 — the `numu_cc_1_*` arrays are index-ordered here and pointer-ordered in the prototype

**Severity: low. Class: determinism improvement (with a caveat).**

The prototype iterates `map_vertex_segments[main_vertex]`, a
`std::set<ProtoSegment*>` — address order — and pushes eight parallel arrays
per segment. The toolkit iterates `sorted_out_edges(vd, graph)` — edge-index
order (`NeutrinoTaggerNuMu.cxx:190`).

The toolkit is strictly better and reproducing the prototype's order would be
an M4 regression. The caveat: the arrays are *ntuple output*, so a
prototype↔toolkit comparison of `numu_cc_1_length[i]` element by element will
disagree even when the two runs are identical in content. Compare as multisets.
The same applies to `cosmic_tagger`'s and `ssm_tagger`'s array fields.

pr/35 P7 recorded the identical pattern in `fill_kine_tree`. This is the
series' most repeated finding and the most consistently benign.

### P10 — a null guard the prototype does not have

**Severity: low. Class: prototype bug not reproduced.**

```cpp
// NeutrinoTaggerSSM.cxx:911
if (!ssm_sg) return exit_ssm();
```

The prototype's SSM selection loop (`NeutrinoID_ssm_tagger.h:545-561`) has two
branches and leaves `ssm_sg` at `0` if `all_ssm_sg` is empty; it then
dereferences it unguarded. The path requires `Nsm > 0` while `all_ssm_sg` is
empty, which the surrounding code makes unlikely but does not exclude. The
toolkit exits cleanly instead of crashing.

Same class as pr/35 P11/P12. **Do not "restore" the prototype behaviour.**

### P11 — `stm_tol_vec` is sized 6 where the prototype sizes it 5

**Severity: low. Class: port defect (probably inert).**

```cpp
// prototype NeutrinoID_cosmic_tagger.h:32
std::vector<double> stm_tol_vec = {-1.5*units::cm, ...};   // 5 entries
// toolkit NeutrinoTaggerCosmic.cxx:536
const std::vector<double> stm_tol_vec(6, -1.5 * units::cm);
```

Harmless if `FiducialUtils::inside_fiducial_volume` reads only the first five
tolerances, a real behaviour change if it reads a sixth boundary the prototype
left untightened. `NeutrinoTaggerNuE.cxx:2357` uses a `stm_tol_vec` too and
should be checked with it. One line of the FV implementation settles it; I did
not read it (§7.4).

### P12 — the PDG gate is stricter here

**Severity: low. Class: port defect (interacts with pr/31 P1).**

Prototype: `if (sg->get_particle_type() != 11) continue;` — a plain member
read. Toolkit: `if (!sg || !sg->has_particle_info() || sg->particle_info()->pdg() != 11) continue;`
(`NeutrinoTaggerNuE.cxx:4274`, and again at `:852`, `:897`), and the same shape
appears at `NeutrinoTaggerNuMu.cxx:199`, `NeutrinoTaggerCosmic.cxx:772`, `:815`,
`NeutrinoTaggerSinglePhoton.cxx:2267` and `NeutrinoTaggerSSM.cxx:913`.

A segment whose PDG was set but whose `ParticleInfo` was never constructed is
**silently skipped** by the toolkit and **accepted** by the prototype. That
combination is not hypothetical: pr/31 P1 found the `cal_4mom` guard dropped at
11 of 13 sites, and pr/35 GOTCHA 4 established that `particle_info()` is only
populated where `segment_cal_4mom` ran. So the population of
"PDG set, no ParticleInfo" segments is plausibly non-empty, and every one of
them is invisible to every tagger.

This is the one finding here that is *downstream* of an earlier round's
finding rather than independent of it. Whether it fires depends entirely on
pr/31 P1's disposition.

### P13 — `photon_flag` is knob-gated off on SBND

**Severity: low. Class: config, already documented.**

`TaggerCheckNeutrino.cxx:832-839` runs `singlephoton_tagger`, then:

```cpp
if (m_sp_photon_flag) { if (flag_sp) tagger_info.photon_flag = 1.0f; }
else { (void)flag_sp; }   // legacy: verdict computed and discarded
```

SBND sets `sp_photon_flag=false` (`clus.jsonnet:758`), so the verdict is
discarded where the prototype sets it (`NeutrinoID.cxx:271`). This is a known,
recorded gap (doc pr/26 §8.2) with a knob already built, listed here only for
completeness of the driver comparison. Note the asymmetry with `cosmic_flag`
(§2.8): the prototype discards *that* one deliberately.

> **CORRECTED 2026-08-05 (doc pr/37 §11.4) — the premise is stale: it is ON in
> production, and there is no gap.** `clus.jsonnet:766`/`:2038` are `false`, but
> those are the **module defaults**. Per doc 68 the SBND operating point lives
> only in `wct-pr-perevt.jsonnet`, where **`:835` reads `sp_photon_flag = true`**
> with a comment citing pr/26 §9.3's own gate (*1215 of 1216 `T_tagger`
> branch-values identical; the one that moves is `photon_flag` 0 → 1 on evt
> 172230*). So the toolkit sets the verdict where the prototype does. This is a
> wrong **negative** in a P-list, produced by reading a knob's value from
> `clus.jsonnet` instead of the operating point — the same mistake §10's knob
> count exists to prevent. §10.10b's "RESOLVED — the knob exists and is wired"
> understates it; the knob is not merely wired, it is on.

---

## §4 The SBND operating point — a calibration surface, not a port defect

Kept structurally separate on purpose. Everything in this section is the
toolkit faithfully applying uBooNE-derived numbers to SBND, which is a
different kind of problem from a mistranslation and should not be triaged as
one.

| knob | SBND value (`clus.jsonnet`) | status |
|---|---|---|
| `cosmic_y_top_main` | `sbnd_y_top − 17` `:991` | **SBND-corrected** |
| `cosmic_y_top_strict` | `sbnd_y_top − 15` `:992` | **SBND-corrected** |
| `cosmic_y_top_loose` | `sbnd_y_top − 37` `:993` | **SBND-corrected** |
| `cosmic_y_small_piece` | default | uBooNE literal |
| `vertex_z_prior_scale` | `100.0` `:1008` | **SBND-corrected** (proto 200) |
| `ssm_target_dir` | `null` `:1018` | **uBooNE BNB-target vector** |
| `ssm_absorber_dir` | `null` | **uBooNE NuMI-absorber vector** |
| `mip_dqdx` / `mip_dqdx_median` | defaults | uBooNE `50e3` / `43e3` |
| BDT weight XML | `uboone/weights/…` | **uBooNE-trained, uncalibrated** |

The asymmetry is worth naming: the *geometric* extents were localised (doc
pr/2 §2e(iv)) but the *beam-line* vectors were not (doc pr/2 §2e(i)), and the
BDT weights themselves are doc pr/2 gap **G1** — uBooNE-trained, applied to
SBND without retraining, with no calibration study in this tree.

That last point dominates every finding in §3 in magnitude. It is also not
something this audit can act on. Stated once, here, so it does not get read as
a port divergence.

---

## §5 Looks like a divergence and is not

Ten entries, so a future reader does not re-chase them.

**5.1 The 20 unwritten `*_score` fields.** `br1_score`, `br3_score`,
`br4_tro_score`, `cme_anc_score`, `gap_score`, `hol_lol_score`,
`mgo_mgt_score`, `mipid_score`, `mipquality_score`, `pio_1_score`,
`stemdir_br2_score`, `stw_spt_score`, `trimuon_score`, `vis_1_score`,
`vis_2_score` and `cosmict_{2_4,3_5,6,7,8}_score` are written by the prototype
and never by the toolkit — but every one of them is written **only inside
`cal_bdts()` / `cal_numu_bdts()`**, the TMVA paths
(`NeutrinoID_nue_bdts.h:325-345`, `_numu_bdts.h:103-112`), and read **only** by
those same TMVA readers. §2.4 establishes production runs XGBoost. The XGBoost
paths write a *different* 19 sub-scores (`br3_3`, `br3_5`, `br3_6`, `pio_2`,
`stw_2/3/4`, `sig_1/2`, `lol_1/2`, `tro_1/2/4/5`, `numu_1/2/3`,
`cosmict_10`) and the toolkit writes all of those. Cost: those 20 ntuple
branches are constant in toolkit output. Not a logic defect.

**5.2 `NeutrinoTaggerSSM.cxx` is 1549 lines against the prototype's 3598.**
Almost all of the gap is `exit_ssm_tagger` (504 lines → a 175-line lambda,
proven equivalent in §2.2) and `print_ssm_tagger` (390 lines, a printer, not
ported). Nothing algorithmic is missing.

**5.3 `NeutrinoTaggerSinglePhoton.cxx` is 2533 lines against 4275.** The
prototype header contains ~1500 lines of **commented-out** duplicates of the
nue sub-taggers (`/*bool ...::single_shower(`, `::angular_cut(`,
`::track_overclustering(`, `::broken_muon_id(`, `::shower_to_wall(`,
`::gap_identification(`, `::mip_quality(`, `::single_shower_pio_tagger(` at
`:599, :715, :826, :1289, :1498, :1717, :1948, :3072`). Dead text.

**5.4 `bad_reconstruction` living in the cosmic translation unit.** File
placement, not a logic change — see §2.1. The toolkit's own comment at
`NeutrinoTaggerNuE.cxx:529` and `:3362` explains it.

**5.5 The cosmic verdict being discarded.** §2.8. Symmetric.

**5.6 The 14 `ssm_kine_*` fields written by neither side.** `ssm_kine_pio_*`,
`ssm_kine_reco_Enu`, `ssm_kine_reco_add_energy` are declared in `TaggerInfo`,
initialised by `init_tagger_info`, and never assigned by any tagger in either
tree. Dead on both sides; not a port gap.

**5.7 `cosmict_score` written by neither side.** Same shape. Named in doc
pr/26's stage-2 plan as "the numu-BDT cosmic score", but no code in either tree
assigns it. If a display shows it, it is showing the init default.

**5.8 The toolkit's five extra written fields.** `nue_score` / `numu_score` are
written by the scorer stage rather than the tagger, which mirrors the
prototype's `NeutrinoID.cxx:277-286`; the other three are pr/35 GOTCHA 3's
renames.

**5.9 `flag_print` / `flag_fill` disappearing from every sub-tagger
signature.** The prototype threads two booleans through all ~30 sub-taggers;
the toolkit's static helpers take neither and always fill. Every live call site
in `nue_tagger` passes `flag_fill = true`, so the fills are unconditional on
both sides. `flag_print` gates `std::cout` only.

**5.10 `dQ_dx_cut * 43e3/units::cm` becoming `dQ_dx_cut * m_mip_dqdx_median`.**
The knob's default is `43000.0` (`TaggerCheckNeutrino.h`), SBND does not set
it, so the value is the uBooNE literal. Doc pr/8's configurability round;
byte-identical when unset.

---

## §6 Determinism

Full sweep of the six tagger translation units at `23bd6783`.

| file | raw `boost::` iteration | pointer-keyed containers | `ordered_*` helper uses |
|---|---|---|---|
| `TaggerCheckNeutrino.cxx` | 0 | 0 | 2 |
| `NeutrinoTaggerCosmic.cxx` | 0 (2 in comments) | 0 | 6 |
| `NeutrinoTaggerNuMu.cxx` | 0 (3 in comments) | 0 | 4 |
| `NeutrinoTaggerNuE.cxx` | 0 (5 in comments) | **2 iterated / 7 declared** (1 unexamined) | 31 |
| `NeutrinoTaggerSinglePhoton.cxx` | 0 (4 in comments) | 0 confirmed iterated / 3 declared | 18 |
| `NeutrinoTaggerSSM.cxx` | 0 (1 in comment) | 0 (1 uses `SegmentIndexCmp`) | 10 |

**Zero raw `boost::edges` / `boost::vertices` / `boost::out_edges` iteration
survives anywhere in the stage** — every hit is inside an explanatory comment.
That is a better result than pr/35's, and it is deliberate: three files carry
comments stating that `ordered_edges` was chosen *because* the following `+=`
is order-sensitive.

The residual hazard is entirely `std::set<SegmentPtr>` / `std::set<ShowerPtr>`
— smart-pointer sets that order by address. Two are iterated in a way that
matters (P3, P5); four more are membership-only and are enumerated in P5 so
they are not re-examined. One was **not** examined: `muon_segs`
(`NeutrinoTaggerNuE.cxx:3494`, inside `track_overclustering`) has the same
shape as P3's `muon_segments` and should be assumed to carry the same hazard
until someone reads it.

`ChargeMap` and the `map_*` structures inherited from earlier stages are
ordered containers and are not implicated here. Do not conflate them with the
sets above (pr/35 GOTCHA 6).

---

## §7 Loose ends

Ranked by cost-to-resolve.

**7.1 P1's diagnostic half — the cheapest and highest-value item this round.**
One event, two `cluster_fc_check` calls (with and without
`m_fiducial`/`m_fv_tolerance`), print both `is_fc` values. If they agree on a
sample of events, P1 collapses to a documentation fix. If they disagree, P1 is
the round's most consequential finding and the FV question has to be settled
before anything else. Nothing else here has that ratio.

**7.2 P3's magnitude, via the M4 protocol.** Run the nue chain N times under
`setarch x86_64 -R` on the same event and compare `brm_acc_length`,
`brm_acc_direct_length`, `brm_flag` and `nue_score`. Same protocol pr/35 §7.3
proposed for the shower-kinematics sum; this stage's version is more direct
because the veto is boolean and visible in one field. Would settle P3 and P5
together.

**7.3 The per-sub-tagger cut values for the ~30 functions not deep-dived.**
The largest remaining unknown by volume. §2.1 proves nothing is *missing* and
§2.3 proves the order is right, but a transposed constant inside
`track_overclustering` or `bad_reconstruction_2` would pass every check in this
doc. The in-tree `clus/docs/tagger/*_review.md` set (15 files, ~230 KB) covers
much of this ground and was **not** read this round — deliberately, because
pr/33 found an in-tree review doc whose proposed fix created a divergence, and
reconciling 15 of them properly is its own round.

**7.4 P11 — read `FiducialUtils::inside_fiducial_volume`.** One function. Tells
you whether the 6th tolerance entry is read, and settles P11 outright.

**7.5 P4 — who reads the `neutrino_type` branch?** A grep of the uBooNE
analysis chain outside this tree. If nobody does, P4 drops to §5.

**7.6 P2's downstream consumers.** `shw_sp_*` is not a BDT input; something
must read it or the whole single-photon tagger is dead weight in the toolkit.
Finding that consumer sets P2's real severity.

**7.7 P6 — how often do the two endpoint rules disagree?** Instrument both at
`NeutrinoTaggerNuE.cxx:4316` and count over the valfast sample. Cheap, and it
converts P6 from "could reverse a direction" to a number.

**7.8 The SSM tagger's non-exit sentinel coverage.** §2.2 proves the *exit*
path is equivalent and §2.1 proves field coverage is identical (362 = 362), but
the toolkit computes its `ParticleBlock`s once (`NeutrinoTaggerSSM.cxx:1080`,
`:1083`) and assigns them in one block at `:1355+` where the prototype
interleaves. A mid-function early return present in one and not the other would
leave different subsets at `-999`. None was found; 976 lines were not searched
exhaustively for one. Listed here rather than as a finding because "I did not
look" is not evidence.

**7.9 The series' outstanding item, unchanged.** pr/34 §7.7: re-check
pr/28–pr/33's prototype citations against `a5fc0b9`. This round's stage is
provenance-clean and pr/35's was too, but pr/28–pr/33 each cite files carrying
78–318 changed lines. Line anchors have shifted in every one of them, and
`08f229a` / `3dc01ac` are algorithm changes, not instrumentation. **This is
still the highest-value follow-up in the series**, and with the eight stages now
mapped there is no longer a reason to defer it.

---

## §8 Summary

| # | finding | severity | class | anchor |
|---|---|---|---|---|
| P1 | `match_isFC` recomputed, on a knowingly-disagreeing FV | **high** | port defect | `TaggerCheckNeutrino.cxx:852` |
| P2 | `singlephoton_tagger` drops SCE position correction | **high** | port gap | `NeutrinoTaggerSinglePhoton.cxx:2243` |
| P3 | `broken_muon_id` sums floats in address order | **high** | determinism (reproduced) | `NeutrinoTaggerNuE.cxx:1518` |
| P4 | `neutrino_type` bitmask dropped; it is an output branch | medium | port gap (output) | `NeutrinoTaggerNuE.cxx:45` |
| P5 | `mip_quality` iterates `std::set<ShowerPtr>` | medium | determinism (reproduced) | `NeutrinoTaggerNuE.cxx:1730` |
| P6 | stem endpoint by proximity, not wcpt index | medium | port defect | `NeutrinoTaggerNuE.cxx:4316` |
| P7 | `3.1415926` vs `M_PI` | low | cosmetic | `NeutrinoTaggerNuE.cxx:4319` |
| P8 | cluster count by pointer, not by id | low | port defect (latent) | `NeutrinoTaggerNuE.cxx:1566` |
| P9 | `numu_cc_1_*` arrays index-ordered, not pointer-ordered | low | determinism improvement | `NeutrinoTaggerNuMu.cxx:190` |
| P10 | `!ssm_sg` guard the prototype lacks | low | prototype bug not reproduced | `NeutrinoTaggerSSM.cxx:911` |
| P11 | `stm_tol_vec` sized 6 vs 5 | low | port defect (probably inert) | `NeutrinoTaggerCosmic.cxx:536` |
| P12 | PDG gate requires `has_particle_info()` | low | port defect | `NeutrinoTaggerNuE.cxx:4274` |
| P13 | `photon_flag` knob-gated off on SBND | low | config (documented) | `TaggerCheckNeutrino.cxx:833` |

---

## §9 What is NOT claimed

* **No event was run.** Every statement is static reading plus the two parsing
  scripts. No frequency is measured for any finding.
* **Coverage is not exhaustive**, and this is the first round in the series
  where that has to be said. §0 states per-file depth; ~30 sub-taggers had
  their field writes and dispatch verified but **not** their internal cut
  values. A wrong constant inside one of them is not excluded by anything here.
* **P1 is not shown to change any score.** It is shown to be a different
  quantity computed on a fiducial volume the tree itself flags as
  disagreeing. Whether the two agree in practice is §7.1.
* **P3 and P5 are hazard arguments, not measurements.** No run-to-run
  difference was observed, because no runs were made. The two in-tree
  determinism nulls are explicitly reconciled in P3 and neither covers this
  code.
* **P2's severity is bounded but not zero.** `shw_sp_*` is not a BDT input on
  either side; the consumer that makes it matter is unidentified (§7.6).
* **P9, P10 and (probably) P11 are the toolkit being better than the
  prototype.** They are listed for completeness of the comparison, not as
  defects to fix.
* **The BDT weights are not evaluated.** Doc pr/2 gap G1 stands. §4 states the
  uBooNE-calibration surface; nothing in §3 should be read as covering it.
* **The 15 in-tree `clus/docs/tagger/` review docs were not reconciled** with
  this audit (§7.3). Where they and this doc disagree, neither has priority
  until someone checks.
* **Nothing was changed.** Zero toolkit files were modified this round;
  verified against a working tree that a concurrent session was editing.

---

## §10 The owner filter — 13 → 7

**What this section is.** The owner's standing rule for this series: *drop the
findings where the toolkit improves on the prototype; keep only the bugs and
the things missing from the port, and propose a fix for each survivor.* Applied
to pr/30 (14→4), pr/31 (15→9), pr/32 (12→4), pr/33 (14→5), pr/34 (14→5) and
pr/35 (14→4). This round lands on **seven**. The count is an output, not a
target.

**Re-verification basis.** §3 was written against toolkit `23bd6783`. HEAD is
now **`407c5ba9`**, twelve commits later. Every anchor below was re-derived at
HEAD, and — because a concurrent session holds twelve modified toolkit files,
two of them in this stage — **every file was read at `git show HEAD:<path>`,
never from the working tree**. Snapshots in
`/home/xqian/tmp/claude-25225/pr36f/`.

**The staleness answer this round is nearly clean, and that is itself worth
recording**, because the series' answer has varied: pr/32's anchors were four
commits stale within a day, pr/33's were up to +19 lines *with a finding created
underneath them*, pr/34's came back clean, pr/35's had seven files move over
eleven commits.

| file | 23bd6783 → 407c5ba9 | anchors |
|---|---|---|
| `NeutrinoTaggerCosmic.cxx` | **unchanged** (1353) | §3 anchors stand |
| `NeutrinoTaggerNuMu.cxx` | **unchanged** (385) | stand |
| `NeutrinoTaggerNuE.cxx` | **unchanged** (4414) | stand |
| `NeutrinoTaggerSSM.cxx` | **unchanged** (1549) | stand |
| `NeutrinoTaggerSinglePhoton.cxx` | **unchanged** (2533) | stand |
| `NeutrinoKinematics.cxx`, `NeutrinoTaggerInfo.h` | **unchanged** | stand |
| `TaggerCheckSTM.cxx`, `Clustering_Util.cxx` | **unchanged** | stand |
| `root/src/Uboone{Nue,Numu}BDTScorer.cxx` | **unchanged** | stand |
| `TaggerCheckNeutrino.cxx` | **940 → 1037** (+97, **0 deleted**) | **all shift, see §10.12** |
| `TaggerCheckNeutrino.h` | 188 → 217 (+29, 0 deleted) | — |

Only the driver moved, and it moved by pure insertion at four hunks
(`@@ -81,0 +82,14 @@`, `@@ -222,0 +237,15 @@`, `@@ -512,0 +542,24 @@`,
`@@ -891,0 +945,44 @@`). Every §3 anchor in that file above old line 512 shifts
by exactly **+53**. §10.12 tabulates them.

### 10.1 The filter

| P | finding | verdict | why |
|---|---|---|---|
| P1 | `match_isFC` recomputed on a disagreeing FV | **KEEP → F1** | definition change on a live BDT input |
| P2 | `singlephoton_tagger` drops SCE correction | **KEEP → F3** | port gap; the plumbing exists 30 lines away |
| P3 | `broken_muon_id` sums floats in address order | **KEEP → F4** | merged with P5 + a third site §6 missed |
| P4 | `neutrino_type` bitmask dropped | **KEEP → F7** | genuinely missing from the port (output branch) |
| P5 | `mip_quality` iterates `std::set<ShowerPtr>` | **merged into F4** | same mechanism, same fix |
| P6 | stem endpoint by proximity, not wcpt index | **KEEP → F5** | port defect — and **22 sites, not one** |
| P7 | `3.1415926` vs `M_PI` | **DROP** | toolkit better; §3 already says do not "fix" it |
| P8 | cluster count by pointer, not by id | **KEEP → F6** | latent port defect; co-located with F4's loop |
| P9 | `numu_cc_1_*` arrays index-ordered | **DROP** | determinism improvement; reproducing it is an M4 regression |
| P10 | `!ssm_sg` guard the prototype lacks | **DROP** | prototype bug not reproduced |
| P11 | `stm_tol_vec` sized 6 vs 5 | **RESOLVED — and inverted** | the 6 is the *correct* translation; see §10.10a |
| P12 | PDG gate requires `has_particle_info()` | **KEEP → F2, mechanism RETRACTED** | the divergence is real, §3's account of *why* is wrong; see §10.3 |
| P13 | `photon_flag` knob-gated off on SBND | **RESOLVED** | the knob exists and is wired; see §10.10b |

**Kept: 7. Resolved outright: 2. Dropped as improvements: 3.** P5 folded into
P3, so thirteen entries produce seven findings.

**Severity is re-ranked.** §8 put P1/P2/P3 at "high" and P12 at "low". F2 (was
P12) is now the finding with the **largest plausible affected population** —
if its measurement comes back non-zero it ranks first, above F1. It is listed
second only to keep §3's headline readable; §10.3 states why.

### 10.1a Findings are not knobs

Seven findings do **not** mean seven default-OFF knobs. Presenting them as
peers would smuggle two category changes past the owner — the pr/33 GOTCHA 20
trap.

| F | = P | shape | new knobs |
|---|---|---|---|
| F1 | P1 | knob implements **one of two readings**; the other is a threading decision | 2 keys, 1 knob |
| F2 | P12 | **measurement first**, then a knob | 1 (+1 counter) |
| F3 | P2 | plumbing; `clus_geom_helper` already exists | 1 gate bool (see §10.4) |
| F4 | P3+P5+§6 | **M4 house-rule, prototype `n/a`** — *not* a fidelity fix | 1 |
| F5 | P6 | 1 knob across **22 independent sub-taggers** ⇒ needs a per-site counter | 1 (+counter) |
| F6 | P8 | one-line | 1 |
| F7 | P4 | cheap fix, **unmeasured value** | 1 (+ a T_tagger branch) |

**F4 is the one to read carefully.** It is a CLAUDE.md §2 determinism
house-rule violation, not a port defect: the prototype does the same
address-ordered thing. Flipping it moves the toolkit **further** from the
prototype's order, not closer. Same class as pr/33's F5.

### 10.2 F1 = P1 — `match_isFC`

Anchor at HEAD: `TaggerCheckNeutrino.cxx:904-907` (was `:851-853`).

§3's two readings stand and I still decline to pick (§5 rule 4, M15). What
§10 adds is that **reading (i) is a knob and reading (ii) is not**, so they are
not symmetric options:

* **(i) mirror `TaggerCheckSTM`'s fiducial knob.** That component already has
  the exact template: `m_use_fiducial = !config["fiducial"].isNull()`
  (`TaggerCheckSTM.cxx:101`), `m_fv_tolerance` parsed at `:110-111`, members
  declared at `:696-697`, used at `:2939-2940`. Add the same two keys to
  `TaggerCheckNeutrino` and pass them through:

  ```cpp
  // TaggerCheckNeutrino.cxx:905, from
  auto fc_result = Facade::cluster_fc_check(*main_cluster, m_dv);
  // to
  auto fc_result = Facade::cluster_fc_check(*main_cluster, m_dv,
                                            m_use_fiducial ? m_fiducial : nullptr,
                                            m_fv_tolerance);
  ```

  `cluster_fc_check`'s own comment (`Clustering_Util.cxx:108-110`) states the
  `fiducial == nullptr` path is "the historical FiducialUtils path,
  **bit-for-bit**" — so the knob is byte-identical while unset, by the callee's
  documented contract rather than by our assertion.

* **(ii) thread the upstream verdict from the matching stage.** No knob
  reproduces this; it is a data-flow decision about whether this stage may
  depend on the light-matching result. Out of scope for a default-OFF change.

**Do the diagnostic before either.** §7.1 remains the highest-value cheap item
in the round: one event, both FV definitions, compare `is_fc`. Agree on a
sample ⇒ F1 collapses to a documentation fix and (ii) becomes moot.

### 10.3 F2 = P12 — the mechanism in §3 is wrong, and the real one is bigger

**This is the largest change §10 makes to §3, and §3's version must be retracted
in words rather than quietly replaced.**

§3's P12 says the divergence is a segment "whose PDG was set but whose
`ParticleInfo` was never constructed", reachable via pr/31 P1's dropped
`cal_4mom` guard. **That account does not survive.** Taking the eight
skip-gates one at a time, they are of two shapes only —
`if (!has_particle_info() || pdg() != 11) continue;` and
`if (!has_particle_info()) continue;` followed by `pdg == 13 || pdg == 2212`.
The prototype's `particle_type` member defaults to `0` (`ProtoSegment.cxx:29`),
and `0` is neither 11 nor 13 nor 2212 — so on a segment with no type set,
*both* trees skip. Identical. This is pr/33 GOTCHA 23's reasoning applied to
the same idiom.

**The real divergence is one level down, and it is verified.**
`get_particle_type()` is **not a member read**:

```cpp
// prototype  pid/src/ProtoSegment.cxx:10-15
int WCPPID::ProtoSegment::get_particle_type(){
  if (get_flag_shower()){
    particle_type = 11;   // shower are all treated as electron
  }
  return particle_type;
}
```

Every prototype read coerces a **shower-flagged** segment to PDG 11 — **and
mutates the member while doing it**, so the coercion latches on first read. The
toolkit has no counterpart: `grep "treated as electron"` over
`PRSegment.{h,cxx}` and `PRCommon.h` returns zero, and `has_particle_info()`
(`PRSegment.h:81`) is a plain null test.

So on a shower-flagged segment carrying no explicit `ParticleInfo`:

| | prototype | toolkit |
|---|---|---|
| `if (type != 11) continue;` | type coerced to **11** ⇒ **processed as an electron** | no info ⇒ **skipped** |

That is the opposite population from the one §3 named, and in an *nue* tagger
it is the majority population, not a corner case.

**The coupling is mutual and the toolkit gets the other half right.**
`seg_is_shower` (`NeutrinoTaggerNuE.cxx:76-80`) does carry the
`std::abs(pdg) == 11` term — the term pr/33 GOTCHA 3 found missing from
`porting_dictionary.md:222`. The prototype's pair is a two-way latch
(`flag_shower ⇒ pdg=11`, `pdg==11 ⇒ flag_shower`); the toolkit ported the
second implication and not the first.

**Verified vs not verified — kept separate on purpose.**

* **Verified**: the coercion exists, mutates, and has no toolkit counterpart.
  Also verified: **parity is not "widen the toolkit gate"**. Widening reproduces
  the *read* and not the *latch*, and the latch is observable — a later
  `seg_is_shower` on the same segment sees the written 11.
* **NOT verified**: that the affected population is non-empty. The toolkit does
  write PDG 11 explicitly at `NeutrinoShowerClustering.cxx:212`, `:432`, `:751`,
  and shower clustering runs before the taggers. Whether that coverage is
  *total* was not established, and I did not check it.

**Discriminating check, to run before acting on F2.** At tagger entry, count
segments satisfying
`(flags_any(kShowerTrajectory) || flags_any(kShowerTopology)) && !has_particle_info()`.
One counter line in the `PR30AUDIT` / `PR32AUDIT` pattern already in this file
(`TaggerCheckNeutrino.cxx:946-969`, `:971-988`). **Zero over the sample ⇒ the eight
skip-gates are equivalent and F2 resolves dead-by-construction, like pr/35's
P10. Non-zero ⇒ F2 is live and that count is its magnitude.**

**Fix, if the count is non-zero.** A default-OFF `pdg_shower_coercion` on the
tagger stage: before the gates, for each segment, if `seg_is_shower(sg)` and
`!sg->has_particle_info()`, construct a `ParticleInfo` with pdg 11. That
reproduces the latch, not just the read. Sites: the eight gates are
`NeutrinoTaggerNuE.cxx:396, :852, :897, :989, :1014, :4089, :4274` and
`NeutrinoTaggerSinglePhoton.cxx:1400`.

**§3's non-NuE site list is also wrong and should be struck.** P12 names
`NeutrinoTaggerNuMu.cxx:199`, `NeutrinoTaggerCosmic.cxx:772`, `:815`,
`NeutrinoTaggerSinglePhoton.cxx:2267` and `NeutrinoTaggerSSM.cxx:913`. All five
are `int pdg = sg->has_particle_info() ? sg->particle_info()->pdg() : 0;`
ternaries — which yield exactly the prototype's member default and are
**faithful**. Of 54 `has_particle_info()` uses across the five tagger files,
only **8** are skip-gates.

### 10.4 F3 = P2 — SCE, and the plumbing is thirty lines away

The cleanest fix in the round, because the driver already does it for the
sibling call.

```cpp
// TaggerCheckNeutrino.cxx:876-885 — singlephoton_tagger gets no helper
pattern_algos.singlephoton_tagger(*pr_graph, main_cluster, final_main_vertex,
                                  showers, map_vertex_to_shower, …, m_dv,
                                  tagger_info);

// TaggerCheckNeutrino.cxx:938 — thirty lines below, in the same function
kine_info = pattern_algos.fill_kine_tree(…, m_dv,
        m_geom_helper,   // nullptr when clus_geom_helper is not configured
        particle_data(), m_recomb_model);
```

`SpContext` already **has the slot** — `IClusGeomHelper::pointer geom_helper;
// nullable; for SCE correction in entry point`
(`NeutrinoTaggerSinglePhoton.cxx:116`) — and the construction hardwires it:
`SpContext ctx{…, dv, nullptr};` (`:2219-2222`). The correction primitive
exists too: `geom_helper->get_corrected_point(p, IClusGeomHelper::SCE, apa, face)`
(`NeutrinoKinematics.cxx:65`).

**Fix.** Add an `IClusGeomHelper::pointer` parameter to
`singlephoton_tagger` (`NeutrinoPatternBase.h:816`), pass `m_geom_helper` from
the driver, drop it into the `SpContext` slot, and apply it at the five sites
the prototype corrects — `NeutrinoID_singlephoton_tagger.h:13` (nu vertex),
`:103`, `:132` (track starts), `:222`, `:317` (shower starts) — plus the
`max_shw_dis` pair at `:324`, `:327-328`, which is a distance between two
*corrected* points and so is not fixed by correcting the coordinates alone.

**Why a separate gate bool, not just `clus_geom_helper`.** SBND leaves
`clus_geom_helper` at `""` (`TaggerCheckNeutrino.cxx:309`), so threading it is a
no-op **today** and byte-identical. But the key is shared with `fill_kine_tree`:
the day someone sets it for kine, the single-photon positions would move in the
same commit, unannounced. Recommend a `sp_sce_correction` bool defaulting
**false** so the two consumers stay independently gateable. Same reasoning that
made pr/35 GOTCHA 10 a *config* gap rather than a silent one.

Damage bound from §3 stands: `shw_sp_*` is not a BDT input on either side, so
this moves the ntuple, not `nue_score`.

### 10.5 F4 = P3 + P5 + the site §6 said it had not examined

**§6 undercounted. `muon_segs` is a third live site, and it is a float sum.**
§6 listed it (`NeutrinoTaggerNuE.cxx:3494`, inside `track_overclustering`) as
"not examined … should be assumed to carry the same hazard until someone reads
it". Read:

```cpp
// NeutrinoTaggerNuE.cxx:3494, :3519-3520
std::set<SegmentPtr> muon_segs;
…
double stem_length_1 = 0;
for (SegmentPtr s : muon_segs) stem_length_1 += segment_track_length(s);
```

`stem_length_1` feeds three hard cuts (`:3563` `> 6 cm`, `:3564` `> 40 cm`,
`:3570` `> 40 cm`) and is pushed to the ntuple at `:3578`. Same shape as P3
exactly. §7.8's own words — *"I did not look" is not evidence* — applied here
and the assumption was right.

The three live sites, and the two the doc already cleared:

| site | container | iterated at | exposure |
|---|---|---|---|
| `:1394` `muon_segments` | `std::set<SegmentPtr>` | `:1518` | **two float sums** → `:1562-1574` cuts, `brm_Ep` |
| `:1685` `connected_showers` | `std::set<ShowerPtr>` | `:1730` | first-wins booleans → `mip_quality_flag_inside_pi0` |
| `:3494` `muon_segs` | `std::set<SegmentPtr>` | `:3520` | **one float sum** → `:3563-3570`, `tro_2_v_stem_length` |
| `:1686` `tmp_pi0_showers` | — | never | `.count()`/`.size()` only — safe |
| `:4264` `good_showers` | — | never | `.count()` only — safe |

**Fix, and it is a type swap the tree already provides.** `IndexedSegmentSet =
std::set<SegmentPtr, SegmentIndexCmp>` (`PRGraph.h:283`) and
`IndexedShowerSet = std::set<ShowerPtr, ShowerIndexCmp>` (`PRShower.h:232`)
exist, and `NeutrinoTaggerSSM.cxx:608` in this same stage already uses
`SegmentIndexCmp`. One default-OFF `tagger_ordered_segment_sets` flipping all
three declarations.

**Class label, stated because it is not optional (pr/33 GOTCHA 20).** This is
an **M4 / CLAUDE.md §2 house-rule violation, prototype `n/a`** — the prototype
does the same address-ordered thing with `std::set<ProtoSegment*>`
(`NeutrinoID_nue_tagger.h:1036`, iterated `:1139`). The hazard is *reproduced*,
not introduced, and the fix makes the toolkit diverge **further** from the
prototype's order. It belongs in the keep-list because address-ordered float
accumulation is forbidden here regardless of provenance — not because it is a
fidelity defect.

Magnitude is still unmeasured; §7.2's M4 protocol (N runs under
`setarch x86_64 -R`, compare `brm_acc_length`, `brm_acc_direct_length`,
`brm_flag`, `tro_2_v_stem_length`, `nue_score`) settles all three sites at once.

### 10.6 F5 = P6 — 22 sites, and the prototype's test is not implementable as written

Two corrections to §3, both enlarging it.

**(a) It is 22 call sites, not one.** `seg_endpoint_near` is called at
`NeutrinoTaggerNuE.cxx:238, :370, :710, :979, :1151, :1383, :1998, :2557,
:2981, :3188, :3413, :3834, :4316` (13) and
`NeutrinoTaggerSinglePhoton.cxx:502, :712, :1003, :1126, :1793` (5), plus the
two definitions. §3's `:4316` is the entry-point site only. This is the
toolkit's *systematic* replacement of the prototype's wcpt-index rule across
the whole stage.

**(b) The prototype's rule cannot be transcribed, because the field is gone.**
`WCPoint` in the toolkit is:

```cpp
// clus/inc/WireCellClus/PRCommon.h:96-110
struct WCPoint {
    WireCell::Point point;   // 3D point
    // int uvw[3] = {-1,-1,-1}; // wire indices
    // int index{-1};           // point index in some container
    // FIXME: WCP had this, does WCT need it?
};
```

`index` is **commented out**. So `main_vertex->get_wcpt().index ==
sg->get_wcpt_vec().front().index` has no toolkit spelling, and
`seg_endpoint_near` is a *forced* substitution, not a careless one. That changes
what "fix it" means.

**Fix.** `Segment::wcpts()` (`PRSegment.h:91`) and `Vertex::wcpt()`
(`PRVertex.h:65`) both survive — only the integer id is gone. wcpts are discrete
skeleton nodes, so index identity is position identity:

```cpp
// prototype NeutrinoID_nue_tagger.h:71-75 — wcpt INDEX identity.
// WCPoint::index is commented out (PRCommon.h:99), so compare positions:
// the wcpts are skeleton nodes, so identity is exact coincidence.
const bool front_at_vertex =
    !seg->wcpts().empty() &&
    ray_length(Ray{vtx->wcpt().point, seg->wcpts().front().point}) < eps;
Point test_p = front_at_vertex ? seg->fits().front().point
                               : seg->fits().back().point;
```

Note both rules return a **fit** point (pr/31: `get_point_vec()` *is*
`fit_pt_vec`); only the selection rule differs, so the fix is scoped to the
selection.

**Ship the knob with a per-site disagreement counter.** One
`stem_endpoint_wcpt_parity` across 22 call sites in ~18 *independent*
sub-taggers cannot tell the gate which sub-tagger moved — the attribution
problem that forced pr/33's F1 into two knobs. A counter keyed by call site is
cheaper than 22 knobs and answers §7.7 in the same run.

A related precedent is already in-tree: `find_vertices`
(`PRGraph.cxx:211-247`) resolves the same front/back question by comparing
`vtx->wcpt().point` against `seg->wcpts().front().point` — i.e. the toolkit
**already** does the wcpt-space comparison in the graph layer while the taggers
do it in fit space. The fix aligns the two.

### 10.7 F6 = P8 — cluster identity vs cluster id

`tmp_clusters` is populated in F4's loop and tested at `:1566`, so one edit
visits both:

```cpp
// NeutrinoTaggerNuE.cxx:1516 / :1521 / :1566
std::set<Facade::Cluster*> tmp_clusters;          // prototype: std::set<int> tmp_ids
    tmp_clusters.insert(mseg->cluster());         // prototype: tmp_ids.insert(cluster_id)
    … tmp_clusters.size() > 1 …                   // prototype :1183
```

Fix: `std::set<int>` keyed on the cluster id, behind
`broken_muon_cluster_id_count`. They agree iff cluster↔id is injective within
the event; doc 53 (`project_real_cluster_id_epochs`) is why that is worth a
knob rather than an assumption. Also note the container is pointer-keyed, so
switching to ids removes an M4 hazard as a side effect — but it is never
iterated, only `.size()`d, so that half is inert.

### 10.8 F7 = P4 — `neutrino_type`

Kept because a dropped output branch is a thing missing from the port, and
§2.10's justification covers only *logic*. The fix is cheap: recompute the
bitmask at the five prototype sites (`cosmic_tagger.h:861`,
`numu_tagger.h:252`/`:254`, `nue_tagger.h:259`,
`singlephoton_tagger.h:538`) behind `neutrino_type_bitmask`.

**Two things the owner should weigh before spending anything on it.** Its value
is unmeasured — §7.5's question (does anything downstream read the branch?) is
unanswered, and if the answer is "nothing", F7 drops to §5. And the fix is not
complete without a writer change: `neutrino_type` is **not booked in
`T_tagger`** (grep = 0 hits in `UbooneTaggerOutputVisitor.cxx`), so computing it
would not make it observable. Ranked last for both reasons.

### 10.9 The gate — two artifacts, and neither alone is sufficient

This series' recurring trap, in a third variant. pr/34: `pctree-pr` would have
PASSed vacuously because the stage was display-only. pr/35: the same gate failed
for the opposite reason — `KineInfo` is not in the pctree at all. Here:

**The right primary gate is `T_tagger`, and it is already produced.**
`tagger_output` is in the SBND PR chain's **default** pipeline string
(`run_pr_chain_batch.sh:115`), so unlike pr/34 and pr/35 **no extra stage is
needed** — the production runner emits the artifact as-is. Compare with the
existing tool:

```bash
sbnd_xin/scripts/analysis/misc/tagger_tree_ab.py \
    <armA>/tracking-*.root <armB>/tracking-*.root T_tagger
```

It diffs branch by branch, dtype- and jagged-aware, on **contents** — so M2
(archive timestamps) does not apply to a ROOT file compared this way. Coverage
was verified field by field for the survivors:

| field | booked in `T_tagger`? |
|---|---|
| `shw_sp_*` (F3) | **yes**, 177 hits |
| `brm_acc_length`, `brm_acc_direct_length` (F4) | **yes** |
| `tro_2_v_stem_length` (F4) | **yes** |
| `cme_angle_beam` (F5) | **yes** |
| `mip_quality_flag_inside_pi0` (F4) | **yes** |
| **`match_isFC` (F1)** | **NO — 0 hits** |
| **`neutrino_type` (F7)** | **NO — 0 hits** |

**So F1 is not gateable on `T_tagger`.** The branches are booked one at a time
(574 explicit `t_tagger->Branch(` calls against 1024 `TaggerInfo` members), and
`match_isFC{0}` (`NeutrinoTaggerInfo.h:1351`) is simply not among them. Its only
outlet is the PR display dump: `out["match_isFC"] = t.match_isFC;`
(`PrDisplayDump.cxx:523`, in `dump_tagger` `:507`), written to
`calib-pr-evt<ID>.json` (`sbnd/clus.jsonnet` `pr_display.output_filename`) —
the same artifact pr/35 landed on, and plain JSON, so directly diffable.

**And the display dump is the wrong *primary* gate.** `dump_tagger` emits
**45** keys — the two scores, the ten cosmic flags with their `_filled`
companions, and the 19 XGBoost sub-scores. It carries none of `shw_sp_*`,
`brm_*`, `tro_2_v_*` or `cme_*`. A gate run on `calib-pr-evt<ID>.json` alone
would PASS on F3, F4 and F5 without looking at anything they move.

**Gate for this stage = `T_tagger` via `tagger_tree_ab.py` for F2–F7, plus the
`tagger` block of `calib-pr-evt<ID>.json` for F1.** Run `T_kine` as a third
comparison if F3 lands, since `clus_geom_helper` is shared.

### 10.10 Resolved outright

**(a) P11 — resolved, and the doc came one step from recommending an
inversion.** §3 asks whether the toolkit's `std::vector<double> stm_tol_vec(6, -1.5cm)`
against the prototype's 5 entries is "a real behaviour change if it reads a
sixth boundary". Both implementations were read.

* Prototype `ToyFiducial::inside_fiducial_volume` (`pid/src/ToyFiducial.cxx`)
  unpacks **five named slots**:
  `[tx_ano, tx_cat, ty_bot, ty_top, tz]` — x-low, x-high, y-low, y-high, and a
  **single** z applied to both faces. All five are `-1.5 cm`
  (`NeutrinoID_cosmic_tagger.h:32`).
* Toolkit `FiducialUtils::inside_fiducial_volume` (`FiducialUtils.cxx:79-121`)
  branches on size: `>= 6` ⇒ `[x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]`; `>= 3` ⇒
  per-axis `[x, y, z]`; else uniform.

The toolkit's size-6 vector is therefore **the correct translation** of the
prototype's 5-slot layout, with `tz` written into both z faces. There is no
sixth boundary and nothing is left untightened.

**And the "fix" §3 gestures at would be a regression.** A 5-entry vector handed
to the toolkit falls into the `>= 3` branch and is reinterpreted as *per-axis*
— silently discarding entries 3 and 4 and reading `tv[2]` as z. With every
entry equal to −1.5 cm the numbers coincide, but the semantics do not, and they
would stop coinciding the moment anyone made the tolerances anisotropic. This
is M7's shape (`<` vs `<=`): a toolkit/prototype difference that exists because
the toolkit's convention is different, not because the port slipped.
**P11 is not a defect. Do not "fix" it toward 5.** `NeutrinoTaggerNuE.cxx:2357`
and `:2501` use the same 6-vector and are correct for the same reason.
§7.4 is closed.

**(b) P13 — `photon_flag`.** Not a divergence to fix: the knob exists, is
documented in-file (`TaggerCheckNeutrino.cxx:868-875`), round-trips in
`default_configuration()` (`:303`), logs unconditionally when on (`:889`), and
SBND's `sp_photon_flag=false` is a *config* decision recorded in doc pr/26
§8.2. §3 lists it "for completeness of the driver comparison", which is the
right reason to have written it down and not a reason to keep it in a fix list.

### 10.11 Dropped, with the reason

* **P7** (`3.1415926` vs `M_PI`) — 1.8×10⁻⁸ relative. §3 already says "do not
  fix it in the direction of the prototype"; the filter agrees.
* **P9** (`numu_cc_1_*` index-ordered) — determinism improvement. Reproducing
  the prototype's address order would be an M4 regression. Keep §3's operational
  note: compare these arrays as **multisets**, not element by element.
* **P10** (`!ssm_sg` guard) — prototype bug not reproduced. Do not restore.

### 10.12 Re-derived anchors

All `TaggerCheckNeutrino.cxx`; every other file's anchors are unchanged (§10
header).

| §3 cites | at HEAD `407c5ba9` | what |
|---|---|---|
| `:735-860` | **`:788-913`** | the tagger block |
| `:751` | **`:804`** | `if (final_main_vertex)` gate on the whole block |
| `:788-804` | **`:841-857`** | apa/face derivation (§2.7) |
| `:832-839` | **`:885-892`** | `singlephoton_tagger` call + `photon_flag` knob |
| `:833` | **`:886`** | `if (m_sp_photon_flag)` |
| `:835` | **`:888`** | the comment; **the `photon_flag` write is `:891`**, not `:835`+53 — §2.1's citation was already off by three |
| `:851-853` | **`:904-906`** | `cluster_fc_check` |
| `:852` | **`:905`** | the call itself (P1 / §8 anchor) |
| `:860` | **`:913`** | `if (final_main_vertex)` before `fill_kine_tree` |

Rule: anything above old `:512` shifts **+53**; `:82-95`, `:237-251`,
`:542-565` and `:945-988` are new.

### 10.13 An observation that corrects a §2 *positive*

Not promoted to a finding, but flagged because a wrong entry in §2 costs more
than a wrong entry in §3 — a false positive tells a future reader not to look.

**§2.9 says the toolkit "keeps the tightened boundary". It keeps the
tightening; it does not keep the boundary switch.** In the prototype,
passing a tolerance vector does not merely inset the volume — it switches which
polygon family is tested. `ToyFiducial::inside_fiducial_volume` with
`tolerance_vec == NULL` runs `pnpoly` against `boundary_xy_x` /
`boundary_xz_x`; with a tolerance vector it runs against
**`boundary_SCB_xy_x_array` / `boundary_SCB_xz_x_array`** — the space-charge
boundary — after temporarily shifting those arrays and reverting them. Two
different surfaces, not one surface at two insets.

The toolkit's `FiducialUtils::inside_fiducial_volume` insets **the same**
`m_sd.fiducial` in both branches (`FiducialUtils.cxx:82-84` and `:113-118`).

Kept as an observation rather than a P for a reason that should be stated:
`boundary_SCB_*` is a uBooNE-specific hard-coded array pair with no SBND
counterpart, and SBND supplies its own `IFiducial` through config. There is no
"restore the SCB family" fix available here. What the entry *does* mean is that
§2.9's parity claim is narrower than it reads: the tolerance is faithful, the
surface it is applied to is not the prototype's, and any future attempt to
reconcile uBooNE numbers against this stage has to account for that.

### 10.14 What §10 does not claim

* **Still no event was run.** Every statement above is static reading at
  `git show HEAD:`. F1's diagnostic, F2's population count, F4's M4 protocol and
  F5's disagreement counter are all *proposed*, none executed.
* **F2's population is not measured** — §10.3 separates the verified mechanism
  from the unverified frequency deliberately, and the fix must not be
  implemented before the counter runs.
* **The filter does not re-audit §0's uncovered ground.** The ~30 sub-taggers
  whose internal cut values were never compared (§7.3) are exactly as unaudited
  after §10 as before it. A wrong constant inside one of them is still not
  excluded.
* **Nothing was built and no gate was run.** Zero toolkit files were modified;
  the concurrent session's twelve dirty files were left untouched and were not
  read.
* **§10.9's coverage table is a grep of the branch-booking block**, not a run.
  It proves the branch is *booked*; it does not prove the writer is reached on
  any event.

### 10.15 Amendments from self-review

Three under-specifications in §10.2-§10.6, each of which would have let an
implementer make a decision the owner did not make.

**(a) F2's counter is two different measurements, and §10.3 named only the
cheap one.** "At tagger entry" plus a citation of the `PR30AUDIT` blocks is
ambiguous: those blocks *emit* at the end of `visit()`, while the population is
per-segment and the eight skip-gates live in two translation units. They are
not the same instrument:

* **The population sweep — run this first.** One pass over the PR graph's
  segments immediately before the tagger block (`TaggerCheckNeutrino.cxx:804`),
  counting `(flags_any(kShowerTrajectory) || flags_any(kShowerTopology)) &&
  !has_particle_info()`, emitted on the existing audit line. Answers *is the
  population non-empty*, which is the only question that decides whether F2
  exists. Cheap, one file, no sub-tagger edits.
* **The eight per-site counters — only if the sweep is non-zero.** One at each
  skip-gate (`NeutrinoTaggerNuE.cxx:396, :852, :897, :989, :1014, :4089,
  :4274`, `NeutrinoTaggerSinglePhoton.cxx:1400`). Answers *which gate loses
  segments*, which is what a fix has to be scoped against.

Stated because whoever runs it will otherwise pick the cheap one and then the
number will not attribute.

**(b) F5's `eps` was left blank, and any non-zero value silently reintroduces
the defect.** §10.6's patch reads `ray_length(...) < eps`. The prototype's test
is **integer index identity**, so a tolerance turns it back into a proximity
rule — just with a different threshold from `seg_endpoint_near`'s, which would
be replacing one proximity test with another while claiming index parity.

**Use exact equality**, `vtx->wcpt().point == seg->wcpts().front().point`. Both
are copies of the same skeleton node; nothing between the graph build and the
tagger does arithmetic on them. The assumption is checkable rather than
assumed: ship the knob with a counter for the case where **neither** endpoint
matches exactly. A non-zero count disproves the coincidence premise, and then
F5 needs redesign rather than a wider tolerance.

**(c) §2.9 now carries a forward pointer.** §10.13 narrows a §2 *positive*, and
in a 1574-line doc a correction that lives only at the end reaches nobody who
stops at §2. The pointer is inserted in §2.9 itself.

---

## §11 The implementation round — F1/F4/F5/F6/F7 SHIPPED SBND ON, F3 plumbed OFF, F2 dead by construction

**Owner instruction (2026-08-04):** repeat the series' knob round for this
doc's filtered list; validate on nueCC48; bug fixes and improvements default ON
in SBND; update the md, commit and push.  Three scoping decisions were taken by
the owner the same day (AskUserQuestion): **F1 = consistent FV, SBND ON**;
**F3 = plumb the SCE helper but keep SBND OFF** (no SBND SCE helper exists;
a separate gate bool keeps kine and single-photon independently switchable);
**F7 = implement the bitmask AND book the T_tagger branch, ON** (branch booked
only when the knob is on, so knob-off stays schema-identical).

Toolkit commit **`2457320d`** (parent `29e8e452`); wcp-porting-img
carries this doc, the runner tri-state block and the new gate driver
`pr36_cmp.py`.

### 11.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit           # at 2457320d
./wcb build --notests -p && ./wcb install --notests -p
./build/clus/wcdoctest-clus                        # 95/95 cases PASS

SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
cd $SX
# knobs-off gate arm + per-knob arms (PR_JOBS=12, pairs of arms concurrent):
PR_JOBS=12 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-nuecc48-0804 work-pr36-off48 data
SBND_NEUTRINO_CONSISTENT_FV=1     ... work-pr36-f1on48   # same shape
SBND_TAGGER_ORDERED_SEGSETS=1     ... work-pr36-f4on48
SBND_STEM_ENDPOINT_WCPT_PARITY=1  ... work-pr36-f5on48
SBND_BROKEN_MUON_CLUSTER_ID_COUNT=1 ... work-pr36-f6on48
SBND_SP_SCE_CORRECTION=1 SBND_NEUTRINO_TYPE_BITMASK=1 ... work-pr36-f3f7on48
# union of the flips, forced on the PRE-flip config:
SBND_NEUTRINO_CONSISTENT_FV=1 SBND_TAGGER_ORDERED_SEGSETS=1 \
SBND_STEM_ENDPOINT_WCPT_PARITY=1 SBND_BROKEN_MUON_CLUSTER_ID_COUNT=1 \
SBND_NEUTRINO_TYPE_BITMASK=1     ... work-pr36-allon48
# after the TLA flips, bare == production:
PR_JOBS=12 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-nuecc48-0804 work-pr36-prod48 data

python3 pr36_cmp.py work-pr35-prod48 work-pr36-off48     # the off gate
python3 pr36_cmp.py work-pr36-off48  work-pr36-<arm>     # per-knob gates
python3 pr36_cmp.py work-pr36-allon48 work-pr36-prod48   # doc-68 invariant
```

Scratch (build logs, compiled-config md5s, harvest scripts):
`/home/xqian/tmp/pr36/`.

### 11.2 What shipped, and the corrections measurement forced on §10

| F | key(s) | SBND | C++ default |
|---|---|---|---|
| F1 | `fiducial` + `fv_tolerance` (via sbnd arg `neutrino_consistent_fv`) | **ON** | absent = legacy FiducialUtils |
| F2 | — (population sweep + 11 per-gate counters only) | n/a | — |
| F3 | `sp_sce_correction` (+ geom_helper param threaded into `singlephoton_tagger`) | **OFF** (owner) | false |
| F4 | `tagger_ordered_segment_sets` | **ON** | false |
| F5 | `stem_endpoint_wcpt_parity` | **ON** | false |
| F6 | `broken_muon_cluster_id_count` | **ON** | false |
| F7 | `neutrino_type_bitmask` (threaded to BOTH `tagger_check_neutrino` and `tagger_output`) | **ON** | false |

Threading: `TaggerCheckNeutrino` gained `Clus::NeedFiducial` (the STM guard —
`NeedFiducial::configure` runs only when the key is present — is copied
verbatim); the F4-F7 switches live on `PatternAlgorithms`; counters live in
`PR::g_pr36_audit` (PRGraph.h) and are emitted per event in a `PR36AUDIT` log
line beside PR30/31/32AUDIT.  `cluster_fc_check` needed **no** change — its
`fiducial`/`fv_tolerance` parameters already exist, defaulted, with the
nullptr path documented bit-for-bit (`Clustering_Util.cxx:108-116`).

**Corrections to §10, found while implementing (each verified in source):**

* **F5 is 18 call sites, not 22** — 13 in NuE (`grep -c` counted the
  definition line too) and 5 in SinglePhoton (two hits are comments).  All 18
  pass a *vertex* fit point; none compares against a non-vertex point.
* **F4 is FOUR sums, not three** — `track_overclustering` iterates the same
  (grown) `muon_segs` set a second time for `tro_3_stem_length` (NuE `:3730`).
  All four sites got the same knob-gated index-ordered iteration.
* **The prototype's singlephoton `neutrino_type` write is commented out**
  (`NeutrinoID_singlephoton_tagger.h:536-539`), so F7 reproduces FOUR live
  writers (cosmic bit 1; numu bit 2 / nc bit 3, unconditional at numu_tagger's
  end; nue bit 5), not five.  Branch type is `/I`
  (`wire-cell-prod-nue-port.cxx:1485-1486`), init 0 (`NeutrinoID.cxx:58`).
* **The prototype's `:222` SCE site is also commented out** — F3 corrects the
  four live positions (nu vertex `:13`, proton `:103` / MIP `:132` track
  starts, shower start `:317`); `max_shw_dis`/`shw_vtx_dis` then derive from
  the corrected coordinates, matching `:318-330`.
* **The §10.3 skip-gate census was short**: NuE `:396` is a *bare*
  `!has_particle_info()` gate (no pdg clause) and three more pdg-11 gates
  exist (`:1693 :1867 :2569`), so the counters cover 11 gates (10 NuE + 1
  SP), and `NeutrinoShowerClustering` has **9** pdg-11 writers (5 `set_pdg`,
  4 fresh constructions), not 3.

### 11.3 The gate (two artifacts, as §10.9 prescribed)

New driver **`pr36_cmp.py`**: leaf-level compare of *every* tree in
`tracking-pr.root` (uproot; doubly-jagged branches canonicalized, NaN==NaN),
the calib-pr JSON with the `tagger` block sub-keyed, mabc member hashes,
pctree rollup, nusel TSVs.  `tagger_tree_ab.py` was not used as a gate — its
exit code is 0 even when branches move.  Evt 116962 has no PR ⇒ calib
comparisons are 47/47.

* **Knobs-off vs production** (`work-pr35-prod48` vs `work-pr36-off48`):
  **trees 48/48, calib 47/47, mabc 48/48, pctree 48/48, TSV 48/48** — all
  counters and the F1 both-ways diagnostic are log-only, proven.
* **Compiled config**: knobs-off md5 `95f069d516cf8ff7ecc6ed10cb179db1` ==
  worktree at `29e8e452` (identical TLAs incl. full `pipeline_names`); with
  the six TLAs forced, every key lands on `TaggerCheckNeutrino:pr` (and
  `neutrino_type_bitmask` also on `UbooneTaggerOutputVisitor:pr`), F1
  receiving exactly the STM objects: `BoxFiducial:sbnd_pr_fv`,
  `[-25,-25,-30,-30,-50,-30]` mm.
* **Doctests**: `wcdoctest-clus` 95/95 cases, 0 failures (three builds).  The
  suite's assertion total moved 984↔983 between runs; attributed by a
  stash-rebuild A/B to `pattern_recognition shower_clustering_with_nv [B]`,
  whose CHECK count is data-dependent (one per `map_vertex_to_shower` entry)
  and ASLR-order sensitive — pre-existing, not this round.
* **`allon48` (env-forced union, pre-flip config) vs `prod48` (bare,
  post-flip)**: **48/48 identical on every artifact class** (trees, calib 47/47, mabc, pctree, TSV) -- a bare runner invocation IS the production operating point.

### 11.4 F1 measured — 6/48 events, all in the same direction

The unconditional both-ways diagnostic (`PR36AUDIT match_isFC disagree`) and
the calib gate agree: **6/48** events flip, every one **contained → exiting**
(`match_isFC` 1→0) — the FiducialUtils sensitive-volume shell calling exiters
contained, exactly the failure mode TaggerCheckSTM's comment documents (96/147
on the STM 30-event sample).  Because `match_isFC` is numu XGBoost var 70,
`numu_score` moves on all six; `nue_score` moves only on 137238 (the others
sit at the +4.3009 nue cap):

| evt | match_isFC | numu_score | nue_score |
|---|---|---|---|
| 54095  | 1→0 | −1.896 → −1.524 | capped |
| 74544  | 1→0 | −0.034 → +0.157 | capped |
| 137238 | 1→0 | +1.270 → +1.159 | +0.007 → −1.063 |
| 168596 | 1→0 | −1.364 → −0.996 | capped |
| 268784 | 1→0 | −0.166 → −0.508 | capped |
| 360535 | 1→0 | −1.353 → −0.643 | capped |

Movement is confined to the calib tagger block + the two score branches of
`T_tagger`; mabc/pctree 48/48, and the nusel TSVs are 48/48 identical — **no
selection verdict flips on this sample**.  `match_isFC`'s only artifact outlet
remains the calib JSON (not booked in `T_tagger`), as §10.9 found.

### 11.5 F2 resolved: dead by construction on nueCC48

The §10.15a population sweep — one pass over the PR graph before the tagger
block — counted **0 shower-flagged, ParticleInfo-less segments in 2998
segments over all 47 PR events** (torn-line events recovered field-by-field;
every event individually 0).  All 11 per-gate counters: 0.  The prototype's
`get_particle_type()` coercion therefore has **no population to act on here**
— `NeutrinoShowerClustering`'s nine pdg-11 writers cover the shower-flagged
population totally on this sample — and F2 resolves like pr/35's P10: dead by
construction, **no knob shipped**.  The sweep and gate counters stay in
production as the tripwire; a future non-zero PR36AUDIT `f2_sweep` is the
signal to revisit (the fix design — reproduce the LATCH, not the read — is in
§10.3).

### 11.6 F5 measured — premise CONFIRMED, two legitimate firings, ON

Across all arms the 18 per-site counters show: **16 of 18 sites have an exact
wcpt match on every call** (f5_disagree = 0).  Site 17
(`low_energy_overlapping_sp`, SP) fired both counters **twice** — evts 268067
and 350186, once each.  §10.15b's rule ("a non-zero neither-match count
disproves the premise ⇒ redesign, never a tolerance") demanded classification
before any flip, so the two events were re-run with a distance diagnostic:

```
evt 268067: d_front=4.77 cm  d_back=6.84 cm
evt 350186: d_front=90.00 cm d_back=94.61 cm
```

Macroscopic distances, not arithmetic drift: the shower's start VERTEX is
genuinely a different skeleton node from either end of its start segment (the
indirectly-connected-shower population, `get_start_vertex_and_type()` type
≠ 1).  In the prototype the integer index test fails on exactly this
population and picks `back` — which is what the knob-on rule does.  **The
coincidence premise stands** (zero near-miss firings anywhere), and the two
disagreements are precisely where the prototype disagrees with the legacy
proximity substitute.  Knob-on movement: those two events only, 4
`shw_sp_lol_*` branches each (268067: `lol_2_v_angle, lol_3_angle_beam,
lol_3_min_angle, lol_3_n_out`; 350186: `lol_3_angle_beam, lol_3_flag,
lol_3_n_out, lol_flag`) — `shw_sp_*` is not a BDT input on either side (§3
P2's bound), and the scores and TSVs are untouched.  **Flipped ON.**

### 11.7 F4, F6, F3, F7 — three nulls and a schema branch

* **F4 (`tagger_ordered_segment_sets`, ON)**: knob-on arm **48/48
  byte-identical on every artifact**.  On this manifest the address order and
  the index order of the four accumulation sets produce bit-identical sums —
  the fix's value is run-to-run stability under a different address layout
  (M4 house rule; the class label of §10.5 stands: this moves *further* from
  the prototype's unreproducible order, deliberately).
* **F6 (`broken_muon_cluster_id_count`, ON)**: 48/48 byte-identical;
  `f6_id_vs_ptr_disagree` = 0 everywhere — cluster↔id is injective inside
  `broken_muon_id`'s sets on this sample, so pointer-count == id-count.  The
  knob makes the prototype's semantics explicit; the counter is the doc-53
  epoch tripwire.
* **F3 (`sp_sce_correction`, OFF — owner)**: the combined F3+F7 arm proves the
  plumbing vacuous as expected: with the knob forced ON and
  `clus_geom_helper` unset (SBND), **zero value branches moved** anywhere.
  The correction becomes reachable the day SBND configures an SCE helper —
  as its own explicit flip, not as a side effect of the kine key.
* **F7 (`neutrino_type_bitmask`, ON)**: knob-on diff = **exactly one new
  `T_tagger` branch (`neutrino_type/I`) on all 47 PR events and nothing
  else** — no value branch, no calib key, no Bee/pctree/TSV movement.
  Knob-off arms show the branch absent: schema-identical, as designed.

### 11.8 Operating point and residuals

After the TLA flips (`neutrino_consistent_fv`, `tagger_ordered_segment_sets`,
`stem_endpoint_wcpt_parity`, `broken_muon_cluster_id_count`,
`neutrino_type_bitmask` = true; `sp_sce_correction` stays false),
`wct-pr-perevt.jsonnet` compiled bare equals the TLA-forced pre-flip config,
and the bare `work-pr36-prod48` arm equals `work-pr36-allon48` (§11.3) — a
bare runner invocation IS the production operating point (doc 68 invariant).
Escape hatches: the runner tri-states (`SBND_NEUTRINO_CONSISTENT_FV=0` etc.).

**Residuals, unchanged by this round:**

* Reading (ii) for F1 — threading the upstream light-matching verdict itself —
  remains open; the shipped knob is the consistent-recompute reading (§10.2).
  The both-ways diagnostic keeps measuring the disagreement rate for free.
* §0's uncovered ground (the ~30 sub-taggers' internal cut values, §7.3)
  is exactly as unaudited as before.
* F2's tripwire logic (sweep + 11 gates) is measurement on THIS sample; MC or
  other-detector samples could populate it (watch PR36AUDIT `f2_sweep`).
* The nusel `stmfit` TSV column remains log-tearing flaky (pr/35 §11.4);
  this round's TSV comparisons happened to be clean 48/48 everywhere.
* uBooNE-trained BDT weights + the §4 calibration surface (doc pr/2 G1) are
  untouched: F1 moves an *input* to a network trained on the uBooNE
  definition of that input — the 6-event movement is definitional
  consistency, not retraining.
