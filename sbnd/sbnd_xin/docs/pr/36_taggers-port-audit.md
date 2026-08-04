# doc pr/36 — the taggers: a prototype↔toolkit fidelity audit

**Why.** Doc pr/27 §0 lists eight PR stages. This audits **step 8, the
taggers** — the last one, and by a wide margin the largest. It is the ninth and
final entry in the series that began with pr/28 (vertex fit + trajectory
dQ/dx) and ran through pr/29 (Steiner graph), pr/30 (proto-vertex + segment),
pr/31 (topology/PID/direction), pr/32 (neutrino vertex ID), pr/33 (EM shower
clustering), pr/34 (particle flow) and pr/35 (energy reconstruction).

**Status. AUDIT ONLY. No code was changed, and no patch is proposed.** The
owner's standing instruction for this series ("please do not change any code
yet") governs. Every finding below is a question for the owner, not a decision
taken.

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
