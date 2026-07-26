# 57 — Where the dQ/dx numbers actually come from: what the STM tagger and the PR
chain read, and every hardcoded charge-scale constant in them

Three questions, asked after [55 §10](55_dqdx-vs-rr-three-bundles.md) put five
free-power reference tables into `nusel_display/stm_ref_dqdx.json`:

1. Does the STM tagger use that json?
2. Does the PR chain use `sbnd/particle_dataset.jsonnet`?
3. Are there hardcoded numbers in the tagger aimed at protons — inherited from
   MicroBooNE — that need the same treatment?

**Short answers: no, yes, and yes — but the third one is worse than expected in
two specific ways.** `detect_proton` never consults the proton table (§3), and
the PR chain has *no* MIP knob at all: `mip_dqdx = 56000` reaches
`TaggerCheckSTM` and nothing else (§4).

Originally a **survey**. §5a/§5a-bis have since become the record of an actual
change and its A/B:
the seven absolute-e/cm literal sites in `TaggerCheckSTM.cxx` are now written
`a x m_mip_dqdx`, bit-identical at MicroBooNE's 50000 and following the MIP
scale at SBND's 56000. Everything else here is still survey — `file:line`, the
number, and what would move it. Where provenance could not be established from
the code it says so rather than inferring it from the value.

---

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev

# 1. is the json in the config chain at all?
grep -rn "stm_ref_dqdx" toolkit/ | grep -v .cache        # -> nothing

# 2. who reads the five *DeDx tables
grep -rn "get_dEdx_function(" toolkit/clus/src toolkit/clus/inc

# 3. the constants
grep -nE "m_mip_dqdx *[<>]|/ *m_mip_dqdx *[<>]" toolkit/clus/src/TaggerCheckSTM.cxx
grep -rn "43e3\|43000" toolkit/clus/src toolkit/clus/inc | wc -l     # -> 102

# 4. the section 5a change, and the proof it is bit-identical at 50e3
cd toolkit && git show efc2c281 --stat
python3 -c "print([a*50000.0 for a in (1.45,1.7,1.85,1.5)], 0.2*50000.0/10.0)"
#   -> [72500.0, 85000.0, 92500.0, 75000.0] 1000.0   exactly the old literals
g++ -fsyntax-only -std=c++17 -DSPDLOG_COMPILED_LIB=1 -DSPDLOG_FMT_EXTERNAL=1 \
    -DEIGEN_HAS_CXX11 -DEIGEN_FFTW_DEFAULT=1 -I. -Iclus/inc -Iutil/inc \
    -Iiface/inc -Iaux/inc -Igen/inc -Iimg/inc -Ibuild/clus/inc \
    -I../local/include -I../local/include/eigen3 clus/src/TaggerCheckSTM.cxx
```

Line numbers are against the working tree as of 2026-07-26.

---

## 1. The json is **not** in the config chain

`nusel_display/stm_ref_dqdx.json` is read by exactly three Python programs, all
under `sbnd_xin/`: `nusel_scan_viewer.py`, `stmfit_showcase.py`,
`stmfit_particle_overlay.py`. Nothing in `toolkit/` opens it — it does not
appear in any `.jsonnet`, any `.cxx`, or any config the pipeline compiles.

So **doc 55 §10 changed no reconstruction behaviour whatsoever.** It changed
what a scan viewer draws and what three analysis scripts compare against. The
free-power tables are a *proposal* living next to the tables in use, which is
why §10.4 kept the `*DeDxBox` keys beside them and anchored `MIP_DQDX` on the
`*Box` muon plateau.

## 2. `particle_dataset.jsonnet` **is** in the chain — for both, with very
different coverage

```
sbnd_xin/wct-pr-perevt.jsonnet:194   local pds = (import '../particle_dataset.jsonnet')()
        :199                         particle_dataset = pds.particle_dataset
        :200                         extra_uses = pds.all          # the 10 LinterpFunctions
   -> ParticleDataSet (clus/src/ParticleDataSet.cxx) — a name -> IScalarFunction map,
      "muon"/"proton"/"pion"/"kaon"/"electron" for dedx_functions and range_functions
```

Two consumers, and they do not use the same subset:

| consumer | `get_dEdx_function` calls | file:line |
|---|---|---|
| **`TaggerCheckSTM`** | `"muon"` **only**, 3 sites | `TaggerCheckSTM.cxx:1500, 1503, 1688` |
| **PR chain** (`do_track_comp`, the track-PID comparator) | `"muon"`, `"proton"`, `"electron"` | `PRSegmentFunctions.cxx:1312–1314` |

The `pion` and `kaon` `*DeDx` tables have **no** `get_dEdx_function` caller
anywhere in `clus/`. They are configured and loaded and never read. (The five
`*Range` tables are a different story — `get_range_function` is called for all
five, and those are genuinely detector-agnostic.)

Consequence for the user's question: rebuilding the shipped `*DeDx` tables would
change **`TaggerCheckSTM` through the muon curve only**, and the PR chain's
track PID through muon + proton + electron. That is where doc 55 §9 item 1's
"moves every STM verdict" cost actually lands.

## 3. **`detect_proton` never uses the proton table**

This is the most surprising item in the survey and it goes straight at the
user's question.

`TaggerCheckSTM.cxx:1287–1542` is the function whose whole job is "is this
stopping thing a proton?" — its `true` return sets pass status 5, *rejected
proton endpoint*. It builds three reference vectors (`1495–1504`):

- `muon_ref`   — `get_dEdx_function("muon")` over rr ∈ (3, 35) cm
- `const_ref`  — the flat `m_mip_dqdx` line
- `muon_ref_p` — `get_dEdx_function("muon")` again, over rr < 20 cm

and then decides from `ks1/ks2/ks3`, `ratio1/2/3` and the peak height
`dQ_dx[max_bin]/m_mip_dqdx`. **There is no `get_dEdx_function("proton")` call in
`TaggerCheckSTM.cxx`.** A proton is identified as "a thing that fits the muon
hypothesis badly and peaks high", never as "a thing that fits the proton
hypothesis well".

Two things follow:

1. It is consistent with doc 55 §5, where all three hand-scanned bundles —
   including the one that sits at 1.14 of the proton curve and 1.91 of the muon
   curve — were accepted as stopping muons.
2. **A better proton table cannot improve proton rejection**, before or after
   §10, because that code path cannot see it. Any gain from doc 55's work would
   have to come either from the muon curve moving or from adding a proton
   hypothesis to this function. The second is a change of algorithm, not of
   constants, and is out of scope here.

## 4. The PR chain has no MIP knob — and two different MicroBooNE values

`mip_dqdx` is a config knob on **`TaggerCheckSTM` only**. `TaggerCheckSTM.cxx`
is the only file in `clus/src/` that mentions it (C++ default `50e3` at
`:406`, SBND config `56000` at `cfg/pgrapher/experiment/sbnd/clus.jsonnet:513`
and `sbnd_xin/wct-pr-perevt.jsonnet:50`).

Everywhere else the MIP scale is a **compiled-in default argument**:

| where | value | note |
|---|---|---|
| `PRSegmentFunctions.h:91, 93, 111` | `50000/units::cm` | `do_track_comp`, `segment_do_track_pid`, `segment_cal_4mom` |
| `PRSegmentFunctions.h:104, 117, 119` | `43000/units::cm` | `segment_determine_dir_track`, the two shower-direction entries |
| `PRShower.cxx:715` | `43e3/units::cm` | local `const double MIP_dQdx` |

Both are MicroBooNE numbers, and **which one you get depends on which function
you land in**. Every call site checked passes no explicit value — e.g.
`NeutrinoShowerClustering.cxx:156, 1034, 1432, 1574, 2743`,
`NeutrinoVertexFinder.cxx:500, 1224–1521` all take the default. The only
explicit overrides found pass `43000/units::cm` again
(`NeutrinoTrackShowerSep.cxx:119, 126, 139`).

Beyond the defaults, the constant is written out inline **102 times across 15
files**:

| file | sites of `43e3`/`43000` |
|---|---|
| `NeutrinoTaggerNuE.cxx` | 19 |
| `NeutrinoVertexFinder.cxx` | 17 |
| `NeutrinoTaggerCosmic.cxx` | 13 |
| `NeutrinoTrackShowerSep.cxx` | 12 |
| `NeutrinoTaggerSSM.cxx` | 10 |
| `NeutrinoTaggerSinglePhoton.cxx` | 9 |
| `NeutrinoShowerClustering.cxx` | 5 |
| `PRSegmentFunctions.h` | 5 |
| `NeutrinoTaggerNuMu.cxx` | 4 |
| `PRSegmentFunctions.cxx`, `NeutrinoPatternBase.cxx` | 2 each |
| `PRShower.cxx`, `NeutrinoStructureExaminer.cxx`, `NeutrinoOtherSegments.cxx`, `NeutrinoDeghoster.cxx` | 1 each |

**This is the single most consequential finding for the PR-chain question.**
Doc 48 raised the SBND MIP to 56000 e/cm; that raise reaches one component. The
rest of the chain still divides by 43000 or 50000. Provenance of the 43000 vs
50000 split is *not established here* — both are MicroBooNE-era, but why two
values coexist is not visible in this code.

## 5. Hardcoded constants inside `TaggerCheckSTM`, by what breaks them

### 5a. Absolute e/cm literals — **found, and converted to `a x MIP` (2026-07-26)**

`TaggerCheckSTM.cxx` had seven thresholds (five distinct values) compared
against a raw dQ/dx in e/cm,
**not** normalised by `m_mip_dqdx`, so they did not move when doc 48 set SBND's
`mip_dqdx = 56000`. They sat alongside ~33 cuts that *are* written
`.../m_mip_dqdx`, and in one case inside the *same* `if`.

**What established the scale.** Every one of the five is an **exact** multiple of
**50000 e/cm** — MicroBooNE's `MIP_dQdx`, and this class's own C++ default. Not
of the 48879 e/cm muon-table plateau, and not of the 43000 the PR direction
routines use. `energy_loss/docs/dqdx_consistency_check.md` §6.1 states the same
convention from the other side: the MicroBooNE tables "define the **scale
convention** the whole tagger is tuned on, and the hard-coded companions sit on
the same scale". The clincher is `check_other_tracks:2240–2241` (was `2209–2210`), which
spells the denominator out in source:

```cpp
else if (track_length1 < 10  && track_medium_dQ_dx <  85/50.) continue;
else if (track_length1 < 3.5 && track_medium_dQ_dx < 110/50.) continue;
```

`track_medium_dQ_dx` is already `measured / m_mip_dqdx`, so those two are
**already** in the `a x MIP` form (a = 1.7 and 2.2) — they needed no change, and
they are the direct evidence that the 50000 is what the absolute literals were
written against. They are left as written fractions, with a comment, for exactly
that reason.

**The conversion.** Each literal became `a * m_mip_dqdx` with
`a = literal / 50000`:

| file:line (new) | was | now | a | at SBND 56000 |
|---|---|---|---|---|
| `:762` `adjust_rough_path` | `min_dQ_dx < 1000` | `< 0.2 * m_mip_dqdx / units::cm` | 0.20 | 10000 → **11200** e/cm |
| `:1744` Michel veto | `ave_res_dQ_dx > 72500` | `> 1.45 * m_mip_dqdx` | 1.45 | 72500 → **81200** |
| `:1745` | `> 72500` | `> 1.45 * m_mip_dqdx` | 1.45 | 72500 → **81200** |
| `:1747` | `> 85000` | `> 1.7 * m_mip_dqdx` | 1.70 | 85000 → **95200** |
| `:1748` | `> 92500` | `> 1.85 * m_mip_dqdx` | 1.85 | 92500 → **103600** |
| `:1749` | `> 72500` | `> 1.45 * m_mip_dqdx` | 1.45 | 72500 → **81200** |
| `:2187` `check_other_tracks` | `segment_track_length_threshold(segment, 75000./units::cm)` | `1.5 * m_mip_dqdx / units::cm` | 1.50 | 75000 → **84000** |

The `/units::cm` on the first and last rows is not decoration: `min_dQ_dx` at
`:762` is `dQ/dx` with `dx` in *internal* length units, and
`segment_track_length_threshold` takes its threshold in the same units. Every
other site in this file compares a dQ/dx that has already been multiplied by
`units::cm`. Getting that wrong would have been a factor 10.

**Adopted as the default going forward (owner decision, 2026-07-26)**, on the
strength of the A/B in §5a-bis: charge thresholds in these taggers are written
`a * m_mip_dqdx`, and an absolute e/cm literal in a new or ported cut is a
defect. The same rule is stated at the `m_mip_dqdx` declaration in the source so
it is discoverable from the code, not only from here. It does **not** by itself
settle the PR chain (§4), which has no such knob to hang the convention on —
that is a separate decision.

**This is a change of convention, not a re-tuning.** `1.45 * 50000`,
`1.7 * 50000`, `1.85 * 50000`, `1.5 * 50000` and `0.2 * 50000 / units::cm` are
each *exactly* representable and equal to the old literal in IEEE double
(checked). So for any configuration that leaves `mip_dqdx` unset — every
MicroBooNE-scale config — the behaviour is **bit-identical to before**. Nothing
was re-tuned; the numbers were only given the scale they were always written on.

**It does change SBND.** All seven sites rise by 12 % (56000/50000). Five of
them are Michel-electron *vetoes* (`return false`, i.e. reject the STM
hypothesis), so raising them makes the veto slightly harder to trigger and STM
acceptance slightly looser. `:2187` feeds a track-length gate and `:762` a
kink-finding break. **A/B'd on the 30-event `d55ton` set: every output bit-identical**, and not for
lack of reach — two of the five converted terms did flip on 8 eval rows, and the
veto's OR absorbed it. Details and the caveats in §5a-bis.

The mixed-convention defect this removes was real and quantified: lines `1742`, `1751`, `1753` of the same `if` use
`ave_res_dQ_dx/m_mip_dqdx > 1.2 / 1.4 / 4.5` and have always followed `mip_dqdx`
to 56000, while their neighbours on `1744`–`1749` stayed on 50000. The two halves of one
veto had silently drifted 12 % apart relative to each other.

### 5a-bis. The A/B on the 30-event `d55ton` set — no output changed, and it was **not** for lack of reach

Two arms, same tree, same inputs, only `TaggerCheckSTM.cxx` differing
(`efc2c281^` vs `efc2c281`), both at SBND's `mip_dqdx = 56000`:

```bash
cd sbnd_xin
git -C ../../../toolkit checkout efc2c281^ -- clus/src/TaggerCheckSTM.cxx
(cd ../../../toolkit && ./wcb build -j8)          # waf is content-hash based:
                                                  # identical text => no rebuild
./run_perf54_nusel.sh base d57mip                 # -> work-*-d57mipbase
git -C ../../../toolkit checkout efc2c281 -- clus/src/TaggerCheckSTM.cxx
(cd ../../../toolkit && ./wcb build -j8 && ./wcb install)
./run_perf54_nusel.sh opt  d57mip                 # -> work-*-d57mipopt
```

Imaging and Q/L products are symlinked from `work-*-d55ton` and never
regenerated, and both arms run under `setarch x86_64 -R` (this chain is
ASLR-non-deterministic, doc 49 §4a), so any difference is the code change.
Flags are the d55ton set exactly, including `-stm-fit -mip 56000`.

**Result — 30 events, both arms `fail=0`:**

| compared | count | differing |
|---|---|---|
| per-bundle label TSVs (`nusel-evt*.tsv`, TGM/STM/FC verdicts) | 30 files | **0** |
| `T_stm_eval` rows (every KS/ratio/residual number the evaluator produced) | 56 | **0** |
| `T_stm_pass` rows (per-pass status, kink, exit/left length and dQ/dx) | 16 | **0** |
| `T_rec_charge` rows (every fitted point's q, dx, rr) | 2767 | **0** |

ROOT trees compared branch-by-branch with `uproot` at `atol = rtol = 0`, i.e.
exact equality, not "close".

**Why that is a real null and not an insensitive test.** `T_stm_eval` dumps
`res_length` (cm) and `ave_res_dqdx` (e/cm) — the two inputs of the veto that
changed — so the conditional can be replayed offline on all 56 rows:

| veto term | old fires | new fires | **flipped** |
|---|---|---|---|
| `L > 16 && ave > 1.45 MIP` | 8 | 8 | 0 |
| `L > 10 && ave > 1.45 MIP && D > −0.05` | 14 | 14 | 0 |
| `L > 10 && ave > 1.70 MIP` | 8 | 4 | **4** |
| `L > 6 && ave > 1.85 MIP` | 8 | 0 | **8** |
| `L > 6 && ave > 1.45 MIP && D > −0.05` | 14 | 14 | 0 |
| **the whole `if` (8-term OR)** | **32** | **32** | **0** |

So the change *did* reach the arithmetic: two of the five converted terms flipped
on 8 distinct rows — `res_length = 31.36 cm, ave = 97807` (1.75 MIP) and
`res_length = 16.68 cm, ave = 94813` (1.69 MIP), both of which clear 92500 but
not 1.85 × 56000 = 103600. The **OR structure absorbed it**: every one of those
rows is still vetoed by a 1.45-MIP term, because 1.75 and 1.69 MIP sit far above
both 72500 and 81200. The veto is redundant across its terms in exactly the
region where the rescaling bites.

**Caveats.** The replay covers the five Michel-veto terms only —
`adjust_rough_path`'s `min_dQ_dx` and `check_other_tracks`' length threshold are
not dumped, so for those the evidence is the end-to-end bit-identical result,
which does exercise them but cannot say how close they came. And the sample is
small: 56 eval rows and 16 pass rows over 30 events, narrowed further by doc 56's
beam-window-only default. A null on 30 events is not a licence to skip the check
on a production-scale run; it is evidence that this change does not move SBND's
current STM answers.

### 5b. Peak-over-MIP thresholds — the ones doc 55's table work moves

All inside `detect_proton`, all of the form `dQ_dx[max_bin]/m_mip_dqdx > X`
(line numbers as surveyed, before the §5a edit shifted them by ~+22):

| line | thresholds |
|---|---|
| `1519` | `> 2.3` (the gate on the whole first proton branch) |
| `1521` | `> 2.5` |
| `1523` | `< 3.0`, `< 4.0` |
| `1533` | `> 3.5` |
| `1535` | `> 4.3` |
| `1536` | `> 3.0` |

These cut on **Bragg-peak height in MIP units** — which is exactly the
peak/plateau contrast that both the field change and the recombination model
move, while the numbers themselves have never moved:

| | peak/plateau, muon | proton |
|---|---|---|
| uBooNE 0.273 kV/cm (doc 48 §3) | 2.525 | 2.664 |
| SBND 0.5 kV/cm, shipped Box (doc 48 §3) | 3.076 | 3.275 |
| SBND 0.5 kV/cm, free power (doc 55 §10.3) | **3.42** | **2.77** |

A threshold of 2.3 that was tuned when a stopping muon peaked at 2.5× MIP is a
very different cut when the same muon peaks at 3.1× or 3.4×. The proton moves
the *other* way. Nothing in this list has been re-tuned for either change.

### 5c. Everything else in `TaggerCheckSTM` that is a dQ/dx ratio

Correctly written against `m_mip_dqdx`, so they follow `mip_dqdx = 56000`
automatically — listed because they are still *tuned* at uBooNE's contrast even
though they are *scaled* correctly. **Untouched** by the §5a change; after it,
these plus the seven converted sites are the whole charge-threshold surface of
this file, all on one convention.

| line(s), **pre-§5a-edit** | what |
|---|---|
| `1354`, `1389` | Michel/delta-ray guards: `seg_dQ_dx > 0.5`, `> 0.8` |
| `1530`, `1533`, `1536` | `track_medium_dQ_dx < 1.0` |
| `1715`, `1716` | residual: `> 0.9`, `> 2.3` |
| `1720`, `1729`, `1731` | residual: `> 1.2`, `> 1.4`, `> 4.5` |
| `2185`–`2210` | `check_other_tracks`: `0.4, 0.7, 0.8, 1.5, 2, 2.5` |
| `2464`, `2481`, `2484`–`2487` | leftover-track gates: `2.0`, `1.5, 1.7, 1.8, 1.9` |

## 6. Two Modified-Box parameter sets live in the same SBND pipeline

| where | A | B | scale | used for |
|---|---|---|---|---|
| the five `*DeDx` tables (`particle_dataset.jsonnet`) | 0.93 | 0.212 | ×0.85 fudge | the reference curves the taggers compare shapes against |
| `sbnd_box_recomb` (`cfg/pgrapher/experiment/sbnd/clus.jsonnet:553`) | **1.0** | **0.255** | none | `segment_cal_4mom` — dQ/dx → energy |

Both at `Efield = 0.5`, `rho = 1.38`, `Wi = 23.6e-6`. Doc 55 already compared
the two parameter sets at both fields (commit `90ab26d`); this entry exists so
the audit is complete, not to re-derive it.

**A units trap, not a live bug.** `Gen::BoxRecombination::operator()` treats
`m_efield`, `m_b` and `m_wi` as *bare numbers* (kV/cm, (kV/cm)(g/cm²)/MeV, MeV),
but the C++ constructor defaults are in WCT units — `Efield = 500*volt/cm`,
`B = 0.212*gram/(MeV*cm2)*(kilovolt/cm)`, `Wi = 23.6*eV`. The SBND config sets
all five keys explicitly, so it is self-consistent today. A config that omitted
one would silently pick up a default in the wrong unit system.

---

## 7. What to look at, in priority order

Ordering is by *how much it changes and how cheaply it can be checked*, not by
how wrong it is.

1. ~~**The absolute e/cm literals.**~~ **Done** (§5a, toolkit `efc2c281` on
   `apply-pointcloud`): seven sites converted to `a * m_mip_dqdx`, bit-identical
   at the MicroBooNE default, +12 % at SBND. `85/50.` and `110/50.` were already
   correct and were only commented. The A/B is **done** too (§5a-bis): 30
   events, every label TSV and every `T_stm_eval`/`T_stm_pass`/`T_rec_charge`
   row identical, with the replay showing the change reached the arithmetic on
   8 rows and was absorbed by the veto's OR. Nothing outstanding on this item
   except re-checking at production scale.
2. **The PR chain's missing `mip_dqdx` knob (§4).** 102 sites and two different
   defaults. Deciding whether the PR chain should see 56000 at all is a design
   question worth answering before touching any of them — but the *inventory*
   is done and the split between 43000 and 50000 needs an explanation from
   someone who knows the uBooNE history.
3. **The peak/MIP thresholds in `detect_proton` (§5b).** These need data, not
   an edit: the right way to re-tune is a proton population (doc 55 §9 item 3),
   because the quantity they cut on moved twice and the sample here is one
   proton.
4. **Whether `detect_proton` should have a proton hypothesis at all (§3).**
   Algorithm change, not a constant. Related to doc 55 §9 item 7 (should the
   non-strong accept path see `ratio1`).
5. **The pion and kaon `*DeDx` tables have no reader (§2).** Either something
   was meant to use them, or they can stop being configured. Harmless either
   way; worth knowing.

## 8. Not audited here

Deliberately out of scope, listed so the gap is visible rather than silent:

- `TaggerCheckNeutrino`, `NeutrinoTaggerSSM`, `NeutrinoTaggerNuE`,
  `NeutrinoTaggerNuMu`, `NeutrinoTaggerCosmic`, `NeutrinoTaggerSinglePhoton`,
  `NeutrinoKinematics`, `NeutrinoEnergyReco`, `NeutrinoDeghoster`,
  `NeutrinoStructureExaminer`, `NeutrinoOtherSegments` — their `43e3` counts are
  in §4's table, but their *logic* was not read. Only the call sites were.
- The five `*Range` (range → kinetic energy) tables. Detector-agnostic by
  construction; not checked against PDG here.
- Angle, length and geometric thresholds throughout. Only charge-scale
  constants were surveyed.
- Whether any of these constants is actually *sensitive* — nothing here was
  varied and re-run. Every claim is a reading of the source.

---

Companion docs: [55](55_dqdx-vs-rr-three-bundles.md) (the reference tables and
the recombination fit), [48](48_sbnd-dqdx-tables-and-mip.md) (where the SBND
tables and `mip_dqdx = 56000` came from), [47](47_stm-bragg-reference-sbnd-retune.md)
(why the uBooNE tables had to be replaced),
[41](41_stm-fit-dump.md) (what `save_stm_fit` writes).
