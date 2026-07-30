# 6 — the `is_dir_weak()` port divergence: history, fix, validation

Status: FIXED behind `dir_weak_use_score` (C++ default false = legacy,
byte-identical; **SBND module default TRUE**, uBooNE untouched). Investigation
requested by the owner after doc pr/5 §3d.1 surfaced the divergence.

## 0. Repro block

```bash
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
# the unwired faithful port (before this fix: zero callers):
grep -rn "segment_is_dir_weak" $TK/clus/
# the knob:
wcsonnet ... sbnd/wct-pr-perevt.jsonnet | grep dir_weak_use_score   # true in the 13-stage PR job
# uBooNE off-gate:
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport/scripts
./sweep_5384.sh dirweakoff_ub 6 && ./ab_check.sh dirweakoff_ub gate3
```

## 1. What the divergence is

Prototype `ProtoSegment` keeps `bool dir_weak` **private** with NO public
getter (`prototype_base/pid/inc/WCPPID/ProtoSegment.h:175`); the only public
read accessor is the score-thresholded method
(`prototype_base/pid/src/ProtoSegment.cxx:1291-1302`):

```cpp
bool is_dir_weak(){ return
    |pdg|==13   && score > 0.07 && len >= 5cm  ||
    |pdg|==13   && score > 0.15 && len <  5cm  ||
    |pdg|==2212 && score > 0.13 && len >= 5cm  ||
    |pdg|==2212 && score > 0.27 && len <  5cm  ||
    dir_weak; }
```

The toolkit port exposed a raw getter (`PRSegment.h:98`) and read it at
**85 sites in 7 files** where the prototype calls `is_dir_weak()`. Census
(prototype side, exhaustive): 131 `is_dir_weak` uses, **zero** raw reads
outside the class — every `flag_dir_weak` string is a BDT TaggerInfo field
fed BY `is_dir_weak()`. So the substitution was a 100% blanket one with no
per-site justification possible.

**Why it bites**: `particle_score` defaults to the sentinel **100** on both
sides (`ProtoSegment.cxx:30`, `PRSegment.h:162`) and is explicitly *reset*
to 100 to invalidate a PID (`NeutrinoID_improve_vertex.h:334/353` ↔
`NeutrinoVertexFinder.cxx:2367/2400`). In the prototype every µ/p-typed
segment with an un-set or invalidated score is therefore **weak**; the raw
flag says strong. The setter side is line-identical between prototype and
toolkit (three conditions, `ProtoSegment.cxx:1583-1594` ↔
`PRSegmentFunctions.cxx:1638-1650`), and `break_segment` propagates the flag
faithfully — nothing compensates the read-side loss.

## 2. How it happened (git + docs archaeology)

- **Nov 2025 – Jan 2026**: the PR chain was ported with raw reads from the
  first commit (`3397df4f` "prepare files for Pattern Recognition" already
  contains `NeutrinoTaggerNuE.cxx`'s translation-key line
  "`sg->is_dir_weak() → sg->dir_weak()`" among mechanical renames).
- **The porter had no written warning**: neither the prototype's own docs
  (`protosegment.md:44` "Indicates if direction determination is weak" and
  ~15 identical one-line glosses) nor any toolkit doc mentions the score
  component. The thresholds appear in NO `.md` file in either tree; the
  semantics lived only in the C++ body. `porting_dictionary.md` and
  `neutrino_id_function_map.md`: zero mentions of dir_weak at all.
- **2026-03-22**: commit `7d494879` "fix various bugs" added the faithful
  `segment_is_dir_weak()` (`PRSegmentFunctions.cxx:1064`) with a correct
  docstring — and never wired a single caller (verified across the whole
  history). A fix begun and forgotten.
- **2026-04-06/09**: the translation comment (`024d0ae0`) and the review-doc
  ✅ (`4d3c9d9a`,
  `clus/docs/tagger/nue_low_energy_..._review.md:126`:
  "| `lol_2_v_flag_dir_weak` | `is_dir_weak()` | `dir_weak()` ✅ |")
  were added — **after** the faithful helper existed. The review
  pattern-matched accessor names, not semantics.
- **qlport validation never caught it**: the proto-vs-toolkit tagger logs do
  contain `flag_dir_weak` mismatches (e.g. `tagger_6529.log:911`
  cosmict_3 proto=1 vs toolkit=0), but every one is confounded by a
  `_filled` or upstream PID mismatch in the same block, so it was never
  isolated. No qlport doc mentions dir_weak.

Three toolkit sites even name a local `bool is_dir_weak = sg->dir_weak();`
(`NeutrinoVertexFinder.cxx:313`, `:2972`, `NeutrinoPatternBase.cxx:1867`) —
the two identifiers were treated as synonyms during translation.

## 3. The fix

`PatternAlgorithms` gains `m_dir_weak_use_score` (default **false**) and the
single accessor

```cpp
bool PatternAlgorithms::seg_dir_weak(SegmentPtr seg) const
{ return m_dir_weak_use_score ? segment_is_dir_weak(seg) : seg->dir_weak(); }
```

- All **85** raw read sites replaced: 67 in member functions →
  `seg_dir_weak(sg)`; 18 in the free tagger helpers (NuE 12, SinglePhoton 6)
  → `ctx.self.seg_dir_weak(sg)` (both context structs already carry
  `PatternAlgorithms& self`).
- Intentional raw reads kept: the helper's own fall-through
  (`PRSegmentFunctions.cxx:1082`), `break_segment` flag propagation
  (`:540-541`), two TRACE prints.
- Knob threading: `TaggerCheckNeutrino` config key `dir_weak_use_score`
  (member default false, `default_configuration` round-trip) →
  `pattern_algos.m_dir_weak_use_score` (same pattern as `m_perf`).
- jsonnet: `cm.tagger_check_neutrino(..., dir_weak_use_score=false)` with
  the key-suppression idiom (`common/clus.jsonnet`); SBND threads it through
  `clus_pr(...)`/`pr(...)` with **default true** (owner 2026-07-30). uBooNE
  call site untouched.
- `segment_is_dir_weak()` itself verified faithful: `segment_track_length(seg,0)`
  = prototype `get_length()` (both geometric sums over fit points);
  pinfo-null ≡ prototype `particle_type==0` (falls through to the raw flag).

## 4. Verification

- `wcdoctest-clus`: 565/565 PASS.
- **Compiled-config proof**: all 16 live job configs
  (`abtest/compile_all_cfg.sh`, after vs 743a55c6 worktree) byte-identical —
  including `sbnd_pr` (whose pipeline stops before `tagger_check_neutrino`)
  and every uBooNE-adjacent job. The 13-stage SBND PR job differs by exactly
  one line: `+ "dir_weak_use_score" : true`.
- **Knob-off C++ path** (new binary + before-tree config, evt 172230 DL):
  `mabc-pr.zip` = `5b4e8158…` — byte-identical to the pre-change pr/4 arm.
- **uBooNE off-gate PASS**: sweep label `dirweakoff_ub` vs `gate3`
  (35 nue events, ASLR off, DL off): **ZIPS 35/35 content-identical**
  (`sweep/dirweakoff_ub/hashes.txt` vs `sweep/gate3/hashes.txt`). The
  ab_check TAGGER stage reports diff=35, but that is an artifact: gate3's
  per-event dirs were disk-consolidated away, so the base-side tagger logs
  cannot be read (file-not-found on all 35). The zip content identity is the
  gate (doc 59 criterion; the tagger-log stage is A/A-unstable anyway,
  doc pr/3).
- **Knob live at runtime** (guards against a vacuous on≡off pass): gdb
  breakpoint on the accessor's first call under the SBND default config
  prints `m_dir_weak_use_score=true`.
- **Knob-on effect on SBND** (both bit-identical to their knob-off arms):
  - evt 172230 DL arm `5b4e8158…`, geometric arm `c5bfe4bf…` — **no change**.
    Consistent with pr/5: this event's decisive segments are pdg 0/11, which
    the score overlay does not touch (thresholds cover only |pdg| 13/2212).
  - evt 444187 (DL default, `nupr_evt444187_dirweak`): `0aeaf413…` —
    bit-identical to the pr/4 `defaultdl` arm. **No change** on either
    validated SBND event.

The knob-on path is therefore adopted for SBND at **zero observed cost** on
the two validated events, and it changes behavior exactly where the
prototype says it should: µ/p-typed segments with poor or sentinel scores.
The places it will matter: numuCC-like topologies (typed muons with marginal
direction fits) in vertex candidacy (`examine_main_vertex_candidate`),
conflict scoring, and the cosmic/nue/single-photon tagger BDT inputs.

## 5. What is NOT done (owner decisions)

1. **uBooNE knob-ON fidelity study**: DONE 2026-07-30 — see §6. Whether to
   adopt knob-ON as the uBooNE default (and re-baseline the gate) remains
   the owner's call.
2. The review-doc ✅ at `nue_low_energy_..._review.md:126` and the
   translation-key comment at `NeutrinoTaggerNuE.cxx:38` are corrected as
   part of this change's commits.
3. pr/5 §6's other two items (proton-template direction vote; mip_dqdx
   threading) remain open — this fix is one of the three.

## 6. uBooNE knob-ON impact study (2026-07-30, owner-requested)

### 6.0 Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport/scripts
DIR_WEAK=true ./sweep_5384.sh dirweakon_ub 6      # 35/35 rc=0
./ab_check.sh dirweakon_ub dirweakoff_ub          # ZIPS 5 mismatch, TAGGER diff=33
# per-event fidelity table (category n_diff_branches, both arms vs the SAME
# prototype references qlport/prototype/nue_5384_<sr>_<ev>.root):
python3 ./fidelity_compare.py sweep/dirweakoff_ub sweep/dirweakon_ub
```

Plumbing added for the study (committed): `uboone-mabc.jsonnet` gained a
`dir_weak_use_score=""` TLA threaded to the `cm.tagger_check_neutrino` call
(default '' = off; compiled JSON proven byte-identical to pre-change, ON
differs by exactly `+ "dir_weak_use_score" : true`); `run_one.sh` passes it
via `DIR_WEAK=true`. The uBooNE **default stays OFF** — gate3 identity is
preserved (§4's off-gate PASS still stands).

### 6.1 Results (35 nue events, ASLR off, DL off — gate conditions)

- **Reco outputs (Bee zips)**: 5/35 events change content hash
  (idx 5/6/7/14/26 = evts 6529, 6532, 6536, 6612, 6837); 30/35 stay
  byte-identical.
- **T_tagger values**: 33/35 events change somewhere (dir_weak feeds many
  BDT input branches directly). Evt 6588 has no T_tagger either arm
  (pre-existing, excluded below).
- **Prototype fidelity** (total differing branches vs prototype, summed over
  the compare tool's category tables, 34 events):
  **off = 20604 → on = 20531 (−73 net)**; per event 10 improved,
  3 worsened, 21 unchanged.
- **`*_flag_dir_weak` branches differing vs prototype**: 91 → 90. Most
  "still differ" because the *particle scores feeding the thresholds* are
  themselves already divergent upstream (PID differences) — faithful
  semantics cannot fix unfaithful inputs.
- The two large movers, both vertex-mediated:
  - **evt 6612 (−117)**: cosmic category 22→**0** differing branches, ssm
    139→**6** — the cosmict_2/cosmict_7 taggers now agree with the
    prototype; `numu_score` flips 1.031 → −1.303 (prototype −0.875), i.e.
    the numu selection decision now AGREES with the prototype.
  - **evt 6536 (+70)**: the main vertex moves (nu_x diff vs prototype
    1.0 → 43.6 cm) and everything downstream cascades; `nue_score` flips
    +4.30 → −4.30 (prototype +8.05) — the nue decision now DISAGREES.
    This event already carried large upstream divergences in both arms
    (600+ differing branches); the knob changed which side of a marginal
    vertex decision it lands on.
- BDT score changes (only events that moved): 6529 numu 1.086→0.935
  (proto −0.139), 6536 numu −0.231→−0.334 (proto −0.283) + the nue flip
  above, 6612 numu sign-flip toward proto, 6642 nue −3.931→−3.960,
  6837 numu 1.026→0.890 (proto −1.150).

### 6.2 Reading

Knob-ON is a modest net fidelity improvement on the uBooNE chain
(−73 branches, 10-vs-3 events better), with one clear selection-level win
(6612) and one selection-level regression (6536) — the regression being a
marginal-vertex coin-flip on an event that was already far from the
prototype in both arms, not a new pathology. The bulk of the residual
proto-vs-toolkit diffs are upstream of dir_weak (PID/score divergences) and
are untouched by this knob. Sweep labels kept for re-checking:
`sweep/dirweakoff_ub/`, `sweep/dirweakon_ub/` (hashes.txt +
ab_check_dirweakoff_ub.log + per-event tagger logs).
