# doc pr/37 — what docs pr/28–36 missed: a cross-audit residual round

**Owner instruction, round 1.** *"Can you take a look at [pr/28, pr/29, pr/30,
pr/31, pr/32] to see which part that we may missed in our existing work. Here, I
am looking for major bugs, non-determinstics, and other crucial things. Please
add a new md file to summarize these for one future round. Please feel free to
look at prototype code, toolkit code to check. No need to modify the code for
now."*

**Owner instruction, round 2.** *"Can you repeat the same for [pr/32, pr/33,
pr/34, pr/35, pr/36] … and put the findings in
`37_cross-audit-residual-round.md`, commit and push."*

Nine documents, eight owner filters, seven implementation rounds. Each ends with
a "what is NOT claimed" section and most name blocks of code that were **never
opened**. Nobody had read them *together*. This document does that, and it adds
things none of them could have found from the inside.

**No toolkit C++ or jsonnet is changed by this document.** Five measurements
were run on already-installed binaries and already-existing artifacts; no build,
no knob, no new gate on new code.

> **Renumbering note.** Round 1 of this document (pushed as `63a9e17` …
> `1e71ce2`) numbered its sections by *which half found them*. Round 2 merges
> both halves into one structure organized **by topic**, at the owner's
> direction. Old → new: §1 → **§1.1** · §2 → **§3.1** · §3 → **§8** ·
> §4 → **§7** · §5 → **§6** · §6 → **§9** · §7 → **§10** · §8 → **§11** ·
> §9 → **§12**. §2, §4 and §5 are new. §7's numbers are **replaced**, not
> extended — see §7's filter note.

### What this round found, ranked

| § | finding | class | status |
|---|---|---|---|
| **2.2** | **doc pr/32 §11.2's "every arm 48/48 identical" is wrong for its knob-ON arms.** Its four SBND-production knobs move `T_tagger` on 9 events, `T_kine` on 4, `T_rec_charge` on 32, and the Bee `mc.json` + `shower_track-global.json` on 4 — **none of it visible to `pr32_cmp.py`, which imports `uproot` zero times while its docstring advertises a `T_tagger`/`T_kine` leaf compare.** The *prime-directive* claim (knobs off ⇒ byte-identical) **holds** under the wider instrument | **measured** | gate language + instrument, **not** a production defect |
| **2.3** | **doc pr/31 §12.4's "all five fixes are NULL on nueCC48" is wrong.** `shower_topo_reset` (F3, SBND DEFAULT ON) flips **23 of 501** `T_rec_charge/flag_shower` entries on evt 52672. pr/31 §12.3 names its own coverage — *"every `tracking-pr.root` `T_tagger`/`T_kine` leaf"* — **two of that file's seven trees**; the movement landed in a third | **measured** | the conclusion survives, the words do not |
| **1.1** | SBND accepts the DL vertex **47/48**; the whole traditional overall-main-vertex layer — including `compare_main_vertices_global`, verified term-for-term in pr/32 §2.5 — runs **0 times**. The DL re-ranker's 4.0 acceptance floor has **never rejected** | **measured** | closes pr/32 loose end 1 |
| **4** | **Three calibrations the prototype applies and the toolkit does not**, compounding on the same quantity, totalled by no document: `cal_corr_factor` is a stub returning **1.0** while the prototype's is live in production; `kine_nu_{x,y,z}_corr` carry the **raw** vertex; the single-photon SCE path is plumbed and off | source, **owner-accepted** | nothing proposed — recorded as one family |
| **3** | **Two by-value boundaries in the same class.** `determine_overall_main_vertex` discards its map and `main_cluster` (latent, reach 0/48); `acc_segment_id` is copied **three deep** so both π⁰ finders mint ids from the same seed and `ssm_tagger` receives 0 — while the same header passes it `int&` to `ssm_tagger` 86 lines later | source-verified; **both-finders reach 0/48 measured** | closes pr/33 loose end 2 |
| **5** | **`fill_sets` was declared out of scope by pr/33, pr/34 *and* pr/35 — and pr/34's F2, SBND production ON, is implemented by adding a new consumer of it.** `WCShower::fill_sets` has never been compared | inherited, cross-read | schedule |
| **8** | **doc pr/33's owner filter was never implemented** — 5 findings, 8 named knobs, zero code. Now that pr/36 has shipped, it is the **sole outlier of eight** | verified by grep | schedule |
| **6** | Repeat-run **floor is 0** at `2457320d` across 59 937 ROOT leaves + every archive **+ the calib dump** (which round 1 could not cover), on two independent comparators. The doctest ASLR canary does **not** reproduce: 52/52 runs at 984 | **measured** | pr/28's zero survives eight knob rounds |
| **7** | The prototype reference is branch `port` @ `53ca938`. Re-measured with **one** filter over all 26 files: **every audited file's substantive diff is print/timing instrumentation**, pr/33's included. The series' self-declared top follow-up is now **closed** | **measured** | citations survive; one-line process fix |
| **10** | The stale valfast/1000 baseline now carries **24 knobs ON plus one unconditional computational change** — and a large fraction of them are measured **null on the only manifest they have ever been gated on** | derived | schedule |

---

## Repro

```sh
# Trees and binaries this document was written against
cd /nfs/data/1/xqian/toolkit-dev/toolkit && git rev-parse --short HEAD   # 2457320d
ls -la --time-style=full-iso /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so
#   2026-08-04 20:55:43 -0700  -- unchanged before, between and after every arm below
ls -la --time-style=full-iso build/clus/wcdoctest-clus                  # 20:55:45, same build
# The prototype reference is a BRANCH, not a tag -- see sec.7:
cd /nfs/data/1/xqian/prototype-dev/wire-cell/pid
git rev-parse HEAD                      # 53ca93824306ea49fe480e8bc9e0a5dd678f81e5  (branch 'port')
git merge-base HEAD master              # a5fc0b9dc1a516c184d5063a8796619d588ab212

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# --- sec.1.1 / sec.3.1: does the traditional overall-main-vertex layer run? ----
SBND_WCT_LOGLEVEL=trace PR_JOBS=5 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr37-trace48 data
cd work-pr37-trace48
grep -l 'switching to DL vertex'          pr_evt*/wct_pr_evt*.log | wc -l   # 47
grep -l 'staying with traditional vertex' pr_evt*/wct_pr_evt*.log | wc -l   # 0
grep -h 'determine_overall_main_vertex timing'    pr_evt*/wct_pr_evt*.log | wc -l   # 0
grep -h 'determine_overall_main_vertex_DL timing' pr_evt*/wct_pr_evt*.log | wc -l   # 47
grep -h 'rerank selected cluster' pr_evt*/wct_pr_evt*.log | wc -l   # 47
grep -h 'rerank rejected'         pr_evt*/wct_pr_evt*.log | wc -l   # 0

# --- sec.3.2: do BOTH pi0 finders ever fire in one event? (pr/33 loose end 2) --
# the two finders carry distinguishable TRACE lines; this arm is the only
# trace-level 48-event capture in the tree
A=$(grep -l 'Pi0 found with mass'                    pr_evt*/wct_pr_evt*.log | sed 's|/.*||' | sort -u)
B=$(grep -l 'Pi0 (displaced vertex) found with mass' pr_evt*/wct_pr_evt*.log | sed 's|/.*||' | sort -u)
echo "$A" | grep -c .            # 11  id_pi0_with_vertex
echo "$B" | grep -c .            #  1  id_pi0_without_vertex
comm -12 <(echo "$A") <(echo "$B") | grep -c .   # 0  BOTH

# --- sec.2: RE-GATE the surviving arms with the only comparator that opens
#     tracking-pr.root.  No wire-cell run: every artifact already existed.
for pair in \
  "work-pr31-f2off48    work-pr32r2-off48"   `# pr/32 byte-identical BAR`   \
  "work-pr31r2-prod48   work-pr34-off48"     `# pr/34 BAR`                  \
  "work-pr34-prod48     work-pr35-off48"     `# pr/35 BAR`                  \
  "work-pr35-prod48     work-pr36-off48"     `# pr/36 BAR`                  \
  "work-pr30-baseHEAD   work-pr30-final"     `# pr/30 knob-ON`              \
  "work-pr31r2-off48b   work-pr31r2-allonb48"`# pr/31 knob-ON`              \
  "work-pr31r2-off48b   work-pr31r2-f3on48"  `# pr/31 F3 attribution`       \
  "work-pr32r2-off48    work-pr32r2-allon48" `# pr/32 knob-ON`              \
  "work-pr34-off48      work-pr34-allon48"   `# pr/34 knob-ON`              \
  "work-pr35-off48      work-pr35-prod48"    `# pr/35 knob-ON`              \
  "work-pr36-off48      work-pr36-allon48"   `# pr/36 CONTROL`              ; do
  python3 pr36_cmp.py $pair
done
# TRAP: work-pr31r2-allon48 and -f1on48 are the arms doc pr/31 sec.12 records as
# CRASHED (1 of 48 tracking files).  Use -allonb48 / -f1onb48, the re-runs.
grep -c uproot pr32_cmp.py        # 0 -- while its docstring line 9 promises
                                  #      "tracking-pr.root T_tagger + T_kine leaves"

# --- sec.6: the run-to-run floor at HEAD, now including the calib dump --------
setarch x86_64 -R env SBND_WCT_LOGLEVEL=debug PR_JOBS=5 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr37b-repeatA data
setarch x86_64 -R env SBND_WCT_LOGLEVEL=debug PR_JOBS=5 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr37b-repeatB data
python3 pr36_cmp.py        work-pr37b-repeatA work-pr37b-repeatB   # all classes 48/48
python3 pr37_repeat_cmp.py work-pr37b-repeatA work-pr37b-repeatB   # FLOOR = 0, 59937 leaves

# --- sec.6.3: the doctest ASLR canary (pr/36 sec.11.3's 984<->983) ------------
cd /nfs/data/1/xqian/toolkit-dev/toolkit
for i in $(seq 1 46); do             ./build/clus/wcdoctest-clus; done | grep assertions: | sort -u
for i in $(seq 1  6); do setarch x86_64 -R ./build/clus/wcdoctest-clus; done | grep assertions: | sort -u
# 52 runs, both ASLR states, one line: "assertions: 984 | 984 passed | 0 failed"

# --- sec.7: is the prototype reference edited where the audits read it? -------
# ONE filter over ALL 26 files.  The 'operator<<' exclusion matters: continuation
# lines of a print statement do NOT contain "std::cout", so without it the
# instrumentation is scored as substantive (that is how round 1 got 92 and 114).
cd /nfs/data/1/xqian/prototype-dev/wire-cell/pid
git diff --numstat a5fc0b9 HEAD | while read a d f; do
  s=$(git diff -U0 -w --ignore-blank-lines a5fc0b9 HEAD -- "$f" \
      | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)' | sed -E 's/^[+-][[:space:]]*//' \
      | grep -vE '^$|^(//|/\*|\*|\*/)' | grep -vE 'std::cout|std::endl|Clock::now|chrono' \
      | grep -vE '^<<' | wc -l)
  printf "%-42s %10s %6s\n" "$f" "+$a/-$d" "$s"; done | sort -k3 -nr

# --- sec.10: the knob count, DERIVED from cfg at HEAD (never summed from docs) --
cd /nfs/data/1/xqian/toolkit-dev/toolkit
git show HEAD:cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet \
  | grep -nE '^ {4}[a-z_0-9]+ *= *(true|false),'
# read at HEAD, not the working tree: a concurrent session commits into this tree.
```

---

## §0 Method, and how far to trust each row

Same tier convention as pr/28 §3b, carried through pr/29–36.

* Every source claim below was read **in both trees myself**, at the anchors
  printed, at `git show HEAD:` (toolkit `2457320d`) and at `pid` `53ca938`.
  Prototype anchors are taken from the **file**, never from `git diff` hunk
  offsets — that produced eight off-by-one anchors in round 1 (§11.4).
* Every count that came from a **run or a comparison** says so and prints the
  command. Nothing is inferred from "the code says it should".
* **Rows inherited from a P-list keep that list's both-readings posture** (M15,
  escalation rule 4). §3 does not: both its members are toolkit-internal
  self-inconsistencies with no prototype-vs-toolkit judgement in them, so §3.1
  carries a recommendation. §4 carries none — the owner has already accepted
  that gap and the "fix" would be the M15 trap the source document names.
* This document proposes **no patch** and changes no code. `git -C toolkit
  status --porcelain` carried only a concurrent session's untracked files
  throughout.

What this round did **not** do: it did not open any of the blocks in §9. Those
are named and ranked, not read.

---

## §1 Reachability — what the SBND operating point does not run

### §1.1 The headline — SBND does not run the layer pr/32 verified

**Symptom.** pr/32 §2.5 verifies `compare_main_vertices_global` term for term;
§2.2 and §2.3 do the same for `compare_main_vertices` and `calc_conflict_maps`.
pr/32 §7 loose end 1 then asks the question that decides what any of it is worth:

> *"How often does the DL re-ranker accept on SBND? This decides whether §2.5's
> global-ladder verification describes production or a fallback. **Nothing else
> in §4.3 can be judged without it.**"*

Nobody measured it. It is one trace-level run and no code change.

**Measured, 48 nueCC events** (`work-pr37-trace48`):

| | events |
|---|---|
| DL vertex **accepted** (`"switching to DL vertex"`) | **47** |
| DL **declined** (`"staying with traditional vertex"`) | **0** |
| `determine_overall_main_vertex` entered (its own `m_perf` timing line) | **0** |
| `determine_overall_main_vertex_DL` entered | 47 |
| DL re-rank **selected** a candidate | **47** |
| DL re-rank **rejected** (score below the 4.0 floor) | **0** |
| no PR at all | 1 — evt **116962**, as in doc pr/28 |

Two independent signals agree: the DL log line fires 47/47, and the traditional
function's own timing line — emitted unconditionally under `m_perf`, which SBND
has on — fires **zero** times.

**What this reclassifies.** `TaggerCheckNeutrino.cxx:785-791` runs the
traditional path only `if (!flag_dl_changed)` (the DL call is `:777`). So on the
SBND operating point these never execute:

| never entered on SBND | pr/32 status |
|---|---|
| `determine_overall_main_vertex` (`NeutrinoVertexFinder.cxx:3941`) | §0 in-scope, audited |
| `examine_main_vertices` (`NeutrinoPatternBase.cxx:2467`) — its only caller is `:3980` | audited |
| `check_switch_main_cluster` / `_2` (`:3355`, `:3449`) | **Tier B** |
| `compare_main_vertices_global` (`:3427` is its only caller) | **§2.5, Tier A, verified term for term** |

**What it does NOT reclassify** — stated because the easy over-reading is that
pr/32 audited dead code. It did not. These all run inside `determine_main_vertex`
(per cluster, `TaggerCheckNeutrino.cxx:708`, `:732`, `:748`), which the DL path
*requires* to have run first:

* `compare_main_vertices` (`:2803`), `compare_main_vertices_all_showers`
  (`:2763`), `examine_main_vertices_local` (`:2780`), `calc_conflict_maps`
  (`:1055`, reached from `compare_main_vertices` at `:859`);
* and pr/32's four shipped knobs are **engaged**, measured on the same run —
  evt 388 reports `f1_reads=104 f1_fit_valid=104 f1_mean_cm=0.5542 f2_gate=4
  f3_cand=50 f4_flagged=98`. They live in `improve_vertex` / `examine_direction`
  / `determine_main_vertex`, all of which run in both paths.

**Parity is not broken by any of this.** The prototype does the same thing:
`NeutrinoID.cxx:202-208` is `if (flag_dl_vtx){ if (!determine_overall_main_vertex_DL()) determine_overall_main_vertex(); }`,
and the prototype's DL body (`NeutrinoID_DL.h:134-176`) carries its own
`swap_main_cluster`, its own short/high-dQ/dx **proton retag** (`:144-147`) and
its own **long-muon cleanup** (`:159-166`) — all three faithfully ported to
`NeutrinoVertexFinder.cxx:3868`, `:3882-3901`, `:3907-3916`. Nothing is lost
relative to the prototype; what is lost is the *audit's* coverage of what
actually runs.

**Two consequences worth carrying forward.**

1. **The 4.0 acceptance floor has never rejected anything.** 47 selections, 0
   rejections. pr/32 §10.6 flagged that `W_MAIN`, `W_CLEN` and the floor "were
   not derived on SBND"; it is now measured that the floor is **not binding** on
   this manifest, i.e. it is an untested guard rather than a tuned one. That
   belongs with doc pr/2 gap G1, as pr/32 said.
2. **Anything that only fires in the traditional path is unvalidated on SBND by
   construction** — including §3.1's defect, and including any future fix aimed
   at `compare_main_vertices_global`. A round that wants to touch that layer must
   first turn DL off, and that is a different operating point.

### §1.2 Knobs ON ≠ knobs with evidence

The same shape one level up. Every measured null in pr/31, pr/34, pr/35 and
pr/36 is on the **same** 48-event nueCC sample, and a large fraction of the
production defaults have no measured effect on it:

| doc | knob | measured effect on nueCC48 |
|---|---|---|
| pr/31 | `cont_muon_dir3_30cm`, `track_comp_empty_abstain`, `reclass_preserve_4mom`, `dir_track_median_local` | **null** (§12.4). `reclass_preserve_4mom` *fires* on 47/48 and still moves nothing |
| pr/31 | `shower_topo_reset` | **NOT null — 1/48**, §2.3 below. Reported null by its own round |
| pr/34 | `pf_shower_vertex_barrier` | 5/48 (`mc.json`) |
| pr/34 | `pf_shower_parent_precedence` | 6/48 (`mc.json`); union with the above = 9/48 |
| pr/34 | `pf_track_main_cluster_only` | **null** — 0 of 63 BFS-claimed track nodes |
| pr/34 | `pf_pi0_node_per_id` | **null** — 13 π⁰ nodes / 12 events, no split id |
| pr/34 | `pf_pdg_name_prototype_fallback` | **null** — every PDG here is in the 11-entry table |
| pr/35 | `kine_shower_pdg_live` | 2/48 (`T_kine` + calib JSON); no BDT-visible move |
| pr/36 | `neutrino_consistent_fv` | 6/48 `numu_score`, 1/48 `nue_score`; **no verdict flips** |
| pr/36 | `stem_endpoint_wcpt_parity` | 2/48, 4 `shw_sp_lol_*` branches each |
| pr/36 | `tagger_ordered_segment_sets`, `broken_muon_cluster_id_count` | **null** — 48/48 byte-identical |
| pr/36 | `neutrino_type_bitmask` | schema-only — one new `T_tagger` branch, 47 events |
| pr/36 | `sp_sce_correction` | **vacuous by construction** — SBND sets no SCE helper |

pr/34's own caveat is the right reading and generalizes: *"the code paths run …
but **the classes they fix are empty here**."* The debt in §10 is therefore
sharper than "the baseline is stale" — for most of these knobs the valfast run
is not a re-confirmation, it is **the first evidence they will ever have**.

### §1.3 Dead by construction — the series' quiet census

Four separate rounds independently found an accepted finding with **no
population at all**. Collected because the pattern is now strong enough to be a
prior, not a surprise:

| | measurement |
|---|---|
| pr/32 P7 | 0 of **2219** |
| pr/35 P10 | `!fits[i].valid()` is **never true** — dead by construction, proven not measured |
| pr/36 F2 | 0 shower-flagged, `ParticleInfo`-less segments in **2998** segments over 47 events; all 11 per-gate counters 0 |
| pr/33 P6 | second site provably unobservable — `(void)n_showers;` |
| §3.2 here | both π⁰ finders in one event: **0 of 48** |

pr/36 §10.1's re-rank is the cautionary half: F2 was promoted to *"the finding
with the largest plausible affected population"* on a source argument and then
measured at zero. Source-level plausibility has been a poor predictor of
population in this series, in both directions — §2.3 is the case where it
under-predicted.

---

## §2 The gate is a different instrument every round

This is the round-2 headline, and it is the one thing that could only come from
reading the nine documents together: **each round discovered that its
predecessor's comparator was blind to the artifact its own stage moved, fixed it
for itself, and never went back.**

### §2.1 The four comparators

| round | comparator | what it opens | what it found out about its predecessor |
|---|---|---|---|
| pr/30 | `scripts/analysis/misc/pr_arm_compare.py` | per-event product hashes | — |
| pr/29–32 | `pr32_cmp.py` | `pctree-pr` member content, the two nusel TSVs, log counters | — |
| pr/34 | `pr34_cmp.py` | + all `mabc-pr.zip` members, `mc.json` broken out | §10.7: *"That gate is vacuous here … Quoting '48/48 byte-identical' from such a run would report a gate that **tested nothing**."* |
| pr/35 | `pr35_cmp.py` | + `calib-pr-evt<ID>.json` | §10.6: *"the standard gate fails it for the **opposite** reason: `KineInfo` is not in the pctree at all."* |
| pr/36 | `pr36_cmp.py` | + **every branch of every tree in `tracking-pr.root`** | §10.9: *"a third variant"*. And §11.3 discards its own §10.9 prescription: *"`tagger_tree_ab.py` was not used as a gate — its exit code is 0 even when branches move."* |

**`pr32_cmp.py`'s docstring promises what its code does not deliver.** Line 9
reads `3. tracking-pr.root T_tagger + T_kine leaves -- the physics scores`, and
`grep -c uproot pr32_cmp.py` returns **0**. It also never opens `mabc-pr.zip`.
So every "48/48 byte-identical" produced by it — pr/29, pr/31 and pr/32 — is a
statement about the pctree rollup and two TSVs. pr/31 §12.3 also claims *"every
`tracking-pr.root` `T_tagger`/`T_kine` leaf identical"*, which `pr32_cmp.py`
cannot produce, so a second tool was used there; pr/36 §11.3's warning that
`tagger_tree_ab.py`'s *"exit code is 0 even when branches move"* is the obvious
hazard, but which tool pr/31 ran is not recorded and is not asserted here.

**The ROOT file was in every arm the whole time.** `tracking-pr.root` is present
in `work-pr30-*`, `work-pr31r2-*`, `work-pr32r2-*`, `work-pr34-*`, `work-pr35-*`
and `work-pr36-*`. The first time anything opened it (pr/36 §11.4) it found
`numu_score` moving on 6/48 and `nue_score` on 1/48 — a class no earlier gate
could see. **M1 tests the rest**, with no wire-cell run.

### §2.2 The re-gate — what holds, and the one sentence that does not

Every surviving 48-event pair, re-compared with `pr36_cmp.py`. **The
byte-identical bar — the prime directive — holds everywhere it can be tested:**

| the bar (knob OFF vs the previous round's production) | trees | mabc | pctree | TSV |
|---|---|---|---|---|
| `work-pr30-baseHEAD` → `work-pr30-final` (pr/30's own knob-off gate arm) | **48/48** | **48/48** | 48/48 | 48/48 |
| `work-pr31-f2off48` → `work-pr32r2-off48` | **48/48** | **48/48** | 48/48 | 48/48 |
| `work-pr31r2-prod48` → `work-pr34-off48` | **48/48** | **48/48** | 48/48 | 48/48 |
| `work-pr34-prod48` → `work-pr35-off48` | **48/48** | **48/48** | 48/48 | 45/48 † |
| `work-pr35-prod48` → `work-pr36-off48` | **48/48** | **48/48** | 48/48 | 48/48 (calib 47/47) |

† the three are `stmfit`-column-only and are pr/35 §11.4's own recorded WCT
log-line tearing, not a regression — reproduced here independently.

**So the thing that matters most was never in doubt.** Five rounds' knob-off
paths — pr/30, pr/32, pr/34, pr/35 and pr/36 — are byte-identical to their
baselines under an instrument strictly wider than any of them used, including all
seven trees of `tracking-pr.root` and every `mabc-pr.zip` member. Say this first,
because the two corrections below are about *language and instruments*, not about
production output.

The knob-ON arms are where a claim breaks:

| knob-ON pair | trees | mabc | the round said |
|---|---|---|---|
| pr/31 `off48b` → `allonb48` | **47/48** | 48/48 | *"all five fixes are NULL"* — **§2.3** |
| pr/32 `off48` → `allon48` | **16/48** | **44/48** | *"every arm 48/48 identical"* — **wrong sentence** |
| pr/34 `off48` → `allon48` | 48/48 | 39/48 (`mc.json` only) | display-only — **confirmed** |
| pr/35 `off48` → `prod48` | 46/48 (`T_kine`, 2 evts) | 48/48 | 2/48 — **confirmed** |
| pr/36 `off48` → `allon48` **(control)** | as §11.4 | 48/48 | **reproduces exactly** |

**pr/32 in detail, and read the class carefully.** Its four knobs
(`vertex_dir_use_fit_point`, `shower_traj_recheck_parity`,
`main_vertex_require_descriptor`, `main_vertex_candidate_flag`) are all SBND
production defaults, and off-vs-allon moves:

* `T_rec_charge` on **32** events, `T_tagger` on **9** (1, 1, 3, 6, 9, 13, 64,
  109 and **193** branches), `T_kine` on **4**;
* `mabc-pr.zip` on **4** events — 388, 38856, 131357, 489330 — where both
  `data/0/0-mc.json` and `data/0/0-shower_track-global.json` change. Verified
  independently by plain `sha256` of the extracted zip members, not only through
  the comparator.

**This is not a defect in the knobs.** They are behaviour changes with a measured
0.613 cm mean fit-to-wcpt vertex gap; moving the vertex is what they are *for*,
and moving `T_tagger`, `T_kine` and the Bee panel is the expected consequence.
The defect is that **doc pr/32 §11 opens by saying "every arm is byte-identical
(§11.2), and every knob is nonetheless [engaged]"** — a sentence that reads as
"these knobs are engaged and change nothing", which is false, and which was
produced by a comparator whose docstring claimed a coverage its code did not
have. pr/36 §10.13 is this document's own precedent for why that costs more than
an ordinary error: *"a false positive tells a future reader not to look."*

**pr/29's bar is not recoverable.** Only single-event arms survive
(`work-pr29-388-*`); there is no 48-event pair to re-gate. Stated as a limit, not
omitted.

### §2.3 pr/31's F3 is not null

**Symptom.** doc pr/31 §12.4 is titled *"Per-knob ON arms — all five fixes are
NULL on nueCC48"*, and its §12 table records `shower_topo_reset` as
*"null — bit-identical"*. That knob is a **SBND production default, ON**.

**Measured.** `work-pr31r2-off48b` vs `work-pr31r2-allonb48`:
`T_rec_charge/flag_shower` differs on evt **52672** — **23 of 501 entries**,
contiguous indices 125–147, every one `1 → 0`. Everything else is identical:
mabc 48/48, pctree 48/48, both nusel TSVs 48/48, and all six other trees.

Attributed to **F3 alone**: `work-pr31r2-off48b` vs `work-pr31r2-f3on48`
reproduces exactly the same single branch on the same event; `f1onb48`, `f2on48`,
`f4on48`, `f5on48`, `f6on48`, `f7on48` are each 48/48.

**No binary confound.** `work-pr31r2-off48` vs `work-pr31r2-off48b` — the
§12.5-rework identity pair — is **48/48 on every artifact class including all
seven trees**, so both off arms are the same generation as `allonb48`.

**Why it hid.** pr/31 did open the ROOT file, and §12.3 names its coverage
precisely: *"every `tracking-pr.root` `T_tagger`/`T_kine` leaf identical"*. That
is **two of the file's seven trees** (`T_bad_ch`, `Trun`, `T_proj_data`,
`T_rec_charge`, `T_proj`, `T_tagger`, `T_kine`). The movement landed in a third.
This is not "they never looked" — it is a stated, partial coverage that the
summary line then rounded up to "bit-identical".

**What survives and what does not.** pr/31's *conclusion* — that these five
fixes have no physics consequence on nueCC48 — survives: no score, no selection
verdict, no Bee member, no pctree entry and no TSV moves. The words *"null"* and
*"bit-identical"* do not. Correct the §12 table row; the flip itself needs no
revisiting.

### §2.4 The consequence for the campaign

The valfast/1000 gate in §10 must be run with a comparator that opens **all**
four artifact families — `tracking-pr.root` leaves, `mabc-pr.zip` members,
`pctree-pr` member content, and the nusel TSVs — i.e. `pr36_cmp.py` or its
successor, never `pr32_cmp.py`. Four of the 24 production knobs were gated with
an instrument that could not see two of those four families.

---

## §3 By-value boundaries — the same oversight twice, once latent, once live

Two parameters whose *type* matches the prototype's member and whose *binding*
does not. Both are toolkit-internal self-inconsistencies with no
prototype-vs-toolkit judgement in them, so M15 does not bind them.

### §3.1 `determine_overall_main_vertex` discards its own work — latent

**Symptom.** Two adjacent functions in one file carry the same two arguments with
**opposite conventions**:

| | `map_cluster_main_vertices` | `main_cluster` | writes propagate? |
|---|---|---|---|
| `determine_overall_main_vertex_DL` (`NeutrinoVertexFinder.cxx:3488-3491`) | `ClusterVertexMap&` | `Facade::Cluster*&` | **yes** |
| `determine_overall_main_vertex` (`:3941`) | `ClusterVertexMap` *(by value)* | `Facade::Cluster*` *(by value)* | **no** |

**Root cause.** Inside the by-value one:

```cpp
// NeutrinoVertexFinder.cxx:3980  -- erases from the map, and can itself swap
examine_main_vertices(graph, map_cluster_main_vertices, main_cluster, other_clusters);
// :3986 (dev chain) / :3996 (frozen chain) -- assigns a LOCAL copy
main_cluster = check_switch_main_cluster  (graph, map_cluster_main_vertices, main_cluster, …);
main_cluster = check_switch_main_cluster_2(graph, temp_main_vertex, max_length_cluster, main_cluster, …);
```

and the function returns only a `VertexPtr`. `TaggerCheckNeutrino.cxx` declares
`Cluster* main_cluster` at `:386` and **never reassigns it after `:492`**.

What makes it a defect rather than dead plumbing is that the swap is only *half*
discarded. `swap_main_cluster` (`NeutrinoPatternBase.cxx:2444-2465`) mutates
state that outlives the copy:

* `other_clusters` is `std::vector<Facade::Cluster*>&` **by reference the whole
  way down** — the old main cluster is pushed in, the new one erased;
* `Facade::Flags::main_cluster` is cleared on the old facade and set on the new.

So after a swap the flag and `other_clusters` say cluster **B** is main while the
caller's pointer still says **A** — and A is now *also* inside `other_clusters`.
`TaggerCheckNeutrino.cxx:813` then runs
`improve_vertex(*pr_graph, *main_cluster, final_main_vertex, …)` on a mismatched
(cluster, vertex) pair, and `:818` files **B's vertex under A's key**
(`map_cluster_main_vertices[main_cluster] = final_main_vertex`).

The prototype has no such boundary: `map_cluster_main_vertices` and
`main_cluster` are `NeutrinoID` **members**, so
`check_switch_main_cluster` → `swap_main_cluster` (`NeutrinoID.cxx:972`) is
persistent for the whole downstream chain by construction.

**Reach — measured.** §1.1 shows `determine_overall_main_vertex` is entered
**0 times in 48 events** at the SBND operating point, and neither
`check_switch_main_cluster` nor `_2` announced a switch. **On SBND production
this defect cannot fire.** It becomes live the moment `dl_weights` is empty or
the DL path declines — i.e. in exactly the configuration a future audit of that
layer would use.

**Why it hid.** pr/32 §0 puts `determine_overall_main_vertex` in its function map
(row 14) but classifies `check_switch_main_cluster{,_2}` **Tier B** — signature
and constants compared, bodies not read end to end. A parameter's `&` is
precisely the kind of thing a signature comparison passes over: the *names* and
*types* match the prototype's members; only the binding differs.

**Fix (recommended, not applied).** Change the two parameters to
`ClusterVertexMap&` and `Facade::Cluster*&`, matching the DL twin one function
above. Byte-identical when it does not fire — which, per §1.1, is every SBND
event on this manifest — so the gate is cheap and the knob question does not
arise. **Do not** make this change without first re-reading §9 row 1: the same
round should decide whether that layer is worth keeping at all.

**The class sweep — 10 by-value sites, exactly one live.** Each was resolved by
asking (a) does the prototype's body write the member, and (b) does anything
downstream read that write. The prototype's complete write list is five lines
(`grep -rn map_cluster_main_vertices src/ inc/`), which is what makes it short.

Declaration lines are `NeutrinoPatternBase.h` at **`2457320d`** — pr/36 inserted
into this header, so round 1's numbers for it are ~34 lines low.

| toolkit site | prototype writes the member? | verdict |
|---|---|---|
| `determine_overall_main_vertex:842` | **yes** — via `examine_main_vertices` (`NeutrinoID.cxx:455`, `:514` erase) **and** `check_switch_main_cluster` (`:972` swap) | **LIVE defect**, unreachable at the SBND operating point |
| `examine_main_vertices:837` | yes | takes `Facade::Cluster*&` and `ClusterVertexMap&` — **correct**; its writes die one frame up |
| `check_switch_main_cluster:840`, `_2:841` | yes | **correct as written** — each *returns* the new `main_cluster`; the loss is the caller's binding |
| `deghost_segments:876` | **no** — `NeutrinoID_deghost.h:187-188` only *reads*; the erase at `:38` is in `deghosting()`, and the toolkit's `deghosting:877` is **by reference** | **inert** (a wasted copy) |
| `shower_clustering_in_other_clusters:864`, `examine_shower_1:865`, `examine_showers:866`, `id_pi0_with_vertex:867`, `id_pi0_without_vertex:868`, `shower_clustering_with_nv:869` | **no** — the only prototype reference in that file is a read-only iteration, `NeutrinoID_shower_clustering.h:1454` | **inert** (6 wasted copies) |

The six shower-clustering copies are a performance smell, not a correctness one,
and are **not** proposed for change — **for `ClusterVertexMap`**. Three of those
same signatures carry a *second* by-value parameter, `int acc_segment_id`, and
that one is not inert: §3.2.

### §3.2 `acc_segment_id` — the same class, copied three deep, and live

pr/33 P3 found this and the owner kept it as part of F3. **pr/33 was never
implemented (§8), so it is live in production today**, and no later document
files it as its own finding — pr/34 §10.8c only notes in passing that P9's
*"inherited half (pr/33 P3's `acc_segment_id` by value) remains pr/33's F3."*

The prototype has `int acc_segment_id;` as a `NeutrinoID` **member**
(`NeutrinoID.h:1982`) — the global segment-id allocator. The toolkit copies it
three times, verified at `2457320d`:

```
TaggerCheckNeutrino.cxx:672   int acc_segment_id = 0;                       // local
                       :839   shower_clustering_with_nv(acc_segment_id, …)   // copy 1
NeutrinoShowerClustering.cxx:3179  void …shower_clustering_with_nv(int acc_segment_id, …)
                       :3304  id_pi0_with_vertex   (acc_segment_id, …)       // copy 2
                       :2458  void …id_pi0_with_vertex(int acc_segment_id, …)
                       :2731      int pio_id = acc_segment_id++;             // dies here
                       :3313  id_pi0_without_vertex(acc_segment_id, …)       // copy 3, SAME seed
                       :2802  void …id_pi0_without_vertex(int acc_segment_id, …)
                       :3133      int pio_id = acc_segment_id;  :3134 acc_segment_id++;
TaggerCheckNeutrino.cxx:919   ssm_tagger(…, acc_segment_id, …)               // still 0
NeutrinoTaggerSSM.cxx:300     int& acc_segment_id                            // BY REFERENCE
                     :307     int temp_acc = acc_segment_id;
                     :382/:390/:393/:408   … temp_acc++ …  fill_ssmsp_pseudo_{1,2,3}
                     :414     acc_segment_id = temp_acc;                     // and writes back
```

**The asymmetry is inside one header.** `NeutrinoPatternBase.h:869` declares
`shower_clustering_with_nv(int acc_segment_id, …)`; `:955` declares
`ssm_tagger(…, int& acc_segment_id, …)`. Eighty-six lines apart, same variable,
opposite bindings — the same shape as §3.1.

**Two unconditional consequences, and their reaches now measured.**

1. **Both π⁰ finders mint ids from the same seed.** `id_pi0_with_vertex` and
   `id_pi0_without_vertex` are called back to back with the same unchanged copy,
   while `map_shower_pio_id`, `map_pio_id_showers`, `map_pio_id_mass` and
   `map_pio_id_saved_pair` are all passed **by reference** and shared. If both
   fire, both allocate `pio_id = 0` and the second overwrites the first's mass
   entry while its showers accumulate under one id.

   **Reach — measured, closing pr/33 §7 loose end 2** (*"How often do both
   finders fire? Not measured. One `SPDLOG` line in each finder would answer
   it."* — the lines already exist, at TRACE, and `work-pr37-trace48` is a
   trace-level 48-event capture):

   | | events |
   |---|---|
   | `id_pi0_with_vertex` fired (`"Pi0 found with mass"`) | **11** (12 π⁰s; evt 269774 found two) |
   | `id_pi0_without_vertex` fired (`"Pi0 (displaced vertex) found with mass"`) | **1** (evt 235435) |
   | **both fired in one event** | **0** |

   So consequence 1 is **latent on this manifest** — the same status as §3.1,
   and for the same reason: the population that would trigger it does not occur
   in 48 nueCC events. Within one finder the increments do work (evt 269774's two
   π⁰s get 0 and 1), because the copy is that function's own.

2. **`ssm_tagger` receives 0, unconditionally.** This one has no reach question:
   `acc_segment_id` reaches `:919` still holding its initial value on **every**
   event, so every `ssmsp_id` / `ssmsp_mother` is offset from the prototype's by
   the number of π⁰s found. Those branches are booked in `T_tagger` and moved in
   §2.3's arm listing, so they are published, not internal.

**Not proposed here, and one half cannot be.** pr/33 §10.4 already worked out
that widening `shower_clustering_with_nv` to `int&` *"leaks the π⁰ increments
into `ssm_tagger` … an unconditional output change with the knob off"*, and that
the deeper half is unfixable as stated: *"the toolkit has no global segment-id
allocator to hook … Restoring the prototype's invariant would mean inventing
one. Recorded as a gap."* Both halves belong to pr/33's implementation round
(§8), not to a patch written here.

---

## §4 Calibrations the prototype applies and the toolkit does not

Three of them, in three different documents, each recorded and then left. No
document adds them up, and they compound on the same quantity. **Nothing here is
proposed** — the owner accepted the largest one explicitly, and reproducing
uBooNE's calibration on SBND would be the M15 trap pr/35 names in its own text.
This section exists so the three are in one place with one total.

| | toolkit at `2457320d` | prototype | state |
|---|---|---|---|
| **charge position correction** (pr/35 P2/F2) | `cal_corr_factor` is a stub returning **1.0** (`NeutrinoEnergyReco.cxx:14-34`), called for every charge hit on both the shower and segment paths (`:221`, `:299`, `:322`) | `NeutrinoID_energy_reco.h:255-272` — three `TGraph`s, **live in production**, established four ways: `flag_calib_corr=1` by default, the production invocations pass no `-q`, the `-port` variant carries the same guard, and all three calibration files exist in the checkout | **owner-accepted gap.** No code, no knob; §11.3: *"The stub keeps returning 1.0; the magnitude on SBND remains unestimated."* |
| **kine neutrino-vertex SCE** (pr/35 P5/F4) | `clus_geom_helper` defaults `""` and SBND never sets it, so `kine_nu_{x,y,z}_corr` publish the **raw** fitted vertex under a `_corr` name | applied unconditionally (`wire-cell-prod-nue.cxx:197`) | shipped as an **unconditional runtime WARN**, firing 46/48. *"the name still lies"*; renaming a published branch is a downstream break and was not proposed |
| **single-photon SCE** (pr/36 P2/F3) | `sp_sce_correction = false` — plumbed, **SBND OFF at the owner's direction**, and measured vacuous (no SCE helper exists to call) | four live sites in `_singlephoton_tagger.h` | knob exists, deliberately off, reachable the day SBND configures a helper |

pr/35's own sentence is the total: *"So every prototype `kine_charge` carries a
per-point, per-wire calibration factor; every toolkit `kine_charge` carries
1.0."* And its §4: the toolkit's absolute energy scale already differs from the
prototype's by the recombination ratios **by design**, so *"P2's missing position
correction sits on top of that, **uncontrolled**."*

pr/36 §3 P2 draws the distinction worth keeping: the SCE gaps are *"a **config**
gap with a wired-up path; this one is **unconditional** — there is no config
that would turn it on."*

**What this section does not claim.** Not that SBND needs any of the three; not
what their magnitudes would be; not that the prototype's uBooNE graphs are the
right answer. Only that three independent unapplied corrections stack on the
same reconstructed energy, that two of them are invisible at runtime, and that
no single document says so.

---

## §5 The shared holes — code every round declared someone else's

### §5.1 `fill_sets`

Three consecutive rounds put the same file out of scope, each for a defensible
reason, and the union is a hole under a shipped production default.

* **pr/33 §0** — `PRShower.cxx` (1273 lines) ↔ `WCShower.cxx` (756) is out of
  scope except `fill_maps` and `complete_structure_with_start_segment`:
  *"A divergence could hide in the other ~1800 lines."*
* **pr/34 §0** lists `fill_sets` under "not audited"; **§9** is explicit —
  *"`PRShower::fill_sets` and `WCShower::fill_sets` **were not compared**. Both
  sides call them to enumerate a shower's vertices and segments, and **P2 and
  P11 both depend on what they return. If they disagree, those two findings
  change shape.**"*
* **pr/35 §0** — *"**The rest of `PRShower.cxx`** — still largely unread."*

And **pr/34's F2 is implemented by adding a new consumer of exactly that
function.** §10.3: the loop *"already calls `fill_sets(sv, ss, false)` per shower
and currently **discards `sv`**. Collect it alongside `shower_segs`."* That knob
— `pf_shower_vertex_barrier`, `wct-pr-perevt.jsonnet:748` — is **SBND production
ON** and moves 5/48 events. pr/34 §10.4 *did* open `PRShower.cxx:410` to clear an
implementation hazard, but toolkit-side only; `WCShower::fill_sets` has still
never been read.

pr/34's own closure ledger mislabels this: §11.8 says *"§7.3 … remains open"*,
but §7.3 is P10's frequency, which §11.4's F4 row *did* answer. The genuinely
open item lives in §9 and §10.10, and it is easy to lose.

### §5.2 The tagger interior

pr/36 §9 says the thing no earlier round had to: *"**Coverage is not exhaustive,
and this is the first round in the series where that has to be said.**"* About
**30 sub-taggers** had their field writes and dispatch verified but **not their
internal cut values** — *"a transposed constant inside `track_overclustering` or
`bad_reconstruction_2` would pass every check in this doc"* — and the 15 in-tree
`clus/docs/tagger/*_review.md` (~230 KB) were deliberately not reconciled,
because pr/33 had just found an in-tree review doc whose proposed fix *created* a
divergence. §10.14 confirms nothing changed: *"The ~30 sub-taggers … are exactly
as unaudited after §10 as before it."*

By volume this is the largest unaudited block in the series, and it sits directly
upstream of the BDT scores that decide selection.

### §5.3 The app nobody has read

§7's regenerated table sizes it for the first time:
`apps/wire-cell-prod-nue-port.cxx` is **+3384/−0** with **2844 substantive
lines** — by a wide margin the largest thing on the prototype's `port` branch,
and it is the app the SBND runner scripts invoke. Round 1 demoted it to a "not
claimed" bullet; it is now measured, so it is ranked in §9 instead.

---

## §6 Determinism

pr/28 §10, §11, §14 and §15 did the heavy work. Three things were checked here:
whether eight knob rounds have introduced anything new, whether the zero floor
still holds at a HEAD those rounds have moved, and whether the one loose
nondeterminism thread in the series is live.

### §6.1 Source sweep — nothing new to fix

* **Raw BGL traversal.** Every live `boost::out_edges` / `boost::edges` /
  `boost::vertices` outside the `ordered_*` / `sorted_out_edges` helpers is
  either a helper definition (`PRGraphType.cxx:7,19`, `PRTrajectoryView.h:155`),
  a debug dumper (`PatternDebugIO.cxx:47`), or JSON export
  (`Facade_Util.cxx:675,695`). Everything else in the grep is a comment. pr/36
  §6 independently swept the six tagger TUs and found *"**zero** raw … iteration
  survives anywhere in the stage"*.
* **`std::unordered_set<SegmentPtr>` / `<VertexPtr>`** at `PRShower.cxx:495-585`
  — the class that bit pr/28 round 9. All three sites are **membership-only** and
  copy into an `IndexedSegmentSet` / `IndexedVertexSet` on return. Compliant.
* **`clustering_deghost.cxx:233`** — an argmax over
  `unordered_map<const Cluster*,int>` with an explicit insertion-index tie-break
  and the reason in-code. Correct.
* **`SteinerGrapher.cxx:1010`** — a raw `boost::edges(base_graph)` feeding a
  strict-`<` best-edge selection, so a tie keeps the first edge seen. Not a
  nondeterminism: `Graphs.h:22-27` declares the edge list **`vecS`**,
  insertion-ordered. Recorded as a **fidelity** note, not a determinism one.
* **Handled by pr/35 and pr/36.** pr/35 P3 (an address-ordered float
  accumulation in `calculate_kinematics`) was **already fixed at HEAD** by
  `026a7501`, whose in-code comment records the measurement it closed
  (`kine_dQdx` moving by 6.9e-16 relative between two `setarch -R` runs). pr/36
  F4 swapped **four** address-ordered `std::set<SegmentPtr>` float sums in
  `NeutrinoTaggerNuE.cxx` to index-ordered sets — and it is worth repeating
  pr/36's own label: that fix moves the toolkit **further** from the prototype's
  order, and is justified by the CLAUDE.md §2 house rule, **not** by fidelity.

### §6.2 The floor at `2457320d`

Round 1 measured the floor at `29e8e452`; pr/36's five knobs have landed since,
and round 1's arms did not emit the display dump (its own §9 named the gap).
Re-measured with `PR_EXTRA_STAGES=pr_display`: two arms, same binary
(`libWireCellClus.so` mtime `20:55:43`, recorded **before, between and after**
both arms and unchanged), `setarch x86_64 -R` (M4), 48 nueCC events,
`PR_JOBS=5`, both 48/48 `rc=0`:

| artifact class | compared | differ |
|---|---|---|
| `tracking-pr.root` — every branch of every tree (`pr36_cmp.py`) | 48 events × 7 trees | **0** |
| `calib-pr-evt<ID>.json` — **new this round** | 47 (evt 116962 has no PR) | **0** |
| `mabc-pr.zip` + `pctree-pr-evt<ID>.tar.gz` member **content** hashes | 96 | **0** |
| `nusel-evt<ID>.tsv` — exact bytes | 48 | **0** |
| `tracking-pr.root` leaves, exact-bit, second instrument (`pr37_repeat_cmp.py`) | **59 937** | **0** |

**FLOOR = 0**, on two independent comparators. `work-pr37b-repeatA` vs
`work-pr37b-repeatB`. The leaf count moved 59 890 → 59 937 because pr/36's
`neutrino_type` branch adds one per PR event.

### §6.3 The doctest canary — the connection nobody made, and it does not hold

pr/36 §11.3 records, as an aside in a residuals list, that `wcdoctest-clus`'s
assertion total *"moved 984 ↔ 983 between runs; attributed by a stash-rebuild A/B
to `pattern_recognition shower_clustering_with_nv [B]`, whose CHECK count is
data-dependent … and **ASLR-order sensitive** — pre-existing, not this round."*

That is worth chasing because of a chain no document draws:

* **pr/34 §6** — *"the graph walk is deterministic; **one input ordering is
  inherited and not proven here**"* — shower construction order, *"a property of
  **pr/33's stage**, not this one."*
* **pr/33 §6** — verdict **"not proven"**. *"`shower_less`'s `a.get() < b.get()`
  fallback is a **live address comparison feeding π⁰ pairing** … the one place in
  the file where determinism rests on an address."* Proposed fix:
  `shower_less_id_tiebreak`.
* **pr/33 never shipped (§8).** So the series' one unfixed live address-ordered
  comparator is the one pr/34's determinism verdict depends on — and the doctest
  reads like evidence for it.

**Measured: it is not.** 52 runs of `build/clus/wcdoctest-clus` at `2457320d`
(46 with ASLR on, 6 under `setarch x86_64 -R`) all report
`assertions: 984 | 984 passed | 0 failed`. The 983 therefore belongs to the
*stashed build* in pr/36's A/B, not to run-to-run variation at one binary. If a
per-run flip existed with probability p, 52 clean runs bound it at **p ≲ 5.6 %**
(95 %).

**What this does and does not settle.** It removes a piece of apparent evidence,
nothing more. The doctest does not exercise `shower_less`'s π⁰ pairing, so
**pr/33 P12 remains exactly what pr/33 called it: a concrete source-level
mechanism, not proven and not disproven.** §6.2's `FLOOR = 0` is likewise
consistent with P12 being live, because `setarch -R` fixes the address layout by
construction — a zero floor under ASLR-off says nothing about an address-keyed
comparator.

### §6.4 A comparator is an instrument — twice now

Round 1 recorded one instance. Round 2 produced a second, by a different
mechanism, in the same session — which promotes this from an anecdote to a rule.

* **Round 1.** `pr37_repeat_cmp.py`'s first version hashed ragged
  (`vector<vector<T>>`) leaves with `repr()`. A numpy **object** array reprs as
  `<… at 0x7f…>`, a heap address, so two bit-identical runs hashed differently.
  It reported **235 differing leaves on 47 of 48 events**, confined to exactly
  the five `T_proj_data` branches pr/28 §14 had driven to zero — maximally
  plausible. Direct value checks showed every value identical and in order.
  Fixed by descending to numeric leaves (`_walk`), never `repr`.
* **Round 2.** An ad-hoc `np.array_equal` snippet, written to attribute §2.3's
  movement to a knob, reported ~30 additional moved `T_tagger` branches
  (`ssmsp_*`, `cosmict_10_*`, `numu_cc_2_*`). All were phantoms: `np.array_equal`
  on jagged/object arrays does not do what it looks like it does. Re-run through
  `pr36_cmp.py`'s own `_to_py` / `branch_equal`, the answer is **one** branch.
  §2.3's `23 of 501` was then re-derived through that same calibrated path
  (`flag_shower` is a flat `int32 (501,)`, so it was never at risk — but the
  number should not rest on an instrument discredited three paragraphs later).

**The rule.** An A-vs-A self-test does not catch either class: the same process
yields the same addresses, and an identical pair compares equal under a broken
comparator too. Calibrate against a *known-different* pair — here, pr/36's own
arm pair reproducing §11.4 exactly is what licensed every other row in §2.2.

---

## §7 The prototype reference is a branch — and the series' top follow-up is now closed

**Symptom.** `prototype_base` → `prototype-dev/wire-cell`, submodule `pid` on
branch **`port`** at **`53ca938`**. Its merge-base with `master` is
**`a5fc0b9`**, and the branch carries **+5833/−989 across 26 files**. Docs pr/34,
pr/35 and pr/36 name the commit; **pr/28–pr/33 do not**.

Three consecutive rounds called re-checking it the series' top item —
pr/34 §7.7, pr/35 §10.10 (*"still owed"*), pr/36 §7.9 (*"**This is still the
highest-value follow-up in the series**, and with the eight stages now mapped
there is no longer a reason to defer it"*). Round 1 did pr/28–32. This closes
pr/33 and the six files round 1's table did not reach.

**The filter, and why round 1's numbers are replaced rather than extended.**
Round 1 filtered whitespace, blank lines, comments and `std::cout` / `chrono`.
That **under-filters**: the continuation lines of a multi-line print statement
(`<< " pdg=" << shower->get_particle_type()`) contain no `std::cout`, so they
were scored as substantive. Adding `grep -vE '^<<'` and re-running **one** filter
over **all 26 files** gives the table below; round 1's 92 and 114 were 37 and 70.
Use these.

| prototype file | raw | substantive | audited by |
|---|---|---|---|
| `apps/wire-cell-prod-nue-port.cxx` | +3384/−0 | **2844** | **nobody** — §5.3, §9 row 3 |
| `apps/wire-cell-prod-stm-port.cxx` | +576/−77 | 179 | nobody (STM app) |
| `src/NeutrinoID.cxx` | +170/−34 | 70 | pr/32 |
| `src/NeutrinoID_proto_vertex.h` | +217/−70 | 37 | pr/30 |
| `src/NeutrinoID_shower_clustering.h` | +137/−25 | **29** | **pr/33** |
| `src/PR3DCluster_dQ_dx_fit.h` | +201/−96 | 22 | pr/28 §4 |
| `src/ProtectOverClustering.cxx` | +26/−26 | 21 | nobody |
| `src/NeutrinoID_track_shower.h` | +66/−12 | 13 | pr/31, pr/32, pr/33 |
| `src/PR3DCluster_multi_track_fitting.h` | +94/−224 | 9 | pr/28 §1, §3b |
| `src/PR3DCluster.cxx` | +118/−55 | 9 | pr/29 |
| `inc/WCPPID/NeutrinoID.h` | +12/−1 | 7 | — |
| `src/NeutrinoID_improve_vertex.h` | +99/−92 | 4 | pr/28 §3, pr/32 P3 |
| `src/ProtoSegment.cxx` | +14/−6 | 3 | pr/31, pr/33 P6, pr/36 |
| `src/NeutrinoID_deghost.h` | +35/−9 | 3 | pr/33 |
| `src/PR3DCluster_steiner.h` | +184/−110 | 2 | pr/29 D1/D3/D6/D12 |
| `src/PR3DCluster_pattern_recognition.h` | +38/−12 | 2 | nobody |
| `src/PR3DCluster_graph.h` | +44/−27 | 2 | pr/29 D3 |
| `inc/WCPPID/PR3DCluster.h` | +4/−0 | 1 | — |
| `ToyFiducial.cxx`, `PR3DCluster_trajectory_fit.h`, `PR3DCluster_path.h`, `PR3DCluster_multi_dQ_dx_fit.h`, `PR3DCluster_crawl.h`, `NeutrinoID_final_structure.h`, `ImprovePR3DCluster.cxx`, `CalcPoints.cxx` | +7/−5 … +153/−61 | **0** | pr/28, pr/29, pr/32 |

**pr/33's file was opened and it is instrumentation.** All 29 substantive lines
of `NeutrinoID_shower_clustering.h` are one added debug-dump block (a per-shower
and per-π⁰ `std::cout` census, plus an eleven-field `scnv_dt_*` timing line) and
the `std::cout` rewrites of the two π⁰ messages §3.2 uses as counters. The same
holds for `NeutrinoID.cxx`'s 70 (timing + `print_segs_info` + the null guard
round 1 triaged), `NeutrinoID_proto_vertex.h`'s 37 (all `chrono`),
`NeutrinoID_track_shower.h`'s 13, and `NeutrinoID_deghost.h`'s 3.
`inc/WCPPID/NeutrinoID.h`'s 7 are the *infrastructure* for that dumping —
`#include <set>`, `<string>`, a `std::set<std::string>` of enabled stage names
and its two accessors.

**So the pr/28–pr/33 citations survive, all of them.** This is deliberately not
written as "every prototype citation is suspect": a broad alarm here would be a
wrong positive, and pr/36 §10.13 is the precedent for what those cost.

**Two files no audit cites, recorded so they are not rediscovered.**
`ProtectOverClustering.cxx`'s 21 lines are a scoring block **commented out** on
the `port` branch — a real algorithm change, in the over-clustering protection,
which no document in this series covers. `PR3DCluster.cxx`'s 9 are timing plus
one `flag_reg_save` guard.

**The six candidate live edits from round 1, triaged.** Four are not live code
and two are real *prototype* bugs the toolkit is structurally immune to, so
nothing in this section is a toolkit defect:

| # | `port`-branch edit | what it actually is | toolkit |
|---|---|---|---|
| 1 | `PR3DCluster_graph.h:528` `if (ref_point_cloud != 0)` | guards a **`std::cout`** | n/a |
| 2 | `PR3DCluster_graph.h:1149` `if (edge.second)` | re-indentation; the new guard has a **commented-out body**, so it is inert — but a dangling-`if` shape nobody should port | n/a |
| 3 | `PR3DCluster_steiner.h:585` `temp_map_new_old_indices = …` | populates a member whose **only reader is a commented-out line** in `apps/wire-cell-prod-stm-port.cxx:901` | n/a |
| 4 | `ProtoSegment.cxx:1286` `particle_score = 0;` | **a real upstream bug** — `do_track_pid`'s reset-before-return left `particle_score` stale | **immune** — `segment_do_track_pid` (`PRSegmentFunctions.cxx:1454`) *returns* `tuple<bool,int,int,double>`; there is no member to leave stale |
| 5 | `ProtoSegment.cxx:1764` `if (fit_index_vec.size()==0) …resize(…)` | **a real upstream bug** — `.at(nbreak_fit)` on a possibly-empty parallel vector | **immune** — the toolkit keeps one `std::vector<PR::Fit> m_fits`, not six parallel vectors; accessors bounds-check (`PRSegment.cxx:20-35`) |
| 6 | `NeutrinoID.cxx:362-369` null guard + early return in `determine_overall_main_vertex` | **a real upstream bug** — `map_cluster_main_vertices[main_cluster]` *inserted a null entry* and the function proceeded with `main_vertex == nullptr` | present — `NeutrinoVertexFinder.cxx:4005-4014` uses `find()` and returns `nullptr`. Independent corroboration for §3.1 |

**Fix (process, one line per doc).** Add the `pid` SHA to the Repro block of
pr/28, pr/29, pr/30, pr/31, pr/32 and pr/33, as pr/34–36 already do. The audits
are sound; undocumented provenance is what turns a sound audit into an
unreproducible one two years from now.

---

## §8 doc pr/33's owner filter was never implemented

Of the eight audits that ran an owner filter, **seven produced toolkit code and
one did not**. Round 1 had to qualify this because pr/36 had been filtered the
same day and had not shipped yet. **pr/36 shipped at `2457320d`. pr/33 is now the
unqualified sole outlier of eight.**

| doc | filter result | implementation | knobs live in `sbnd/wct-pr-perevt.jsonnet` at HEAD |
|---|---|---|---|
| pr/29 | 12 → 5 | §11, §12 | 3 ON (`steiner_terminal_wire_tol=1`, `steiner_terminal_adjacent_slice`, `steiner_edge_charge_forward_dead_mix`) |
| pr/30 | 14 → 4 | §12 | 5 present, **1 ON** (`oov_prototype_parity`) |
| pr/31 | 15 → 9 | §11, §12 | 7 present, **5 ON** |
| pr/32 | 12 → 4 | §11 | **4 ON** |
| **pr/33** | **14 → 5, "8 knobs"** | **none — the doc stops at §10.10** | **0 — none of the names exists anywhere in `clus/`** |
| pr/34 | 14 → 5 | §11 | **5 ON** (`pf_*`) |
| pr/35 | 14 → 4 | §11 | **1 ON** (`kine_shower_pdg_live`) + one unconditional change |
| pr/36 | 13 → 7 | §11 (`2457320d`) | 6 present, **5 ON**; `sp_sce_correction` OFF by owner decision |

`grep -rn 'shower_pdg_from_start_segment\|shower_pdg_exact_muon_test\|shower_less_id_tiebreak' clus/`
returns nothing.

**What is live because of it.** This is not only a scheduling note — three items
elsewhere in this document trace back to pr/33's non-implementation:

* **§3.2** — `acc_segment_id`'s two unconditional consequences are pr/33's F3,
  and pr/34 §10.8c records that its own P9 *compounds* with them.
* **§6.3** — pr/33's P12 (`shower_less`'s address comparison) is the series' one
  unfixed live address-ordered comparator, and pr/34's determinism verdict is
  explicitly conditional on it.
* **§11.3** — pr/33 ordered a `porting_dictionary.md` correction
  *unconditionally* ("whether or not F4 ships"), and it is still unmade.

pr/33 is upstream of every shower quantity in `T_kine` and the Bee mc tree, and
it is the largest block of accepted-but-unimplemented work in the series. Named
so it is scheduled rather than forgotten.

---

## §9 The unaudited blocks the nine documents name, ranked

Not discoveries — the docs' own admissions, ranked once, with why each matters.

| # | block | named by | why it ranks here |
|---|---|---|---|
| 1 | **The ~30 sub-taggers' internal cut values**, plus the 15 unreconciled `clus/docs/tagger/*_review.md` | pr/36 §0, §7.3, §9, §10.14 | the largest unaudited volume in the series, sitting directly upstream of the BDT scores that decide selection. pr/36 states plainly that a transposed constant inside one of them *"would pass every check in this doc"* |
| 2 | **`examine_structure_*` / `NeutrinoStructureExaminer.cxx`** | pr/32 §0, §9 | ~700 lines per side, called from `determine_main_vertex` — which **does** run on SBND (§1.1) — and pr/32 says it *"can move vertices before the final `examine_direction` sees them"*. The owner's original "vertex is a bit off" question |
| 3 | **`apps/wire-cell-prod-nue-port.cxx`** | round 1 §9; sized here | **2844 substantive lines** (§7), the app the SBND runners invoke, read by nobody. Every prototype-behaviour claim in nine documents assumes what this file configures |
| 4 | **`PRShower.cxx` ↔ `WCShower.cxx`, and `fill_sets` in particular** | pr/33 §0, pr/34 §0/§9, pr/35 §0 | §5.1 — three rounds deferred it and a production-ON knob (`pf_shower_vertex_barrier`) now consumes it |
| 5 | **Base-graph builders** — `Create_graph` / `Establish_close_connected_graph` ↔ `find_graph("ctpc_ref_pid")` | pr/29 §0, §13.2 | *"Everything in §4 is about inputs to the solver; the biggest input of all was not opened."* §7 items 1–2 land in these functions |
| 6 | **`improve_maps_no_dir_tracks`** | pr/31 §1, §9 | 331 vs 474 lines — the largest unread pair — containing **8 of P1's 11 sites** |
| 7 | **`examine_direction`'s PDG-reassignment ladder** (`:1232-1348`, `:1529-1548`) | pr/32 §9; pr/33 §7.5 | Tier B, 428 vs 632 lines, and it runs in **both** vertex paths, so unlike §1.1's casualties it is live on SBND. pr/33 §7.5 says its own `examine_shower_1` / `examine_showers` gap is *"the same gap pr/32 GOTCHA 12 flagged … narrowed but not closed"* |
| 8 | **`TrackFitting::collect_2D_charge`** | pr/33 §0/P7, pr/35 §0/§1 | pr/33 closed P7 on a source argument resting on three greps (*"If any of those changes, P7 reopens"*), and pr/35's whole F3 class argument rests on *"reads `.charge` and nothing else"*. Two rounds lean on a function neither read |
| 9 | **`clustering_points_segments`** (317 lines) and pr/30's `update_association` coordinate question | pr/31 §0; pr/30 §9 | *"one instrumented run settles it"* and nobody ran it |
| 10 | **Downstream consumers of `steiner_graph` / `steiner_pc`**; **the retile step** | pr/29 §13.2, §0 | one consumer was opened and found a deliberate substitution — *"That one was found by looking; nobody has looked at the rest."* The retile step is de-risked by §7 (`ImprovePR3DCluster.cxx` substantive = **0**) but still unread |

Also unclosed and cheap: pr/29's **D8 and D10 reaches** are dismissed as *forced*,
which is a statement about *why* the toolkit differs, not *how much*; and
**pr/34 §7.5** — is `conn_type == 4 ∩ pi0_showers` reachable? — is the series' one
loose end never revisited in any later section of its own document.

---

## §10 The measurement debt — the count, derived

pr/32 §11.8 called the stale valfast/1000 baseline *"past the point where 'stale
baseline' is a footnote"* and put **ten** knobs on it. Round 1 derived nineteen.
With pr/36 shipped it is **24**, and it must not be obtained by adding the
per-doc sections — per doc 68 the SBND operating point lives **only** in cfg, and
`clus.jsonnet`'s parameter defaults are the *legacy/off* values. Derived from
`wct-pr-perevt.jsonnet` at `2457320d`:

| doc | knobs present | **ON at HEAD** |
|---|---|---|
| pr/29 | 3 | **3** — `steiner_terminal_wire_tol=1`, `steiner_terminal_adjacent_slice`, `steiner_edge_charge_forward_dead_mix` |
| pr/30 | 5 | **1** — `oov_prototype_parity` (`fit_exclusion`, `graph_endpoint_strict` false; two `null`) |
| pr/31 | 7 | **5** — `cont_muon_dir3_30cm`, `track_comp_empty_abstain`, `shower_topo_reset`, `reclass_preserve_4mom`, `dir_track_median_local` |
| pr/32 | 4 | **4** — `vertex_dir_use_fit_point`, `shower_traj_recheck_parity`, `main_vertex_require_descriptor`, `main_vertex_candidate_flag` |
| pr/33 | 0 | 0 — never implemented (§8) |
| pr/34 | 5 | **5** — the `pf_*` family |
| pr/35 | 1 | **1** — `kine_shower_pdg_live`, plus one **unconditional** change (the segment `cal_kine_charge` cache reuse) |
| pr/36 | 6 | **5** — `neutrino_consistent_fv`, `tagger_ordered_segment_sets`, `stem_endpoint_wcpt_parity`, `broken_muon_cluster_id_count`, `neutrino_type_bitmask` (`sp_sce_correction` false) |
| | | **24 knobs ON, plus one unconditional computational change** |

Several unconditional *log-only* additions ride along too — pr/35's F1 counter
and F4 WARN, pr/36's F2 population sweep with 11 per-gate counters and F1's
both-ways diagnostic. They are artifact-inert (proven by their rounds' gates) but
they are in production.

**How the table was built, so the number can be re-derived rather than trusted.**
The *total* is a mechanical count of boolean TLAs set away from the C++ default
in `wct-pr-perevt.jsonnet` at HEAD (the grep is in the Repro block). The
*per-doc split* is assigned **by knob name**, from each document's own §11/§12
implementation section — not by scanning the `// doc pr/NN` comments, which is
unreliable: a comment's attribution carries forward to every TLA below it until
the next one, so a naive scan credits pr/23 with 30 TLAs and pr/35 with 18. If a
re-run of the grep gives a different split, trust the knob names.

**Two things this round adds to the debt, both from §1.2 and §2.4.** First, the
campaign is not a re-confirmation: for most of these 24 knobs it is the *first*
evidence they will have, because their measured effect on nueCC48 is null.
Second, it must be run with a comparator that opens all four artifact families
(§2.4) — four of the 24 were gated with one that could not see two of them.

Carried forward from the nine documents, one line each, so they are in one place:

* **pr/29 §13.1** — D1+D12 has never been separated from D2; the pi0 question
  (evt 388 lost one) is open; the 24-starved-clusters number's scaling is only
  partially answered.
* **pr/28 §15.9** — round 8's two missing tests; **`work-tfix388-r9` must be
  hand-added to the next retire round's `PROTECTED` list** (there is no automatic
  protection). Add **`work-pr37b-repeat{A,B}`** to the same list if the floor is
  to stay re-checkable.
* **pr/30 §7.1/§7.2** — the `walk_history` asymmetry between the two
  `proto_extend_point` calls; and the `init_first_segment` main-cluster
  flag-vs-pointer warning, which §3.1 supplies a mechanism for.
* **pr/31 §7.3/§7.4** — `segment_cal_4mom`'s dead `MIP_dQdx` parameter;
  `kslike_compare` divides by zero on an all-zero window. **§7.1's anchor has
  drifted** — `PRSegmentFunctions.cxx:2028-2029` is now an `(apa,face)` angle
  cache, so that loose end must be re-anchored before it can be answered.
* **pr/31 §12.9, pr/32 §10.6** — F7/F8 and P2's SBND-tuning concern deliberately
  not moved. **And §12.4's null claim needs the §2.3 correction.**
* **pr/32 loose ends 5 and 6** — three coexisting formulations of "which end of
  the segment touches this vertex"; the `else if` that makes
  `angle_beam < 45 && max_angle < 70` unreachable in **both** trees (M15 applies).
* **pr/33 §7.1/§7.4** — P1's proton-skip reach and P10's degenerate inputs, both
  unmeasured; and the whole of §8 above.
* **pr/34 §7.5** — the one loose end never revisited (§9).
* **pr/35 §7.1** — the F1 counter's **572-event valfast** demotion test *"has not
  been run"*; §11.8 also records that the calib JSON gate **cannot see BDT
  scores**, which §2.1's `pr36_cmp.py` now can.
* **pr/36 §7.5/§7.6/§7.8** — who reads the `neutrino_type` branch (F7 shipped
  with *"unmeasured value"*); P2's downstream consumer, never found, which is
  what would set F3's real severity; the SSM sentinel-coverage sweep.
* **The porting dictionary still has no section for the pr/30–pr/36 stages.**
  pr/33 §7.7 counted *"sixth audit in a row"*; on its own counting rule it is now
  **nine**. pr/29's was filled (§13.4); the others' divergences are undocumented
  by construction, which is what keeps escalation rule 4 binding on all of them.

---

## §11 Small, named, cheap

**§11.1 Raw `std::cout` on the production PR path.** Eight at `2457320d` — five
in `TaggerCheckNeutrino.cxx` (`:716`, `:820`, `:824`, `:834`, `:852`, each
followed by a `print_segs_info`) and three in `NeutrinoPatternBase.cxx`
(`:2313`, `:2317`, `:2321`, inside `print_segs_info` itself). Round 1's anchors
for these have drifted by ~46 lines; these are HEAD's. CLAUDE.md §2 mandates `Aux::Logger` /
`SPDLOG_LOGGER_DEBUG`. These print unconditionally, at any log level, once per
event, and bypass the log-level plumbing every other diagnostic in this chain
respects. Reported, not fixed.

**§11.2 Log lines tear mid-word**, e.g. `PR31AUDIT … examine_showerusters=91>,
40562 points …` — two loggers interleaving on one line. Known
(`project_wct_log_line_tearing`); it makes counter lines unsafe to parse with a
strict regex (**parse per-key, not per-line** — pr/36 §11.5 recovered its sweep
that way), and it is the recorded cause of pr/35's three `stmfit` TSV columns,
reproduced independently in §2.2.

**§11.3 One unconditional fix, ordered and not made.** pr/33 §10.7:
`clus/docs/porting/porting_dictionary.md:222` maps `get_flag_shower()` →
`flags_any(kShowerTrajectory | kShowerTopology)` with the `abs(pdg)==11` term
simply absent — *"a documentation bug, not a behaviour change, and **it is what
makes this class recur** … Correct it whether or not F4 ships."* **Verified
unchanged at `2457320d`.**

**§11.4 Citation hazards in pr/33 and pr/34**, recorded because a future reader
will otherwise quote withdrawn text:

* **pr/33's §3 and §8 anchors are stale by up to +19 lines and were left that way
  deliberately** (§10 preamble, so the two revisions stay distinguishable). Cite
  its **§10.x** anchors. HEAD function heads: `:76 :230 :473 :762 :1223 :1304
  :1641 :2099 :2458 :2802 :3179`.
* **pr/33 §9 bullets 3 and 4 are superseded** by §10.8 corrections 5 and 6.
* **pr/33 gives P12 three different line numbers for one line** — `:2829` (§3,
  §8), `:2831` (§6), `:2848` (§10.6).
* **pr/34 §11.8 mis-cites `§7.3`** for the `fill_sets` question (§5.1).
* **Prototype anchors taken from `git diff` hunk offsets are systematically off
  by one** versus the file — round 1 shipped eight such and fixed them in
  `1e71ce2`. Read the file.

**§11.5 The crashed pr/31 arms are a trap for exactly this kind of re-analysis.**
`work-pr31r2-allon48` and `work-pr31r2-f1on48` hold **1 of 48** `tracking-pr.root`
files — they are the arms pr/31 §12 records as having crashed on the
`ParticleInfo` validator. The re-runs are `-allonb48` / `-f1onb48`. A comparator
pointed at the wrong one dies on a `FileNotFoundError`, which is the good case;
a hash-only comparator would have silently compared 48 pctree files and reported
a clean PASS.

**§11.6 The tree moved under both rounds.** A concurrent session committed
`29e8e452` during round 1 and `2457320d` before round 2. Every anchor here was
re-read at `git show HEAD:`, and the binary and doctest mtimes (`20:55:43`,
`20:55:45`) were checked before, between and after every arm. "Same binary" is a
premise of §6.2 and deserves evidence rather than assumption.

---

## §12 What is NOT claimed

* **The ten blocks in §9 were not opened.** They are ranked from what the nine
  documents say about them, not from reading them.
* **§3.1's fix was not written, built, run or gated**, and **§3.2's is not
  proposed at all** — its shallow half belongs to pr/33's implementation round
  and its deep half (a global segment-id allocator) does not exist to be hooked.
* **§2's re-gate covers only the arms that survive.** pr/29 has no 48-event pair
  (single-event `work-pr29-388-*` only), so its bar is **unrecoverable**. pr/28
  predates the arm-naming scheme entirely.
* **§2.2 does not say pr/32's knobs are wrong.** They are behaviour changes doing
  what they were shipped to do. What is wrong is a summary sentence, produced by
  a comparator whose docstring over-promised. The **byte-identical bar itself
  holds** — verified here under a strictly wider instrument than the round used.
* **§2.3 does not overturn pr/31's conclusion**, only its words. No score, no
  verdict, no Bee member and no pctree entry moves.
* **§1.1 is 48 nueCC events, not valfast.** *"DL accepts on SBND"* is a statement
  about this manifest and this operating point. A different `dl_vtx_cut`, a
  different weights file, or cosmics rather than nueCC could change it — and the
  traditional path is exactly the fallback that would then start running,
  carrying §3.1 with it.
* **§1.1 does not say pr/32 audited dead code.** Four of its Tier-A functions run
  in production and its four knobs are engaged.
* **§3.2's 0/48 is a reach, not an impossibility proof.** pr/33 shows from source
  that both finders' gates are compatible; the population simply does not occur
  in 48 nueCC events. Consequence 2 (`ssm_tagger` receives 0) is unconditional
  and is **not** bounded by that zero.
* **§4 proposes nothing and estimates nothing.** Not whether SBND needs any of
  the three corrections, not their magnitudes, not that the prototype's uBooNE
  graphs are the right answer.
* **§6.1 is a source sweep, not an N-run identity test per site**, and §6.2's
  zero says no *observed* artifact moved, not that every pointer-ordered
  traversal is provably unobservable. Two arms cannot bound a rare
  nondeterminism; pr/28 §14.5 found its residual only after a larger family was
  removed. The comparators are **order-sensitive**, so a pure permutation would
  be reported as a difference — moot at a floor of zero.
* **§6.3 does not settle pr/33 P12 either way.** It removes one piece of apparent
  evidence and bounds a per-run flip at p ≲ 5.6 %. The doctest does not exercise
  `shower_less`'s π⁰ pairing, and `setarch -R` fixes the address layout by
  construction.
* **§7 does not clear the `port` branch.** It shows the *audited* files' diffs
  are instrumentation plus six triaged hunks. `wire-cell-prod-nue-port.cxx`
  (2844 substantive lines) and `wire-cell-prod-stm-port.cxx` (179) were **not**
  read, and the 26-file diff was not read line by line — the filter is a
  heuristic, and a semantic change hidden inside a print-adjacent hunk would
  survive it.
* **§7's immunity claims are structural, not measured.** Items 4 and 5 argue from
  the toolkit's data model; no event was run to demonstrate the prototype bug
  firing.
* **§10's count is knobs ON at one HEAD.** Not a claim that all 24 interact, nor
  that a valfast run would move any of them.
* **Nothing here is implemented.** No toolkit C++ or jsonnet was changed by this
  document; `git -C toolkit status --porcelain` carried only a concurrent
  session's untracked files throughout.
