# Doc 77 — Knob lifecycle and the shape of the final-production configuration (2026-08-24, analysis only, no code)

Owner ask, ahead of final production: "we have added probably hundreds of
knobs (default C++ off) and on for SBND running. There might be some failed
attempts as well. For clearly-failed attempts, shall we remove them from the
code, or leave them as record? For hundreds of knobs, shall we keep them, or
consolidate, or reduce them as default?"

**This doc answers with numbers, not impressions, and changes nothing.** No
`.cxx`, `.h`, `.jsonnet`, default value, or doctest is touched. Every
recommendation below is staged as a later, separately-gated round (§8). One
question — whether the uBooNE chain may ever be re-baselined — is presented
as open rather than decided (§5c, §9): it is the single decision that gates
the largest possible cleanup, and it is the owner's, not this doc's.

## Repro block

Every number in this doc comes from one of these three greps, run against
`toolkit` at `HEAD` on `apply-pointcloud` (283 commits ahead of
`origin/master` — `git rev-list --count origin/master..HEAD`) and `wcp-porting-img`
at `HEAD` on `main`. Re-run to check any figure below.

```bash
T=/nfs/data/1/xqian/toolkit-dev/toolkit
cd $T

# branch size
git diff --shortstat origin/master...HEAD
git diff --name-status origin/master...HEAD | grep -c '^A'
git diff --name-only  origin/master...HEAD -- cfg/

# the SBND operating point: 380 TLAs, split by default
F=cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet
B=$(awk '/^function\(/,/^\)/' $F)
echo "$B" | grep -oE '^\s+[a-z_][a-z0-9_]*\s*=' | tr -d ' =' | sort -u | wc -l   # 380
echo "$B" | grep -cE '=\s*true\s*,'   # 155
echo "$B" | grep -cE '=\s*false\s*,'  # 19
echo "$B" | grep -cE '=\s*null\s*,'   # 121

# uBooNE: how many behavior knobs it sets (answer: 1)
grep -n "dir_weak_use_score\|fit_exclusion\|dqdx_fit_keep_all_points" \
  /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport/uboone-mabc.jsonnet

# knob surface per package (union of get() + bracket reads)
grep -rhoE '\bget(<[^>]*>)?\(\s*(config|cfg|jcfg|jconfig|m_cfg|m_config)[^,]*,\s*"[^"]+"' \
     --include=*.cxx --include=*.h clus | grep -oE '"[^"]+"$' | tr -d '"' | sort -u | wc -l   # ~560
grep -c 'get(config\|get<' clus/src/TaggerCheckNeutrino.cxx                                    # 335

# the hand-written mirror block
grep -cE '^\s*pattern_algos\.m_[a-z0-9_]+\s*=' clus/src/TaggerCheckNeutrino.cxx                # 290

# default-pinning doctest
grep -c 'TEST_CASE' clus/test/doctest_clus_knob_defaults.cxx   # 21
grep -c 'CHECK'      clus/test/doctest_clus_knob_defaults.cxx  # 402

# env-var side channel, invisible to both jsonnet and any doc grep above
grep -rhoE 'getenv\("[A-Z_0-9]+"' clus/src/*.cxx | sort -u | wc -l   # 37

# deletion precedent already in the log
git log --oneline origin/master..HEAD | grep -iE 'revert|remove|retire|drop|pulled from'
```

---

## 1. Four populations that look identical in the source

The tree contains four things that all present the same way — a
`get(config, "foo", m_foo)` line plus a jsonnet key — but are not the same
kind of thing:

| # | Population | Size | What it is |
|---|---|---:|---|
| 1 | **The reference** | C++ default set | uBooNE / prototype parity. Frozen, pinned by `doctest_clus_knob_defaults.cxx` (21 cases, 402 `CHECK`s). Not configuration — a **baseline**. |
| 2 | **The operating point** | 229 keys | What SBND actually turns ON. This is the algorithm, not debt. Already single-sourced by doc 68. |
| 3 | **Held-off, verdict recorded** | 19 keys | Measured or deliberately parked; explicit `false`. |
| 4 | **Plumbed but null** | 121 keys | Threaded through three jsonnet layers, emits **no key at all** when unset. |

Conflating these four is why the surface *feels* like "hundreds of knobs"
rather than what it actually is: one 229-key operating point, one frozen
reference, and about 140 handles that see little or no use. The rest of this
doc treats them separately because the right action differs by population —
population 1 must never move without a stop-and-ask (CLAUDE.md §5 rule 1);
population 2 is the algorithm and stays; populations 3 and 4 are where the
actual "hundreds of knobs" feeling — and most of the real cleanup opportunity
— live.

## 2. The measured picture

### 2.1 Scale of the branch

283 commits ahead of `origin/master`; 132 files changed, +50 805 / −2 551; 41
new source files (34 in `clus/`). Against that: **only 5 cfg files touched**
(`cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/sbnd/{clus,
wct-pr-perevt,wct-clus-matching-perevt}.jsonnet`,
`cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json`). The knob surface is a
C++-side phenomenon that surfaces through a narrow cfg funnel.

### 2.2 The knob surface is bigger than the SBND job alone suggests

Distinct config keys read from JSON, by package (union **968**, some names
shared across many components so per-package sums exceed the union):

| package | distinct keys |
|---|---:|
| `clus` | 593 |
| `match` | 218 |
| `root` | 136 |
| `img` | 117 |
| `flash` | 80 |

`clus/src/TaggerCheckNeutrino.cxx` alone holds **335 `get()` calls — half of
the whole `clus` config surface**. Of clus's ~216 boolean flags, **196
default false** — that is the size of the porting/improvement campaign's
feature-flag surface, and ~144 of the 196 live in that one file.

There is also a **fourth, undocumented config channel**: **37 distinct
`getenv("WCT_*")` debug knobs** across `clus/src/*.cxx` — census counters and
dump switches, in no jsonnet, cited in no doc. They are real knobs by the
CLAUDE.md definition (they change what the code does) but invisible to any
grep of cfg or docs. §4 addresses them.

### 2.3 The SBND operating point — `wct-pr-perevt.jsonnet`

**380 TLAs** in one function signature (`:43`–`:2493`), 2451 lines of which
2117 are comment:

| Default | Count | Meaning |
|---|---:|---|
| `true` | 155 | ON — boolean feature flags |
| explicit value | 85 | ON — numeric/string tuning (4 numerically neutral, e.g. `mvga_reseat_angle = 0`) |
| `false` | 19 | plumbed, verdict recorded, held off |
| `null` | 121 | plumbed, emits **no key**, never exercised |

**229 keys are actually turned ON** for SBND (155 bool + 74 valued, excluding
14 infra TLAs like `input`/`event`/`output_dir`). By family: `shower_*` 36,
`mvga_*` 27, `pf_*` 14, `kine_*` 13, `tgm_*` 13, `protect_*` 10, `stm_*` 9,
`nu_*` 7, a long tail of 1–4 after that.

### 2.4 The uBooNE chain sets one behavior knob

`qlport/uboone-mabc.jsonnet` imports `pgrapher/common/clus.jsonnet`
**directly** (no detector-specific layer) and sets **exactly one** behavior
knob ON — `dir_weak_use_score = "true"` (doc pr/6, owner 2026-07-30). Its
visitor list is shorter too: no TGM, no FC, no `protect_bundle`, STM
commented out. `fit_exclusion` and `dqdx_fit_keep_all_points` are
explicit `false`.

**Ratio: 229 ON for SBND vs 1 ON for uBooNE.**

### 2.5 The load-bearing fact

Because uBooNE overrides essentially nothing, **the C++ default set *is* the
uBooNE porting reference**, not an abstract baseline. `doctest_clus_knob_defaults.cxx`
says this in its own header: a moved default "would make every 'no behavior
change' claim false while every gate still passed, because the gate compares
two runs of the same (already-wrong) default." `wct-pr-perevt.jsonnet:348`
states the same guarantee from the SBND side: *"The C++ defaults stay OFF, so
uBooNE/ICARUS/PDHD/PDVD are unaffected."*

This is the fact that makes **"change the defaults" and "consolidate the
config" different kinds of action**, not two strengths of the same one — the
first destroys the branch's central validation invariant; the second never
touches it. §5 keeps them separate for that reason.

### 2.6 A knob costs six hand-written restatements

There is **no central knob struct and no profile/variant/preset mechanism
anywhere** — `variant` appears exactly twice in SBND cfg, both times in
prose comments, never as a data structure. What exists instead is a manual
copy chain, restated by hand at each stage:

| Stage | Where | Size |
|---|---|---|
| A — `get()` into members | `TaggerCheckNeutrino.cxx:141`+ → `TaggerCheckNeutrino.h` | 335 calls; 924-line header, 143 bool + 188 numeric members |
| B — **hand-written mirror** `pattern_algos.m_X = m_X;` | `TaggerCheckNeutrino.cxx:1731`–`3028` | **290 lines** |
| C — the de-facto knob bag | `NeutrinoPatternBase.h:201` `class PatternAlgorithms` | 3298-line header; 114 bool + 154 double members |
| D — side channel for call sites too deep to thread a parameter through | 6 file-scope `extern` globals (`PRGraph.h:187` `g_graph_endpoint_policy`, four audit counters, `g_shower_traj_refresh_flag`) | `PRGraph.h:280` states the reason directly: it can't reach `PatternAlgorithms::m_traj_cover_probe` from six files |
| jsonnet L1 | `common/clus.jsonnet:483` — one physical line | **344 named params**, 329 `+ (if …)` key-suppression clauses |
| jsonnet L2 | `sbnd/clus.jsonnet:707` `clus_pr(...)` | **406 params**; body forwards ~200 of them one-by-one |
| jsonnet L3 | `wct-pr-perevt.jsonnet:43` | **380 TLAs** |

So a single knob is restated by hand roughly six times end to end, and the
SBND-vs-default divergence is recorded only as inline literals plus
free-text `// doc pr/NN` comments — nowhere as structured data.

**The consolidation target already exists — in the naming, not in any
data structure.** `two_end_break` gates 18 `teb_*` sub-knobs;
`vertex_kink_snap` gates 11 `vks_*`; `main_vertex_graph_audit` gates ~28
`mvga_*`. Every family in §2.3's table is already a gate/sub-knob group by
convention; §5b turns that convention into a real grouping.

### 2.7 103 keys read in C++ but set in no cfg at all

`clus` 38 · `img` 21 · `flash` 18 · `match` 17 · `root` 9 — keys with a live
`get()` call site that no jsonnet in the tree ever sets. The `clus` list is
dominated by ~18 `guard_*` STM accept-guard cut values: the *gate* that turns
guard-checking on is exposed as a knob, but its individual cut parameters are
hard-defaulted in C++ and never exposed at all. This is a smaller, sharper
version of the population-4 problem (§1): a handle nobody has ever needed to
turn.

## 3. Deletion is already established practice — not a new policy

The owner has already chosen "pull it from the code, not just switch it off"
at least four times on this branch:

- `225d7e7e` — *"clus/cfg: revert doc pr/43 — five PID knobs pulled from code,
  not just off (too broad, owner requested rollback; will revisit)"*, reverting
  `4aabef3e`. pr/43's own doc: *"owner … asked for the five knobs to be
  **pulled from the code entirely**, to be revisited later rather than left
  dead-OFF in the tree."*
- pr/24's `dl_vtx_swap_min_len` / `dl_vtx_swap_min_frac` — "removed from the
  tree" (per pr/100's summary table: "tried, removed from the tree — also
  does not work well; remove").
- `df40b2a4` — *"REMOVE the two round-1 knobs"* (pr/38 round 2).
- `ff541919` — *"drop the doc pr/24 `dl_vtx_swap_*` wiring (fix withdrawn)"*.
- `60bad894` — `Revert "clus: doc pr/72 round 3 — es3sg_vertex_fit … NEGATIVE
  result, DEFAULT OFF"`.

**7 knobs are already deleted after rejection.** So question 1 in the owner's
ask is not a new policy to invent; it is making an existing, already-applied
practice explicit and giving it a ledger so it stops being ad hoc.

### 3.1 The counter-weight, from the same corpus

Deletion is not free of risk, and the docs already record the opposing
pressure:

- `pr/32:1564` — *"P7 stays behind a knob until a wider census agrees;
  **retiring it is a decision, not a consequence.**"*
- `pr/99:339` — after `mvga_ac_veto_radius` measured ADVERSE: *"**Knob
  retained for future scans.**"*
- `pr/112:335` — conditional retirement: *"A knob, as the owner said, and
  retired when a retrained net lands."*
- `pr/32:1190` — the rule that dissolves this into a checkable test: *"if it
  is 0 everywhere, the finding is confirmed dead and **the knob can be
  retired rather than carried forever.**"*

§4 turns this into a four-kind taxonomy so "remove" is never applied to a
knob that is actually a live diagnostic or a genuinely open question.

### 3.2 Two traps for anyone doing the removal round

**Trap 1 — doc prose goes stale; the jsonnet does not.** Four knobs whose
docs currently say "stays OFF" / "not flipped" are ON in production today:
`fit_exclusion` (`pr/30:1465` "stays OFF" vs `= true` today, flipped by
pr/98), `nu_per_bundle` (`pr/94:1463` "stays OFF in production" vs `= true`,
pr/94 §9.13 Phase 6), `sgp_max_sep` (`pr/73` header "OFF and is not flipped"
vs `= 3.0` today, owner flip 2026-08-13), `shower_proton_daughter_pion`
(`pr/40` "left OFF, not flipped" vs `= true` today — `pr/40:1851` already
self-corrects this in its own text). **Consequence: any ledger or removal
decision must be generated by grepping the current jsonnet + `git blame`,
never by transcribing a doc's prose verdict**, because the doc that recorded
"OFF" may predate a later flip recorded in a different doc.

**Trap 2 — "retire" is a taken word in this tree.** It already means deleting
work-tree run directories to reclaim disk (`work-tags.md`'s repeated
"RETIREMENT ROUND `<date>`" sections, `scripts/retire/*`, doc pr/76 "retire to
20 G"). If a future round reuses "retire" for knob removal, commit messages
and this doc's own vocabulary will collide with that established sense.
**Recommendation: use "knob retirement" and "arm retirement" as distinct
terms whenever both are in play**, and say which one explicitly in every
commit message that could be read either way.

### 3.3 A precondition before anything gets called "final"

`pr/32:1560` — *"**48 events, not 1000.** The valfast/1000 population gate
has been stale since round 6, and the count of knobs riding on it is now
**ten** … regenerating it should be **scheduled, not mentioned.**"* Any
freeze of the operating point (§5b, §8) inherits this debt — ten knobs'
validation currently rests on a stale population gate. §8 lists regenerating
it as a prerequisite, not an afterthought.

## 4. Answering "remove or keep as record?" — a four-kind taxonomy

The OFF population (populations 3 and 4 from §1, plus the tooling layer) is
not one thing. Treating it as one thing is exactly what makes "remove vs.
keep" feel like a single yes/no question when it is actually four different
answers:

| Kind | Examples | Recommendation |
|---|---|---|
| **Diagnostic / tooling, off by design** | `rough_path_probe`, `sgp_edge_probe`, `traj_cover_probe`, `vertex_scoreboard`, `dl_vtx_harvest`, `save_stm_fit`, `PrDisplayDump`, `NeutrinoGraphAudit`, the 37 `WCT_*` env probes | **Keep.** These pay for themselves every debugging round; they are not attempts at a physics change. Re-label as *tooling*, not *knob*, in any future census — and give the `WCT_*` env channel a one-page doc, since it is currently invisible to anyone who only greps cfg. |
| **Measured negative / ADVERSE, verdict closed** | `other_seg_uncover_3d` (23 ADVERSE movers), `other_seg_keep_isolated_min_nnf`, `teb_second_max` (negative on its own motivating events), `dl_vtx_topo_weight`/`_center` (live A/B −8/1014), `dl_vtx_swap_guard` (live A/B −36/1014), `graph_endpoint_strict` ("must stay OFF" — pr/30 P8), `shower_connect_protected_pion_guard` ("measured-dead negative result, never flipped" — porting_dictionary.md) | **Remove** — code, all 6 plumbing layers (§2.6), and the corresponding doctest lines, once logged in the ledger (§6). |
| **Deferred — mixed evidence, no closed verdict** | `dqdx_fit_keep_all_points` (vertex 35→36 but a 12/47-event EM-shower reshuffle — "NOT flipped, owner decides the next step"), `dl_vtx_cloud_no_exclusion`, `main_vertex_swap_apply`, `fit_blob_coverage_defer`, `mvga_ac_veto_radius` | **Keep, but timestamp.** These are genuinely open questions, not failures — removing them would erase live options. But "no verdict yet" should not mean "forever": give each a stated re-visit horizon (e.g. next full-1k census) after which it is either promoted, rejected, or explicitly re-parked with a fresh doc entry. |
| **Detector-differentiating** | fiducial-volume / geometry-scoped knobs, the `*_consistent_fv` family | **Keep permanently.** This is real configuration difference between detectors, not campaign debt, and belongs in the tree the way `sbnd_track_fitting.json` does. |

### 4.1 Why removal beats "leave the code as the record", for kind 2

- **Dead code is a poor record.** No gate exercises it, so it silently
  bit-rots under continued churn around it (283 commits and counting); by the
  time anyone revisits it, it may no longer even compile against its
  neighbours. Git preserves it losslessly regardless of whether the
  source tree does.
- **The pr/NN doc is a *better* record than the source.** It holds the
  measurement (which events moved, by how much, ADVERSE count) — the source
  code alone holds none of that.
- **Each dead knob costs the six-stage restatement in §2.6, forever**,
  whether or not it is ever read again.
- **This branch is expected to merge to `WireCell/wire-cell-toolkit`
  eventually.** A dead default-OFF branch that will never be flipped is pure
  review burden there, with no offsetting value.

### 4.2 What makes removal safe: it is recoverable, and cheap to prove so

`git log`/`git show` recovers any deleted knob byte-for-byte from its
commit hash — nothing here proposes deleting history, only the live tree.
And per CLAUDE.md's own bar (§4), a kind-2 removal is provable the same way
any other unknobbed, byte-identical change is: an OFF-gate PASS on the
standard manifest (the removed knob was already OFF in production, so the
compiled config for every existing arm is unchanged) plus `wcdoctest-clus`
green after the corresponding `CHECK`s are dropped from
`doctest_clus_knob_defaults.cxx` — see doc 70 sec on why that file must be
amended, never left to silently start failing.

## 5. Answering "keep, consolidate, or reduce as default?"

### 5a. Flip C++ defaults to the SBND values — recommend against

This would silently redefine the uBooNE porting reference (§2.5), void the
premise of the 402-CHECK defaults doctest, and destroy the "key absent ⇒
pre-knob behavior" invariant that every A/B gate on this branch rests on. It
is only reversible by pinning ~229 keys explicitly `false` in
`uboone-mabc.jsonnet` — which relocates the same clutter into a different
file and still loses the invariant, since the *meaning* of "key absent" has
changed. It is also, independently, a standing stop-and-ask by CLAUDE.md §5
rule 1 ("changing any existing knob's default" requires the owner to be
asked first).

### 5b. Consolidate the surface — recommend; both halves provable without a physics gate

Two distinct, independently-doable pieces:

**b1 — jsonnet: turn the naming convention (§2.6) into real structure.**
Measured by prefix-grouping the 380 TLA names: 39 are true singletons (no
sibling with the same first token) and stay flat; the remaining 341 fall into
44 first-token families (`teb_*`, `vks_*`, `mvga_*`, etc.) covering 90% of
the total. Three of those families are umbrella prefixes spanning several
independent algorithmic features rather than one gated group — `shower_*`
(56 TLAs), `kine_*` (30), `pf_*` (18) — so at "one nested object per real
feature" granularity (`two_end_break: {enabled, min_len, turn_angle, …}`
instead of 19 flat siblings) the honest count is **roughly 70–90 named
groups**, not a single round number. TLAs are the right shape for per-run
things — `input`,
`event`, `output_dir`, `anode_indices`, `reality` — not for 229 physics
settings threaded by hand through three jsonnet layers. The payoff: "what did
production run" becomes one hashable JSON object that can be **frozen and
tagged** for the final production, rather than 380 individually-defaulted
function arguments. This is the direct continuation of docs 64 → 68 (doc 68:
*"the SBND production operating point now exists in exactly one place"*), and
doc 68's own gate machinery (`scripts/cfg/compile_prjob_cfg.sh`, compiled-JSON
identity) already proves this kind of change byte-identical with no rebuild
and no physics run. Doc 68's standing rule (`:101`, diagnostics stay OFF in
cfg) survives unchanged.

**b2 — C++: the 290-line mirror block is pure mechanical boilerplate.**
`TaggerCheckNeutrino.cxx:1731`–`3028` is a hand-written copy of one struct's
members into another (`pattern_algos.m_X = m_X;`, repeated 290 times).
Removing it — e.g. by having `TaggerCheckNeutrino::configure()` populate
`PatternAlgorithms` directly — is byte-identical by construction; it is a
representation change, not a behavior change. **But** it is a mechanical
edit inside a heavily-used production file, and CLAUDE.md's M10 ("fork by
duplication, don't extract shared helpers... the owner has repeatedly
rejected this review burden") makes any such refactor of a shared production
file the owner's call, not a default action. This doc flags it as the
single largest mechanical win on the table and leaves the decision open.

### 5c. Promote settled ON knobs to unconditional code — recommend, gated on one owner decision

A knob that is ON for SBND and will never plausibly be turned back off is not
a knob; it is the algorithm. Deleting the OFF branch and keeping only the ON
one is the *only* route that actually shrinks the C++ (§5b only reshapes the
cfg and removes boilerplate — it does not remove logic branches). But doing
this re-baselines uBooNE: some of the 229 ON keys, if hard-coded, would move
uBooNE's C++-default output for the first time since the porting reference
was established (§2.5). It is possible **only if** the uBooNE chain is
treated as a historical porting reference that may be deliberately
re-baselined once at a chosen point (e.g. at final production), rather than a
live target that must stay bit-identical to its 2026 porting state forever.

**This is presented as an open decision, not a recommendation, per the
owner's direction**:

- *Reading A — uBooNE stays frozen forever.* The 402-CHECK doctest keeps its
  full meaning indefinitely; §5c never happens; cleanup tops out at §4
  (remove kind-2 dead code) + §5a/b (consolidate cfg, remove boilerplate).
  Lower ceiling, zero re-baselining risk.
- *Reading B — uBooNE may be re-baselined once, at a chosen point.* The
  ~229-key operating point (or a settled subset of it) becomes the new
  default, the doctest is regenerated against the new baseline in one
  reviewed commit, and every future OFF-branch that will never be exercised
  again can be deleted rather than merely defaulted off. Highest ceiling for
  actual code reduction, but it is a one-time, deliberate, fully-gated event
  — not a background process — and it changes what "byte-identical" has
  meant on this branch since its start.

## 6. The ledger format

Every knob that is removed under §4's kind-2 rule gets one line, generated —
never hand-transcribed (Trap 1, §3.2) — from the current jsonnet plus `git
log --follow` on the knob name:

```
name | component | originating_doc(§) | add_commit | remove_commit | verdict_class | one_line_why | was_doctest_pinned
```

Example (illustrative, not yet generated):

```
other_seg_uncover_3d | TaggerCheckNeutrino | pr/102 §... | 8c128dc0 | <future> | ADVERSE | 23/N movers ADVERSE on the owner mcp1k census | no
```

Proposed home: `sbnd_xin/docs/77_knob-ledger.tsv`, regenerated (not hand-edited)
each time a removal round runs, so it never drifts from the source the way
doc prose has (§3.2 Trap 1).

## 7. Sequencing (nothing executed by this doc)

- **Phase 0 — generated census, no code.** Every knob × {component, C++
  default, SBND value, originating pr/NN doc, verdict class, doctest-pinned?}.
  Worth doing on its own even if nothing else in this doc happens — it is
  what makes every later phase's decision cheap instead of a fresh
  archaeology exercise.
- **Phase 1 — drop the closed-negative nulls (population 4 ∩ kind-2).** Pure
  plumbing deletion of TLAs nobody overrides; compiled-JSON-identical by
  construction. The cheapest real reduction available, and it needs no
  physics gate at all — only the compiled-config identity proof (§2.5-style).
- **Phase 2 — retire the measured-negative code branches (§4 kind 2).** One
  commit per family; each carries an OFF-gate PASS on the standard manifest,
  a `doctest_clus_knob_defaults.cxx` amendment (doc 70's own rule: a moved
  default is a stop-and-ask, so the corresponding `CHECK`s must be dropped
  deliberately, not left to rot), and a ledger line (§6).
- **Phase 3 — config consolidation (§5b1)**, with optional §5b2 boilerplate
  removal if the owner takes that M10 call. Compiled-JSON-identity gate only.
- **Phase 4 — promote settled knobs (§5c)**, only if and when the owner
  chooses Reading B in §5c. Full A/B gate per family, same bar as any other
  behavior-affecting change (CLAUDE.md §4).
- **Prerequisite that applies across every phase that touches production
  numbers** — regenerate the stale valfast/1000 population gate (§3.3,
  `pr/32:1560`); ten currently-shipped knobs' validation rests on it.

## 8. Open decisions for the owner

1. **§5c** — is the uBooNE chain a frozen reference forever, or a historical
   baseline that may be deliberately re-baselined once? This is the one
   decision that gates the largest possible cleanup; the rest of this doc's
   recommendations (§4, §5a, §5b) do not depend on it.
2. Scope of Phase 0's census — worth running as its own round regardless of
   how #1 resolves?
3. Whether §5b2 (the 290-line mirror-block removal) is in scope given M10, or
   left alone as "working code, don't touch."
4. Timing — should any of this happen before or after the final production
   run, given §3.3's population-gate prerequisite?

## 9. Round 1 executed (2026-08-24)

Owner: *"safe to remove the clearly negative ones, but leave the debugging
knobs in for now"* — confirmed the kind-2 taxonomy (§4) and requested
implementation. This section records Phase 2 (§7) for exactly the 10 knobs
with a closed, measured-negative verdict; nothing else was touched (the 6
diagnostic knobs, the 37 `WCT_*` env probes, and every deferred/mixed-
evidence knob stay).

| Knob | Verdict | Source |
|---|---|---|
| `graph_endpoint_strict` | must stay OFF — false positive as placed; 22/48 events change, 5 nue lost vs 1 gained | pr/30 P8; pr/86:450 |
| `shower_connect_protected_pion_guard` (F13) | measured dead, never flipped | pr/40:1459 |
| `dl_vtx_swap_guard` | live A/B **−36/1014** (6 fixed / 42 regressed); rider closed | pr/89 r5; pr/100:113 |
| `dl_vtx_topo_weight` / `dl_vtx_topo_center` (Arm C2) | live A/B **−8/1014** | pr/89 C2 / r5 |
| `other_seg_uncover_3d` (P2) | **23 ADVERSE** movers, stays OFF | pr/102 r2 |
| `teb_second_max` | negative on its own motivating events; superseded by `teb_chain_topology` | pr/90 §8.5, :714, :984 |
| `pf_touch_cross_main` / `pf_touch_cross_max` (F1 rung 2) | **zero movers** on all 7 census candidates — F1.0 probe failure | pr/84:607, :622 |
| `mvga_carry_max` | not needed — class A cleared 8/8 with it OFF | pr/83 r3 §8.5, :835, :854 |

**Why this is not a behavior change.** All 10 were key-suppressed in jsonnet
when off (`[if k then …]` / `!= null then …`, verified per knob before
editing), so the compiled SBND JSON never carried them; their C++ defaults
(`false` / `0`) made every guarded branch unreachable. Deletion is
unreachable-code removal, held to the full byte-identity bar anyway
(CLAUDE.md §4).

**Scope, full depth (owner's choice, not minimal).** Removed the 10 config
keys and their guarded branches, plus the scaffolding that existed only to
serve them: `GraphEndpointPolicy::strict` and the `endpoint_refused` counter
(the `endpoint_mismatch` tripwire and `graph_endpoint_tol` stay — that is the
kept diagnostic); `TopoVote`/`topo_rule1_vote()` in `NeutrinoVertexFinder.cxx`;
the vertex-scoreboard fields `s_topo`/`topo_frac`/`topo_votes`/`topo_used`/
`topo_weight`/`topo_center`/`skipped_by_swap_guard` (every other scoreboard
field is untouched — a kept diagnostic). One accepted, understood consequence:
new diagnostic calib JSON (`PrDisplayDump`) no longer emits the always-`false`
`skipped_by_swap_guard` key; the `topo_*` keys were already absent with the
knob off, so that half is a no-op. All `dl_vtx_training/*.py` readers use
`.get(...)` with falsy defaults, so both archived and new dumps still parse.
Six 6-CHECK-line doctest pins dropped deliberately (doc 70's rule — a removed
default must show up in the diff, never silently); the `pr30 P8 strict mode`
`TEST_CASE` (only test of the removed refusal) removed; the preceding
non-strict tripwire `TEST_CASE` is untouched.

**Files touched** — 19, all `clus/` + `cfg/pgrapher/{common,experiment/sbnd}/`
(toolkit repo) plus `sbnd_xin/run_pr_chain_batch.sh` (this repo, M9): the 7
env-var passthroughs (`SBND_DL_VTX_SWAP_GUARD`, `SBND_MVGA_CARRY_MAX`,
`SBND_TEB_SECOND_MAX`, `SBND_OSEG_UNCOVER_3D`, `SBND_SHOWER_CONNECT_
PROTECTED_PION_GUARD`, `SBND_PF_TOUCH_CROSS_MAIN`, `SBND_PF_TOUCH_CROSS_MAX`,
`SBND_DL_VTX_TOPO_WEIGHT`/`_CENTER`) removed so a stale env var can no longer
reference a jsonnet TLA that no longer exists. No other experiment's cfg
touched; no C++ default changed (escalation rule 1); nothing under
`work/`/`abtest/snap/`/`decisions*/` touched or regenerated.

**Verification.**
- Compiled-config: unreachable by construction (verified per knob before
  editing — see "why this is not a behavior change" above).
- `./build/clus/wcdoctest-clus`: **232/232 cases, 2379/2379 assertions,
  0 failed** (fresh install, `local/lib/libWireCellClus.so` 2026-08-24 01:15,
  after all source edits).
- Byte-identity gate, base (pre-edit HEAD) vs after (this round), 308 events
  = 241 mcp1k (first 241 sorted `work-mcp1k-prod0823` events) + 48 nueCC48 +
  19 NCpi0, `PR_JOBS=8`, manifest saved at
  `$SCRATCH/knob-rm10/gate308.txt`:
  - `pr85_hash_gate.py` (mabc-pr.zip + pctree member hashes): **PASS**,
    482+96+38 = 616 archives byte-identical, 0 missing/unpaired.
  - `pr94_root_gate.py` (every `tracking-pr.root` branch): **PASS**,
    241+48+19 = 308 events identical, 0 differing.
  - `nusel-table.tsv` diff (sorted): **0 lines** on all three samples.
  - Arm labels: `base-{mcp1k,nuecc48,ncpi0}` vs `rm10-{mcp1k,nuecc48,ncpi0}`
    under the session scratchpad (not a `work-*` label).

**Ledger** — `sbnd_xin/docs/77_knob-ledger.tsv` (new), one row per knob:
the 10 above plus the 7 already deleted before this round (`225d7e7e`,
`df40b2a4`, `ff541919`, `60bad894`), so the ledger is the complete removal
record, not just this round's.

§7 **Phase 2 status: DONE** for the kind-2 population identified as of
2026-08-24. Phase 0's full census (the un-audited remainder of the 121-null
population) would be needed before calling kind-2 exhaustively covered.
