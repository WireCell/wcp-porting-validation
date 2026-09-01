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
  **CLOSED 2026-09-01: the owner chose Reading A (§8 #1, §11.5). Phase 4 will
  not run. The sequence tops out at Phase 3.**
- **Prerequisite that applies across every phase that touches production
  numbers** — regenerate the stale valfast/1000 population gate (§3.3,
  `pr/32:1560`); ten currently-shipped knobs' validation rests on it.

## 8. Open decisions for the owner

1. **§5c** — is the uBooNE chain a frozen reference forever, or a historical
   baseline that may be deliberately re-baselined once? This is the one
   decision that gates the largest possible cleanup; the rest of this doc's
   recommendations (§4, §5a, §5b) do not depend on it.
   **RESOLVED 2026-09-01 → Reading A: uBooNE stays frozen, no re-baselining.
   §5c never happens. See §11.5 for the consequences.**
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

## 10. Round 2 executed (2026-08-24) — §5b1 consolidation, cfg only

Owner: *"I believe we have removed the bad knobs and their implementations.
Now, it is time to consolidate the knobs."* Two scope decisions, taken before
any edit:

- **Plumbing only.** Every TLA name and every runner flag keeps working exactly
  as it did. The 373-TLA signature of `wct-pr-perevt.jsonnet` is **untouched**,
  so no historical `-A` recipe, arm script, or doc-quoted invocation breaks.
- **§5b2 (the 290-line `pattern_algos.m_X = m_X;` mirror block) stays.** No
  `.cxx`/`.h` is touched in this round — doc 68's precedent, so no rebuild, no
  `wcdoctest`, no abtest/qlport *binary* gate. (It is also not the pure copy
  §5b2 assumed: ~half its lines carry `* units::cm` conversions, so removing it
  means relocating a unit conversion inside a shared production file — M10.)

### 10.1 What was actually there: four hand-written layers, not three

§2.6 counted six restatements per knob. Measured on the tree at `2cd25e82`, the
PR path was worse than the table suggested — `pr()` and `clus_pr()` were the
*same signature twice*:

| layer | file | size |
|---|---|---:|
| job TLAs | `sbnd/wct-pr-perevt.jsonnet:43-2469` | 373 |
| job → `pr()` | `wct-pr-perevt.jsonnet:2502-2871` | 368 kwargs, 351 pure `x=x` |
| `pr(...)::` | `sbnd/clus.jsonnet:2985-3632` | 394 params |
| `pr()` → `clus_pr` | `sbnd/clus.jsonnet:3633-4023` | 399 kwargs, **zero** non-`x=x` |
| `clus_pr(...)` | `sbnd/clus.jsonnet:719-1836` | 400 params |
| `clus_pr` → components | `sbnd/clus.jsonnet:1837-2915` | one `x=x` per knob |
| `tagger_check_neutrino(...)` | `common/clus.jsonnet:483` | **336 params on one 9147-char line** |
| its suppression clauses | `common/clus.jsonnet:484-1446` | 364 `+ (if …)` |

`clus_pr` had exactly one caller; `tagger_check_neutrino` has exactly two
(SBND's `pr()` and `qlport/uboone-mabc.jsonnet:1258` — PDHD/PDVD never call it).

### 10.2 What changed

**Step 1 (`436915f2`) — merge `clus_pr()` into `pr()`.** The kept signature is
`clus_pr`'s, the richer one (its per-knob comments carry the doc reference and
the C++ default), with `pr()`'s *public* defaults preserved verbatim where the
two differed (`dump`, `cathode_x`, `cathode_kink_xcut`,
`cathode_wide_kink_angle`, `protect_cathode_x` and the three
`protect_cathode_rejoin_*` lengths), so no caller sees a different default. The
two `name: 'clus_pr'` component names are string literals in the compiled config
and are untouched. **−1039 lines.**

**Step 2 (`b5384c2c`) — one knob bag into `TaggerCheckNeutrino`.** 293 knobs
now travel in a single object:

- `common/clus.jsonnet`: `tagger_check_neutrino` **336 → 44** named parameters
  plus `knobs={}`, merged last into `data`; the 293 clauses are gone. An absent
  key still means the C++ default — which is exactly what those clauses said.
- `sbnd/clus.jsonnet`: `pr()` **394 → 107** params, gains `tcn_knobs={}`. Five
  knobs (`cathode_x`, `cosmic_consistent_fv`, `mip_dqdx`,
  `neutrino_type_bitmask`, `nue_sp_consistent_fv`) are read elsewhere in `pr()`,
  so they stay named parameters and `pr()` adds them to the bag itself.
- `wct-pr-perevt.jsonnet`: TLA signature unchanged; its 370-line `pr()` call
  becomes an 80-line call plus a 288-entry `local tcn_knobs`, one line per knob,
  carrying the family comments. The 11 knobs the job passes as an *expression*
  (the four `pr_y_top` offsets, the seven null-coalesced `kine_*` flags) keep
  that expression.
- `qlport/uboone-mabc.jsonnet`: `fit_exclusion` and `dqdx_fit_keep_all_points`
  move into an inline two-entry bag.

**The rule for what gets absorbed:** a knob is absorbed only where **the job's
TLA is its single documented home**. That keeps the prose where it is richest
(the job's TLA block averages ~5.5 comment lines per knob) and never orphans a
knob whose only description lives at a lower layer. Deliberately **not**
absorbed, and unchanged by this round:

- the **20** knobs `pr()` declares but the job never sets — the whole `teb_*`
  family, `stm_proton_*`, `kink_dqdx_hot_ratio`, `dir_weak_use_score`,
  `endpoint_trim_retry`, `mip_dqdx_median`, `proton_dir_vote`, …;
- the **3** that are not `pr()` parameters at all — `fiducial`,
  `proton_dir_score_max`, `proton_dir_asym_min`;
- everything with a **compound predicate** — `nu_per_bundle` and its two riders,
  `nu_skip_cosmic_bundle_min_length` (`> 0`), `fv_tolerance`
  (`std.length(…) > 0`).

**Deliberate deviation from §5b1's wording.** The bag keys are the **existing
C++ key names, grouped visually by family**, not doc 77's nested-and-renamed
fields (`two_end_break: {enabled, min_len, …}`). A rename table would be a new
per-knob restatement and a new silent-failure surface — the very thing this
round removes. The payoff §5b1 asked for is still delivered: "what did
production run" is one object, and `scripts/cfg/compile_prjob_cfg.sh` hashes it.

**Step 3 was evaluated and skipped.** The other knob-taking builders were
measured for the same treatment: `tagger_check_stm` (11 absorbable),
`tagger_check_tgm` (8). Both were left alone, on purpose. Their parameter names
differ from the job's TLA names (`stm_accept_guards` → `accept_guards`,
`tgm_component_extremes` → `component_extremes`), so absorbing them would
require exactly the rename table rejected above; and four of the nineteen
(`beam_window_only`, `evaluate_demoted_mains`, `require_in_scope`,
`save_stm_fit`) genuinely fan out to several components, which a named
parameter expresses better than a bag. ~19 knobs and ~95 lines are not worth
either cost.

### 10.3 Net

| file | lines |
|---|---|
| `cfg/pgrapher/common/clus.jsonnet` | 2301 → 1459 |
| `cfg/pgrapher/experiment/sbnd/clus.jsonnet` | 4052 → 2460 |
| `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` | 2911 → 2923 |
| **cfg total** | **+1169 / −3591 = −2422** |

Restatements per absorbed knob: **6 → 2** (its TLA declaration, and one line in
the bag). A knob that is added from now on needs no plumbing at all — any C++
key is reachable by naming it in the bag.

The 373 job TLAs, their defaults, and their comments are bit-for-bit what they
were.

### 10.4 Verification

Three gates, each run against a pristine tree extracted from the pre-round
commit (`git archive 2cd25e82 cfg | tar -x`) and the working tree.

**Gate 1 — compiled-config identity, 21/21 byte-identical.** Every live consumer
of the SBND/common clustering config, `cmp` on the compiled JSON (jsonnet emits
object keys sorted, so source reordering cannot hide in it):

| what | result |
|---|---|
| SBND PR job at the **production operating point** (`compile_prjob_cfg.sh`: full PR pipeline + BDTs) | identical |
| SBND PR job bare (default pipeline) | identical |
| `sbnd_pr`, `sbnd_img`, `sbnd_clus`, `sbnd_ql` (abtest `compile_all_cfg.sh`) | identical |
| `wcls-img-clus.jsonnet`, legacy standalone Q/L (`compile_sbnd_prod.sh`) | identical |
| **uBooNE MABC** (`qlport/scripts/compile_ub_cfg.sh`) | identical |
| PDHD nfsp / img / clus, PDVD nfsp / img / clus | identical ×6 |
| PDHD / PDVD / SBND sim-check, PDHD sim-track + sim-noise, PDVD sim-track | identical ×6 |

**Gate 2 — per-TLA probe sweep, 373/373 and 46/46 identical.** The production
compile only exercises the knobs that are ON; the 115 `null` + 15 `false` TLAs
emit no key, so a broken forward for any of them would pass Gate 1 silently —
M6 with a 130-knob blast radius. Every TLA is therefore set **in turn** to a
distinctive non-default value and compiled against both trees, which must agree
(including agreeing on the failure text where a probe value is illegal):

- PR job: 373 TLAs — 9 already supplied by the base invocation, 364 probed,
  **373/373 identical**. Coverage check on the pristine tree: **359 of the 373
  probes visibly move the compiled JSON**. The five that do not are `DL`, `DT`,
  `lifetime`, `driftSpeed` (documented inert at `wct-pr-perevt.jsonnet:57-73` —
  they feed only the sim Drifter) and `neutrino_consistent_fv`, which at this
  operating point only ever appears inside `neutrino_consistent_fv ||
  cosmic_consistent_fv || nue_sp_consistent_fv` with the other two already ON.
- Q/L job (`wct-clus-matching-perevt.jsonnet`, untouched but downstream of
  `common/clus.jsonnet`): **46/46 identical**.
- uBooNE knob probes: `dir_weak_use_score`, `fit_exclusion`,
  `dqdx_fit_keep_all_points`, each ON and OFF, pristine `uboone-mabc.jsonnet` +
  pristine cfg vs the new pair — **6/6 identical**, key present exactly when ON.

**Gate 3 — physics, the owner's samples.** 167 PR-chain events, `reality=data`
on all three (the `-ql0819` Q/L roots carry no `.lineage_reality`, so the
runner's check is a silent no-op — doc 78 was bitten by this), `PR_JOBS=8`,
all `rc=0`. A side = the doc-79 arms, which sit at the last PR-affecting commit
(`c8e0b9f5`; `2cd25e82` moved only the Q/L job):

| sample | events | `pr85_hash_gate.py` | `pr94_root_gate.py` | sorted `nusel-table.tsv` diff |
|---|---:|---|---|---|
| numuCC (mcp1k, first 100 of the doc-77 241) | 100 | **PASS** 200 archives | **PASS** 100 events | **0 lines** (1171 rows) |
| nueCC48 | 48 | **PASS** 96 archives | **PASS** 48 events | **0 lines** |
| NCpi0 | 19 | **PASS** 38 archives | **PASS** 19 events | **0 lines** |
| **total** | **167** | **334 archives byte-identical** | **167 events identical** | **0** |

Arms: `$SCRATCH/cons/c2-{mcp1k,nuecc48,ncpi0}` vs
`$SCRATCH/dgtail/dg79pr-{mcp1k,nuecc48,ncpi0}`; event list
`$SCRATCH/cons/numu100.txt`.

Not run, and why. **No C++ was touched**, so there is no `wcbuild`, no
freshness proof, no `wcdoctest-clus`, and no abtest/qlport *binary* gate;
`common/clus.jsonnet` is shared, which is exactly why uBooNE, PDHD and PDVD
appear in Gate 1 as compile checks. **The doc-68 runner-argv capture** (shim
`wire-cell`, compile what each runner really passes) was planned and dropped:
it proves "no runner flag lost its TLA", and Gate 2 already proves that
strictly harder — it fails outright if the TLA name set moves, and it compiles
*every* TLA individually against both trees, whereas the shim only exercises
the ones a runner happens to set. The bare-job compile in Gate 1 pins the
defaults for the rest.

**Third-party callers.** `tagger_check_neutrino(` and `.pr(` were grepped
across both repos: exactly three call sites exist — the SBND job and
`qlport/uboone-mabc.jsonnet` (both rewritten here, both gate-proven) and the
stale file in §10.7. The three other importers of `common/clus.jsonnet`
(`sbnd/sbnd_abhat/`, `sbnd/obsolete/`, `pdhd/old/`) call neither, so nothing
they use changed.

### 10.5 Tooling added (`scripts/cfg/`)

- `compile_consumers.sh <cfgroot> <outdir>` — compiles all 21 consumers above
  against an arbitrary cfg tree.
- `cmp_consumers.sh <dirA> <dirB>` — exact `cmp` of the two output dirs.
- `tla_probe_gate.py <cfgA> <cfgB> [--job pr|ql] [--jobs N]` — the per-TLA probe
  sweep. **Any future cfg refactor should run all three**; Gate 1 alone cannot
  see the OFF knobs.

### 10.6 Repro

```bash
T=/nfs/data/1/xqian/toolkit-dev/toolkit
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
D=/home/xqian/tmp/cons                      # scratch
mkdir -p $D/pristine && git -C $T archive 2cd25e82 cfg | tar -x -C $D/pristine

$SX/scripts/cfg/compile_consumers.sh $D/pristine/cfg $D/before
$SX/scripts/cfg/compile_consumers.sh $T/cfg          $D/after
$SX/scripts/cfg/cmp_consumers.sh     $D/before $D/after            # 21/21 identical

$SX/scripts/cfg/tla_probe_gate.py $D/pristine/cfg $T/cfg --job pr --jobs 24
$SX/scripts/cfg/tla_probe_gate.py $D/pristine/cfg $T/cfg --job ql --jobs 12

cd $SX                                       # physics, 167 events
PR_JOBS=8 ./run_pr_chain_batch.sh work-mcp1k-ql0819   $D/c2-mcp1k   data $(cat $D/numu100.txt)
PR_JOBS=8 ./run_pr_chain_batch.sh work-nuecc48-ql0819 $D/c2-nuecc48 data $(cat <48-evt list>)
PR_JOBS=8 ./run_pr_chain_batch.sh work-ncpi0-ql0819   $D/c2-ncpi0   data $(cat <19-evt list>)
for s in mcp1k nuecc48 ncpi0; do
  python3 scripts/pr85_hash_gate.py --jobs 8 $D/c2-$s <dg79pr-$s>; python3 scripts/pr94_root_gate.py $D/c2-$s <dg79pr-$s>
done
```

### 10.7 Known-stale file, unchanged in kind

`wcp-porting-img/sbnd/wcls-img-clus-matching-xin.jsonnet` (HaiwangYu's, doc 68
§5) does not compile before **or** after, and fails at the same line both ways
(`function has no parameter rse_from_metadata`). If it is ever repaired it will
additionally need its `iso_endpoint` argument moved into the knob bag.

§7 **Phase 3 status: DONE** for the PR path. The Q/L path
(`wct-clus-matching-perevt.jsonnet`, 46 TLAs, zero physics-knob overlap with the
PR job) was not consolidated — there is nothing there to consolidate. §5b2 and
§5c remain open decisions (§8).

---

## 11. Round 3 — planned (2026-09-01), post EM/pi0 campaign

Status: **PLAN. Nothing executed, nothing deleted, no knob moved.** The work
runs in the next session. Toolkit unchanged at `ddce7430`.
**EXECUTED 2026-09-01 — see §12**, which carries the results and corrects four
defects found in this section and in §4/§9 while executing it (§12.6).  In
particular §11.3.1's recommendation for `shower_samevtx_track_absorb` is
**overturned by measurement** — the knob was NOT deleted (§12.3).

Owner asked whether a cleanup round like round 1 (§9) / round 2 (§10) is worth
repeating after the EM/pi0 campaign (docs pr/117 → pr/142), and **answered §8
decision #1 in the same breath**:

> *"The uBooNE chain should be a frozen reference, I think, we do not want to
> re-baseline it, since we may not have sufficient effort to do the
> validation."*

That is **§5c Reading A**, and it is now settled policy for this branch — see
§11.5 for what it forecloses and what it leaves on the table.

### 11.0 Repro

```bash
T=/nfs/data/1/xqian/toolkit-dev/toolkit
F=cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# knob-surface growth across the campaign window (§11.1)
for r in 8d93260d ddce7430; do git -C $T show $r:$F \
  | awk '/^function\(/,/^\)/' | grep -oE '^\s+[a-z_][a-z0-9_]*\s*=' \
  | tr -d ' =' | sort -u | wc -l; done                                   # 377 -> 501
for r in 8d93260d ddce7430; do git -C $T show $r:clus/src/TaggerCheckNeutrino.cxx \
  | grep -cE '^\s*pattern_algos\.m_[a-z0-9_]+\s*='; done                 # 285 -> 387
for r in 8d93260d ddce7430; do git -C $T show $r:clus/test/doctest_clus_knob_defaults.cxx \
  | grep -c CHECK; done                                                  # 400 -> 513
for r in b5384c2c 8d93260d ddce7430; do git -C $T show $r:$F \
  | sed -n '/local tcn_knobs = {/,/^    };/p' \
  | grep -cE "^\s+(\[if .*then )?'?[a-z_][a-z0-9_]*'?\]?\s*:"; done      # 288 -> 292 -> 408

# the fire census (§11.3) -- the generator, not a hand count
cd $SX && scripts/cfg/fire_census.py work-mcp1k-prod0901 work-mcp2k-prod0901 \
    work-ncpi0-prod0901 work-nuecc48-prod0901 --tsv docs/77-firecensus-prod0901.tsv

# the one live case (§11.3.1), three arms, same event
for a in work-pr125r1-flipK598-ncpi0 work-ncpi0-empre0901 work-ncpi0-prod0901; do
  printf '%-34s samevtx=%s\n' $a "$(grep -c samevtx $a/pr_evt37112/wct_pr_evt37112.log)"; done
```

### 11.1 What the campaign cost the knob surface

| measure | `8d93260d` (pre) | `ddce7430` (post) | delta |
|---|---:|---:|---:|
| SBND PR job TLAs | 377 | 501 | **+124** |
| … shipped ON (`true` or a value) † | — | 315 | — |
| … shipped OFF (`false` or `null`) † | — | 186 | — |
| … of which `false` | 17 | 26 | +9 |
| `pattern_algos.m_X =` mirror lines | 285 | 387 | **+102** |
| `doctest_clus_knob_defaults.cxx` CHECKs | 400 | 513 | +113 |
| `WCT_*` env probes | 39 | 51 | +12 |
| `clus/` source lines | 125,781 | 133,770 | +7,989 |
| knobs **removed** | — | — | **0** |

† **Mechanical split by literal value, not a feature count.** It reads every
non-`false`/`null` TLA as ON, which sweeps in sub-parameters on both sides — a
`null` tuning param under an ON parent lands in the OFF column, and a numeric
sub-param of an ON feature lands in the ON column. See §11.2: null-ness is
*not* the discriminator. Treat 315 as "TLAs carrying a value", never as
"315 live features"; the same caveat applies to `fire_census.py`'s coverage
line, which quotes this number.

Round 1 retired 10; the campaign added 124 and retired none, so the surface is
~33 % larger than at the last cleanup. Four values were re-tuned rather than
added: `kine_shower_fudge_factor` null→0.86, `mcs_enable` false→true,
`mcs_muon_source` `'pf_muon'`→`'long_muon_else_pf'`, `sccc_max_gap` 6→10.
(`mcs_enable` is the family *gate* — it appears in the bag only as the
condition in `[if mcs_enable then 'mcs_muon_source']`, never as an entry of
its own.)

**The cfg side needs no work.** Round 2's consolidation held and the campaign
used it correctly: `tagger_check_neutrino` is still 44 named params, and the
`tcn_knobs` bag absorbed the campaign cleanly (**292 → 408 entries**) with
`shower_split`, `kine_shower_fudge_factor`, `shower_pass4_prune_gap2`,
`shower_samevtx_track_absorb` all riding it under the key-suppression idiom.
The flat TLA growth is the job-level signature §10 deliberately left untouched
so no historical `-A` recipe would break. **The debt landed on the C++ side** —
the piece §10 explicitly deferred under M10.

### 11.2 The removal pool is not 124, and not the 56 that look OFF

56 of the 124 added TLAs are `false` or `null`, but a `null` under an ON parent
means *"use the C++ default"*, not *"feature off"* — the eight
`shower_merge_relax_cont_*` nulls are the live tuning surface of
`shower_merge_relax = true`. **The discriminator is the parent boolean**, not
null-ness.

Clean kind-2 candidates, each with the verdict already stated in the tree:

| knob | verdict in cfg comment | dead sub-params to go with it |
|---|---|---|
| `shower_flank_absorb` | "no targets in the marked set" (pr/117 §6) | `_max_dis`, `_max_len` |
| `shower_ex1_conn3_body_dis` | "measured ZERO yield — 1 admit in 98 events" (pr/118 §4a) | — |
| `shower_ex1_walk_em_track_guard` | "measured ZERO yield" (pr/120 §5) | `_len` |

The other added-`false` knobs (`pi0_nv_allow_type2`, `pi0_nv_retry_paired`,
`pi0_reseat_start_assoc`, `pi0_mu_shower_hypothesis`, the three
`shower_split_*`) carry only "C++ default false" — that states a *default*, not
a verdict. Each needs its doc read before classification; none is assumed here.

### 11.3 New instrument: the fire census

`scripts/cfg/fire_census.py` (new), output `docs/77-firecensus-prod0901.tsv`.

The sentinel suite (`scripts/pr127_sentinels.py`) asks *"does this fix still
produce the right answer on ITS event?"*. It cannot ask *"does this fix still
run anywhere?"* — and doc pr/142 §5.3 showed the two questions have different
answers: `406125`'s pr/124 prune stopped firing on its own event while firing
on 70 others. The fire census asks the second question, using the `prNN <tag>:`
log convention, across all 3067 events of `prod0901`.

**The instrumented set is taken from the toolkit source, never from the logs.**
Inverting the log table would silently equate "no tag exists" with "knob is
dead". Three buckets, and only the third is evidence:

- **fires on N of 3067** — 25 tags, `pr55 do_rough_path` (45.1 %) down to
  `pr121 ex1_dedup_rehome` at 1 event, which matches pr/121's recorded
  "sole fire in 239".
- **uninstrumented** — no tag exists. Coverage is **35 tags against 315 ON job
  TLAs**, so silence for an untagged knob is a gap in the instrument, *not*
  evidence.
- **instrumented, ZERO fires** — 10 tags, and a zero is only a finding after
  adjudication:
  - 6 × `pr67 *` are gated by OFF diagnostics (`m_traj_cover_probe`,
    `m_pr_find_other_rounds`) → expected; §4 kind-1 tooling, keep.
  - `pr117 flank_absorb`, `pr139 shower_split` sit under knobs shipped OFF →
    expected, and they are §11.2's removal candidates.
  - `pr84 shower_dedup` → see §11.3.2, **not** a finding.
  - `pr125 samevtx_absorb` → §11.3.1, the one live case.

#### 11.3.1 `shower_samevtx_track_absorb` — ON, inert, superseded in place

Flipped SBND PRODUCTION ON 2026-08-29 (doc pr/125; owner: *"one shower,
everything"*). Its cfg comment records that it fires on exactly 2 fragments,
both in NCpi0 event **37112**, which **is** in the production sample.

| arm | knob | `samevtx` fires | showers | id 67048 `kine_best` / `kine_dQdx` |
|---|---|---:|---:|---|
| `work-pr125r1-flipK598-ncpi0` (validation) | ON | **1** | 8 | 797.3 / 840.9 |
| `work-ncpi0-empre0901` (campaign OFF) | OFF | 0 | **13** | 511.3 / 356.8 |
| `work-ncpi0-prod0901` (production) | ON | **0** | 8 | 741.7 / 840.9 |

The campaign-off arm is the discriminator. With the campaign off the merge is
genuinely absent (13 showers, 511.3 MeV), so the merge is a real campaign
effect and the knob was **not** already redundant when it was flipped. In
production the merged result is back with identical composition to the
validation arm (`kine_dQdx` 840.9 both, same 8 shower ids) — **but this knob's
code path never runs.** Another campaign knob now produces the same merge.

This is the *benign* form of the `406125` pattern: redundant, not broken. §4's
taxonomy has no bucket for it — all four kinds describe OFF knobs. It needs a
fifth: **ON but inert / superseded in place**.

*Not attributed, and it does not bear on the merge claim:* `kine_best` 797.3 →
741.7 between the validation and production arms. Composition is identical
(`kine_dQdx` unchanged), so this is charge accounting — the 0.86 scale and doc
85 r2's excluded-energy census both landed in between — not a merge change.
Isolating it was not attempted.

#### 11.3.2 `shower_dedup_start_seg` — zero fires, but NOT the same class

`pr84 shower_dedup:` also never fires and its knob is ON, but the emit site
sits inside `if (group.size() < 2) continue;` — it fires only when two showers
share a start segment. Zero fires is therefore a statement about the **input**
(no duplicate start segments in 3067 events), not about the knob, and it has no
recorded motivating event to check against. Listed for completeness; **not**
part of the plan.

### 11.4 The kind-3 knobs are now due

§4 kind 3 parked five knobs as *"keep, but timestamp"*, with the explicit
horizon *"next full-1k census"*:

`dqdx_fit_keep_all_points`, `dl_vtx_cloud_no_exclusion`,
`main_vertex_swap_apply`, `fit_blob_coverage_defer`, `teb_chain_topology`.

**doc pr/142 is that census.** Each is now promote / reject / re-park with a
fresh horizon. This is the cheapest item on the board: the commitment already
exists, the trigger has fired, and no new instrument is needed.

### 11.5 §8 decision #1 — RESOLVED: Reading A, uBooNE stays frozen

The owner's answer (quoted above) settles the one decision that gated the
largest possible cleanup. Consequences, stated plainly so no future round
re-opens it by accident:

**What it forecloses.** §5c never happens. Settled ON knobs are **not** promoted
to unconditional code, because hard-coding an SBND-ON knob would move uBooNE's
C++-default output for the first time since the porting reference was
established (§2.5) — and the owner's stated reason is precisely that the
validation effort for that is not available. The 513-CHECK defaults doctest
keeps its full meaning indefinitely. **"Key absent ⇒ pre-knob behavior" remains
an invariant of this branch**, which is what every A/B gate here rests on.

**What it leaves.** The ceiling is §4 kind-2 removal + §5a/§5b:

- §11.2's three closed-negative knobs — removable, and *cheaper* under Reading
  A than under B: they are C++-defaulted false and set by no compiled config,
  so removal is compiled-config-identical for SBND **and** uBooNE.
- §11.3.1's ON-but-inert knob — note the asymmetry Reading A creates. An
  ON-but-inert knob can now be retired **only by deleting the feature**, never
  by hard-coding it ON (that is a §5c move and is now off the table). Deleting
  is the provable direction anyway: the knob is measured inert on 3067 events,
  so an OFF-gate PASS is the expected result rather than a hope.
- **§5b2, the mirror block, is now the largest remaining mechanical win** and
  is *unaffected* by Reading A — it is a representation change, byte-identical
  by construction, and touches no default. It grew +102 lines in one campaign
  (§11.1). It remains §8 decision #3 (M10: a mechanical refactor inside a
  shared production file is the owner's call, not a default action), and §10
  added a real complication: ~half its lines carry `* units::cm` conversions,
  so removing it relocates a unit conversion inside that file.

### 11.6 The plan for the next session, in order

1. **Adjudicate the five kind-3 knobs (§11.4).** Read each one's originating
   doc and the pr/142 population read-out; promote, reject, or re-park with a
   dated horizon. Output: five ledger-ready verdicts. No code.
2. **Retire the three kind-2 knobs (§11.2)** + their dead sub-params, by §9's
   recipe: delete C++ + all plumbing layers + the `doctest_clus_knob_defaults`
   CHECKs, then `wcdoctest-clus` green, `pr85_hash_gate.py` PASS on the
   standard manifest, and a **generated** ledger line each (§6 Trap 1 — the
   ledger is generated from jsonnet + `git log --follow`, never hand-typed).
3. **Adjudicate `shower_samevtx_track_absorb` (§11.3.1).** Identify which knob
   took the merge over — bisect the campaign flips on 37112 — *then* delete the
   inert one behind a full A/B. Do not delete before the successor is named:
   "inert on 3067 events" is a measurement on this sample, not a proof that the
   path can never fire.
4. **Extend the instrument (§11.3).** Every knob flipped ON from now on ships
   with a `prNN <tag>:` line, so the fire census can see it; 35-of-315 coverage
   is the real limit on this round's conclusions. Pair the two instruments in
   the standard round checklist: sentinel = *still right on its event*, fire
   census = *still fires at all*. Together they would have caught both `406125`
   and §11.3.1 in the week they happened rather than at campaign close.
5. **Re-run the fire census against the OFF arm too.** This round measured only
   `prod0901`. `empre0901` exists and is free to scan; a tag that fires in both
   arms is not campaign-attributable, which sharpens every future adjudication.

### 11.7 Not in scope

- Anything under §5c (Reading A, §11.5).
- Any knob **default** change, any re-tune, any flip — §11 removes measured-dead
  wiring and nothing else. CLAUDE.md §5.1 governs the rest.
- The Q/L job (`wct-clus-matching-perevt.jsonnet`) — §10.7 already found
  nothing there to consolidate.
- `406125` itself (doc pr/142 §5.3): a shipped fix that no longer fires on its
  own event is a **sentinel** adjudication, not a cleanup item. It stays on the
  pr/142 §7.2 list.

---

## 12. Round 3 executed (2026-09-01) — §11.6's five items

Status: **DONE.** Three kind-2 knobs retired (toolkit `6f30c079`); the fourth
candidate, `shower_samevtx_track_absorb`, was **measured and kept** — §11.3.1's
"delete it" reading is overturned (§12.3). Six kind-3 knobs adjudicated, no
knob default moved, no flip, nothing under `work/`/`abtest/snap/` rewritten.

Owner: *"Can you start to prepare the code for the production running?"* — with
§11.6's five items named in order, and two scope decisions taken before any
edit: retire **only** §11.2's three (newly-evidenced closed-negatives get
documented, not deleted, on today's adjudication), and for the samevtx knob
*name the successor first, delete only behind a full A/B*. That second decision
is what stopped a deletion the evidence turned out not to support.

### 12.0 Repro

```bash
T=/nfs/data/1/xqian/toolkit-dev/toolkit
F=cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
W=<session scratchpad>/knob77r3   # arms, snapshots, manifests, worktree

# knob-surface delta -- §11.0's OWN commands, so before/after are comparable
git -C $T show ddce7430:$F | awk '/^function\(/,/^\)/' \
  | grep -oE '^\s+[a-z_][a-z0-9_]*\s*=' | tr -d ' =' | sort -u | wc -l   # 501 -> 495
git -C $T show ddce7430:clus/src/TaggerCheckNeutrino.cxx \
  | grep -cE '^\s*pattern_algos\.m_[a-z0-9_]+\s*='                       # 387 -> 381
git -C $T show ddce7430:clus/test/doctest_clus_knob_defaults.cxx | grep -c CHECK  # 513 -> 507

# item 5: the OFF-arm census + the four-way classification (§12.5)
cd $SX && scripts/cfg/fire_census.py work-mcp1k-empre0901 work-mcp2k-empre0901 \
    work-ncpi0-empre0901 work-nuecc48-empre0901 --tsv docs/77-firecensus-empre0901.tsv
scripts/cfg/census_ab.py docs/77-firecensus-prod0901.tsv \
    docs/77-firecensus-empre0901.tsv --tsv docs/77-censusab-prod-vs-empre.tsv

# item 4: which shipped-ON knobs the census cannot see (§12.4)
scripts/cfg/tag_coverage.py --tsv docs/77-tagcoverage-prod0901.tsv   # 46 tagged / 269 not

# item 3: the samevtx bisect, one event, leave-one-out from production (§12.3)
#   ctl == prod0901 byte-identical; prefilter OFF => samevtx fires, SAME output;
#   both OFF => the merge is LOST.  That is why the knob stays.
for a in ctl t2-prefilter t2-prefilter-nosv; do
  printf '%-22s samevtx=%s\n' $a \
    "$(grep -c 'pr125 samevtx_absorb:' work-77r3-bis-$a-ncpi0/pr_evt37112/wct_pr_evt37112.log)"
done

# item 2 gates: compiled-config identity, then the 308-event byte-identity gate
scripts/cfg/compile_consumers.sh $W/tk-base/cfg $W/cons-base   # 21 artifacts
scripts/cfg/compile_consumers.sh $T/cfg           $W/cons-rm3
scripts/cfg/cmp_consumers.sh $W/cons-base $W/cons-rm3          # 21/21 identical
for s in mcp1k nuecc48 ncpi0; do
  python3 scripts/pr85_hash_gate.py work-77r3-base-$s work-77r3-rm3-$s; echo "rc=$?"; done
```

### 12.1 Item 1 — the kind-3 knobs, adjudicated (six, not five)

**First, an erratum in §11.4 itself.** §4's kind-3 row parks
`dqdx_fit_keep_all_points`, `dl_vtx_cloud_no_exclusion`, `main_vertex_swap_apply`,
`fit_blob_coverage_defer` and **`mvga_ac_veto_radius`**. §11.4 listed the first
four plus **`teb_chain_topology`** — a silent substitution, made without comment
when §11 was written. Adjudicating "the five" from §11.4 would have dropped a
knob §4 committed to revisiting and picked up one §4 never parked, so the
**union of six** is adjudicated here.

§4's horizon was *"next full-1k census"* and doc pr/142 is that census, so the
commitment is now discharged. §4 carries per-knob evidence for exactly one of
the six, so each verdict below cites the originating doc or the cfg comment —
which, per §3.2 Trap 1, is the record that does not go stale.

| knob | prod | measured evidence | verdict |
|---|---|---|---|
| `dqdx_fit_keep_all_points` | OFF | vertex +1/47, but a **12/47 EM-shower reshuffle** with two large electron-energy losses (81597 Enu 1578→696, 196649 1614→365, nue 4.24→−2.32) — pr/107 §7 | **re-park**, horizon: the next EM-clustering round. The cost lands squarely on what the campaign just spent 25 docs improving. |
| `dl_vtx_cloud_no_exclusion` | OFF | vertex 35→38/47 **but nue-selected 35→32**, churn 11/47 for net +1, +8 % PR wall — pr/112 §5.1 | **re-park**, horizon: any round that makes vertex accuracy an objective in its own right. Owner already ruled it out *as the answer to idea 2* (pr/112:243), which is not a verdict on the knob. |
| `main_vertex_swap_apply` | OFF | **never measured ON** — pr/51 r3 shipped gates only (knob-off JSON identical, key correct when on); a latent-bug fix, not a tested change | **re-park**, horizon: needs a measured arm before any verdict is possible. Cheapest of the six to resolve. |
| `fit_blob_coverage_defer` | OFF | fixes the 172230 partition-reshuffle class; costs 57441 cid 20 ghost 1.12→1.23 cm; pr/51:192 finds it fixes **nothing** in 268067/360535 | **re-park**, horizon: now answerable — pr/51:650 parked its open question on "if the owner flips mvga on", and mvga **is** production ON. |
| `teb_chain_topology` | OFF | cfg:2344 *"STAY OFF: the D1+D3 live A/B was net NEGATIVE"* — **19 ADVERSE vs 6 toward** on harv3 labels, two cosmict flips (pr/90 §10.6) | **re-classify kind 2.** The measurement is closed-negative; only *"keep for a future vertex-anchored redesign"* (pr/90:1265) kept it out of kind 2. → §12.7 pool. |
| `mvga_ac_veto_radius` | OFF | cfg:2592 *"stays OFF: 0.2 cm measured **ADVERSE**"* — kills the 349945 design case, re-confirming pr/86 Stage A's deliberate 1.0 cm relax | **re-classify kind 2.** §4 filed it kind 3 on the strength of pr/99:339's *"knob retained for future scans"*, which is an intent, not evidence. → §12.7 pool. |

Two structural facts worth recording with the verdicts:

- **`teb_chain_topology` is not independently flippable.** Its use site
  (`NeutrinoPatternBase.cxx:3035`) also requires `m_teb_r3_turn > 0 &&
  m_teb_r3_hot > 0`, and both are OFF. Flipping the knob alone is a no-op, so it
  could never have been A/B'd on its own — the "deferred, needs a measurement"
  filing was unreachable as written.
- **None of the six has a `prNN <tag>:` line**, so none is visible to the fire
  census. Worse, `dqdx_fit_keep_all_points`'s nearest emit
  (`TrackFitting.cxx:9074`, *"pre-dQ/dx form_map_graph dropped N ... points"*)
  is guarded by `n_fits_after != n_fits_before` — it fires when the knob is
  **OFF** and goes silent when it is ON. A log-grep for "did it fire" returns
  the answer backwards. This is item 4's motivating case (§12.4).

### 12.2 Item 2 — three kind-2 knobs retired

**Why this is not a behavior change.** All three are C++-defaulted `false`,
key-suppressed in jsonnet when off (`[if k then …]`), and — verified across the
whole `cfg/` tree, not assumed from §11.5 — set by **no compiled config of any
detector**. They appear in exactly six files repo-wide, and in cfg only in
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`. Their guarded branches
were therefore unreachable in every arm ever run, and removal is
compiled-config-identical for SBND **and uBooNE** — which §11.5 asserted and
Gate 1b below measures. Held to the full byte-identity bar anyway.

**Scope, and why the three do not share one recipe.** They have genuinely
different shapes, and treating them alike is how a "mechanical" deletion goes
wrong:

| knob | shape | what came out |
|---|---|---|
| `shower_flank_absorb` (+`_max_dis`, `_max_len`) | a knob-exclusive **function** plus a whole `if` block | `flank_absorb_orphans` (83 lines) + its sole call + the declaration. The shared helpers `segment_confident_nonelectron_pid` (8 other callers) and `pr93_probe_absorb_direct` (9 other sites) **stay**. |
| `shower_ex1_conn3_body_dis` | **one behavioural line** inside a measure-first probe | only `min_dis = body_dis;`. The guard `m_shower_ex1_conn3_body_dis \|\| pr91_merge_dbg()` collapses to `pr91_merge_dbg()`, and the probe region is **kept intact** — `scripts/pr118_probe_census.py:65,128` parses its `tag=ex_shower1_p2dis` output. Collapsing the region would have silently broken that census. |
| `shower_ex1_walk_em_track_guard` (+`_len`) | a ternary argument that **cascades out of `clus/`'s knob files** | the 6th argument at the one call site; then `em_straight_min_len` is dead, so `PRShower.h`'s defaulted parameter and `PRShower.cxx`'s `guard_excludes` branch go too. The lambda collapses to the legacy unconditional `if (std::abs(pdg) == 11) return false;`, and the call site now matches the other **eight** verbatim. |

**Files touched** — 8 in the toolkit (`clus/src/{NeutrinoShowerClustering,
TaggerCheckNeutrino,PRShower}.cxx`, `clus/inc/WireCellClus/{NeutrinoPatternBase,
TaggerCheckNeutrino,PRShower}.h`, `clus/test/doctest_clus_knob_defaults.cxx`,
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`) plus
`sbnd_xin/run_pr_chain_batch.sh` in this repo (M9): the **6 env passthroughs**
(`SBND_SHOWER_FLANK_ABSORB` +`_MAX_DIS`/`_MAX_LEN`,
`SBND_SHOWER_EX1_CONN3_BODY_DIS`, `SBND_SHOWER_EX1_WALK_EM_TRACK_GUARD`
+`_LEN`) removed so a stale env var can no longer name a TLA that no longer
exists. Net **62 insertions / 265 deletions**.

Every rationale comment block these knobs sat in is **shared with a
PRODUCTION-ON knob** (`shower_pass4_best_owner`, `shower_merge_relax`,
`shower_merge_relax_continuity`, `stem_backfill_back_guard`), so each was
edited down rather than deleted, and the surviving item numbering ("(1)/(2)/(3)")
was renumbered so the prose still matches the members below it. Three
retirement breadcrumbs point at the ledger from the code that remains.

**Knob-surface delta**, measured with §11.0's own commands so it composes with
§11.1's table:

| measure | `ddce7430` | after | delta |
|---|---:|---:|---:|
| SBND PR job TLAs | 501 | 495 | **−6** |
| `pattern_algos.m_X =` mirror lines | 387 | 381 | −6 |
| `doctest_clus_knob_defaults.cxx` CHECKs | 513 | 507 | −6 |
| `tcn_knobs` bag entries | 408 | 402 | −6 |

Round 1 retired 10; this round retires 3, and the campaign's +124 stands at
**+118 net**. The surface is still ~31 % larger than at the last cleanup — §12.7
names where the next reduction can come from.

### 12.3 Item 3 — `shower_samevtx_track_absorb`: measured, and **KEPT**

§11.3.1 found this knob ON in SBND production and firing on **0 of 3067**
events, named it *"ON but inert / superseded in place"*, and §11.6 item 3
planned to delete it once the successor was named. **The measurement overturns
that.** The knob is not superseded; it is *pre-empted*, and it still works.

**Method.** Leave-one-out from the production config on NCπ⁰ event **37112**
alone (the knob's own motivating event), one arm per candidate, reading the
**outcome** from the calib dump — shower count and shower 67048's
`kine_best`/`kine_dQdx` — rather than tag presence, since a knob can fire on the
event without performing the merge. Candidates came from two directions that
disagreed: the three tags that fire only in the production arm on 37112
(`pr117 merge_relax`, `pr123 pass4_prune`, `pr125 satellite_absorb`), and four
untagged knobs whose own documentation describes a track-into-shower fold.
**The control arm reproduces `work-ncpi0-prod0901` byte-identically** (2/2
archives), so every row below rests on a verified baseline.

| arm (leave-one-out from production) | showers | 67048 `kine_best` | `kine_dQdx` | `samevtx` fires |
|---|---:|---:|---:|---:|
| control = production | 8 | 741.7 | 840.9 | 0 |
| `shower_merge_relax` off | 9 | 595.1 | 442.6 | **1** |
| `shower_pass4_prune_detached` off | 6 | 741.7 | 840.9 | 0 |
| `shower_satellite_absorb` off | 12 | 730.1 | 832.6 | 0 |
| `stem_backfill_back_dvtx` off | 8 | 741.7 | 840.9 | 0 |
| `shower_pass3_backfill_guard_len` off | 8 | 741.7 | 840.9 | 0 |
| `shower_pass4_prox_guard_len` off | 8 | 741.7 | 840.9 | 0 |
| **`shower_pass4_prefilter_v1_escape` off** | **8** | **741.7** | **840.9** | **1** |

The last row is the answer. With the pr/136 pass-4 prefilter escape turned off,
`samevtx` fires and absorbs **2 track-typed fragments** — exactly the count
pr/125 §4.2 recorded at ship time — and the arm is **byte-identical to
production** (`pr85_hash_gate.py` PASS, 2/2 archives). Two independent knobs
converge on the same object; in production the pass-4 escape simply gets there
first (`NeutrinoShowerClustering.cxx` pass 4, well before the samevtx pass), so
by the time samevtx runs its precondition — a *separate* track-typed fragment
shower at a shared non-main vertex — no longer holds.

**The decisive control**, and the reason the knob is not deleted:

| arm | showers | 67048 `kine_best` | `kine_dQdx` | `samevtx` |
|---|---:|---:|---:|---:|
| production | 8 | 741.7 | 840.9 | 0 |
| prefilter escape OFF | 8 | 741.7 | 840.9 | 1 |
| prefilter escape OFF **+ samevtx OFF** | **12** | **522.1** | **390.9** | 0 |

Turn both off and **the merge is lost**. So `shower_samevtx_track_absorb` is a
live, load-bearing fallback sitting exactly one flip away from being the only
thing holding 37112's merge together. Deleting it would have passed the gate
today — it is inert, so the bytes would not move — and would have silently
removed that cover. **Verdict: KEEP.**

**What this corrects, generally.** §11.3.1 proposed a fifth taxonomy kind, *"ON
but inert / superseded in place"*, and treated it as removable. The measurement
splits that kind in two, and only one half is removable:

- **superseded** — another knob does the job and this one no longer can. Removable.
- **pre-empted** — another knob merely gets there *first*; this one still works
  and still produces the identical result when reached. **Not** removable, and
  indistinguishable from "superseded" by fire census alone.

Zero fires therefore does not mean dead code. It can be a statement about **pass
ordering among redundant paths**, and only a leave-one-out arm can tell the two
apart. That is the general lesson from this item, and it is why §11.6 item 3's
"name the successor first" guard was worth the arms it cost.

*Not attributed, unchanged from §11.3.1:* `kine_best` 797.3 (pr/125 validation
arm) → 741.7 (production). Composition is identical (`kine_dQdx` 840.9 both), so
this is charge accounting — the 0.86 scale and doc 85 r2's excluded-energy
census both landed in between — not a merge change.

### 12.4 Item 4 — the instrument's blind spot, now a measured number

§11.3 could only remark that the census covers *"35 tags against 315 ON job
TLAs"*. Nothing regenerated that ratio and nothing said **which** knobs were
invisible. `scripts/cfg/tag_coverage.py` (new) names them, output
`docs/77-tagcoverage-prod0901.tsv`:

| measure | value |
|---|---:|
| SBND PR job TLAs | 495 |
| … carrying a value ("ON" by §11.1's mechanical split) | 315 |
| instrumented tag literals in `clus/src` | 34 (33 distinct names) |
| ON knobs with a matching tag | 46 |
| **ON knobs with no tag — invisible to the fire census** | **269** |
| coverage (lower bound) | **14.6 %** |

The three retirements took 6 TLAs off the OFF side (186 → 180) and none off the
ON side, so this round's own edits do not move the coverage figure.

The matcher is a name-token heuristic, so the coverage figure is a **lower
bound** and an entry on the untagged list is a lead to check, not a verdict —
seven tags are named for the fix rather than the knob and match nothing. The
script says so in its own output; that honesty is the point, since the whole
value of the number is that it bounds what §11's conclusions could see.

**The rule going forward:** every knob flipped ON ships a `prNN <tag>:` line.
The pair is the standard round check — sentinel = *still right on its event*,
fire census = *still fires at all* — and §12.3 adds the third question the pair
still cannot answer alone: *if it does not fire, is it superseded or merely
pre-empted?* Only a leave-one-out arm settles that.

### 12.5 Item 5 — the OFF-arm census, and what a second arm buys

`fire_census.py` re-run over the four campaign-OFF arms (`empre0901`, the same
3067 events): **35 tags, 13 fire, 22 ZERO**, committed as
`docs/77-firecensus-empre0901.tsv`. `scripts/cfg/census_ab.py` (new) classifies
every tag against the production census — `docs/77-censusab-prod-vs-empre.tsv`:

| class | n | meaning |
|---|---:|---|
| **CAMPAIGN** | 12 | fires in production, dark with the campaign off |
| **PERTURBED** | 6 | pre-campaign knob whose firing the campaign *moved* |
| **IDENTICAL** | 7 | pre-campaign, untouched |
| **ZERO** | 10 | fires in neither — says nothing on its own |

Two results §11 could not reach with one arm:

- **The 12 CAMPAIGN tags go to exactly zero** with the campaign off
  (`satellite_absorb` 372→0, `pass4_prune` 96→0, `pass4_prune2` 67→0,
  `shower_split` 60→0, `pass4_prox_guard` 41→0, `merge_relax` 18→0,
  `pass3_cone_guard` 10→0, `pass4_track_guard` and `pass3_backfill_guard` and
  `stem_backfill_back_guard` 7→0, `stem_backfill_back_dvtx` 3→0,
  `ex1_dedup_rehome` 1→0). That is an **independent empirical confirmation of
  doc pr/142's restore file** — measured at the instrument rather than argued
  from the config, and by a route pr/142's own Proofs A–C do not use.
- **Six pre-campaign knobs were perturbed**, all in single digits:
  `detach_track_stem` 625→621, `stem_backfill` 38→40, `conn3_unreachable` 25→28,
  `pf_id_collision` 10→12, `cone_absorb_guard` 6→5, `accept_pid_guard` 2→3.
  Seven others are bit-for-bit unchanged (`do_rough_path` 1383 both sides,
  `shower_in_cascade_guard` 86, `absorb_unreachable_main` 75, `conn3_stitch` 17,
  `case_b_dqdx_guard` 15, `michel_stem_michel_check` 9, `ghost_member` 1). Small
  and local, exactly the blast radius pr/142 §8 predicted in advance.

**A ZERO row still proves nothing by itself**, and this round is careful not to
use it as if it did: `pr117 flank_absorb` reads ZERO in both arms, but it is OFF
in both, so that zero is not evidence for its retirement. §12.2's justification
is the compiled-config argument and the byte-identity gate, never the census.

### 12.6 Errata found in this doc while executing it

Four, all found by re-deriving rather than reading, and all fixed here rather
than left for a future round to trip on. They are listed because §3.2 Trap 1 is
about exactly this failure mode — prose goes stale, and a cleanup round that
trusts prose deletes the wrong thing.

1. **§11.4's kind-3 list ≠ §4's** — `teb_chain_topology` was silently
   substituted for `mvga_ac_veto_radius`. Both are adjudicated in §12.1; the
   union is six.
2. **`other_seg_keep_isolated_min_nnf` is a stranded kind 2.** §4:284 files it
   measured-negative/verdict-closed, round 1 never retired it, and it is in
   none of the ledger's 20 rows. Its cfg comment (`:2849`) is more cautious
   than §4 — *"STAYS OFF — validation FAILED at 4 … Owner hand-scan before any
   flip"* — which reads as kind 3, not kind 2. **The cfg comment governs**
   (Trap 1), so it is neither retired here nor treated as closed. → §12.7.
3. **§9:533's "the 7 already deleted before this round"** contradicts its own
   generated ledger, which carries **10** such rows (20 data rows total, of
   which 10 have `remove_commit = 0e8b7334`). Per §6 the ledger is generated
   and the prose is not: **10** is authoritative, "7" is stale.
4. **The `WCT_*` env-probe count drifts** — 37 in §4 and §9, 39 → 51 in §11.1.
   Any future census must not quote 37.

### 12.7 The documented removal pool (not touched this round)

Per the owner's scope decision, knobs whose closed-negative verdict was
established *today* are documented rather than deleted — a deletion should rest
on a verdict that predates the round doing the deleting. This is the
ledger-ready pool for a later round:

| knob | why it qualifies | caution |
|---|---|---|
| `teb_chain_topology` (+ riders `teb_r3_turn`, `teb_r3_hot`) | 19 ADVERSE vs 6 toward, 2 cosmict flips — pr/90 §10.6, cfg:2344 *"STAY OFF"* | pr/90:1265 keeps it *"for a future vertex-anchored redesign"*. That is an intent to re-use, and it is the owner's call whether it survives. `NeutrinoPatternBase.h:478` also cites it as `teb_second_max`'s superseder, so the ledger's existing row needs a note if it goes. |
| `mvga_ac_veto_radius` | 0.2 cm measured **ADVERSE**, kills the 349945 design case — cfg:2592 | §4 filed it kind 3 on pr/99:339's *"retained for future scans"*. |
| `other_seg_keep_isolated_min_nnf` | §4 filed it kind 2 (§12.6 item 2) | but its cfg comment asks for an **owner hand-scan before any flip** and records a named nue loss — treat as kind 3 until the owner says otherwise. |

**Not** in the pool, and not assumed: the added-`false` knobs `pi0_nv_allow_type2`,
`pi0_nv_retry_paired`, `pi0_reseat_start_assoc`, `pi0_mu_shower_hypothesis` and
the three `shower_split_*`. Their cfg comments state a *default*, not a verdict
(§11.2), and none has been read against its originating doc.

The largest remaining mechanical win is unchanged and unaffected by any of
this: **§5b2, the mirror block** (381 lines after this round), still §8
decision #3 and still the owner's call under M10.

### 12.8 Verification

- **Compiled-config, Gate 1** — SBND PR job at the production operating point,
  `ddce7430` cfg tree vs this round's: **byte-identical** (266,930 bytes each).
- **Compiled-config, Gate 1b** — every live consumer via
  `scripts/cfg/compile_consumers.sh` + `cmp_consumers.sh`: **21/21 identical, 0
  differ**, including the uBooNE MABC job and the PDHD/PDVD + sim-check set.
  This *measures* §11.5's claim that a kind-2 removal is compiled-config-
  identical for uBooNE as well as SBND, rather than inheriting it.
- **Unit tests** — `./build/clus/wcdoctest-clus`: **235/235 cases, 2627/2627
  assertions, 0 failed** (after the 6 `CHECK` lines were dropped deliberately;
  doc 70's rule — a removed default must appear in the diff).
- **Freshness proof (M1)** — `local/lib/libWireCellClus.so` 2026-09-01 10:07:34
  vs last source edit 10:02:15; base and post-edit snapshots differ by md5
  (`430ffa3e…` → `ba7324ac…`), so the gate is not comparing a binary to itself.
- **Binary pin** — each arm ran under `LD_LIBRARY_PATH=<snapshot>`; snapshots at
  `$SX/knob77r3/lib-{base,rm3}`. Note this tree resolves `toolkit/build/<pkg>`
  *before* `local/lib`, so the snapshot is prepended rather than relied on
  through the prefix.
- **Byte-identity gate**, §9's standard manifest shape — **308 events** = 241
  mcp1k (first 241 sorted from `work-mcp1k-grp0825`) + 48 nueCC48 + 19 NCπ⁰,
  `PR_JOBS=16`, manifest at `$SX/knob77r3/gate308-*.txt`. Arms
  `work-77r3-base-{mcp1k,nuecc48,ncpi0}` vs `work-77r3-rm3-*`, both `rc=0` on
  308/308 events:
  | sample | events | `pr85_hash_gate.py` | `pr94_root_gate.py` | sorted `nusel-table.tsv` |
  |---|---:|---|---|---|
  | mcp1k | 241 | **PASS** 482 archives | **PASS** 241 identical | 0 lines |
  | nueCC48 | 48 | **PASS** 96 archives | **PASS** 48 identical | 0 lines |
  | NCpi0 | 19 | **PASS** 38 archives | **PASS** 19 identical | 0 lines |
  | **total** | **308** | **616 byte-identical, 0 unpaired** | **308 identical, 0 differing** | **0** |
- **Item 3's arms** — `work-77r3-bis-*-ncpi0`, nine leave-one-out single-event
  arms plus the two-knob control; the control is byte-identical to
  `work-ncpi0-prod0901` on 37112 (2/2 archives).
- **Not run, and why** — no new full-3067 production arm: this round changes no
  compiled config and no reachable code, and the 308-event gate plus the 21-way
  compiled-config identity bound it more tightly than a re-run would.

**Ledger** — three rows appended to `docs/77_knob-ledger.tsv`, **generated** by
`scripts/cfg/ledger_line.py` (new) from the jsonnet plus `git log -S`, never
hand-typed (§6, Trap 1). `verdict_class = INERT`, following round 1's own
precedent of reserving `ADVERSE` for knobs that moved something the wrong way
and using `INERT` for measured-zero-yield. The ledger now carries **23** rows.

§7 **Phase 2 status: DONE** for the kind-2 population identified as of
2026-09-01. §12.7's pool is the next tranche and is owner-gated. §7 Phase 4
remains **CLOSED** (Reading A, §11.5); §5b2 and §8 decision #3 remain open.
