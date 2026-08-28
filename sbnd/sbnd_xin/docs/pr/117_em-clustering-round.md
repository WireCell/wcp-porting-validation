# pr/117 — EM clustering round 1: three knobs out of the pr/115 census

**Status**: complete.  `shower_pass4_best_owner` + `shower_merge_relax` SBND
PRODUCTION ON 2026-08-28 (owner pre-authorized "if validation passed";
flip-equivalence PASS); `shower_flank_absorb` shipped OFF, not selected.
**Toolkit commits**: `ff239030` (knobs, DEFAULT OFF) + `c559d84c` (flip).
**Toolkit**: three default-OFF knob families in `clus/` (`shower_pass4_best_owner`,
`shower_merge_relax`, `shower_flank_absorb`), OFF path proven byte-identical.
**Scope**: EM shower clustering only — under-clustering, over-clustering,
misidentified stem parts.  π⁰ pairing/reporting deferred (owner instruction);
tracks unaffected (verified per knob below).

## Repro

```
# toolkit (apply-pointcloud), after the pr/117 commits:
direnv exec ~/toolkit-dev bash -c './wcb build --notests -p && ./wcb install --notests -p'
./build/clus/wcdoctest-clus                      # 235/235 pass

cd sbnd_xin
# bare arm (pre-change binary), off arm (new binary, knobs off), gate:
#   events = the 98 scan events of em_display/em114-manifest.tsv, run per sample
#   from the work-<sample>-grp0825 Q/L roots
WCT_SHOWER_CONTENT_DEBUG=1 WCT_SHOWER_ABSORB_DEBUG=1 WCT_SHOWER_MERGE_DEBUG=1 \
PR_EXTRA_STAGES=pr_display PR_JOBS=24 \
  ./run_pr_chain_batch.sh work-<sample>-grp0825 work-pr117r1-bare-<sample> data <events>
python3 scripts/pr85_hash_gate.py work-pr117r1-bare-<s> work-pr117r1-off3-<s>

# knob arms via the runner envs (see sec 4), then:
cd em_display
./prep_pr117.py --tag 117onK1 work-pr117r1-onK1-{mcp1k,mcp2k,ncpi0,nuecc48}
./em117_score.py --manifest em117-117onK1-manifest.tsv --prepdir emprep-117onK1 \
                 --tsv ../docs/pr/pr117-onK1-score.tsv
./em117_score.py --baseline <bare tsv> --compare <on tsv>
./em117_score.py --diffstat emprep-117bare emprep-117onK1
```

## 1. What this round ships

The pr/115 §16.5 absorb census turned the hand scan's 28 failing events into a
ranked list of call sites.  This round implements one default-OFF knob family
per measured defect class:

| knob family | defect (share) | mechanism |
|---|---|---|
| `shower_pass4_best_owner` | wrong-owner absorption — 48 % of wrongly-held charge; 25/30 of that pass's misses sit in a *neighbour* shower | the pass-4 direct cone (`shower_clustering_with_nv_from_vertices`) accepts greedily with no competition between showers; when on, each accepted segment goes to the argmin over all existing showers of the pass-3 ellipsoid metric `(d·cosθ/40cm)² + (d·sinθ/5cm)²` (rivals gated by the pass-3 cone disjunction) |
| `shower_flank_absorb` (+`_max_dis` 6 cm, `_max_len` 25 cm) | orphan stubs — 41 marks (33 % of all misses) never absorbed by any pass | a new late seat (after `stem_backfill`, before calc-kine-2): unclaimed shower-like segments absorbed into the argmin shower by body distance; main-cluster segments deliberately eligible (the legacy seats' `cluster==main_cluster ⇒ continue` guard is what orphans them) |
| `shower_merge_relax` (+`_dis` 6 cm, `_angle` 15°) | unmerged fragments — the 20-event merge class | a new late consolidation pass (`merge_shower_fragments`, after `examine_showers`): a strictly-smaller shower merges into a bigger one on body gap + local-pivot axis agreement; hard γγ guard — never merges two main-vertex conn-1/2 showers |

Supporting evidence for the arbitration design: pass 3's cone — the one
absorber that already argmin-competes — causes only 5 % of the damage; the
greedy pass-4 cone causes 48 %.

**A census correction to pr/115 §16.5** (recorded here, the pr/115 doc is
append-only): its "where" column attributes the `pass4_angle` tag to the pass-3
cone at `NeutrinoShowerClustering.cxx:1310` and quotes pass 3's 80/130/200 cm
constants.  The tag is in fact emitted by the **pass-4 direct cone** in
`shower_clustering_with_nv_from_vertices` (accept site, probe tag
`pass4_angle`); pass 3's cone tags as `pass3_cone` (8 marks, 5 %).  The census
*numbers* are unaffected — only the file/line attribution moves — and the
conclusion sharpens: the dominant offender is the one cone with no competition.

## 2. Implementation

Per knob, the six seats (pattern: `shower_ghost_member_drop`):
`NeutrinoPatternBase.h` member + rationale block; `TaggerCheckNeutrino.h`
mirror (cm/deg); `configure()` read; `default_configuration()` echo;
`pattern_algos` push (cm→internal at the push); pin in
`doctest_clus_knob_defaults.cxx`.

- **K1** (`shower_pass4_best_owner`): rival list hoisted per candidate cluster
  (all other showers, cluster-id/segment-id order, start point + 30 cm local
  direction; near-zero directions skipped so 1-segment stubs cannot steal).
  At the accept site the current shower's metric is the better of its two
  anchors (start vertex / closest-approach point, matching the acceptance
  disjunction); a rival competes only if it passes the pass-3 cone gate from
  its own start point.  Divert = `add_segment` into the argmin (probe tag
  `pass4_angle_divert`); the accepted segment *set* is unchanged — only the
  owner — so the track pool is untouched by construction.  Structural
  limitation (by design, measured by the probe): rivals must already exist at
  take-time; a wrong owner vs a *not-yet-seeded* cluster's future shower is
  not fixed.  The pass-4 flood-fill (Path A, the pr/84 last-writer-wins hole)
  is untouched.
- **K3** (`shower_flank_absorb`): single deterministic sweep, segments in
  graph-index order × showers in cluster/segment-id order, strict-`<` argmin
  on `shower_get_closest_dis`.  Candidate filter is the track safety:
  kShowerTrajectory/kShowerTopology or |pdg|==11; not in a long muon; not
  `segment_confident_nonelectron_pid`; length < max_len and < 0.75× the
  recipient.  Long-muon pseudo-showers (|type|==13) excluded as recipients.
  Grown showers get `set_flag_kinematics(false)` so calc-kine-2 recosts them.
- **K2** (`shower_merge_relax`): two-phase plan/execute forked in shape from
  `examine_merge_showers` (which stays byte-untouched).  Absorber = strictly
  longer shower; gap = min over fragment members of distance to the absorber's
  fit cloud; directions at the *meeting point* (30 cm), axis-folded (|cos|) —
  fragments continuing each other read anti-parallel.  Post-merge bookkeeping
  verbatim from the legacy pass.
- Byte-neutral probe: `WCT_SHOWER_ABSORB_DEBUG` now also prints, per pass-4
  cone accept, `SHOWER_ABSORB PASS4_OWNER seg=… cur=… cur_metric=… best=…
  best_metric=… divert=… applied=…` — the would-divert measurement runs with
  the knob off.

Config: keys join the `tcn_knobs` bag in
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` with the key-suppression
idiom; runner envs in `run_pr_chain_batch.sh` (`SBND_SHOWER_PASS4_BEST_OWNER`,
`SBND_SHOWER_MERGE_RELAX{,_DIS,_ANGLE}`,
`SBND_SHOWER_FLANK_ABSORB{,_MAX_DIS,_MAX_LEN}`; bools tri-state, numerics
pass-through in cm/deg).

## 3. Gates (all PASS)

- `./build/clus/wcdoctest-clus`: **235/235 pass** (7 new default pins).
- Compiled-config, knob off: base-vs-new `wcsonnet` output **byte-identical**
  (full tagger pipeline, `dl_weights=` empty).  Knob on: all 7 keys present in
  the compiled `TaggerCheckNeutrino` node with the passed values.
- Runtime OFF gate: arms `work-pr117r1-bare-<s>` (pre-change binary, HEAD
  `8d93260d`) vs `work-pr117r1-off3-<s>` (new binary, knobs off) over all 98
  scan events: `pr85_hash_gate.py` **PASS 28+34+38+96 = 196/196 archives
  byte-identical** (mabc-pr.zip + pctree per event), and every
  `nusel-evt*.tsv` byte-equal.  Freshness proof done before the arms
  (`local/lib/libWireCellClus.so` 12:15 > sources 12:11/12:12, Aug 28).

## 4. The measurement harness (cross-run)

`em_display/em117_score.py` — a fork of `em115_score.py` (fork-by-duplication)
adding `--manifest/--prepdir` overrides, charge-weighted-overlap shower
matching (the label's shower key is the scan-time start-segment display id; a
re-rooted shower renames, and the exact join would score a rename as a
catastrophe), a cross-run drift reinterpretation, and `--diffstat A B` — a
one-to-one greedy membership diff between two prepdirs, the hold-flat check
for the 50 mark-free events the score table cannot see.
`em_display/prep_pr117.py` builds a knob arm's sidecars + a manifest pointing
at the arm's **own** dumps (prep_em_scan.py's manifest hardcodes prod0825,
correct for the scan display, wrong for a changed reconstruction).

Validation of the chain itself:
- fork fidelity: `./em117_score.py` with no flags reproduces
  `./em115_score.py` stdout **byte-identically**;
- diffstat identity: a prepdir against itself = 0 moved segments
  (after fixing the matcher to one-to-one — overlapping showers otherwise
  orphan their twin and invent churn);
- the bare arm reproduces the scan-time reconstruction **exactly**: diffstat
  `emprep` vs `emprep-117bare` = 0 moved segments over all 98 events, and its
  cross-run score lands on the pr/115 §16.2 bucket medians (0.887 / 0.740 /
  0.740).

Baseline (bare arm, cross-run matching), the numbers every knob is judged
against — 33 marked showers over 25 events:

| bucket | n | med qF1 |
|---|---|---|
| 1 under-clustered | 17 | 0.887 |
| 1+2 both | 3 | 0.740 |
| 2 over-clustered | 5 | 0.740 |

## 5. K1 (`shower_pass4_best_owner`) — small, clean, and it measures its own ceiling

Arms `work-pr117r1-onK1-<s>` (K1 alone).  Sentinel `pr117 pass4_best_owner
on` present; across the 98 events the pass-4 cone accepted **586** segments
and arbitration diverted **16** (probe `divert=1`).

Delta vs the bare baseline (rows = changed marked showers only):

| event | shower | qF1 base → new |
|---|---|---|
| 469665 | 15003 | 0.891 → 0.899 (+0.008) |
| 314838 | 110088 | 0.740 → 0.741 (+0.001) |

No marked shower moves down.  Diffstat: **7 of 98 events change, 13 segments
move**; none of the 37 good events is touched (the changed unmarked events are
2× scanned-no-correction, 1× vertex-bad, 1× too-busy).  `nusel` scores:
unchanged on every event (bare vs onK1 spot-checked via the score TSVs).

**Why the yield is small, measured.**  Of the 25 `pass4_angle` IN marks
sitting in a neighbour shower (the 25-of-30 signature):

- 9 were taken before the scanned shower existed at all (pass 4 runs before
  `in_other_clusters` creates the conn-3/4 showers) — arbitration cannot see
  a rival that does not exist yet;
- of the 16 whose scanned shower did exist, **21 of the full 25 fail the
  pass-3 cone gate against the scanned shower's own axis** — e.g. evt409634's
  six segments sit 55–65° off-axis at 75–116 cm; evt415278's six at 20–29°
  beyond 130 cm.  No start-anchored cone metric can hand them back; they are
  flank/late-cascade geometry, which is exactly why the scanned shower missed
  them the first time.

So the wrong-owner class is *not* recoverable by owner arbitration on any
cone metric — the recoverable route is merging the wrongly-holding fragment
into the scanned shower wholesale (K2), or pr/91 P2's distance-to-charge
admission for the conn-3 case.  K1 stays: it is cheap, strictly
non-negative on the marked set, zero-churn on the controls, and its probe
now measures the divert rate in production.

## 6. The stub class dissolves into the merge class (measurement)

Two measurements that redraw §15.4b:

1. Body distance from each of the 41 "orphan" IN marks to the scanned
   shower's members (computed from the dump fit points): median **19.1 cm**,
   quartiles 9.7–29.7 cm; only 10 of 41 sit within 6 cm.
2. **Every one of the 41 is already a member of some shower in the sidecar**
   — 40 of 41 are pdg-11, median length 1.7 cm, and they typically form their
   own one-segment shower (created by the seeding passes, never *absorbed*,
   which is why the census tag reads "never absorbed").  evt444187's "stub"
   is a 52 cm electron that is literally its own second shower at gap 0.0 cm.

Consequence: **`shower_flank_absorb` (K3) has no targets in the marked set**
— there are no unclaimed shower-like segments to absorb on the failing
events.  Its knob-on run at the 6 cm default fires 8 times across all 98
events, all on unmarked segments, and moves no marked shower's score.  K3
ships as built (OFF, byte-identical; the class it guards against can exist —
nothing prevents a future graph from stranding an unclaimed stub) but it is
**not a production-flip candidate** on this evidence.  The stub class is
K2's: one-segment showers merging into their big neighbour on proximity.
This is also why K2 gained `shower_merge_relax_short_len` (5 cm): a 1.7 cm
fragment has no measurable 30 cm direction, so below that length the merge
is judged on gap alone — an angle test against a noise direction fires at
random.

## 7. K2 (`shower_merge_relax`) — the exploration that set its final shape

First trial (arms `work-pr117r1-onK12-<s>`, K1+K2, gap 6 cm / 15° / short_len
5 cm): **245 merges** over the 98 events — 242 by the gap-only short-fragment
path, 3 directional.  Net on the marked set **+0.085**, but decomposed:

- the single directional merge on a marked event is the entire headline:
  **evt168596 0.852 → 1.000 (+0.148)** — the reference merge-class event's
  neighbour shower (node 14058, holding 19 of its 20 marks) merges into the
  scanned shower at gap 0.00 cm, axis-fold 14.97°;
- the 242 short-path merges are **net negative on the marked set** (+0.044
  across 105946/169626/469665/314838 vs −0.107 across
  21073/47212/173093/269774/423981/463565, worst evt47212 −0.061, a
  1.000-pure shower gaining foreign charge).  Winners' and losers' gap
  distributions overlap completely (2.0–5.9 cm both) — blind proximity
  cannot tell an under-clustered stub from an over-clustered one, so no gap
  threshold separates them.

Second finding: of the 3 directional merges, the two on non-improving events
each had a **proton-typed side** — evt37112 (scanned-no-correction): pdg-2212
fragment into an e⁻; evt389538 (**good** event — control churn): e⁻ fragment
into a pdg-2212 absorber at 5.2 cm / 12.8°.

Final K2 semantics, from those two measurements:

1. **EM ↔ EM only** — both showers must be |type| 11 (hard guard, not a
   knob).  Kills both spurious directional merges, keeps evt168596, and
   keeps the pass out of the track pool entirely.
2. **`shower_merge_relax_short_len` defaults 0 = gap-only path disabled.**
   The stub class it aimed at is real but unservable by blind proximity;
   the knob remains for study.
3. Fragment-first argmin planning (each fragment picks its min-gap absorber;
   no absorber/fragment chains), after the first-trial code let the first
   bigger shower in iteration order claim any fragment.

## 8. Final configuration — result

Arms `work-pr117r1-onK12c-<s>`: K1 + K2 final semantics (EM↔EM, directional
only, 5 cm fragment floor), all 98 events.  K2 fires **3** merges; K1 diverts
**16** cone accepts.

Delta vs the bare baseline (`pr117-bare-score.tsv` → `pr117-onK12c-score.tsv`):

| event | shower | qF1 base → new |
|---|---|---|
| 168596 | 14153 | 0.852 → **1.000** (+0.148) |
| 469665 | 15003 | 0.891 → 0.899 (+0.008) |
| 314838 | 110088 | 0.740 → 0.741 (+0.001) |

**No marked shower moves down.**  Bucket medians: under-clustered 0.887 →
**0.899**, both 0.740, over 0.740→0.741.

Hold-flat (`--diffstat emprep-117bare emprep-117onK12c`): 10 of 98 events
change, 41 segment slots move.  Of the 37 good events exactly **one** changes:
evt389538, where two *touching* (gap 0.0 cm) EM showers 8.5° apart merge
(fragment 65071, 14.3 cm, into 54056) — a collinear-fragment cleanup on an
event bucketed "good (no major change)"; flagged here for owner review.  The
other changed events are the three winners plus scanned-no-correction
(37112, 281165), vertex-bad (56982, 76350), too-busy (396222), and one
1-segment move on 235435 with no score effect.

**`nusel-evt*.tsv` is byte-identical on all 98 events** — the knobs change
shower composition only; the neutrino selection scalars (main vertex, nue/numu
scores, cosmic flags) are untouched.  This is also the tracks-unaffected
verification: no track enters or leaves any reconstruction product.

## 9. Production flip (owner pre-authorized)

The owner authorized in advance: *"if validation passed, you can turn the
knobs on for SBND production."*  Validation above passed every §4 criterion,
so `wct-pr-perevt.jsonnet` now sets:

- `shower_pass4_best_owner = true` — SBND PRODUCTION ON 2026-08-28
- `shower_merge_relax = true` — SBND PRODUCTION ON 2026-08-28
- `shower_flank_absorb = false` — shipped, **not selected** (§6: no targets)
- numerics stay at C++ defaults (keys suppressed)

Compiled-config proof: the flipped config carries exactly the two true keys;
`shower_flank_absorb` / numerics absent.  Flip-equivalence: arms
`work-pr117r1-flipchk-<s>` (flipped config, **no** env) hash-gated against
the validated `onK12c` arms — see §10 for the gate verdict.

C++ defaults remain false everywhere — no other detector is affected.

## 10. Gate ledger

| gate | arms | verdict |
|---|---|---|
| OFF, final binary | `bare` vs `off6` (98 events × mabc + pctree) | PASS 196/196 byte-identical + all nusel `cmp` equal |
| OFF, intermediate binaries | `bare` vs `off3`, `off4`, `off5` | PASS 196/196 each |
| flip-equivalence | `onK12c` vs `flipchk` | PASS 196/196 byte-identical + all nusel `cmp` equal |
| doctest | `./build/clus/wcdoctest-clus` | 235/235, 2424 assertions |
| compiled-config OFF | base vs new jsonnet, full tagger pipeline | byte-identical |
| fork fidelity | `em117_score.py` (no flags) vs `em115_score.py` | stdout byte-identical |
| harness identity | `--diffstat` prepdir vs itself; bare vs scan-time | 0 moved segments |

Discarded arm: `work-pr117r1-onK13d15-*` ran while a rebuild was in flight
(binary-mixed) — no claims drawn from it.

## 10b. Owner-review Bee pair (uploaded 2026-08-28 on owner request)

Two-event before/after sets, index `bee/pr117r1/pr117r1.index.txt`
(idx 0 = 389538 control change, idx 1 = 168596 rescue):

- OFF (bare): https://www.phy.bnl.gov/twister/bee/set/c01a0af0-4f84-40a5-b3cb-942ee1c8beeb/event/list/
- ON (flipped config): https://www.phy.bnl.gov/twister/bee/set/2df29aff-4b56-4e4e-a2ed-540349b9c17c/event/list/

## 11. What stays open

- The 21-of-25 wrong-owner marks that fail every start-anchored cone (§5):
  the recoverable route is pr/91 P2's distance-to-charge admission for
  conn-3, now with this round's measurement in hand.
- The sub-5 cm stub class (§6): real, low-charge, unservable by blind
  proximity; needs a discriminator (e.g. charge continuity along the axis)
  before any knob can act on it.
- The pass-4 flood-fill last-writer-wins hole (pr/84) and the pi0 absorb
  ownership hole (`id_pi0_*`) — out of scope, unchanged.
- π⁰ pairing/reporting (pr/115 §16.3) — deferred by owner instruction.

## 12. Files

| file | role |
|---|---|
| toolkit `clus/{inc,src,test}` + `cfg/.../sbnd/wct-pr-perevt.jsonnet` | knobs, passes, pins, flip |
| `em_display/em117_score.py` | cross-run scorer (fork of em115_score.py) |
| `em_display/prep_pr117.py` | knob-arm sidecars + arm-own-dump manifest |
| `run_pr_chain_batch.sh` | pr/117 env→TLA block |
| `docs/pr/pr117-bare-score.tsv` | baseline rows (md5 52b46bec…) |
| `docs/pr/pr117-onK12c-score.tsv` | final-config rows (md5 1663c39b…) |
| `em_display/emprep-117{bare,onK12c}/` + their manifests | committed measurement products (fresh tags, M13) |
| `em_display/emprep-117{onK1,onK12}/` + manifests | intermediate-tuning products, left on disk uncommitted |

