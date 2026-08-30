# doc pr/131 — the pr/128 denominator, measured

**Status: MEASURED. Census probe only — stderr, env-gated, no knob, no
production gate relaxed, nothing flipped.** Closes the item doc pr/128 left at
the top of its "Still open" list:

> Every completeness claim so far is anecdotal because the `same_cluster` gate
> hides the audit line as well as the pools, so there is **no denominator**: we
> cannot say what fraction of near-candidate reconstructed charge is missing
> from `kine_reco_Enu`, only that these 13 objects existed. Measuring that
> denominator (debug-gated, no knob, no predicate filter) is the prerequisite
> for deciding whether more completeness rounds are worth running.

**The answer is 0.77%, and it says stop running population-scale completeness
rounds.** The detail says something more interesting than the headline.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./scripts/pr131_arms.sh 98  denom98  1
./scripts/pr131_arms.sh 141 denom141 1
./scripts/pr131_denominator.py "work-pr131-denom98-*" "work-pr131-denom141-*" \
    > docs/pr/pr131-denominator.txt
```

Arms `work-pr131-denom{98,141}-*`, all 239 events of both standard manifests,
at toolkit **`95346dc5`** — today's final SBND production point, after all
three of today's flips (`shower_pass4_prox_guard_len=50`,
`shower_pass3_backfill_guard_len=15`, `stem_backfill_back_dvtx=45`). That hash
matters: §5 below turns on one of them.

## The probe

`MultiAlgBlobClustering.cxx`, end of `fill_bee_pf_tree`, under
`WCT_PFDENOM_DEBUG`. It enumerates the whole PR graph and classifies every
segment. It **appends no node, touches no `particles`/`next_id`, and relaxes no
gate** — the `same_cluster` filters are left exactly as they are. (doc pr/128
§4 cleared M15 for the *knobs pr/128 shipped*, not for a blanket un-gating.
This round only counts.)

Three design points that the numbers depend on:

1. **It does not live inside the audit block.** The pr/65 audit line is gated on
   `pf_shower_vertex_barrier && pf_orphan_audit_only` *and* on `same_cluster`,
   and its rollup counts only the post-gate survivors. Sitting outside all of
   that is what makes a denominator possible at all.
2. **Three pools draw without inserting into `used_segs`** — pr/93
   confident-track, pr/123 guard-freed, pr/128 near-cross-cluster. Treating
   `used_segs` as "what the tree draws" would book them as unclaimed and
   overstate the hidden population by exactly the loudest events in the
   manifests (171572, 393505, 315167, and — see §5 — 399118). They are tracked
   in a lookup-only set filled only under the env gate.
3. **conn-4 and cross-cluster overlap.** `conn4_skip_segs` is tested *before*
   `same_cluster` in every pool, so a single `class=` string would not add up.
   The summary line uses five mutually exclusive buckets; the per-segment lines
   carry the raw flags independently, so cuts can move without a re-run.

### Currency warning

Segment KE (`particle_info()->kinetic_energy()`) is **not** the currency of
`kine_reco_Enu`, which carries shower charge-based energy, the pr/101 mass
rules and long-muon range KE. Every ratio in this doc is like-for-like segment
KE throughout. `kine_reco_Enu` appears only beside the per-event numbers as a
scale reference, never as the denominator of a bucket ratio.

## 1. The buckets — 239 events

| bucket | meaning | nseg | KE (MeV) | % |
|---|---|---|---|---|
| `drawn` | in `used_segs`, not conn-4 — the PF tree shows it | 9323 | 191810.1 | 95.52% |
| `conn4` | in `conn4_skip_segs` (pr/128 class B) | 570 | 3349.7 | 1.67% |
| `extra` | a late pool drew it without touching `used_segs` | 17 | 4118.4 | 2.05% |
| `audited` | unclaimed, same cluster — the pr/65 line names it | 4 | 34.1 | 0.02% |
| **`hidden`** | **unclaimed, cross-cluster — named nowhere** | **33** | **1495.2** | **0.74%** |
| | **TOTAL** | 9947 | 200807.5 | |

**Which source each figure comes from.** The bucket table and the totals
(200807.5, and the 197457.8 universe in §2) are summed from the per-event
`PFDENOM_SUM` lines; the conn-4 split in §3 is summed from the per-segment
`PFDENOM` rows, which is the only place the `xclus` flag lives. The two agree
to 0.04% on conn-4 (3349.7 vs 3350.9 MeV) because the summary line prints one
decimal per event before being added, so `200807.5 − 3350.9` does not land
exactly on 197457.8. The gap is 1.2 MeV — 0.0006% — and moves no conclusion.

## 2. The number pr/128 asked for

The near-candidate universe, on the owner's own line: everything the candidate
has a claim on = `drawn + extra + audited + hidden + conn4_main`. `conn4_far`
is excluded — see §3, and the exclusion is the owner's rule, not convenience.

> **Of 197457.8 MeV of near-candidate reconstructed segment KE over 239 events,
> 1529.3 MeV — 0.77% — reaches no PF output. Of that, 1495.2 MeV (0.76% of the
> whole) is the cross-cluster `hidden` class that is counted nowhere in the
> toolkit at all.**

Within the unclaimed population alone (`audited + hidden`), the hidden share is
**97.8%** (1495.2 of 1529.3 MeV, 33 of 37 segments). So the `same_cluster` gate
is not hiding *some* of the problem — it is hiding essentially all of it. That
part of doc pr/128's suspicion was exactly right.

**The decision this was the prerequisite for: population-scale completeness
rounds are not worth running.** Under one percent of near-candidate charge is
missing, and the campaign has spent four rounds on this front.

## 3. `conn4_main = 0` — pr/128's class-B knob already finished the job

The owner drew the line when approving 105074 (pr/128): **main-cluster
membership is a sufficient admission rule; vertex reachability is not
required.** So conn-4 material inside the main cluster belongs in the
denominator, and conn-4 in a genuinely distant cluster does not (conn-4 *means*
"cluster >80 cm from the candidate", `NeutrinoShowerClustering.cxx:3733`).

Splitting the 570 conn-4 segments on that line:

| | nseg | KE (MeV) |
|---|---|---|
| `conn4_main` (inside the main cluster) | **0** | **0.0** |
| `conn4_far` (distant cluster) | 570 | 3350.9 |

**Zero.** Not a parsing failure — verified on 105074 itself, whose two class-B
showers are now in the `drawn` bucket:

```
pr128 pf-conn4-near: KEEP shower_id=4 cluster=23 pdg=13 ke_mev=162.03 len_cm=58.2 gap_cm=0.08
pr128 pf-conn4-near: KEEP shower_id=5 cluster=23 pdg=13 ke_mev=215.08 len_cm=82.9 gap_cm=0.07
```

`pf_conn4_near_candidate` (shipped and flipped by pr/128) recovered them. Every
conn-4 segment that remains skipped is in a distant cluster, which is what the
gate is for. **Class B is closed** — there is nothing left in it under the
owner's own rule.

## 4. The hidden charge is a concentrated tail, and it is touching the candidate

The population fraction is <1%, but it is not spread thin:

- **19 of 239 events** carry any hidden KE.
- Per event it is large: **318769 hides 298.7 MeV against a reco Enu of 826.6
  (36.1%)**; 169626 17.9%; 54341 18.3%; 51546 15.0%; 259542 202.1 MeV (10.8%).

| cut on `dmin` (gap to what the tree draws) | hidden segs | hidden KE | events |
|---|---|---|---|
| ≤ 0.5 cm | 18 | **1238.1 MeV** | 14 |
| ≤ 5 cm | 22 | 1332.6 MeV | 16 |
| ≤ 10 cm | 27 | 1382.1 MeV | 16 |
| ≤ 50 cm | 32 | 1473.6 MeV | 18 |

**83% of the hidden charge sits at a gap of 0.5 cm or less — literally touching
content the PF tree already draws.** That is the same shape doc pr/128 saw
anecdotally ("many sit at gap 0.00 cm"), now with a denominator under it.

What it is: 33 segments, **max length 39.8 cm, median 9.5 cm, none over 50 cm**;
by charge 724.2 MeV muon, 445.0 MeV electron, 317.5 MeV proton. Three would
also be dropped by the display filters (`dirsign==0`) even if the cluster gate
were opened, so opening the gate alone would not recover them.

> So the honest statement is two-sided: **as a fraction of reconstructed energy
> this is a rounding error; as a per-event effect on ~8% of events it is tens of
> percent of Enu.** A round aimed at the population average is not worth it. A
> round aimed at the 19-event tail is a different proposition, and this census
> is its target list.

## 5. pr/128's headline open item closed itself — and pr/129 overruled it

doc pr/128 named one item as "the strongest single remaining loss, and the
natural next front": **399118's 108.8 cm / 481.0 MeV proton**, 4.9 cm from the
ν vertex, rejected by the shipped continuation predicate because it kinks
47.3°. Its recommendation was a **vertex-proximity arm** (small `d_mainvtx`, no
kink requirement), and it measured the arm clean — one candidate at any cut
from 5 to 20 cm, nearest cosmic at 35.3 cm.

That proton is **no longer missing from the PF tree**:

```
PFDENOM seg=16017 cluster=16 pdg=2212 ke_mev=481.02 len_cm=108.75 dirsign=1
        conn4=0 xclus=1 gf=1 shown=1 bucket=extra dmin_cm=0.00 d_mainvtx_cm=4.94
pr123 pf-orphan-guard-freed: EMIT pseudo-n id=8 -> seg=16017 cluster=16 pdg=2212 ke_mev=481.02
```

Today's `shower_pass4_prox_guard_len=50` flip declined it from the EM shower,
which stamped `kPass4GuardFreed` on it, and **pr/123's guard-freed pool drew
it**. A knob shipped for a different reason closed pr/128's top item as a side
effect — which is why the census had to be run at `95346dc5` and not earlier.

**But it is still not in the energy**, and that is a decision, not a gap.
`kine_count_guard_freed` is ON, and pr/129's pointing test refuses it:

```
kine_guard_freed_impact: seg idx=17 cluster=16 ke_mev=481.02 d_vtx_cm=4.94
                         impact_cm=5.23 miss_deg=151.6 -> SKIP
```

It sits 4.94 cm from the vertex and points **151.6° away from it**. pr/128's
proposed vertex-proximity arm — *small `d_mainvtx`, no kink requirement* — would
have admitted exactly this object, and pr/129's DIRECTION discriminator was
built to exclude exactly this object. **The two proposals are in direct
conflict and the later one won.** Anyone reopening pr/128's "natural next
front" has to argue against pr/129's pointing test first.

`kine_reco_Enu` for 399118 is 803.2 MeV with no 481 MeV proton in
`kine_energy_particle` — confirming the SKIP reached the output.

### PF and kine are different gates — do not read this doc as an Enu number

This census measures the **PF tree**. The kine side has its own gate
(`kine_count_orphan_tracks`, `NeutrinoKinematics.cxx:552`) and its own
pr/129 pointing test on top. 399118 is the proof that they disagree: the
proton is `bucket=extra` (drawn) here and absent from Enu there. **The 0.77%
is "reaches no PF output", not "missing from Enu".** A kine-side denominator
is a separate measurement and is not made here.

## Validation

- **Byte-identical with `WCT_PFDENOM_DEBUG` unset**: PASS on 72786 + 55740 —
  `mabc-pr.zip` and `pctree-pr-evt*.tar.gz` by `hash_archive.py` member content,
  plus the calib dump compared with timer fields excluded — against
  `work-pr130r1-bon141-mcp2k` at the same production point.
- **Freshness proof (M1)**: `local/lib/libWireCellClus.so` 18:38:44 >
  `MultiAlgBlobClustering.cxx` 18:38:05.
- `./build/clus/wcdoctest-clus`: **235 passed, 0 failed**.
- **The probe emits** (a census that prints nothing is indistinguishable from
  one that is not wired): 9947 classified segments over 239 events, and the
  arm's compiled config confirms the production point —
  `pf_orphan_audit_only`, `pf_shower_vertex_barrier`, all four pr/128 pools ON,
  and all three of today's flips at 50 / 15 / 45.
- `fill_bee_pf_tree` runs once per evaluated candidate; **3 of 239 events**
  (18625, 179369, 286681) evaluate two. Both bundles accumulate into the same
  PF output, so their sums add, and per-segment rows are de-duplicated on
  segment id.

## What is NOT established

- **No kine-side denominator.** See §5. Everything here is the PF tree.
- **Segment KE is not Enu currency.** The `hid/Enu` column in
  `pr131-denominator.txt` is a scale reference; it is not a conserved fraction
  and must not be summed.
- **The 19-event tail is not adjudicated.** No scanner has looked at these 33
  segments. "Hidden" means the toolkit counts them nowhere — not that they
  belong to the neutrino. 72786's two (70.1 MeV) are in the event pr/128 uses
  as its cosmic CONTROL, so at least some of the tail is correctly excluded.
- **Whether the 0.5 cm touching population is recoverable** is not measured.
  Three of the 33 fail the display filters independently of the cluster gate.

## Where this leaves the completeness front

1. The denominator is **0.77%**; the hidden share of the unclaimed is **97.8%**.
2. **Population-scale completeness rounds: stop.** The prerequisite question
   doc pr/128 posed is answered against running them.
3. **Class B is closed** — `conn4_main` is zero under the owner's own rule.
4. pr/128's headline item is **closed for display** by today's flip and
   **deliberately excluded from energy** by pr/129. Its proposed fix is
   superseded, not pending.
5. What remains is a **19-event tail**, 83% of it touching drawn content, with
   a target list on disk (`pr131-denominator.txt`). Whether that tail is worth
   a round is an owner call, and it is a different question from the one
   pr/128 asked.

Related: [`128_pf-kine-completeness.md`](128_pf-kine-completeness.md),
[`129_pointing-guard-freed.md`](129_pointing-guard-freed.md),
[`130_guard-freed-overcount.md`](130_guard-freed-overcount.md),
[`pr130-qextra-98set.md`](pr130-qextra-98set.md).
