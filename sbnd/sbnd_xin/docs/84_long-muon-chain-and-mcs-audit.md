# Doc 84 — long-muon chain construction and the MCS delta-ray question: audit + case list

Owner ask (2026-08-28): *"audit the long muon formation and construction in the
toolkit. 1. Against the prototype code 2. based on the existing events 1000 and
2000 numu events, there are some of them contained long muons, which may not be
recognized, but treated as several particles. Note, the motivation for long
muons is that a long muon with clearly delta ray, so the vertex can be
3-segement vertex, which broke one muon to several. … If we broke a long muon to
several segments, then we cannot get the range correctly. In addition, I also
want you to audit the MCS calculation for the long muon. In this case, we need
to exclude the delta ray which can bias the calculation of the MCS. I am not
ready to modify the code yet, but for this round, you can audit, identify the
cases from the existing events, and make suggestions on how to improve and plan
the validation step."*

**Status. AUDIT + MEASUREMENT ONLY. No C++, no jsonnet and no existing arm is
touched by this doc.** Every proposal in §6 is a *shape* for the owner to
filter; none is implemented.

**Headline.** The owner's mechanism is real and now counted. On 1366 beam events
with a PR dump, the toolkit builds **242 muon-typed pseudo-showers**; **62** of
them carry one of the two defects, and **22** carry the literal signature the
owner described — a muon whose chain runs through an internal junction with a
non-muon arm hanging off it. But the two halves of the ask land differently:

- **Chain construction is a faithful 1:1 port** (§1; already certified by
  doc pr/113 §1.3/§2.5, re-verified here). What breaks the muon is not a port
  bug — it is the **10° collinearity cut inside `find_cont_muon_segment`**, and
  an offline replay that **reproduces the shipped chain length exactly** on the
  reference cases shows the muon being dropped at **10.99°** (evt 313847) and
  **14.62°** (evt 66366). §4.
- **The range is not merely "wrong when broken" — it is often literally zero.**
  `kine_range == 0` on **28 / 242 (11.6%)** muon pseudo-showers, because
  `calculate_kinematics_long_muon` is dispatched on the shower's *pdg*, but
  sums length over `segments_in_long_muon` *membership*. When the two disagree
  the range energy is 0 no matter how much muon is there: **26 of those 28
  showers hold ≥2 muon-typed segments, 12 hold more than 45 cm and the worst
  holds 332.8 cm.** §3.2.
- **Delta rays already bias the shipped dQ/dx energy**, exactly as the owner
  feared, but in the *charge* estimator rather than in MCS: **128 / 242 (52.9%)**
  pseudo-showers contain a non-muon member whose charge feeds `kine_dQdx`, and
  the induced shift has a heavy tail — median 0.9% of the energy, **p90 20%,
  max 100%**. §3.3. **Removing it is not a free win** (§6, P6): on the clean
  population it moves the median 2% the wrong way, so it fixes the tail and
  needs the mode-2 ratio window re-derived.
- **For MCS the delta-ray premise is already satisfied by the shipped default**
  (`point_source = "muon_segments"`), with three named residual holes. The live
  MCS problem is different: in the doc-80 production arm **every one of the 290
  MCS evaluations used `muon_source = pf_muon` with `nseg = 1`** — a single
  segment — and in **37 / 287** of those the selected segment is short of the PR
  muon by ≥20%. §5.

**These are SBND *data*** (MCP2025C reco1, `reality=data`, no `simb::MCTruth`;
doc 83, doc pr/113 §6.1). "numu events" means beam-data numu candidates. Every
number below is a reconstruction verdict; purity is unknown and unknowable on
these arms. The one truth handle in the tree is named in §7.

---

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin

# the whole census (pure read of existing arms; no WCT job is run)
python3 scripts/d84_longmu_census.py \
    --calib-arms work-mcp1k-prod0825:mcp1k work-mcp2k-prod0825:mcp2k \
    --mcs-arm    work-mcp1k-mcs80on \
    --out docs/84_longmu --out-exist-ok
echo rc=$?          # exit-code discipline: never judged through a pipe (M14)
```

Writes `docs/84_longmu/{summary.txt, d84-longmu-showers.tsv, d84-gate-replay.tsv,
d84-broken-muons.tsv, d84-mcs-sentinel.tsv, d84-mcs-vs-pr-muon.tsv,
d84_longmu_overview.png}`. Runtime ~4 min single-threaded.

### Anchors, and a warning about line numbers

| what | anchor |
|---|---|
| toolkit | **`06dfa09f`** (`apply-pointcloud`), 2026-08-28 |
| wcp-porting-img | **`e16624d`** (`main`) |
| the census arms `work-mcp{1k,2k}-prod0825` | produced 2026-08-25 at toolkit **`2aba11dc`**, wcp `b04a94e` (doc 81, "Binary:" line) — current production at the shipped operating point |
| the MCS arm `work-mcp1k-mcs80on` | `SBND_MCS=1 ./run_pr_chain_batch.sh work-mcp1k-grp0825 work-mcp1k-mcs80on data`, all 1000 events rc=0 (doc 80 §18) |
| prototype | `prototype_base/` → `/home/xqian/work/scratch_wcgpu1/prototype-dev`, `wire-cell/pid/` |

**Every toolkit line number in this doc is `git show 06dfa09f:<file>`, not the
working tree.** At the time of writing the tree carried uncommitted probe edits
from an unrelated in-flight round (doc pr/122: `WCT_SHOWER_PID_DEBUG` in
`PRShower.cxx` / `NeutrinoShowerClustering.cxx`, plus `wct-pr-perevt.jsonnet`,
`TaggerCheckNeutrino.{h,cxx}`, `NeutrinoPatternBase.h`,
`doctest_clus_knob_defaults.cxx`), and the set *grew during this session*.
Working-tree lines are already shifted by up to 47 (`calculate_kinematics_long_muon`
is HEAD `PRShower.cxx:1737`, worktree `:1784`). Re-derive before quoting.

---

## 0. Scope, and what this doc does NOT repeat

`sbnd_xin/docs/pr/113_em-shower-pi0-long-muon-coverage-audit.md` §1.3 and §2.5
already did the **coverage/parity** audit of long-muon construction against the
prototype and found the port complete, with one candidate divergence (F2, the
dropped `cal_4mom()` guard). That verdict was **re-verified here** by reading
both sides again (§1); it is not re-derived, and F2 is not re-litigated.

This doc is about the **consumers** — what the chain's length and membership are
used *for*, and where that produces a wrong number. Four things are in scope:

1. §1 the construction site and its structural limits (short);
2. §3 the energy consumer, `calculate_kinematics_long_muon`;
3. §4 an offline replay of `find_cont_muon_segment` that names which gate breaks
   each muon;
4. §5 the MCS driver.

Not in scope: `cal_kine_charge` internals; the tagger bodies (docs 74/75, pr/36)
except their use of the long-muon sets; the upstream `Mcs::MuonMCS` numerics
(doc 80/83 own those).

---

## 1. Construction vs the prototype: a faithful port, with faithful limits

Toolkit `NeutrinoVertexFinder.cxx:1860-1919` (inside
`PatternAlgorithms::examine_direction`, `:1466`) is 1:1 with prototype
`NeutrinoID_track_shower.h:2137-2190`:

| step | toolkit | prototype |
|---|---|---|
| fill-once-per-cluster guard | `:1861-1867` | `:2137-2142` |
| seed cut, `median dQ/dx / m_mip_dqdx_median > 1.3 → skip` | `:1877-1878` | `:2145` |
| chain walk `find_cont_muon_segment` | `:1888-1893` | `:2157-2162` |
| accept gate `total > 45 cm && max > 35 cm && size > 1` | `:1902` | `:2169` |
| retype every member to pdg 13, clear both shower flags | `:1905-1913` | `:2171-2183` |
| insert into `segments_in_long_muon` / `vertices_in_long_muon` | `:1912`, `:1915` | `:2183`, `:2186` |
| purge non-main-cluster members at the end | `:5605-5632` | `NeutrinoID.cxx:393-409` |

`find_cont_muon_segment` (toolkit `:1339-1464`, prototype `:2304-2369`) is
likewise 1:1: 15 cm-lever direction, `angle < 10°` (or `< 15°` when the incoming
segment is `< 6 cm`), a 50 cm-lever re-test for candidates `> 50 cm`, candidate
`dQ/dx ratio < 1.3`, winner by `length·cos(angle)`. The toolkit adds two things
and nothing else: a zero-magnitude direction guard (`:1377`, the prototype has
none) and the SBND-ON `flag_stub_bridge` (`:1415-1432`, doc pr/46). Iteration is
`sorted_out_edges` rather than the prototype's pointer-ordered map — more
deterministic, not a behaviour change.

### 1.1 Four structural limits a parity check cannot see

All four are present in the prototype too, so **none is a port defect (M15-clean)**
— and all four are load-bearing for the owner's question:

- **L1 — the chain is only ever seeded at the main vertex.** `examine_direction`
  is called only as `examine_direction(graph, main_vertex, main_vertex, …)`
  (`:3964`, `:5362`, `TaggerCheckNeutrino.cxx:2521`; prototype
  `NeutrinoID_track_shower.h:1381`, `NeutrinoID.cxx:225`). A muon whose first
  piece *at the main vertex* fails the `1.3×MIP` seed cut yields no chain at
  all, however long. Measured: **1419** main-vertex seeds rejected here.
- **L2 — an unbroken muon can never be in the set.** `acc_segments.size() > 1`.
  Measured: **936** main-vertex seeds rejected on this alone, of which **532 are
  longer than 45 cm by themselves** (median 143 cm, max 498 cm). Those counts
  are over *seeds*, not over muons — many are cosmic arms at the main vertex —
  but they bound the population the `long_muon` MCS arm can never see (§5.3).
- **L3 — an evenly-broken muon is rejected outright.** `max_length > 35 cm`
  kills e.g. 3 × 30 cm = 90 cm. Measured: only **4** seeds fail on this alone —
  a real but rare mechanism.
- **L4 — the chain labels, it never re-joins.** Members stay separate
  `PR::Segment`s. Only a consumer that iterates the *set* sees the full length;
  a consumer holding one segment sees a fragment. That is the whole of §5.

The walk is also greedy (best `length·cos(angle)`, no backtracking) and upstream
carries no visited set; the replay in §4 adds a 64-step guard and reports `loop`
separately (it never fired).

---

## 2. The measurement, and what is on disk

`calib-pr-evt<ID>.json` (written only under `PR_EXTRA_STAGES=pr_display`, which
prod0825 used, `PrDisplayDump.cxx`) carries everything needed to reconstruct the
PR graph offline: `segments[]` with `id = cluster_id*1000 + graph_index`,
`start/end_vertex_id`, `length` (fitted arc length, cm), `particle_id` (pdg),
`flag_shower`, `shower_id`, and per-fit `points[].{x,y,z,dQ,dx,rr}`;
`vertices[]` with `fit`, `degree`, `is_main`; `showers[]` with `kine_range`,
`kine_dQdx`, `kine_charge`, `kine_best`, `total_length`. The membership join is
`segments[].shower_id == showers[].id` (`PrDisplayDump.cxx:490` vs `:577`).

**Two denominators, both reported** (pr/113 §6.2 precedent, they are not
interchangeable):

| pool | mcp1k | mcp2k | total |
|---|---|---|---|
| events in the arm | 1000 | 2000 | 3000 |
| events with `calib-pr-evt*.json` (TaggerCheckNeutrino evaluated a main cluster) | **461** | **905** | **1366** |
| muon-typed pseudo-showers found | 88 | 154 | **242** |

### 2.1 One real closure test, one weak one, and the precondition that matters

All three are printed by the script:

```
CLOSURE max |dqdx_all/kine_dQdx - 1| = 4.44e-16 over 242 showers
(weak)  max |range(L_chain) - kine_range| = 9.00e-04 MeV
PRECONDITION start-segment pdg == 13: 241 / 242
  L_chain MEANINGLESS (wrong range table): mcp2k evt 294174 shower 16028 start_pdg=11 kine_range=0.0001
```

- The first re-integrates `cal_kine_dQdx` offline — the SBND
  `PowerBoxRecombination` operating point (`clus.jsonnet` `sbnd_power_recomb`,
  `use_power_recomb=true`; `Gen::PowerBoxRecombination::dE`,
  `gen/src/RecombinationModels.cxx:160-176`) over the *same member set* — and
  reproduces the shipped `kine_dQdx` to machine precision. So "what happens if
  the delta rays are excluded" (§3.3) is a real subtraction, not a model guess.
- The second inverts the shipped SBND `muon_range_function` (a 1067-point
  `LinterpFunction`, range cm → KE MeV, `particle_dataset.jsonnet:136`) on
  `kine_range` and re-forwards it. **On its own this shows only that the
  interpolation is monotone and invertible — it is not independent evidence
  the way the dQ/dx closure is.** What makes `L_chain` trustworthy is a
  separate precondition, checked explicitly: `cal_kine_range` is called
  (`PRShower.cxx:1781`) with `particle_type = abs(m_start_segment->
  particle_info()->pdg())` — the **start segment's** pdg, *not* the shower type
  this census filters on (`PrDisplayDump.cxx:579`) — so the muon table is the
  right one only if the start segment is itself pdg 13.

  Checked over all 242: `showers[].id` is `pf_node_id(shower->start_segment())`
  (`PrDisplayDump.cxx:577`), so the start segment is `segments[id]`.
  **241 / 242 have a pdg-13 start segment.** The single exception is
  `mcp2k` evt 294174 shower 16028, whose start segment is **pdg 11** — its
  `kine_range = 0.0001 MeV` came from the *electron* table and its `L_chain`
  (and therefore `L_unchained_cm`) is meaningless. That row is classified
  `deltaray_charge` on charge fraction, not on chain length, so the case list
  is unaffected; it is flagged here and in §4.3 rather than silently kept.
  (That a muon-typed shower can carry a non-muon start segment is the subject
  of the in-flight doc pr/122; one case in 242 here.)

---

## 3. The energy consumer

### 3.1 Where the numbers come from

`NeutrinoEnergyReco.cxx:331` dispatches every shower with `|pdg| == 13` to
`Shower::calculate_kinematics_long_muon` (`PRShower.cxx:1737`), which:

- accumulates `total_length` **only over `segments_in_long_muon` members**
  (`:1765-1771`, guarded by `bool in_muons`), then
  `data.kenergy_range = cal_kine_range(total_length, 13, …)` (`:1781`);
- accumulates `vec_dQ`/`vec_dx` over **every segment in the pseudo-shower**
  (`:1773-1777` — *outside* the `if (in_muons)` block), then
  `data.kenergy_dQdx = cal_kine_dQdx(...)` (`:1782`);
- sets `data.kenergy_best = data.kenergy_dQdx` (`:1785`), which doc pr/101 K4's
  `kine_long_muon_mode` (SBND = **2**, `wct-pr-perevt.jsonnet:1779`) may replace
  with range when `kenergy_range > 0`, `end_degree == 1`, and
  `kine_dQdx/kine_range ∈ [1-0.3, 1+0.5] = [0.7, 1.5]` (`:1815-1826`).

The prototype is **identical**, including the length/charge asymmetry, and
carries the original author's own TODO in-code:

```cpp
   kenergy_range = start_segment->cal_kine_range(L);
   kenergy_dQdx  = start_segment->cal_kine_dQdx(vec_dQ, vec_dx);
   // long muon ... // should be improve in the future using range ...
   kenergy_best  = kenergy_dQdx;
```
`prototype_base/wire-cell/pid/src/WCShower.cxx:288-303`. **M15-clean: not a
divergence, a carried limitation.**

### 3.2 The range is zero on one muon shower in nine

| outcome for `kine_best` | count / 242 |
|---|---|
| resolved to `kine_range` | 184 (76.0%) |
| resolved to `kine_dQdx` | 30 (12.4%) |
| resolved to `kine_charge` (a later SBND-ON override) | 28 (11.6%) |

and the reason mode 2 did **not** take range:

| why | count / 242 |
|---|---|
| `kine_range == 0` — the chain contributed **no length at all** | **28 (11.6%)** |
| `kine_dQdx/kine_range` outside `[0.7, 1.5]` | 9 (3.7%) |
| `end_degree != 1`, or a later override won | 21 (8.7%) |

**The `kine_range == 0` class is a dispatch/membership mismatch, and it is the
single largest clean defect this doc found.** `NeutrinoEnergyReco.cxx:331`
routes a shower into `calculate_kinematics_long_muon` on `|pdg| == 13` — the
shower's own particle type — but the function measures length over
`segments_in_long_muon` *membership*. When a muon-typed shower carries no chain
member, `total_length = 0` and `cal_kine_range(0, …) = 0`, however much muon
track the shower actually holds:

| composition of the 28 `kine_range == 0` showers | count |
|---|---|
| carry ≥2 muon-typed member segments | **26 / 28** |
| carry > 45 cm of muon-typed length | 12 / 28 |
| carry > 100 cm | 8 / 28 |
| largest | **332.8 cm** |

Worked example, `mcp2k` evt 497311 shower 15000: one muon-typed segment of
**332.8 cm**, `kine_range = 0`, `kine_dQdx = 722.9 MeV`, shipped
`kine_best = kine_charge = 508.5 MeV`. A 333 cm muon's range energy is ≈760 MeV.
(This particular event is also an instance of L2 — a single segment can never be
a "long muon" — but L2 explains only **2 of the 28**; the other 26 have multiple
muon-typed members and simply have no chain.)

### 3.3 Delta-ray charge in the dQ/dx integral — the tail, not the median

| metric over the 242 muon pseudo-showers | count |
|---|---|
| ≥1 **non-muon** member segment | **128 (52.9%)** |
| non-muon member *length* > 5% of the muon length | 55 (22.7%) |
| non-muon member *charge* > 5% of the total | **45 (18.6%)** |

The energy this adds (offline subtraction, closure-verified §2.1):

```
dQ/dx delta-ray shift:  median 5.0 MeV   p90 42.5 MeV   max 404.8 MeV
as a fraction of kine_dQdx: median 0.009   p90 0.204   max 1.000
```

Two illustrations from the case list:

- **`mcp2k` evt 99232 shower 17013** (hand-verified, §4.2): a 236.6 cm + 10.3 cm
  muon chain, complete, plus an absorbed **40.0 cm proton** (median dQ/dx =
  2.21×MIP) and a 16.8 cm electron. 26.6% of the charge is non-muon.
  `kine_range = 570.5 MeV`, `kine_dQdx = 795.9 MeV`, and `kine_best` takes
  **dQdx** because the ratio 1.395 sits *inside* the `[0.7, 1.5]` window. Excluding
  the two non-muon members gives 570.5 MeV — agreeing with range to the printed
  precision (one case; see the population result below before reading anything
  general into it).
- **`mcp2k` evt 294174 shower 16028**: `dQ_frac_oth = 1.000` — a 1.8 cm muon
  segment and **8** non-muon members totalling 157.6 cm. The entire
  `kine_dQdx = 404.8 MeV` is daughter charge.

**And the honest population result, which changes the recommendation.**
Restricting the chain's dQ/dx to chain members does **not** globally improve
agreement with range:

| population (`kine_range > 50 MeV`) | n | `dQdx_all/range` median | `dQdx_chain/range` median | outside `[0.7,1.5]` |
|---|---|---|---|---|
| all | 203 | 0.982 | 0.961 | 18 → 20 |
| contaminated (`dQ_frac_oth > 0.05`) | 25 | 1.078 | **0.912** | 3 → 5 |
| clean (`dQ_frac_oth ≤ 0.05`) | 178 | 0.979 | 0.967 | 15 → 15 |

`docs/84_longmu/d84_longmu_overview.png` shows all three: the non-muon charge
fraction, the unchained muon length, and the `kine_dQdx` vs `kine_range` scatter
with the mode-2 `[0.7, 1.5]` fallback window drawn.

So the delta-ray contamination is a **heavy-tail** effect, not a median bias, and
the naive fix pulls the median ~2% low and by the shipped ratio-window metric is
a wash. §6 P6 is written accordingly: it is a tail fix that requires the mode-2
window to be re-derived, not a drop-in improvement.

---

## 4. Gate replay — which cut breaks the muon

`scripts/d84_longmu_census.py` re-implements `find_cont_muon_segment` and the
accept gate directly from the calib JSON (`segment_cal_dir_3vector`,
`PRSegmentFunctions.cxx:2363-2390`, is a mean-of-nearby-fit-points minus the
vertex, normalised; `segment_median_dQ_dx`, `:1608`, is a median of
`dQ/(dx+1e-9)`; both reproduce trivially), then walks from every main-vertex
seed with `flag_stub_bridge` ON and OFF.

> **Caveat, stated because it bounds every number in this section.** The calib
> JSON is the **final** PR state; `examine_direction` ran earlier, on a graph
> that may since have changed. This is *not* a bit-exact re-execution.
> Measured agreement: on the 105 events where both a shipped chain length and an
> accepted replay chain exist, **86 / 105 (82%)** agree within 5%. The 18% is the
> epoch gap. On the reference cases below the agreement is exact.

### 4.1 Where the seeds die

| gate (stub_bridge ON) | seeds |
|---|---|
| `median dQ/dx > 1.3×MIP` on the seed (L1) | 1419 |
| `acc_segments.size() > 1` (L2) | 936 |
| **accepted** | **103** |
| `max_length > 35 cm` (L3) | 4 |

and where an accepted or attempted walk *stopped*:

| stop reason | ON | OFF |
|---|---|---|
| seed rejected before the walk | 1419 | 1419 |
| genuine dead end (degree-1 vertex) | 811 | 811 |
| **angle test** | **116** | 117 |
| angle **and** dQ/dx | 86 | 86 |
| dQ/dx only | 30 | 29 |

**`long_muon_stub_bridge` (SBND ON) changes exactly one walk in 1366 events on
the final graph.** That is not evidence the knob is dead — pr/46 measured it at
`examine_direction` time, where short stubs are far more common — but on the
final graph its `sg_length < 6 cm` precondition almost never holds. It is
therefore *not* a meaningful delta-ray-into-the-chain vector here (§5).

### 4.2 The angle cut is what breaks the muon, and it breaks it by a hair

Of the 202 walks stopped by the angle test, the distribution of the rejected
candidate's angle:

| angle of the best rejected continuation | walks | of which the candidate is > 20 cm |
|---|---|---|
| 10–12° | 6 | 5 |
| 12–15° | 9 | 8 |
| 15–20° | 11 | 6 |
| 20–30° | 31 | 7 |
| > 30° | 145 | 20 |

**25 candidates longer than 50 cm were rejected on angle alone; their angles have
median 15.6° and minimum 10.4°** — i.e. a population of long, MIP-like, nearly
collinear continuations sitting just outside a 10° cut.

Two hand-verified reference cases (raw calib JSON read by hand, then compared
with the automated verdict):

**`mcp2k` evt 66366, shower 22006 — the owner's picture exactly.**

```
seg 22009  pdg=13  L=174.43 cm  medianQ/MIP=1.13  v(22007,22000) deg(2,1)
seg 22006  pdg=13  L= 91.68 cm  medianQ/MIP=1.15  v(22005,22001) deg(3,2)
seg 22005  pdg=13  L= 27.67 cm  medianQ/MIP=1.11  v(22003,22005) deg(2,3)
seg 22007  pdg=11  L=  6.85 cm  medianQ/MIP=0.86  v(22005,22006) deg(3,1)   <- the delta ray
seg 22008  pdg=13  L=  6.81 cm  medianQ/MIP=1.04  v(22007,22003) deg(2,2)   <- the stub
```
One muon of **300.6 cm** in four MIP pieces (all four well under the 1.3 seed
cut), with a 6.85 cm electron hanging off the degree-3 vertex 22005 — the
3-segment vertex. Replay: seed 22006 → accepted, 3 segments, **126.2 cm**,
which is **exactly the shipped `L_chain = 126.2 cm`**. The walk stopped with
`stop=angle, candidate 174.4 cm, angle 14.62°, ratio 1.129, stop vertex degree 2`.
Shipped result: `kine_range = 307.0 MeV` (for 126 cm) instead of ≈680 MeV;
ratio 2.14 is outside the window so `kine_best` falls to `kine_dQdx = 656.0 MeV`.
Note the stub that carries the direction into the break is **6.81 cm** — 0.81 cm
*above* `long_muon_stub_bridge`'s `< 6 cm` precondition.

**`mcp1k` evt 313847, shower 19014 — one degree of arc.**

```
seg 19001  pdg=13  L=162.40 cm  medianQ/MIP=1.09  v(19000,19002) deg(1,2)
seg 19022  pdg=13  L= 78.28 cm  medianQ/MIP=1.12  v(19002,19008) deg(2,3)
seg 19014  pdg=13  L= 20.27 cm  medianQ/MIP=1.11  v(19008,19006) deg(3,3)
seg 19015  pdg=11  L=  6.85 cm  medianQ/MIP=1.21  v(19008,19009) deg(3,1)   <- the delta ray
```
260.9 cm of muon in three pieces; delta ray at the degree-3 vertex 19008.
Replay: accepted, 2 segments, **98.54 cm** = the shipped `L_chain` exactly.
Stop: `angle, candidate 162.4 cm, **angle 10.99°**, ratio 1.092, degree 2`.
**The 162 cm continuation missed the 10° cut by 0.99°.** Shipped
`kine_range = 248.3 MeV` for a 261 cm muon (true range energy ≈ 575 MeV);
`kine_best = kine_dQdx = 547.5 MeV`.

**`mcp2k` evt 99232, shower 17013 —** the contamination case quoted in §3.3;
chain complete, defect is entirely in the dQ/dx integral.

### 4.3 The delivered case list

`docs/84_longmu/d84-broken-muons.tsv`, **62 rows**, one per muon pseudo-shower
that shows either defect, sorted worst-first:

| class | rows |
|---|---|
| `deltaray_charge` — chain complete, >5% of the charge from non-muon members | 35 |
| `broken_chain` — >20 cm of muon-typed length outside the chain, or `kine_range==0` with >45 cm of muon | 17 |
| `both` | 10 |
| of these, with an internal junction carrying a non-muon arm (`n_deg3_internal > 0`) | **22** |
| of these, with an MCS sentinel in `work-mcp1k-mcs80on` | 19 |

Columns: `arm run subrun evt shower defect n_mu n_oth L_mu_cm L_chain_cm
L_unchained_cm n_deg3_internal min_arm_cm dQ_frac_oth dqdx_shift_MeV kine_range
kine_dQdx kine_charge kine_best best_is fallback_why mcs_*`. `L_chain_cm` is the
inverted-`kine_range` chain length (§2.1), `L_unchained_cm = L_mu − L_chain`.

Worst broken chains: evt 497311 (332.8 cm, chain 0), 177536 (294.8, 0),
74784 (291.3, chain 29.9), 53793 (240.9, 0), 281595 (351.0, chain 130.0),
178885 (195.0, 0), 66366 (300.6, chain 126.2), 313847 (260.9, chain 98.5).
Worst contamination: 294174 (100% of charge), 286681 (73%), 54629 (28%),
99232 (27%), 288327 (25%).

> **Three honesty notes on the table.** (i) `L_unchained` is clamped at 0; in a
> few rows `L_chain > L_mu` (e.g. evt 101828: 128.9 vs 110.6 cm), meaning the
> chain contains a member whose *final* pdg is no longer 13 — the epoch gap
> again. (ii) `n_deg3_internal` counts junctions shared by ≥2 muon-typed members
> that also carry a non-chain arm; it is a proxy for the owner's 3-segment
> vertex, not a truth label. (iii) **`mcp2k` evt 294174's `L_chain_cm` /
> `L_unchained_cm` are meaningless** — its start segment is pdg 11 so
> `cal_kine_range` used the electron table (§2.1). The row is kept because its
> defect is charge contamination (`dQ_frac_oth = 1.000`), which does not depend
> on the range inversion.

---

## 5. MCS

**Operating point first: `mcs_enable = false` in SBND production**
(`wct-pr-perevt.jsonnet:1196`). Nothing in this section affects the shipping
chain; it is about the doc-80 arms and about what to enable.

### 5.1 The delta-ray premise is already satisfied — with three named holes

With the shipped default `point_source = "muon_segments"`
(`MuonMCSDriver.cxx:228-230`) the cloud handed to `Mcs::MuonMCS::run` is *only*
the selected muon segments' `fits()` points. **A delta ray that is its own
`PR::Segment` never enters.** The residual holes, in order of how live they are:

1. **A chain member admitted only by `long_muon_stub_bridge`'s 45° relaxation**
   (`NeutrinoVertexFinder.cxx:1415-1432`, SBND ON). Its explicit veto already
   excludes shower-flagged and pdg-±11 arms (`:1422-1425`), and §4.1 measures it
   changing **one** walk in 1366 events on the final graph. **Effectively not a
   vector.**
2. **Delta-ray charge merged inside a muon segment's own fit trajectory.** Not
   separable at this layer at all — the fit points *are* the trajectory. This is
   the only genuinely open channel, and it is a track-fitting question, not an
   MCS one.
3. **`point_source = "whole_event"`** (`:210-227`) harvests every segment and
   every vertex in the graph. Documented validation-only (doc 80 §7.3); it must
   never be a production setting, and this doc records why.

The upstream header says extra activity "is tolerated — stage 1 trims it"
(`mcs/inc/WireCellMcs/MuonMCS.h:215-217`). **Tolerated is weaker than excluded**,
and the shipped default does not rely on it. Say so rather than leaning on trim.

### 5.2 The live MCS problem is L4, not delta rays

In `work-mcp1k-mcs80on` — 1000 events, the doc-80 round-4 production arm — the
log sentinel fires on **289 events / 290 evaluations**, and:

```
muon_source: {'pf_muon': 290}
nseg (segments handed to MCS): {'1': 290}
```

**Every single MCS evaluation used one segment.** `muon_source = "pf_muon"`
(`MuonMCSDriver.cxx:80-97`) ranks `|pdg| == 13` segments by kinetic energy and
pushes exactly one. Three consequences:

- the MCS trajectory is that fragment's points;
- `muon_min_length_cm = 40` is applied to the fragment, so a muon that is long
  only in total can be skipped entirely;
- the logged comparator `ke_range_toolkit = cal_kine_range(total_length_cm, 13, …)`
  (`:270`) is the **fragment's** range energy.

Measured **within the same arm** (no cross-arm join — the arms are different
binaries and segment the same track differently), against `T_kine`'s
`kine_energy_particle` for the muon, i.e. what the PR chain itself assigned:

```
PR muon KE / ke_range_toolkit(selected segment):  median 1.000   p90 1.328
the selected segment is short of the PR muon by >= 20%:  37 / 287
ke_MCS / ke_range_toolkit:  median 1.123 ;  > 1.5 in 101 / 287
```

So in the median event the pf muon *is* the whole muon (consistent with L2: most
muons are single segments). But in **37 / 287 (12.9%)** the MCS comparator is
computed on a fragment carrying ≥20% less energy than the PR muon — and those
events land in the high-side `ke_MCS/ke_range` tail that doc 83 investigated.

> **Relation to doc 83.** Doc 83 §2 step 4b tested "is there an adjacent
> fragment" and correctly found it non-discriminating (67% of outliers, 73% of
> all contained muons). The measurement here is a different one — *how much
> muon-typed energy sits outside the segment MCS was given* — and it is not a
> refutation of doc 83's conclusion, which was about the statistical mechanism.
> It is a second, additive contribution worth separating before the tail is
> re-attributed. **An earlier cross-arm version of this number (a "median 74%
> length coverage") is retracted:** joining `work-mcp1k-prod0825` to
> `work-mcp1k-mcs80on` by event number is invalid, and the within-arm number
> above (median 1.000) is the one to quote.

### 5.3 Two further findings in the driver

- **`muon_source = "long_muon"` (`:98-111`) is the complement of `pf_muon`, not
  a superset.** It takes the whole chain — but by L2 the chain requires ≥2
  segments, so this arm **silently skips every unbroken muon** (532 of the
  main-vertex seeds here would be >45 cm on their own). **There is no hybrid
  arm.** That is why the default is `pf_muon`, and why neither arm is right.
- **The endpoint logic (`:147-200`) assumes exactly two degree-1 extremes.** A
  chain with a mid-chain branch falls into the most-separated-pair fallback
  (`:174-193`). Of the 62 case-list rows, **22** have at least one internal
  junction with an extra arm, so the fallback is not a corner case for the
  `long_muon` arm.
- **Closed, not a concern:** `segments_in_long_muon` cannot mix clusters by the
  time MCS runs — `determine_overall_main_vertex` purges non-main-cluster
  members at `NeutrinoVertexFinder.cxx:5605-5632`, matching prototype
  `NeutrinoID.cxx:393-409`, and `mcs_fill_kine` is called after that
  (`TaggerCheckNeutrino.cxx:2834`).

### 5.4 The prototype has no MCS

A full sweep of `prototype_base/` finds **no scattering-angle momentum
estimator** — no Highland formula, no `theta_rms`/`sigma_theta`, no MCS fitter.
The only `wire-cell/mcs/` there is truth-level *simulation*
(`WCPMCSSim/MCSSim.h`, `Eloss.h::get_mcs_angle`), never called from any
`NeutrinoID_*` / `ProtoSegment` reconstruction file. The prototype's estimators
are range (`ProtoSegment::cal_kine_range`, `ProtoSegment.cxx:1380`) and dQ/dx
only. **So "audit MCS against the prototype" has no referent**: the MCS audit is
against the ubreco/MicroBooNE upstream and doc 80/83, and is recorded as such.

---

## 6. Improvement proposals — shapes only, nothing shipped

Each is a default-OFF knob with a hit counter, so a rate can be measured before
any flip (pr/32 F3 / pr/113 precedent: ship the counter, read it, *then* decide).
Ordered by how well the measurement supports them.

| # | knob | site (HEAD `06dfa09f`) | what, and what the measurement says |
|---|---|---|---|
| **P1** | `long_muon_range_empty_chain_fallback` | `PRShower.cxx:1765-1781` | When the chain contributes **zero** length, fall back to the muon-typed member length of the pseudo-shower (or, if there is none, the start segment's own length) instead of leaving `kenergy_range = 0`. **Directly addresses the 28/242 `kine_range == 0`** — the largest and least ambiguous defect found, and one whose cause (dispatch on pdg, measure on membership, §3.2) is unambiguous. Strictly widens where mode 2 can take range; changes nothing when the chain is non-empty. Ship the counter first: 26 of the 28 have ≥2 muon members, so the fallback is a *sum*, not a single segment. |
| **P2** | `long_muon_angle_relax_long` | `NeutrinoVertexFinder.cxx:1401-1402` | Raise the continuation angle cut for a **long** (> 50 cm) MIP candidate from 10° to ~16°, keeping 10° elsewhere. **25 such candidates were rejected on angle, median 15.6°, min 10.4°**; evt 313847 misses by 0.99°, evt 66366 by 4.62°. Cheapest fix for the owner's exact mechanism. Must ship with the junction veto `long_muon_stub_bridge` already uses (`:1418-1430`) so a hadronic vertex is not bridged. |
| **P3** | `long_muon_stub_bridge_len` | `NeutrinoVertexFinder.cxx:1416` | Make the stub-bridge precondition `sg_length < 6 cm` a configurable length. evt 66366's stub is **6.81 cm** — the shipped knob misses it by 0.81 cm. A one-number generalisation of an already-approved mechanism. |
| **P4** | `mcs_muon_source = "long_muon_else_pf"` | `MuonMCSDriver.cxx:80-127` | The missing hybrid arm: the chain when one exists, else the pf muon segment. Without it neither shipped arm covers both populations (§5.3). Prerequisite for enabling MCS on broken muons at all. |
| **P5** | `mcs_range_comparator_chain` | `MuonMCSDriver.cxx:264-276` | Make the logged `ke_range_toolkit` comparator use the long-muon chain rather than the single selected segment. **Log-only, no output bytes move** — the cheapest way to test §5.2's 37/287 against doc 83's tail. Would be the natural first MCS step. |
| **P6** | `kine_long_muon_dqdx_chain_only` | `PRShower.cxx:1773-1777` | Restrict `vec_dQ`/`vec_dx` to chain members so delta-ray charge stops feeding the muon's dQ/dx energy. **Deliberately ranked last.** §3.3 measures it as a tail fix that moves the clean-population median 2% low and, by the shipped `[0.7, 1.5]` metric, is a net wash (18 → 20 outside). It should not ship without re-deriving `kine_long_muon_ratio_{lo,hi}`, and probably wants an `if contaminated` guard rather than an unconditional restriction. |
| **P7** | `mcs_exclude_shower_members` | `MuonMCSDriver.cxx:228-230` | Drop shower-flagged / pdg ±11 chain members from the harvested cloud. **Listed for completeness and NOT recommended now**: §5.1 measures the only route by which such a member enters (the stub bridge) at one walk in 1366 events. |

Not proposed, deliberately: relaxing L2 (`acc_segments.size() > 1`) at the
construction site. It would put every long single-segment muon into
`segments_in_long_muon`, which is read by the numu and cosmic taggers
(`NeutrinoTaggerNuMu.cxx:122-144`, `NeutrinoTaggerCosmic.cxx:588-604`) and by
shower clustering — a large blast radius for a problem P1 fixes locally at the
consumer.

---

## 7. Validation plan for whichever of §6 the owner picks

Every item is a *separate* implementation round under `/wct-knob` + `/ab-verify`.

**Gate 1 — knob OFF is byte-identical.** `scripts/pr85_hash_gate.py` against the
current production arms on all four samples (`nuecc48`, `ncpi0`, `mcp1k`,
`mcp2k`; 6134 archives + `pr94_root_gate.py` 3067 ROOT files, the prod0825
counts), labels reported. Compiled-config proof that the key appears only when
on (`wcsonnet … | grep <key>`). M1 freshness proof (`local/lib/libWireCellClus.so`
mtime newer than the last source edit) *before* the A/B. `./build/clus/wcdoctest-clus`
green, and the knob pinned in `doctest_clus_knob_defaults.cxx`.

**Gate 2 — the ON effect is the intended one.** The knob's hit counter is
non-zero, and the movers are drawn from `docs/84_longmu/d84-broken-muons.tsv`.
For P2/P3 specifically, the diff of `d84-gate-replay.tsv` before/after must show
the recovered continuations and nothing else.

**Gate 3 — the physics metric, reco-internal** (these arms are data, §0):
re-run this census on the ON arm and require, on the case list:

- P1: `kine_range == 0` count falls from 28/242, and no shower that previously
  had a non-zero `kine_range` changes it;
- P2/P3: `L_unchained_cm` falls; `kine_dQdx/kine_range` concentrates *into*
  `[0.7, 1.5]` (currently 18/203 outside) rather than out of it;
- P5: `ke_MCS/ke_range_toolkit` (currently median 1.123, >1.5 in 101/287) moves
  toward 1 on the 37 fragment events specifically, with the other 250 unmoved.
  Doc 83's `80_mcs/mcs_vs_range_scatter.png` is the baseline plot.

A metric that moves the *median* of the clean population is a red flag, not a
win — that is the P6 lesson (§3.3).

**Gate 4 — the one truth handle.** For the delta-ray question specifically,
reco-internal cross-checks can only say two estimators agree, never that either
is right. **`work-mcsim-stmon`** (52 events, MC with truth dQ/dx *and* delta
rays; docs 44/46) is the only place in the tree where "did excluding the delta
rays move the estimate the right way" can be answered against truth. It is small
and is not a numu sample, so it validates the **mechanism**, not the rate. If
the owner wants a rate, a numu MC sample with truth has to be produced — that is
a request, not something this round can substitute for.

**Gate 5 — owner hand-scan.** The 22 rows with `n_deg3_internal > 0` are the
scan set; `pr_display` (docs pr/26, pr/27) renders each from its own
`calib-pr-evt<ID>.json`. A Bee before/after package only on request
(`/bee-review`; upload is owner-gated, CLAUDE.md §5.6).

---

## 8. What is NOT claimed

- No statement about efficiency or purity. These are data; there is no truth.
- The §4 replay is on the **final** graph and agrees with the shipped chain on
  86/105 events (§4 caveat). Per-gate counts are therefore indicative for the
  population and exact only on the cases hand-verified in §4.2.
- `n_deg3_internal` is a proxy for "3-segment vertex", not a label.
- The MCS numbers cover **`mcp1k` only** — `work-mcp2k-mcs80{off,offb,ref}` exist
  but there is **no `work-mcp2k-mcs80on`**, so the MCS join is 289 events while
  the topology census is 1366.
- Doc pr/113's F1/F2 candidates are neither re-derived nor closed here.
- Nothing about `Mcs::MuonMCS`'s own numerics; docs 80 and 83 own those.

---

# Round 1 — implementation: MCS ON + P1 shipped, P2/P3 built and HELD, P4/P5 shipped

**Status.** Owner-directed implementation round (2026-08-28): turn MCS ON for
SBND production, act on P1 (kine_range==0), implement P2+P3 (held OFF for the
owner hand-scan), P5, and the P4 hybrid MCS source (owner choice over pf_muon).
All C++ knobs DEFAULT OFF; the SBND production flip is a separate cfg commit
under owner pre-authorization.  Line numbers in this round refer to the doc-84
round-1 toolkit commits (the audit sections above cite `06dfa09f`).

## R1.0 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild > /home/xqian/tmp/build.log 2>&1; echo rc=$?     # then freshness proof
./build/clus/wcdoctest-clus && ./build/mcs/wcdoctest-mcs

# gate + smoke arms (98-evt manifest = the pr/117-121 gate family; 31-evt Bee set):
/home/xqian/tmp/d84_arms.sh off1                          # OFF gate arm
/home/xqian/tmp/d84r1/bee_arms.sh beeoff                  # Bee OFF (31 evts)
/home/xqian/tmp/d84r1/bee_arms.sh beeonB SBND_MCS=1 SBND_MCS_MUON_SOURCE=long_muon_else_pf \
    SBND_MCS_RANGE_COMPARATOR_CHAIN=1 SBND_LONG_MUON_RANGE_FALLBACK=1
/home/xqian/tmp/d84r1/bee_arms.sh beeon2B <same> SBND_LONG_MUON_ANGLE_RELAX=1 SBND_LONG_MUON_STUB_BRIDGE_LEN=7.5
python3 scripts/pr85_hash_gate.py work-pr120r1-flipchk-<s> work-d84r1-off1-<s>
python3 scripts/pr85_hash_gate.py work-d84r1-on98-<s> work-d84r1-flipchk98-<s>   # flip equivalence
```

## R1.1 What shipped (knob table)

| # | config key | site | C++ default | SBND 2026-08-28 |
|---|---|---|---|---|
| P1 | `long_muon_range_empty_chain_fallback` | `PRShower.cxx` `calculate_kinematics_long_muon` + TCN post-reconcile recompute pass | false | **PRODUCTION ON** |
| P2 | `long_muon_angle_relax_long` / `long_muon_angle_relax_deg` | `NeutrinoVertexFinder.cxx` `find_cont_muon_segment` (6th param, formation walk only) | false / 16.0 | **PRODUCTION ON** (owner-directed post Bee scan; measured latent — see R1.9) |
| P3 | `long_muon_stub_bridge_len` | stub-bridge `< 6 cm` precondition as a member | 6.0 cm | **PRODUCTION ON at 7.5 cm** (owner-directed post Bee scan; one mover, 66366 — see R1.9) |
| P4 | `mcs_muon_source = "long_muon_else_pf"` | `MuonMCSDriver.cxx` new arm: chain when one exists, else the pf muon | `"pf_muon"` | **PRODUCTION ON** |
| P5 | `mcs_range_comparator_chain` | second log-only `mcs: chain comparator` INFO sentinel | false | **PRODUCTION ON** |
| — | `mcs_enable` | existing doc-80 gate | false | **PRODUCTION ON** (books the five `kine_mcs_*` T_kine branches) |

**P1 is two coupled pieces under one key** (the audit's §3.2 attribution was
incomplete — see R1.3): (a) the empty-chain fallback inside
`calculate_kinematics_long_muon` (sum muon-typed members when
`segments_in_long_muon` contributed nothing, and feed their vertices to the
endpoint/end_degree logic); (b) a post-`reconcile_particle_flags` pass in
TaggerCheckNeutrino that clears `flag_kinematics` on showers that are
muon-typed with `kine_range == 0` and reruns `calculate_shower_kinematics`
once — because the dominant mechanism is a shower (re)typed |13| AFTER its
kinematics pass, which keeps the generic multi-track verdict
`kenergy_range = 0 / kenergy_best = 0` forever (retyping never clears the
flag).

## R1.2 Gate ledger

| gate | arms | result |
|---|---|---|
| knob-off byte-identical (final binary) | `work-pr120r1-flipchk-*` vs `work-d84r1-off1-*` (98 evts, 4 samples) | **PASS 196/196** (28+34+38+96); `nusel-evt*.tsv` **98/98** identical |
| knob-off, first binary iteration | `work-pr120r1-flipchk-*` vs `work-d84r1-off0-*` | PASS 196/196 (superseded by off1 after the P1b recompute pass was added) |
| doctests | `build/clus/wcdoctest-clus`, `build/mcs/wcdoctest-mcs` | 2466/2466, 5651/5651 (incl. new default pins) |
| compiled config, knobs off (Step A) | `wcsonnet` full tagger pipeline, pre-edit vs post-TLA | byte-identical (`diff` empty) |
| compiled config, production flip (Step B) | same | exactly 8 keys appear: `mcs_enable/muon_source/point_source/cathode_x/cathode_xcut/range_comparator_chain`, `long_muon_range_empty_chain_fallback`, `mcs_output`; P2/P3 keys absent |
| flip equivalence | `work-d84r1-on98-*` (env TLAs, pre-flip cfg) vs `work-d84r1-flipchk98-*` (post-flip cfg, no env) | **PASS 196/196** (28+34+38+96) — one nuecc48 event (256587) rerun after an operator stop mid-arm, then gated |
| ON-state divergence signature (31 Bee evts) | `beeoff` vs `beeonB` | calib-pr json 31/31 differ (kine_mcs scalars), `mabc-pr.zip` 19/31 (PF labels where kine_best moved), `nusel` **0/31**, `pctree` **0/31** |

`work-d84r1-flipchk98-*` (98 evts × 4 samples, post-flip config, no env) is the
**new SBND production baseline** for subsequent rounds, superseding
`work-pr120r1-flipchk-*`.

## R1.3 P1 measured effect — 20/20 range_zero rescued, nothing else moves

The audit's §3.2 said the 28 `kine_range==0` showers were "dispatch (shower
pdg) vs measurement (chain membership) mismatch."  Half right.  The knob-on
runs show **none** of the dump-level range_zero population even reaches
`calculate_kinematics_long_muon`: their kinematics ran the generic
multi-track branch (`nsegments != nconnected_segs` ⇒ `kenergy_range = 0`,
`kenergy_best = 0`) while the shower was NOT yet muon-typed, and the later
muon-typing (reconcile / muon-slot competition / trunk re-rooting) never
clears `flag_kinematics`.  Hence P1b (R1.1).  With both pieces on, over the
31-event Bee set:

- **20/20 `kine_range==0` muon showers rescued** (range now > 0; best moves to
  range wherever mode 2's dead-end + ratio window admits it), e.g.
  `mcp2k` 497311 shower 15000: 332.8 cm, best **508.5 → 766.3 MeV** (charge →
  range) — the audit's worst case; `mcp2k` 177536: 294.8 cm, 511.8 → 679.0;
  `mcp1k` 286191: 13.1 → 114.3.
- **0/21 muon showers with nonzero range changed** — the pass touches only its
  targets.
- **nusel selection verdicts: 31/31 identical** — the change is energy
  bookkeeping, not selection.
- In-function fallback (P1a) alone fired on 2/31 events (chain-dispatch with
  empty set); the recompute pass (P1b) accounts for the other 18.

## R1.4 MCS ON operating point (P4+P5)

Production sentinel now reads e.g. (`mcp2k` 66366):

```
mcs: source=long_muon_else_pf nseg=3 npoints=213 len=126.2cm seg_id=22006 cluster=22 ->
  ke_MCS=869.3 MeV amb=0.7975 tracklen=125.6cm ke_range=304.3 MeV
  ke_range_toolkit=307.0 MeV ke_dqdx_toolkit=267.6 MeV (nsegs14=9 bad_path=false cathode_drop=1/1)
mcs: chain comparator nseg_chain=3 len_chain=126.2cm ke_range_chain=307.0 MeV seg_id=22006
```

- The hybrid arm uses the long-muon chain when `segments_in_long_muon` is
  non-empty (verbatim copy of the `long_muon` arm), else falls back to the
  pf muon (verbatim copy of the `pf_muon` arm).  `muon_min_length_cm` and both
  toolkit comparators therefore see the chain sum whenever a chain exists —
  the doc-83 nseg=1-always pathology is closed at the source.
- P5's second sentinel is emitted only when `mcs_range_comparator_chain` (no
  format change to the doc-80 line); under the hybrid source it is redundant
  when the chain was selected and diagnostic when the pf fallback ran
  (497311: `nseg=1 len=332.8cm` from pf, `nseg_chain=0` — an unbroken muon,
  correctly single-segment).
- Determinism (M4): two `setarch x86_64 -R` runs of `mcp2k` 66366 at the
  post-flip point — `mabc-pr.zip` + `pctree` byte-identical
  (`work-d84r1-det{1,2}-mcp2k`), `kine_mcs_*` sentinel values identical to
  every printed digit.

## R1.5 P2/P3 status — P3 delivers its case; P2 is NOT yet effective, held

`beeon2B` (P2 16° + P3 7.5 cm on top of the production point):

- **66366 (P3's case): full rescue.**  The 6.81 cm stub bridges; chain
  **126.2 → 300.6 cm** (nseg 3 → 4), range 307.0 → 692.2 MeV, ratio
  2.14 → 0.95, end_degree 2 → 1, so **best flips dqdx 656.0 → range 692.2**.
  MCS follows the same chain: npoints 213 → 503, `ke_MCS` 869.3 → 881.0 with
  ambiguity **0.80 → 0.35**.
- **313847 (P2's case): RESIDUAL.**  The chain still stops at nseg=2
  (98.5 cm) with the 162.4 cm continuation un-taken — even in a probe run at
  `relax_deg=25`.  The audit replay (§4) says the candidate passes every
  condition the knob codes (angle 10.99°, ratio 1.092, degree-2 junction), so
  the C++ walk is being stopped by something the replay does not model.  A
  knob-gated candidate probe (`long_muon_angle_relax: cand ...` DEBUG line)
  is in the code for the follow-up; **P2 stays HOLD**.  Diagnosed this
  round (`work-d84r1-p2diag-mcp1k`, probe log): at formation time the
  junction's actual candidates are a 5.6 cm arm at **120.2°** and a 63.0 cm
  arm at ratio **1.99** (correctly non-MIP) — **the 162.4 cm / 10.99°
  continuation of the audit replay does not exist in the graph when the
  chain forms; it is final-frame geometry** (same class as doc pr/120's
  scanner-start-override artifact).  Consequence: the audit's §4.2
  angle-rejection census (25 candidates, median 15.6°) is a final-frame
  measurement and cannot be used to tune P2.  Follow-up shape: a
  formation-time candidate census via this probe (env-gated arm over the
  case list), and possibly a post-refinement re-examine pass instead of a
  wider angle — the fill-once guard (§1, L4) prevents the chain from being
  re-formed after track refinement improves the geometry.

## R1.6 Bee packages for the owner scan

Three sets over the 31-event case list (owner refs first: idx 0 = `mcp2k`
66366, idx 1 = `mcp1k` 313847; then broken_chain/both by descending muon
length, range_zero movers > 30 cm, 4 worst delta-ray dQdx shifts):

| zip | arm | config |
|---|---|---|
| `bee/d84r1/d84r1-off.zip` | `work-d84r1-beeoff-*` | pre-flip production |
| `bee/d84r1/d84r1-on.zip` | `work-d84r1-beeonB-*` | the shipped flip (MCS+P4+P5+P1) |
| `bee/d84r1/d84r1-on2.zip` | `work-d84r1-beeon2B-*` | + P2 (16°) + P3 (7.5 cm) preview |

Energy IS visible in the scan: the PF tree bakes `get_kine_best()` into every
node label (`MultiAlgBlobClustering.cxx` `fill_bee_pf_tree`), so the P1
rescues read directly as label changes (497311: `mu- 508 MeV` → `mu- 766 MeV`
class).  MCS scalars are NOT in Bee (T_kine + sentinel only).
Index: `bee/d84r1/d84r1.index.txt`.  URLs:

- OFF: https://www.phy.bnl.gov/twister/bee/set/361f9e58-3883-4dcd-aba1-cd6ab57928d9/event/list/
- ON:  https://www.phy.bnl.gov/twister/bee/set/c1bbb7c6-2d1a-48b4-8e03-7fa075ee497b/event/list/
- ON2: https://www.phy.bnl.gov/twister/bee/set/ea9b4914-6152-4cc4-9da7-df104939e6e0/event/list/

## R1.9 P2/P3 flip (owner-directed, post Bee scan)

After scanning the Bee sets the owner directed the P2/P3 flip
(2026-08-28, same day).  Evidence backing it:

- **Mover census, 129 distinct events** (98-manifest + 31-evt case list):
  exactly **one** physics mover, `mcp2k` 66366 — the P3 rescue of R1.5.
  `work-d84r1-on2_98-*` (P2 16° + P3 7.5 on top of the shipped point) vs
  `work-d84r1-flipchk98-*`: **PASS 196/196 byte-identical** — both knobs are
  latent on the standard gate manifest.  On the 31-evt case list, on-vs-on2
  differs only on 66366 (`mabc-pr.zip` + calib; every other calib delta is
  the `vertex_scoreboard/dual_chain/off_ms` wall-clock field, not physics).
- **P2 is ON but measured latent everywhere probed** — the formation-time
  census (R1.5) still owns the follow-up; flipping it now means any future
  formation-time geometry that satisfies 10–16° + MIP + junction-veto will
  be taken without another cfg round.
- Compiled-config proof: post-flip config differs from R1.2's Step B by
  exactly two keys (`long_muon_angle_relax_long: true`,
  `long_muon_stub_bridge_len: 7.5`); `long_muon_angle_relax_deg` stays
  omitted (C++ 16.0).
- Flip equivalence on the 31-evt set: `work-d84r1-flip2-*` (post-flip cfg,
  no env) vs `work-d84r1-beeon2B-*` (env TLAs): **PASS 62/62** (16+46 archives) byte-identical, the 66366 mover included.
- Baseline note: because P2/P3 are latent on the 98-manifest,
  `work-d84r1-flipchk98-*` **remains** the valid production baseline for
  gate purposes; the 66366-class movers live only in case-list territory.

## R1.7 T_kine schema note

`mcs_enable=true` derives `mcs_output=true` (`sbnd/clus.jsonnet`), so
`tracking-pr.root` gains the five `kine_mcs_*` branches — an intended schema +
output change, stated here once for downstream consumers.  `calib-pr-evt*.json`
carries the same scalars via `dump_kine`, which is why every post-flip calib
dump differs even on events with no P1 mover (R1.2 divergence row).

## R1.8 Coordination note

This round ran concurrently with the pr/121-123 EM-clustering session in the
same working tree.  All toolkit commits here were staged hunk-selectively
against a snapshot of that session's in-flight diff; the shared binary carried
both sessions' default-OFF code through every gate above (both sides'
OFF-gates make that sound).  Runner env→TLA seats for the six new keys are in
`run_pr_chain_batch.sh` (`SBND_LONG_MUON_RANGE_FALLBACK`,
`SBND_LONG_MUON_ANGLE_RELAX[_DEG]`, `SBND_LONG_MUON_STUB_BRIDGE_LEN`,
`SBND_MCS_MUON_SOURCE`, `SBND_MCS_RANGE_COMPARATOR_CHAIN`).

# Round 2 (2026-08-28) -- members geometry, cathode bridge, formation-time census

## R2.0 Repro

```
# census (read-only, current-production arms; outputs docs/84_longmu/r2_census/)
python3 scripts/d84r2_census.py \
  --arms work-d84r1-flipchk98-mcp1k:mcp1k work-d84r1-flipchk98-mcp2k:mcp2k \
         work-d84r1-flipchk98-ncpi0:ncpi0 work-d84r1-flipchk98-nuecc48:nuecc48 \
         work-d84r1-flip2-mcp1k:mcp1k work-d84r1-flip2-mcp2k:mcp2k \
  --out docs/84_longmu/r2_census
# OFF gate (98-evt manifest, four samples), then per sample:
#   d84r2_arms.sh off1        (no env; production post-flip config)
python3 scripts/pr85_hash_gate.py work-d84r1-flipchk98-<s> work-d84r2-off1-<s>
# ON smoke (31-evt case list):
#   bee2_arms.sh on1 SBND_LONG_MUON_MEMBERS_GEOMETRY=1 SBND_LONG_MUON_CATHODE_BRIDGE=1
```

## R2.1 Census over the current production state (read-only)

`scripts/d84r2_census.py` over `work-d84r1-flipchk98-*` (98-evt manifest) +
`work-d84r1-flip2-*` (31-evt case list): 129 distinct events, 54 muon showers
with `flag_kinematics`.  Outputs in `docs/84_longmu/r2_census/`.

**Pop A — chain-vs-membership truncation** (`d84r2-members-gap.tsv`,
L_members − L_chain > 10 cm, L_chain from inverting the shipped muon
range→KE table on `kine_range`): **6/54 muon showers**, all in the case-list
events:

| evt | L_chain | L_members | arrow at | kine_range | kine_best |
|---|---|---|---|---|---|
| mcp1k 313847 | 98.5 | 260.9 | 90 cm (38%) | 248.3 | 547.5 (dqdx) |
| mcp1k 281595 | 130.0 | 351.0 | 124 cm | 315.1 | 750.1 (dqdx) |
| mcp2k 74784  | 29.9  | 291.3 | 29 cm  | 98.9  | 717.8 (dqdx) |
| mcp1k 278684 | 167.3 | 187.8 | 165 cm | 395.3 | 395.3 |
| mcp2k 393538 | 292.5 | 331.7 | 279 cm | 673.7 | 673.7 |
| mcp2k 499577 | 294.7 | 324.4 | 285 cm | 678.6 | 678.6 |

**Pop B — cathode-split muons** (`d84r2-cathode-pairs.tsv`; both facing ends
within 6 cm of x=0, far ends on opposite drift sides, gap < 25 cm): the three
scan events plus one new candidate.  Two census-corrections to the round-1
narrative, baked into the knob design:

- the halves do **not** share a PR `cluster_id` (53793: clus 37 vs 12) —
  imaging-level stitching does not survive into the PR graph, so the guards
  are purely geometric (no same-cluster requirement);
- near ends can jitter onto ONE side (77978's far half itself crosses x=0,
  near end at +2.17 beside the muon end at +4.77) — the opposite-side test
  is on the far ends.

| evt | muon end x | partner | partner x | gap | ∠gap | ∠tan |
|---|---|---|---|---|---|---|
| mcp2k 53793  | +4.71 | shower 528 MeV (240.5 cm, μ) | −4.53 | 13.9 | 14.5/11.8 | 6.5 |
| mcp2k 177536 | −2.50 | bare 98.1 cm | +3.29 | 9.4 | 6.0 | 9.0 |
| mcp2k 77978  | +4.77 | bare 23.5 cm (+9.8 cm Bragg stub typed 2212 beyond) | +2.17 | 6.5 | 12.6 | 14.9 |
| mcp2k 392901 | +2.10 | shower 106 MeV (**EM, pdg 11**) | −4.01 | 17.8 | 19.7 | 14.0 |

392901's partner is an EM shower — the one candidate the muon-typed-partner
guard excludes by design (absorbing EM showers across the seam is the failure
mode to avoid, not the target).

## R2.2 What shipped (knob table)

| config key | C++ default | SBND this round | what it does |
|---|---|---|---|
| `long_muon_members_geometry` | `false` | OFF (hold for scan) | `calculate_kinematics_long_muon`: muon-typed members outside the chain ADD to the range length and their vertices join the endpoint/end_degree search (round-1 P1's accumulators, partial-truncation form) |
| `long_muon_cathode_bridge` | `false` | OFF (hold for scan) | TCN post-reconcile pass: a \|13\| shower with a member end at the cathode absorbs its facing far half — `add_shower` merge for a muon-typed shower partner, BFS absorb + retype-to-13 for a bare segment chain — then `update_shower_maps` + kinematics recompute |
| `long_muon_cathode_bridge_x` | `0.0` cm | (default) | cathode plane |
| `long_muon_cathode_bridge_xcut` | `6.0` cm | (default) | end admission window (census ends at ±4.8) |
| `long_muon_cathode_bridge_gap` | `20.0` cm | (default) | max 3D gap (census 6.5–13.9) |
| `long_muon_cathode_bridge_angle` | `25.0` deg | (default) | continuation angle cap, gap vector AND partner tangent (census ≤ 19.7) |

Scope limits, deliberate: the bridge adds the far half to the SHOWER only —
`segments_in_long_muon` (tagger features, MCS chain) keeps its legacy
content, so `nusel` stays put and the change is confined to kine/PF output.
The two knobs compose: the bridged far half is outside the chain, so its
length reaches `kine_range` through `members_geometry`.  Ship/flip them
together.

## R2.3 Gate ledger

| gate | arms | result |
|---|---|---|
| knob-off byte-identical | `work-d84r1-flipchk98-*` vs `work-d84r2-off1-*` (98 evts, 4 samples) | **PASS 196/196** (28+34+38+96); per-event `nusel-evt*.tsv` **98/98** identical (the flipchk98-nuecc48 arm-level table holds only rerun evt 256587 — round-1 operator-stop artifact, not a diff) |
| doctests | `build/clus/wcdoctest-clus` | 2484/2484 (incl. six new default pins) |
| compiled config, knobs off | full-pipeline `wcsonnet` (reality=data + PR pipeline_names), HEAD vs edited jsonnet | byte-identical |
| compiled config, knobs on | same + 4 TLAs | exactly `long_muon_members_geometry`, `long_muon_cathode_bridge`, `_gap`, `_angle` appear in the TCN node; nothing else moves |
| ON smoke | `work-d84r2-on2-*` (31-evt case list, both knobs ON, pr/123 pass4 pinned OFF) vs `work-d84r1-flip2-*` | exactly **9 mover events**, all \|13\| showers; divergence confined to `mabc-pr.zip` on those 9; `pctree` 31/31 and `nusel` 31/31 byte-identical |
| determinism | `work-d84r2-det1/det2` (53793 merge + 177536 bare absorb, ASLR off) | PASS 4/4 byte-identical |

Freshness: `local/lib/libWireCellClus.so` 20:14 > last source edit 20:10.
Attribution note: a first ON arm (`on1`) overlapped the concurrent session's
pr/123 `shower_pass4_*` SBND flip (jsonnet mtime 20:24:36, mid-arm) and
picked up two of THEIR movers (55740, 75554); `on2` pins
`SBND_SHOWER_PASS4_PRUNE=0 SBND_SHOWER_PASS4_TRACK_GUARD_LEN=0` and is the
arm quoted everywhere here.  The OFF arm compiled all 196 configs before
that flip (verified: zero carry pass4 keys).

## R2.4 ON smoke — all five scan events land, nothing else moves

The 9 movers (on2 vs flip2), quoted from `kine_long_muon:` / bridge lines:

**members_geometry (6):**
- 313847: L 98.5→260.9 cm, range 248.3→602.2, best 547.5 (dqdx) → **602.2 (range)**; end_degree 2→1 — the Bee arrow lands at the true end (owner finding #1).
- 281595: L 130.0→351.0 cm, best 750.1 → **808.5 (range)** (owner finding #2).
- 74784: L 29.9→291.3 cm, range 98.9→670.8, best 717.8 (dqdx) → **670.8 (range)** — census discovery, worst truncation of all.
- 278684: 167.3→187.8 cm, 395.3→439.9.  393538: 292.5→331.7 cm, 673.7→757.9.  499577: 294.7→324.4 cm, 678.6→746.9.

**cathode_bridge (3):**
- 53793 (shower partner): `merge shower sid=0 (mu_len=154.1cm) <- sid=2 (mu_len=240.9cm) gap=13.9cm ends x=(4.7,-4.5)cm` — single 395.0 cm muon, best 367.0 → **912.9 MeV** (dqdx/range ratio 1.00); the 527.8 MeV far-half shower absorbed (owner finding #3).
- 177536 (bare far half): `absorb bare chain into sid=1 nseg=2 len=122.1cm gap=9.4cm` — L 294.8→416.9 cm, best 679.0 → **963.9 (range)** (owner finding #4).
- 77978 (bare far half + Bragg stub): `absorb bare chain into sid=0 nseg=2 len=33.4cm gap=6.5cm` — the 23.5 cm muon piece AND the 9.8 cm stub typed 2212 (Bragg rise) beyond it, retyped 13; best 337.5 → **409.4 (range)** (owner finding #5).

Regression checks: 497311 keeps its round-1 P1 rescue (fallback=1, 766.3),
66366 keeps its P3 chain (300.6 cm, 692.2), 392901 unchanged BY DESIGN (its
census partner is a 106 MeV EM shower — excluded by the muon-typed-partner
guard; flagged in the Bee index for owner opinion).

## R2.5 P2 formation-time census — the audit question, closed

The round-1 probe (`long_muon_angle_relax: cand`) is knob-gated and P2 is
production-ON, so the flipchk98 + flip2 debug logs already carry the census:
**105 distinct formation-time candidates in 24 events**
(`r2_census/d84r2-formation-cands.tsv`).  Angle distribution: median 60.4°,
p25 35.3° — the rejected population is genuinely non-collinear; only 20
candidates sit below 30°, and only 4 pass the ratio+length preconditions
(`considered=1`).

The sharpest finding closes 281595: its 220.8 cm continuation IS present at
formation time at 15.6° / ratio 1.019 / considered=1 — inside the 16° cap —
but the P2 block's own junction veto kills it: the third arm at vertex
20003 is the 13.9 cm stub (> 10 cm, track-like), which reads as "genuine
hadronic vertex".  P2-as-designed can never fire where a >10 cm competing
arm exists, which is exactly the greedy-stub topology it was meant to heal.
`members_geometry` recovers these events without touching the walk, so no
P2 re-tuning is proposed; the veto interplay is recorded for any future
walk-level fix (re-examine after refinement is still blocked by the
fill-once guard).

## R2.6 Bee package (held for owner)

`bee/d84r2/d84r2-on.zip` (31 events, round-1 index order) is built from the
on2 arm and annotated in `bee/d84r2/d84r2.index.txt`; **not uploaded** —
outward-facing, owner call.  BEFORE for the scan is the already-uploaded
round-1 ON2 set `ea9b4914-6152-4cc4-9da7-df104939e6e0` (byte-equivalent to
current production, flip-equivalence PASS 62/62).  Both knobs are SBND OFF
until that scan.

## R2.7 Coordination note

Concurrent with the pr/123 session in the same tree.  Its uncommitted
pass4 hunks (TCN.{h,cxx}, NPB.h, NSC.cxx, PRShower.{h,cxx}, doctest,
jsonnet) were recorded before editing and excluded from this round's commits
by tag-selective staging; its 20:24:36 jsonnet flip mid-arm is handled in
R2.3.  New runner env→TLA seats: `SBND_LONG_MUON_MEMBERS_GEOMETRY`,
`SBND_LONG_MUON_CATHODE_BRIDGE[_GAP|_ANGLE]`.
