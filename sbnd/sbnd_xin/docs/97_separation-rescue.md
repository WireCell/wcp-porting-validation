# Doc 97 — rescuing the two separation cases: `sep_track_recarve` and `sep_fv_point`

**Status: two SBND knobs implemented and validated on the full 3134-event
sample. `sep_fv_point` is SBND PRODUCTION as of `ref/prod-2026-09-04`
(owner flip 2026-09-02, §9). `sep_track_recarve` stays default OFF and
byte-identical when off.**

The owner's instruction for this round: *"we do not need to worry about the STM
case, but focus on whether we can rescue the two separation cases, at least one
of them. We want to test against the 3000 numu events that we have (also the 48
nueCC, and 19 NCpi0) to ensure there is no regression."*

The two cases are doc 96's, from the doc-95 Bee scan set (indices 8 and 11):
`272-2-30 separation not working???` and `105-23-21 separation not working???`.
Doc 96 established that neither cluster was ever separated — the two tracks
arrive from imaging already blob-connected — and measured, at the *cluster*
level, that two default-OFF mechanisms could reach them.

**Both are rescued, and both by the same knob.** The answer to "at least one of
them" is two.

| the owner's case | `sep_track_recarve` | `sep_fv_point` |
|---|---|---|
| `272-2-30 separation not working???` | **rescued** — TGM → nu-candidate | **rescued** |
| `105-23-21 separation not working???` | no fire | **rescued** — TGM → nu-candidate |
| events changed at the clustering stage, of 3067 | 103 (3.4%) | 1277 (41.6%) |
| in-beam neutrino candidates, of 1599 | **+3, −0** | +3, **−2** |
| bundles that gain a cosmic tag | **0** | +3 STM |
| shipped-fix sentinels (31) | 29 PASS / **2 FAIL** | 30 PASS / **1 FAIL** |

Both break a shipped, owner-approved fix, so both were **reported rather than
flipped** when this doc was first written. The owner then scanned the three-arm
Bee set of §6 and adjudicated `sep_fv_point`'s three apparently-adverse flips
(idx 5, 6, 7) as **improvements**, which turns its ledger positive and its one
sentinel failure into an expected consequence. `sep_fv_point` is production as
of `ref/prod-2026-09-04`; `sep_track_recarve` is not. §9 records the flip and
its verification; §7 is left as written, as the reasoning the owner ruled on.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin   # symlink into wcp-porting-img/sbnd/sbnd_xin

# A. is the on-disk stage-A half of production still reproducible today? (sec 2)
/home/xqian/tmp/d97/idgated_run.sh                  # -> work-d97idgd-*, ~10 min
python3 scripts/d97_ql_gate.py d97idgd grp0825      # 367 events / 1468 products

# B. the two knobs, byte-identity when off (sec 3)
python3 scripts/cfg/prod_cfg_gate.py                # 21/21 vs ref/prod-2026-09-03
./build/clus/wcdoctest-clus -tc="clus sep fv inset*"   # the 3 new cases, 59 assertions

# C. the MC control: the owner's two symptom events (sec 4)
./scripts/d97_dbg25_arm.sh both a b                 # sep_track_recarve OFF/ON
./scripts/d97_dbg25_fv.sh a b                       # sep_fv_point
python3 scripts/d97_flip_report.py d97onpr  d97offpr dbg25a dbg25b
python3 scripts/d97_flip_report.py d97fvpr3 d97offpr dbg25a dbg25b

# D. which of sep_fv_point's four values does the work (sec 5)
python3 scripts/d97_fv_decomp.py 21 30
./scripts/d97_fv_decomp_run.sh 21 30

# E. the full-sample arms and the regression census (sec 6)
./scripts/d97_off_arms.sh all                       # the BASELINE, ~1.5 h
./scripts/d97_on_arms.sh  all                       # sep_track_recarve
./scripts/d97_fv_arms.sh  all                       # sep_fv_point
python3 scripts/d97_ql_gate.py       d97on    d97off2      # stage-A blast radius
python3 scripts/d97_flip_report.py   d97onpr  d97off2pr    # verdicts
python3 scripts/d97_physics_census.py d97onpr d97off2pr    # Enu / showers / vertex
python3 scripts/d97_stagea_sets.py d97off2 d97on d97fv grp0825   # change sets + overlap
python3 scripts/d97_sentinels.py --arms 'work-*-d97onpr'   # shipped-fix sentinels

# F. the owner's Bee A/B (sec 6) -- three arms, ONE event order
./scripts/d97_bee.sh 30 21 290654 318703 95911 105074 162363 392901 401780 \
    94392 53793 256587 360535 54095 390182 316729 280159 101828 98470 \
    175871 289508 49742
./upload-to-bee.sh bee/d97/d97-{off,on,fv}.zip     # one at a time

# G. the flip and its verification (sec 9) -- AFTER the owner's verdict
python3 scripts/cfg/prod_cfg_gate.py --keep /home/xqian/tmp/d97/cfgflip   # drift
git -C ../.. archive HEAD cfg | tar -x -C /home/xqian/tmp/d97/cfgbase     # pre-flip tree
python3 scripts/cfg/prod_cfg_gate.py --cfg /home/xqian/tmp/d97/cfgbase/cfg \
    --keep /home/xqian/tmp/d97/cfgoff                                     # must PASS
python3 scripts/d97_flip_drift.py /home/xqian/tmp/d97/cfgoff /home/xqian/tmp/d97/cfgflip
cp -a ref/prod-2026-09-03 ref/prod-2026-09-04                             # never --refresh in place
python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-04 --refresh
/home/xqian/tmp/d97/prodchk.sh                    # production default, NO flag
python3 scripts/d97_ql_gate.py d97prodchk d97fv   # must be byte-identical
```

Saved outputs under `docs/97_sep/`: `97-idgate.txt`, `97-idgate-final.txt`,
`97-dbg25-control.txt`, `97-dbg25-fv-control.txt`, `97-fv-decomposition.txt`,
`97-stagea-sets.txt`, `97-flip-3067.txt`, `97-physics-3067.txt`,
`97-sentinels.txt`, `97-flip-drift.txt`, `97-prodchk-gate.txt`; Bee sidecars
under `bee/d97/`; the new operating point in `ref/prod-2026-09-04/`.

Operating point `ref/prod-2026-09-03`, `reality=data` for the four validation
samples and `sim` for the doc-95 MC debug set. Binary pinned to
`~/tmp/d97b-libsnap` (md5-equal to `local/lib` at launch, 229 of 229 shared
objects) for every arm built after the C++ change, and to `~/tmp/doc94r3b-libsnap`
for the ones before it; PR config tree pinned to `~/tmp/dbg25-cfgsnap`.

---

## 1. What shipped

Two knobs, each an SBND jsonnet argument defaulting **false**, each suppressing
its keys when off so the compiled configuration is byte-identical to
`ref/prod-2026-09-03`:

| knob | what it turns on | C++ |
|---|---|---|
| `sep_track_recarve` | `ClusteringSeparate.track_recarve` — the post-separation k=2 3D-line self-split of a member holding two long crossing track arms | **existing**, already production on PDHD and PDVD |
| `sep_fv_point` | `fv_inset_yz` 15 cm + `far_point_x_cut` 14 cm + `far_point_mid_dis` 60 cm + `dec1_guard_main_angle` 45° | `fv_inset_yz` is **new** (this round); the other three existed |

(§9: `sep_fv_point`'s SBND default became **true** after the owner's
2026-09-02 verdict. Everything in §1–§8 describes the knob as it was
validated — default OFF — and the tables below are the proof that the OFF path
was byte-identical, which is what made the flip a one-line change with a named
drift.)

`fv_inset_yz` is the only new C++ in the round. It exists because the PDHD/PDVD
recipe for this blindness (`clus/docs/clustering-separate-fv.md`) insets
`FV_ymin/ymax/zmin/zmax` in the **shared** `DetectorVolumes` metadata, which
`select_scope_fv` also hands to `clustering_neutrino` and the containment
taggers — doc 96 §9 flagged that coupling as the reason an inset could not ship
as a config edit. `fv_inset_yz` applies the same inset to the single `ScopeFV`
that `clustering_separate()` builds, and nothing else reads it.

```
    const ScopeFV fv = inset_scope_fv(select_scope_fv(dv, drift_side_fv_x), fv_inset_yz);
```

`inset_scope_fv()` (`clus/inc/WireCellClus/ClusteringFuncs.h`,
`clus/src/clustering_separate.cxx:134`) returns its input unchanged for
`inset_yz <= 0`, so OFF is bit-identical as a property of the code, not only of
the gates.

### Files

| file | change |
|---|---|
| `clus/src/clustering_separate.cxx` | `inset_scope_fv()`; `fv_inset_yz` threaded through `clustering_separate()` and read by `ClusteringSeparate::configure` (default 0) |
| `clus/inc/WireCellClus/ClusteringFuncs.h` | `inset_scope_fv()` declaration + why it exists |
| `clus/test/doctest_sep_fv_inset.cxx` | **new**, 3 test cases / 59 assertions |
| `cfg/pgrapher/common/clus.jsonnet` | `separate(fv_inset_yz=null)`, key-suppressed |
| `cfg/pgrapher/experiment/sbnd/clus.jsonnet` | `sep_track_recarve`, `sep_fv_point` threaded to the one `cm.separate()` call |
| `cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet` | the two TLAs, default false |
| `sbnd_xin/run_ql_evt.sh` | `-sep-recarve`, `-sep-fv-point` (and the `-no-` forms) |

No other detector's configuration is touched: the compiled-config gate below
covers all 21 consumer artifacts, uboone and the eleven pdhd/pdvd jobs included.

### Byte-identity when off

| proof | result |
|---|---|
| compiled config, both knobs off, vs `ref/prod-2026-09-03` | **PASS 21/21 artifacts** — run twice, before and after the C++ change |
| compiled config, `sep_track_recarve=true` | `track_recarve: true` on exactly **2** `ClusteringSeparate` nodes (2 anodes × 1 face; the all-APA chain has no `Separate`) |
| `./build/clus/wcdoctest-clus` | **all pass**; this change adds **3 cases / 59 assertions**, confirmed by `-tc="clus sep fv inset*"`. The suite total was 237 → 240 when the change landed and reads 248 / 2733 now, a peer's PDVD commits (`fb0579c5`, `784dc837`) having landed since — quote the delta, not the total. |
| stage-A output, both knobs off, 367 events | byte-identical, see §2 |

The new doctest carries the causal negative control the quality bar asks for:
it takes the two **measured** wall exits of 105-23-21 (3.0 cm short of the
y-min wall, 1.4 cm short of z-max), shows that the production FV counts
**zero** surface contacts on them, that a 15 cm inset counts two, and then that
feeding the same points through `inset_scope_fv(fv, 0.0)` returns to zero. The
effect disappears when exactly the quantity the knob supplies is removed.

---

## 2. Method, and the baseline that had to be rebuilt

Both knobs live in `ClusteringSeparate`, which runs inside the **Q/L job**, not
the PR tagger tail. Every A/B harness in this tree (`doc94r3_arm.sh`,
`stm_campaign/run_round.sh`, …) re-runs only the PR tail on top of a frozen Q/L
root, because every knob those rounds touched was a PR knob. This round needs
the Q/L stage re-run and the PR chain rebuilt on top of it, which is what
`scripts/d97_ql_arm.sh` does.

The group-mode stage-A driver could not be reused: doc 89's retire round pruned
the `g<K>/` group scratch out of the four `work-<s>-grp0825` roots, so the group
imaging checkpoint is gone. The **per-event** imaging survives
(`evt<ID>/icluster-apa{0,1}-{active,masked}.npz`), and doc 81 §7 gated
group-mode products byte-identical to per-event ones (24536/24536), so a
per-event Q/L re-run off those npz is the same reconstruction.

### 2.1 The intended baseline, and why it is not safe

The plan was to gate `work-<s>-grp0825` (2026-08-25, the on-disk stage-A half of
production) against today's binary and configuration, and — if it reproduced —
use it, and the PR arm `work-<s>-r3entry` built on it, as the OFF side.

It reproduced, three separate times, on a 367-event manifest
(`gate308-{mcp1k,ncpi0,nuecc48}` plus 59 mcp2k events sampled every 34th):

| arm | binary | result |
|---|---|---|
| `work-d97idg-{ncpi0,mcp1k,mcp2k}` + `work-d97idgb-nuecc48` | pre-change | **367/367 events, 1468/1468 products byte-identical** |
| `work-d97idgc-*` | first `fv_inset_yz` build | **367/367** |
| `work-d97idgd-*` | the shipped build | **367/367** |

**That manifest was too small.** A four-event spot check of mcp2k events that
the census had flagged found **two** — `18255-x-74190` and `18255-x-53793` —
where today's knob-off run does *not* reproduce `grp0825`:

```
work-d97chk2-mcp2k vs work-mcp2k-grp0825:  4 events, 2 identical, 2 differing
  DIFF mcp2k 53793 pctree-evt53793.tar.gz
  DIFF mcp2k 74190 mabc-all-apa.zip  +  pctree-evt74190.tar.gz
```

Two knob-off runs **today** reproduce each other exactly
(`work-d97chk3-mcp2k` vs `work-d97chk2-mcp2k`: 4/4 events, 16/16 products), so
this is **not run-to-run nondeterminism**. It is an epoch drift between
2026-08-25 and now — the master merge (`6100d8f0`, 397 commits) and the doc-88
recombination fix both landed in between, and neither was ever gated at the Q/L
stage: doc 89's `prod0901b` re-ran only the PR chain over `grp0825`.

Consequences, and they are the reason this section exists:

* **`grp0825` is not a safe OFF baseline** for attributing a knob's effect, and
  neither is `r3entry`, which sits on it. A fresh knob-off arm
  (`work-<s>-d97off2` / `-d97off2pr`) was built over the whole sample and is
  what §6 compares against.
* The drift is small and the gate manifest is what missed it: over all 3067
  events the ON arm differs from `grp0825` on **107** events while the knob
  fires on **103**, so the drift set is **4 events (0.13%)** — but two of the
  four are sentinel events for shipped, owner-approved fixes, so its visible
  cost was out of all proportion to its size (§8).
* A 367-event manifest cannot see a 0.13% effect. That is not a subtle
  statistical point: 0.13% of 367 is half an event.

### 2.2 The instruments

* **`scripts/d97_ql_gate.py`** — member-content hash of every `ql_evt<ID>/`
  product (the three Bee zips and the pctree tarball) between two stage-A
  roots. Never `md5sum`/`cmp` on the archive (M2).
* **`scripts/d97_flip_report.py`** — per-bundle verdict A/B. It cannot key on
  `main_id` the way `doc94r2_flip_report.py` does: this is a *clustering* knob,
  and `cluster_id_order: 'tree'` renumbers every cluster after a split, so on a
  firing event all 23 of 272-2-30's bundles change `main_id` while all 23 keep
  their flash. The key is the flash, with one correction found by an assertion:
  `(flash_apa, flash_gid)` is **not** unique — nueCC48 `18255-1-10550` matches a
  131.8 cm nu-candidate *and* a 374.1 cm TGM to flash `1/1000002` — so bundles
  sharing a flash are paired by nearest `npts_main`. Without that, one bundle
  per collision is silently dropped and every count is understated.
* **`scripts/d97_physics_census.py`** — `kine_reco_Enu`, `kine_pio_mass`,
  main-vertex position, shower/segment/vertex counts and summed shower energy
  from the calib dumps. Deliberately not a whole-file diff:
  `vertex_scoreboard.dual_chain.off_ms` is a wall-clock timer and makes two
  byte-identical reconstructions "differ" every time.
* **`scripts/pr127_sentinels.py`** — the shipped-fix sentinel suite. Run it with
  **one** arm glob at a time: `find_arm` takes the first match, so passing
  several globs silently reports only the first.
* Two blind spots are covered on purpose. A verdict table counts labels, so it
  cannot see an in-time main that got *longer* — a neutrino fused into a cosmic
  — when the PR chain's unmerge hides it from the calib dump too; the flip
  report therefore prints the direction of every in-beam size change. And the
  physics census reads the calib dump, which only exists where a candidate was
  selected, so it covers 1436 of 3067 events and says nothing about the rest;
  the flip report covers all of them.

---

## 3. The MC control: both symptom events are rescued

The two symptom events live in the doc-95 25-event MC debug sample, not in the
data samples. The control arms re-run that sample's Q/L stage with each knob and
rebuild the PR chain; the knob-off arm reproduces `work-dbg25{a,b}-ql`
**byte-identically** (25 events, 100 products), which is what makes the rest
attributable.

| | `sep_track_recarve` | `sep_fv_point` |
|---|---|---|
| 272-2-30 (`dbg25a` evt 30) | **TGM → nu-candidate** | **TGM → nu-candidate** |
| 105-23-21 (`dbg25a` evt 21) | no fire, unchanged | **TGM → nu-candidate** |
| 105-23-5 (`dbg25a` evt 5) | unchanged (out of scope, doc 96 §8.4) | unchanged |
| in-beam bundles moved, 25 events | 1 | 2 |
| stage-A events changed | 2 of 25 (exactly the fires) | 2 of 25 |

**272-2-30.** The in-beam bundle's main goes from 20516 points / 422.6 cm,
`tgm=1`, **TGM**, to 9887 points / 342.6 cm, `fc=1`, **nu-candidate** — and a
new bundle appears at `t = −563.3 µs` holding 10621 points / 411.6 cm. That is
doc 96 §8.2's prediction confirmed end to end: the through-going cosmic leaves
the beam bundle, flash-matches separately, and the in-beam object becomes the
343 cm track alone. The `track_recarve` line is identical to doc 96's:

```
Separate track_recarve: len 486.724 cm split into arms 10608 / 9844 npoints,
    cross frac 0.56278 / 0.126344, resid 2.82947 / 6.78415 cm
```

**105-23-21.** 4760 points / 439.1 cm, **TGM** → 3852 points / 333.0 cm,
**nu-candidate**. Doc 96 §8.3 reached this event only by insetting the *shared*
`DetectorVolumes` FV, so it could not say whether the rescue was the separation
or the containment tagger seeing a smaller volume. `sep_fv_point` insets only
inside `clustering_separate`, and the flip is the same, so **the rescue is the
separation**. Doc 96 §8.3's other reservation — that 908 of the points land in a
different 28389-point host — is still true and is why the owner should look at
this one (§7).

The second `track_recarve` fire in the 25-event sample, `304-6-28`, is benign:
the in-beam main goes 109.0 → 106.5 cm and every verdict, tag and PF quantity is
unchanged.

---

## 4. Which of `sep_fv_point`'s four values does the work

The knob moves four numbers at once because doc 96 §8.3 measured that neither
half reached 105-23-21 alone — but it measured that with the inset applied to
the *shared* FV. The separation-scoped decomposition had never been measured,
and the owner is being asked to flip this. Each arm below patches the event's
own compiled Q/L config from the byte-identical OFF arm, so the only difference
is the listed keys; the numbers are the final `clustering-global` cluster sizes.

| arm | 105-23-21 (main 4760 pts) | 272-2-30 (main 20516 pts) |
|---|---|---|
| off | 4760 | 20516 |
| `fv_inset_yz` only | 4760 | **10433 + 9846** |
| `far_point` pair only | 4760 | 20516 |
| `dec1_guard_main_angle` only | 4760 | **10433 + 9827** |
| inset + far_point | **3852 (+908 into the 28389-pt host)** | 10433 + 9846 |
| all four | **3852** | 10433 + 9827 |

* `fv_inset_yz` is load-bearing for both events — the only value that is.
* The `far_point` pair is required **only** by 105-23-21, and only together
  with the inset.
* `dec1_guard_main_angle` is required by **neither**. It reaches 272-2-30 by an
  independent route and shifts that split by 19 points; it is carried for
  parity with the PDHD/PDVD operating point.

A three-value variant (drop `dec1_guard_main_angle`) would rescue both symptom
events. It has **not** been validated on the 3134-event sample; the four-value
point has, and that is what the flip would adopt.

---

## 5. The population: 3067 data events, both knobs

Baseline is `work-<s>-d97off2pr`, the fresh knob-off arm of §2.1 — **not**
`r3entry`. Verdicts come from `nusel-evt<ID>.tsv` keyed on the flash;
energies from the calib dumps; the sentinel suite is the shipped-fix tripwire.

### 5.1 Blast radius at the clustering stage

| | events whose all-APA Q/L output changes | of 3067 |
|---|---|---|
| `sep_track_recarve` | **103** — exactly the events where the recarve line fires | 3.36% |
| `sep_fv_point` | **1277** | 41.6% |
| (the 2026-08-25 stored stage A, for scale) | 3 | 0.10% |

The two are **not nested**: 53 events are changed by both, **50 by
`sep_track_recarve` alone** and 1224 by `sep_fv_point` alone. Neither knob
subsumes the other, and a **both-on arm has not been measured** — that is the
one arm this round is missing.

41.6% is the number to be suspicious of, so here is what it consists of. Of
those 1277 events, **10** show any in-beam movement at all: 6 verdict flips
(§5.2), 3 events where an in-beam bundle appears or disappears, and **1** whose
in-beam main changes length by more than 1 cm. Of the 40 in-beam mains that
change *at all*, **38 keep their length to within 0.05 cm** and move only their
point count. So the 41.6% is overwhelmingly **re-partition at unchanged
extent** — points moving between neighbouring clusters along the same track —
not splitting or merging. `sep_track_recarve`'s 103 events are the opposite
shape: 13 in-beam mains change size and 6 of them by more than 1 cm, because
every one of its changes *is* a split.

### 5.2 Verdicts

| | `sep_track_recarve` | `sep_fv_point` |
|---|---|---|
| bundles compared | 34826 | 34799 |
| field flips (in-beam) | **1 (1)** | **8 (6)** |
| in-beam nu-candidates | 1599 → **1602 (+3)** | 1599 → **1597 (−2)** |
| in-beam TGM | 509 → 508 (−1) | 509 → 508 (−1) |
| in-beam STM | 414 → **414 (0)** | 414 → **417 (+3)** |
| in-beam LM | 54 → 54 | 54 → 54 |
| in-beam mains that GREW > 1 cm | 2 (max +22 cm) | **0** |
| in-beam mains that SHRANK > 1 cm | 4 (max −61 cm) | 1 |

`sep_track_recarve` **gains three in-beam neutrino candidates and loses none,
and no bundle anywhere gains a cosmic tag.** One is a verdict flip
(mcp2k 290654, TGM → nu-candidate); two are new in-beam bundles freed by a
split onto their own beam flash (mcp1k 401780, mcp2k 94392), so those events now
carry two in-beam candidates instead of one.

`sep_fv_point` gains three (290654, 318703, and 95911 where a `no-bundle` event
acquires a 15 cm candidate) and **loses two** — mcp2k 105074 becomes STM at
unchanged length and mcp2k 392901 becomes TGM with its main *growing* 384.5 →
482.1 cm — plus a third STM tag from 162363 (TGM → STM). Net −2 candidates,
+3 STM.

### 5.3 Reconstructed physics, where a candidate exists

The calib dump exists only where `TaggerCheckNeutrino` selected something, so
this covers 1436 of 3067 events; the verdict table above covers all of them.

| | `sep_track_recarve` | `sep_fv_point` |
|---|---|---|
| events with any moved quantity | **18** of 1436 (1.3%) | **33** of 1436 (2.3%) |
| largest ΔEnu | −779 MeV (mcp2k 53793) | **−1807 MeV** (mcp2k 392901) |
| next largest | −317 (94392), −290 (390182), +247 (72759) | +1028 (280159), +1022 (101828), −265 (98470) |
| largest main-vertex move | 42.9 cm (mcp1k 401780) | **249.8 cm** (392901) |

### 5.4 The shipped-fix sentinels — both knobs break one

`scripts/d97_sentinels.py`, one arm glob at a time.

| arm | result |
|---|---|
| `work-*-r3entry` (production) | **31 PASS / 0 FAIL** |
| `work-*-d97off2pr` (this round's baseline) | **31 PASS / 0 FAIL** |
| `work-*-d97onpr` (`sep_track_recarve`) | **29 PASS / 2 FAIL** |
| `work-*-d97fvpr2` (`sep_fv_point`) | **30 PASS / 1 FAIL** |

The baseline passing all 31 is what makes these attributable: the failures are
the knobs', not the epoch drift's.

* `sep_track_recarve` breaks **mcp2k 94392** (pr/129 — `Enu=822.2`, want
  1136–1149) and **mcp2k 53793** (doc 84 r2 — the cathode-bridged 808 MeV muon
  comes back as 237 + 32 MeV, i.e. the bridge's two halves are separated again).
  Both are events where the recarve fires.
* `sep_fv_point` breaks **mcp2k 105074** (pr/128 class B) so hard that the event
  has no PF tree at all — which is why `scripts/d97_sentinels.py` exists as a
  fork: the production `pr127_sentinels.py` raises
  `KeyError: "There is no item named 'data/0/0-mc.json'"` and reports **nothing**
  for that arm, hiding the other 30 sentinels behind a crash.

Under CLAUDE.md §5.7 these are reported, not tuned away. Each is a *shipped,
owner-approved* fix losing its event.

---

## 6. What the owner is asked to judge

Three Bee sets, one event order, 22 events (`bee/d97/d97.index.txt` carries the
per-row evidence):

| arm | link |
|---|---|
| OFF (baseline) | https://www.phy.bnl.gov/twister/bee/set/4514b121-ce67-4647-ba60-20424cdecfcf/event/list/ |
| `sep_track_recarve` | https://www.phy.bnl.gov/twister/bee/set/ce2b4924-a466-4149-8af1-274ea28c3b3c/event/list/ |
| `sep_fv_point` | https://www.phy.bnl.gov/twister/bee/set/8bc27e44-d4c7-48ed-b252-100b0585f2c5/event/list/ |

idx 0–1 are the two symptom events, idx 2–7 every in-beam verdict flip, idx
8–18 the largest energy/size movers, idx 19–21 controls neither knob touches by
a byte. The questions that actually need an eye:

1. **idx 0, 1** — are the two splits right? (I believe so; they are the whole
   point of the round.)
2. **idx 5, 7** — `sep_fv_point`'s two candidate losses. 392901 in particular:
   the main *grows* 384.5 → 482.1 cm and Enu falls 1807 MeV. Is the OFF-side
   candidate real?
3. **idx 9, 10** — `sep_track_recarve`'s two sentinel breaks. 53793 is the
   cleaner question: does the recarve cut a genuine cathode-crossing muon in
   two?
4. **idx 8, 12** — the two events where a knob makes an in-beam main *bigger*
   or splits off a second in-beam candidate. Is one event with two in-beam
   nu-candidates an improvement or a double count?

---

## 7. Recommendation

**Neither knob is flipped.** Both are ready to be, and the choice is a physics
judgement I should not make alone. What the measurements say:

`sep_track_recarve` is the **narrower and, on verdicts, the strictly
non-negative** one: it touches 3.4% of events, gains three in-beam neutrino
candidates, loses none, and adds no cosmic tag anywhere. Its cost is two
shipped-fix sentinels (§5.4) and a −779 MeV energy movement on one of them. It
rescues **272-2-30 only**.

`sep_fv_point` is the only thing that rescues **both** symptom events, and it
is the answer to doc 96 §9's open question — the 105-23-21 rescue is the
*separation*, not the containment tagger seeing a smaller volume, because the
inset is scoped to the pass. Its cost is a 41.6% blast radius, a net loss of two
in-beam candidates, three new STM tags, one sentinel, and the round's single
worst movement (idx 7).

So my recommendation, in order:

1. **Scan the Bee sets (§6) first.** Four of the questions there decide the
   flip; none of them can be settled from a census. In particular, if idx 5 and
   idx 7 turn out to be OFF-side over-clustering that *deserved* to be tagged,
   `sep_fv_point`'s ledger goes from −2/+3 to positive and it becomes the clear
   choice.
2. **If only one flips, flip `sep_track_recarve`** — but only after idx 9 and
   idx 10 are adjudicated, because a knob that breaks two shipped fixes should
   not ship on a "+3 candidates" ledger alone.
3. **Measure the both-on arm** before adopting either as final. §5.1 shows 50
   events that only `sep_track_recarve` reaches and 1224 that only
   `sep_fv_point` reaches; the two together have never been run, and neither
   census predicts the union.
4. **A narrower `sep_fv_point` is available and unmeasured.** §4 shows
   `dec1_guard_main_angle` is required by neither symptom event. Dropping it
   gives a three-value variant that still rescues both; whether it also shrinks
   the 41.6% blast radius is a one-arm question.

---

## 8. Reported, not fixed

* **The stored stage A of production has drifted.** `work-<s>-grp0825`
  (2026-08-25) is the only on-disk carrier of the imaging + Q/L half of
  production (doc 89), and `work-<s>-r3entry` — today's production PR baseline —
  is built on it. Re-running the Q/L stage today with the same configuration
  reproduces it on **3064 of 3067 events** and differs on three: `mcp1k 390182`,
  `mcp2k 74190`, `mcp2k 99438` (plus `mcp2k 53793` in the pctree only). Two runs
  today agree with each other exactly, so this is an epoch difference, not
  nondeterminism — the master merge `6100d8f0` and the doc-88 recombination fix
  both landed after 2026-08-25 and neither was gated at the Q/L stage. It is
  0.1% and it cost nothing here once a fresh baseline was built, but it means
  **any future round that A/Bs a clustering knob against `grp0825` will
  mis-attribute up to three events**, and one of the three (74190) presents as a
  clean nu-candidate → TGM flip.
* **A 367-event gate manifest cannot see it.** `gate308` plus 59 mcp2k events
  passed 367/367 three separate times while the drift was live. If the standing
  byte-identity manifest is meant to protect the Q/L stage as well as the PR
  stage, it needs to be a good deal larger, or event-sampled per round.
* **`pr127_sentinels.py` crashes instead of failing when a sentinel event loses
  its PF tree** (`KeyError: 'data/0/0-mc.json'`), which suppresses the verdicts
  of every other sentinel in that arm. `scripts/d97_sentinels.py` is a fork with
  the one-line guard; the production file is untouched (M10). Worth folding in.
* **`(flash_apa, flash_gid)` is not a unique bundle key.** nueCC48
  `18255-1-10550` matches a 131.8 cm nu-candidate and a 374.1 cm TGM to the same
  flash `1/1000002`. Any future flash-keyed A/B needs the size pairing this
  round's `d97_flip_report.py` does, or it will silently drop one bundle per
  collision.

---

## 9. The flip — `sep_fv_point` is SBND production

**Owner, 2026-09-02, after scanning the §6 Bee sets:** *"These are good idx 5-7
are all improvements. We can set this new running as default on for SBND
production."*

That verdict is what changes the arithmetic. idx 5 (mcp2k 105074,
nu-candidate → STM), idx 6 (162363, TGM → STM) and idx 7 (392901,
nu-candidate → TGM with the main growing 384.5 → 482.1 cm) were §5.2's entire
cost. Adjudicated as improvements, `sep_fv_point`'s in-beam ledger reads **six
verdict flips, all of them right**, and its one sentinel failure (§5.4) is a
consequence of one of them rather than a regression — mcp2k 105074 is *supposed*
to lose its candidate.

### 9.1 What was flipped, and what was not

`sep_fv_point` defaults **true** in `cfg/pgrapher/experiment/sbnd/clus.jsonnet`
(both `clus_per_face` and `per_apa`) and in
`wct-clus-matching-perevt.jsonnet`. Setting the default on `clus_per_face` is
what carries it into the **LArSoft** entry point, which reaches
`clus_per_face` through `per_volume()` and takes its defaults — the same
mechanism `sep_vertex_veto` uses.

**`sep_track_recarve` was NOT flipped.** The owner's sentence names the arm he
scanned, which is the `sep_fv_point` arm, and three things say to stop there:
`sep_fv_point` already rescues both symptom events on its own; the two knobs
reach different events (§5.1 — 50 events only `sep_track_recarve` touches), so
turning both on is a configuration **nobody has run**; and
`sep_track_recarve`'s own two sentinel failures (mcp2k 94392, 53793 — Bee idx 9
and 10) are still unadjudicated. It remains available, default OFF and
byte-identical when off.

### 9.2 The compiled drift, key by key

`scripts/d97_flip_drift.py` walks every node of every artifact by
`(type, name)`, never by array index:

| artifact | drift |
|---|---|
| `sbnd_clus.json`, `sbnd_ql.json`, `prod.standalone`, `prod.wcls` | **8 keys each** — `fv_inset_yz = 150`, `far_point_x_cut = 140`, `far_point_mid_dis = 600`, `dec1_guard_main_angle = 45` on **both** `ClusteringSeparate` nodes |
| `sbnd_pr.json`, `sbnd_img.json`, `prod_prjob.json`, `bare_prjob.json`, `uboone.json`, the six pdhd and five pdvd jobs, the sim checks | **identical** (17 of 21 artifacts) |

The pre-flip tree, extracted with `git archive HEAD cfg`, still compiles to
`prod-2026-09-03` exactly (21/21) — so the drift above is the flip and nothing
else.

### 9.3 Production default == the validated arm

The arm that produced every number in §5 ran with `-sep-fv-point` on the command
line. Production takes the knob from the configuration instead, and that has to
be shown to be the same reconstruction, not assumed:

| | |
|---|---|
| data (`work-d97prodchk-*`, **no flag**) vs `work-<s>-d97fv` | 119 events / 476 products **byte-identical** |
| MC debug group, both symptom events | 20 events / 80 products **byte-identical** |
| manifest coverage | all six in-beam flip events, all three appear/disappear events, the full ncpi0 and nueCC48 sets, 20 gate308 mcp1k + 20 mcp2k, and 272-2-30 / 105-23-21 |
| new reference | `ref/prod-2026-09-04`, `prod_cfg_gate.py --ref ref/prod-2026-09-04` **PASS 21/21** |
| `prod-2026-09-03` | left byte-untouched (M13); it is the escape point, reachable with `run_ql_evt.sh -no-sep-fv-point` |

### 9.4 What this changes for the next round

* The stage-A baseline moves. `work-<s>-d97fv` / `work-<s>-d97fvpr2` are the
  production arms at `ref/prod-2026-09-04`; `work-<s>-r3entry` and the
  `grp0825` root behind it are **two generations stale** now — once for the
  epoch drift of §2.1 and once for this flip.
* The sentinel suite needs re-baselining: `pr/128 class B` on mcp2k 105074
  asserts a PF node that production deliberately no longer produces. Left as it
  stands here so the change is visible in the record; the next round should
  retire or re-anchor that sentinel and say so.
* `sep_track_recarve` and the both-on arm are the open items (§7 steps 2–4).
