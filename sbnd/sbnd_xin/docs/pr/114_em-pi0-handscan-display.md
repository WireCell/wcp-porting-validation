# doc pr/114 — `em_display`: a hand-scan display for EM shower clustering and π⁰

**Status: SHIPPED, scan-ready.** 94-event sample, probes parsed, 30/30 self-test
checks pass. **No C++ and no jsonnet changed — the toolkit repo is untouched, so
no A/B gate is owed and none is claimed.**

**Result.** A Bokeh display at `sbnd_xin/em_display/` for the owner's next
validation step. EM mode marks segments in/out of a selected shower against the
clustering's own acceptance geometry and, crucially, against **the pass that
actually absorbed each one** — read from probes pr/91 and pr/93 already built
into the code, not inferred. π⁰ mode builds a π⁰ from scratch (assign two γ,
set start points, choose or back-project the vertex) and shows the mass under
**both** direction conventions, because the code's own stored mass and angle use
different recipes and do not close. Four measurements came out of building it:
the stage-2 re-run **reproduces prod0825 on every physics field for all 94
events** (only the per-process `shower_id` counter moves); the lossy
`segments[].shower_id` join is **15 of 1567 showers (1.0 %), all 15 repaired** by
the probe; the gate the display draws (`pass3_cone`) is the **largest single
absorber at 41 % of 4030 absorptions**, with `pass4_angle` second at 21 %; and
**doc pr/113's census had a falsy-zero bug** that undercounted π⁰-paired events
by 10× — found because this display re-derives the pairing independently.

> **Read doc pr/113 §10 first if you have the round-1 lists.** They changed:
> nueCC 98 → **90**, NCπ⁰ 23 → **46**, numuCC-EM 79 → **79** (byte-identical).

## Repro

```sh
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin

# stage 2 -- re-run the 94-event sample with the shower probes on (~10 min, 4 fresh arms).
# ALREADY DONE: the four work-em114-* arms exist, so this now prints "NOTHING WAS RUN"
# and exits 0.  That is the expected steady state, not a success -- move an arm aside
# to genuinely redo it.
./em_display/run_em114_probe.sh

# parse the probes, rebuild the manifest, resolve Bee links offline (no network)
python em_display/prep_em_scan.py --parse-probes

# the two self-tests behind every number in this doc
python em_display/selftest_repro.py         # reproduction + membership repair
python em_display/selftest_em_display.py    # drives the viewer's callbacks, 30 checks

# serve
./em_display/serve_em_display.sh 5021 --scan-tag <your-tag>
#   ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -L 5021:localhost:5021 wcgpu1
#   http://localhost:5021/em_display_viewer
```

Anchors: toolkit `8d93260d` (**unchanged**), wcp-porting-img `6373efa` (the
pr/113 round-2 correction, which this round depends on).

---

## 0. Two corrections to the brief, made once and then built around

### 0.1 The π⁰ vertex is already reconstructed — the display judges it

The ask was *"for NCpi0 … it would also need to reconstruct the pi0 vertex, and
then display."* The code already does. `id_pi0_without_vertex` back-projects the
two γ rays with `WireCell::ray_closest_points`
(`clus/src/NeutrinoShowerClustering.cxx:4177`; helper `util/src/Point.cxx:125-171`)
and then **overwrites the main vertex** with the answer:

```cpp
// NeutrinoShowerClustering.cxx:4338-4340
main_vertex->fit().point = vtx_point;   // "(hack) - set to reconstructed pi0 decay point"
main_vertex->fit().dQ = 0;              // <- the tell-tale
```

Also, *"without vertex"* does **not** mean the event has no neutrino vertex: the
function returns early on `!main_vertex` (`:3961`) and only considers showers
with `start_connection_type == 3`, i.e. **detached** γ. A genuinely vertex-less
event gets no π⁰ at all.

So the deliverable is smaller than asked and more useful: show the vertex the
code computed, mirror the same back-projection so its inputs can be varied, mark
whether the code's own 25° gates would have accepted it, and let the owner
override. All three are built.

**A by-product worth knowing: `id_pi0_without_vertex` never fires on this
sample.** Across all 1433 prod0825 dumps there are **70 events carrying 78
accepted π⁰ groups**, and every one is flag-1 (`id_pi0_with_vertex`). Two
independent lines agree:

- **0 of 70** events have the `fit.dQ == 0` tell-tale on the main vertex, so no
  main vertex was ever overwritten by the back-projection;
- **0 of 78** groups contain a `start_connection_type == 3` member — the only
  kind `id_pi0_without_vertex` considers (`:4126`). The observed pairs are
  (2,2) × 46 and (1,2) × 32.

So the detached-γ path is live code that has produced nothing here. That is a
finding for a later round, not something this display fixes — but it is also what
makes §8's inference safe, and it would stop being safe the moment the path
fires.

### 0.2 Direction and π⁰ pairing needed no new dump field

`showers[]` already carries `start`, `end`, `start_connection_type`,
`start_vertex_id`, `num_segments`, `kine_charge`, `pio_id`, `pio_mass`,
`stem_dqdx`; `segments[]` carries `shower_id`, `dirsign`, `length`, `points[]`.
The shower axis is not a dump key, but the stage-2 probe prints it (`dir15`), and
where the probe is absent `em_geom.shower_init_dir` mirrors the C++ branch logic
(`PRShower.cxx:1552-1562` single-segment, `:1618-1640` multi). Two one-line
getter adds to `PrDisplayDump.cxx` were considered and **declined** (§6).

---

## 1. What was built

```
em_display/em_display_viewer.py    the app: one program, two modes
em_display/em_geom.py              the C++ geometry, mirrored with citations
em_display/prep_em_scan.py         manifest + probe parser + Bee index/build
em_display/run_em114_probe.sh      stage-2 launcher
em_display/serve_em_display.sh     bokeh serve wrapper, port 5021
em_display/selftest_em_display.py  30 headless checks over the real sample
em_display/selftest_repro.py       reproduction + membership repair
em_display/em114-manifest.tsv      the 94-event sample, links, per-event stats
em_display/emprep/emprep-evt*.json 94 probe sidecars
em_display/README.md               operating manual
```

Fork by duplication (CLAUDE.md §2 Code / M10): `pr_display/pr_display_viewer.py`
is **byte-untouched**. Its projection block, detector box, reentrancy guards,
Bokeh-3.9 DataTable repaint workaround and label-writer shape were copied and
then changed freely. No shared helper was extracted.

---

## 2. The sample

94 events = the whole corrected NCπ⁰ list (46) + the curated nueCC48 arm (48).
By arm: ncpi0 19, nuecc48 48, mcp1k 10, mcp2k 17.

| | n |
|---|---|
| events | 94 |
| …with a reconstructed π⁰ group | **42** |
| …with a lossy shower join | 12 |
| …with a probe sidecar | 94 |
| …with an offline Bee link, no upload needed | **78** |

42 of 94 carrying an actual reco π⁰ pair is what makes the π⁰ mode worth
building as an *audit* as well as a construction tool — and that number only
exists because of the pr/113 §10 correction. Under the round-1 census it would
have been 4.

---

## 3. Stage 2 reproduces prod0825 — measured, not assumed

The concern was real: prod0825 ran in **group mode** (`PR_GROUP_SIZE=16`) against
`libWireCellClus.so` installed 2026-08-25 **02:04**, and the re-run is per-event
against the lib installed **16:34** the same day, with the doc-80 MCS commits in
between.

`selftest_repro.py` compares every one of the 94 events, field by field:

> **all 94 events identical on `main_vertex`, `kine`, `tagger`, `showers` and
> `segments`.**

The **only** field that moves is `showers[].shower_id`, and it moves everywhere:
prod0825 numbers evt21073's showers 20…33 where the re-run numbers them 0…13,
because `Shower::get_shower_id()` is a **per-process** counter and prod0825 ran
16 events per process. It is not a physics quantity. Everything in this round
therefore joins on `showers[].id` / the probe's `node_id`
(`cluster_id*1000 + segment index`), which is stable.

*(Consequence worth recording: any consumer joining on `shower_id` across arms or
across group boundaries is wrong. `scripts/pr93_shower_composition.py:85` labels
its two id columns the wrong way round — it emits the start-segment id as
`shower_id` and the sequential id as `start_seg`. Mentioned, not fixed,
CLAUDE.md §5.)*

Because the two agree, **the display reads the prod0825 dumps** — the canonical
products — and takes only probe text from the em114 arms.

---

## 4. Membership: the dump is lossy in 1 %, and the display says so

`segments[].shower_id` stores **one** shower per segment
(`PrDisplayDump.cxx:490`), so when two showers overlap the loser's members
disappear from the join and it comes back looking *empty rather than nested* —
the condition `NeutrinoShowerClustering.cxx:116-126` describes.

Measured over the 94-event sample (1567 showers):

| | n |
|---|---|
| showers | 1567 |
| lossy join (`join != num_segments`) | **15 (1.0 %)** |
| repaired by the probe | **15 (100 %)** |
| probe member count == `num_segments` | **1567 / 1567** |

Worst cases, both in the sample and both used as the display's regression tests:
ncpi0 **evt84229 shower 69134** (43 of 50 joined, a 958 MeV shower) and ncpi0
**evt463565 shower 109073** (**0 of 5** — renders as an empty shower).

**The key point is that the dump is lossy but not silent.** `num_segments` is the
shower's own count, so the mismatch is computable with no re-run at all. The
display shows a `joined` column and a banner on affected events, and with a probe
sidecar it uses the non-lossy membership and the banner reads `REPAIRED`. Without
a sidecar it still refuses to pretend: the banner says the join is incomplete and
by how much.

**Axis provenance:** 1565 of 1567 shower axes come from the probe's `dir15`, i.e.
the C++'s own `shower_cal_dir_3vector(shower, start, 15 cm)`. Only 2 fall back to
the Python mirror. The probe prints components at `%.3f`, so vectors arrive with
|v| in 0.9994–1.0004 and are re-normalised on read — ≈0.06° on any angle, far
below anything a scan turns on, but they must *be* unit vectors for the
arithmetic. (Caught by the self-test, which asserted unit length.)

---

## 5. Where segments actually come from — a census

The `site=` tag names the pass that absorbed each segment. Over 4030 absorptions
in the 94 events:

| absorbing pass | n | share |
|---|---|---|
| `pass3_cone` (`:1310-1350`) | 1665 | **41.3 %** |
| `pass4_angle` (`:1964-1977`) | 838 | 20.8 % |
| `in_main_cluster` | 714 | 17.7 % |
| `from_vertices` | 408 | 10.1 % |
| `pass4_proximity` | 114 | 2.8 % |
| `pass3_cluster_map` | 67 | 1.7 % |
| 9 others (`in_other_clusters_*`, `examine_shower_1_*`, `conn3_unreachable`, `stem_backfill`, …) | 224 | 5.6 % |

Mechanism split: 2754 `direct` adds, 1259 flood-fill `walk_add`, 17
`walk_exclude`. Shower–shower splices are rare (22 total, led by
`examine_shower_1_assoc` 8 and `examine_showers_vtxcontain` 6).

**This is what justifies the acceptance plot and simultaneously bounds it.** The
gate it draws is the largest single absorber by 2×, so the panel is looking at
the dominant mechanism — but 59 % of absorptions happen elsewhere, so the panel
is a *geometry aid*, never a verdict. The UI says so in two directions:

- **above every step ≠ rejected** — `pass4_angle` uses different constants
  (`:1964-1967`);
- **below a step ≠ absorbed** — `shower_cone_absorb_guard` is **SBND-ON**
  (`wct-pr-perevt.jsonnet:1671`) and declines a confidently-PID'd non-electron
  straight track longer than 50 cm (`:1336-1351`, pr/93 Cause D).

The `absorbed by` column is the authority.

---

## 6. What the display shows, and the two traps it keeps apart

### 6.1 π⁰ mass, both conventions, side by side

`mass = √(4·E₁·E₂·sin²(θ/2))` with `E = kine_charge` — the code's own formula and
the code's own energy (`:3771`, `:4199`, `:4250`). The display computes θ two
ways: from the showers' **own axes**, and from the two **vertex→start chords**.

Both are shown because the code itself is inconsistent here: the mass it stores
uses `local_dirs` (`get_init_dir()` or the vertex chord, `:3760-3771`) while the
angle it stores is recomputed from `shower_cal_dir_3vector(…,15 cm)`
(`:3812-3831`). Over 282 events carrying both, `2√(E₁E₂)·sin(angle/2)` reproduces
the stored mass within 1 % in only **171**, and misses by >5 % in **78**.

Two mechanisms are located and **neither is picked** (CLAUDE.md §5.4): the
within-block recipe mismatch above, *and* the fact that the stored pair may be
one no reconstruction accepted (§6.2). The 78 is a mix of both and is not
attributed to either.

*Unit convention verified on the way:* `kine.kine_pio_angle` is in **degrees**
in the dump although `:3830` is an `acos()` — converted at
`NeutrinoKinematics.cxx:597-598`. Treating it as radians yields a mass wrong by a
factor you would not notice. (House precedent: the dQ/dx unit trap, doc pr/42.)

### 6.2 `pio_id` is the pairing; `kine_pio_*` is a BDT feature

They can name **different pairs**. On ncpi0 evt21073 the accepted groups are
(60081 + 31023, 127.2 MeV) and (11008 + 63100, 111.2 MeV) — while `kine_pio_*`
reports `energy_1` 680.2 (from 60081) and `energy_2` 104.7 (from 63100): a third
pairing, mass 207.25, that no reconstruction accepted. Cause: `pio_kine` comes
from a separate highest-energy scan over *all* candidate pairs (`:3777-3832`,
`:4260-4297`), while `map_pio_id_mass` records only what the winner loop accepted
(`:3836-3891`, `:4319-4334`).

The display draws them in **separate blocks** and never merges them. The
self-test asserts this (it would fail if a future edit merged the tables), and
`docs/pr/pr113-ncpi0.index.txt`'s `pio_mass` column is the `kine` variety — which
is why that column runs 10 → 1166 MeV and is almost never 135.

---

## 6.3 The back-projection has three branches, and the middle one is the common case

Caught by review *after* the first commit, and worth recording because a
plausible-looking mirror was wrong on the majority of real pairs.

`id_pi0_without_vertex` does not compute one vertex; it computes one of three:

| condition | vertex | line |
|---|---|---|
| both γ longer than 15 cm | midpoint of closest approach | `:4182-4200` |
| **exactly one** longer than 15 cm | **re-ray the short γ from that midpoint, re-intersect, keep the closest point on the LONG γ's ray** (not the midpoint; and no 3 cm fallback on this branch) | `:4203-4247` |
| both ≤ 15 cm | no π⁰ | `:4254-4255` |

The first mirror implemented only the first branch. Over the 78 accepted pairs on
disk the split is **29 both-long / 49 one-short** — so the majority took a branch
that did not exist in the code. Measured displacement between the midpoint the
first version would have shown and the vertex the code actually computes, over
those 49 pairs:

| | cm |
|---|---|
| median | **2.68** |
| p90 | **24.89** |
| max | **43.40** |
| pairs off by >1 cm / >5 cm / >20 cm | 36 / 20 / 6 |

All three branches are now mirrored, `pi0_backproject` returns which one ran, the
UI names it, and `selftest_em_display.py` asserts both live branches are
exercised and that one-short is 49 of 78 — so a regression that silently drops a
branch fails the test rather than quietly moving a vertex 25 cm.

## 7. Verification

`selftest_em_display.py`, 30 checks, all passing, over the real sample:

- **evt463565 shower 109073** renders as `5 / 5` with the note *"join lossy 0/5,
  REPAIRED by probe"* — not as an empty shower. The §4 regression case.
- **evt84229 shower 69134** has all 50 members, a unit-length probe axis, arrows
  in ≥2 panels, a populated candidate table with `absorbed by` filled, and at
  least one segment inside a pass-1 tier.
- **evt21073** yields exactly 2 accepted groups at 111.2 / 127.2 MeV, the kine
  block is labelled a BDT feature and its 207.25 shown separately.
- A hand-built pair (60081 + 31023) gives θ = 32.1°, m = **125.7 MeV**; the
  back-projection returns verdict `ok` with a 2.44 cm closest-approach gap.
  (The reco's own accepted group for those same two showers reads **127.2 MeV** —
  not a discrepancy: 125.7 is the *axis* convention, the reco's is `local_dirs`
  per §6.1. A 1.5 MeV spread between the two conventions on a well-behaved pair
  is the scale to have in mind when reading the >5 % misses.)
- Both back-projection branches fire on the real sample, 29 both-long and
  **49 one-short**, and the test pins that split (§6.3).
- Snap moves a γ start onto a real fitted point; a label round-trips through
  save → reload with both γ slots and both verdicts restored; the record carries
  the reco's groups **and** the `kine_pio_*` block **and** full provenance.
- The M13 tag guard refuses an implicit tag that already holds labels and accepts
  an explicit one.

`selftest_repro.py` produces §3 and §4's numbers.

**No A/B gate is owed.** No C++, no jsonnet, no config: there is no
byte-identicality claim to make and none is made. The stage-2 probes are
env-gated and stderr-only, and §3 measures rather than assumes that.

---

## 8. Declined, and what travels instead

Two one-line plain-getter adds to `PrDisplayDump.cxx` were offered and the owner
chose the re-run without them:

- `showers[].init_dir` from `Shower::get_init_dir()` (`PRShower.h:158`);
- `showers[].pio_method` from `map_pio_id_mass.at(id).second`
  (1 = `id_pi0_with_vertex`, 2 = `..._without_vertex`, currently dropped at
  `PrDisplayDump.cxx:618-620`).

So the toolkit stays unchanged and two caveats travel in the display instead. The
first turned out to cost almost nothing — the probe's `dir15` covers 1565/1567
axes (§4). The second means the display cannot *state* which π⁰ finder produced a
vertex; it infers it from `start_connection_type == 3` plus the `fit.dQ == 0`
tell-tale and labels the inference as such. §0.1's finding — that the flag-2 path
never fires on this sample — makes that inference safe for now, and would stop
being safe the moment it does fire.

## 9. What is NOT claimed

- **No physics claim about any event.** This round ships an instrument and the
  measurements needed to trust it. No shower is judged, no π⁰ is confirmed, and
  no knob is proposed.
- **No purity or efficiency.** The sample is reconstruction-defined on data;
  there is no truth to measure against (doc pr/113 §6.1).
- **The acceptance plot is not a verdict** (§5). 59 % of absorptions happen
  outside the gate it draws.
- **The 78/282 mass–angle non-closure is not attributed** to either of its two
  located mechanisms (§6.1).
- **`em_geom.shower_cal_dir_3vector` is not bit-exact.** The C++ walks the
  shower's view graph; the Python walks the dump's `fill_sets` membership. It
  affects only the 2 showers not covered by the probe's `dir15`.
- **`ray_closest_points` is mirrored, including its quirks.** The C++ signals the
  parallel case by returning the two ray origins, indistinguishable from an
  answer; and `scale1` divides by a quantity that vanishes when the rays already
  intersect (`Point.cxx:153`), yielding inf/nan silently. The Python reports
  `parallel` / `degenerate` instead of propagating either. That is a deliberate
  divergence in the *display*, not a proposed change to the code.

## 10. Two operational notes

**One prod0825 log is truncated, and a shipped tool greps it.**
`scripts/bee/make_pr_bee.py:78` decides whether an event was evaluated by
grepping the per-event log for `TaggerCheckNeutrino: selected main cluster`.
ncpi0 **evt399860**'s prod0825 log is **22 KB against a 207 KB median** and is
missing that line, so the builder refuses an event whose dump is perfectly good
(main vertex, 9 showers, 47 segments). It is isolated — 18 of 19 ncpi0 and 48 of
48 nuecc48 prod0825 logs carry the line, and the em114 arms carry it 94 of 94 —
so this is one truncated file, not a group-mode pattern. `prep_em_scan.py`
therefore points `make_pr_bee.py` at the **em114** arms, which is safe precisely
because §3 shows the two are equal on every physics field. Mentioned, not fixed
(CLAUDE.md §5): the general lesson is that a **log-derived** predicate is weaker
than a **product-derived** one, and `has_main` from the dump disagrees with it
here.

**The Bee set is built and waiting.** `bee/em114/` holds four zips (94 events,
51 MB) with their `.index.txt` and `.prid-map.txt` sidecars; every zip's first
member is `data/0/...`, which is what the server validates (`views.py:241`).
Upload is the owner's step (§5.6) — after it, dropping each returned URL beside
its zip as `.url` and re-running `prep_em_scan.py` fills the remaining 16 links
with no other change.
