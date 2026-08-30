# 86 — Showcase events for a reconstruction video: nine Bee sets from `prod0830`

Nineteen SBND events picked out of the `prod0830` production arm (doc 85 §9,
3067 events) to illustrate what the pattern-recognition chain currently does:
six "this works" categories and three failure modes, each its own Bee set with
the per-event reconstruction numbers recorded next to it.

**Scope.** No arm is re-run and no code changes. Everything below is read out
of products that already existed at `prod0830`. The picks are made by
*quantitative* gates on the reconstruction (listed in §2) — the visual pass is
what the Bee links are for.

## Repro block

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# 1. the candidate pools, the wide feature table, and the final set manifest
python3 scripts/analysis/d86_video_picks.py        # -> docs/86_video/

# 2. the nine Bee zips (LOCAL; upload is a separate owner authorisation)
./scripts/bee/build_d86_bee.sh                     # -> bee/d86/*.zip

# 3. prove each zip holds the events the manifest names, then upload
python3 scripts/bee/verify_d86_bee.py; echo rc=$?  # 19/19 OK, rc=0
for c in nuecc numucc-cathode numucc ccpi0 ncpi0 cosmiclike multinu \
         fail-busy fail-em; do
  ./upload-to-bee.sh bee/d86/d86-$c.zip            # owner-authorised 2026-08-30
done
```

---

## 1. Read this before using the labels

**There is no truth anywhere in this document.** All four `prod0830` samples
are SBND *data* selections (doc 85 §1; doc 21 line 51 — the reco1 input carries
no MC truth and no `raw::RawDigits`). So:

* "ν<sub>e</sub>CC", "ν<sub>μ</sub>CC", "CC π⁰", "NC π⁰" name a **reconstructed
  topology on a selected sample**, never a known interaction. Every one of them
  should be read as "…candidate".
* The CC/NC split for the π⁰ events is *defined here* as "the reconstruction
  put a primary muon at the vertex, and the ν<sub>μ</sub> BDT agrees" — that is
  the only handle available without truth. It is a reconstruction statement.
* "Well reconstructed" means the event passed the gates in §2. It is not a
  claim that the reconstruction is *correct*, which only a scan can say.

**`nue_score = -15` is a sentinel, not a score.** It means `br_filled != 1` —
the ν<sub>e</sub> BDT never ran on that candidate. It appears on most of the
ν<sub>μ</sub>-like picks below, which is the expected behaviour there, and on
one *failure* pick (§5.3) where it is the whole point. Since doc 85 §9 removed
the score clamp the physical range reaches ±16.25562, so −15 now sits *inside*
it: test this sentinel as **equality with −15**, never `< -14.9`.

### 1.1 `kine_pio_flag == 1` does not mean "a π⁰ was found"

This one would have produced a bad video, so it is recorded in full.

`NeutrinoShowerClustering.cxx:6046-6100` fills the **entire** `pio_kine` block
for the highest-energy shower *pair* in the event, whatever that pair is, as a
BDT feature. It is not a π⁰ identification verdict. On `prod0830`:

| | rows | median mass | median γ energies |
|---|---:|---:|---:|
| `kine_pio_flag == 1` | 973 / 1458 | 5 MeV | 2.7, 2.7 MeV |
| …of those, mass ∈ [100, 170] | 64 | — | 114, 108 MeV |

The 900-odd low-mass rows are pairs of shower *fragments*. Gating a video on
`flag == 1` would have selected them.

The field semantics also do not read the way the header comment suggests:

| field | what it actually is |
|---|---|
| `kine_pio_vtx_dis` | π⁰ decay vertex → **main ν vertex**. This is the "γ's point back at the neutrino vertex" quantity. |
| `kine_pio_dis_1/2` | π⁰ vertex → each shower **start point** = the photon **conversion gap**. For a real π⁰ this is *expected to be large*, so gating on "small `dis`" would select the mis-reconstructed pairs. |
| `kine_pio_angle` | opening angle. Note `dis > 3 cm` makes the direction the vertex→start vector *by construction* (line 6082), so for those the pointing is assumed, not independently measured. |

`kine_shower_fudge_factor = 0.84` is ON in this production (`d9814518`), so
every γ energy and π⁰ mass quoted here carries that shower-energy scale.

---

## 2. The gates

Applied to every **showcase** pick (the failure picks in §5 deliberately skip
them):

* `rc == 0` — read from `pr_evt<ID>/rc.txt`. This arm did not write the
  `.time.meta` that `pr_scores_table.py` reads, so `rc`/`wall_s`/`maxrss_kb` are
  **blank in the score TSVs**; the table here uses the real source.
* `nu_evaluated == 1` — the PR log carries `TaggerCheckNeutrino: selected main
  cluster …`, i.e. `TaggerInfo`/`KineInfo` hold real features, not struct
  defaults.
* **Not** the doc 85 §1.1 degenerate class (`kine_reco_Enu == 0` at vertex
  `(0,0,0)`) — 49 such events exist and would otherwise walk straight in.
* `dl_warn == 0`; vertex inside the SBND active volume (doc 2:
  x ±201.45, y ±200, z 0–500 cm) with a 20 cm margin.
* **Completeness:** `kine_energy_excluded / kine_reco_Enu < 5 %`, using the
  variable added in doc 85 §9 — the charge in the PR graph that the E<sub>ν</sub>
  sum did *not* consume. Most picks are at 0.0 %.
* Category-specific: `nue_score > 7.0` / `numu_score > 0.9` (the MicroBooNE
  working points, doc 85 §7), reconstructed particle content, and for the π⁰
  sets the kinematic gate below.
* **Cosmic verdict is `cosmict_flag`**, not `cosmic_flag` — the latter is
  `1.0` on all 48 `nuecc48` events and is not a cosmic verdict.

**π⁰ gate** (from §1.1): mass ∈ [90, 180] MeV, both γ ∈ [40, 800] MeV,
energy asymmetry < 0.70, opening angle > 15°, `vtx_dis` < 5 cm. 31 of 1458 rows
survive (3 in the NC π⁰ sideband sample). The asymmetry and angle gates are for
the *display* — a 1.7 GeV shower paired with a 4 MeV fragment can land on
135 MeV by accident and shows nothing on screen.

**Cathode crossing** is measured on the **muon's own endpoints** from the
particle-flow tree, not on the candidate's overall x-extent — the latter counts
two unrelated prongs on opposite sides as a "crosser". SBND's cathode is at
x = ∓0.45 cm (doc 2).

---

## 3. The showcase sets

Bee numbers events by **upload order**, not by event id, so the `event/<i>/`
index below is part of the record. Full per-event numbers:
`docs/86_video/d86-final.tsv`.

### 3.1 ν<sub>e</sub>CC candidates — [set `ad10e755`](https://www.phy.bnl.gov/twister/bee/set/ad10e755-07d8-481c-91d9-29e8b0469007/event/list/)

| # | run/sub/evt | `nue_score` | `numu_score` | E<sub>ν</sub> | excluded | reconstructed final state |
|---|---|---:|---:|---:|---:|---|
| [0](https://www.phy.bnl.gov/twister/bee/set/ad10e755-07d8-481c-91d9-29e8b0469007/event/0/) | 18313/1/**81597** | **15.05** | −0.56 | 1511 MeV | 1.4 MeV (0.1 %) | e⁻ 1362 + proton 139 |
| [1](https://www.phy.bnl.gov/twister/bee/set/ad10e755-07d8-481c-91d9-29e8b0469007/event/1/) | 18279/1/**267597** | **14.42** | −0.70 | 1792 MeV | 8.7 MeV (0.5 %) | e⁻ 1256 + proton 395 + n 89 + p 89 |

81597 is the cleanest 1e1p in the sample: a single 1.36 GeV electron shower, one
139 MeV proton, and 1.4 MeV of charge unaccounted for. Both events clear the
ν<sub>e</sub> working point by more than the *entire* dynamic range the score
had before doc 85 §9 removed the clamp (old ceiling 4.30103).

### 3.2 ν<sub>μ</sub>CC candidates — [set `ef6c82c7`](https://www.phy.bnl.gov/twister/bee/set/ef6c82c7-33aa-4b30-88bc-54d550d26f3f/event/list/)

| # | run/sub/evt | `numu_score` | E<sub>ν</sub> | excluded | reconstructed final state |
|---|---|---:|---:|---:|---|
| [0](https://www.phy.bnl.gov/twister/bee/set/ef6c82c7-33aa-4b30-88bc-54d550d26f3f/event/0/) | 18255/1/**290718** | **4.78** | 1250 MeV | **0.0 MeV** | μ⁻ 1138 + n 3 + p 3 |
| [1](https://www.phy.bnl.gov/twister/bee/set/ef6c82c7-33aa-4b30-88bc-54d550d26f3f/event/1/) | 18255/1/**94293** | 4.19 | 1516 MeV | **0.0 MeV** | μ⁻ 1067 + π⁺ 107 + γ 35 + γ 32 |

290718 is a single 1.14 GeV muon from a vertex at (−23.7, 141.7, 32.7) running
to (−108.8, 141.8, 500.9) — it exits the downstream face. Zero excluded charge.

### 3.3 ν<sub>μ</sub>CC **crossing the cathode** — [set `4f929df8`](https://www.phy.bnl.gov/twister/bee/set/4f929df8-7e36-404f-b1f6-6b42fec543df/event/list/)

| # | run/sub/evt | `numu_score` | E<sub>ν</sub> | excluded | muon x: start → end | final state |
|---|---|---:|---:|---:|---|---|
| [0](https://www.phy.bnl.gov/twister/bee/set/4f929df8-7e36-404f-b1f6-6b42fec543df/event/0/) | 18255/1/**283591** | 4.11 | 836 MeV | **0.0 MeV** | **−79.6 → +62.3 cm** | μ⁻ 621 + proton 100 |
| [1](https://www.phy.bnl.gov/twister/bee/set/4f929df8-7e36-404f-b1f6-6b42fec543df/event/1/) | 18255/1/**313979** | 4.37 | 919 MeV | **0.0 MeV** | **−18.2 → +94.4 cm** | μ⁻ 607 + proton 197 |

Both are clean 1μ1p with the muon passing through the cathode plane — the muon
is reconstructed as **one** particle spanning both drift volumes, which is the
thing worth showing. 283591's muon runs (−79.6, 126.4, 26.3) → (62.3, 198.0,
236.5); 313979's runs (−18.2, −37.3, 269.4) → (94.4, −194.5, 439.4). Zero
excluded charge on both.

### 3.4 CC π⁰ candidate — [set `2af3970b`](https://www.phy.bnl.gov/twister/bee/set/2af3970b-9d97-4dff-b499-f592fe704512/event/list/)

| # | run/sub/evt | π⁰ mass | γ₁, γ₂ | opening ∠ | `vtx_dis` | `numu_score` | excluded | final state |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0](https://www.phy.bnl.gov/twister/bee/set/2af3970b-9d97-4dff-b499-f592fe704512/event/0/) | 18255/1/**400504** | **138.9 MeV** | 64, 146 MeV | 73.0° | **0.00 cm** | 2.06 | 0.8 MeV (0.2 %) | μ⁻ 159 + **π⁰ 138** (γ 146 + γ 64) |
| [1](https://www.phy.bnl.gov/twister/bee/set/2af3970b-9d97-4dff-b499-f592fe704512/event/1/) | 18261/1/**285567** | **139.0 MeV** | 82, 97 MeV | 101.4° | **0.00 cm** | 0.99 | 181.7 MeV (10.0 %) | p 323 + π⁺ 197 + μ⁻ 187 + **π⁰ 139** |

**400504 is the primary pick**: mass 138.9 MeV from two balanced photons, both
back-projecting to the neutrino vertex (`vtx_dis` = 0.00 cm), a muon at the
vertex, and 0.2 % excluded charge. The particle-flow tree carries an explicit
`pi0 138 MeV` node.

285567 is the busier alternate — it comes from the **NC π⁰ sideband sample** yet
the reconstruction puts a 187 MeV muon at its vertex and the ν<sub>μ</sub> BDT
agrees (0.99, just over the working point). Recorded as-is; it is a genuine
sideband-vs-reconstruction disagreement, not a pick error.

### 3.5 NC π⁰ candidate — [set `5e5b7a94`](https://www.phy.bnl.gov/twister/bee/set/5e5b7a94-09aa-43f7-96b5-fcfe953ef8f8/event/list/)

Two picks with **different claims**, because no single event has both.

| # | run/sub/evt | sample | π⁰ mass | γ₁, γ₂ | ∠ | `vtx_dis` | `numu_score` | muon? | final state |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| [0](https://www.phy.bnl.gov/twister/bee/set/5e5b7a94-09aa-43f7-96b5-fcfe953ef8f8/event/0/) | 18255/1/**57709** | mcp2k | **147.9 MeV** | 186, 94 | 68.2° | **0.00 cm** | **−0.29** | none | p 420 + **π⁰ 147** (γ 185 + γ 93) |
| [1](https://www.phy.bnl.gov/twister/bee/set/5e5b7a94-09aa-43f7-96b5-fcfe953ef8f8/event/1/) | 18255/1/**180801** | **ncpi0 sideband** | 99.3 MeV | 164, 293 | 23.9° | **0.00 cm** | +1.36 | none | e⁻ 292 + **π⁰ 139** + γ 164 + π⁺ 129 |

* **57709** is *self-consistent*: no reconstructed muon, and the ν<sub>μ</sub> BDT
  agrees (−0.29, below the working point). Two photons at 186 and 94 MeV both
  back-projecting to the vertex, mass 147.9 MeV.
* **180801** is the best muon-free π⁰ the **NC π⁰ sideband sample itself**
  offers — but its `numu_score` is +1.36, i.e. the ν<sub>μ</sub> BDT *would*
  select it. That tension is reported rather than gated away. Its opening angle
  (23.9°) is also the tightest of the π⁰ picks.

The sideband sample has only **3** rows passing the §2 π⁰ gate at all, and the
other two carry reconstructed muons — so there is no NC π⁰ pick that is both
from the sideband and BDT-consistent. Stated rather than papered over.

### 3.6 Cosmic-like — [set `214131fd`](https://www.phy.bnl.gov/twister/bee/set/214131fd-39b4-475b-bc71-5e15af4130b8/event/list/)

| # | run/sub/evt | `cosmict_flag` | `numu_score` | muon | track extent | final state |
|---|---|---:|---:|---:|---|---|
| [0](https://www.phy.bnl.gov/twister/bee/set/214131fd-39b4-475b-bc71-5e15af4130b8/event/0/) | 18259/1/**180698** | **1** | **−5.08** | 617 MeV | 681 cm diagonal | μ⁻ 617 + γ 8 |
| [1](https://www.phy.bnl.gov/twister/bee/set/214131fd-39b4-475b-bc71-5e15af4130b8/event/1/) | 18255/1/**99563** | **1** | **−5.10** | 1159 MeV | 501 cm diagonal | μ⁻ 1159 + γ 22 + γ 14 + γ 13 |

Both are long muons that the cosmic tagger flagged (`cosmict_flag = 1`) *and*
that the ν<sub>μ</sub> BDT independently scores at the far cosmic end (−5.1,
against +4 to +5 for the ν<sub>μ</sub>CC picks in §3.2/3.3) — the two
independent verdicts agreeing is the point. 180698's muon enters at
(−158.5, 183.2, 1.1), near the top of the upstream face; 99563's runs
(−91.2, −14.7, 1.8) → (3.4, 83.9, 468.9), nearly through the whole detector.

### 3.7 More than one neutrino in one event — [set `45c4cc54`](https://www.phy.bnl.gov/twister/bee/set/45c4cc54-a9ae-461e-8301-e6a71be071ce/event/list/)

| # | run/sub/evt | ν candidates | vertex 1 | final state 1 | vertex 2 | final state 2 |
|---|---|---:|---|---|---|---|
| [0](https://www.phy.bnl.gov/twister/bee/set/45c4cc54-a9ae-461e-8301-e6a71be071ce/event/0/) | 18255/1/**487303** | **2** | (132.0, 72.3, 375.9) | μ⁻ 298 + p 84, E<sub>ν</sub> 503 MeV | (−82.6, −160.7, 409.7) | p 161 + e⁻ 99, E<sub>ν</sub> 273 MeV |
| [1](https://www.phy.bnl.gov/twister/bee/set/45c4cc54-a9ae-461e-8301-e6a71be071ce/event/1/) | 18259/1/**174661** | **2** | (100.8, 40.3, 114.9) | μ⁻ 304 + p 200, E<sub>ν</sub> 619 MeV | (−27.4, −153.6, 137.4) | μ⁻ 335, E<sub>ν</sub> 457 MeV |

Two independently reconstructed neutrino candidates per event, each with its own
vertex, its own particle flow, its own `T_kine` row and its own E<sub>ν</sub> —
and in both events the two vertices sit in **opposite drift volumes** (x > 0 and
x < 0), ~250–300 cm apart. Both have **zero excluded charge**.

487303 is the more illustrative: one ν<sub>μ</sub>CC-like (μ + p) and one
NC/ν<sub>e</sub>-like (p + e⁻). Note the `mc` tree gives them different flash
group ids (`gid 1000005` vs `gid 6`), i.e. they are separately flash-matched
bundles, not one interaction split in two.

> Caveat worth stating: two reconstructed candidates is not proof of two
> neutrinos. It is what the reconstruction produced, which is what the video
> shows. `pr_scores_table.py` reports only one row per event
> (`primary_index()`), so these numbers come from `T_kine` directly.

---

## 4. Summary table

| category | Bee set | events | headline number |
|---|---|---|---|
| ν<sub>e</sub>CC | [`ad10e755`](https://www.phy.bnl.gov/twister/bee/set/ad10e755-07d8-481c-91d9-29e8b0469007/event/list/) | 81597, 267597 | `nue_score` 15.05 / 14.42 |
| ν<sub>μ</sub>CC | [`ef6c82c7`](https://www.phy.bnl.gov/twister/bee/set/ef6c82c7-33aa-4b30-88bc-54d550d26f3f/event/list/) | 290718, 94293 | 1.14 GeV muon, 0 MeV excluded |
| ν<sub>μ</sub>CC, cathode crosser | [`4f929df8`](https://www.phy.bnl.gov/twister/bee/set/4f929df8-7e36-404f-b1f6-6b42fec543df/event/list/) | 283591, 313979 | muon x −79.6 → +62.3 cm |
| CC π⁰ | [`2af3970b`](https://www.phy.bnl.gov/twister/bee/set/2af3970b-9d97-4dff-b499-f592fe704512/event/list/) | 400504, 285567 | mass 138.9 MeV, `vtx_dis` 0.00 |
| NC π⁰ | [`5e5b7a94`](https://www.phy.bnl.gov/twister/bee/set/5e5b7a94-09aa-43f7-96b5-fcfe953ef8f8/event/list/) | 57709, 180801 | mass 147.9 MeV, `numu_score` −0.29 |
| cosmic-like | [`214131fd`](https://www.phy.bnl.gov/twister/bee/set/214131fd-39b4-475b-bc71-5e15af4130b8/event/list/) | 180698, 99563 | `cosmict_flag` 1, `numu_score` −5.1 |
| multiple ν | [`45c4cc54`](https://www.phy.bnl.gov/twister/bee/set/45c4cc54-a9ae-461e-8301-e6a71be071ce/event/list/) | 487303, 174661 | 2 vertices, opposite drift volumes |
| **failure** — busy | [`1752786d`](https://www.phy.bnl.gov/twister/bee/set/1752786d-d8c5-4c47-8f30-58532983a07b/event/list/) | 389538, 67868 | 82.3 % of E<sub>ν</sub> excluded |
| **failure** — EM shower | [`172e006a`](https://www.phy.bnl.gov/twister/bee/set/172e006a-a499-4228-854a-4a94257701e2/event/list/) | 69314, 138009, 271851 | shower in 31 pieces; two near-misses |

---

## 5. The failure cases

These deliberately **skip** the §2 gates — failing them is the point.

### 5.1 Busy event — [set `1752786d`](https://www.phy.bnl.gov/twister/bee/set/1752786d-d8c5-4c47-8f30-58532983a07b/event/list/)

| # | run/sub/evt | E<sub>ν</sub> | **excluded** | `n_excluded` | `nue_score` | final state |
|---|---|---:|---:|---:|---:|---|
| [0](https://www.phy.bnl.gov/twister/bee/set/1752786d-d8c5-4c47-8f30-58532983a07b/event/0/) | 18255/1/**389538** | 981 MeV | **807.4 MeV (82.3 %)** | **26** | −15 (never ran) | μ⁻ 388 + π⁺ 142 + p 69 + γ 43 … |
| [1](https://www.phy.bnl.gov/twister/bee/set/1752786d-d8c5-4c47-8f30-58532983a07b/event/1/) | 18255/1/**67868** | 921 MeV | 169.1 MeV (18.4 %) | 21 | −15 (never ran) | p 297 + μ⁻ 261 + p 81 + p 70 … |

**389538** is the sharpest illustration in the whole set: a ν<sub>e</sub>CC-selected
event where the PR graph holds **26 segments carrying 807 MeV** that the
E<sub>ν</sub> sum does not account for — 82 % of the reconstructed energy again
over — and the ν<sub>e</sub> BDT never ran at all. Its π⁰ row is junk in the
§1.1 sense (a 937 MeV "γ" paired with 17.8 MeV, `vtx_dis` 160 cm), which is a
useful counter-example to show next to §3.4.

The excluded-energy variable itself is new (doc 85 §9); across all four samples
its median is 0.27–3.2 % of E<sub>ν</sub>, so 82.3 % is a genuine outlier and
not a typical event.

### 5.2 EM shower clustering — near misses

| # | run/sub/evt | `nue_score` | E<sub>ν</sub> | excluded | final state |
|---|---|---:|---:|---:|---|
| [1](https://www.phy.bnl.gov/twister/bee/set/172e006a-a499-4228-854a-4a94257701e2/event/1/) | 18255/1/**138009** | **5.83** | 1381 MeV | 1.4 MeV (0.1 %) | e⁻ 1127 + p 222 + p 6 |
| [2](https://www.phy.bnl.gov/twister/bee/set/172e006a-a499-4228-854a-4a94257701e2/event/2/) | 18255/1/**271851** | **5.55** | 1476 MeV | 9.1 MeV (0.6 %) | e⁻ 1211 + μ⁻ 156 |

Both are ν<sub>e</sub>CC-selected events with a single big electron shower and
almost no excluded charge — they look like §3.1 — and both land at 5.6–5.8,
**below** the 7.0 working point. These are the interesting failures: not broken
reconstruction, but a score that lands short. Doc 85 §9.7 counts six such
near-misses between 3.9 and 5.8 in the 48-event sample.

### 5.3 EM shower clustering — the shower never became one object

| # | run/sub/evt | `nue_score` | electron-type objects | largest | final state |
|---|---|---:|---:|---:|---|
| [0](https://www.phy.bnl.gov/twister/bee/set/172e006a-a499-4228-854a-4a94257701e2/event/0/) | 18255/1/**69314** | **−15** (never ran) | **31** | **71 MeV** | μ⁻ 362 + p 252 + p 227 + π⁺ 128 + e⁻ 71 + e⁻ 62 |

The clearest EM-clustering failure available. A 1497 MeV ν<sub>e</sub>CC-selected
event whose electromagnetic activity is reconstructed as **31 separate
electron-type objects, the largest only 71 MeV** — the shower never became one
object, so no electron candidate exists, `br_filled != 1`, and the
ν<sub>e</sub> BDT never ran (`nue_score` = −15 exactly). 69314 is one of the
four such events doc 85 §9.7 identified, and is already named in the pr/125
round as a satellite-absorption case.

---

## 6. What this document does not claim

1. **No truth, so no efficiency, purity or resolution statement.** These are 19
   hand-picked events out of 3067, chosen to be *illustrative*, and they are not
   a measurement of anything.
2. **"Well reconstructed" is §2's gate list, not a verdict.** Nothing here has
   been visually scanned; the Bee links exist so that it can be.
3. **The CC/NC π⁰ split is a reconstruction statement** (muon present + the
   ν<sub>μ</sub> BDT agreeing), and §3.4/§3.5 record two events where the sample
   selection and the reconstruction disagree.
4. **Bee indices follow upload order.** The `event/<i>/` numbers above are from
   the pick lists and were re-verified against each zip
   (`verify_d86_bee.py`, 19/19, rc=0) and against the live set pages by counting
   `event/N/` links per set (2/2 ×8, 3/3 ×1). A 200 status code alone proves
   nothing — the viewer returns 200 for out-of-range indices too.

## 7. Artifacts

| path | what |
|---|---|
| `scripts/analysis/d86_video_picks.py` | pools, gates, feature table, final manifest |
| `scripts/bee/build_d86_bee.sh` | builds the nine zips |
| `scripts/bee/verify_d86_bee.py` | run/subrun/event + layer check per zip member |
| `docs/86_video/d86-final.tsv` | **the per-event record** — every number quoted above |
| `docs/86_video/d86-features.tsv` | all 1458 evaluated `(event, T_kine row)` rows |
| `docs/86_video/d86-set-*.txt` | the nine pick lists, in Bee upload order |
| `bee/d86/*.zip` | the nine uploaded zips (kept for rebuild) |
