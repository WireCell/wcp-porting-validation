# 86 — Showcase events for a reconstruction video: nine Bee sets from `prod0830`

Twenty SBND events picked out of the `prod0830` production arm (doc 85 §9,
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
python3 scripts/bee/verify_d86_bee.py; echo rc=$?  # 20/20 OK, rc=0
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
| `kine_pio_dis_1/2` | π⁰ vertex → each shower **start point** = the photon **conversion gap**. For a real π⁰ this is *expected to be large*, so gating on "small `dis`" would select the mis-reconstructed pairs. The good-mass population sits at 21 / 15 cm median. |
| `kine_pio_angle` | opening angle — but **not** the one the mass was built from. See §1.2. |
| `kine_pio_vtx_dis` | π⁰ decay vertex → main ν vertex. **Near-tautologically 0** and *not* usable as a pointing gate: the candidate π⁰ vertex is constrained to be one of the two showers' own start vertices and the loop keeps whichever is closest to the main vertex (:6062-6066), so it reads 0.00 whenever a shower starts at the main vertex. 487 of the 712 *junk* rows (mass < 20 MeV) pass `vtx_dis < 2 cm`. It is reported below for completeness, never leaned on. |

`kine_shower_fudge_factor = 0.84` is ON in this production (`d9814518`), so
every γ energy and π⁰ mass quoted here carries that shower-energy scale.

### 1.2 The reported mass does not always close from the reported γ's and angle

The finder builds `mass = sqrt(4·E₁·E₂·sin²(θ/2))` (:6034) — but **the E's and θ
it stores are not the ones it used**, because the two are chosen by *different
rules*:

* `mass` uses `local_dirs[sh]`, which is the shower's own `get_init_dir()` when
  the shower is attached to the candidate π⁰ vertex (:5964, :5978) and the
  vertex→start vector when it was associated by angle (:5997);
* `kine_pio_angle` is **recomputed** in the fill loop (:6078-6084) from a
  different rule — a fresh 15 cm direction fit when the conversion gap is
  < 3 cm, the vertex→start vector otherwise.

`kine_pio_energy_1/2` are safe (`get_kq`, and the K12 virtual-sum knob
`pi0_collinear_merge_deg` is `null` ⇒ C++ default 0 ⇒ off in this production),
so **the whole discrepancy is the angle**. Across the 35-row π⁰ pool the
reported triple reproduces the reported mass **25 times** and misses on 10, by
up to 20 %:

| event | reported E₁, E₂, θ | recomputed | reported mass |
|---|---|---:|---:|
| 285567 | 82.5, 97.2, 101.4° | 138.6 | 139.0 ✓ |
| 57709 | 185.8, 93.8, 68.2° | 148.0 | 147.9 ✓ |
| **400504** | 64.4, 146.2, **73.0°** | **115.4** | **138.9** ✗ (mass implies 91.4°) |
| **180801** | 164.1, 292.6, **23.9°** | **90.7** | **99.3** ✗ (implies 26.2°) |

**Consequence for the video: `kine_pio_mass` is the finder's number and must not
be re-derived on screen from the E₁, E₂ and angle printed beside it.** Every π⁰
pick in §3.4/§3.5 was therefore re-ranked to require the triple to close
(`pio_mass_closure` in both TSVs, tolerance 0.5 %) — a *display* criterion, not
a physics-quality claim. 400504, which an earlier draft of this document had as
the primary CC π⁰ pick, is the worked example above rather than a pick.

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
survive (3 in the NC π⁰ sideband sample); after the FV cut the CC pool is 16 and
the muon-free BDT-consistent NC pool is 5. The asymmetry and angle gates are for
the *display* — a 1.7 GeV shower paired with a 4 MeV fragment can land on
135 MeV by accident and shows nothing on screen. Picks additionally require the
**mass to close** from the reported triple (§1.2): 12 of the 16 CC pool rows and
5 of 5 NC pool rows do.

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
236.5); 313979's runs (−18.2, −37.3, 269.4) → (94.4, −194.5, 439.4).  Zero
excluded charge on both.

Neither muon is contained: 283591's ends 2 cm from the y = +200 boundary and
313979's 5.5 cm from y = −200, so both **exit through the top/bottom**. The 621
and 607 MeV are the chain's energy estimates for exiting tracks, not
range-based measurements — worth not narrating as the latter. (The §3.2 picks
exit downstream at z ≈ 500 for the same reason.)

### 3.4 CC π⁰ candidate — [set `e915a828`](https://www.phy.bnl.gov/twister/bee/set/e915a828-177b-49d8-9d06-a2aab3bcbc97/event/list/)

| # | run/sub/evt | π⁰ mass | closes? | γ₁, γ₂ | ∠ | conversion gaps | `numu_score` | excluded | final state |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [0](https://www.phy.bnl.gov/twister/bee/set/e915a828-177b-49d8-9d06-a2aab3bcbc97/event/0/) | 18255/1/**99838** | **126.3 MeV** | **1.000** | 125, 104 MeV | 67.3° | 32.5 / 3.1 cm | **4.11** | 4.6 MeV (0.3 %) | μ⁻ 1046 + **π⁰ 126** (γ 125 + γ 103) |
| [1](https://www.phy.bnl.gov/twister/bee/set/e915a828-177b-49d8-9d06-a2aab3bcbc97/event/1/) | 18259/1/**242726** | **128.1 MeV** | **1.000** | 111, 65 MeV | 98.6° | 18.4 / 3.0 cm | 3.62 | **0.0 MeV** | μ⁻ 394 + p 355 + **π⁰ 128** (γ 110 + γ 64) |

**99838 is the primary pick**: a 1046 MeV muon at the vertex, a π⁰ from two
well-balanced photons (125 and 104 MeV, 67° apart) whose mass recomputes exactly
from the numbers shown, and 0.3 % of the charge unaccounted for. The larger
photon converts 32.5 cm from the π⁰ vertex — the visible gap that makes a π⁰
read as a π⁰ on screen rather than as an electron. The particle-flow tree
carries an explicit `pi0 126 MeV` node.

242726 is the same topology with a proton as well (μ 394 + p 355 + π⁰ 128) and
**zero** excluded charge, at a wider 98.6° opening.

Both have `numu_score` of 3.6–4.1, so the "CC" half of the label rests on the
muon *and* the BDT, not on one number. Closer-in-mass alternates exist (285567
at 139.0, 392901 at 129.1) but carry 10 % / 5.8 % excluded charge — the pools
are in `d86-features.tsv`.

### 3.5 NC π⁰ candidate — [set `2b944df2`](https://www.phy.bnl.gov/twister/bee/set/2b944df2-7b85-4b3e-ba92-62263c36bf5f/event/list/)

| # | run/sub/evt | sample | π⁰ mass | closes? | γ₁, γ₂ | ∠ | gaps | `numu_score` | muon? | final state |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| [0](https://www.phy.bnl.gov/twister/bee/set/2b944df2-7b85-4b3e-ba92-62263c36bf5f/event/0/) | 18255/1/**57709** | mcp2k | **147.9 MeV** | **1.000** | 186, 94 | 68.2° | 14.8 / 37.4 cm | **−0.29** | none | p 420 + **π⁰ 147** (γ 185 + γ 93) |
| [1](https://www.phy.bnl.gov/twister/bee/set/2b944df2-7b85-4b3e-ba92-62263c36bf5f/event/1/) | 18259/1/**176986** | mcp2k | **108.4 MeV** | **1.000** | 168, 114 | 46.0° | 23.2 / 7.2 cm | **−0.66** | none | **π⁰** (γ 168 + γ 114) + γ 48, E<sub>ν</sub> 352 MeV |
| [2](https://www.phy.bnl.gov/twister/bee/set/2b944df2-7b85-4b3e-ba92-62263c36bf5f/event/2/) | 18255/1/**180801** | **ncpi0 sideband** | 99.3 MeV | **0.912** ✗ | 164, 293 | 23.9° | 28.9 / **0.0** cm | **+1.36** | none | e⁻ 292 + γ 164 + π⁺ 129 |

* **57709 is the primary pick** and is fully self-consistent: no reconstructed
  muon, the ν<sub>μ</sub> BDT agrees (−0.29, below the working point), the mass
  closes exactly, and **both** photons have a real conversion gap (14.8 and
  37.4 cm) — the clearest two-γ topology in the set.
* **176986** is the cleanest by completeness (**0.0 MeV** excluded, E<sub>ν</sub>
  352 MeV) with two balanced photons at 168 and 114 MeV.
* **180801, index 2, is the only muon-free π⁰ the NC π⁰ sideband sample itself
  offers, and it is included with three caveats**, none of which should be
  discovered on camera: its triple does not close (0.912), its second photon has
  **no conversion gap at all** (0.0 cm), and its `numu_score` is +1.36 — the
  ν<sub>μ</sub> BDT would select it. Use it only if a sideband-sample event is
  specifically wanted.

The sideband sample has only **3** rows passing the §2 π⁰ gate, and the other
two carry reconstructed muons — so there is no NC π⁰ pick that is at once from
the sideband, BDT-consistent and internally consistent. Stated rather than
papered over.

> One display caveat: the particle-flow tree's own `pi0 <E>` node and
> `kine_pio_mass` are the same number on 57709 (147 / 147.9), 99838 (126 /
> 126.3) and 242726 (128 / 128.1), but differ on 176986 (124 vs 108.4), which
> has a third 48 MeV photon and so a different pairing in the flow tree. Quote
> one or the other, not both.

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
| CC π⁰ | [`e915a828`](https://www.phy.bnl.gov/twister/bee/set/e915a828-177b-49d8-9d06-a2aab3bcbc97/event/list/) | 99838, 242726 | mass 126.3 MeV, closes 1.000 |
| NC π⁰ | [`2b944df2`](https://www.phy.bnl.gov/twister/bee/set/2b944df2-7b85-4b3e-ba92-62263c36bf5f/event/list/) | 57709, 176986, 180801 | mass 147.9 MeV, `numu_score` −0.29 |
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
   (`verify_d86_bee.py`, 20/20, rc=0) and against the live set pages by counting
   `event/N/` links per set (2/2 ×7, 3/3 ×2). A 200 status code alone proves
   nothing — the viewer returns 200 for out-of-range indices too.
5. **The CC/NC π⁰ sets were rebuilt and re-uploaded once**, after §1.2's closure
   check disqualified the first CC pick. Their UUIDs above (`e915a828`,
   `2b944df2`) are the second upload; the first (`2af3970b`, `5e5b7a94`) still
   exists on the server and should **not** be used.

## 7. Artifacts

| path | what |
|---|---|
| `scripts/analysis/d86_video_picks.py` | pools, gates, feature table, final manifest |
| `scripts/bee/build_d86_bee.sh` | builds the nine zips |
| `scripts/bee/verify_d86_bee.py` | run/subrun/event + layer check per zip member |
| `docs/86_video/d86-final.tsv` | **the per-event record** — every number quoted above |
| `docs/86_video/d86-features.tsv` | all 1458 evaluated `(event, T_kine row)` rows |
| `docs/86_video/d86-set-*.txt` | **the nine pick lists the build script reads**, in Bee upload order |
| `docs/86_video/d86-<category>.txt` | the wider *candidate pools* the ranking produced — NOT what was uploaded (e.g. `d86-nuecc.txt` has 3 entries, `d86-set-nuecc.txt` the 2 that shipped). Rebuild from `d86-set-*` only. |
| `bee/d86/superseded-v1/` | the first ccpi0/ncpi0 build, before the §1.2 closure re-rank; kept so the earlier UUIDs stay explicable |
| `bee/d86/*.zip` | the nine uploaded zips (kept for rebuild) |

---

## 8. Video round 2 — what the next build needs (2026-08-30)

Three narrated Bee videos were built from these sets
(`wire-cell-bee3/bee-video/make_sbnd_{pr_chain,topologies,failures}.mjs`, see
`wire-cell-bee3/docs/sbnd-pattern-recognition-video.md`). Video 1, the
ν<sub>e</sub>CC pattern-recognition walkthrough on 81597, is finished. Videos 2
and 3 need re-cuts that this machine **cannot** build yet: the events the owner
asked for are in the `prod0830` feature table but have **no Bee set**, and the
grid work dirs (`work-*-grp0825` / `work-*-prod0830`) and `bee/d86/*.zip` do not
exist here — only the nine uploaded sets, mirrored read-only from the server.

Everything below is either **(N)** something needed from a machine with the
`prod0830` outputs, or **(Q)** a question whose answer changes the narration.

### 8.1 (N) Three Bee sets for events that already exist in `d86-features.tsv`

Same recipe as §Repro step 2 — `make_pr_bee.py` with the sample's `-q/-p`
roots, then upload and record the set UUID here. Layers needed:
`img-global`, `clustering-global`, `track_fit-global`, `shower_track-global`,
`vertices-global`, `mc`. **`op` is no longer needed** — the owner's round-2
direction drops all light/flash content from videos 2 and 3, so charge–light
matching appears only in video 1.

| for | run/sub/evt | sample (`-q/-p` root) | `n_part` | E<sub>ν</sub> | excluded | `numu` / `nue` | reconstructed final state |
|---|---|---|---:|---:|---:|---:|---|
| video 2 — two ν candidates, **replaces 487303** | 18259/1/**179054** | `mcp2k` | 11 / 5 (2 rows) | 1070.3 + 563.1 | 131.4 (12.3 %) / 0.0 | 0.81 / −15 | `proton 496 + neutron 205 + mu- 205 + pi+ 185 + proton 131 + e- 23` ‖ `mu- 263 + proton 173 + neutron 4 + proton 4` |
| video 3 — failure | 18259/1/**169626** | `mcp1k` | 6 | 638.2 | 142.2 (22.3 %) | −2.64 / **−10.07** | `gamma 511 + e- 511 + pi0 138 + gamma 102 + e- 102 + gamma 12` |
| video 3 — failure | 18255/1/**142421** | `ncpi0` | 19 | 2707.1 | 422.2 (15.6 %) | **3.54** / −15 | `gamma 672 + e- 672 + gamma 555 + e- 555 + pi+ 502 + mu- 350` |

Note 169626 carries a **real** ν<sub>e</sub> score (−10.07), not the −15
sentinel — the BDT ran and rejected it. That is a different failure mode from
§5.3's 69314, and worth keeping distinct in the narration.

**Status: DONE (this machine has the `prod0830` outputs).** Three local Bee
zips built with the recipe above and re-verified the same way
`verify_d86_bee.py` checks the original nine (run/subrun/event read back out
of each member's `img-global` layer):

| zip | bee index | run/sub/evt | layers |
|---|---:|---|---|
| `bee/d86r2/d86r2-video2.zip` | 0 | 18259/1/179054 | img-global, clustering-global, track_fit-global, shower_track-global, vertices-global, mc |
| `bee/d86r2/d86r2-video3-fail.zip` | 0 | 18259/1/169626 | same |
| `bee/d86r2/d86r2-video3-fail.zip` | 1 | 18255/1/142421 | same |

All 3/3 OK, rc=0. **Not yet uploaded** — upload is the owner-authorised step
(CLAUDE.md escalation rule 6), same as the original nine.

### 8.2 (N) The `kine_mcs_*` branches for the video events

`wct-pr-perevt.jsonnet:1234` sets `mcs_enable = true` (SBND production, doc 84
round 1, 2026-08-28), so `prod0830` should carry the five T_kine branches
`kine_mcs_energy`, `kine_mcs_ambiguity`, `kine_mcs_tracklen`,
`kine_mcs_range_energy`, `kine_mcs_segment_id`
(`NeutrinoTaggerInfo.h:105-109`). `d86_video_picks.py` never read them, so
they are **not** in `d86-features.tsv` and nothing here can quote them.

Needed: those five values (per T_kine row) for at least the two ν<sub>μ</sub>CC
events whose muons leave the detector — 18255/1/**290718** (μ 1138 MeV,
476 cm fitted) and 18255/1/**283591** (μ 621 MeV, 264 cm fitted) — and
ideally for every event in §4. Cheapest route is one more column block in
`d86_video_picks.py` and a re-emit of `d86-features.tsv`.

Why: the owner's direction is to quote MCS for the exiting muons instead of
leaving them as "the chain's estimate for an exiting track". `kine_mcs_energy`
is an MCS KE, and `kine_mcs_range_energy` is MCS's own CSDA range KE over the
**trimmed** path (doc 80 §8.3) — deliberately not `cal_kine_range`'s number, so
the two must not be presented as the same quantity.

**Status: DONE.** `mcs_output` is not its own knob — `cfg/pgrapher/experiment
/sbnd/clus.jsonnet:2263` sets `mcs_output=mcs_enable`, and `mcs_enable=true`
in this arm's `wct-pr-perevt.jsonnet:1234` — so the five branches are already
booked in every `prod0830` `tracking-pr.root`; nothing needed re-running.
`scripts/analysis/d86r2_video_round2.py` reads them for every §4 event's every
`T_kine` row → `docs/86_video/d86r2-mcs.tsv` (28 rows). The two named events:

| event | `kine_energy_particle` (muon, legacy) | `kine_mcs_energy` | ambiguity | trimmed len | `kine_mcs_range_energy` |
|---|---:|---:|---:|---:|---:|
| 290718 | 1138.7 MeV (range) | **1089.5 MeV** | 0.0019 | 495.2 cm | 1125.5 MeV |
| 283591 | 621.3 MeV (range) | **613.3 MeV** | 0.0147 | 268.9 cm | 614.9 MeV |

Both ambiguities are near 0 (unambiguous fit). The three numbers per muon
(legacy range-based `kine_energy_particle`, MCS's own `kine_mcs_energy`, and
MCS's own range estimate over its trimmed path) differ by 4–5 %, which is the
worked example §8's `kine_mcs_range_energy` note above warns not to blur
together on screen. MCS values exist for most of the other §4 events too
(not just exiting muons) — full table in `d86r2-mcs.tsv`.

### 8.3 (N + Q) A "very busy showers and tracks" event to open video 3

The owner ruled 389538 (§5.1) not busy enough. Ranking `d86-features.tsv` by
`n_part` with ≥2 electron-type objects and ≥1 muon or ≥2 protons gives 498
candidates; the head of that list, none of which has a Bee set:

| run/sub/evt | sample | `n_part` (e / μ / p) | E<sub>ν</sub> | excluded | `numu` / `nue` |
|---|---|---|---:|---:|---:|
| 18255/1/**318769** | `mcp2k` | 36 (30 / 1 / 4) | 807 | 310 (**38.5 %**) | 0.10 / −15 |
| 18255/1/**281781** | `mcp2k` | 34 (22 / 1 / 11) | 1007 | 5 (0.5 %) | 1.78 / −15 |
| 18261/1/**285567** | `ncpi0` | 34 (25 / 1 / 4) | 1821 | 182 (10.0 %) | 0.99 / −6.69 |
| 18259/1/**174771** | `mcp2k` | 31 (21 / 3 / 5) | 1802 | 394 (21.9 %) | 2.37 / −15 |
| 18259/1/**168526** | `mcp1k` | 31 (24 / 2 / 1) | 2410 | 265 (11.0 %) | 3.72 / −15 |

Recommendation: **318769** if the beat is meant to end on a failure (38.5 % of
the energy never consumed), **281781** if it is meant to show the chain coping
with a genuinely crowded event (11 protons, 0.5 % excluded). Two sets would let
the visual choice be made on the picture rather than on the table. Note
`n_part` counts particle-flow objects, which is a proxy for busy, not a
measurement of it — the picture is the arbiter.

**Status: (N) DONE, (Q) still the owner's call.** Both candidates built as one
local Bee set — `bee/d86r2/d86r2-video3-busy.zip`, index 0 = 18255/1/318769,
index 1 = 18255/1/281781, both verified (layers + run/sub/evt) OK. Not
uploaded pending owner authorisation. Which one opens video 3 is a visual
judgment on the picture, as recommended above — not resolved here.

### 8.4 (Q) Open questions for the round-2 cut

1. **Does video 3 keep 69314?** The direction reads as "replace 389538 with a
   busier event, then use 169626 and 142421" — silent on whether §5.3's
   never-merged EM shower stays. Default taken: keep it, since it is the only
   event where the −15 sentinel is the whole story.
2. **Which candidate owns the BDT score in a two-candidate event?** **Answered
   — 0.81 belongs to candidate 1 only; it was never candidate 2's score to
   begin with.** `T_tagger` carries `numu_score`/`nue_score` **per bundle**
   (one row per `nu_index`); `pr_scores_table.py` (via `pr94_rows.primary_index()`)
   collapses that to ONE row per event — "the longest selected main activity" —
   for backward compatibility with the pre-pr/94 single-candidate schema, and
   `d86-features.tsv` inherited that single value. Reading `T_tagger` directly
   (`scripts/analysis/d86r2_video_round2.py` → `docs/86_video/d86r2-candidates.tsv`)
   gives both candidates their **own** score:

   | event | candidate | `nu_index` | selected len | `numu_score` | clears 0.9 WP? |
   |---|---|---:|---:|---:|---|
   | 179054 | 1 (primary) | 0 | 130.9 cm | **0.81** | no |
   | 179054 | 2 | 1 | 110.5 cm | **4.24** | **yes** |
   | 487303 (§3.7, for comparison) | 1 (primary) | 0 | 126.0 cm | 4.29 | yes |
   | 487303 | 2 | 1 | 21.7 cm | −0.04 | no |
   | 174661 (§3.7) | 1 (primary) | 0 | 135.2 cm | 3.53 | yes |
   | 174661 | 2 | 1 | 130.1 cm | 2.91 | yes |

   So for 179054 the picture inverts what the single scalar suggested: it is
   candidate 2 (μ⁻ 263 + proton 173, the simpler final state) that is the
   confident ν<sub>μ</sub>CC-like read, and candidate 1 (the busier proton +
   π⁺ + μ⁻ + fragments state, 1070 MeV) that the BDT is unsure about. Do not
   caption 0.81 next to both objects on screen.
3. **Is a per-candidate flash still wanted anywhere in video 2?** The "two
   separate flash-matched bundles" line was the strongest evidence that the two
   179054 candidates are not one interaction split in half, and it is light
   information. Dropping light entirely costs that argument. **Data point, not
   a resolution:** 179054's two candidates carry `gid 1` and `gid 1000005` in
   the particle-flow tree — same pattern as 487303's `gid 1000005`/`gid 6`, so
   the argument is available if wanted. Whether to spend a beat on it is still
   the owner's call.

### 8.5 What is *not* needed — already verified locally

- **NC π⁰ 57709's proton (`13001`) dQ/dx.** The owner asked whether the
  short prong is a proton and whether dQ/dx shows it. It does, textbook: 148
  fitted points over 85 cm from the vertex, mean dQ/dx rising monotonically
  3521 → 3974 → 4241 → 4460 → 4961 → 6021 → 7627 → **11711** in 11 cm bands.
  That beat can be built with the set already mirrored.
- **The cathode crosser's muon-vs-proton contrast** (14001 flat at 2348 mean
  over 264 cm; 14002 rising 4356 → 8869 over 8 cm) and **290718's muon**
  (8000, 831 points, 476 cm, flat at 2311) — both already used in video 2.
- **Video 1's energy sum.** E<sub>ν</sub> is a sum, not a measurement
  (`NeutrinoKinematics.cxx:96`, `NeutrinoPatternBase.h:112`): electron adds
  nothing, a nucleon adds the 8.6 MeV average binding energy, μ/π/K add their
  rest mass. For 81597 that is 1362.9 (e⁻, from `kine_energy_particle`) +
  139.3 (proton KE) + 8.6 = **1510.8**, which is why the flow tree's rounded
  `e- 1362` and `proton 139` do not visibly add up.

### 8.6 (N) Neutrino-energy reconstruction — the per-object breakdown

Video 1 now ends on E<sub>ν</sub> explained as a sum rather than quoted as a
number, which is as far as `d86-features.tsv` allows: it carries only `Enu`,
`e_e` (the largest electron-type KE) and `e_mu` (the largest muon). For the
1e1p event 81597 that is enough — the proton's KE is what the sum leaves,
139.33, and it round-trips to the flow tree's printed `proton 139`. For every
other event in §4 it is not: with 5, 11 or 19 objects there is no way to
decompose E<sub>ν</sub> from the table at all.

Needed, per `(sample, event, T_kine row)`, for the §4 events — the four
**parallel** vectors plus the scalars (`NeutrinoTaggerInfo.h:36-39`):

| branch | what |
|---|---|
| `kine_particle_type` | PDG code per object |
| `kine_energy_particle` | kinetic energy [MeV] per object, full precision |
| `kine_energy_info` | **0 = dQ/dx, 1 = range, 2 = charge** — how that object's KE was measured |
| `kine_energy_included` | 1 = this row entered the E<sub>ν</sub> sum |
| `kine_reco_Enu`, `kine_energy_excluded`, `kine_energy_excluded_main/_other`, `kine_n_excluded` | already in the table, repeated here so a re-emit is one block |

`kine_energy_info` is the interesting one and is not in the table at all. It
turns the energy card from arithmetic into a statement about *method* — "the
electron's energy is calorimetric, from charge; the proton's is from its
range; the exiting muon's is neither, which is why §8.2's MCS number matters".
That is the single most useful addition to video 1's closing beats.

Two further questions on the same subject:

1. **Is `kine_shower_fudge_factor = 0.84` (§1, build `d9814518`) already inside
   `kine_energy_particle` for EM objects, or applied downstream?** The card
   would be wrong to call 1362.9 MeV "the charge, converted" if a 0.84 scaling
   sits between the two unstated.
2. **Is the 8.6 MeV binding term visible anywhere in the branches, or only in
   the difference?** The video derives it from `NeutrinoKinematics.cxx:96`;
   confirming that `sum(kine_energy_particle[included]) + rest terms` closes to
   `kine_reco_Enu` on 81597 would let the card be built from the branches
   rather than from a subtraction.

**Status: DONE — both answered, with code citations and a numeric check.**
`scripts/analysis/d86r2_video_round2.py` re-emits the four vectors +
`kine_reco_add_energy` for every §4 event's every `T_kine` row to
`docs/86_video/d86r2-objects.tsv` (313 rows) and `docs/86_video/d86r2-mcs.tsv`
(28 rows, one per (event, krow), also carrying `kine_reco_Enu`,
`kine_reco_add_energy` and their sum).

1. **Yes, baked in — there is no unstated 0.84 sitting between the two.**
   `push_shower_kine()` (`NeutrinoKinematics.cxx:204-210`) stores
   `shower->get_kine_best()/MeV` into `kine_energy_particle`.
   `PRShower::get_kine_best()` (`PRShower.h:154-155`) falls back to
   `kenergy_charge` whenever `kenergy_best == 0` (true for a plain EM shower —
   nothing upstream sets `kenergy_best` for it). `kenergy_charge` is whatever
   `PatternAlgorithms::cal_kine_charge()` returned, and that function
   (`NeutrinoEnergyReco.cxx:188`) divides by `shower_fudge_factor` **before**
   returning: `return overall / recom_factor / fudge_factor * kopts.w_value / 1e6`.
   So a `kine_energy_info == 2` ("charge") row already has the fudge factor
   applied at storage time. Confirmed on 81597: the 1362.9 MeV electron is
   `kine_energy_info == 2`, i.e. it *is* `cal_kine_charge()`'s return value,
   fudge and all.
2. **Visible, but only as one event-level scalar — never split per object.**
   `fill_kine_tree()` accumulates `ktree.kine_reco_add_energy` across every
   pushed particle via `rest_term_rules(pdg, mass)` (0 for e⁻, the 8.6 MeV
   `ave_binding_energy` for p/n, rest mass for μ/π/K —
   `NeutrinoKinematics.cxx:96,102-107`), and at the end
   (`NeutrinoKinematics.cxx:889-893`) `kine_reco_Enu = Σ kine_energy_particle
   + kine_reco_add_energy`, **unconditional of `kine_energy_included`** — the
   struct's own comment at `NeutrinoTaggerInfo.h:59-64` says so explicitly, so
   the round-2 question's `[included]` above is not quite how the sum works;
   this script's closure check therefore sums *every* row, not just the
   included ones. Checked, not assumed: **28/28 (event, krow) rows close**
   (`sum(kine_energy_particle) + kine_reco_add_energy == kine_reco_Enu` to
   float precision). On 81597: `1362.865 + 139.332 = 1502.197`, `+8.600 =
   1510.797` = the stored `kine_reco_Enu` exactly. `kine_reco_add_energy` is a
   plain scalar `T_kine` branch (`NeutrinoTaggerInfo.h:33`) — the card can read
   it directly ("148.2 MeV of rest mass + binding energy" for 179054's
   candidate 1) but cannot break it down further into which object contributed
   how much; that attribution does not exist in the output.

---

## 9. Round-2 delivery (2026-08-30, later same day)

Everything in §8 tagged **(N)** is done; §8.4's three **(Q)** items are
answered where the answer is a fact (Q2), supplied with new data where the
answer is a creative call (Q3's flash gids), or left exactly as posed where it
genuinely is one (Q1's default, §8.3's visual pick). No arm was re-run, no
`clus`/`root` code was touched — this is the same "read what `prod0830`
already produced" scope as §§1-8.

**Machine note for whoever reads this next.** §8 was written on a machine
that mirrors only the nine uploaded Bee sets read-only; this update was made
on `wcgpu1.phy.bnl.gov`, which holds the full `work-<sample>-grp0825` /
`work-<sample>-prod0830` trees §8 said were missing. If a future round hits
the same "(N), needs a machine with the outputs" wall, that is the machine to
use.

### 9.1 What is still not done

* **The five new Bee zips are local only, not uploaded.** Upload is the
  owner-authorised step (CLAUDE.md escalation rule 6) — the same gate the
  original nine went through (§ Repro block step 3). Awaiting authorisation.
* **§8.3's visual pick (318769 vs 281781)** and **§8.4 Q1 (keep 69314?)** and
  **Q3 (spend a video-2 beat on the per-candidate flash?)** are editorial
  calls on the picture/cut, not reconstruction questions — not resolved here,
  by design.
* **How the five new events fold into video 2's/video 3's final Bee-set
  membership** (replace vs. append vs. new standalone sets) is also an
  editorial call the round-2 note did not specify beyond "replaces 487303" for
  179054. The zips below are built standalone, one per §8 ask, so that
  question stays open rather than pre-decided.

### 9.2 Artifacts (round 2)

| path | what |
|---|---|
| `scripts/analysis/d86r2_video_round2.py` | reads MCS + per-object kine branches + per-bundle `T_tagger` scores for every §4 event and the five new events |
| `docs/86_video/d86r2-mcs.tsv` | `kine_mcs_*` + `kine_reco_add_energy` + the closure check, one row per (event, krow) |
| `docs/86_video/d86r2-objects.tsv` | `kine_particle_type`/`kine_energy_particle`/`kine_energy_info`/`kine_energy_included`, one row per (event, krow, object) |
| `docs/86_video/d86r2-candidates.tsv` | `T_tagger` per-bundle `numu_score`/`nue_score`, every row (not just `primary_index()`'s pick) |
| `docs/86_video/d86r2-set-video2.txt`, `-video3-fail.txt`, `-video3-busy.txt` | the three new pick lists |
| `bee/d86r2/d86r2-video2.zip` | 179054 (video 2, §8.1) |
| `bee/d86r2/d86r2-video3-fail.zip` | 169626, 142421 (video 3 failures, §8.1) |
| `bee/d86r2/d86r2-video3-busy.zip` | 318769, 281781 (video 3 busy-event candidates, §8.3) |

Repro:

```bash
cd wcp-porting-img/sbnd/sbnd_xin
python3 scripts/analysis/d86r2_video_round2.py     # -> docs/86_video/d86r2-*.tsv
python3 scripts/bee/make_pr_bee.py -q work-mcp2k-grp0825 -p work-mcp2k-prod0830 \
    -o bee/d86r2/d86r2-video2.zip 179054
python3 scripts/bee/make_pr_bee.py -q work-mcp1k-grp0825 -q work-ncpi0-grp0825 \
    -p work-mcp1k-prod0830 -p work-ncpi0-prod0830 \
    -o bee/d86r2/d86r2-video3-fail.zip 169626 142421
python3 scripts/bee/make_pr_bee.py -q work-mcp2k-grp0825 -p work-mcp2k-prod0830 \
    -o bee/d86r2/d86r2-video3-busy.zip 318769 281781
# verify (adapt scripts/bee/verify_d86_bee.py's per-member run/sub/evt check;
# all three zips read back 18259/1/179054, {18259/1/169626, 18255/1/142421},
# {18255/1/318769, 18255/1/281781} respectively, 5/5 OK)
```
