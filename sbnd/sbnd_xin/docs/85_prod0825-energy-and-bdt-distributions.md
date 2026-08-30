# 85 — Reconstructed energy and BDT scores of the four SBND production samples, the MicroBooNE working points, and six Bee hand-scan sets

Three figures over the **whole** `prod0825` PR product (3067 events, four
samples), the MicroBooNE νμ/νe BDT working points applied to them (§7 — the νe
one does **not** transfer, and why), and six Bee sets.

**Nothing is tuned here and no code changed.** This is a read-out of an existing
production arm: `pr_scores_table.py` over `work-<sample>-prod0825`, then
matplotlib. No knob moved, no A/B was run, no arm was regenerated.

## Repro block

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# 1. the per-event score tables (T_tagger / T_kine scalars + nusel labels)
mkdir -p products/prod0825
for s in nuecc48 ncpi0 mcp1k mcp2k; do
  python3 pr_scores_table.py --root work-$s-prod0825 --sample $s \
      --out products/prod0825/$s-scores-prod0825.tsv
done

# 2. both figures, the summary table, and the three Bee pick lists
python3 scripts/analysis/d85_dists.py          # -> docs/85_dists/

# 3. the Bee zips (LOCAL; the upload below is a separate owner authorisation)
mkdir -p bee/d85
for c in cosmiclike nuelike neither; do
  python3 scripts/bee/make_pr_bee.py \
      -q work-mcp1k-grp0825   -q work-mcp2k-grp0825 \
      -p work-mcp1k-prod0825  -p work-mcp2k-prod0825 \
      -o bee/d85/d85-$c.zip $(cat docs/85_dists/d85-$c.txt)
  ./upload-to-bee.sh bee/d85/d85-$c.zip        # owner-authorised 2026-08-30
done

# 4. sec 7: the MicroBooNE working points, its figure and its three Bee sets
python3 scripts/analysis/d85b_cuts.py         # -> docs/85_dists/d85b-*
mkdir -p bee/d85b
python3 scripts/bee/make_pr_bee.py -q work-nuecc48-grp0825 -p work-nuecc48-prod0825 \
    --allow-unevaluated -o bee/d85b/d85b-nuecc-failnue.zip \
    $(cat docs/85_dists/d85b-nuecc-failnue.txt)
for c in ncpi0-numupass ncpi0-nearmiss; do
  python3 scripts/bee/make_pr_bee.py -q work-ncpi0-grp0825 -p work-ncpi0-prod0825 \
      -o bee/d85b/d85b-$c.zip $(cat docs/85_dists/d85b-$c.txt)
done

# 5. the fudge-flip caveat of sec 4, measured not assumed
for a in off98 onfudge98; do for s in nuecc48 ncpi0 mcp1k mcp2k; do
  python3 pr_scores_table.py --root work-pr132-$a-$s --sample $s \
      --out /home/xqian/tmp/d85/$a-$s.tsv
done; done
```

## 1. The samples, and what "the ones with PR" means

All four are **data** selections — there is no truth label anywhere in this
document, so "NCπ⁰" and "ν<sub>e</sub>CC" name the *sample selection*, not the
per-event interaction, and the two ν<sub>μ</sub> samples are beam data, not a
ν<sub>μ</sub>CC-truth subset.

| sample | source | events | provenance |
|---|---|---:|---|
| `ncpi0` | `input_files_reco1/extracted-ncpi0` | 19 | NC π⁰ sideband (doc 71 §3) |
| `nuecc48` | `extracted-2025fall-48evt-fsprod` | 48 | ν<sub>e</sub>CC-selected 2025-fall data (doc 71, doc pr/115) |
| `mcp1k` | `staged-mcp2025c-1000evt` | 1000 | MCP2025C reco1, first 1000 entries (doc 59) |
| `mcp2k` | `staged-mcp2025c-2nd-2000evt` | 2000 | MCP2025C reco1, 2nd 2000 entries |

**Population = `nu_evaluated == 1`.** That flag is set iff the PR log carries
`TaggerCheckNeutrino: selected main cluster …`, i.e. iff `TaggerInfo`/`KineInfo`
were filled for real. Anything else holds struct *defaults* run through the
BDTs and is blanked by `pr_scores_table.py` rather than reported — plotting it
would manufacture a spike at the sentinel values.

| | ncpi0 | nuecc48 | mcp1k | mcp2k |
|---|---:|---:|---:|---:|
| events in sample | 19 | 48 | 1000 | 2000 |
| **PR-evaluated** | **18** | **47** | **447** | **879** |
| …of which `cosmic-tagged` | 0 | 1 | 36 | 74 |
| − degenerate (§1.1) | 0 | 0 | 16 | 33 |
| − no `tracking-pr.root` row at all | 0 | 0 | 2 | 12 |
| **plotted** | **18** | **47** | **429** | **834** |

Three honest caveats on that table:

* the evaluated population is **not** a neutrino-purity cut — 110 of the 1326
  evaluated ν<sub>μ</sub>-sample events carry the chain's own `cosmic-tagged`
  event label, and they are in the histograms.
* 14 evaluated events (2 mcp1k, 12 mcp2k) have no `T_tagger`/`T_kine` values at
  all — not a blank energy field but a missing ROOT row; 8 of the 14 are
  cosmic-tagged. Counted as evaluated, absent from both figures.
* 49 more are degenerate — next section. They are what would otherwise make
  the first energy bin and one spike of the score plot look like physics.

### 1.1 The degenerate class: 49 evaluated events with no reconstruction

`TaggerCheckNeutrino` emits its `selected main cluster …` line — so
`nu_evaluated = 1` — even when the main it picked is an **unmerge shard a
couple of cm long**. `unmerge_bundle`/`unmerge_assoc` run *before* the tagger
and mint PR-internal cluster idents, so the tagger can select one of those
(this is exactly the non-joinability `pr_scores_table.py`'s own docstring
warns about). `KineInfo` then never fills — and, unlike the
`nu_evaluated == 0` case, **nothing blanks the row**.

Three signatures coincide *exactly* on these events:

| signature | value |
|---|---|
| `kine_reco_Enu` | `0.0` exactly |
| `nu_x/y/z_cm` | `(0, 0, 0)` exactly |
| `numu_score` | `-1.9416895` — the *same* value for every one of them (the BDT on a default feature vector) |

They number **16 mcp1k + 33 mcp2k = 49**, and **0 in ncpi0/nuecc48**. Median
selected-main length is 1.5–1.8 cm (max 52 cm); 47 of the 49 are
`cosmic-tagged`.

**They are excluded from both figures**, by the energy+vertex test (not by
hard-coding the score value) in `d85_dists.py:degenerate()`. Left in, all 49
would sit in the 0–200 MeV bin — a quarter of the mcp1k first bin — and in a
single 49-event spike at `numu_score = -1.94`. That is a reconstruction
failure mode worth its own investigation; it is not an energy spectrum.

## 2. Reconstructed neutrino energy

![reconstructed Enu](85_dists/d85_enu.png)

`kine_reco_Enu` (MeV) from `T_kine`, four samples overlaid. **Left** is
area-normalised — this is the panel to read for shape, because 18 against 834
events on a common count axis makes two of the four samples invisible.
**Right** is raw counts on a log *y*. The bin at 3000–3200 MeV is an explicit
**overflow** bin, not a clip: 1 nuecc48, 1 mcp1k and 3 mcp2k events live there
(max 4654 MeV). The 49 degenerate events of §1.1 are out.

| | ncpi0 | nuecc48 | mcp1k | mcp2k |
|---|---:|---:|---:|---:|
| N | 18 | 47 | 429 | 834 |
| min / median / max [MeV] | 126 / **1401** / 2768 | 428 / **1578** / 3948 | 11 / **575** / 3583 | 3 / **530** / 4654 |

The two ν<sub>μ</sub>-beam samples land on top of each other (median 575 vs
530 MeV) — expected, they are consecutive slices of the same MCP2025C data, and
their agreement is the only internal consistency check this figure offers. The
two hand-selected samples sit a factor 2.5–3 higher and are much flatter. That
is a **selection** statement, not a physics one: `ncpi0` and `nuecc48` were
picked as π⁰/EM-rich events, and the ncpi0 curve is 18 events across 15 bins,
so every structure in it is within its own Poisson noise.

## 3. ν<sub>e</sub> BDT score vs ν<sub>μ</sub> BDT score

![nue vs numu BDT](85_dists/d85_bdt.png)

Three features of this plot are real and would otherwise read as defects:

1. **`nue_score = -15` is not missing data.** `UbooneNueBDTScorer.cxx:1991`
   assigns it when `br_filled != 1` — "background-like default; matches
   prototype `default_val = -15`". The nue BDT was never evaluated for those
   events. It is the *majority* of the ν<sub>μ</sub> samples, so it gets its own
   bottom panel (their `numu_score`, projected) instead of crushing the top
   panel's scale.
2. **Both scores saturate at ±4.301.** Both are
   `log10((1+v)/(1-v))` of an xgboost output clamped to ±0.9999. The rows of
   points sitting exactly on the dashed frame are that clamp — and that clamp
   is a toolkit addition the prototype does not have, which is a bigger deal
   than a plotting artefact: see **§7.1**.
3. **`cosmic_flag` is not usable as a discriminator here** and is deliberately
   absent from the figure: it is 1 for 1307 of the 1326 evaluated
   ν<sub>μ</sub>-sample events (it is the struct default, and nothing in the
   scored path overwrites it). The live cosmic-side handle is the nusel
   `event_label`, which is what section 5 uses.

| | ncpi0 | nuecc48 | mcp1k | mcp2k |
|---|---:|---:|---:|---:|
| plotted (both scores, non-degenerate) | 18 | 47 | 429 | 834 |
| **nue BDT filled** (`> -15`) | **3** | **44** | **20** | **71** |
| median `numu_score` | +0.57 | −0.45 | +1.79 | +1.59 |
| at the `numu_score` clamp | 0 | 1 | 31 | 70 |

The separation the plot is for: the ν<sub>e</sub>CC sample fills the nue BDT
**44/47** and piles up at `nue_score = +4.30`, while the same BDT fills for only
**20/429** and **71/834** of the ν<sub>μ</sub>-sample events. Median
`numu_score` runs the other way (+1.79/+1.59 for the ν<sub>μ</sub> samples,
−0.45 for ν<sub>e</sub>CC). NCπ⁰ fills the nue BDT 3/18 and all three land
*negative* (−3.21 … −1.29) — on this sample of 18 the nue BDT is not being fooled
by the π⁰.

## 4. Caveat that matters: prod0825 predates the 0.84 fudge flip

`prod0825` was produced 2026-08-25. `kine_shower_fudge_factor` flipped
**0.80 → 0.84** in SBND production on 2026-08-29 (doc pr/132 §1, commit
`d9814518`). Since `E = … / recom / fudge`, current production returns
*slightly lower* shower energy than what is plotted above. Measured, per sample,
on the pr/132 98-event arms (`work-pr132-off98-*` = 0.80 vs
`work-pr132-onfudge98-*` = 0.84):

| sample | events | median Δ | range |
|---|---:|---:|---|
| nuecc48 | 48 | **−3.9 %** | −4.8 … −0.4 % |
| ncpi0 | 19 | **−3.5 %** | −4.4 … −0.3 % |
| mcp1k (98-subset) | 14 | **−3.4 %** | −32.4 … −0.9 % |
| mcp2k (98-subset) | 17 | **−3.0 %** | −4.7 … −0.8 % |

So the shift is a near-uniform few-percent compression, comparable across all
four samples — it does **not** distort the sample-to-sample comparison in §2,
and it is far below the 2.5–3× median separation between the hand-selected and
beam samples. Two things to keep straight:

* the ceiling is the naive `0.80/0.84 = ×0.952` (−4.8 %); events land above it
  in proportion to how much of their energy is shower-flagged.
* one event moved far more than scaling can explain — **mcp1k evt 64591,
  944.5 → 638.5 MeV (−32.4 %)**. That is the documented *indirect* coupling
  (pi0 acceptance → `calculate_kinematics` recompute → `kine_charge`,
  pr/126 §2.6/§4f), not a scale factor. It is one event of 98.
* the mcp1k/mcp2k rows are the EM-enriched 98-event scan subset, not a random
  draw from the 1000/2000, so treat them as an order of magnitude rather than
  the full-sample number.

Re-running the 3067-event chain at 0.84 was not done: at order 1 min/event it
is ~10 h at the standing job caps, which is not a reasonable cost for a
distribution plot. (That rate is a spot check, not a measured mean — prod0825
was pruned and only one of its per-event logs still carries its `Timer: Total`
line, at 77 s.) If the owner wants the figures at the exact current operating point, that
is the price and it should be a fresh tag (`prod0830`), never a write into
`prod0825` (M13).

## 5. Three Bee hand-scan sets from the ν<sub>μ</sub> samples

Drawn from the **1263** `mcp1k`+`mcp2k` events that are PR-evaluated, carry both
scores, and are not degenerate (§1.1) — the same population as the figures.
Event ids do not collide between mcp1k and mcp2k (checked: 0 overlaps), so each
category is one Bee set spanning both.

**These are rank-order cuts, not thresholds.** There is no documented
`numu_score` / `nue_score` operating point anywhere in these docs, so each
category is "the 10 most extreme by the named score" within its pool. Say so
before quoting any of them as a selection.

Charge cloud = `work-mcp{1k,2k}-grp0825` (Q/L), PR layers =
`work-mcp{1k,2k}-prod0825` — the same arm as both figures.

### 5.1 Cosmic-like — [Bee set 7b4ee14a](https://www.phy.bnl.gov/twister/bee/set/7b4ee14a-13c3-4a66-b1ff-abf0034c1df8/event/list/)

Pool: the **55** plotted events the chain itself labels `cosmic-tagged`
(in-beam bundle tagged TGM/STM/LM); picked the 10 lowest `numu_score`, i.e.
where the tagger and the BDT agree. That 55 is what is left of the **931**
cosmic-tagged events in the two samples: 931 → 110 evaluated (**821** never got
a selected main cluster at all — no scores; they would need
`--allow-unevaluated` to reach Bee) → 63 non-degenerate (**47** are the §1.1
shard class, unsurprisingly concentrated here) → **55** with a ROOT row (8 lack
one). So this set is the cosmic-like events the chain actually *scored*, not
the cosmic population.

| Bee | sample | run | event | numu | nue | E [MeV] | link |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | mcp2k | 18259 | 172262 | −3.588 | −15 | 347 | [event/0](https://www.phy.bnl.gov/twister/bee/set/7b4ee14a-13c3-4a66-b1ff-abf0034c1df8/event/0/) |
| 1 | mcp1k | 18255 | 401450 | −3.575 | +0.982 | 1609 | [event/1](https://www.phy.bnl.gov/twister/bee/set/7b4ee14a-13c3-4a66-b1ff-abf0034c1df8/event/1/) |
| 2 | mcp2k | 18259 | 172012 | −3.359 | −15 | 462 | [event/2](https://www.phy.bnl.gov/twister/bee/set/7b4ee14a-13c3-4a66-b1ff-abf0034c1df8/event/2/) |
| 3 | mcp2k | 18255 | 319275 | −3.168 | −15 | 514 | [event/3](https://www.phy.bnl.gov/twister/bee/set/7b4ee14a-13c3-4a66-b1ff-abf0034c1df8/event/3/) |
| 4 | mcp2k | 18255 | 75113 | −2.850 | −15 | 200 | [event/4](https://www.phy.bnl.gov/twister/bee/set/7b4ee14a-13c3-4a66-b1ff-abf0034c1df8/event/4/) |
| 5 | mcp2k | 18255 | 91609 | −2.653 | −15 | 49 | [event/5](https://www.phy.bnl.gov/twister/bee/set/7b4ee14a-13c3-4a66-b1ff-abf0034c1df8/event/5/) |
| 6 | mcp2k | 18255 | 414442 | −2.441 | −15 | 427 | [event/6](https://www.phy.bnl.gov/twister/bee/set/7b4ee14a-13c3-4a66-b1ff-abf0034c1df8/event/6/) |
| 7 | mcp2k | 18255 | 73038 | −2.243 | −15 | 51 | [event/7](https://www.phy.bnl.gov/twister/bee/set/7b4ee14a-13c3-4a66-b1ff-abf0034c1df8/event/7/) |
| 8 | mcp2k | 18255 | 287332 | −2.211 | −15 | 284 | [event/8](https://www.phy.bnl.gov/twister/bee/set/7b4ee14a-13c3-4a66-b1ff-abf0034c1df8/event/8/) |
| 9 | mcp1k | 18255 | 288327 | −1.961 | −15 | 523 | [event/9](https://www.phy.bnl.gov/twister/bee/set/7b4ee14a-13c3-4a66-b1ff-abf0034c1df8/event/9/) |

Bee 1 (evt 401450) is the odd one and worth a look first: cosmic-tagged, most
cosmic-like `numu_score` in the whole pool, and yet the nue BDT filled and
returned **+0.98** at 1609 MeV.

### 5.2 ν<sub>e</sub>CC-like inside the ν<sub>μ</sub> samples — [Bee set 7dfadf68](https://www.phy.bnl.gov/twister/bee/set/7dfadf68-bfe2-4d7c-8391-70a1d6b13912/event/list/)

Pool: the **1204** `nu-candidate` events; picked the 10 highest `nue_score`.
Six of the ten are *at* the +4.301 clamp, so their internal order is arbitrary —
they are tied, not ranked.

| Bee | sample | run | event | numu | nue | E [MeV] | link |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | mcp1k | 18255 | 64409 | +0.091 | **+4.301** | 1077 | [event/0](https://www.phy.bnl.gov/twister/bee/set/7dfadf68-bfe2-4d7c-8391-70a1d6b13912/event/0/) |
| 1 | mcp2k | 18255 | 75954 | −0.587 | **+4.301** | 753 | [event/1](https://www.phy.bnl.gov/twister/bee/set/7dfadf68-bfe2-4d7c-8391-70a1d6b13912/event/1/) |
| 2 | mcp2k | 18259 | 176502 | −1.589 | **+4.301** | 4600 | [event/2](https://www.phy.bnl.gov/twister/bee/set/7dfadf68-bfe2-4d7c-8391-70a1d6b13912/event/2/) |
| 3 | mcp2k | 18259 | 176533 | −0.515 | **+4.301** | 688 | [event/3](https://www.phy.bnl.gov/twister/bee/set/7dfadf68-bfe2-4d7c-8391-70a1d6b13912/event/3/) |
| 4 | mcp2k | 18255 | 291072 | −0.276 | **+4.301** | 830 | [event/4](https://www.phy.bnl.gov/twister/bee/set/7dfadf68-bfe2-4d7c-8391-70a1d6b13912/event/4/) |
| 5 | mcp2k | 18255 | 394167 | −0.738 | **+4.301** | 482 | [event/5](https://www.phy.bnl.gov/twister/bee/set/7dfadf68-bfe2-4d7c-8391-70a1d6b13912/event/5/) |
| 6 | mcp1k | 18255 | 386442 | −1.371 | +4.055 | 267 | [event/6](https://www.phy.bnl.gov/twister/bee/set/7dfadf68-bfe2-4d7c-8391-70a1d6b13912/event/6/) |
| 7 | mcp2k | 18255 | 281567 | −0.453 | +1.604 | 1897 | [event/7](https://www.phy.bnl.gov/twister/bee/set/7dfadf68-bfe2-4d7c-8391-70a1d6b13912/event/7/) |
| 8 | mcp2k | 18255 | 396222 | −0.795 | +1.321 | 4654 | [event/8](https://www.phy.bnl.gov/twister/bee/set/7dfadf68-bfe2-4d7c-8391-70a1d6b13912/event/8/) |
| 9 | mcp2k | 18259 | 47212 | −0.480 | +0.852 | 1342 | [event/9](https://www.phy.bnl.gov/twister/bee/set/7dfadf68-bfe2-4d7c-8391-70a1d6b13912/event/9/) |

Bee 2 and 8 are also the two highest-energy events in the entire ν<sub>μ</sub>
population (4600 and 4654 MeV) — worth checking whether the ν<sub>e</sub>-like
score and the energy have a common cause. Bee 9 (evt 47212) is the pr/120
backward-stem-guard event, already hand-scanned in that round.

### 5.3 ν<sub>μ</sub>-sample events that are neither — [Bee set f6c75e50](https://www.phy.bnl.gov/twister/bee/set/f6c75e50-8f04-4283-9ad5-ddac614add75/event/list/)

Pool: the **1120** `nu-candidate` events whose nue BDT never filled
(`nue_score = -15`); picked the 10 lowest `numu_score` — i.e. accepted as a
neutrino candidate by the chain, then rejected by the ν<sub>μ</sub> BDT and never
even offered to the ν<sub>e</sub> one. The first four are at the −4.301 clamp
and so are tied.

| Bee | sample | run | event | numu | nue | E [MeV] | link |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | mcp1k | 18255 | 391766 | **−4.301** | −15 | 837 | [event/0](https://www.phy.bnl.gov/twister/bee/set/f6c75e50-8f04-4283-9ad5-ddac614add75/event/0/) |
| 1 | mcp2k | 18255 | 99563 | **−4.301** | −15 | 1348 | [event/1](https://www.phy.bnl.gov/twister/bee/set/f6c75e50-8f04-4283-9ad5-ddac614add75/event/1/) |
| 2 | mcp2k | 18259 | 160031 | **−4.301** | −15 | 947 | [event/2](https://www.phy.bnl.gov/twister/bee/set/f6c75e50-8f04-4283-9ad5-ddac614add75/event/2/) |
| 3 | mcp2k | 18259 | 180698 | **−4.301** | −15 | 732 | [event/3](https://www.phy.bnl.gov/twister/bee/set/f6c75e50-8f04-4283-9ad5-ddac614add75/event/3/) |
| 4 | mcp2k | 18255 | 99284 | −4.123 | −15 | 13 | [event/4](https://www.phy.bnl.gov/twister/bee/set/f6c75e50-8f04-4283-9ad5-ddac614add75/event/4/) |
| 5 | mcp1k | 18255 | 62613 | −4.123 | −15 | 805 | [event/5](https://www.phy.bnl.gov/twister/bee/set/f6c75e50-8f04-4283-9ad5-ddac614add75/event/5/) |
| 6 | mcp2k | 18255 | 70766 | −3.928 | −15 | 660 | [event/6](https://www.phy.bnl.gov/twister/bee/set/f6c75e50-8f04-4283-9ad5-ddac614add75/event/6/) |
| 7 | mcp2k | 18255 | 93234 | −3.924 | −15 | 29 | [event/7](https://www.phy.bnl.gov/twister/bee/set/f6c75e50-8f04-4283-9ad5-ddac614add75/event/7/) |
| 8 | mcp2k | 18259 | 173450 | −3.800 | −15 | 788 | [event/8](https://www.phy.bnl.gov/twister/bee/set/f6c75e50-8f04-4283-9ad5-ddac614add75/event/8/) |
| 9 | mcp2k | 18255 | 321235 | −3.662 | −15 | 465 | [event/9](https://www.phy.bnl.gov/twister/bee/set/f6c75e50-8f04-4283-9ad5-ddac614add75/event/9/) |

Bee 4 and 7 carry `kine_reco_Enu` of **13** and **29 MeV** while still being
accepted as neutrino candidates — the two cheapest checks in this set for
whether the candidate is real at all.

## 7. The MicroBooNE BDT working points applied here

### 7.1 The νe working point does not transfer — the toolkit clamps the score

The two working points, as this repo records them
(`toolkit/clus/docs/tagger/tagger_validation_plan.md:229`, "representative
working points, e.g. nue_score > 7.0, numu_score > 0.9"), plus the cosmic
verdict `cosmict_flag == 0` (the OR of the ten cosmic tests,
`NeutrinoTaggerCosmic.cxx:1384`; **`cosmic_flag` is not this** — it is a BDT
input equal to `!cosmict_flag_9`, and it reads 1 on 1307 of 1326 events, §3).

Prototype and toolkit use the **same formula**,
`score = log10((1+v)/(1−v))` on the raw BDT output `v`. They differ in one
line: the toolkit **added a clamp** `v → [−0.9999, 0.9999]`
(`root/docs/examination/bdt-scorers-examination.md` Bug 1, marked fixed;
`UbooneNueBDTScorer.cxx:1986-1989`, `UbooneNumuBDTScorer.cxx:586-590`). The
prototype has none, so its range is bounded only by float precision: **≈ ±15**
— which is also why the "not filled" sentinel is `-15`, chosen to sit just
below the physical floor.

The clamp caps |score| at 4.301, and the two scorers disagree with each other
by one type:

| scorer | clamp variable | ceiling | seen in the data |
|---|---|---:|---:|
| `UbooneNueBDTScorer` | `float val1` | **4.3009362** | nue ceiling 4.300936 |
| `UbooneNumuBDTScorer` | `double val1` | **4.3010083** | numu clamp 4.301008 |

Two independent pieces of evidence, and they are worth keeping distinct.

**(a) The clamp is what bites, not a forest that saturates on its own.** If the
xgboost forest simply topped out just under 0.9999, the approach to the ceiling
would be empty and the pile-up would be its own natural endpoint. It is not:
both scores climb *continuously* to the ceiling and then hit a delta function.

| | distinct values in the approach | at the ceiling |
|---|---|---:|
| `nue_score` | 3.402, 4.055, 4.159, 4.242 | **38** at 4.300936 |
| `numu_score` | 24 distinct values, 4.182 … 4.297 | **98** at 4.301008 |

```bash
awk -F'\t' 'NR>1 && $15==1 && $24!="" && $24+0>3.0 && $24+0<4.4 {print $24}' \
    products/prod0825/*-scores-prod0825.tsv | sort -g | uniq -c   # $23 for numu
```

That is a live distribution being truncated.

**(b) The prototype really does exceed 4.301.**
`qlport/docs/perf-optimization.md:194` records uBooNE ev6786 at toolkit
`nue_score +4.30` vs **prototype +10.7**. That is one event, quoted inside a
perf-round note rather than a score-range study — it establishes *that* the
prototype goes past the toolkit ceiling, which is the load-bearing claim. The
**≈ ±15** figure is not measured here: it is inferred from the choice of `-15`
as the "background-like default", a value only meaningful if it sits below the
real floor.

**Consequence.** `nue_score > 7.0` requires `v > 0.9999998`; the clamp erases
everything above `v = 0.9999`. The cut therefore selects **0 events in all
four samples, by construction** — and the 32 of 47 νeCC events sitting at
exactly 4.300936 are precisely the ones whose true score is *somewhere above*
4.30, with no way to tell which of them would clear 7.0. That is the
information destroyed before it reaches `T_tagger`: not the existence of a
signal region, but all ordering inside it.

This is **reported, not fixed**: removing the clamp changes production output
unconditionally (escalation rules 1 and 4). Until it is decided, this document
uses the only reachable stand-in,

> **NUE_SEL** = `nue_score >= 4.30` — the clamp ceiling, a **strict superset**
> of the MicroBooNE selection

and reports `nue_score > 0.7` (the *other* number in that same validation-plan
doc, §4.1.4 — the doc is internally inconsistent) beside it as a bracket. The
νμ working point is unaffected: 0.9 ⇒ `v > 0.776`, far below the clamp.

### 7.2 The νμ sample under each working point

![numu sample under the BDT cuts](85_dists/d85b_numusample_cuts.png)

The 3000-event νμ data sample (mcp1k + mcp2k), **cosmics excluded**
(`cosmict_flag == 0`), same population as §1 otherwise.

| selection | mcp1k | mcp2k | total | median E<sub>ν</sub> |
|---|---:|---:|---:|---:|
| all, cosmics excluded | 359 | 709 | **1068** | 587 MeV |
| `numu_score > 0.9` | 240 | 451 | **691** | 703 MeV |
| `nue_score >= 4.30` (ceiling) | 1 | 5 | **6** | 792 MeV |
| `nue_score > 0.7` | 3 | 8 | **11** | 1077 MeV |
| `nue_score > 7.0` (uB point) | 0 | 0 | **0** | — |

195 of the 1263 plotted νμ-sample events are cosmic-flagged and drop out
before any BDT cut. The νμ selection keeps 65 % of what is left and pushes the
median up 587 → 703 MeV, as a CC selection should. The νe rows are the §7.1
story in numbers: 6 events at the ceiling, 0 at the working point that is
supposed to define the selection.

### 7.3 νeCC events the νe selection does NOT keep — [Bee set 4e5c2cda](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/list/)

**16 of 48** fail `nue_score >= 4.30`. (Under the literal MicroBooNE cut all
48 fail — §7.1 — so this set is the meaningful reading of the question.)

| Bee | run | event | numu | nue | E [MeV] | label | link |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 18342 | 30504 | −0.989 | −1.388 | 1251 | nu-candidate | [event/0](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/0/) |
| 1 | 18304 | 38856 | +0.597 | −0.367 | 1280 | nu-candidate | [event/1](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/1/) |
| 2 | 18259 | 52672 | −0.509 | **−4.301** | 928 | nu-candidate | [event/2](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/2/) |
| 3 | 18255 | 69314 | +3.960 | **−15** | 1499 | nu-candidate | [event/3](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/3/) |
| 4 | 18259 | 111412 | +0.072 | +0.037 | 1019 | nu-candidate | [event/4](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/4/) |
| 5 | 18259 | 116962 | +1.469 | **−4.301** | 773 | **cosmic-tagged** | [event/5](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/5/) |
| 6 | 18255 | 122660 | −0.064 | +2.339 | 1334 | nu-candidate | [event/6](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/6/) |
| 7 | 18264 | 137238 | +1.136 | −3.415 | 739 | nu-candidate | [event/7](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/7/) |
| 8 | 18255 | 163543 | +0.332 | +1.402 | 1270 | nu-candidate | [event/8](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/8/) |
| 9 | 18255 | 196649 | −1.117 | +4.242 | 1614 | nu-candidate | [event/9](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/9/) |
| 10 | 18255 | 235435 | −0.210 | **−15** | 428 | nu-candidate | [event/10](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/10/) |
| 11 | 18255 | 389538 | **+4.301** | **−15** | 986 | nu-candidate | [event/11](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/11/) |
| 12 | 18355 | 400474 | −0.069 | +4.159 | 1799 | nu-candidate | [event/12](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/12/) |
| 13 | 18255 | 423981 | +0.013 | +3.402 | 2133 | nu-candidate | [event/13](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/13/) |
| 14 | 18255 | 447477 | — | — | — | nu-candidate (no PR evaluation) | [event/14](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/14/) |
| 15 | 18255 | 469665 | −1.534 | +1.225 | 876 | nu-candidate | [event/15](https://www.phy.bnl.gov/twister/bee/set/4e5c2cda-0b88-46e6-90ab-0151771da5df/event/15/) |

Three of these are worth a look first: **69314** and **389538** have the νe BDT
*never filled* (`-15`) while the νμ BDT is strongly positive (+3.96, +4.30 at
the clamp) — they were routed away from the νe path entirely; **116962** is a
νeCC-sample event the chain labels `cosmic-tagged`; and **196649 / 400474**
(+4.24, +4.16) miss the ceiling by less than 0.06, i.e. by the clamp's own
rounding rather than by anything physical.

### 7.4 NCπ⁰ events passing the νμ selection — [Bee set 0133c243](https://www.phy.bnl.gov/twister/bee/set/0133c243-b77e-45d1-b57c-55303db14b8a/event/list/)

**8 of 18** evaluated NCπ⁰ events pass `numu_score > 0.9`; all 8 are
non-cosmic (`cosmict_flag = 0`).

| Bee | run | event | numu | nue | E [MeV] | link |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 18259 | 18625 | +2.409 | −15 | 1478 | [event/0](https://www.phy.bnl.gov/twister/bee/set/0133c243-b77e-45d1-b57c-55303db14b8a/event/0/) |
| 1 | 18345 | 21073 | +0.932 | −15 | 1019 | [event/1](https://www.phy.bnl.gov/twister/bee/set/0133c243-b77e-45d1-b57c-55303db14b8a/event/1/) |
| 2 | 18255 | 56982 | +1.112 | −1.285 | 1231 | [event/2](https://www.phy.bnl.gov/twister/bee/set/0133c243-b77e-45d1-b57c-55303db14b8a/event/2/) |
| 3 | 18255 | 71372 | +2.626 | −15 | 2597 | [event/3](https://www.phy.bnl.gov/twister/bee/set/0133c243-b77e-45d1-b57c-55303db14b8a/event/3/) |
| 4 | 18255 | 142421 | **+3.602** | −15 | 2768 | [event/4](https://www.phy.bnl.gov/twister/bee/set/0133c243-b77e-45d1-b57c-55303db14b8a/event/4/) |
| 5 | 18255 | 180801 | +1.356 | −3.210 | 1324 | [event/5](https://www.phy.bnl.gov/twister/bee/set/0133c243-b77e-45d1-b57c-55303db14b8a/event/5/) |
| 6 | 18261 | 285567 | +0.987 | −15 | 1935 | [event/6](https://www.phy.bnl.gov/twister/bee/set/0133c243-b77e-45d1-b57c-55303db14b8a/event/6/) |
| 7 | 18255 | 506746 | +1.112 | −15 | 1926 | [event/7](https://www.phy.bnl.gov/twister/bee/set/0133c243-b77e-45d1-b57c-55303db14b8a/event/7/) |

Bee 4 (**142421**, `numu_score +3.60`, 2768 MeV) is the strongest νμ-like NCπ⁰
event in the sample and the same event flagged in the earlier rescan as "a
major shower not tagged as shower" — the two observations are likely the same
defect seen from two sides.

### 7.5 NCπ⁰ passing cosmic *or* the νe selection — **empty**, and the nearest misses — [Bee set b6e6fefd](https://www.phy.bnl.gov/twister/bee/set/b6e6fefd-df80-4c81-be3d-2da90318d74e/event/list/)

**0 of 18.** No NCπ⁰ event is cosmic-tagged by either handle (`cosmict_flag`
is 0 on all 18; the nusel `event_label` is `nu-candidate` on all 19), and none
reaches `nue_score >= 4.30`. Only **three** have the νe BDT filled at all, and
all three are negative — so on this sample the νe selection is not being fooled
by π⁰s, with the §7.1 caveat that the ceiling proxy is looser than the real cut
and the sample is 19 events.

The set below is those three nearest misses, not a passing selection:

| Bee | run | event | numu | nue | E [MeV] | link |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 18255 | 56982 | +1.112 | −1.285 | 1231 | [event/0](https://www.phy.bnl.gov/twister/bee/set/b6e6fefd-df80-4c81-be3d-2da90318d74e/event/0/) |
| 1 | 18259 | 37112 | +0.336 | −2.468 | 1539 | [event/1](https://www.phy.bnl.gov/twister/bee/set/b6e6fefd-df80-4c81-be3d-2da90318d74e/event/1/) |
| 2 | 18255 | 180801 | +1.356 | −3.210 | 1324 | [event/2](https://www.phy.bnl.gov/twister/bee/set/b6e6fefd-df80-4c81-be3d-2da90318d74e/event/2/) |

## 8. What this document is not

* **Not efficiency or purity.** No truth is used, no selection cut is applied
  beyond `nu_evaluated == 1` and the §1.1 degenerate cut, and the plotted
  population still contains 55 chain-labelled cosmics in the two
  ν<sub>μ</sub> samples (56 across all four). Nothing here says how well the BDTs work.
* **Not the current operating point** — §4. Add ~3–4 % downward to every energy
  to reach production as of 2026-08-30.
* **Not a MicroBooNE-equivalent selection** — §7.1. The νμ working point
  transfers; the νe one is unreachable under the toolkit's score clamp, and
  every νe number here uses a strictly looser stand-in.
* **Not a gate.** No A/B, no hash comparison, no claim of byte-identicality is
  made or needed: the arms were read, not produced.

## Artifacts

| what | path |
|---|---|
| score tables (4 samples, 3067 rows) | `products/prod0825/<sample>-scores-prod0825.tsv` |
| figures | `docs/85_dists/d85_enu.png`, `docs/85_dists/d85_bdt.png` |
| summary numbers behind §2/§3 | `docs/85_dists/d85-summary-prod0825.tsv` |
| Bee pick lists + their scores | `docs/85_dists/d85-{cosmiclike,nuelike,neither}.{txt,tsv}` |
| Bee zips + index/UUID sidecars | `bee/d85/d85-*.{zip,index.txt,prid-map.txt,url}` |
| §7 figure + census | `docs/85_dists/d85b_numusample_cuts.png`, `docs/85_dists/d85b-numusample-census.tsv` |
| §7 Bee pick lists | `docs/85_dists/d85b-{nuecc-failnue,ncpi0-numupass,ncpi0-nearmiss}.{txt,tsv}` |
| §7 Bee zips + sidecars | `bee/d85b/d85b-*.{zip,index.txt,prid-map.txt,url}` |
| plotting + picking scripts | `scripts/analysis/d85_dists.py` (§2/§3/§5), `scripts/analysis/d85b_cuts.py` (§7) |

All six Bee sets uploaded 2026-08-30 with owner authorisation and verified
reachable (HTTP 200 on `event/list/` and `event/0/`).
