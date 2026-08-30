# 85 — Reconstructed energy and BDT scores of the four SBND production samples, and three Bee hand-scan sets

Two figures over the **whole** `prod0825` PR product (3067 events, four samples)
plus three 10-event Bee sets drawn from the ν<sub>μ</sub> samples.

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

# 4. the fudge-flip caveat of sec 4, measured not assumed
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
| with a `kine_reco_Enu` value | 18 | 47 | 445 | 867 |

Two honest caveats on that table:

* the evaluated population is **not** a neutrino-purity cut — 110 of the 1326
  evaluated ν<sub>μ</sub>-sample events carry the chain's own `cosmic-tagged`
  event label. They are in the histograms.
* 14 evaluated events have an empty `kine_reco_Enu` (2 mcp1k, 12 mcp2k); they
  are counted as evaluated but excluded from the energy figure, hence
  `N=445`/`N=867` in its legend.

## 2. Reconstructed neutrino energy

![reconstructed Enu](85_dists/d85_enu.png)

`kine_reco_Enu` (MeV) from `T_kine`, four samples overlaid. **Left** is
area-normalised — this is the panel to read for shape, because 18 against 879
events on a common count axis makes two of the four samples invisible.
**Right** is raw counts on a log *y*. The bin at 3000–3200 MeV is an explicit
**overflow** bin, not a clip: 1 nuecc48, 1 mcp1k and 3 mcp2k events live there
(max 4654 MeV).

| | ncpi0 | nuecc48 | mcp1k | mcp2k |
|---|---:|---:|---:|---:|
| N | 18 | 47 | 445 | 867 |
| min / median / max [MeV] | 126 / **1401** / 2768 | 428 / **1578** / 3948 | 0 / **562** / 3583 | 0 / **516** / 4654 |

The two ν<sub>μ</sub>-beam samples land on top of each other (median 562 vs
516 MeV) — expected, they are consecutive slices of the same MCP2025C data, and
their agreement is the only internal consistency check this figure offers. The
two hand-selected samples sit a factor ~2.7 higher and are much flatter. That
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
   points sitting exactly on the dashed frame are that clamp.
3. **`cosmic_flag` is not usable as a discriminator here** and is deliberately
   absent from the figure: it is 1 for 1307 of the 1326 evaluated
   ν<sub>μ</sub>-sample events (it is the struct default, and nothing in the
   scored path overwrites it). The live cosmic-side handle is the nusel
   `event_label`, which is what section 5 uses.

| | ncpi0 | nuecc48 | mcp1k | mcp2k |
|---|---:|---:|---:|---:|
| evaluated with both scores | 18 | 47 | 445 | 867 |
| **nue BDT filled** (`> -15`) | **3** | **44** | **20** | **71** |
| median `numu_score` | +0.57 | −0.45 | +1.69 | +1.42 |
| at the `numu_score` clamp | 0 | 1 | 31 | 70 |

The separation the plot is for: the ν<sub>e</sub>CC sample fills the nue BDT
**44/47** and piles up at `nue_score = +4.30`, while the same BDT fills for only
**20/445** and **71/867** of the ν<sub>μ</sub>-sample events. Median
`numu_score` runs the other way (+1.69/+1.42 for the ν<sub>μ</sub> samples,
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
and it is far below the ~2.7× median separation between the hand-selected and
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

Re-running the 3067-event chain at 0.84 was not done: at ~77 s/event it is ~11 h
at the standing job caps, which is not a reasonable cost for a distribution
plot. If the owner wants the figures at the exact current operating point, that
is the price and it should be a fresh tag (`prod0830`), never a write into
`prod0825` (M13).

## 5. Three Bee hand-scan sets from the ν<sub>μ</sub> samples

Drawn from the **1312** `mcp1k`+`mcp2k` events that are PR-evaluated and carry
both scores. Event ids do not collide between mcp1k and mcp2k (checked: 0
overlaps), so each category is one Bee set spanning both.

**These are rank-order cuts, not thresholds.** There is no documented
`numu_score` / `nue_score` operating point anywhere in these docs, so each
category is "the 10 most extreme by the named score" within its pool. Say so
before quoting any of them as a selection.

Charge cloud = `work-mcp{1k,2k}-grp0825` (Q/L), PR layers =
`work-mcp{1k,2k}-prod0825` — the same arm as both figures.

### 5.1 Cosmic-like — [Bee set 7b4ee14a](https://www.phy.bnl.gov/twister/bee/set/7b4ee14a-13c3-4a66-b1ff-abf0034c1df8/event/list/)

Pool: the **102** evaluated events the chain itself labels `cosmic-tagged`
(in-beam bundle tagged TGM/STM/LM); picked the 10 lowest `numu_score`, i.e.
where the tagger and the BDT agree. Note this is the *evaluated* cosmic
population — a further 821 cosmic-tagged events never got a selected main
cluster and so carry no scores; they would need `--allow-unevaluated` to reach
Bee at all.

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

Pool: the **1206** `nu-candidate` events; picked the 10 highest `nue_score`.
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

Pool: the **1122** `nu-candidate` events whose nue BDT never filled
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

## 6. What this document is not

* **Not efficiency or purity.** No truth is used, no selection cut is applied
  beyond `nu_evaluated == 1`, and the evaluated population still contains 110
  chain-labelled cosmics in the two ν<sub>μ</sub> samples (111 across all four). Nothing here says how well the BDTs work.
* **Not the current operating point** — §4. Add ~3–4 % downward to every energy
  to reach production as of 2026-08-30.
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
| plotting + picking script | `scripts/analysis/d85_dists.py` |

Bee sets uploaded 2026-08-30 with owner authorisation; all three verified
reachable (HTTP 200 on `event/list/` and `event/0/`).
