# doc pr/141 M1+M2 — the PRE-REGISTERED prediction

Written from the owner's verdicts and the offline geometry **alone**, committed
**before** the `onboth` arm was run or scored. Same discipline as doc pr/139 §8.

## The population

Six μ-typed, no-shower-flag, `conn_type` 2/3, ≥ 30 MeV objects of length
40–80 cm — every one in the 239-event production population. K20 refuses all six
today (`length < 40 cm` bound), and their energy is priced with the **track**
recombination.

## The owner's verdicts (2026-08-31)

| event/object | owner | in-window EM partner at the shower price |
|---|---|---|
| 283713 / 23011 | **electron** | `47021`, m = 115.2 |
| 350354 / 18009 | **electron** | `46023`, m = 117.8 |
| 122660 / 54071 | **electron shower** | none |
| 392901 / 23017 | track | `15010`, m = 124.6 |
| 280159 / 90098 | likely muon track | none |
| 294174 / 25030 | overclustered muon | none |

**Purity of the class is 3 of 6 (0.500).** That is the number that governs, and
it was not knowable before the scan.

## The prediction

With `pi0_mu_shower_hypothesis = true` **and** `pi0_mu_shower_max_len = 80`:

1. **Exactly three new π⁰ pairs** should be accepted that involve one of these
   objects — in events **283713, 350354 and 392901** — and none in the other
   three, because no EM partner in those events reaches the window even at the
   shower price.
2. **Two of the three are correct** (283713, 350354 — both owner-called
   electrons) and **one is a false π⁰** (392901/23017, owner-called a track).
   So the effect's own purity is **2/3**, on n = 3.
3. Census: **at most +2 exact**, and only if 283713 and 350354 are in the
   66-π⁰ hand set and were not already exact.
4. **No event outside these six should move.** Any other census or mover change
   is unexplained by the intended mechanism and stops the round (CLAUDE.md §5.5).

## The bar, set before scoring

M1+M2 is worth recommending only if **(a)** the arm produces exactly the three
predicted acceptances and no others, **and (b)** the census gain is strictly
positive after the 392901 false π⁰ is counted against it.

If the arm accepts pairs beyond the three, the mechanism is not what this
prediction says it is, and the knobs stay off regardless of the census.

## What is already decided regardless of the arm

`122660/54071` is an **owner-called electron shower typed `pdg = 13`** with no
in-window partner. It changes no π⁰, but it is a mis-typed 170 MeV EM object
whose energy enters `kine_reco_Enu` at the **track** price — a real defect, in a
different subsystem, that this round found and does not fix.
