# doc pr/141 — the final round: what is left after the splitter campaign

**Status: OPEN (session 1, 2026-08-31).** doc pr/139 §26 closed the splitter
sub-campaign and ranked what remains (§26.5). The owner asked for one last round
against that list:

> *"I would like to act one last round '1. The k=3 kernel recursion … 2. The 8
> missing γs … 3. A hand scan of the 7 badly-wrong masses … 4. Cheap leftovers:
> the pdg=211 EM-only restriction and 278420's separability'. I assume you want
> me to do a scan, can you serve them in port 5022 first, so that I can provide
> you the results while you act on the others. Note, the goal is to have this one
> last round of campaign tonight for improvements."*

**Why 141 and not 140.** The `pr140_*` scripts belong to doc pr/139's sessions
2–4 (§8–§26); doc number 140 is left unused so a script prefix always names the
doc that owns it.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# item 3 -- the pre-scan analysis of the nine mass failures
python3 scripts/pr141_massfail.py --tsv docs/pr/pr141-massfail.tsv

# item 3 -- the owner's scan (port 5022, tag pi0mass-0904-owner)
./em_display/serve_em_display.sh 5022 --scan-tag pi0mass-0904-owner \
    --manifest em_display/em141-massfail9-manifest.tsv \
    --prepdir em_display/emprep-140r2off
```

Arm under scan: `work-pr140r2-off-*` — the **shipped production** configuration
(`onV1c90` + splitter, every doc pr/139 knob off), the arm whose census is
35/66 exact.

---

## 1. Item 3 — the nine mass failures, re-read the way the FINDER reads them

### 1.1 The correction that comes first

doc pr/139 §26.3 classified nine π⁰ as "pair found, mass outside (100,160)"
using the label's `mass_vertex_convention` — the mass from the angle between the
two rays main-vertex → γ start. **That is not the finder's convention.**

`id_pi0_with_vertex` builds a per-γ direction (`local_dirs`,
`NeutrinoShowerClustering.cxx:7553-7625`) and it is keyed on the *start vertex*,
not on `conn_type`:

| case | direction used |
|---|---|
| shower attached to the candidate vertex (`conn_type == 1`) | `get_init_dir()` — the shower's own axis (`:7572`) |
| disconnected shower whose **start vertex IS the candidate vertex** | `get_init_dir()` — the axis again (`:7586`) |
| any other disconnected shower | the candidate-vertex → start ray (`:7620`) |

and the mass is `sqrt(4 kq1 kq2 sin²(θ/2))` at `:7707`, accepted when
`-25 < m - 135 + offset < 35` with `m_pi0_mass_offset = 10` ⇒ **(100, 160) MeV**.

So the label's two stored masses *bracket* the finder's, and which end applies
is decided per γ by `start_vertex_id == <main vertex id>` in the dump. Six of
the nine events take `axis/axis`, two take `axis/ray`, one takes `ray/ray`.

### 1.2 What that changes

`scripts/pr141_massfail.py` recomputes all nine on the finder's own geometry, at
the arm's energy scale (labels are scan-time fudge 0.80, the arm runs 0.86, so
every energy and every mass is multiplied by 0.80/0.86). Two events cross the
window edge:

| event | census said | finder's own geometry | class |
|---|---|---|---|
| 168432 | 57 MeV | **108.9 — IN WINDOW** | A |
| 280159 | 70 MeV | **100.3 — IN WINDOW** | A |
| 21073 | 203 | 187.8 | B (too high) |
| 286655 | 226 | 226.5 | B |
| 348691 | 277 | 311.4 | B |
| 397630 | 54 | 67.5 | C (too low) |
| 409634 | 78 | 38.2 | C |
| 71872 | 94 | 97.3 | C (near-miss) |
| 283713 | 88 | 84.5 | C (near-miss) |

**So "mass" was never the blocker for two of the seven.** For those two the
finder's own arithmetic accepts the hand pair; what stops it is admission or the
greedy pick:

- **168432** — the hand pair would **win** the greedy competition
  (|m−125| = 16.1) against the pair the reco actually took (`49028`+`19032`,
  m = 156.8, |Δ| = 31.8). So the hand pair never entered the pool: an
  **admission** failure, and the only live "free" π⁰ in the whole set.
- **280159** — the hand pair is admissible but **loses** the greedy pick to
  `12119`+`24015` at m = 131.5, where `12119` is a 453 MeV / 118 cm /
  33-segment object. Whether that is a wrong pick depends on whether `12119`
  is one γ — a scan question.

### 1.3 What the remaining seven need, quantitatively

`need_eprod` is the factor the *product* E₁E₂ must move by to land at 135:

| event | m_finder | need E₁E₂ × | need θ | reco paired instead |
|---|---|---|---|---|
| 21073 | 187.8 | 0.52 | 42.0° → 29.9° | `60081`+`34027` → 106.1 **and** `63100`+`11008` → 102.0 |
| 286655 | 226.5 | 0.36 | 67.8° → 38.8° | **nothing** |
| 348691 | 311.4 | 0.19 | 120.1° → 44.1° | `52089`+`50073` → 141.8 |
| 397630 | 67.5 | 3.99 | 34.9° → 73.6° | `19010`+`15047` → 111.5 |
| 409634 | 38.2 | 12.51 | 28.3° → 119.5° | **nothing** |
| 71872 | 97.3 | 1.92 | unreachable at any angle | `79074`+`13008` → 129.4 |
| 283713 | 84.5 | 2.55 | unreachable at any angle | **nothing** |

Two structural facts fall straight out of the table and need no scan:

1. **Five of the nine events already carry a reco π⁰ that shares a γ with the
   hand pair** (21073 shares *both*, each re-paired with a different partner).
   So the finder is not failing to *find* a π⁰ in these events — it is choosing
   a different partner. That makes "wrong pairing" the leading hypothesis over
   "no candidate", which is what §26.5 item 3 asked.
2. **Three events (286655, 409634, 283713) have no reco π⁰ at all**, and all
   three are the ones whose hand pair is furthest from the window. Their partner
   γs are small: `69032` is 36 MeV / 8.7 cm / 2 seg and `47021` is 25 MeV /
   8.2 cm / **1 seg**. Crumb-sized — `pi0_crumb_assoc_max` territory.

### 1.4 The scan, and what it decides

Served on port 5022, tag `pi0mass-0904-owner`, brief in
`docs/pr/pr141-scan-brief.md`. **`em_display`, not `split_display`** — the
question is the π⁰ *pair* (which two showers, are the energies right), which is
`em_display`'s PI0 mode; `split_display` answers "cut this object in two", the
question doc pr/139 closed.

The tag is fresh and **pre-seeded with a copy** of each event's existing label
(from `emscan-0827`, `emscan-0828-agent5`, `pi0scan-0829-agent`; base wins on
conflict, the same precedence `pr132_pi0_census.py` uses), so the recorded
pairing is already in the slots. Originals untouched — M13.

Per event the scan settles one question, listed in the brief. What each class
buys:

- **A (2)** — is there an *admission* fix? 168432 would flip on its own.
- **B (3)** — is over-clustering still costing π⁰ after the splitter shipped,
  i.e. does doc pr/139's "the clustering share is spent" conclusion hold?
  21073's `60081` — 703 MeV over **188.7 cm and 39 segments** — is the test.
- **C (4)** — are the crumb partners real γs?
