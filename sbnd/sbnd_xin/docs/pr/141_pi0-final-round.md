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

# item 3 -- the MEASUREMENT: run the finder's own pair tape on the nine events
#   (stderr-only, byte-neutral; binary pinned /home/xqian/tmp/pin-pr140r4)
LD_LIBRARY_PATH=/home/xqian/tmp/pin-pr140r4 WCT_PI0_PAIR_DEBUG=1 PR_JOBS=4 \
  ./run_pr_chain_batch.sh work-mcp1k-grp0825 work-pr141dbg-pair-mcp1k data 168432 286655
#   ... and pair-mcp2k (280159 71872), pair2-mcp1k (348691 409634 283713),
#       pair2-mcp2k (397630), pair2-ncpi0 (21073)
python3 scripts/pr141_pairtape.py --tsv docs/pr/pr141-pairtape.tsv

# item 3 -- energies, alternative partners (geometry model SUPERSEDED by the tape)
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

## 1. Item 3 — the nine mass failures, measured with the finder's own tape

### 1.1 A correction I made and then had to retract

The first pass through this item modelled the finder's geometry offline.
`id_pi0_with_vertex` builds a per-γ direction (`local_dirs`,
`NeutrinoShowerClustering.cxx:7553-7625`): `get_init_dir()` when the shower's
start vertex *is* the candidate vertex (`:7572`, `:7586`), the candidate-vertex →
start ray otherwise (`:7620`). I read the dump's `start_vertex_id`, matched it
against the main vertex id, concluded that six of the nine events take the
*axis* branch, and reported that two events (168432, 280159) were therefore
inside the (100, 160) window and had never been mass failures at all.

**That was wrong.** The dump's `start_vertex_id` is not the quantity `get_svc()`
compares against `cand_vtx`. Running the finder with `WCT_PI0_PAIR_DEBUG=1` on
168432 prints, for the owner's pair:

```
PI0_PAIR P1 pair sh1=22006 sh2=49028 ct1=2 ct2=2 vtx=19001 E1=241.0 E2=115.8 m=53.5
```

— the **ray** branch, m = 53.5, which is exactly the `mass_vertex_convention` the
census used. Across the nine events the finder's mass at the main vertex equals
the label's `mass_vertex_convention` in **7 of 9** (the two exceptions are the
events whose label used a non-main π⁰ vertex). **doc pr/139 §26.3's
classification was right and my re-read of it was not.**

The lesson is the one this campaign keeps re-learning: *a rule read off the
source is a hypothesis.* The probe already existed
(`WCT_PI0_PAIR_DEBUG`, `NeutrinoShowerClustering.cxx:106`), costs one event, and
emits no bytes. `scripts/pr141_massfail.py` is kept for its energy and
alternative-partner columns, with the geometry model marked superseded;
`scripts/pr141_pairtape.py` is the measurement.

### 1.2 What the finder actually did, per event

Nine events, `WCT_PI0_PAIR_DEBUG=1`, production binary (pin
`/home/xqian/tmp/pin-pr140r4`, md5 `28e2b85a…` = `local/lib` at run time), arms
`work-pr141dbg-pair*-{mcp1k,mcp2k,ncpi0}`. Energies are the finder's own
`get_kq` = `kine_charge`.

| event | the owner's pair, as the finder scored it | why it is not the reco's π⁰ |
|---|---|---|
| 21073 | m = 190.5 at main | out of window (high) |
| 168432 | m = 53.5 | out (low); θ from the vertex is only 18.4° |
| 280159 | m = 73.6 | out (low) |
| 286655 | m = 205.4 (at 79033) / 226.1 (at main) | out (high); `80055` also fails the 30° assoc test at main (48.6°) |
| 348691 | m = 276.9, θ = 100.8° | out (high) |
| 409634 | m = 77.4 | out (low) |
| **397630** | **m = 114.4 — IN WINDOW**, at candidate vertex 15000 | **outranked**: K24's main-first tiering took `19010`+`15047` at m = 111.5 **at the main vertex** |
| **71872** | **no pair row at all** | **admission**: `64044` failed the 30° association test at all three candidate vertices (73.3°, 70.4°, 38.7°) |
| **283713** | **m = 99.9** at candidate vertex 17004 | **the window edge**: the tape's own line reads `pi0 window reject: best mass=99.9 MeV delta=-25.1` against a bound of −25.0 |

So the answer to §26.5 item 3, before any scan: **six of the nine are genuine
mass failures; three are not, and each of the three has a different named
mechanism.**

### 1.3 The three mechanisms, and what each is worth

**283713 — a π⁰ lost by 0.1 MeV.** The acceptance test is
`-25 < m - 135 + offset < 35` with `m_pi0_mass_offset = 10` (C++ default and
production), i.e. (100, 160) MeV. The best pair in the event is the owner's, at
**99.9**. `delta = -25.1`. Nothing else was accepted.

This is a knife edge on a *production-wide* constant: an offset of 10.2 instead
of 10.0 admits it, and would move every other event in the sample too. **That is
CLAUDE.md §5.1 territory — changing an existing knob's default — so it is
reported, not tuned.** What makes it worth reporting rather than filing away is
that it is a *one-sided* miss: the lower bound is −25 and the upper is +35, so
the window is already asymmetric about the peak, and the campaign's own
measurement (doc pr/135: the in-window peak sits at 134.7 at fudge 0.86, i.e.
self-consistent) gives no reason for the asymmetry to sit exactly where it does.
The scan settles whether `47021` — 25 MeV, 8.2 cm, **one segment** — is a real γ
at all; if it is not, there is nothing here.

**397630 — the main-vertex preference costs a π⁰.** The owner's pair is in
window at 114.4, at candidate vertex 15000. It loses to `19010`+`15047` at
111.5 at the **main** vertex, because `pi0_prefer_main_vertex` (K24, SBND
production ON 2026-08-31, on the owner's own instruction *"if there is a
direction ambiguity … the preference should give to neutrino vertex"*) ranks any
in-window pair at main above any non-main pair **outright**, before the |Δ|
comparison runs. On the legacy comparator the owner's pair would have won: its
key is |114.4 − 125| − 6 = 4.6 (the 6 MeV bonus for two detached members,
`:7770`) against 13.5 for the pair that was taken.

This is the rule working as specified, not a defect — but it is **the first
measured case of K24 costing a π⁰**, and doc pr/134 shipped it on a census that
went 32 → 33. Whether the trade is still right is an owner question, and the
scan answers the input to it: is `15047` (73 MeV, attached at the ν vertex) a γ,
or is `33038` the true partner?

**71872 — an admission failure, with a knob already written for it.** `64044`
(23.0 MeV, 12.4 cm, typed 211) never entered the pool: the disconnected-shower
association test needs the shower's own direction within
`pi0_assoc_angle_deg = 30°` of the vertex→start ray, and `64044` measured
73.3° / 70.4° / 38.7° at the three candidate vertices. doc pr/132 K9 named this
exact shower as the specimen for `pi0_crumb_assoc_max` — "a low-energy crumb's
15-cm PCA direction is noise" — and that knob is **default off and has never
been flipped**. If the scan says `64044` is the γ, K9 is a live, already-built
candidate; if it says `13008` (86 MeV, which the finder paired at 129.4) is the
γ, K9 stays dead and the reco is simply right.

### 1.4 What the six real mass failures still need

Two structural facts hold for the six, from the tape and the dumps:

1. **Five of the nine events carry a reco π⁰ that shares a γ with the owner's
   pair** — 21073 shares *both*, each re-paired with a different partner
   (`60081`+`34027` → 106.1 and `63100`+`11008` → 102.0). The finder is not
   failing to find a π⁰; it is choosing a different partner.
2. **The three events with no reco π⁰ at all** (286655, 409634, 283713) are the
   ones whose partner γ is smallest: `69032` is 36 MeV / 8.7 cm / 2 seg and
   `47021` is 25 MeV / 8.2 cm / **1 seg**.

Direction of the error splits them cleanly: **too high** — 21073 (190.5),
286655 (205.4), 348691 (276.9), all needing the charge product × 0.19–0.50, which
is where a grossly over-clustered γ would show; **too low** — 168432 (53.5),
280159 (73.6), 409634 (77.4), needing × 1.8–12.6, which at these energies is an
angle, i.e. a start-point, problem. `60081` (634.5 MeV over **188.7 cm and 39
segments**) and `12119` (452.9 MeV over 118.5 cm and 33 segments) are the two
over-clustering candidates the scan can settle by eye.

### 1.5 The scan

Served on port 5022, tag `pi0mass-0904-owner`, brief in
`docs/pr/pr141-scan-brief.md`, one question per event. **`em_display`, not
`split_display`** — the question is the π⁰ *pair*, which is `em_display`'s PI0
mode; `split_display` answers "cut this object in two", the question doc pr/139
closed. The tag is fresh and pre-seeded with a copy of each event's existing
label (base wins over overlay, the same precedence `pr132_pi0_census.py` uses);
originals untouched — M13.

