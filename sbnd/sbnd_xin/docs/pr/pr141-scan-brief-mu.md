# doc pr/141 recommendation 2 — the μ-typed scan brief  (tag `pi0mu-0905-owner`, port 5022)

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./em_display/serve_em_display.sh 5022 --scan-tag pi0mu-0905-owner \
    --manifest em_display/em141-smu6-manifest.tsv \
    --prepdir em_display/emprep-140r2off
ssh -o ServerAliveInterval=30 -L 5022:localhost:5022 <user>@wcgpu1.phy.bnl.gov
#   then http://localhost:5022/em_display_viewer
```

## The one question

**Is this μ-typed object a photon?**

These are the six objects in the whole 239-event production population that are
typed `pdg = 13`, carry **no** shower flag, sit in the disconnected pool
(`conn_type` 2 or 3), are above 30 MeV, and are **40–80 cm long** — i.e. exactly
the class K20's `trackish` rule refuses (`length < 40 cm` is its bound).

Everything downstream depends on this one judgement, so it is worth stating what
each answer buys before you look:

- **If they are photons** — the π⁰ finder is refusing real γs on a *type* test,
  and §8's `pi0_mu_shower_hypothesis` knob (which re-prices them under the
  shower recombination) has something real to rescue. Three of the six then
  reach the mass window.
- **If they are muons or pion tracks** — the type test is doing its job, the
  knob is correct to be default-OFF, and this whole line closes cleanly.

Judge it the way you judged the splitter set: is it a *track* (a MIP-like line,
constant dE/dx, one prong) or a *shower* (spreading, branching, growing)? These
objects carry no shower flag, which is *why* they were typed 13 — so the reco's
own opinion is already "track", and the question is whether it is wrong.

## The six

`E_track` is what the finder uses today; `E_shower` is the same collected charge
under the shower recombination and fudge (an exact ×1.657 at the production
factors — nothing is re-measured). `partner` is the EM shower that would give the

**mcp2k evt 392901, object `23017`** — E_track = 83.7 MeV, **E_shower = 138.7 MeV**, length 42.4 cm, 1 segment(s), conn 2
  · best EM partner `15010` at θ = 64.3° → **m = 124.6 MeV** — **in the (100,160) window**

**mcp2k evt 350354, object `18009`** — E_track = 100.2 MeV, **E_shower = 166.0 MeV**, length 70.2 cm, 2 segment(s), conn 3
  · best EM partner `46023` at θ = 99.5° → **m = 117.8 MeV** — **in the (100,160) window**

**mcp1k evt 283713, object `23011`** — E_track = 168.4 MeV, **E_shower = 279.0 MeV**, length 57.3 cm, 11 segment(s), conn 2
  · best EM partner `47021` at θ = 86.6° → **m = 115.2 MeV** — **in the (100,160) window**

**mcp2k evt 294174, object `25030`** — E_track = 59.8 MeV, **E_shower = 99.1 MeV**, length 78.2 cm, 1 segment(s), conn 3
  · best EM partner `71067` at θ = 61.3° → **m = 95.7 MeV** — outside the window

**mcp2k evt 280159, object `90098`** — E_track = 73.5 MeV, **E_shower = 121.8 MeV**, length 40.3 cm, 1 segment(s), conn 3
  · no EM partner in the event gives an in-window mass even at the shower price

**nuecc48 evt 122660, object `54071`** — E_track = 102.6 MeV, **E_shower = 170.0 MeV**, length 53.6 cm, 5 segment(s), conn 2
  · no EM partner in the event gives an in-window mass even at the shower price


## How to record it

Per object, in the note: `GAMMA` / `MUON` / `PION` / `UNSURE`, plus one line of
why (spread vs MIP line, branching, dE/dx). If it is a γ, set the two γ slots in
PI0 mode so the pairing is on record — the partner above is a geometric best
fit, not a claim.

**280159 and 283713 you have already scanned for other reasons** (their labels
are seeded in), so the genuinely new judgements are `294174/25030`,
`350354/18009`, `392901/23017` and `122660/54071` — four objects.

`122660/54071` is worth a note: doc pr/138 listed it among the eight
**track-typed** objects excluded from the splitter scan set, so it has been
called a track once already. If it still reads as a track, say so — a confirmed
negative is as useful here as a positive.

## One correction to carry in

An earlier note of mine put the track→shower ratio at **1.906**, measured off
283713's `23011`. That was wrong: 1.906 was contaminated by **152.5 MeV of
charge you had hand-added to that object as `in` marks**
(`energy_marks_delta`). The ratio is not measured at all — it is exact and
global, because the energy conversion is a pure division
(`overall / recom / fudge`, `NeutrinoEnergyReco.cxx:188`):
(0.87 × 0.95) / (0.58 × 0.86) = **1.657**. Every mass above uses 1.657.
