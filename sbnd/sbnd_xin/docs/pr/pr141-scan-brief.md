# doc pr/141 item 3 — the scan brief  (tag `pi0mass-0904-owner`, port 5022)

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./em_display/serve_em_display.sh 5022 --scan-tag pi0mass-0904-owner \
    --manifest em_display/em141-massfail9-manifest.tsv \
    --prepdir em_display/emprep-140r2off
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -L 5022:localhost:5022 <user>@wcgpu1.phy.bnl.gov
#   then http://localhost:5022/em_display_viewer
```

**It is `em_display`, not `split_display`, on purpose.** `split_display` answers
"should this one object be cut in two" — the question doc pr/139 closed. This
round's question is about the π⁰ *pair*: which two showers, and are their
energies right. That is `em_display`'s PI0 mode (two γ slots, both mass
conventions, per-segment marks), the same instrument the 66-π⁰ census was
built on.

**The tag is fresh and pre-seeded** with a copy of each event's existing label,
so the pairing you recorded before is already in the slots. Originals
(`emscan-0827`, `emscan-0828-agent5`, `pi0scan-0829-agent`) untouched — M13.

---

## What these 9 events are, measured by the finder's own tape

doc pr/139 §26.3 ended with nine π⁰ whose hand pair reconstructs outside the
finder's acceptance window (100, 160) MeV, and asked which of two things does
it: **wrong pairing**, or **a grossly mis-clustered γ**.

Rather than model the finder offline, this round ran it with
`WCT_PI0_PAIR_DEBUG=1` on all nine events (stderr-only, byte-neutral) and read
what it did. Six of the nine are exactly what the census said — the pair forms
and its mass is far outside. **Three are not, and each has a different, specific
mechanism** that no offline model would have found:

| | n | events |
|---|---|---|
| **the mass really is the blocker** | 6 | 21073, 168432, 280159, 286655, 348691, 409634 |
| **admission** — a γ never reached the pool | 1 | 71872 |
| **ranking** — in window, outranked by the main-vertex rule | 1 | 397630 |
| **the window edge, by 0.1 MeV** | 1 | 283713 |

---

## Per event — the one question the scan has to settle

`m_finder` is the mass **the finder itself computed** for your pair (from its own
tape line), and energies are the arm's `kine_charge`. Where a mass is quoted at
a non-main candidate vertex it says so.

### The three with a mechanism other than mass

**283713** (mcp1k) — g1 `67051` 152.5 MeV / 30.9 cm · g2 `47021` 25.2 MeV / 8.2 cm / **1 seg**
· **m_finder = 99.9 MeV** at candidate vertex 17004, and the tape's own line is
`pi0 window reject: best mass=99.9 MeV delta=-25.1` against a bound of −25.0.
**This π⁰ is lost by 0.1 MeV of window.** Nothing else in the event was accepted.
→ **Q: is `47021` — 25 MeV, 8.2 cm, one segment — a real γ?** If yes, this π⁰ is
lost to the window edge alone and to nothing about the clustering.

**397630** (mcp2k) — g1 `19010` 119.7 MeV / 35.9 cm / 2 seg · g2 `33038` 106.0 MeV / 37.6 cm / 11 seg
· **m_finder = 114.4 MeV — IN WINDOW** at candidate vertex 15000. It was not
taken: the finder accepted `19010`+`15047` (73.1 MeV) at m = 111.5 **at the main
vertex**, and K24's three-tier main-first ranking (`pi0_prefer_main_vertex`,
SBND production ON since 2026-08-31, your own rule: *"the preference should give
to neutrino vertex"*) wins outright over any non-main pair.
→ **Q: which pairing is right — is `15047` (73 MeV, attached at the ν vertex) a
γ, or is `33038` the true partner?** If `33038` is right, this is the first
measured case of the main-vertex preference costing a π⁰.

**71872** (mcp2k) — g1 `79074` 194.8 MeV / 38.1 cm · g2 `64044` **23.0 MeV / 12.4 cm, pdg = 211**
· **the pair never formed.** `64044` failed the 30° disconnected-association
test at all three candidate vertices (73.3°, 70.4°, 38.7°). The finder took
`79074`+`13008` (86 MeV) at 129.4 instead. This is the K9 "crumb" specimen from
doc pr/132.
→ **Q: is `64044` the γ, or is `13008` (86 MeV) the γ and the reco right?**
If `64044` is the γ, `pi0_crumb_assoc_max` (default off) is the knob that admits it.

### The six where the mass really is wrong

**21073** (ncpi0) — g1 `60081` **634.5 MeV / 188.7 cm / 39 seg** · g2 `63100` 94.8 MeV / 30.0 cm
· m_finder = **190.5** (needs E₁E₂ × 0.50). The finder used **both** your γs but
paired each with something else: `60081`+`34027` → 106.1 **and** `63100`+`11008` → 102.0.
→ **Q: is `60081` one γ?** 188 cm and 39 segments is the largest object in the set.
If it is over-clustered by ~2× in charge, everything else follows.

**168432** (mcp1k) — g1 `22006` 241.0 MeV / 37.0 cm / 3 seg · g2 `49028` 115.8 MeV / 50.8 cm / 7 seg
· m_finder = **53.5** — the opening angle from the ν vertex is only **18.4°**.
The finder took `49028`+`19032` (71.3 MeV) at 156.8.
→ **Q: are the two γ start points right?** A mass this low at these energies is
an angle problem, not a charge problem.

**280159** (mcp2k) — g1 `24015` 74.1 MeV / 22.7 cm · g2 `95114` 129.3 MeV / 56.7 cm
· m_finder = **73.6**. The finder took `24015`+`12119` at 131.5, where `12119` is
a **452.9 MeV / 118.5 cm / 33-segment** object.
→ **Q: is `12119` one γ, or is it over-clustered (does it hold `95114`'s charge,
or a track)?** If over-clustered, the reco's 131.5 is a coincidence.

**286655** (mcp1k) — g1 `19006` 219.4 MeV / 81.3 cm · g2 `80055` 187.3 MeV / 50.3 cm
· m_finder = **205.4** (at 79033) / **226.1** (at main 79034). Nothing was accepted
in this event. `80055` also fails the 30° association test at main (48.6°).
→ **Q: are both objects really γs from this vertex, or is one a track / a
different interaction?**

**348691** (mcp1k) — g1 `49046` 301.6 MeV / 85.9 cm · g2 `50073` 107.1 MeV / 37.3 cm
· m_finder = **276.9**, opening angle **100.8°**. The finder took `50073`+`52089`
(157 MeV) at 141.8.
→ **Q: a 100° opening is very wide for a π⁰ — is the pair right at all, or is
`52089` the true partner and `49046` something else?**

**409634** (mcp1k) — g1 `21002` 167.2 MeV / 48.4 cm · g2 `69032` **36.3 MeV / 8.7 cm / 2 seg**
· m_finder = **77.4**. Nothing accepted in the event.
→ **Q: is `69032` a γ or a fragment of `21002`?** 8.7 cm and 2 segments is crumb-sized.

---

## How to record the answer

For each event, either

1. **confirm the pairing** and say the mechanism in the note — one of
   `OVERCLUSTERED g<id>` / `FRAGMENT g<id>` / `WRONG START` / `WRONG VERTEX` /
   `PAIR OK, ENERGY WRONG`; or
2. **re-pair** in PI0 mode if the reco's partner — or a third object — is the
   right γ. The stored candidate keeps your old row as `supersedes`.

**If time is short, the three highest-value events are 283713, 397630 and
71872** — each one is a single yes/no that settles a specific mechanism, and
each has a named knob behind it. 21073 and 280159 are next: they are the two
"is this big object over-clustered" tests, which is the question of whether doc
pr/139's *"the clustering share is spent"* conclusion still holds.
