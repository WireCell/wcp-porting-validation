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
conventions, per-segment marks), the same instrument the 66-π⁰ census was built
on.

**The tag is fresh and pre-seeded.** `em_labels/pi0mass-0904-owner/` was created
for this scan and holds a **copy** of each event's existing label, so the
pairing you recorded before is already in the slots when the event opens. The
originals (`emscan-0827`, `emscan-0828-agent5`, `pi0scan-0829-agent`) are
untouched — M13.

---

## What these 9 events are

doc pr/139 §26.3 ended with nine π⁰ whose hand pair reconstructs at a mass
outside the finder's acceptance window, and asked (§26.5 item 3) which of two
things does it: **wrong pairing**, or **a grossly mis-clustered γ**.

Before the scan, `scripts/pr141_massfail.py` re-evaluated all nine **the way the
finder evaluates them**, which turns out not to be the convention the census
used. The finder's opening angle is per-γ (`NeutrinoShowerClustering.cxx:7553-7625`):
the shower's own axis when its start vertex *is* the candidate vertex, the
vertex→start ray otherwise. That moves two events across the window edge, so
the nine now split three ways:

| class | n | events |
|---|---|---|
| **A — mass is NOT the blocker** (hand pair is inside (100,160) under the finder's own geometry) | 2 | 168432, 280159 |
| **B — mass far too HIGH** (needs E₁E₂ × 0.19–0.52) | 3 | 21073, 286655, 348691 |
| **C — mass far too LOW** (needs E₁E₂ × 2.5–12.5) | 4 | 397630, 409634, 71872, 283713 |

Class B is where "grossly over-clustered γ" would show up; class C is where
"the partner is a fragment, or the pairing is wrong" would.

---

## Per event — the one question the scan has to settle

Energies below are on the **arm** scale (fudge 0.86); the viewer shows the same
numbers. `m_finder` is the mass the finder would compute for **your** pair.

### Class A — the pair is admissible; something else blocked it

**168432** (mcp1k) — g1 `22006` 241.0 MeV / 37.0 cm / 3 seg · g2 `49028` 133.0 MeV / 50.8 cm / 7 seg
· θ 35.4° · **m_finder = 108.9, IN WINDOW**, and it would even *win* the greedy
pick (|Δ| 16.1) against what the reco actually took (`49028`+`19032`, m = 156.8, |Δ| 31.8).
→ **Q: is `22006` a real γ, and is it a separate object from `19032` (45 MeV)?**
If yes, the blocker is admission, not mass — a live lead.

**280159** (mcp2k) — g1 `24015` 72.7 MeV / 22.7 cm / 10 seg · g2 `95114` 120.5 MeV / 56.7 cm / 8 seg
· θ 64.8° · **m_finder = 100.3, just inside**. It loses the greedy pick to
`12119`+`24015` at m = 131.5 — and `12119` is a **453 MeV, 118 cm, 33-segment**
object.
→ **Q: is `12119` one γ, or is it over-clustered (does it contain `95114`'s charge,
or a track)?** If it is over-clustered, the reco's 131.5 is a coincidence and
your pair is right.

### Class B — mass far too high

**21073** (ncpi0) — g1 `60081` **703.4 MeV / 188.7 cm / 39 seg** · g2 `63100` 97.4 MeV / 30.0 cm / 7 seg
· θ 42.0° · m_finder = 187.8 (needs E₁E₂ × 0.52). The reco used **both** your γs
but paired each with something else: `60081`+`34027` (73 MeV) → 106.1 and
`63100`+`11008` (70 MeV) → 102.0.
→ **Q: is `60081` one γ?** 188 cm and 39 segments is the largest object in the
set. If it is over-clustered by ~2× in charge, everything else follows.

**286655** (mcp1k) — g1 `19006` 220.2 MeV / 81.3 cm / 11 seg · g2 `80055` 187.3 MeV / 50.3 cm / 12 seg
· θ 67.8° · m_finder = 226.5 (needs × 0.36). **The reco found no π⁰ at all here.**
Note both γ start vertices are `79033` while the main vertex is `79034`.
→ **Q: are both objects really γs from this vertex, or is one a track / a
different interaction?**

**348691** (mcp1k) — g1 `49046` 301.6 MeV / 85.9 cm / 10 seg · g2 `50073` 107.1 MeV / 37.3 cm / 12 seg
· θ **120.1°** · m_finder = 311.4 (needs × 0.19). The reco paired `50073` with
`52089` (157 MeV) → 141.8.
→ **Q: a 120° opening is very wide for a π⁰ — is the pair right at all, or is
`52089` the true partner and `49046` something else?**

### Class C — mass far too low

**397630** (mcp2k) — g1 `19010` 119.7 MeV / 35.9 cm / **2 seg** · g2 `33038` 106.0 MeV / 37.6 cm / 11 seg
· θ 34.9° · m_finder = 67.5 (needs × 4.0). Reco paired `19010`+`15047` (73 MeV) → 111.5.
Your label used a **back-projected** π⁰ vertex, not the main vertex.
→ **Q: are the two γ start points right?** A too-small opening angle at plausible
energies means the starts (or the vertex) are wrong, not the charge.

**409634** (mcp1k) — g1 `21002` 168.0 MeV / 48.4 cm / 11 seg · g2 `69032` **36.3 MeV / 8.7 cm / 2 seg**
· θ 28.3° · m_finder = 38.2 (needs × 12.5 — hopeless as an energy error).
**The reco found no π⁰.**
→ **Q: is `69032` a γ or a fragment of `21002`?** 8.7 cm and 2 segments is crumb-sized.

**71872** (mcp2k, near-miss) — g1 `79074` 186.6 MeV / 38.1 cm · g2 `64044` **18.5 MeV / 12.4 cm, pdg = 211**
· θ 111.9° · m_finder = 97.3. Reco paired `79074`+`13008` (86 MeV) → 129.4.
This is the K9 "crumb" specimen from doc pr/132.
→ **Q: is `64044` the γ, or is `13008` (86 MeV) the γ and the reco right?**

**283713** (mcp1k, near-miss) — g1 `67051` 152.5 MeV / 30.9 cm · g2 `47021` **25.2 MeV / 8.2 cm / 1 seg**
· θ 85.9° · m_finder = 84.5. **The reco found no π⁰.**
→ **Q: same as 409634 — is `47021` a γ or a crumb?**

---

## How to record the answer

For each event, either

1. **confirm the pairing** and say the mechanism in the note — one of
   `OVERCLUSTERED g<id>` / `FRAGMENT g<id>` / `WRONG START` / `WRONG VERTEX` /
   `PAIR OK, ENERGY WRONG`; or
2. **re-pair** in PI0 mode (assign the two slots, store the pairing) if the
   reco's partner — or a third object — is the right γ. The stored candidate
   keeps your old row as `supersedes`, so nothing is lost.

Confidence field as usual. Nine events; the two near-misses (71872, 283713) are
the least valuable — drop them if time is short.

**What the answer buys.** Class A says whether there is an *admission* fix
(2 events, and 168432 would flip on its own). Class B says whether over-clustering
is still costing π⁰ after the splitter shipped — i.e. whether doc pr/139's
"clustering is spent" conclusion holds. Class C says whether the crumb partners
are real, which is the `pi0_crumb_assoc_max` knob's territory.
