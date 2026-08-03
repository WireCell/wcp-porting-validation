# Bee scan key — B0 knob OFF vs ON, the events whose PR output moved

Both sets carry the **same 14 events in the same Bee index order**, so index *i*
is the same event in both tabs.

- OFF (B0 disabled, = today's SBND production):
  https://www.phy.bnl.gov/twister/bee/set/00ec3bae-3461-4046-a282-b5d3cecda13e/event/list/
- ON  (`cathode_kink_xcut = 5 cm`, `cathode_x = 0`):
  https://www.phy.bnl.gov/twister/bee/set/9b16b249-3bfd-4c69-8882-44605e2584ac/event/list/

Both arms come from the identical Q/L tree `work-mcp1kall-cathA12on2` (A1+A2 ON)
and the identical binary; the only difference is the two TLAs. Sources:
`work-b0pr{300,700}-{off,on}/pr_evt<ID>/mabc-pr.zip`.

The question to answer per event: **is the ON reconstruction better, worse, or a
wash?** The cases are ordered worst-first.

| idx | event | what moved | reco vertex OFF → ON (cm) | `kine_reco_Enu` MeV |
|---|---|---|---|---|
| 0 | **286400** | **different nu candidate**, `cosmict_flag` 1→0, numu 4.11→−0.73 | (−2, −164, 56) → **(154, −87, 309)** | 1085.6 → 815.5 |
| 1 | **289559** | **different nu candidate**, `nue_score` −15→−4.3, numu 3.79→1.11 | (−3, 13, 355) → **(39, 0, 444)** | 1834.5 → 763.4 |
| 2 | 281214 | `nue_score` −4.3 → **−15 (the "not filled" sentinel)**, numu 2.14→2.86 | (32, 32, 298) → (31, 33, 299) | 1639.7 → **1043.8** |
| 3 | 386948 | vertex 55 cm, numu 0.63→0.22 | (−17, 143, 355) → (−7, 185, 388) | 1104.0 → 1278.4 |
| 4 | 172794 | **a break B0 *created*** at x = 21 cm; numu −0.43→4.30 | (27, 151, 184) → (21, 148, 178) | 802.6 → 1215.3 |
| 5 | 349549 | `nue_score` −2.10 → **+1.05**, numu 3.05→2.51 | (3, −78, 111) → (3, −78, 112) | 1215.2 → 1215.1 |
| 6 | 287654 | vertex 7 cm; 2 off-cathode 1-cm fragments also re-split | (−8, 58, 463) → (−3, 56, 460) | 652.8 → 687.4 |
| 7 | 285971 | vertex 1.4 cm, numu 2.98→3.43 | (−123, −18, 96) → (−124, −17, 95) | 717.3 → 835.7 |
| 8 | 486247 | `cosmict_flag` **1→0**, numu −0.60→−0.24 | (7, −30, 380) → (8, −24, 381) | 477.1 → 537.2 |
| 9 | 292643 | vertex 0.8 cm, scores ~unchanged | (−13, −29, 55) → (−13, −30, 56) | 949.4 → 987.4 |
| 10 | 395654 | `cosmict_flag` **0→1**, `nue_score` −4.3→−15 (no vertex move) | — | 629.4 → 803.6 |
| 11 | 409634 | a cathode stub **survives** B0 (6.81 → 4.26 cm), built by another breaker | — | 632.5 → 631.5 |
| 12 | **315497** | the class-A crosser: A1 joins it, the kink finder re-splits it into a 4.76 cm cathode stub, **B0 keeps it whole** | — | **382.0 → 962.8** |
| 13 | **169824** | the design case: the 4.67 cm stub and both cathode vertices go | — | 1125.6 → 1071.2 |

Indices 12–13 are the intended wins and are included as contrast, not as
questions. Indices 0–1 are the ones that decide whether B0 can be default ON.
