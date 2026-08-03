# doc pr/25 §3.9 — the 10 nu-candidate-main long shower-flagged segments

Bee set: https://www.phy.bnl.gov/twister/bee/set/34d88b08-af0b-4d31-a3fa-868ee573dd0b/event/list/

Built from `work-pr25s3r2-dbgall` (PR) + `work-mcp1kall-vfprodoff` (QL/img),
`make_pr_bee.py` with cluster-id remap to img-global space.

**The question to answer by eye:** each of these has a segment longer than
50 cm on the selected nu-candidate main cluster that the PR chain flagged
`kShowerTopology` (→ `pdg=11`, `score=100`, no track PID attempted). Is it a
shower, or a track? See doc pr/25 §3.8 — a 249 cm EM shower should not exist
(X₀ ≈ 14 cm in LAr), and the flag fired on drift-direction quantization noise
in 86 of 91 such cases across the manifest.

**Layer note (doc pr/13):** `img-global` is the only raw layer; the PR layers
(`shower_track-global`, `track_fit-global`, `vertices-global`) are in the
remapped frame.

| bee idx | evt | seg L (cm) | angle to drift | numu_score | kine_reco_Enu (MeV) | pdg now | cosmict |
|---|---|---|---|---|---|---|---|
| 0 | **321107** | **248.7** | **88.5°** | **−0.783** | **550.1** | **11** | **1** |
| 1 | 286353 | 271.3 | 71.3° | 2.023 | 624.9 | 11 | 0 |
| 2 | 284013 | 149.6 | 48.9° | 0.880 | 394.1 | 11 | 0 |
| 3 | 277276 | 140.4 | 76.6° | 1.291 | 1566.3 | 11, 2212 | 0 |
| 4 | 57903 | 69.9 | 87.5° | 1.070 | 391.2 | 11 | 0 |
| 5 | 280972 | 68.5 | 71.2° | 3.409 | 3088.2 | 11, 211 | 0 |
| 6 | 315167 | 62.9 | 64.0° | 2.069 | 1354.5 | 11 | 0 |
| 7 | 278684 | 59.2 | 38.9° | −0.485 | 680.0 | 11 | 1 |
| 8 | 287621 | 54.0 | 75.7° | 1.785 | 716.0 | 13 | 0 |
| 9 | 400504 | 50.4 | 84.2° | 1.190 | 508.5 | 11 | 0 |

Notes: 321107 (idx 0) is the reported event and the only one with a negative
`numu_score` *and* a near-perpendicular angle. 287621 (idx 8) already reads
`pdg=13` on its flagged segment — a partial case, not a clean muon-as-electron.
`nue_score` is the −4.300936 / −15.0 sentinel on 10/10, so it carries no
information for this scan.

A verdict on these 10 makes the cut in §3.8 derivable instead of tuned.
