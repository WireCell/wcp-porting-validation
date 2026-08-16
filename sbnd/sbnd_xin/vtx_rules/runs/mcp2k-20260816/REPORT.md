# mcp2k vertex scan — run record

Full write-up: `sbnd_xin/docs/pr/88_mcp2k-vertex-scan.md`. This directory is
the evidence behind it — every scanner pick, every wave's bucketing, every
served pile and calibration draw, as they were written.

Repro:

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# 845 scannable dumps of the 2000-event mcp2k arm (doc pr/88 sec 2)
python3 vtx_rules/selfscan.py prepare --kit new --workers 8 \
    --dumps /home/xqian/tmp/scan-mcp2k/wave-N.txt \
    --out   /home/xqian/tmp/scan-mcp2k/waveN
#   ... N scanner subagents, each handed vtx_rules/scan_worker_prompt.md ...
python3 vtx_rules/selfscan.py review --dir /home/xqian/tmp/scan-mcp2k/waveN
python3 vtx_rules/b2_checkpoint.py  --dir /home/xqian/tmp/scan-mcp2k/waveN
python3 vtx_rules/build_review_pile.py --runs /home/xqian/tmp/scan-mcp2k \
    --rank /home/xqian/tmp/pr88/rank-mcp2k.tsv --allow-partial \
    --drop-unscannable --exclude <prior piles> --out /home/xqian/tmp/pr88/pileN
./pr_display/serve_pr_display.sh 5017 --scan-tag vtxscan-mcp2k <pile-dumps>
```

Kit at wcp `d5f8534`; `scankit.py selftest` 25 dumps / 0 failures (it had been
silently unrunnable since the pr/82 retirement — doc pr/88 sec 1.1).

## Scale

845 events blind-scanned by 60 subagents across 8 waves. The B2 co-located
merge checkpoint passed on every wave: **zero off-list picks across all 845**,
which is the sharp signal that the 0.8 cm merge did not hide a candidate the
scanner wanted.

```
REVIEW                                  418  (49.5%)
auto-accept                             341  (40.4%)
REVIEW (scanner abstained)               49   (5.8%)
REVIEW FIRST (confident disagreement)    37   (4.4%)
```

## The two gates

**Auto-accept, blind 40-event calibration: 39/40 = 97.5%** (95% CI
[86.8%, 99.9%]). Bar was 90%, pre-registered in `piles/calib-draw*.json`
before the owner scanned. 39/39 excluding the one Tier-D event, but 97.5% is
the number of record. The 341-event auto-accept tier is admitted to training
on this gate, not on a human's say-so.

**Scanner confidence, on the 174 owner labels available when it was measured** (doc pr/88 sec 7.1): certain
86.4% / likely 35.0% / unclear 14.6%. Steeper than pr/80's, and the reason
`likely`/`unclear` are not admitted as labels.

## Owner review

242 labels over SEVEN port-5017 instalments (257 served, 15 declined), **112
corrective (46.3%)** against a ~21% base rate. Yield by tier:

```
REVIEW FIRST      70-100%      abstained     67%
ranker-hot fill    38-59%      calibration    3%
```

The fill tier did not decay as instalments went on — it ran 38.9 / 52.0 /
58.8 / 56.5 / 43.2% across instalments 2-6, which was not what I predicted
and is recorded because the prediction was wrong.

## What is deliberately NOT here

Label JSONs. No scan tag in this repo has ever tracked them (`*.json` is
gitignored and every prior tag has 0 tracked files); they live in
`sbnd_xin/vertex_labels/vtxscan-mcp2k/` on disk. Panels are not here either —
~380 MB, regenerable from the dumps by `selfscan.py prepare`.

## Residue, after instalment 6

```
              total  scannable
REVIEW FIRST      4     4
abstained        26     3      <- 23 of 26 are "only dots"
REVIEW disagree  30    17
REVIEW agree    251   228      <- never sampled, see below
auto-accept     301   299
```

The enriched material is **24 scannable events** and that is the end of it.

The 228 scannable `REVIEW`-agreeing events (scanner picked what the
reconstruction picked, but at `likely`/`unclear` confidence) are the one class
with **zero owner labels** — every fill tier selected on `not agrees`, so
their precision is unmeasured. Admitting them would add confirming labels
only; whether that is worth 40 owner-minutes of calibration depends on
whether confirming volume is the training constraint, which the 339
just-admitted auto-accepts will answer for free.

## The training input

`dl_vtx_training/data/pr88_pool_combined/` — **999 events** (213 corrective,
786 confirming; 700 human, 299 AI-scanner), built from all six tags across
four arms with `--drop-unscannable`. Provenance and the two open questions
(no lockbox drawn; 449 unvalidated carried labels) travel with the snapshot
in its `PROVENANCE.md`, and the build is doc pr/88 §8.7.
