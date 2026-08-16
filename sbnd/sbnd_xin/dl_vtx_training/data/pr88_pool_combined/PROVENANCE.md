# pr88_pool_combined — 999 events, the round-3 training input

Built 2026-08-16 (doc pr/88 §8.7). **Read this before training on it**: the
999 are not 999 events of equal quality, and three of the differences are not
visible in `manifest.tsv` unless you look for them.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 dl_vtx_training/build_dataset.py --name pr88_pool_combined \
  --tags vtxscan-mcp2k vtxscan-mcp2k-auto vtxscan-harv3-nuecc48 \
         vtxscan-harv3-ncpi0 vtxscan-harv3-mcp1k vtxscan-harv3-delta \
  --harvest-roots vtxscan-mcp2k=work-mcp2k-harv3 \
                  vtxscan-mcp2k-auto=work-mcp2k-harv3 \
                  vtxscan-harv3-nuecc48=work-nuecc48-harv3 \
                  vtxscan-harv3-ncpi0=work-ncpi0-harv3 \
                  vtxscan-harv3-mcp1k=work-mcp1k-harv3 \
                  vtxscan-harv3-delta=@arm \
  --drop-unscannable
→ wrote 999 events (213 corrective, 786 confirming)
  "only dots" events (longest fitted segment < 5.0 cm): 15 DROPPED
```

`vtxscan-harv3-delta=@arm` is not a typo: that tag holds 19 events on
`work-mcp1k-harv3` and 5 on `work-nuecc48-harv3` (it is the tag for events a
human re-answered, wherever they lived), and `--harvest-roots` maps one root
per tag. `@arm` resolves each label to its own recorded arm. It is opt-in per
tag because forcing an explicit arm is the point of harvest mode everywhere
else — pr/79 §11 exists so labels are paired with the dumps the live net
actually saw.

## Composition

| | n |
|---|---:|
| **total** | **999** |
| human-labelled | 700 |
| AI-scanner, admitted by the pr/88 §7 gate | 299 |
| corrective (label ≠ reconstruction) | 213 |
| confirming | 786 |

By tag:

```
vtxscan-mcp2k          240   human, doc pr/88 owner review (111 corrective)
vtxscan-mcp2k-auto     299   AI-scanner, gated                (2 corrective)
vtxscan-harv3-mcp1k    375   human, current epoch            (79 corrective)
vtxscan-harv3-delta     24   human, re-answered              (10 corrective)
vtxscan-harv3-nuecc48   42   human                            (6 corrective)
vtxscan-harv3-ncpi0     19   human                            (5 corrective)
```

## Three things to know before you train

**1. 299 labels were never seen by a human.** `label_source` in the manifest
and in every npz says which: `ai-scanner` vs `human`. They are blind-scanner
picks admitted by a measured gate — doc pr/88 §7, a blind 40-event
calibration on this sample, 39/40 = 97.5% correct at 1 cm, 95% CI
[86.8%, 99.9%] against a 90% bar. That is a *measured* 2.5% expected label
error, not a guarantee. Weight or ablate on `label_source` if it matters.

**2. 449 of the current-epoch labels are an unvalidated carry.** pr/82 §4.3:
they were bulk-written in one second at 08:44 on 2026-08-16 by carrying
earlier-epoch picks forward onto the harv3 arms, and the pre-registered ≥95%
blind re-scan of ~60 of them **never ran**. That is ~45% of this pool resting
on a carry nobody checked. It was correctly out of scope for the scan round;
it is *not* out of scope for a training input, and it is the largest single
quality unknown here.

**3. There is no lockbox.** `lockbox` is 0 for all 999, deliberately. The
current-epoch 473 have never been built into a snapshot (`harv473` is the
prod0813 tags on prod0813 arms), so there is no manifest to
`--inherit-manifest` from — and `--inherit-manifest` keys on `(scan_tag,
evt)`, which the epoch relabelling changed for every row. A fresh
`--lockbox` draw here would produce a *different* held-out set from the one
pr/79–81 already spent, quietly contaminating any held-out claim made against
those numbers. **How the old lockbox carries across the epoch relabelling is
an open question and the training round has to answer it before any held-out
result from this snapshot means anything.**

## What was dropped

15 "only dots" events (longest fitted segment < 5 cm), the owner's own cut —
*"they are impossible to hand scan the true vertices, so I just skipped them.
For our later fine tuning etc, we should filter out this kind of events."*
Definition and validation in `vtx_rules/scannability.py`. 1014 labels − 15 =
999.

## Two manifest columns that mislead

- `corrective` is `dis_to_main > 1e-9` — *any* nonzero separation, not the
  1 cm tolerance used everywhere else in pr/78–88. That is why 2 of the 299
  auto labels count corrective despite agreeing with the reconstruction
  inside tolerance. Do not read 213 as "213 events where the reconstruction
  was wrong"; recompute at 1 cm if that is the quantity you want.
- `sample` is derived from the tag string, so every mcp2k event lands in
  `nuecc` by fallthrough (`sample_of_label` matches `ncpi0`/`mcp1k` and
  defaults to `nuecc`). The 605 `nuecc` are not 605 nueCC events.
