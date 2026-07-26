# batch2/ — raw sub-agent returns for doc 61 §5b

**`../handscan-batch2.tsv` is the canonical record.** It has 402 rows, one per
in-beam bundle of the 383 events, with the `--ai-scan` join keys
(`flash_gid`, `t_us`, `len_cm`, `auto_label`) taken from each event's own
`nusel-evt<ID>.tsv` rather than typed by an agent.

`slice-verdicts/s*.tsv` are what the ten agents actually returned
(`event main_id verdict quality conf reason`). **Do not concatenate them as a
scan record**: each slice also contains the six *control* bundles that were
seeded into all ten slices, so the concatenation is 462 rows with 60 duplicates.
The controls (48301:13, 48895:17, 50787:6, 48367:19, 51865:11, 52723:7) were
already scanned in `../handscan-first20.tsv`; the merge drops them from the batch
TSV and turns them into the agreement table instead. Where a control's *reason*
text differs between the two batches (52723:7 reads as proton-like here, "vertex
+ stub" in batch 1) the verdicts still agree — batch 1 is the earlier record.

`merge_verdicts.py` rebuilds `../handscan-batch2.tsv` from `slice-verdicts/`
plus the event tables and is also the gate: it fails on a missing, extra,
duplicate or key-unmatched row and on any illegal `verdict`/`quality`/`conf`,
and prints the tagger-vs-scan matrix, the per-slice STM rate and the control
agreement. It reads the per-slice files from `/home/xqian/tmp/nusel-b2/verdicts/`
(the working tree), so point `B` at this directory's `slice-verdicts/` to re-run
it here.

`AGENT_INSTRUCTIONS.md` is the operating manual the ten agents were given — the
scan criteria in the form that actually produced these verdicts, including the
two-value verdict rule and the containment arithmetic against `DET_BOX`/`FV_BOX`.
