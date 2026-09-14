# doc pdvd/100 round 2 — pre-registration (written 2026-09-13, before own100m was served and before any round-2 arm ran)

Owner, 2026-09-13, after round 1 (gain flip stopped at the pre-registered Michel-purity criterion): "Let's go with your
recommendation, I am happy to scan in display port 5017. I feel that we do not need to have a separate round, but can do it
in the same round for this round." The recommendation was (1) an owner look at the Michel movers between production and
the gain-flip candidate, then (2) a readout-edge guard exemption when the chain finds a Michel (doc 100 §1). Both are this
round. Scans are not blind (owner, same day).

## 1. own100m — what the gain flip moves (owner look, port 5017)

**Set** (`scripts/d100_michel_scan_set.py --out /home/xqian/tmp/p100/mscan`, built before this file was finished; its
listing is `mscan_set.log`). Every object whose chain answer differs between `p99wflip` (production) and `p100c` (gain
ON + v7 wire order + C 0.8630) on the latest carried record, items joined by original record key exactly as `d99_grade.py`
lists movers:
- tranche 1: the 46 `michel_found` movers, **both directions** (record Michel lost 16, found 6; record no-Michel gained 21,
  dropped 3);
- tranche 2: the 19 `is_stm` movers not already in tranche 1.
Shown on `p100c` (its payload, `prep_p100c`), with both arms' chain answers and the record verdict in the question.
Tag `own100m` (new; `own100` is untouched). Label shas of every existing tag taken first: `label_shas_before_r2.txt`.

**Fold.** `d100_michel_scan_score.py` writes a NEW record `scan/pdvd_stm_michel_own100m_verdicts.json`, one item per arm
key (the p100c = p99rwon key and the p99wflip key, same verdict), `confidence="owner"`. UNCLEAR / MESSY stay as labelled
(the graders exclude them as for any record item).

**Corrected records** (new files, `/home/xqian/tmp/p100/carry_r2/`; the carried records are never edited):
`latest_on_p99wflip` and `latest_on_p99rwon` with the verdict of every key present in `own100m` or `own100` replaced by
the owner's (own100m wins where both name a key; none expected).

**Re-grade, both arms on the corrected records:** `d99_grade.py --arms p99wflip:<corrected> p100c:<corrected>
--movers p99wflip,p100c`. **No purity is computed on the mover set itself** — it is selected by the disputed metric.

**Criterion, unchanged from `prereg.md` §2:** `p100c`'s Michel purity on its corrected record within 1.5σ (binomial, the
two arms' errors in quadrature) of `p99wflip`'s on its corrected record. is_stm purity and both efficiencies reported beside
it with the same error. PASS → the gain flip is recommended to the owner (the flip itself, `prereg.md` §4, only on the
owner's yes). FAIL → no flip; the per-volume movers by owner verdict are reported, no threshold is retuned.

Caveat registered now: the owner judges on the gain-ON display. A Michel visible only there is still a Michel, so the
corrected verdict applies to the object on both arms.

## 2. The readout-edge guard exemption (same round)

**Why it cannot be written inside the guard.** `readout_edge_guard` rejects in TaggerCheckSTM, before `Flags::STM` is
set; `michel_found` is computed later in CheckSTM_Michel, which only reads STM-flagged clusters (`require_stm_flag`
default true; absent from PDVD's compiled config). `protect_bundle` sits between them and, with PDVD's
`stm_only_bundles` / `open_convicted_bundles` / `skip_convicted` (default true), opens an STM cluster's bundle and never
splits the STM cluster. The own100 separation (chain `michel_found` 1: 4 stoppers / 0 non-stoppers; 0: 8 / 6) was
measured on arms where the guard did not fire, i.e. with the flag set.

**Knobs (all default OFF; keys omitted when off):**
- TaggerCheckSTM `readout_edge_defer` (C++ false): when the guard fires on a pass, the pass continues exactly as with the
  guard off; if THAT pass accepts the cluster, it gets the cluster scalar `stm_readout_edge` = 1 (Flags::STM as usual).
  An INFO log line names each deferral.
- CheckSTM_Michel `readout_edge_require_michel` (C++ false): a candidate with `stm_readout_edge` = 1 and final
  `michel_found` = 0 (after the T2c / T3c vetoes) gets the new reject bit `R_READOUT_EDGE` (1u<<14). The topology clear
  never clears it. Free function `stm_michel_readout_edge_bits` with a doctest.
- Config: toolkit `protodunevd/pr.jsonnet` arg `stm_readout_edge_defer=false` → the tagger key (emitted only with the
  guard on); the driver forwards it; the CheckSTM_Michel key goes through the existing bag (`stm_michel_extra`).
  Defer without the bag key = the guard effectively off; that is the control arm below, and the comment says so.

**Build and pins.** Round 1's arms ran on `libpin_p96` (Clus 4e1db810); `local/lib` is already a different binary. The
change is built with `wcbuild`, freshness-proved, and copied to a NEW pin `/home/xqian/tmp/p100/libpin_p100b`
(`libpin_p96` untouched). `d100_arms.sh` takes `PIN` from the environment (default unchanged).

**Gates, in order; a FAIL stops the round and is reported with its first divergent event:**
- **X0** compiled config: PDVD PR entry at defaults md5-identical before/after; with `stm_readout_edge_defer=true` exactly
  one added key (`readout_edge_defer`); with the bag key exactly one more.
- **X1** `./build/clus/wcdoctest-clus` passes (new test case included).
- **X2 OFF identity, PDVD:** `p100boff` = PR-only on `p99rwon`'s pctree, pin p100b, `-S stm_recomb_C=0.8630` (p100c's
  config) == `p100c` on 120/120 (`d99rw_identity.py --nt all`: pctree, tlas, every PR branch, mabc-pr). It is
  cross-binary; on a diff, a HEAD build without the change is made to attribute it, and nothing else is read until then.
- **X3 OFF identity, PDHD:** the PDHD PR-only runner (`d53_run_arms.sh DET=pdhd SRC=d16hnu`, production config) on pin
  p96 vs pin p100b, same events, every PR branch identical. SBND: its configs never set either key (grep reported);
  stated by construction, not run.
- **X4 control:** `p100bd` (defer on, no bag key) vs `p100bg` (`stm_readout_edge_guard=false`), both pin p100b on p99rwon's
  pctree with C 0.8630: every PR branch identical on 120/120. This proves the deferral reproduces the guard-off flow the
  separation was measured on. A diff stops the exemption measurement.

**Measurement arms** (pin p100b, PR-only): `p100bx` = p100c's config + defer + bag key (the gain-flip candidate with the
exemption); `p100bxp` = `p99wflip`'s pctree, production config + defer + bag key (what the exemption does to production
today).

**Predictions, fixed now:**
- E1 (implementation): `p100bx` is_stm = `p100c` is_stm ∪ {clusters `p100bg` tags is_stm with `michel_found` 1 whose
  tagger pass was deferred on `p100bd`}; nothing leaves. Every exception is listed.
- E2 (own100): of the 14 latest-side objects the guard removed, `p100bx` restores those with `michel_found` 1 on `p100bg`.
  The own100 payload (`p98vonq`, 10000-tick window, C 0.7941) had 3 of them (40/48, 67/35 and — on the production side —
  48/21, 81/41); the p100bg count is reported beside it, not assumed. `p100bxp` likewise on production's 11 (own100: 3,
  all stoppers).

**Adoption rule, fixed now.** Objects `p100bx` gains over `p100c` that no owner record judges are scanned by the owner
under a NEW tag `own100x` (not blind). The exemption is recommended for production only if the owner-judged gained set's
purity (stoppers / (stoppers + non-stoppers)) is at least `p100c`'s is_stm purity on the corrected record of §1. Below it,
it is reported as "N stoppers for M non-stoppers" and left to the owner. `stm_readout_edge_ticks` is not changed (doc 100
§1: no tick window separates).
