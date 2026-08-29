# doc pr/124 — EM residuals round: gap band, recognition, pass3_cone

**Status: IN PROGRESS (opened 2026-08-28 night, owner directive).**

Follow-on to doc pr/123 (pass4 over-reach round, closed VALIDATED/ON). The
owner picked three fronts from the post-pr/123 residual assessment for tonight,
"the rest ... the same as before" — i.e. the standing pr/117–123 bar:
measure-first, default-OFF knobs, dual-manifest byte-identical gates
(`scripts/pr85_hash_gate.py`), census + movers adjudication, Bee A/B, flip
only on validation, commit + push.

Owner wording (2026-08-28): start work on

1. *"The 25–40 cm gap band (over-clustering, round-3 candidate). G=40 was
   deliberately conservative. ... The offline prune-scan can sweep G with
   qualifiers (component charge, PID mix, direction w.r.t. the body) against
   the existing labels with zero new arms — the question is whether a
   qualifier tames the 1:1 collateral that killed G=25."*
2. *"Recognition / fake showers. 489327, 69232, 54332's 16014, plus the
   171143/277298 fakes — the root itself is wrong, so no membership knob
   helps. ... a seed-time re-classification campaign is its own measure-first
   round. Related housekeeping: the score-100 sentinel class (rule-assigned
   PIDs) is systematically excluded from every 'confident' gate — worth an
   audit."*
3. *"pass3_cone — now the largest untouched absorber (20 OUT marks on the
   141-set). ... the final-body prune is absorber-agnostic, so pass3's
   detached mistakes are already partially handled; what remains is its
   contiguous share. Same probe-census choreography as pass4 would settle
   it."*

The neutrino-vertex identification thread stays closed (owner, pr/123).

## 0. Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# Front A -- gap-band qualifier scan (offline, zero new arms):
./scripts/pr124_gapband_scan.py 'work-pr123r1-r21flip141-*'

# Front B -- sentinel audit is grep-only (sec B.1); seed census after dbg arms:
#   WCT_SHOWER_ABSORB_DEBUG=1 WCT_SHOWER_TOPO_DEBUG=1 dbg arms, then
./scripts/pr124_seed_census.py 'work-pr124r1-dbg-*' 'work-pr124r1-dbg98-*'

# Front C -- pass3 absorber census from the same dbg arms:
./scripts/pr124_pass3_census.py 'work-pr124r1-dbg-*' 'work-pr124r1-dbg98-*'
```

Baselines: 98-set `work-d84r2-prod98-*`; 141-set `work-pr123r1-r21flip141-*`
(current production, incl. pr/123 r2.1 pseudo-neutron correction). Labels
(read-only, M13): `em_labels/emscan-0827` (98-manifest) and
`em_labels/emscan-0828-agent5` (141-manifest).

**Concurrency note (opening night).** The doc-84 session's 3000-event census
(`work-d84r3-cens-mcp{1,2}k`, full current production config) was running when
this round opened; per its request, no builds/installs or shared-jsonnet edits
until it completes. Phase 1 here is purely offline. Once landed,
`work-d84r3-cens-*` becomes the shared full-sample baseline and — logs
permitting — a free 3000-event out-of-sample input for the front B/C censuses
(read-only over its dirs, no reruns).

## A. Front A — the 25–40 cm gap band (prune round 3 candidate)

pr/123 shipped the final-body single-linkage prune at G=40 cm
(`shower_pass4_prune_detached` / `shower_pass4_prune_gap`, SBND ON). G=25/30
were rejected because the collateral was 1:1 — every additional OUT member
pruned cost roughly one TARGET member (the 54332-122091 / 281325 satellite
class; the owner later ruled those two rows vertex-bad, but the bar for a
tighter G remains zero collateral on correctly-vertexed showers).

Worst post-pr/123 residual rows, all sitting in the 25–40 band:

| event | qF1 (r21flip141) | note |
|---|---|---|
| 406125 | 0.097 | q_extra 2.6e6; all four OUT marks pruned at G=25 in the pr/123 offline scan |
| 94392  | 0.221 | |
| 175896 | 0.327 | |
| 286655 | low | band component(s) |
| 283515 | partial | parts in band |
| 278420 | — | contiguous far chain, first link 23.9 cm — inside even G=25; NOT a gap-band target, listed as a known non-catch |

**Question under measurement:** with G tightened into the band (25/30/35),
does a per-component qualifier — summed charge (absolute and as a fraction of
the shower), PID mix (track-like member 13/211/2212), summed length, component
direction w.r.t. the kept body axis, gap distance — separate the OUT
components from the TARGET (collateral) components on BOTH label sets?

Tool: `scripts/pr124_gapband_scan.py` (fork of `pr123_prune_scan.py`; same
final-body single-linkage machinery, adds per-component qualifier columns and
per-qualifier collateral tables). Runs on the `work-pr123r1-r21flip141-*`
calib dumps — production arms, so components already surviving the G=40 prune.

Guard rows (must NOT newly regress owner-approved outcomes): 54332-122091,
281325, 348471 (0.912, sheds only unlabeled 31011 today), and the 8 rows at
1.000 from pr/123.

Outcome gate: a qualifier reaching zero collateral on both manifests graduates
to a default-OFF second-tier knob (`shower_pass4_prune_gap2` + qualifier
param); anything short of that ships as a measurement section only (the
pr/118-P2 / pr/119 precedent), or as a morning decision table if the trade-off
is a genuine owner call.

*(results to follow)*

## B. Front B — recognition / fake showers + the score-100 sentinel audit

Cases where the shower ROOT is wrong, so no membership knob helps: 489327,
69232, 54332's seg 16014, and the 171143 / 277298 fakes. The pr/122 finding
stands: long seeds carry inherited, never re-validated `kShowerTopology`
flags; the `in_main_cluster` seeder accepts the flag with no further test.

Existing instrumentation (reused, no new probes expected for round 1):

- `SHOWER_SEED` line (`pr122_probe_seed`, NeutrinoShowerClustering.cxx:735,
  under `WCT_SHOWER_ABSORB_DEBUG`): per accepted seed — which disjunct
  (traj/topo/pdg11), length, median dQ/dx, in_long_muon.
- `WCT_SHOWER_TOPO_DEBUG` (PRSegmentFunctions.cxx:4112-4124): evaluation-time
  branch + features for every `segment_is_shower_topology` verdict.
- Existing default-OFF knobs at the choke points: `shower_topo_demote_len`,
  `shower_topo_dqdx_guard`, `shower_traj_straight_guard`.

Census (`scripts/pr124_seed_census.py`): every accepted seed × disjunct ×
seed-time features × label verdict on the resulting shower (fake/real), both
manifests. Measured question: does any existing knob threshold — or a
seeder-side re-validation (re-running the topology classifier on seed-time
geometry) — separate the fake-seed class from real stems with zero collateral?
pr/122 measured the *classifier-side* features interleaved; the new angle is
seed-time staleness (flag written on early geometry, segment since regrown).

### B.1 The score-100 sentinel audit (housekeeping, read-only)

Rule-assigned PIDs carry `particle_score = 100`; every "confident PID" gate
written as `particle_score < 1.0` (or similar) silently excludes them. The
pr/123 lost-muon defect was one instance (`segment_orphan_confident_track`).
This section enumerates every such site in `clus/` with a verdict:
load-bearing (the gate SHOULD skip rule-assigned PIDs) vs accidental (the
sentinel class was simply forgotten). Audit only — no code change in this
round without owner sign-off per site.

*(table to follow)*

## C. Front C — pass3 absorbers (largest untouched)

pass3 has three absorb sites (NeutrinoShowerClustering.cxx): `pass3_proximity`
(:1388), `pass3_cone` (:1538), `pass3_cluster_map` (:1583). pass3_cone carries
20 OUT marks on the 141-set — the largest absorber not yet measured with the
pr/120/123 choreography. Notes:

- The pr/123 final-body prune is absorber-agnostic: pass3's *detached*
  mistakes are already (partially) reclaimed. The open share is contiguous.
- Existing instrumentation (reused): `P120_P3CONE` line (~:1490, pr/120
  admission census: seg, pdg, len, site angle, dist, scan-frame ang15/ang60,
  angle_offset) + `SHOWER_ABSORB DIRECT site=pass3_*` tags, all under
  `WCT_SHOWER_ABSORB_DEBUG`.
- An existing decline knob already guards one class:
  `shower_cone_absorb_guard` (pr/93 Cause D) for confidently-PID'd
  straight-long non-electrons.

Census (`scripts/pr124_pass3_census.py`, fork of `pr123_pass4_census.py`):
every pass3 absorb joined to labels; classify contiguous-vs-detached against
the final body (detached = already covered by the prune); feature table
(site_ang, dist, ang15/ang60, pdg, len, dQ/dx) for OUT vs IN absorbs. Outcome
gate as in front A: zero-collateral separator → default-OFF guard knob;
otherwise the census ships as measurement.

*(results to follow)*

## Plan of record (tonight)

1. Doc opened, committed, pushed (this commit).
2. Front A offline scan on existing dumps → §A tables.
3. §B.1 sentinel audit → table.
4. After the d84r3 census window clears: dbg arms
   `work-pr124r1-dbg-{mcp1k,mcp2k}` (141) + `work-pr124r1-dbg98-*` (98),
   production knobs + probe envs, PR_JOBS ≤ 16, hash-gated vs baselines
   (probes are stderr-only ⇒ byte-identical required).
5. Front B/C censuses → §B/§C tables; 3000-event read-only pass over
   `work-d84r3-cens-*` if logs permit.
6. Knobs only where a separator measures zero-collateral on both label sets;
   ambiguous thresholds → morning decision table for the owner. Existing-knob
   default flips are never made overnight.
