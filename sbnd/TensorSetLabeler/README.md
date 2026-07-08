# TensorSetLabeler validation area

Output/validation area for `wclsTensorSetLabeler` (larwirecell `aiml/`), which
sits between the all-APA MultiAlgBlobClustering and its TensorFileSink in
`wcls-img-clus-matching-xin.fcl` (MC only, `truth_labeler = reality=='sim'` in
the entry jsonnet) and attaches truth to the clustering ITensorSet:

- tensor-set metadata: `runNo/subRunNo/eventNo`, `nu_pdg/nu_ccnc/nu_int_type/
  nu_energy/nu_vtx_{x,y,z}/nu_flavor` (from `generator` MCTruth),
- a `truth_per_track` 2D tensor `[ntracks x 22]` (columns in the tensor
  metadata; cm/ns/GeV).  With `truth_tracks_nu_only` (default true) only
  BEAM-neutrino primaries are saved — the larreco CellTree "nuOnly" cut
  (assns MCTruth `Origin()==kBeamNeutrino` + primary; CellTree's
  `Mother()==0` becomes `Process()=="primary"` under the trackid-offset
  scheme).  Rockbox events can carry several beam-nu interactions:
  primaries of ALL of them are kept, and the `nu_idx` column records each
  row's MCTruth index (0 = the interaction the `nu_*` metadata describes;
  -1 = non-beam rows in the full table).  No cosmic-muon or secondary
  truth (4-event check: 6/3/14/3 rows); `false` restores the full
  `largeant` MCParticle table,
- per-blob dominant G4 `trackid` written into each live blob `scalar` PC
  (`-1` = no matching deposit), from a BlobDepoFill-style association of
  `ionandscint:priorSCE` SimEnergyDeposits with blob (slice tick, u/v/w wire)
  bounds in the raw (non-t0-corrected) coordinates,
- a `truth_trackid` Bee points set in the shared `mabc.zip` (raw x,y,z;
  `cluster_id` = blob trackid) for visual validation.

## Run (SL7 + setup-ap.sh)

```bash
export FHICL_FILE_PATH=/exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd:$FHICL_FILE_PATH
cd /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/TensorSetLabeler
lar -n 10 -c wcls-img-clus-matching-xin.fcl \
    -S /exp/sbnd/app/users/yuhw/2025-fall-prod-sample/mc_paths-10files.lst --no-output
```

Outputs land in cwd: `mabc.zip` (Bee, incl. truth_trackid) and
`trash-all-apa.tar.gz` (labeled tensors; with `truth_labeler` on the sink's
`dump_mode` is off so the tar actually contains the labeled pctree +
truth_per_track + metadata).

## Sanity checks

```bash
python3 check_truth_trackid.py mabc.zip trash-all-apa.tar.gz
```

10-event MC (corsika+GENIE 2025C Fall, run 32/31): every event non-empty;
44-71% of points labeled; nearest-neighbor trackid coherence 99.7-100%
(visually close points share the trackid).  Unlabeled blobs are dominated by
ghosts: per-event diagnosis showed ~2/3 of them have no SimEnergyDeposit
consistent in >=2 of 3 wire views (2-view recovery / deghosting leftovers) and
they carry half the median charge of labeled blobs.

BEE (10 events): https://www.phy.bnl.gov/twister/bee/set/f11538d6-a265-4546-8617-ffcb10716887/event/list/
Compare `truth_trackid` against `clustering` — same geometry, colors = true
tracks.

## Time-offset guidance (the "tricky" part)

Raw blob x is `x = x_Wplane + dirx*(t_sig + time_offset)*drift_speed`
(BlobSampler, SBND `time_offset = -205us` = `sim.tick0_time`, `drift_speed =
1.563 mm/us`, `tick = 0.5us`).  The WCT sim arranges an in-time depo to
reconstruct at its true x, so the inverse map for a depo at `(x_true, t_dep)`:

```
x_app  = x_true + dirx*drift_speed*t_dep
t_sig  = (x_app - x_Wplane)*dirx/drift_speed - time_offset
itick  = floor(t_sig/tick)
```

The labeler's `drift_speed/time_offset/tick` MUST match the BlobSampler
config (they are threaded from the same locals in
`cfg/pgrapher/experiment/sbnd/clus.jsonnet`); `depo_time_offset` (default 0)
absorbs any residual sim-frame shift.  `wire_slop=1`/`tick_slop=2` widen the
blob bounds to stand in for the diffusion extents that BlobDepoFill would
integrate (SimEnergyDeposits are point-like); without slop ~1/4 of real blobs
are missed at one-plane boundaries.

## SCE correction of the depos (low-label-rate fix)

The blobs are reconstructed from post-SCE (spatially distorted) charge while
`ionandscint:priorSCE` depos are at TRUE positions — up to ~1.4 cm apart vs
3 mm wire pitch.  The labeler therefore applies (default on, `sce_correction`)
the TrueFwd (true→reco) displacement from a second `SCEFieldTH3` instance
(`sbnd_dualmap_fwd`, `th3_name_* = TrueFwd_Displacement_*`, `sign: 1`) to each
depo before association.  Probes (logged at configure): |d| ≈ 0.8–1.0 cm near
the cathode, ≈ 0.1 cm near the anodes, opposite in direction to the
SCECorrection (reco→true) probes.

Event 1 comparison (blobs labeled / points labeled):
- no SCE: 2347/4031 (58%) / 64.8% of points —
  https://www.phy.bnl.gov/twister/bee/set/c98b73cd-a5cb-4f0d-9af1-7cca11b6231e/event/list/
- with SCE: 2690/4031 (67%) / 76.7% of points, NN-coherence 99.9% —
  https://www.phy.bnl.gov/twister/bee/set/8fcc50ca-1889-4b0d-9f44-5b02dbd495f2/event/list/

Debug Bee sets (both uploads): `truth_unlabeled` shows only the points of
unlabeled blobs (cluster_id = reco cluster ident); `data/{i}/{i}-mc.json`
holds the truth particle-flow tree (MCParticles with KE > 10 MeV
(`pf_ke_min`), nested under the nearest kept ancestor by Mother() tracing;
node id = G4 trackid, cross-references the `truth_trackid` cluster ids).
The mc tree additionally applies, in order:
- `pf_nu_only` (default true): ONLY particles derived from a beam neutrino
  (assns MCTruth `Origin()==kBeamNeutrino`) — no cosmics at all.  Event 1:
  just the numu + 2 vertex protons over 10 MeV:
  https://www.phy.bnl.gov/twister/bee/set/a0edceb4-1494-4a97-bc2f-bfaeacfd82c2/event/list/
- with `pf_nu_only: false`, an FV cut instead (`pf_fiducial` = the
  `all-overall-fv` BoxFiducial): a cosmic is kept if its start OR end point
  lies inside the FV, or the start-end line section crosses it (sampled
  every 5 cm) — through-going multi-GeV cosmic muons then appear as roots
  (event 1: 43 nodes, set 59c44411).  Beam-nu-derived particles always
  skip the FV cut.

## Files

- `check_truth_trackid.py` — the sanity checker described above.
- `run-10evt.log`, `check-10evt.txt` — the validated 10-event run + checks.
- `mabc.zip`, `trash-all-apa.tar.gz` — 10-event outputs (uploaded to BEE).
