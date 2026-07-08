# TensorSetLabeler validation area

Output/validation area for `wclsTensorSetLabeler` (larwirecell `aiml/`), which
sits between the all-APA MultiAlgBlobClustering and its TensorFileSink in
`wcls-img-clus-matching-xin.fcl` (MC only, `truth_labeler = reality=='sim'` in
the entry jsonnet) and attaches truth to the clustering ITensorSet:

- tensor-set metadata: `runNo/subRunNo/eventNo`, `nu_pdg/nu_ccnc/nu_int_type/
  nu_energy/nu_vtx_{x,y,z}/nu_flavor` (from `generator` MCTruth),
- a `truth_per_track` 2D tensor `[ntracks x 21]` (one row per `largeant`
  MCParticle; columns in the tensor metadata; cm/ns/GeV),
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

## Files

- `check_truth_trackid.py` — the sanity checker described above.
- `run-10evt.log`, `check-10evt.txt` — the validated 10-event run + checks.
- `mabc.zip`, `trash-all-apa.tar.gz` — 10-event outputs (uploaded to BEE).
