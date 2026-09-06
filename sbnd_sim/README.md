# sbnd_sim — self-contained WCT track-simulation driver for SBND (doc pdvd/47)

Mirrors `pdhd_sim/` and `pdvd_sim/`: `wct-sim-xtrack-sp.jsonnet` runs
`TrackDepos → Drifter → DepoBagger → DepoTransform → Reframer → (AddNoise) → Digitizer →
OmnibusNoiseFilter → OmnibusSigProc (production sbnd/sp.jsonnet) → FrameFileSink` for ONE
anode on a list of straight tracks passed as a TLA, entirely outside LArSoft (the SBND MC in
`sbnd/sbnd_xin/` is LArSoft-produced and cannot be re-simulated here).

Run it through `pdvd/docs/nf_sp_img_clus/scripts/run_d47_xtrack_arms.sh` (`DET=sbnd`), which
compiles every arm first (the compiled-config proof), runs on the pinned libraries and writes
one `<arm> rc=<n>` line per arm. Tracks: `d47_make_xtracks.py --det sbnd`. Noiseless arms bypass
the noise filter (`nf=false`): SBND's `mbOneChannelNoise` flags every flat channel bad whatever
the RMS cuts. The sim channel DB has the data bad-channel list and RMS cuts neutralised.
The SBND base pieces follow `cfg/pgrapher/experiment/sbnd/wct-sim-check.jsonnet` (which cannot
be compiled: it imports the removed `pgrapher/common/fileio.jsonnet`).
