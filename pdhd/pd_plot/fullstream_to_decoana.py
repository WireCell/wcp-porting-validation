#!/usr/bin/env python3
"""Convert PDHD -x full-stream PD waveforms into the decoana snippet format.

The -x APA at z 35-195 (opch 120-159) is read out in *full-stream* mode: one
continuous 343808-sample (~5.5 ms) waveform per channel per event, stored in
rawdump/raw_waveform.  These channels have no LArSoft OpHits and are not in the
self-trigger `decoana` container, so the toolkit light chain never reconstructs
them.

This converter writes the full-stream waveforms into the exact ROOT layout the
existing PDHDOpWaveformSource expects -- one TH1D per channel under
decoana/run_<R>_evt_<E>/ch<N>/raw/waveform_0 -- so the unmodified chain
(PDHDOpWaveformSource -> OpDecon -> OpHitFinder -> OpFlashFinder) reconstructs
them.  Each channel becomes a *single, very long* snippet (343808 bins, x-axis
low edge 0 = tbin 0).  We write only the `raw` subdir (no `deconv`): with no
deconv snippets the source's t_first vote is empty and it falls back to
t_first = rd_timestamp (PDHDOpWaveformSource.cxx:182) -- exactly right, since the
full stream starts at the readout window origin.

The source also requires trigoff/trigger_offset (for the trigger) and
opflashana/PerOpHitTree (it throws if absent); we copy the event's trigger rows
and write an empty PerOpHitTree (its content is unused for full-stream channels).

Usage:
  fullstream_to_decoana.py <input.root> <run> <event> <output.root> [lo hi]
    lo,hi = channel range (default 120 160), half-open.
"""
import sys
import numpy as np
import uproot


def convert(infile, run, event, outfile, lo=120, hi=160):
    fin = uproot.open(infile)

    # --- full-stream waveforms for this event ---
    rw = fin["rawdump/raw_waveform"]
    ev = rw["event"].array(library="np")
    opch = rw["opch"].array(library="np")
    tstamp = rw["timestamp"].array(library="np")  # absolute DTS tick of each waveform start
    sel = np.where((ev == event) & (opch >= lo) & (opch < hi))[0]
    if len(sel) == 0:
        raise SystemExit("no full-stream channels %d-%d for event %d" % (lo, hi, event))
    # The full-stream rows are NOT contiguous in the tree (they are interleaved
    # with the 1024-sample snippet rows), so read the spanning slice and index it
    # by local offset (global index - base), not by enumeration position.
    base = int(sel.min())
    adc_all = rw["adc"].array(library="np", entry_start=base,
                              entry_stop=int(sel.max()) + 1)

    # --- trigger rows for this (run, event) ---
    to = fin["trigoff/trigger_offset"].arrays(library="np")
    tsel = (to["run"] == run) & (to["event"] == event)
    if not tsel.any():
        raise SystemExit("no trigger candidate for run %d event %d" % (run, event))
    # Readout-window origin: the tbin of each waveform is its DTS start relative
    # to rd_timestamp (the source uses t_first = rd_timestamp when no OpHit votes,
    # so OpHit abs time = rd + tbin + sample = timestamp + sample, correct).
    rd = int(to["rd_timestamp"][tsel][0])

    fout = uproot.recreate(outfile)

    # trigoff/trigger_offset -- preserve the branch dtypes the C++ reader binds
    # (run/subrun/event/tc_type Int_t, rd_timestamp/tc_time_candidate ULong64_t,
    #  offset_us Double_t).  Use mktree/extend so these are classic TTrees: uproot
    # writes ROOT::RNTuple for a plain dict assignment, which the C++ ROOT reader
    # (TKey::ReadObj / dynamic_cast<TTree*>) cannot read.
    trig = {
        "run": to["run"][tsel].astype(np.int32),
        "subrun": to["subrun"][tsel].astype(np.int32),
        "event": to["event"][tsel].astype(np.int32),
        "tc_type": to["tc_type"][tsel].astype(np.int32),
        "rd_timestamp": to["rd_timestamp"][tsel].astype(np.uint64),
        "tc_time_candidate": to["tc_time_candidate"][tsel].astype(np.uint64),
        "offset_us": to["offset_us"][tsel].astype(np.float64),
    }
    fout.mktree("trigoff/trigger_offset", {k: v.dtype for k, v in trig.items()})
    fout["trigoff/trigger_offset"].extend(trig)

    # opflashana/PerOpHitTree -- must exist; full-stream channels have no entries,
    # so an empty tree (with the 3 branches the source binds) is sufficient.
    fout.mktree("opflashana/PerOpHitTree",
                {"EventID": np.int32, "OpChannel": np.int32, "PeakTimeAbs": np.float64})

    # decoana/run_<R>_evt_<E>/ch<N>/raw/waveform_<i>.  One long TH1D per
    # full-stream channel (120-159, 1 row); many short snippet TH1Ds per
    # self-trigger channel (0-119, several 1024-tick rows).  The x-axis low edge
    # = tbin = waveform start relative to rd_timestamp, so OpHit absolute time =
    # rd + tbin + sample = timestamp + sample (correct on the trigger-relative
    # clock shared with the other readout mode).
    evname = "decoana/run_%d_evt_%d" % (run, event)
    nrows = 0
    nsamp = 0
    nperch = {}  # channel -> next waveform index
    for gi in sel:
        ch = int(opch[gi])
        adc = np.asarray(adc_all[gi - base], dtype=np.float64)
        n = adc.size
        tbin = int(tstamp[gi]) - rd
        edges = np.arange(n + 1, dtype=np.float64) + tbin
        wi = nperch.get(ch, 0)
        nperch[ch] = wi + 1
        fout["%s/ch%d/raw/waveform_%d" % (evname, ch, wi)] = (adc, edges)
        nrows += 1
        nsamp = n

    fout.close()
    print("wrote %d waveforms over %d channels (last nsamples=%d) -> %s"
          % (nrows, len(nperch), nsamp, outfile))


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        raise SystemExit(1)
    infile, run, event, outfile = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    lo = int(sys.argv[5]) if len(sys.argv) > 5 else 120
    hi = int(sys.argv[6]) if len(sys.argv) > 6 else 160
    convert(infile, run, event, outfile, lo, hi)
