#!/usr/bin/env python3
"""Extract per-event trigger time / readout-window offset from DUNE DAQ raw HDF5.

For each TriggerRecord in the file(s) this prints:
  - trigger number / sequence
  - trigger timestamp (DTS ticks, 16 ns) from the TriggerRecordHeader
  - TriggerCandidate type (beam / cosmic / laser / ...) and time_candidate
  - TPC fragment window_begin / window_end
  - offset = trigger_timestamp - window_begin  (the trigger-to-waveform-start
    offset: ~2500 us for cosmics, ~411 us for beam in PDVD)

Pure h5py + struct: no DUNE DAQ software needed.  Binary layouts taken from
  daqdataformats  FragmentHeader v5      (marker 0x11112222)
  daqdataformats  TriggerRecordHeader v4 (marker 0x33334444)
  trgdataformats2 TriggerCandidateData v3

Usage:
  pdvd_event_trigger_offset.py FILE_OR_DIR [...]   # .hdf5 files or dirs of them
  options: --csv OUT.csv   also write a csv
"""

import argparse
import csv
import struct
import sys
from pathlib import Path

import h5py

DTS_TICK_US = 16e-3  # 62.5 MHz DTS clock -> 16 ns per tick

TC_TYPE_NAMES = {
    0: "kUnknown", 1: "kTiming", 2: "kTPCLowE", 3: "kSupernova", 4: "kRandom",
    5: "kPrescale", 6: "kADCSimpleWindow", 7: "kHorizontalMuon",
    8: "kMichelElectron", 9: "kPlaneCoincidence", 10: "kDBSCAN",
    11: "kChannelDistance", 12: "kBundle", 13: "kCTBFakeTrigger",
    14: "kCTBBeam", 15: "kCTBBeamChkvHL", 16: "kCTBCustomD", 17: "kCTBCustomE",
    18: "kCTBCustomF", 19: "kCTBCustomG", 20: "kCTBBeamChkvHLx",
    21: "kCTBBeamChkvHxL", 22: "kCTBBeamChkvHxLx", 23: "kNeutronSourceCalib",
    24: "kChannelAdjacency", 25: "kCIBFakeTrigger", 26: "kCIBLaserTriggerP1",
    27: "kCIBLaserTriggerP2", 28: "kCIBLaserTriggerP3", 29: "kCTBOffSpillSnapshot",
    30: "kCTBOffSpillCosmicJura", 31: "kCTBOffSpillCRTCosmic", 32: "kCTBCustomA",
    33: "kCTBCustomB", 34: "kCTBCustomC", 35: "kCTBCustomPulseTrain",
    36: "kDTSPulser", 37: "kDTSCosmic", 38: "kSSPLEDCalibration",
}

FRAG_HEADER_SIZE = 72


def parse_trh(raw):
    """TriggerRecordHeaderData: marker,u32 ver,u32 | trig_num u64, trig_ts u64,
    n_components u64 | run u32, err u32 | type u16, seq u16, max_seq u16."""
    marker, version = struct.unpack_from("<II", raw, 0)
    if marker != 0x33334444:
        raise ValueError(f"bad TRH marker 0x{marker:08x}")
    trig_num, trig_ts, _ncomp = struct.unpack_from("<QQQ", raw, 8)
    run, _err = struct.unpack_from("<II", raw, 32)
    _ttype, seq, _maxseq = struct.unpack_from("<HHH", raw, 40)
    return dict(trig_num=trig_num, trig_ts=trig_ts, run=run, seq=seq)


def parse_frag_header(raw):
    marker, version = struct.unpack_from("<II", raw, 0)
    if marker != 0x11112222:
        raise ValueError(f"bad fragment marker 0x{marker:08x}")
    trig_ts, win_begin, win_end = struct.unpack_from("<QQQ", raw, 24)
    return dict(trig_ts=trig_ts, win_begin=win_begin, win_end=win_end)


def parse_tc_data(raw):
    """trgdataformats2::TriggerCandidateData v3 (payload after fragment header):
    u8 version, pad7 | u64 time_start, time_end, time_candidate |
    u8 detid, pad3 | i32 type | i32 algorithm."""
    version = raw[0]
    t_start, t_end, t_cand = struct.unpack_from("<QQQ", raw, 8)
    detid = raw[32]
    tc_type, algo = struct.unpack_from("<ii", raw, 36)
    return dict(version=version, time_start=t_start, time_end=t_end,
                time_candidate=t_cand, detid=detid, type=tc_type, algo=algo)


def process_record(rec):
    out = {}
    trh_ds = tc_ds = tpc_ds = None
    rawdata = rec.get("RawData")
    if rawdata is None:
        return None
    # TPC fragment: must be an Ethernet TPC stream (WIBEth for HD / VD bottom,
    # TDEEth for VD top).  Other Detector_Readout datasets (DAPHNEStream PDS,
    # CRT, ...) have different frame formats and possibly different windows.
    for name in sorted(rawdata):
        if name.endswith("TriggerRecordHeader"):
            trh_ds = rawdata[name]
        elif "Trigger_Candidate" in name and tc_ds is None:
            tc_ds = rawdata[name]
        elif ("Detector_Readout" in name and tpc_ds is None
              and (name.endswith("WIBEth") or name.endswith("TDEEth"))):
            tpc_ds = rawdata[name]

    if trh_ds is not None:
        out.update(parse_trh(trh_ds[:].tobytes()))
    if tpc_ds is not None:
        # 72-byte fragment header + first frame's DAQEthHeader (timestamp at +8)
        raw = tpc_ds[: FRAG_HEADER_SIZE + 16].tobytes()
        out["tpc"] = parse_frag_header(raw)
        if len(raw) >= FRAG_HEADER_SIZE + 16:
            out["tpc"]["first_frame_ts"] = struct.unpack_from(
                "<Q", raw, FRAG_HEADER_SIZE + 8)[0]
        out["tpc_name"] = tpc_ds.name.rsplit("/", 1)[-1]
    if tc_ds is not None:
        raw = tc_ds[:].tobytes()
        parse_frag_header(raw)  # validate marker
        if len(raw) > FRAG_HEADER_SIZE + 44:
            out["tc"] = parse_tc_data(raw[FRAG_HEADER_SIZE:])
    return out


def process_file(path, writer=None):
    print(f"\n=== {path} ===")
    hdr = (f"{'trig':>8} {'seq':>4} {'tc_type':<24} {'trig_ts (DTS)':>18} "
           f"{'win_begin':>18} {'offset_us':>10} {'frame_off_us':>12} {'win_us':>9}")
    print(hdr)
    with h5py.File(path, "r") as f:
        for recname in sorted(f.keys()):
            rec = f[recname]
            if not isinstance(rec, h5py.Group):
                continue
            try:
                r = process_record(rec)
            except ValueError as e:
                print(f"{recname}: {e}")
                continue
            if not r or "tpc" not in r:
                continue
            trig_ts = r.get("trig_ts", r["tpc"]["trig_ts"])
            offset_us = (trig_ts - r["tpc"]["win_begin"]) * DTS_TICK_US
            win_us = (r["tpc"]["win_end"] - r["tpc"]["win_begin"]) * DTS_TICK_US
            frame_ts = r["tpc"].get("first_frame_ts")
            frame_off_us = (trig_ts - frame_ts) * DTS_TICK_US if frame_ts else float("nan")
            tc = r.get("tc")
            tc_name = TC_TYPE_NAMES.get(tc["type"], f"type{tc['type']}") if tc else "n/a"
            print(f"{r.get('trig_num', -1):>8} {r.get('seq', -1):>4} {tc_name:<24} "
                  f"{trig_ts:>18} {r['tpc']['win_begin']:>18} "
                  f"{offset_us:>10.2f} {frame_off_us:>12.2f} {win_us:>9.1f}")
            if writer:
                writer.writerow([Path(path).name, r.get("run"), r.get("trig_num"),
                                 r.get("seq"), tc_name, trig_ts,
                                 tc["time_candidate"] if tc else "",
                                 r["tpc"]["win_begin"], r["tpc"]["win_end"],
                                 f"{offset_us:.3f}", f"{frame_off_us:.3f}",
                                 f"{win_us:.3f}"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".hdf5 files or directories")
    ap.add_argument("--csv", help="also write results to this csv file")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        p = Path(p)
        files.extend(sorted(p.glob("*.hdf5")) if p.is_dir() else [p])
    if not files:
        sys.exit("no hdf5 files found")

    writer = None
    csvfh = None
    if args.csv:
        csvfh = open(args.csv, "w", newline="")
        writer = csv.writer(csvfh)
        writer.writerow(["file", "run", "trig_num", "seq", "tc_type", "trig_ts",
                         "tc_time_candidate", "win_begin", "win_end",
                         "offset_us", "frame_offset_us", "window_us"])
    for fp in files:
        process_file(fp, writer)
    if csvfh:
        csvfh.close()
        print(f"\ncsv written: {args.csv}")


if __name__ == "__main__":
    main()
