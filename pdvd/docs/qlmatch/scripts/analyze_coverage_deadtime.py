#!/usr/bin/env python3
"""Is an uncovered PDVD self-trigger channel BUSY (World B) or ARMED-AND-QUIET (World A)?

Read-only.  Reconstructs each DAPHNE snippet's live interval from the rawwf tree in
the chain's own tick base (PDVDOpWaveformSource.cxx:123-124: start = llround(ts*62.5)
- 64 for nsamp<=1024), rebuilds OpFlashFinder's per-flash coverage exactly
(OpFlashFinder.cxx:711-763: window = [min,max] ophit peak_time over the flash's
refined hits; per-OpDet frac = MIN over ganged sub-channels), and then asks, for every
(flash, opdet) the chain marks uncovered:

    was this channel reading out something else just before the flash window?

World B (busy)  => a snippet ends shortly before t_lo; the silence is dead time and
                   carries no information about the light -> excluding is right.
World A (armed) => no snippet anywhere near; the channel was listening and simply did
                   not cross threshold -> PE ~ 0 is a real measurement, excluding it
                   throws away an upper limit.

Validation gate: the rebuilt coverage must reproduce the chain's own flash_cov tensor.
If it does not, the tick base / ganging is being misapplied -- fix that first.

Repro:
    cd wcp-porting-img/pdvd
    python3 <this> [event]
"""
import sys, glob, json
import numpy as np
import uproot

PD = "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd"
RAW = PD + "/input_data_light/np02vd_raw_run039252_1176_df-s03-d3_dw_0_20250830T054542_rawwf.root"
LIGHT = PD + "/work/039252_light%d_spcov"
TICK_US = 0.016
TRIG_SAMPLE = 64
SNIPPET_NSAMP = 1024


def load_tensors(evt):
    """name -> array, from the light archive's already-extracted npy files."""
    import tarfile, io, tempfile, os
    d = LIGHT % evt
    tf = tarfile.open(d + "/opflash_pdvd-wct.tar.gz")
    names, arrs = {}, {}
    for m in tf.getmembers():
        if m.name.endswith("_metadata.json") and "_tensor_" in m.name:
            idx = m.name.split("_tensor_")[1].split("_")[1]
            names[idx] = json.load(tf.extractfile(m))["name"]
    for m in tf.getmembers():
        if m.name.endswith("_array.npy"):
            idx = m.name.split("_tensor_")[1].split("_")[1]
            arrs[names[idx]] = np.load(io.BytesIO(tf.extractfile(m).read()))
    return arrs


def snippets(evt):
    """raw opchannel -> sorted list of (t_begin, t_end) us, chain base; plus raw->od map."""
    f = uproot.open(RAW)
    a = f["rawdump/raw_waveform"].arrays(
        ["run", "event", "opchannel", "opdet", "nsamp", "timestamp"], library="np")
    m = (a["run"] == 39252) & (a["event"] == evt)
    ch, od = a["opchannel"][m], a["opdet"][m]
    ns, ts = a["nsamp"][m], a["timestamp"][m]
    start = np.round(ts * 62.5) - np.where(ns <= SNIPPET_NSAMP, TRIG_SAMPLE, 0)
    # PDVDOpWaveformSource.cxx:155 builds the frame at time 0.0 and :148 sets each
    # trace tbin = start - t0 (t0 = min start over ALL records of the event), so the
    # OpHitFinder coverage rows and ophit peak_time live on a FRAME-RELATIVE axis, not
    # the absolute DTS timestamp.  Work in us here; ophit peak_time is WCT-internal ns.
    t0_tick = start.min()
    tb = (start - t0_tick) * TICK_US
    te = (start - t0_tick + ns) * TICK_US
    iv = {}
    for c, b, e in zip(ch, tb, te):
        iv.setdefault(int(c), []).append((b, e))
    for v in iv.values():
        v.sort()
    chmap = {int(c): int(o) for c, o in zip(ch, od)}
    isfull = {int(c): False for c in ch}
    for c, n in zip(ch, ns):
        if n > SNIPPET_NSAMP:
            isfull[int(c)] = True
    return iv, chmap, isfull


def covered_fraction(iv, ch, t_lo, t_hi):
    """OpFlashFinder.cxx:711-725 verbatim."""
    v = iv.get(ch)
    if not v:
        return 0.0
    if t_hi <= t_lo:
        for a, b in v:
            if a <= t_lo < b:
                return 1.0
        return 0.0
    live = 0.0
    for a, b in v:
        if a >= t_hi:
            break
        lo, hi = max(a, t_lo), min(b, t_hi)
        if hi > lo:
            live += hi - lo
    return min(1.0, live / (t_hi - t_lo))


def prev_gap(iv, ch, t_lo):
    """us from the end of the last snippet ending at/before t_lo; None if no prior snippet."""
    v = iv.get(ch)
    if not v:
        return None
    ends = [b for a, b in v if b <= t_lo]
    return (t_lo - max(ends)) if ends else None


def main(events):
    NCH = 40
    gaps_unc, gaps_cov, nosnip_unc, nosnip_cov = [], [], 0, 0
    gate_ok = gate_tot = 0
    for evt in events:
        try:
            T = load_tensors(evt)
        except Exception as e:
            print("  (skip %d: %s)" % (evt, e)); continue
        oh, fc = T["ophits"], T["flash_cov"]
        iv, chmap, isfull = snippets(evt)
        od_sub = {}
        for c, o in chmap.items():
            od_sub.setdefault(o, []).append(c)
        ncol = oh.shape[1]
        # ophit peak_time is WCT-internal (ns); snippet intervals above are us.
        fid, pt = oh[:, 7].astype(int), oh[:, 1] / 1000.0
        nflash = fc.shape[0] if fc.ndim == 2 else len(fc) // NCH
        fcm = fc.reshape(nflash, NCH)
        for f in range(nflash):
            sel = fid == f
            if not sel.any():
                continue
            t_lo, t_hi = pt[sel].min(), pt[sel].max()
            for od in range(NCH):
                subs = od_sub.get(od)
                if not subs:
                    continue
                frac = 1.0
                for c in subs:
                    frac = min(frac, covered_fraction(iv, c, t_lo, t_hi))
                # ---- validation gate against the chain's own tensor ----
                gate_tot += 1
                if abs(frac - fcm[f, od]) < 0.02:
                    gate_ok += 1
                # ---- the question: busy or armed? (self-trigger channels only)
                if all(isfull.get(c, False) for c in subs):
                    continue  # full-stream cathode: always covered, not at issue
                g = [prev_gap(iv, c, t_lo) for c in subs]
                g = [x for x in g if x is not None]
                tgt = gaps_unc if fcm[f, od] < 1.0 else gaps_cov
                if g:
                    tgt.append(min(g))
                else:
                    if fcm[f, od] < 1.0:
                        nosnip_unc += 1
                    else:
                        nosnip_cov += 1
    print("VALIDATION GATE: rebuilt coverage matches chain flash_cov on %d/%d (%.2f%%) "
          "(flash,opdet) cells" % (gate_ok, gate_tot, 100.0 * gate_ok / max(gate_tot, 1)))
    if gate_ok < 0.98 * gate_tot:
        print("*** gate FAILED -- tick base or ganging misapplied; stop. ***")
        return
    print()
    u, c = np.array(gaps_unc), np.array(gaps_cov)
    print("UNCOVERED (cov<1) self-trigger channels: %d with a prior snippet, "
          "%d with NO snippet anywhere in the event" % (len(u), nosnip_unc))
    print("COVERED   (cov=1) self-trigger channels: %d with a prior snippet, "
          "%d with none" % (len(c), nosnip_cov))
    print()
    print("Gap from END of previous snippet to the flash window start t_lo:")
    print("%-14s %10s %10s" % ("gap (us)", "uncovered", "covered"))
    bins = [(0, 5), (5, 16.4), (16.4, 50), (50, 200), (200, 1000), (1000, 1e9)]
    for lo, hi in bins:
        fu = ((u >= lo) & (u < hi)).mean() if len(u) else 0
        fcv = ((c >= lo) & (c < hi)).mean() if len(c) else 0
        print("%-14s %9.1f%% %9.1f%%" % ("%g-%g" % (lo, hi), 100 * fu, 100 * fcv))
    if len(u):
        print("\nuncovered: median gap %.1f us   frac within 16.4us (one snippet length) %.1f%%"
              % (np.median(u), 100 * (u < 16.4).mean()))
    if len(c):
        print("covered:   median gap %.1f us   frac within 16.4us %.1f%%"
              % (np.median(c), 100 * (c < 16.4).mean()))


if __name__ == "__main__":
    ev = [int(sys.argv[1])] if len(sys.argv) > 1 else [
        298567, 298581, 298595, 298609, 298623, 298637, 298651, 298665, 298679,
        298693, 298707, 298721, 298735, 298749, 298763, 298777, 298791, 298805]
    main(ev)
