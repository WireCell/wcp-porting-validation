#!/usr/bin/env python3
"""Doc 23 section 7d: the "two reconstructed flashes" at the matched time of
tracks B (evt 298609, raw 5399.99 + 5401.16) and C (evt 298651, raw 2800.64 +
2801.95) are ONE physical flash split by the finder.  This script prints the
evidence:

  1. family-resolved PE + hit-time spans of each pair member (cathode XA
     opdets 4-11 / membrane XA 0-3,12,13,18,19 / PMT the rest; raw DAPHNE
     channels 1xxx / 2xxx / 3xxx) and the PE left unassigned (flash_id -1)
     in the pair window;
  2. the largest hits of the second member (the cathode-XA slow-component
     tail, 1.6-1.7 us wide, 1.2-1.9 us after the fast peak);
  3. per-family summed RAW waveform peak times in the window: the cathode
     stream has a single fast pulse; the self-trigger families peak
     +0.7..0.9 us later (the doc 23 section 4 record lateness) -- run with
     --raw (needs uproot + the rawwf ROOT file, ~1 min).

Run from pdvd/docs/qlmatch:  python3 scripts/aca_flash_split.py [--raw]
"""
import io, os, sys, tarfile
import numpy as np

PDVD = os.path.join(os.path.dirname(__file__), "..", "..", "..")
RAW = os.path.join(PDVD, "input_data_light",
    "np02vd_raw_run039252_1176_df-s03-d3_dw_0_20250830T054542_rawwf.root")

CATH = set(range(4, 12))
MEMB = {0, 1, 2, 3, 12, 13, 18, 19}

# event -> (work tag, pair flash row indices, window us, track label)
PAIRS = {
    298609: ("039252_light298609_keep", (313, 314), (5398.0, 5406.0), "B"),
    298651: ("039252_light298651_keep", (150, 151), (2798.0, 2806.0), "C"),
    # track A for contrast: same topology, ONE flash (bin phase absorbed the
    # late self-trigger hits; its slow tail went unassigned instead)
}
A_FLASH = (298609, 215, "A")


def load(ev, tag):
    with tarfile.open(os.path.join(PDVD, "work", tag, "opflash_pdvd-wct.tar.gz")) as tf:
        op = np.load(io.BytesIO(tf.extractfile(f"opflash_tensor_{ev}_0_array.npy").read()))
        oh = np.load(io.BytesIO(tf.extractfile(f"opflash_tensor_{ev}_2_array.npy").read()))
    return op, oh  # ophit cols: ch, book_time_ns, width_ns, area, peak, pe, start_ns, flash_id, f2t


def fam_of_opdet(od):
    return "cathXA" if od in CATH else ("membXA" if od in MEMB else "PMT")


def fam_of_daphne(ch):
    return {1: "cathXA", 2: "membXA", 3: "PMT"}[ch // 1000]


def report(ev, tag, pair, win, trk):
    op, oh = load(ev, tag)
    t_fl = op[:, 0] / 1000.0
    print(f"== track {trk} (evt {ev}): flash pair "
          f"{t_fl[pair[0]]:.3f} / {t_fl[pair[1]]:.3f} us, dt = "
          f"{t_fl[pair[1]] - t_fl[pair[0]]:+.3f} us ==")
    for fid in pair:
        pe = op[fid, 1:41]
        byfam = {}
        for od in np.nonzero(pe > 0)[0]:
            byfam.setdefault(fam_of_opdet(od), [0.0, 0])
            byfam[fam_of_opdet(od)][0] += pe[od]
            byfam[fam_of_opdet(od)][1] += 1
        m = oh[:, 7] == fid
        th = oh[m, 1] / 1000.0
        parts = ", ".join(f"{f} {v[0]:.1f} PE/{v[1]} PD" for f, v in byfam.items())
        print(f"  flash row {fid}: t={t_fl[fid]:9.3f} tot={pe.sum():8.1f} PE"
              f"  [{parts}]  hits {th.min():.3f}-{th.max():.3f} us")
    # biggest hits of the late member = the cathode slow-component tail
    m2 = oh[:, 7] == pair[1]
    big = oh[m2][np.argsort(oh[m2, 5])[::-1][:4]]
    print("  largest hits of the late member (ch, t_us, width_us, PE):")
    for r in big:
        print(f"    ch{int(r[0])} ({fam_of_daphne(int(r[0]))}) t={r[1]/1000:.3f}"
              f" w={r[2]/1000:.2f} pe={r[5]:.1f}")
    # unassigned light in the window (fell between/after the accumulator bins)
    th = oh[:, 1] / 1000.0
    un = (th >= win[0]) & (th <= win[1]) & (oh[:, 7] < 0) & (oh[:, 5] > 0)
    pe_un = {f: 0.0 for f in ("cathXA", "membXA", "PMT")}
    for r in oh[un]:
        pe_un[fam_of_daphne(int(r[0]))] += r[5]
    print(f"  unassigned positive-PE hits in window: n={un.sum()}"
          f" pe={oh[un, 5].sum():.1f}  ({', '.join(f'{k} {v:.1f}' for k, v in pe_un.items())})")
    print()


def report_A():
    ev, fid, trk = A_FLASH
    op, oh = load(ev, PAIRS[ev][0])
    m = oh[:, 7] == fid
    th = oh[m, 1] / 1000.0
    fam = np.array([fam_of_daphne(int(c)) for c in oh[m, 0]])
    print(f"== track {trk} contrast: single flash row {fid} t={op[fid,0]/1000:.3f} "
          f"tot={op[fid,1:41].sum():.1f} PE ==")
    for f in ("cathXA", "membXA", "PMT"):
        mm = fam == f
        if mm.any():
            print(f"  {f}: n={mm.sum()} pe={oh[m][mm,5].sum():9.1f}"
                  f"  t {th[mm].min():.3f}-{th[mm].max():.3f} us")
    late = (oh[:, 1] / 1000.0 > th.max()) & (oh[:, 1] / 1000.0 < th.max() + 3.0) \
        & (oh[:, 7] < 0) & (oh[:, 5] > 0)
    print(f"  slow tail after the flash: {late.sum()} unassigned hits, "
          f"{oh[late,5].sum():.1f} PE (dropped, not a second flash)\n")


def raw_peaks():
    import uproot
    d = uproot.open(RAW)["rawdump/raw_waveform"].arrays(
        ["event", "opchannel", "nsamp", "timestamp", "adc"], library="np")
    TICK = 16.0
    for ev, (tag, pair, win, trk) in PAIRS.items():
        sel = np.nonzero(d["event"] == ev)[0]
        starts = {i: int(round(d["timestamp"][i] * 62.5))
                  - (64 if d["nsamp"][i] <= 1024 else 0) for i in sel}
        t0 = min(starts.values())
        print(f"== track {trk} (evt {ev}) raw per-family peaks in {win} ==")
        for famcode, name in [(1, "cathXA(stream)"), (2, "membXA(snip)"), (3, "PMT(snip)")]:
            grid = {}
            for i in [j for j in sel if d["opchannel"][j] // 1000 == famcode]:
                a = np.asarray(d["adc"][i], dtype=float)
                a -= np.median(a[:200])
                s0 = starts[i] - t0
                t_us = (np.arange(a.size) + s0) * TICK / 1000.0
                m = (t_us >= win[0]) & (t_us <= win[1])
                for tv, sv in zip(np.round(t_us[m] * 62.5).astype(int), a[m]):
                    grid[tv] = grid.get(tv, 0.0) + sv
            if not grid:
                continue
            ks = sorted(grid)
            tv = np.array(ks) / 62.5
            sv = np.array([grid[k] for k in ks])
            pk = [(tv[j], sv[j]) for j in range(1, len(sv) - 1)
                  if sv[j] > sv[j - 1] and sv[j] >= sv[j + 1] and sv[j] > 0.10 * sv.max()]
            merged = []
            for t_, s_ in pk:
                if merged and t_ - merged[-1][0] < 0.15:
                    if s_ > merged[-1][1]:
                        merged[-1] = (t_, s_)
                else:
                    merged.append((t_, s_))
            print(f"  {name}: peaks>10%:",
                  ", ".join(f"{t_:.3f}({s_:.0f})" for t_, s_ in merged[:6]))
        print()


if __name__ == "__main__":
    for ev, (tag, pair, win, trk) in PAIRS.items():
        report(ev, tag, pair, win, trk)
    report_A()
    if "--raw" in sys.argv:
        raw_peaks()
