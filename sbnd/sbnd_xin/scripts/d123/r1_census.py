#!/usr/bin/env python3
"""doc sbnd_xin/123 round 1 -- the close-flash census: what SBND's reco1 SimpleFlashAlgo merged,
vetoed or dropped, measured on the hit flashes.

Input: a flash dump written by wct-reco1-dump.jsonnet with flash_source=hits reco1_reference=true
(hits_arm.sh --ref): per group or per entry, opflash_apa<N>.tar.gz (ours: opflash [nflash,313],
flash_summary, ophits with col 7 = our flash id) and reco1flash_apa<N>.tar.gz (SBND's OpFlash).
Both archives may hold many tensor sets (one per event); members are keyed by event id.

Per event and TPC, every reco1 flash R (time t_R) takes the nearest hit flash within +-match_us as
its MATCH (SBND's time is ours + a few ns of light travel).  A reco1 flash with no match within
match_us but one at -74.06 +- 0.5 us is BUGGED (sbndcode's failed-X time shift).  Every hit flash
that is nobody's match is then classed by where it sits relative to the reco1 flashes:
    absorbed   inside a reco1 flash's integration window (t_R, t_R + 8 us]: light SBND folded into R
    vetoed     inside (t_R - 8 us, t_R) of a reco1 flash: an EARLIER pulse SBND's veto threw away
    dropped    outside every window: light SBND put in no flash at all
(a hit flash inside two windows takes the nearer reco1 flash).  The census also reports, for the
beam-window reco1 flashes (corrected time in [0.3, 1.9] us), how many carry an absorbed or vetoed
partner, and the reco1-vs-ours count of beam-window flashes per TPC.

usage: r1_census.py <dump_root> [--glob 'g*'] [--match-us 0.5] [--min-pe 20] [--mc]
                    [--tsv flashes.tsv] [--summary summary.json]
"""
import argparse, glob, io, json, os, re, sys, tarfile
import numpy as np

RE_SET = re.compile(r"^opflash_tensorset_(\d+)_metadata\.json$")
RE_ARR = re.compile(r"^opflash_tensor_(\d+)_(\d+)_array\.npy$")
VETO_NS, INT_NS = 8000.0, 8000.0
BUG_NS = (999999.0 - 201.0) / 13.5 * 1000.0  # 74 059 ns, sbndcode failed-X shift


def read_sets(path):
    """{event: (metadata, {tensor index: array})}"""
    out = {}
    with tarfile.open(path) as t:
        for m in t.getmembers():
            a = RE_SET.match(m.name)
            if a:
                out.setdefault(int(a.group(1)), [None, {}])[0] = json.load(t.extractfile(m))
                continue
            b = RE_ARR.match(m.name)
            if b:
                out.setdefault(int(b.group(1)), [None, {}])[1][int(b.group(2))] = \
                    np.load(io.BytesIO(t.extractfile(m).read()))
    return out


def flashes(arrs):
    x = arrs.get(0)
    if x is None or x.ndim != 2 or x.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    return x[:, 0].astype(float), x[:, 1:].sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--glob", default="g*")
    ap.add_argument("--match-us", type=float, default=0.5)
    ap.add_argument("--min-pe", type=float, default=20.0, help="hit flashes below this PE are not classed")
    ap.add_argument("--min-dt-us", type=float, default=0.3,
                    help="an unmatched hit flash closer than this to a reco1 flash is a 'piece' of it (the "
                         "finder cut the same pulse), not a separate absorbed/vetoed pulse")
    ap.add_argument("--mc", action="store_true", help="no frame_apply_at_caf expected")
    ap.add_argument("--prepulse-us", type=float, default=4.0)
    ap.add_argument("--prepulse-frac", type=float, default=0.01)
    ap.add_argument("--tsv")
    ap.add_argument("--summary")
    a = ap.parse_args()
    match_ns = a.match_us * 1000.0

    rows = []      # one per hit flash (matched or not) and per bugged/unfound reco1 flash
    S = dict(events=0, tpcsets=0, n_reco1=0, n_ours=0, matched=0, bugged=0, unfound=0,
             absorbed=0, vetoed=0, dropped=0, piece=0, prepulse=0, ours_small=0,
             beam_reco1=0, beam_reco1_with_absorbed=0, beam_reco1_with_vetoed=0,
             beam_ours=0, beam_ours_unmatched=0, no_offset=0)
    dt_abs, pe_abs, dt_vet, pe_vet, pe_drop, dt_match = [], [], [], [], [], []
    evts = set()
    for tpc in (0, 1):
        for opath in sorted(glob.glob(os.path.join(a.root, a.glob, "opflash_apa%d.tar.gz" % tpc))):
            rpath = os.path.join(os.path.dirname(opath), "reco1flash_apa%d.tar.gz" % tpc)
            if not os.path.exists(rpath):
                sys.exit("no reference %s (dump with reco1_reference=true)" % rpath)
            ours, ref = read_sets(opath), read_sets(rpath)
            for ev in sorted(ours):
                if ev not in ref:
                    continue
                evts.add(ev)
                S["tpcsets"] += 1
                md = ours[ev][0] or {}
                off = md.get("frame_apply_at_caf")
                if off is None:
                    if not a.mc:
                        S["no_offset"] += 1
                    off = 0.0
                to, po = flashes(ours[ev][1])
                tr, pr = flashes(ref[ev][1])
                S["n_reco1"] += len(tr); S["n_ours"] += len(to)
                match_of_r = np.full(len(tr), -1)      # reco1 -> ours index
                matched_o = np.zeros(len(to), bool)
                # reco1 flashes brightest first take the nearest free hit flash within match_ns
                for i in np.argsort(-pr):
                    if len(to) == 0:
                        break
                    d = np.abs(to - tr[i]); d[matched_o] = np.inf
                    j = int(np.argmin(d))
                    if d[j] <= match_ns:
                        match_of_r[i] = j; matched_o[j] = True
                        S["matched"] += 1; dt_match.append(tr[i] - to[j])
                        rows.append((ev, tpc, "match", i, (tr[i] + off) / 1e3, pr[i], j, (to[j] + off) / 1e3, po[j], (to[j] - tr[i]) / 1e3))
                    else:
                        # the 74 us bug: SBND's time is ours + 74.06 us
                        d2 = np.abs((to + BUG_NS) - tr[i]); d2[matched_o] = np.inf
                        k = int(np.argmin(d2)) if len(d2) else -1
                        if k >= 0 and d2[k] <= match_ns:
                            match_of_r[i] = k; matched_o[k] = True; S["bugged"] += 1
                            rows.append((ev, tpc, "bugged", i, (tr[i] + off) / 1e3, pr[i], k, (to[k] + off) / 1e3, po[k], (to[k] - tr[i]) / 1e3))
                        else:
                            S["unfound"] += 1
                            rows.append((ev, tpc, "unfound", i, (tr[i] + off) / 1e3, pr[i], -1, float("nan"), 0.0, float("nan")))
                # the reco1 time the windows are anchored on: the corrected (un-bugged) one
                tr_win = tr.copy()
                for i in range(len(tr)):
                    if match_of_r[i] >= 0 and rows and rows[-1][2] == "bugged" and rows[-1][3] == i:
                        tr_win[i] = to[match_of_r[i]]
                for i in range(len(tr)):
                    if match_of_r[i] >= 0 and abs(to[match_of_r[i]] + BUG_NS - tr[i]) <= match_ns:
                        tr_win[i] = to[match_of_r[i]]
                has_abs = np.zeros(len(tr), bool); has_vet = np.zeros(len(tr), bool)
                for j in range(len(to)):
                    if matched_o[j]:
                        continue
                    if po[j] < a.min_pe:
                        S["ours_small"] += 1
                        continue
                    cls, i_near, dt = "dropped", -1, float("nan")
                    if len(tr_win):
                        d = to[j] - tr_win           # >0: after the reco1 flash
                        inside_abs = (d > 0) & (d <= INT_NS)
                        inside_vet = (d < 0) & (d > -VETO_NS)
                        cand = np.where(inside_abs | inside_vet)[0]
                        if len(cand):
                            i_near = int(cand[np.argmin(np.abs(d[cand]))])
                            dt = d[i_near]
                            cls = "absorbed" if dt > 0 else "vetoed"
                            if abs(dt) < a.min_dt_us * 1000.0:
                                cls = "piece"
                            elif cls == "vetoed" and dt > -a.prepulse_us * 1000.0 and \
                                    po[j] < a.prepulse_frac * pr[i_near]:
                                # a faint flash shortly BEFORE a very bright one: seen on 40 of the
                                # 87 beam flashes of nueCC48 at 0.03-0.3 % of its PE, 1-3 us early --
                                # the deconvolution's pre-ringing of the big pulse is the likely
                                # source (xning's open item on fake hits), not a real earlier pulse
                                cls = "prepulse"
                    S[cls] += 1
                    if cls == "absorbed":
                        dt_abs.append(dt / 1e3); pe_abs.append(po[j]); has_abs[i_near] = True
                    elif cls == "vetoed":
                        dt_vet.append(dt / 1e3); pe_vet.append(po[j]); has_vet[i_near] = True
                    elif cls == "dropped":
                        pe_drop.append(po[j])
                    rows.append((ev, tpc, cls, i_near, (tr_win[i_near] + off) / 1e3 if i_near >= 0 else float("nan"),
                                 pr[i_near] if i_near >= 0 else 0.0, j, (to[j] + off) / 1e3, po[j], dt / 1e3))
                # beam-window bookkeeping (corrected times)
                tb = (tr_win + off) / 1e3
                beam = (tb >= 0.3) & (tb <= 1.9)
                S["beam_reco1"] += int(beam.sum())
                S["beam_reco1_with_absorbed"] += int((beam & has_abs).sum())
                S["beam_reco1_with_vetoed"] += int((beam & has_vet).sum())
                ob = (to + off) / 1e3
                beam_o = (ob >= 0.3) & (ob <= 1.9) & (po >= a.min_pe)
                S["beam_ours"] += int(beam_o.sum())
                S["beam_ours_unmatched"] += int((beam_o & ~matched_o).sum())
    S["events"] = len(evts)

    def q(x):
        x = np.asarray(x, float)
        return [round(float(v), 3) for v in np.percentile(x, [0, 5, 50, 95, 100])] if len(x) else []
    S["dt_match_ns_q"] = q(dt_match)
    S["absorbed_dt_us_q"] = q(dt_abs); S["absorbed_pe_q"] = q(pe_abs)
    S["vetoed_dt_us_q"] = q(dt_vet); S["vetoed_pe_q"] = q(pe_vet)
    S["dropped_pe_q"] = q(pe_drop)
    S["absorbed_dt_us_hist_1us"] = np.histogram(dt_abs, bins=np.arange(0, 9, 1.0))[0].tolist() if dt_abs else []
    S["vetoed_dt_us_hist_1us"] = np.histogram(dt_vet, bins=np.arange(-8, 1, 1.0))[0].tolist() if dt_vet else []
    for k in ("absorbed", "vetoed", "dropped"):
        S[k + "_per_event"] = round(S[k] / max(1, S["events"]), 3)
    print(json.dumps(S, indent=1))
    if a.summary:
        json.dump(S, open(a.summary, "w"), indent=1)
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("event\ttpc\tclass\treco1_idx\treco1_t_us\treco1_pe\tours_idx\tours_t_us\tours_pe\tdt_us\n")
            for r in rows:
                f.write("%d\t%d\t%s\t%d\t%.4f\t%.1f\t%d\t%.4f\t%.1f\t%.4f\n" % r)


if __name__ == "__main__":
    main()
