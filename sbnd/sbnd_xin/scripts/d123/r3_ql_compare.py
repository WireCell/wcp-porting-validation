#!/usr/bin/env python3
"""doc sbnd_xin/123 round 3 -- compare the Q/L matches of two stage-A arms that share their imaging.

Both arms are run_chain_group.sh --layout perevt roots; the Q/L job's log g*/wct_ql.log carries,
per event, one 'flash_bundles_map: flash id I time T ns, cluster gidx G ...' line per matched
cluster, followed by its 'bundle_flags: flash id I cluster ident N gidx G | ...' line (the same
lines tools/ql_ab_compare.py and xning's ql_compare_ours.py parse), and the cathode-rescue
'rescue round' / 'unmatched rescue round' lines.  A group log holds all of its events; lines are
assigned to events by TIMESTAMP against the 'loading tensor set ident=' lines, as
scripts/multi/slice_group_log.py does (file order is not time order across spdlog sinks).

Clusters are keyed by (event, anode group, cluster ident): the imaging is shared, so an ident names the same
object in both arms unless clustering itself changed (the rescue merges; see the 'ident' counts).
A cluster's flash is identified by its corrected time; two matches are the SAME flash when the
times agree within --same-us (ours and SBND's times of one flash agree to ~10 ns).  Flash PE comes
from the arm's own opflash_apa*.tar.gz (tensor 0 column 0 + frame_apply_at_caf = the logged time).

Per event: matched clusters A / B, same / moved / lost / gained, beam-window (0.3..1.9 us) matches,
rescue moves A / B.  --tsv writes one row per (event, cluster); --events writes one row per event.

usage: r3_ql_compare.py <arm_A> <arm_B> [--same-us 0.2] [--tsv out.tsv] [--events out_events.tsv]
"""
import argparse, glob, io, json, os, re, sys, tarfile
import numpy as np

RE_TS = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\.(\d\d\d)\]")
RE_LOAD = re.compile(r"loading tensor set ident=(\d+) ")
RE_MAP = re.compile(r"flash_bundles_map: flash id (\d+) time (-?[\d.e+]+) ns, cluster gidx (\d+) total_pred_light (-?[\d.e+]+)")
RE_ANODE = re.compile(r"matching_joint> anode (\d+) group-bbox")
RE_FLAGS = re.compile(r"bundle_flags: flash id (\d+) cluster ident (\d+) gidx (\d+)")
RE_RESCUE = re.compile(r"(unmatched rescue round|rescue round|fragment adopt round) (\d+): c(\d+) \(gid (\d+), t0 (-?[\d.]+) us")
RE_SET = re.compile(r"^opflash_tensorset_(\d+)_metadata\.json$")
RE_T0 = re.compile(r"^opflash_tensor_(\d+)_0_array\.npy$")


def ts(line):
    m = RE_TS.match(line)
    if not m:
        return None
    h, mi, s, ms = (int(x) for x in m.groups())
    return ((h * 60 + mi) * 60 + s) * 1000 + ms


def parse_log(path):
    """{event: {'matches': {(anode, ident): (t_ns, pe_pred, flash_id, gidx)}, 'rescue': [lines]}}

    Event assignment: the joint matcher runs per anode group AFTER both per-APA clusterings of an
    event and BEFORE the all-APA MABC step loads that event ('<...clus_all_apa> loading tensor set
    ident=E'), while the NEXT event's per-APA clustering may already have started (the graph is a
    pipeline).  So a matcher line belongs to the event of the first clus_all_apa load line at or
    after its timestamp.  Cluster idents restart per anode group; the 'anode N group-bbox' line
    that opens each group's block gives the anode.
    """
    lines = open(path, errors="replace").read().splitlines()
    ends = []   # (ts, ident) of the clus_all_apa load lines
    for ln in lines:
        if "clus_all_apa> loading tensor set ident=" in ln:
            m = RE_LOAD.search(ln); t = ts(ln)
            if m and t is not None:
                ends.append((t, int(m.group(1))))
    ends.sort()
    if not ends:
        return {}
    end_ts = np.array([t for t, _ in ends])

    def event_of(t):
        i = int(np.searchsorted(end_ts, t, side="left"))
        return ends[min(i, len(ends) - 1)][1]

    out = {ev: {"matches": {}, "rescue": [], "blocks": 0} for _, ev in ends}
    pending = {}   # (event, anode, flash id, gidx) -> (t, pred)  awaiting its bundle_flags line
    last_t = ends[0][0]; anode = -1
    for ln in lines:
        t = ts(ln)
        if t is not None:
            last_t = t
        if "matching_joint>" not in ln and "rescue round" not in ln:
            continue
        m = RE_ANODE.search(ln)
        if m:
            anode = int(m.group(1)); out[event_of(last_t)]["blocks"] += 1
            continue
        m = RE_MAP.search(ln)
        if m:
            ev = event_of(last_t)
            pending[(ev, anode, int(m.group(1)), int(m.group(3)))] = (float(m.group(2)), float(m.group(4)))
            continue
        m = RE_FLAGS.search(ln)
        if m:
            ev = event_of(last_t)
            key = (ev, anode, int(m.group(1)), int(m.group(3)))
            if key in pending:
                tt, pred = pending[key]
                out[ev]["matches"][(anode, int(m.group(2)))] = (tt, pred, int(m.group(1)), int(m.group(3)))
            continue
        m = RE_RESCUE.search(ln)
        if m:
            out[event_of(last_t)]["rescue"].append(m.group(0))
    bad = [ev for ev, d in out.items() if d["blocks"] != 2]
    if bad:
        sys.stderr.write("%s: %d events without exactly 2 matcher blocks: %s\n" % (path, len(bad), bad[:5]))
    return out


def flash_pe(root):
    """{event: (corrected times ns, pe)} over both TPCs"""
    out = {}
    for tpc in (0, 1):
        for path in sorted(glob.glob(os.path.join(root, "g*", "opflash_apa%d.tar.gz" % tpc))):
            md, arr = {}, {}
            with tarfile.open(path) as t:
                for m in t.getmembers():
                    a = RE_SET.match(m.name)
                    if a:
                        md[int(a.group(1))] = json.load(t.extractfile(m)); continue
                    b = RE_T0.match(m.name)
                    if b:
                        arr[int(b.group(1))] = np.load(io.BytesIO(t.extractfile(m).read()))
            for ev, d in md.items():
                x = arr.get(ev)
                if x is None or x.ndim != 2 or x.shape[0] == 0:
                    continue
                off = d.get("frame_apply_at_caf", 0.0) or 0.0
                tt, pe = x[:, 0] + off, x[:, 1:].sum(axis=1)
                if ev in out:
                    out[ev] = (np.concatenate([out[ev][0], tt]), np.concatenate([out[ev][1], pe]))
                else:
                    out[ev] = (tt, pe)
    return out


def pe_at(fp, ev, t_ns):
    if ev not in fp or len(fp[ev][0]) == 0:
        return float("nan")
    tt, pe = fp[ev]
    i = int(np.argmin(np.abs(tt - t_ns)))
    return float(pe[i]) if abs(tt[i] - t_ns) < 100.0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a"); ap.add_argument("arm_b")
    ap.add_argument("--same-us", type=float, default=0.2)
    ap.add_argument("--tsv"); ap.add_argument("--events")
    a = ap.parse_args()
    A, B = {}, {}
    for root, dst in ((a.arm_a, A), (a.arm_b, B)):
        for lg in sorted(glob.glob(os.path.join(root, "g*", "wct_ql.log"))):
            dst.update(parse_log(lg))
    fpA, fpB = flash_pe(a.arm_a), flash_pe(a.arm_b)
    evts = sorted(set(A) | set(B))
    same_ns = a.same_us * 1000.0
    tot = dict(events=len(evts), matched_A=0, matched_B=0, same=0, moved=0, lost=0, gained=0,
               beam_A=0, beam_B=0, beam_same=0, beam_moved=0, beam_lost=0, beam_gained=0,
               rescue_A=0, rescue_B=0, events_changed=0, moved_to_small=0, gained_small=0)
    rows, erows = [], []
    for ev in evts:
        ma = A.get(ev, {}).get("matches", {}); mb = B.get(ev, {}).get("matches", {})
        ra = A.get(ev, {}).get("rescue", []); rb = B.get(ev, {}).get("rescue", [])
        tot["matched_A"] += len(ma); tot["matched_B"] += len(mb)
        tot["rescue_A"] += len(ra); tot["rescue_B"] += len(rb)
        e = dict(same=0, moved=0, lost=0, gained=0)
        for key in sorted(set(ma) | set(mb)):
            anode, ident = key
            xa, xb = ma.get(key), mb.get(key)
            beam_a = xa is not None and 300.0 <= xa[0] <= 1900.0
            beam_b = xb is not None and 300.0 <= xb[0] <= 1900.0
            tot["beam_A"] += beam_a; tot["beam_B"] += beam_b
            pa = pe_at(fpA, ev, xa[0]) if xa else float("nan")
            pb = pe_at(fpB, ev, xb[0]) if xb else float("nan")
            if xa and xb:
                cls = "same" if abs(xa[0] - xb[0]) <= same_ns else "moved"
            elif xa:
                cls = "lost"
            else:
                cls = "gained"
            e[cls] += 1; tot[cls] += 1
            if beam_a or beam_b:
                tot["beam_" + cls] += 1
            if cls == "moved" and pb < 100.0:
                tot["moved_to_small"] += 1
            if cls == "gained" and pb < 100.0:
                tot["gained_small"] += 1
            rows.append((ev, anode, ident, cls,
                         xa[0] / 1e3 if xa else float("nan"), pa, xa[1] if xa else float("nan"),
                         xb[0] / 1e3 if xb else float("nan"), pb, xb[1] if xb else float("nan"),
                         int(beam_a), int(beam_b)))
        changed = e["moved"] + e["lost"] + e["gained"] > 0 or len(ra) != len(rb)
        tot["events_changed"] += changed
        erows.append((ev, len(ma), len(mb), e["same"], e["moved"], e["lost"], e["gained"], len(ra), len(rb)))
    print(json.dumps(tot, indent=1))
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("event\tanode\tident\tclass\tA_t_us\tA_pe\tA_pred\tB_t_us\tB_pe\tB_pred\tA_beam\tB_beam\n")
            for r in rows:
                f.write("%d\t%d\t%d\t%s\t%.4f\t%.1f\t%.1f\t%.4f\t%.1f\t%.1f\t%d\t%d\n" % r)
    if a.events:
        with open(a.events, "w") as f:
            f.write("event\tmatched_A\tmatched_B\tsame\tmoved\tlost\tgained\trescue_A\trescue_B\n")
            for r in erows:
                f.write("\t".join(str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
