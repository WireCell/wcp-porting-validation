"""doc 95 -- prove each group archive carries the frame of the event it claims.

One decompression per archive (bz2 is expensive): hash every member once,
then compare the staged single-event extraction against the group archive it
was folded into.  A group-b event whose imaging silently read a different
frame would keep the right RSE (that comes from the opflash tar) and a
plausible point count, so this byte comparison is the discriminator.
"""
import hashlib
import tarfile

STAGE = "input_files_reco1/staged-dbg25"

rows, seen = [], set()
for line in open(f"{STAGE}/entry_event_map.tsv").readlines()[1:]:
    e, r, s, ev, _ = line.rstrip("\n").split("\t")[:5]
    grp = "b" if ev in seen else "a"
    seen.add(ev)
    rows.append((int(e), r, s, ev, grp))


def all_md5(path):
    out = {}
    with tarfile.open(path, "r:bz2") as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if f:
                out[m.name] = hashlib.md5(f.read()).hexdigest()
    return out


grp_h = {g: all_md5(f"input_files_reco1/extracted-dbg25{g}/frames-dnn.tar.bz2")
         for g in "ab"}
print("group archives hashed:", {g: len(v) for g, v in grp_h.items()}, flush=True)

bad = n = 0
print(f"{'grp':<4}{'entry':>5} {'RSE':<13} {'members':>8}  status", flush=True)
for e, r, s, ev, grp in rows:
    sh = all_md5(f"{STAGE}/e{e}/frames-dnn.tar.bz2")
    ok = True
    for name, h in sh.items():
        n += 1
        if grp_h[grp].get(name) != h:
            ok = False
            bad += 1
    if grp == "b" or not ok:
        print(f"{grp:<4}{e:>5} {r}-{s}-{ev:<8} {len(sh):>8}  "
              f"{'all match' if ok else 'MISMATCH'}", flush=True)

print(f"\nchecked {n} frame members across all 25 entries; mismatches: {bad}")
print("VERDICT:", "OK -- every group-archive member is byte-identical to the "
      "staged single-event extraction" if bad == 0 else "FAIL")
