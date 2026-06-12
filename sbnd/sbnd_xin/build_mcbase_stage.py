#!/usr/bin/env python3
"""Stage the input-10files-mc files with globally-unique event IDs.

The 10 MC files share the event-id space 1..50, so the standard work/evt<ID>
scheme would clobber them.  Remap file N event E -> UID = 700000 + N*1000 + E
and repackage frames + opflash under canonical names (frames-dnn.tar.bz2,
opflash_apa{0,1}.tar.gz) so the run scripts (via SBND_INPUT_DIR) can process
each file independently.  Output dir default: $MCBASE_STAGE or /home/xqian/tmp/mcbase_in.

Member naming (verified): frames <prefix>_<E>.npy with prefix in {frame_dnnsp,
channels_dnnsp,tickinfo_dnnsp,summary_dnnsp,chanmask_bad}; opflash
opflash_tensorset_<E>_*, opflash_tensor_<E>_<k>_*.  Metadata json contents are
null/{} (no internal id refs) -> rename files only.
"""
import io
import os
import re
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "input_files", "input-10files-mc")
OUT = os.environ.get("MCBASE_STAGE", "/home/xqian/tmp/mcbase_in")
NFILES = 10


def uid(n, e):
    return 700000 + n * 1000 + e


def remap_tar(src_path, dst_path, rename, mode_w):
    with tarfile.open(src_path, "r:*") as ti, tarfile.open(dst_path, mode_w) as to:
        for m in ti.getmembers():
            if not m.isfile():
                continue
            new = rename(m.name)
            if new is None:
                continue
            data = ti.extractfile(m).read()
            ni = tarfile.TarInfo(new)
            ni.size = len(data); ni.mtime = m.mtime; ni.mode = m.mode
            to.addfile(ni, io.BytesIO(data))


FRAME_RE = re.compile(r"^(frame_dnnsp|channels_dnnsp|tickinfo_dnnsp|summary_dnnsp|chanmask_bad)_(\d+)\.npy$")
OPSET_RE = re.compile(r"^opflash_tensorset_(\d+)_(.*)$")
OPTEN_RE = re.compile(r"^opflash_tensor_(\d+)_(.*)$")


def main():
    for n in range(1, NFILES + 1):
        d = os.path.join(OUT, str(n))
        os.makedirs(d, exist_ok=True)

        def frame_rename(name, n=n):
            m = FRAME_RE.match(name)
            return f"{m.group(1)}_{uid(n, int(m.group(2)))}.npy" if m else None

        remap_tar(os.path.join(SRC, f"frames-dnn-{n}.tar.bz2"),
                  os.path.join(d, "frames-dnn.tar.bz2"), frame_rename, "w:bz2")

        for apa in (0, 1):
            def op_rename(name, n=n):
                m = OPSET_RE.match(name)
                if m:
                    return f"opflash_tensorset_{uid(n, int(m.group(1)))}_{m.group(2)}"
                m = OPTEN_RE.match(name)
                if m:
                    return f"opflash_tensor_{uid(n, int(m.group(1)))}_{m.group(2)}"
                return None
            remap_tar(os.path.join(SRC, f"opflash_apa{apa}-{n}.tar.gz"),
                      os.path.join(d, f"opflash_apa{apa}.tar.gz"), op_rename, "w:gz")

        with tarfile.open(os.path.join(d, "frames-dnn.tar.bz2")) as t:
            ids = sorted({int(re.match(r"frame_dnnsp_(\d+)\.npy", x).group(1))
                          for x in t.getnames() if x.startswith("frame_dnnsp_")})
        print(f"file {n}: {len(ids)} events  uid {ids[0]}..{ids[-1]}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
