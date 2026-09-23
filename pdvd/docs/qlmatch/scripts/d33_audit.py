#!/usr/bin/env python3
"""doc qlmatch/33 -- blindness audit (fork of d32_audit.py for the d33 recorder) of one scanner agent's transcript (JSONL).  Lists every tool call the agent made
and flags any Read/Glob/Grep path or Bash command that touches something other than its own wave directory, the
RUBRIC, or the recorder script.  Prints only the flagged calls and a summary (never the transcript itself).

    python3 d33_audit.py <transcript.jsonl> <wave> [--scan-root /home/xqian/tmp/p33/scan/r2d]
"""
import argparse
import json
import re

REC = "docs/qlmatch/scripts/d33_record.py"


def calls(path):
    with open(path) as fh:
        for ln in fh:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            msg = d.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for c in content:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    yield c.get("name"), c.get("input") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("wave", type=int)
    ap.add_argument("--scan-root", default="/home/xqian/tmp/p33/scan/r2d")
    a = ap.parse_args()
    wave_dir = f"{a.scan_root}/wave{a.wave}"
    allowed = (wave_dir + "/", f"{a.scan_root}/RUBRIC.md")
    n, nrec, flagged, npng = 0, 0, [], 0
    for name, inp in calls(a.transcript):
        n += 1
        if name == "Read":
            p = inp.get("file_path", "")
            if p.endswith(".png"):
                npng += 1
            if not (p.startswith(allowed[0]) or p == allowed[1] or p.endswith(REC)):
                flagged.append((name, p))
        elif name in ("Glob", "Grep", "LS"):
            flagged.append((name, json.dumps(inp)[:200]))
        elif name == "Bash":
            cmd = inp.get("command", "")
            if REC in cmd and "--wave" in cmd:
                nrec += 1
                m = re.search(r"--wave\s+(\d+)", cmd)
                if m and int(m.group(1)) != a.wave:
                    flagged.append(("Bash(wrong wave)", cmd[:200]))
                continue
            # anything else must stay inside the wave dir (e.g. ls of it) or read the rubric/recorder
            paths = re.findall(r"(/[\w./-]+)", cmd)
            ok = all(p.startswith(wave_dir) or p.startswith(allowed[1]) or p.endswith(REC)
                     or p in ("/dev/null",) for p in paths)
            if not ok:
                flagged.append((name, cmd[:200]))
        elif name not in ("TodoWrite", "TaskCreate", "TaskUpdate"):
            flagged.append((name, json.dumps(inp)[:200]))
    print(f"wave {a.wave}: {n} tool calls, {npng} sheet reads, {nrec} recorder calls, {len(flagged)} flagged")
    for f in flagged:
        print("  FLAG", f[0], f[1])


if __name__ == "__main__":
    main()
