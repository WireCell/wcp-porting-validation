#!/usr/bin/env python3
"""doc pdhd/25 sec 6 -- audit one scanner's transcript for any read that would break the blind.

    python3 d25_audit.py TRANSCRIPT.jsonl OWN_TAG          (OWN_TAG e.g. rv5_a3)
    python3 d25_audit.py --selftest                          (negative control: must flag every bad line)

Fork by duplication of the doc pdhd/19 round's /home/xqian/tmp/h20/audit.py (never committed); the h20
script stays as that round's record.  Changed: the earlier-round dirs are h18..h24 plus h25/ (the arm runs,
NOT h25r); the doc-25 files that name the chain's answer or the tranche's groups (controls, preregistered,
movers, census, twin, misses, the prep dirs) are forbidden; another scanner's item list is forbidden too.
Prints the number of tool calls and every flagged input; exit 1 if anything is flagged.
"""
import json, re, sys

def bad_patterns(own):
    B = [r'/home/xqian/tmp/h(1[89]|2[0-4])\b', r'/home/xqian/tmp/h25(/|"|$)', r'pdhd/docs', r'pdhd/work',
         r'pdvd/docs', r'stm_michel_scan/prep-', r'stm_michel_labels', r'_key|key_|\.tsv', r'labels',
         r'verdicts', r'queue', r'rulings', r'/adj', r'smprep-', r'controls', r'preregistered', r'movers',
         r'census', r'twin', r'misses', r'v_parts/?"', r'v_parts/?$']
    n = int(own.rsplit("a", 1)[1])
    B += [r'rv5_a%d\b' % i for i in range(1, 10) if i != n]
    B += [r'scratch_a%d\b' % i for i in range(1, 10) if i != n]
    B += [r'items_a%d\b' % i for i in range(1, 10) if i != n]
    return B

def audit(lines, own):
    B = bad_patterns(own)
    n, hits = 0, []
    for line in lines:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        msg = ev.get('message') or {}
        content = msg.get('content') if isinstance(msg.get('content'), list) else []
        for c in content:
            if not isinstance(c, dict) or c.get('type') != 'tool_use':
                continue
            n += 1
            inp = json.dumps(c.get('input', {}))
            for b in B:
                if re.search(b, inp):
                    hits.append((c.get('name'), b, inp[:220]))
                    break
    return n, hits

def tu(inp):
    return json.dumps({"message": {"content": [{"type": "tool_use", "name": "Bash", "input": inp}]}})

if sys.argv[1:] == ["--selftest"]:
    good = [tu({"command": "cat /home/xqian/tmp/h25r/shots/029107_1_85/context.json"}),
            tu({"file_path": "/home/xqian/tmp/h25r/RUBRIC.md"}),
            tu({"file_path": "/home/xqian/tmp/h25r/items_a3.txt"}),
            tu({"command": "python3 /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/campaign/mkv.py "
                           "029107_1/85 THRU none high --shots-dir /home/xqian/tmp/h25r/shots "
                           "--out-dir /home/xqian/tmp/h25r/v_parts/rv5_a3"}),
            tu({"file_path": "/home/xqian/tmp/h25r/scratch_a3/x.png"})]
    bad = [tu({"command": "ls /home/xqian/tmp/h25/arm_h25k"}),
           tu({"file_path": "/home/xqian/tmp/h21/shots/x/context.json"}),
           tu({"command": "cat /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/docs/scan/h25/controls_frozen.txt"}),
           tu({"command": "ls /home/xqian/tmp/h25r/v_parts/rv5_a1"}),
           tu({"file_path": "/home/xqian/tmp/h25r/items_a5.txt"}),
           tu({"command": "ls /home/xqian/tmp/h25r/v_parts"}),
           tu({"file_path": "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd-h25base/smprep-029107_1-c85.json"}),
           tu({"file_path": "/home/xqian/tmp/h25r/smx25_key.tsv"})]
    ng, hg = audit(good, "rv5_a3")
    nb, hb = audit(bad, "rv5_a3")
    ok = ng == len(good) and not hg and nb == len(bad) and len(hb) == len(bad)
    print(f"selftest: clean {ng} calls, {len(hg)} flagged (want 0); bad {nb} calls, {len(hb)} flagged (want {len(bad)})"
          f" -> {'PASS' if ok else 'FAIL'}")
    for h in hg: print("   false alarm:", h)
    sys.exit(0 if ok else 1)

f, own = sys.argv[1:3]
n, hits = audit(open(f), own)
print(f"{f}: tool calls {n} | flagged {len(hits)}")
for h in hits:
    print("  ", h)
sys.exit(1 if hits else 0)
